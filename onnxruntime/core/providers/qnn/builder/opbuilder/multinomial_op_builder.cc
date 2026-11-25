// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <limits>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Inverse-CDF sampler. HTP uniforms have only 8 bits of precision;
// seed forwarding does not guarantee an advancing random stream.
// Supports one sample per row of static, non-quantized, finite logits
// within the backend FP16 range.
class MultinomialOpBuilder : public BaseOpBuilder {
 public:
  MultinomialOpBuilder() : BaseOpBuilder("MultinomialOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MultinomialOpBuilder);

 protected:
  Qnn_DataType_t GetSupportedOutputDataType(size_t index,
                                            Qnn_DataType_t qnn_data_type) const override;

  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Qnn_DataType_t MultinomialOpBuilder::GetSupportedOutputDataType(size_t index,
                                                                Qnn_DataType_t qnn_data_type) const {
  ORT_UNUSED_PARAMETER(index);
  // Follow ArgMaxMinOpBuilder: graph INT64 outputs get a trailing Cast.
  return qnn_data_type == QNN_DATATYPE_INT_64 ? QNN_DATATYPE_INT_32 : qnn_data_type;
}

Ort::Status MultinomialOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const Ort::Logger& logger,
                                                std::vector<std::string>& input_names,
                                                bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  RETURN_IF_NOT(IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()),
                "Multinomial requires the HTP backend.");

  OrtNodeAttrHelper attrs(node_unit);
  RETURN_IF_NOT(attrs.Get("sample_size", int64_t{1}) == 1,
                "Multinomial supports sample_size=1 only.");

  const auto& input = node_unit.Inputs()[0];
  RETURN_IF_NOT(input.shape.has_value() && input.shape->size() == 2 &&
                    (*input.shape)[0] > 0 && (*input.shape)[1] > 0 &&
                    (*input.shape)[1] <= std::numeric_limits<int32_t>::max(),
                "Multinomial requires positive static rank-2 logits dimensions.");

  TensorInfo info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input, info));
  RETURN_IF_NOT(!info.quant_param.IsQuantized() &&
                    (info.qnn_data_type == QNN_DATATYPE_FLOAT_16 || info.qnn_data_type == QNN_DATATYPE_FLOAT_32),
                "Multinomial requires non-quantized FLOAT or FLOAT16 logits.");

  TensorInfo output{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output));
  RETURN_IF_NOT(output.qnn_data_type == QNN_DATATYPE_INT_32 || output.qnn_data_type == QNN_DATATYPE_INT_64,
                "Multinomial requires INT32 or INT64 output.");

  RETURN_IF_NOT(output.shape == std::vector<uint32_t>({info.shape[0], 1}),
                "Multinomial requires output shape [batch, 1].");

  return ProcessInput(qnn_model_wrapper, input, logger, input_names);
}

Ort::Status MultinomialOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                              const OrtNodeUnit& node_unit,
                                                              std::vector<std::string>&& input_names,
                                                              const Ort::Logger& logger,
                                                              bool do_op_validation) const {
  TensorInfo info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], info));
  const auto& shape = info.shape;
  const std::vector<uint32_t> row_shape{shape[0], 1};
  const auto type = info.qnn_data_type;
  const std::string& output = node_unit.Outputs()[0].name;
  auto name = [&](const char* suffix) { return utils::UniqueNameGenerator().New(output, suffix); };
  auto tensor = [&](const std::string& tensor_name, Qnn_DataType_t dtype,
                    const std::vector<uint32_t>& dims) -> Ort::Status {
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(QnnTensorWrapper(tensor_name, QNN_TENSOR_TYPE_NATIVE, dtype,
                                                                      QnnQuantParamsWrapper(), std::vector<uint32_t>(dims))),
                  "Failed to add Multinomial intermediate tensor.");
    return Ort::Status();
  };

  auto op = [&](const char* op_type, std::vector<std::string> in, const std::string& out,
                std::vector<std::string> params = {}) -> Ort::Status {
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(name(op_type), QNN_OP_PACKAGE_NAME_QTI_AISW, op_type,
                                                  std::move(in), {out}, std::move(params), do_op_validation),
                  "Failed to add Multinomial decomposition node.");
    return Ort::Status();
  };

  auto reduce_params = [&](const std::string& owner, std::vector<std::string>& params) -> Ort::Status {
    // ReduceMaxOpBuilder uses uint32 axes and keep_dims=true for row reductions.
    QnnParamWrapper axes(node_unit.Index(), owner, QNN_OP_REDUCE_MAX_PARAM_AXES,
                         std::vector<uint32_t>{1}, std::vector<uint32_t>{1});
    params.push_back(axes.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(axes)), "Failed to add Multinomial reduction axis.");
    return AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), owner, true, QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS, params);
  };

  const std::string& logits = input_names[0];

  // Stabilization keeps every weight in [0,1] and at least one weight equal to 1
  // for each row with finite logits.
  const auto maximum = name("_maximum");
  RETURN_IF_ERROR(tensor(maximum, type, row_shape));
  std::vector<std::string> max_params;
  RETURN_IF_ERROR(reduce_params(maximum, max_params));
  RETURN_IF_ERROR(op(QNN_OP_REDUCE_MAX, {logits}, maximum, std::move(max_params)));

  const auto centered = name("_centered");
  RETURN_IF_ERROR(tensor(centered, type, shape));
  RETURN_IF_ERROR(op(QNN_OP_ELEMENT_WISE_SUBTRACT, {logits, maximum}, centered));

  // HTP floating-point Exp uses FP16 arithmetic (HTP supplement).
  // Large negative inputs can produce invalid weights instead of underflowing:
  // Exp(-1000) can produce a large positive value or infinity. Bound
  // its input as in ClipOpBuilder; Exp(-80) already rounds to zero in FP16,
  // preserving underflowed weights without adding probability mass.
  const auto bounded = name("_bounded");
  RETURN_IF_ERROR(tensor(bounded, type, shape));
  std::vector<std::string> bounds_params;
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), bounded, -80.0f,
                                      QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE, bounds_params));
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), bounded, 0.0f,
                                      QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE, bounds_params));
  RETURN_IF_ERROR(op(QNN_OP_RELU_MIN_MAX, {centered}, bounded, std::move(bounds_params)));

  const auto weights = name("_weights");
  RETURN_IF_ERROR(tensor(weights, type, shape));
  RETURN_IF_ERROR(op(QNN_OP_ELEMENT_WISE_EXP, {bounded}, weights));

  const auto cdf = name("_cdf");
  RETURN_IF_ERROR(tensor(cdf, type, shape));
  std::vector<std::string> cdf_params;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), cdf, 1, QNN_OP_CUMULATIVE_SUM_PARAM_AXIS, cdf_params));
  // Inclusive forward accumulation, as in CumSumOpBuilder.
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), cdf, false, QNN_OP_CUMULATIVE_SUM_PARAM_EXCLUSIVE, cdf_params));
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), cdf, false, QNN_OP_CUMULATIVE_SUM_PARAM_REVERSE, cdf_params));
  RETURN_IF_ERROR(op(QNN_OP_CUMULATIVE_SUM, {weights}, cdf, std::move(cdf_params)));

  const auto total = name("_total");
  RETURN_IF_ERROR(tensor(total, type, row_shape));
  std::vector<std::string> total_params;
  RETURN_IF_ERROR(reduce_params(total, total_params));
  // The nonnegative CDF's maximum is its final value; reuse that accumulation
  // rather than computing the mass using a separate ReduceSum.
  RETURN_IF_ERROR(op(QNN_OP_REDUCE_MAX, {cdf}, total, std::move(total_params)));

  const auto random_shape = name("_random_shape");
  std::vector<uint8_t> shape_data(SafeInt<size_t>(row_shape.size()) * sizeof(uint32_t));
  memcpy(shape_data.data(), row_shape.data(), shape_data.size());
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(QnnTensorWrapper(random_shape, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32,
                                                                    QnnQuantParamsWrapper(), std::vector<uint32_t>{2},
                                                                    std::move(shape_data))),
                "Failed to add Multinomial random shape.");
  std::vector<std::string> random_inputs{random_shape};

  OrtNodeAttrHelper attrs(node_unit);
  if (attrs.HasAttr("seed")) {
    const float seed = attrs.GetFloat("seed").value();
    const auto seed_name = name("_seed");
    std::vector<uint8_t> seed_data(sizeof(seed));
    memcpy(seed_data.data(), &seed, sizeof(seed));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(QnnTensorWrapper(seed_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32,
                                                                      QnnQuantParamsWrapper(), std::vector<uint32_t>{1},
                                                                      std::move(seed_data))),
                  "Failed to add Multinomial seed.");
    random_inputs.push_back(seed_name);
  }

  const auto uniform = name("_uniform");
  RETURN_IF_ERROR(tensor(uniform, type, row_shape));
  std::vector<std::string> random_params;
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), uniform, 0.0f, QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_LOW, random_params));
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), uniform, 1.0f, QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_HIGH, random_params));
  {
    const auto quantized_uniform = name("_uniform_u8");
    // HTP supplement: UFIXED_POINT_8 output. Codes [0,255] map to [0,1),
    // preventing the highest code from becoming the inclusive endpoint 1.
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(QnnTensorWrapper(quantized_uniform, QNN_TENSOR_TYPE_NATIVE,
                                                                      QNN_DATATYPE_UFIXED_POINT_8,
                                                                      QnnQuantParamsWrapper::PerTensor(1.0f / 256.0f, 0),
                                                                      std::vector<uint32_t>(row_shape))),
                  "Failed to add Multinomial quantized uniform tensor.");
    RETURN_IF_ERROR(op(QNN_OP_RANDOM_UNIFORM_LIKE, std::move(random_inputs), quantized_uniform, std::move(random_params)));
    RETURN_IF_ERROR(op(QNN_OP_DEQUANTIZE, {quantized_uniform}, uniform));
  }

  const auto threshold = name("_threshold");
  RETURN_IF_ERROR(tensor(threshold, type, row_shape));
  RETURN_IF_ERROR(op(QNN_OP_ELEMENT_WISE_MULTIPLY, {uniform, total}, threshold));

  const auto crossed = name("_crossed");
  RETURN_IF_ERROR(tensor(crossed, QNN_DATATYPE_BOOL_8, shape));
  RETURN_IF_ERROR(op(QNN_OP_ELEMENT_WISE_LESS, {threshold, cdf}, crossed));

  // Force the final comparison true so rounding cannot yield an all-false
  // row (ArgMax would incorrectly return class zero). Earlier CDF entries
  // retain their original comparisons; no artificial probability is added.
  const auto guard = name("_last_class_guard");
  std::vector<uint8_t> guard_data(shape[1], 0);
  guard_data.at(SafeInt<size_t>(shape[1]) - 1) = 1;
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(QnnTensorWrapper(guard, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_BOOL_8,
                                                                    QnnQuantParamsWrapper(), std::vector<uint32_t>{1, shape[1]},
                                                                    std::move(guard_data))),
                "Failed to add Multinomial last-class guard.");

  const auto guarded = name("_guarded");
  RETURN_IF_ERROR(tensor(guarded, QNN_DATATYPE_BOOL_8, shape));
  RETURN_IF_ERROR(op(QNN_OP_ELEMENT_WISE_OR, {crossed, guard}, guarded));

  const auto indicators = name("_indicators");
  RETURN_IF_ERROR(tensor(indicators, QNN_DATATYPE_INT_32, shape));
  RETURN_IF_ERROR(op(QNN_OP_CAST, {guarded}, indicators));

  std::vector<std::string> argmax_params;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), output, 1, QNN_OP_ARGMAX_PARAM_AXIS, argmax_params));
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), output, true, QNN_OP_ARGMAX_PARAM_KEEP_DIMS, argmax_params));
  return ProcessOutputs(qnn_model_wrapper, node_unit, {indicators}, std::move(argmax_params), logger, do_op_validation, QNN_OP_ARGMAX);
}

void CreateMultinomialOpBuilder(const std::string& op_type, OpBuilderRegistrations& registrations) {
  registrations.AddOpBuilder(op_type, std::make_unique<MultinomialOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
