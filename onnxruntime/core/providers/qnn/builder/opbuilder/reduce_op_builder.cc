// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

enum ReduceOpType {
  REDUCE_OP_TYPE_MAX = 0,
  REDUCE_OP_TYPE_MIN,
  REDUCE_OP_TYPE_MEAN,
  REDUCE_OP_TYPE_PROD,
  REDUCE_OP_TYPE_SUM,
  REDUCE_OP_TYPE_L2,
  REDUCE_OP_TYPE_LOG_SUM_EXP,

  REDUCE_OP_TYPE_COUNT,
  REDUCE_OP_TYPE_UNKNOWN,
};

ReduceOpType GetReduceOpType(const std::string& op_type) {
  if (op_type == "ReduceMax") {
    return REDUCE_OP_TYPE_MAX;
  } else if (op_type == "ReduceMin") {
    return REDUCE_OP_TYPE_MIN;
  } else if (op_type == "ReduceMean") {
    return REDUCE_OP_TYPE_MEAN;
  } else if (op_type == "ReduceProd") {
    return REDUCE_OP_TYPE_PROD;
  } else if (op_type == "ReduceSum") {
    return REDUCE_OP_TYPE_SUM;
  } else if (op_type == "ReduceL2") {
    return REDUCE_OP_TYPE_L2;
  } else if (op_type == "ReduceLogSumExp") {
    return REDUCE_OP_TYPE_LOG_SUM_EXP;
  } else {
    return REDUCE_OP_TYPE_UNKNOWN;
  }
}

class ReduceOpBuilder : public BaseOpBuilder {
 public:
  ReduceOpBuilder() : BaseOpBuilder("ReduceOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReduceOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit, const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation = false) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names, const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  using AxesOnnxIntType = int64_t;
  using AxesQnnIntType = uint32_t;

  Ort::Status GetAxesSet(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                         std::set<AxesOnnxIntType>& axes_set) const;

  // Maps an operator type to the opset in which "axes" became an input instead of an attribute.
  static const std::array<int, REDUCE_OP_TYPE_COUNT> opset_with_axes_as_input;
};

const std::array<int, REDUCE_OP_TYPE_COUNT> ReduceOpBuilder::opset_with_axes_as_input = {
    18,  // ReduceMax
    18,  // ReduceMin
    18,  // ReduceMean
    18,  // ReduceProd
    13,  // ReduceSum
    18,  // ReduceL2
    18,  // ReduceLogSumExp
};

Ort::Status ReduceOpBuilder::GetAxesSet(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                        std::set<AxesOnnxIntType>& axes_set) const {
  ReduceOpType reduce_op_type = GetReduceOpType(node_unit.OpType());
  RETURN_IF(reduce_op_type == ReduceOpType::REDUCE_OP_TYPE_UNKNOWN,
            ("QNN EP: Unknown reduce operator " + node_unit.OpType()).c_str());

  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape");
  const size_t input_rank = input_shape.size();

  std::vector<AxesOnnxIntType> reduce_axes;

  const int opset_axes_as_input = ReduceOpBuilder::opset_with_axes_as_input[reduce_op_type];
  const int opset = node_unit.SinceVersion();
  OrtNodeAttrHelper node_helper(node_unit);

  // Extract the axes values from either the attribute or initializer input (depending on opset).
  if (opset < opset_axes_as_input) {  // Axes is in ONNX node attribute.
    reduce_axes = node_helper.Get(QNN_OP_REDUCE_MAX_PARAM_AXES, reduce_axes);
  } else if (inputs.size() > 1) {  // Axes is in ONNX input[1] initializer.
    const auto& axes_input = inputs[1];

    std::vector<uint32_t> axes_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(axes_input.shape, axes_shape), "Cannot get shape of axes input");

    RETURN_IF(axes_shape.size() != 1, "QNN EP: \"axes\" input must have shape [M] where 0 < M <= rank(input[0])");

    bool noop_with_empty_axes = static_cast<bool>(node_helper.Get("noop_with_empty_axes", (int64_t)0));
    RETURN_IF(axes_shape[0] == 0 && noop_with_empty_axes,
              "QNN EP: does not support NoOp for reduction operators with empty axes.");

    // Empty axes means to use default axes (when noop_with_empty_axes is 0).
    if (axes_shape[0] > 0) {
      const std::string& axes_input_name = inputs[1].name;

      // Check that the axes input is an initializer.
      RETURN_IF(!qnn_model_wrapper.IsConstantInput(axes_input_name),
                "QNN EP: \"axes\" input for reduce operator must be an initializer");

      // Get axes initializer bytes.
      const auto* axes_tensor = qnn_model_wrapper.GetConstantTensor(axes_input_name);
      std::vector<uint8_t> axes_bytes;

      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(axes_tensor, axes_bytes));
      RETURN_IF_NOT(input_rank * sizeof(AxesOnnxIntType) >= axes_bytes.size(),
                    "Expect QNN Reduce* operator to have at most rank(input[0]) axes elements.");
      reduce_axes.resize(axes_bytes.size() / sizeof(AxesOnnxIntType));

      auto src_span = gsl::make_span(axes_bytes.data(), axes_bytes.size());
      auto dst_span = gsl::make_span(reduce_axes.data(), reduce_axes.size());

      std::memcpy(dst_span.data(), src_span.data(), src_span.size_bytes());
    }
  }

  if (reduce_axes.size() == 0) {
    // Use default axes of (0, 1, 2, ..., input_rank - 1)
    for (size_t i = 0; i < input_rank; ++i) {
      axes_set.insert(static_cast<AxesOnnxIntType>(i));
    }
  } else {
    // QNN does not support negative axes values. Fix negative values by adding the input rank.
    for (auto ax : reduce_axes) {
      AxesOnnxIntType positive_axis = (ax < 0) ? (ax + static_cast<AxesOnnxIntType>(input_rank)) : ax;
      axes_set.insert(positive_axis);
    }
  }

  RETURN_IF(axes_set.size() > input_rank, "QNN EP: \"axes\" input must have shape [M] where 0 < M <= rank(input[0])");

  return Ort::Status();
}

Ort::Status ReduceOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger) const {
  ReduceOpType reduce_op_type = GetReduceOpType(node_unit.OpType());
  RETURN_IF(reduce_op_type == ReduceOpType::REDUCE_OP_TYPE_UNKNOWN,
            ("QNN EP: Unknown reduce operator " + node_unit.OpType()).c_str());

  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  RETURN_IF(reduce_op_type == ReduceOpType::REDUCE_OP_TYPE_PROD && is_npu_backend,
            "QNN EP: ReduceProd operator not supported by HTP backend.");

  if (reduce_op_type == ReduceOpType::REDUCE_OP_TYPE_LOG_SUM_EXP) {
    RETURN_IF(node_unit.Inputs()[0].quant_param.has_value(),
              "QNN EP: ReduceLogSumExp operator does not support quantized input.");
    ONNXTensorElementDataType input_type = node_unit.Inputs()[0].type;
    RETURN_IF(input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                  input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
                  input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
              "QNN EP: ReduceLogSumExp operator only supports float input dtypes.");
  }

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Ort::Status ReduceOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger, std::vector<std::string>& input_names,
                                           bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();

  // Only need to process input[0]. In newer opset versions, input[1] corresponds to the reduce axes,
  // which needs to be set as a QNN parameter.
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  return Ort::Status();
}

Ort::Status ReduceOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& logger,
                                                         bool do_op_validation) const {
  OrtNodeAttrHelper node_attr_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  //
  // Handle axes param.
  //
  std::set<AxesOnnxIntType> axes_set;
  RETURN_IF_ERROR(GetAxesSet(qnn_model_wrapper, node_unit, axes_set));
  const size_t num_axes = axes_set.size();

  // Truncate int64 ONNX axes values to QNN's required type (uint32_t).
  std::vector<AxesQnnIntType> axes_shape{SafeInt<AxesQnnIntType>(num_axes)};
  std::vector<AxesQnnIntType> axes_data;
  axes_data.resize(num_axes);
  std::transform(axes_set.begin(), axes_set.end(), axes_data.begin(),
                 [](AxesOnnxIntType item) { return SafeInt<AxesQnnIntType>(item); });

  QnnParamWrapper axes_param(node_unit.Index(), node_unit.Name(), QNN_OP_REDUCE_MAX_PARAM_AXES, std::move(axes_shape),
                             std::move(axes_data));
  param_tensor_names.push_back(axes_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axes_param));

  //
  // Handle keepdims param.
  //
  const int32_t onnx_keepdims = node_attr_helper.Get("keepdims", (int32_t)1);
  if (node_unit.OpType() != "ReduceLogSumExp") {
    RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), onnx_keepdims != 0,
                                       QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS, param_tensor_names));
  }

  if (node_unit.OpType() == "ReduceL2") {
    // If ReduceL2, QNN doesn't have a single Op for it, we need to add a
    // ElementWiseMultiply->ReduceSum->ElementWiseSquareRoot node sequence.
    //
    // The intermediate Multiply/ReduceSum activations don't have well-defined quantization
    // parameters (squaring a quantized value doesn't map to a simple rescale), so when the
    // node unit is quantized we Dequantize the input to float32, run the decomposition in
    // float32, then Quantize the final result back using the QDQ node unit's output quant
    // params -- mirroring how BatchNormalizationOpBuilder handles the same problem.
    const auto& input = node_unit.Inputs()[0];
    const auto& output = node_unit.Outputs()[0];
    std::vector<uint32_t> input_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.shape, input_shape), "Cannot get input shape.");
    std::vector<uint32_t> output_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.shape, output_shape), "Cannot get output shape.");
    const bool is_quantized_op = input.quant_param.has_value();
    Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
    if (!is_quantized_op) {
      RETURN_IF_ERROR(utils::GetQnnDataType(false, output.type, qnn_data_type));
    }
    std::string input_name = input_names[0];

    if (is_quantized_op) {
      // Insert Dequantize (quantized -> float32) before the float32 decomposition.
      const std::string dq_output_name = utils::UniqueNameGenerator().New(input_name, "_to_f32");
      QnnTensorWrapper dq_tensorwrapper(dq_output_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                                        QnnQuantParamsWrapper(), std::vector<uint32_t>(input_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(dq_tensorwrapper)), "AddTensorWrapper failed");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_dequant_in"),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_DEQUANTIZE,
                                                    {input_name}, {dq_output_name}, {},
                                                    do_op_validation),
                    "Failed to add dequantize node");
      input_name = dq_output_name;
    }

    // Step 1: y_pow2 = x * x, using ElementWiseMultiply instead of ElementWisePower so we don't need to add a new
    // initializer tensor for the power value. The performance difference is negligible.
    const std::string pow2_output_name = utils::UniqueNameGenerator().New(input_name, "_pow2");
    QnnTensorWrapper pow2_tensorwrapper(pow2_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type, QnnQuantParamsWrapper(),
                                        std::vector<uint32_t>(input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(pow2_tensorwrapper)), "AddTensorWrapper failed");
    std::string pow2_node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_BINARY);
    std::vector<std::string> pow2_param_names;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), pow2_node_name,
                                           static_cast<uint32_t>(QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY),
                                           QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, pow2_param_names));
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(pow2_node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_BINARY,
                                                  {input_name, input_name},
                                                  {pow2_output_name},
                                                  std::move(pow2_param_names),
                                                  do_op_validation),
                  "CreateQnnNode failed");

    // Step 2: y_pow2_sum = ReduceSum(y_pow2)
    const std::string reduce_output_name = utils::UniqueNameGenerator().New(input_name, "_sum");
    QnnTensorWrapper reduce_tensorwrapper(reduce_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type, QnnQuantParamsWrapper(),
                                          std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reduce_tensorwrapper)), "AddTensorWrapper failed");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_REDUCE_SUM),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_REDUCE_SUM,
                                                  {pow2_output_name},
                                                  {reduce_output_name},
                                                  std::move(param_tensor_names),
                                                  do_op_validation),
                  "CreateQnnNode failed");

    // Step 3: y = Sqrt(y_pow2_sum)
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output.name);
    const std::string sqrt_output_name =
        is_quantized_op ? utils::UniqueNameGenerator().New(output.name, "_f32") : output.name;
    Qnn_TensorType_t sqrt_tensor_type =
        (!is_quantized_op && is_graph_output) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper sqrt_tensorwrapper(sqrt_output_name, sqrt_tensor_type, qnn_data_type,
                                        QnnQuantParamsWrapper(), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sqrt_tensorwrapper)), "AddTensorWrapper failed");
    std::string sqrt_node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_UNARY);
    std::vector<std::string> sqrt_param_names;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), sqrt_node_name,
                                           static_cast<uint32_t>(QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT),
                                           QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION, sqrt_param_names));
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(sqrt_node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_UNARY,
                                                  {reduce_output_name},
                                                  {sqrt_output_name},
                                                  std::move(sqrt_param_names),
                                                  do_op_validation),
                  "CreateQnnNode failed");

    if (is_quantized_op) {
      // Insert Quantize (float32 -> quantized) after the float32 decomposition, using the
      // QDQ node unit's output quant params.
      TensorInfo output_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, output_info));
      Qnn_TensorType_t final_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
      QnnTensorWrapper final_tensorwrapper(output.name, final_tensor_type, output_info.qnn_data_type,
                                           std::move(output_info.quant_param), std::move(output_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_tensorwrapper)), "AddTensorWrapper failed");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_quant_out"),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_QUANTIZE,
                                                    {sqrt_output_name}, {output.name}, {},
                                                    do_op_validation),
                    "Failed to add quantize node");
    }
  } else if (node_unit.OpType() == "ReduceLogSumExp") {
    // Decompose ReduceLogSumExp(x, axes, keepdims) numerically-stably as:
    //   m       = ReduceMax(x, axes, keepdims=True)
    //   result  = Add(Log(ReduceSum(Exp(Sub(x, m)), axes, keepdims=True)), m)
    //   output  = result if user keepdims=True else Reshape(result, output_shape)
    // Subtracting per-axis max keeps Exp in (0, 1] so the chain is numerically safe on FP16.
    // All intermediate reduce outputs use keepdims=True so shapes line up without scalar tensors.
    const auto& input = node_unit.Inputs()[0];
    const auto& output = node_unit.Outputs()[0];
    std::vector<uint32_t> input_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.shape, input_shape), "Cannot get input shape.");
    std::vector<uint32_t> output_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.shape, output_shape), "Cannot get output shape.");
    Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
    RETURN_IF_ERROR(utils::GetQnnDataType(false, output.type, qnn_data_type));
    const std::string input_name = input_names[0];

    // kept_shape: input shape with reduced axes set to 1 (full input rank). All internal reduce/elementwise
    // outputs share this shape, so broadcasting and the final Add work without rank mismatches.
    std::vector<uint32_t> kept_shape = input_shape;
    for (auto ax : axes_set) {
      RETURN_IF_NOT(ax >= 0 && static_cast<size_t>(ax) < input_shape.size(),
                    "QNN EP: ReduceLogSumExp axis out of range.");
      kept_shape[static_cast<size_t>(ax)] = 1;
    }

    // Reuse the user's axes param (already in param_tensor_names[0]) for both inner ReduceMax and ReduceSum.
    const std::string user_axes_param_name = param_tensor_names[0];

    // Build a separate keep_dims=True param for the inner reduces.
    std::vector<std::string> kd_true_param_names;
    RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(),
                                       utils::UniqueNameGenerator().New(node_unit, "_inner"),
                                       true, QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS, kd_true_param_names));
    const std::string kd_true_param_name = kd_true_param_names[0];

    const bool needs_reshape = (onnx_keepdims == 0);

    // Step 1: m = ReduceMax(x, axes=user, keepdims=True).
    const std::string max_output_name = utils::UniqueNameGenerator().New(input_name, "_max");
    QnnTensorWrapper max_tensorwrapper(max_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>(kept_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(max_tensorwrapper)), "AddTensorWrapper failed");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_REDUCE_MAX),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_REDUCE_MAX,
                                                  {input_name},
                                                  {max_output_name},
                                                  {user_axes_param_name, kd_true_param_name},
                                                  do_op_validation),
                  "CreateQnnNode failed");

    // Step 2: d = Sub(x, m).
    const std::string sub_output_name = utils::UniqueNameGenerator().New(input_name, "_normalized");
    QnnTensorWrapper sub_tensorwrapper(sub_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>(input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sub_tensorwrapper)), "AddTensorWrapper failed");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_SUBTRACT),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_SUBTRACT,
                                                  {input_name, max_output_name},
                                                  {sub_output_name},
                                                  {},
                                                  do_op_validation),
                  "CreateQnnNode failed");

    // Step 3: e = Exp(d).
    const std::string exp_output_name = utils::UniqueNameGenerator().New(input_name, "_exp");
    QnnTensorWrapper exp_tensorwrapper(exp_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>(input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(exp_tensorwrapper)), "AddTensorWrapper failed");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_EXP),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_EXP,
                                                  {sub_output_name},
                                                  {exp_output_name},
                                                  {},
                                                  do_op_validation),
                  "CreateQnnNode failed");

    // Step 4: s = ReduceSum(e, axes=user, keepdims=True).
    const std::string reduce_sum_output_name = utils::UniqueNameGenerator().New(input_name, "_sum");
    QnnTensorWrapper reduce_sum_tensorwrapper(reduce_sum_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type,
                                              QnnQuantParamsWrapper(), std::vector<uint32_t>(kept_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reduce_sum_tensorwrapper)), "AddTensorWrapper failed");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_REDUCE_SUM),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_REDUCE_SUM,
                                                  {exp_output_name},
                                                  {reduce_sum_output_name},
                                                  {user_axes_param_name, kd_true_param_name},
                                                  do_op_validation),
                  "CreateQnnNode failed");

    // Step 5: l = Log(s).
    const std::string log_output_name = utils::UniqueNameGenerator().New(input_name, "_log");
    QnnTensorWrapper log_tensorwrapper(log_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>(kept_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(log_tensorwrapper)), "AddTensorWrapper failed");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_LOG),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_LOG,
                                                  {reduce_sum_output_name},
                                                  {log_output_name},
                                                  {},
                                                  do_op_validation),
                  "CreateQnnNode failed");

    // Step 6: result_kept = Add(l, m). Both have kept_shape, no broadcast.
    const std::string add_output_name = needs_reshape
                                            ? utils::UniqueNameGenerator().New(input_name, "_kept")
                                            : output.name;
    Qnn_TensorType_t add_output_tensor_type =
        (!needs_reshape && qnn_model_wrapper.IsGraphOutput(output.name)) ? QNN_TENSOR_TYPE_APP_READ
                                                                         : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper add_tensorwrapper(add_output_name, add_output_tensor_type, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>(kept_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(add_tensorwrapper)), "AddTensorWrapper failed");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_ADD),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_ADD,
                                                  {log_output_name, max_output_name},
                                                  {add_output_name},
                                                  {},
                                                  do_op_validation),
                  "CreateQnnNode failed");

    // Step 7 (only when user keepdims=False): squeeze the reduced axes via Reshape.
    if (needs_reshape) {
      Qnn_TensorType_t output_tensor_type =
          qnn_model_wrapper.IsGraphOutput(output.name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
      QnnTensorWrapper reshape_tensorwrapper(output.name, output_tensor_type, qnn_data_type, QnnQuantParamsWrapper(),
                                             std::move(output_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_tensorwrapper)), "AddTensorWrapper failed");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_RESHAPE),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_RESHAPE,
                                                    {add_output_name},
                                                    {output.name},
                                                    {},
                                                    do_op_validation),
                    "CreateQnnNode failed");
    }
  } else {
    RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names),
                                   std::move(param_tensor_names), logger, do_op_validation,
                                   GetQnnOpType(node_unit.OpType())));
  }

  return Ort::Status();
}

void CreateReduceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ReduceOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
