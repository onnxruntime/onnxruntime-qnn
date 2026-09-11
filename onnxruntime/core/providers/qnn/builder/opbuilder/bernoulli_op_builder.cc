// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Implements the ONNX Bernoulli op via a three-step QNN decomposition.
// Semantics: output[i] = 1 if Uniform(0,1) < input[i], else 0.
//
// CPU path:
//   1. QNN_OP_RANDOM_UNIFORM_LIKE(input, low=0, high=1) → rand (float)
//   2. QNN_OP_ELEMENT_WISE_LESS(rand, input_prob)       → cmp (BOOL_8)
//   3. QNN_OP_CAST(cmp → target_dtype)
//
// HTP (NPU) path:
//   1a. QNN_OP_RANDOM_UNIFORM_LIKE(input, low=0, high=1) → rand_uint8 (UFIXED_POINT_8)
//   1b. QNN_OP_DEQUANTIZE(rand_uint8)                    → rand (float)
//   2.  QNN_OP_ELEMENT_WISE_LESS(rand, input_prob)       → cmp (BOOL_8)
//       HTP's ElementWiseLess only accepts BOOL_8 as output type.
//   3.  QNN_OP_CAST(cmp → target_dtype)
class BernoulliOpBuilder : public BaseOpBuilder {
 public:
  BernoulliOpBuilder() : BaseOpBuilder("BernoulliOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BernoulliOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override;
};

Ort::Status BernoulliOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnit& node_unit,
                                              const Ort::Logger& logger,
                                              std::vector<std::string>& input_names,
                                              bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto& input_tensor = inputs[0];
  const std::string& input_tensor_name = input_tensor.name;

  // Reject dynamic shapes; the underlying QNN_OP_RANDOM_UNIFORM_LIKE needs a
  // static shape tensor baked at compile time.
  if (!input_tensor.shape.has_value()) {
    return MAKE_EP_FAIL(
        "QNN EP Bernoulli requires static input dimensions. "
        "Input shape is unknown.");
  }
  for (const auto& dim : *input_tensor.shape) {
    if (dim < 0) {
      return MAKE_EP_FAIL(
          "QNN EP Bernoulli requires static input dimensions. "
          "Found symbolic/unknown dimension in input shape.");
    }
  }

  // Register the probability input tensor as a QNN tensor.
  // RandomUniformLike only uses the static shape tensor, but the input is also
  // consumed by the ElementWiseLess node in Step 2. Without this registration,
  // ORT's SetupTensors fails with "Zero tensor size!".
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_tensor_name)) {
    QnnTensorWrapper input_tensorwrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_tensor, input_tensorwrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                  "Failed to add input tensor wrapper.");
  }

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_tensor.shape, input_shape),
                ("Failed to get shape for input tensor: " + input_tensor_name).c_str());

  const std::string shape_tensor_name = utils::UniqueNameGenerator().New(input_tensor_name, "_shape");
  std::vector<uint8_t> shape_data(input_shape.size() * sizeof(uint32_t));
  memcpy(shape_data.data(), input_shape.data(), shape_data.size());
  std::vector<uint32_t> shape_tensor_shape = {static_cast<uint32_t>(input_shape.size())};

  QnnTensorWrapper shape_tensor_wrapper(shape_tensor_name,
                                        QNN_TENSOR_TYPE_STATIC,
                                        QNN_DATATYPE_UINT_32,
                                        QnnQuantParamsWrapper(),
                                        std::move(shape_tensor_shape),
                                        std::move(shape_data));

  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(shape_tensor_wrapper)),
                "Failed to add shape tensor.");

  input_names.push_back(shape_tensor_name);

  // If seed attribute is present, add it as the second input to RandomUniformLike.
  if (node_helper.HasAttr("seed")) {
    auto seed_value = node_helper.GetFloat("seed");

    std::vector<uint32_t> scalar_shape = {1};
    std::vector<uint8_t> seed_data(sizeof(float));
    memcpy(seed_data.data(), &seed_value, sizeof(float));

    const std::string seed_tensor_name = utils::UniqueNameGenerator().New(input_tensor_name, "_ort_qnn_ep_seed");

    QnnTensorWrapper seed_tensor(seed_tensor_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32,
                                 QnnQuantParamsWrapper(), std::move(scalar_shape), std::move(seed_data));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(seed_tensor)),
                  "Failed to add seed tensor");

    input_names.push_back(seed_tensor_name);
  }
  return Ort::Status();
}

Ort::Status BernoulliOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            std::vector<std::string>&& input_names,
                                                            const Ort::Logger& logger,
                                                            bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);

  const auto& inputs = node_unit.Inputs();
  const auto& input_tensor = inputs[0];
  const std::string& input_prob_name = input_tensor.name;

  const auto& outputs = node_unit.Outputs();
  const std::string& final_output_name = outputs[0].name;

  // Retrieve the shape for intermediate tensors.
  TensorInfo input_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_tensor, input_info));

  // --- Step 1: QNN_OP_RANDOM_UNIFORM_LIKE with hardcoded low=0, high=1 ---
  constexpr float kLow = 0.0f;
  constexpr float kHigh = 1.0f;

  std::vector<std::string> rul_param_names;
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), kLow,
                                      QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_LOW, rul_param_names));
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), kHigh,
                                      QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_HIGH, rul_param_names));

  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  const std::string rand_name = utils::UniqueNameGenerator().New(final_output_name, "_rand");

  if (is_npu_backend) {
    // HTP path: emit UFIXED_POINT_8 intermediate, then add QNN_OP_DEQUANTIZE to float.
    const std::string intermediate_uint8_name = utils::UniqueNameGenerator().New(final_output_name, "_rand_uint8");

    QnnQuantParamsWrapper quantize_param;
    float scale = 0.0f;
    int32_t zero_point = 0;
    RETURN_IF_ERROR(utils::GetQuantParams(kLow, kHigh, QNN_DATATYPE_UFIXED_POINT_8, scale, zero_point));
    quantize_param = QnnQuantParamsWrapper::PerTensor(scale, zero_point);

    QnnTensorWrapper intermediate_uint8_wrapper(intermediate_uint8_name,
                                                QNN_TENSOR_TYPE_NATIVE,
                                                QNN_DATATYPE_UFIXED_POINT_8,
                                                std::move(quantize_param),
                                                std::vector<uint32_t>(input_info.shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(intermediate_uint8_wrapper)),
                  "Failed to add intermediate UFIXED_POINT_8 tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, "_rul"),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_RANDOM_UNIFORM_LIKE,
                      std::move(input_names),
                      {intermediate_uint8_name},
                      std::move(rul_param_names),
                      do_op_validation),
                  "Failed to create RandomUniformLike node (NPU).");

    // Dequantize UFIXED_POINT_8 → float32 (same shape, same dtype as probability input).
    QnnTensorWrapper rand_wrapper(rand_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  input_info.qnn_data_type,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(input_info.shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(rand_wrapper)),
                  "Failed to add dequantized random tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, "_dequantize"),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_DEQUANTIZE,
                      {intermediate_uint8_name},
                      {rand_name},
                      {},
                      do_op_validation),
                  "Failed to create Dequantize node.");
  } else {
    // Non-NPU path: RandomUniformLike outputs float directly.
    QnnTensorWrapper rand_wrapper(rand_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  input_info.qnn_data_type,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(input_info.shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(rand_wrapper)),
                  "Failed to add random tensor (non-NPU).");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, "_rul"),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_RANDOM_UNIFORM_LIKE,
                      std::move(input_names),
                      {rand_name},
                      std::move(rul_param_names),
                      do_op_validation),
                  "Failed to create RandomUniformLike node (non-NPU).");
  }

  // --- Step 2: QNN_OP_ELEMENT_WISE_LESS: Less(rand, input_prob) → BOOL_8 ---
  // Produces: output[i] = (rand[i] < prob[i])  ≡  Bernoulli sample.
  // HTP's ElementWiseLess only accepts BOOL_8 as its output type; using UFIXED_POINT_8
  // fails op validation (error 3110) and causes the node to fall back to CPU.
  const std::string cmp_name = utils::UniqueNameGenerator().New(final_output_name, "_cmp");
  QnnTensorWrapper cmp_wrapper(cmp_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_BOOL_8,
                               QnnQuantParamsWrapper(), std::vector<uint32_t>(input_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cmp_wrapper)),
                "Failed to add comparison (BOOL_8) tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::UniqueNameGenerator().New(node_unit, "_less"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_ELEMENT_WISE_LESS,
                    {rand_name, input_prob_name},
                    {cmp_name},
                    {},
                    do_op_validation),
                "Failed to create ElementWiseLess node.");

  // --- Step 3: comparison tensor → target_dtype via QNN_OP_CAST ---
  // Determine target QNN data type from the 'dtype' attribute.
  // ONNX Bernoulli 'dtype' is an int64 TensorProto_DataType value; 0 means "same as input".
  Qnn_DataType_t target_qnn_dtype = QNN_DATATYPE_UNDEFINED;
  int64_t onnx_dtype_attr = node_helper.Get("dtype", static_cast<int64_t>(0));

  if (onnx_dtype_attr == 0) {
    // dtype unset: output type matches input probability type.
    target_qnn_dtype = input_info.qnn_data_type;
  } else {
    RETURN_IF_ERROR(utils::GetQnnDataType(
        false,
        static_cast<ONNXTensorElementDataType>(onnx_dtype_attr),
        target_qnn_dtype));
  }

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  Qnn_TensorType_t output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  QnnTensorWrapper output_wrapper(final_output_name,
                                  output_tensor_type,
                                  target_qnn_dtype,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_wrapper)),
                "Failed to add output tensor.");

  // Both backends feed BOOL_8 into Step 3. Cast(BOOL_8 → target_dtype) is
  // supported on both CPU and HTP.
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::UniqueNameGenerator().New(node_unit, "_cast"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_CAST,
                    {cmp_name},
                    {final_output_name},
                    {},
                    do_op_validation),
                "Failed to create Cast node.");

  return Ort::Status();
}

void CreateBernoulliOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<BernoulliOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
