// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Implements the ONNX RandomUniform op (opset 1+), which unlike RandomUniformLike
// takes zero tensor inputs: shape, low, high, and seed all come from node attributes.
// Lowered onto the same QNN_OP_RANDOM_UNIFORM_LIKE hardware op used by
// RandomUniformLikeOpBuilder: QNN's "Like" op structurally only consumes a small
// static tensor holding the output dims (see RandomUniformLikeOpBuilder, which
// synthesizes that same dims tensor from its reference input's shape instead of
// from an attribute) -- so here we synthesize it directly from the `shape` attribute.
//
// RandomNormal is the exact structural twin of RandomUniform (same attribute-only,
// zero-tensor-input shape) and is intentionally not implemented here -- tracked as a
// known follow-up, not an oversight.
class RandomUniformOpBuilder : public BaseOpBuilder {
 public:
  RandomUniformOpBuilder() : BaseOpBuilder("RandomUniformOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RandomUniformOpBuilder);

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

Ort::Status RandomUniformOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const OrtNodeUnit& node_unit,
                                                  const Ort::Logger& logger,
                                                  std::vector<std::string>& input_names,
                                                  bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);

  // `shape` is a required attribute per the ONNX RandomUniform-1 schema; ONNX's own
  // graph validation should already guarantee it is present, but we check explicitly
  // rather than assume, since there is no input tensor to fall back on.
  std::optional<std::vector<int64_t>> shape_attr = node_helper.GetInt64s("shape");
  if (!shape_attr.has_value() || shape_attr->empty()) {
    return MAKE_EP_FAIL(
        "QNN EP RandomUniform requires a non-empty 'shape' attribute.");
  }
  for (const auto& dim : *shape_attr) {
    if (dim < 0) {
      return MAKE_EP_FAIL(
          "QNN EP RandomUniform requires all 'shape' attribute entries to be "
          "non-negative integers (no symbolic/unknown dimensions are possible via "
          "an attribute).");
    }
  }

  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(shape_attr, output_shape),
                "Failed to convert the 'shape' attribute to a QNN shape.");

  // Node has no NodeArg-backed ORT input to name this after; use the node unit
  // itself as the base for uniqueness instead (mirrors the "_shape" suffix pattern
  // used by RandomUniformLikeOpBuilder for its synthesized dims tensor). The
  // NodeUnit overload falls back to OpType()+Index() for unnamed nodes, unlike
  // node_unit.Name() alone.
  const std::string shape_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_shape");
  std::vector<uint8_t> shape_data(output_shape.size() * sizeof(uint32_t));
  memcpy(shape_data.data(), output_shape.data(), shape_data.size());
  std::vector<uint32_t> shape_tensor_shape = {SafeInt<uint32_t>(output_shape.size())};

  QnnTensorWrapper shape_tensor_wrapper(shape_tensor_name,
                                        QNN_TENSOR_TYPE_STATIC,
                                        QNN_DATATYPE_UINT_32,
                                        QnnQuantParamsWrapper(),
                                        std::move(shape_tensor_shape),
                                        std::move(shape_data));

  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(shape_tensor_wrapper)),
                "Failed to add shape tensor.");

  input_names.push_back(shape_tensor_name);

  // --- If seed attribute is present, add it as second input ---
  if (node_helper.HasAttr("seed")) {
    auto seed_value = node_helper.GetFloat("seed");

    std::vector<uint32_t> scalar_shape = {1};
    std::vector<uint8_t> seed_data(sizeof(float));
    memcpy(seed_data.data(), &seed_value, sizeof(float));

    const std::string seed_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_ort_qnn_ep_seed");

    QnnTensorWrapper seed_tensor(seed_tensor_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32,
                                 QnnQuantParamsWrapper(), std::move(scalar_shape), std::move(seed_data));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(seed_tensor)),
                  "Failed to add seed tensor");

    input_names.push_back(seed_tensor_name);  // Seed is always second
  }
  return Ort::Status();
}

Ort::Status RandomUniformOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                 const OrtNodeUnit& node_unit,
                                                                 std::vector<std::string>&& input_names,
                                                                 const Ort::Logger& logger,
                                                                 bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_names;

  // Extract 'low' attribute
  float low = node_helper.Get("low", 0.0f);
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), low,
                                      QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_LOW, param_names));

  // Extract 'high' attribute
  float high = node_helper.Get("high", 1.0f);
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), high,
                                      QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_HIGH, param_names));

  // NOTE: the ONNX 'dtype' attribute is intentionally not read here. By the time
  // this builder runs, ORT's own ONNX type/shape inference has already resolved
  // the output NodeArg's type from that attribute, and GetTensorInfo() below
  // reflects it -- identical to how RandomUniformLikeOpBuilder never reads a
  // dtype attribute either (its output dtype comes from the same place).
  const auto& outputs = node_unit.Outputs();
  const std::string& output_name = outputs[0].name;

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  // Get output info
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  // Determine if we need to use UFIXED_POINT_8 and add a dequantize node
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  bool need_dequantize = is_npu_backend;

  if (need_dequantize) {
    // Create an intermediate tensor with UFIXED_POINT_8 data type
    const std::string intermediate_output_name = utils::UniqueNameGenerator().New(output_name, "_uint8");

    // Calculate quantization parameters based on low and high values
    QnnQuantParamsWrapper quantize_param;
    float scale = 0.0f;
    int32_t zero_point = 0;
    RETURN_IF_ERROR(utils::GetQuantParams(low, high, QNN_DATATYPE_UFIXED_POINT_8, scale, zero_point));
    quantize_param = QnnQuantParamsWrapper::PerTensor(scale, zero_point);
    QnnTensorWrapper intermediate_output_wrapper(intermediate_output_name,
                                                 QNN_TENSOR_TYPE_NATIVE,
                                                 QNN_DATATYPE_UFIXED_POINT_8,
                                                 std::move(quantize_param),
                                                 std::vector<uint32_t>(output_info.shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(intermediate_output_wrapper)),
                  "Failed to add intermediate output tensor.");

    // Create the RandomUniformLike node with uint8 output
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_RANDOM_UNIFORM_LIKE,
                      std::move(input_names),
                      {intermediate_output_name},
                      std::move(param_names),
                      do_op_validation),
                  "Failed to create RandomUniformLike node.");

    // Create the final output tensor with the original data type
    QnnTensorWrapper output_wrapper(output_name,
                                    tensor_type,
                                    output_info.qnn_data_type,
                                    output_info.quant_param.Copy(),
                                    std::vector<uint32_t>(output_info.shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_wrapper)),
                  "Failed to add output tensor.");

    // Create a Dequantize node to convert from uint8 to float32
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, "_dequantize"),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_DEQUANTIZE,
                      {intermediate_output_name},
                      {output_name},
                      {},
                      do_op_validation),
                  "Failed to create Dequantize node.");
  } else {
    // For non-NPU backends, use the original data type directly
    QnnTensorWrapper output_wrapper(output_name,
                                    tensor_type,
                                    output_info.qnn_data_type,
                                    output_info.quant_param.Copy(),
                                    std::vector<uint32_t>(output_info.shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_wrapper)),
                  "Failed to add output tensor.");

    // Create the RandomUniformLike node with the original data type output
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_RANDOM_UNIFORM_LIKE,
                      std::move(input_names),
                      {output_name},
                      std::move(param_names),
                      do_op_validation),
                  "Failed to create RandomUniformLike node.");
  }

  return Ort::Status();
}

void CreateRandomUniformOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RandomUniformOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
