// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class RandomNormalLikeOpBuilder : public BaseOpBuilder {
 public:
  RandomNormalLikeOpBuilder() : BaseOpBuilder("RandomNormalLikeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RandomNormalLikeOpBuilder);

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

Ort::Status RandomNormalLikeOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
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

  // Explicitly reject dynamic (symbolic/unknown) input shapes.
  // This builder materializes a static shape tensor at compile time, so all dimensions
  // must be statically known. Dynamic shapes would bake incorrect values into the graph.
  if (!input_tensor.shape.has_value()) {
    return MAKE_EP_FAIL(
        "QNN EP RandomNormalLike requires static input dimensions. "
        "Input shape is unknown.");
  }
  for (const auto& dim : *input_tensor.shape) {
    if (dim < 0) {
      return MAKE_EP_FAIL(
          "QNN EP RandomNormalLike requires static input dimensions. "
          "Found symbolic/unknown dimension in input shape.");
    }
  }

  // Register `x` as a QnnTensorWrapper even though the QNN op only consumes the static
  // shape tensor — without it, ORT's SetupTensors fails with "Zero tensor size!".
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_tensor_name)) {
    QnnTensorWrapper input_tensorwrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_tensor, input_tensorwrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                  "Failed to add input tensor wrapper.");
  }

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_tensor.shape, input_shape),
                ("Failed to get shape for input tensor: " + input_tensor_name).c_str());

  // Create a static shape tensor from the input's shape for the QNN RandomNormalLike node
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

  // --- If seed attribute is present, add it as second input ---
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

    input_names.push_back(seed_tensor_name);  // Seed is always second
  }
  return Ort::Status();
}

Ort::Status RandomNormalLikeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                   const OrtNodeUnit& node_unit,
                                                                   std::vector<std::string>&& input_names,
                                                                   const Ort::Logger& logger,
                                                                   bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_names;

  // Extract 'mean' attribute
  float mean = node_helper.Get("mean", 0.0f);
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), mean,
                                      QNN_OP_RANDOM_NORMAL_LIKE_PARAM_MEAN, param_names));

  // Extract 'scale' attribute
  float scale = node_helper.Get("scale", 1.0f);
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), scale,
                                      QNN_OP_RANDOM_NORMAL_LIKE_PARAM_SCALE, param_names));

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

    // Calculate quantization parameters based on the expected range of the normal distribution.
    // Use mean ± 3*scale (std_dev) to cover ~99.7% of values.
    float quant_min = mean - 3.0f * scale;
    float quant_max = mean + 3.0f * scale;
    float quant_scale = 0.0f;
    int32_t zero_point = 0;
    RETURN_IF_ERROR(utils::GetQuantParams(quant_min, quant_max, QNN_DATATYPE_UFIXED_POINT_8, quant_scale, zero_point));
    QnnQuantParamsWrapper quantize_param(quant_scale, zero_point);

    QnnTensorWrapper intermediate_output_wrapper(intermediate_output_name,
                                                 QNN_TENSOR_TYPE_NATIVE,
                                                 QNN_DATATYPE_UFIXED_POINT_8,
                                                 std::move(quantize_param),
                                                 std::vector<uint32_t>(output_info.shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(intermediate_output_wrapper)),
                  "Failed to add intermediate output tensor.");

    // Create the RandomNormalLike node with uint8 output
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_RANDOM_NORMAL_LIKE,
                      std::move(input_names),
                      {intermediate_output_name},
                      std::move(param_names),
                      do_op_validation),
                  "Failed to create RandomNormalLike node.");

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

    // Create the RandomNormalLike node with the original data type output
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_RANDOM_NORMAL_LIKE,
                      std::move(input_names),
                      {output_name},
                      std::move(param_names),
                      do_op_validation),
                  "Failed to create RandomNormalLike node.");
  }

  return Ort::Status();
}

void CreateRandomNormalLikeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RandomNormalLikeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
