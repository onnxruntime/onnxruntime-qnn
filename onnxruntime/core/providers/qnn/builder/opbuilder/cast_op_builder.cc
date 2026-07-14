// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class CastOpBuilder : public BaseOpBuilder {
 public:
  CastOpBuilder() : BaseOpBuilder("CastOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CastOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation = false) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status CastOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  RETURN_IF_NOT(inputs.size() == 1, "QNN Cast node must have a single input.");
  const auto& input = inputs[0];

  const auto& input_name = input.name;
  RETURN_IF(qnn_model_wrapper.IsGraphInput(input_name) && input.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            "Unsupported FP64 data type in graph IO.");

  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_name).c_str());
    input_names.push_back(input_name);
    return Ort::Status();
  }

  std::vector<uint8_t> unpacked_tensor;
  bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
  if (is_constant_input) {
    const auto* input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_tensor, unpacked_tensor));
  }

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input_name);
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.shape, input_shape),
                "Cannot get shape for QNN Cast node's input.");

  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
  ONNXTensorElementDataType input_type = input.type;

  RETURN_IF_ERROR(utils::GetQnnDataType(false,  // Do not try to get the quantized type. HTP cast supports normal types.
                                        input_type,
                                        qnn_data_type));

  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::move(input_shape), std::move(unpacked_tensor));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                "Failed to add input tensor for QNN Cast node.");
  input_names.push_back(input_name);

  return Ort::Status();
}

Ort::Status CastOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& outputs = node_unit.Outputs();
  RETURN_IF_NOT(outputs.size() == 1, "QNN Cast node must have a single output.");
  const auto& output = outputs[0];
  const auto& output_name = output.name;

  ONNXTensorElementDataType output_type = output.type;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(false,  // Do not try to get the quantized type. HTP cast supports normal types.
                                        output_type,
                                        qnn_data_type));

  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.shape, output_shape),
                "Cannot get shape for QNN Cast node's output.");
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  const Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  if (qnn_data_type == QNN_DATATYPE_INT_64 && tensor_type == QNN_TENSOR_TYPE_NATIVE) {
    qnn_data_type = QNN_DATATYPE_INT_32;
  } else if (qnn_data_type == QNN_DATATYPE_FLOAT_64) {
    RETURN_IF(is_graph_output, "Unsupported FP64 data type in graph IO.");
    qnn_data_type = QNN_DATATYPE_FLOAT_32;
  }
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        qnn_data_type,
                                        QnnQuantParamsWrapper(),
                                        std::move(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                "Failed to add output tensor for QNN Cast node.");

  const std::string qnn_op_type = GetQnnOpType(node_unit.OpType());
  const std::string cast_node_name = utils::UniqueNameGenerator().New(node_unit);
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                qnn_op_type,
                                                std::move(input_names),
                                                {output_name},
                                                {},
                                                do_op_validation),
                ("Failed to create " + qnn_op_type + " node.").c_str());

  return Ort::Status();
}

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<CastOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
