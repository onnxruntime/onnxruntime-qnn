// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// QNN does not have a native Tan op, so it's decomposed into Sin, Cos, and Div ops
class TanOpBuilder : public BaseOpBuilder {
 public:
  TanOpBuilder() : BaseOpBuilder("TanOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TanOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status TanOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);

  const auto& inputs = node_unit.Inputs();
  RETURN_IF_NOT(inputs.size() == 1, "Tan operator must have exactly 1 input.");

  const auto& outputs = node_unit.Outputs();
  RETURN_IF_NOT(outputs.size() == 1, "Tan operator must have exactly 1 output.");

  RETURN_IF_NOT(inputs[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    inputs[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                "Tan operator only supports float and float16 input.");
  RETURN_IF_NOT(outputs[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    outputs[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                "Tan operator only supports float and float16 output.");

  return Ort::Status();
}

Ort::Status TanOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      std::vector<std::string>&& input_names,
                                                      const Ort::Logger& logger,
                                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& outputs = node_unit.Outputs();
  const std::string& output_name = outputs[0].name;

  TensorInfo input_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  // Create intermediate tensor for Sin output
  std::string sin_output_name = utils::GetUniqueName(node_unit, "_sin_out");
  QnnTensorWrapper sin_output_tensor_wrapper(sin_output_name, QNN_TENSOR_TYPE_NATIVE, input_info.qnn_data_type,
                                             input_info.quant_param.Copy(), std::vector<uint32_t>(input_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sin_output_tensor_wrapper)),
                "Failed to add Sin output tensor.");

  // Create intermediate tensor for Cos output
  std::string cos_output_name = utils::GetUniqueName(node_unit, "_cos_out");
  QnnTensorWrapper cos_output_tensor_wrapper(cos_output_name, QNN_TENSOR_TYPE_NATIVE, input_info.qnn_data_type,
                                             input_info.quant_param.Copy(), std::vector<uint32_t>(input_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cos_output_tensor_wrapper)),
                "Failed to add Cos output tensor.");

  // Create Sin node: input -> sin_out
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit, "_Sin"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_SIN,
                                                {input_names[0]},
                                                {sin_output_name},
                                                {},
                                                do_op_validation),
                "Failed to create Sin node.");

  // Create Cos node: input -> cos_out
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit, "_Cos"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_COS,
                                                {input_names[0]},
                                                {cos_output_name},
                                                {},
                                                do_op_validation),
                "Failed to create Cos node.");

  // Create final output tensor
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper output_tensor_wrapper(output_name, tensor_type, output_info.qnn_data_type,
                                         output_info.quant_param.Copy(), std::move(output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                "Failed to add output tensor.");

  // Create Div node: sin_out / cos_out -> output
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit, "_Div"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_DIVIDE,
                                                {sin_output_name, cos_output_name},
                                                {output_name},
                                                {},
                                                do_op_validation),
                "Failed to create Div node.");

  return Ort::Status();
}

void CreateTanOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<TanOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime