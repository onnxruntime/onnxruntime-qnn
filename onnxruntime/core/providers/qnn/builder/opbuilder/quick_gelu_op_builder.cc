// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class QuickGeluOpBuilder : public BaseOpBuilder {
 public:
  QuickGeluOpBuilder() : BaseOpBuilder("QuickGeluOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QuickGeluOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status QuickGeluOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const NodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const logging::Logger& logger,
                                                       bool do_op_validation) const {
  LOGS(logger, VERBOSE) << "Processing QuickGelu operator: " << node_unit.Name();

  // Get the input and output tensor name
  const std::string& input_name = input_names[0];
  const auto& outputs = node_unit.Outputs();
  const std::string& output_name = outputs[0].node_arg.Name();

  // Get the alpha attribute value (default is 1.0f)
  NodeAttrHelper node_helper(node_unit);
  float alpha = node_helper.Get("alpha", 1.0f);

  // Get input tensor info
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  // Create intermediate tensor names
  const std::string alpha_mul_output_name = utils::GetUniqueName(node_unit.Name() + "_alpha_mul");
  const std::string sigmoid_output_name = utils::GetUniqueName(node_unit.Name() + "_sigmoid");

  // Create alpha tensor for multiplication
  std::string alpha_tensor_name = utils::GetUniqueName(node_unit.Name() + "_alpha");
  Qnn_Scalar_t alpha_qnn_scalar = QNN_SCALAR_INIT;
  alpha_qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  alpha_qnn_scalar.floatValue = alpha;

  // Create alpha tensor wrapper
  std::vector<uint32_t> alpha_shape{1};
  std::vector<uint8_t> alpha_data(sizeof(float), 0);
  memcpy(alpha_data.data(), &alpha, sizeof(float));

  QnnTensorWrapper alpha_tensor_wrapper(alpha_tensor_name,
                                        QNN_TENSOR_TYPE_STATIC,
                                        input_info.qnn_data_type,
                                        QnnQuantParamsWrapper(),
                                        std::move(alpha_shape),
                                        std::move(alpha_data));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(alpha_tensor_wrapper)), "Failed to add alpha tensor.");

  // Create intermediate tensor wrappers
  // 1. alpha_mul_output: result of alpha * x
  // Using default quantization parameters for intermediate tensors as per guideline
  QnnTensorWrapper alpha_mul_output_tensor_wrapper(alpha_mul_output_name,
                                                   QNN_TENSOR_TYPE_NATIVE,
                                                   input_info.qnn_data_type,
                                                   QnnQuantParamsWrapper(),
                                                   std::vector<uint32_t>(input_info.shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(alpha_mul_output_tensor_wrapper)),
                    "Failed to add alpha_mul_output tensor.");

  // 2. sigmoid_output: result of sigmoid(alpha * x)
  // Using default quantization parameters for intermediate tensors as per guideline
  QnnTensorWrapper sigmoid_output_tensor_wrapper(sigmoid_output_name,
                                                 QNN_TENSOR_TYPE_NATIVE,
                                                 input_info.qnn_data_type,
                                                 QnnQuantParamsWrapper(),
                                                 std::vector<uint32_t>(input_info.shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sigmoid_output_tensor_wrapper)),
                    "Failed to add sigmoid_output tensor.");

  // 3. final_output: result of x * sigmoid(alpha * x)
  // Check if the output is a graph output and set the tensor type accordingly
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.IsGraphOutput(output_name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper output_tensor_wrapper(output_name,
                                         tensor_type,
                                         input_info.qnn_data_type,
                                         input_info.quant_param.Copy(),
                                         std::vector<uint32_t>(input_info.shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                    "Failed to add output tensor.");

  // Step 1: Create Mul node for alpha * x
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit.Name() + "_alpha_mul"),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                    {alpha_tensor_name, input_name},
                                                    {alpha_mul_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to create alpha_mul node.");

  // Step 2: Create Sigmoid node for sigmoid(alpha * x)
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit.Name() + "_sigmoid"),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_SIGMOID,
                                                    {alpha_mul_output_name},
                                                    {sigmoid_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to create sigmoid node.");

  // Step 3: Create Mul node for x * sigmoid(alpha * x)
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit.Name() + "_final_mul"),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                    {input_name, sigmoid_output_name},
                                                    {output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to create final_mul node.");

  return Status::OK();
}

void CreateQuickGeluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QuickGeluOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
