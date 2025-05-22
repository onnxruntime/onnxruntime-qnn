// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class QuickGeluOpBuilder : public BaseOpBuilder {
 public:
  QuickGeluOpBuilder() : BaseOpBuilder("QuickGeluOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QuickGeluOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names, const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status QuickGeluOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names, const logging::Logger& logger,
                                                       bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  NodeAttrHelper node_helper(node_unit);
  // QuickGelu - X * Sigmoid (X * alpha)

  // Get Input and Output
  const auto& input = node_unit.Inputs()[0];
  const auto& output = node_unit.Outputs()[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.node_arg, input_shape), "Cannot get input shape.");
  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, output_shape), "Cannot get output shape.");
  ORT_ENFORCE(!input.quant_param.has_value(), "Input tensor must not be quantized.");
  const auto* type_proto = output.node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false, type_proto, qnn_data_type));
  const std::string input_name = input_names[0];

  // Get Alpha parameter
  auto alpha = node_helper.Get("alpha", (float)1.702);

  // Step-1: Add Mul(X, alpha)
  // skip if alpha = 1.0
  // if (alpha != 1.0) {

  // Prepare alpha as input tensor to mul operator
  const std::string alpha_name = input_name + "_ort_qnn_ep_alpha";
  std::vector<uint32_t> alpha_shape{1};
  std::vector<uint8_t> unpackage_data(sizeof(float));
  std::memcpy(unpackage_data.data(), &alpha, sizeof(float));
  QnnTensorWrapper alpha_tensorwrapper(alpha_name, QNN_TENSOR_TYPE_STATIC, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::move(alpha_shape), std::move(unpackage_data));

  const std::string mul_name = input_name + "_ort_qnn_ep_mul";
  QnnTensorWrapper mul_tensorwrapper(mul_name, QNN_TENSOR_TYPE_UPDATEABLE_NATIVE, qnn_data_type, QnnQuantParamsWrapper(),
                                     std::move(input_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mul_tensorwrapper)), "AddTensorWrapper failed");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(mul_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                    {input_name, alpha_name},
                                                    {mul_name},
                                                    {},
                                                    do_op_validation),
                    "CreateQnnNode failed");

  // }

  // Step-2: Add Sigmoid(y_mul)
  const std::string sigmoid_name = input_name + "_ort_qnn_ep_mul_sigmoid";
  QnnTensorWrapper sigmoid_tensorwrapper(sigmoid_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type, QnnQuantParamsWrapper(),
                                         std::vector<uint32_t>(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sigmoid_tensorwrapper)), "AddTensorwrapper failed");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(sigmoid_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_SIGMOID,
                                                    {mul_name},
                                                    {sigmoid_name},
                                                    {},
                                                    do_op_validation),
                    "CreateQnnNode failed");

  // Step-3: Add mul(y_sigmoid)
  Qnn_TensorType_t output_tensor_type = qnn_model_wrapper.IsGraphOutput(output.node_arg.Name()) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper mul2_tensorwrapper(output.node_arg.Name(), output_tensor_type, qnn_data_type,
                                      QnnQuantParamsWrapper(), std::move(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mul2_tensorwrapper)), "AddTensorWrapper failed");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(input_name + "_ort_qnn_ep_mul_sigmoid_mul",
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                    {input_name, sigmoid_name},
                                                    {output.node_arg.Name()},
                                                    {},
                                                    do_op_validation),
                    "CreateQnnNode failed");
  return Status::OK();
}

void CreateQuickGeluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QuickGeluOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime