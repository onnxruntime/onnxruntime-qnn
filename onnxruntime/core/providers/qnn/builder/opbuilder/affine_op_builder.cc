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

class AffineOpBuilder : public BaseOpBuilder {
 public:
  AffineOpBuilder() : BaseOpBuilder("AffineOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AffineOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names, const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status AffineOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names, const logging::Logger& logger,
                                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  NodeAttrHelper node_helper(node_unit);

  // Affine - Add( Mul( X, alpha), Beta)

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

  // Get Alpha parameter & prepare it as input tensor to mul operator
  auto alpha = node_helper.Get("alpha", (float)1.0f);
  const std::string alpha_name = input_name + "_ort_qnn_ep_alpha";
  std::vector<uint32_t> alpha_shape{1};
  std::vector<uint8_t> alpha_data(sizeof(float));
  std::memcpy(alpha_data.data(), &alpha, sizeof(float));
  QnnTensorWrapper alpha_tensorwrapper(alpha_name, QNN_TENSOR_TYPE_STATIC, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::move(alpha_shape), std::move(alpha_data));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(alpha_tensorwrapper)), "AddTensorWrapper failed");

  // Get Beta parameter & prepare it as input tensor to mul operator
  auto beta = node_helper.Get("beta", (float)0.0f);
  const std::string beta_name = input_name + "_ort_qnn_ep_beta";
  std::vector<uint32_t> beta_shape{1};
  std::vector<uint8_t> beta_data(sizeof(float));
  std::memcpy(beta_data.data(), &beta, sizeof(float));
  QnnTensorWrapper beta_tensorwrapper(beta_name, QNN_TENSOR_TYPE_STATIC, qnn_data_type, QnnQuantParamsWrapper(),
                                      std::move(beta_shape), std::move(beta_data));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(beta_tensorwrapper)), "AddTensorWrapper failed");

  // Step-1 : Mul(X, alpha)
  const std::string mul_name = input_name + "_ort_qnn_ep_mul";
  QnnTensorWrapper mul_tensorwrapper(mul_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type, QnnQuantParamsWrapper(),
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

  // Step-2: Add(mul_alpha, beta)
  Qnn_TensorType_t output_tensor_type = qnn_model_wrapper.IsGraphOutput(output.node_arg.Name()) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper output_tensorwrapper(output.node_arg.Name(), output_tensor_type, qnn_data_type,
                                        QnnQuantParamsWrapper(), std::move(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "AddTensorWrapper failed");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(input_name + "_ort_qnn_ep_mul_add",
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ELEMENT_WISE_ADD,
                                                    {mul_name, beta_name},
                                                    {output.node_arg.Name()},
                                                    {},
                                                    do_op_validation),
                    "CreateQnnNode failed");

  return Status::OK();
}

void CreateAffineOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<AffineOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime