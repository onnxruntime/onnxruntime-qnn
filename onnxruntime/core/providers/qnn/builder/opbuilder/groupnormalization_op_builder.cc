// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class GroupNormOpBuilder : public BaseOpBuilder {
 public:
  GroupNormOpBuilder() : BaseOpBuilder("GroupNormOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GroupNormOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
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

Ort::Status GroupNormOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnit& node_unit,
                                              const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // Check input type is float for CPU. Can't use Qnn Op validation API since it's before layout transformation
  ONNXTensorElementDataType input_type = inputs[0].type;
  RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, input_type,
                                             "QNN GroupNorm only supports float input for CPU backend."));

  RETURN_IF(outputs.size() > 1, "QNN GroupNorm only support 1 output.");

  TensorInfo input_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
  const std::vector<uint32_t>& input_shape = input_info.shape;
  const size_t input_rank = input_shape.size();

  if (input_rank <= 2) {
    return MAKE_EP_FAIL("QNN GroupNorm only supports input ranks greater than 2.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  uint32_t num_channels;
  if (node_unit.Domain() == kMSDomain) {
    // Handle channels_last attribute for com.microsoft.GroupNorm
    const int64_t channels_last = node_helper.Get("channels_last", static_cast<int64_t>(1));
    num_channels = (channels_last == 1) ? input_shape.back() : input_shape[1];
  } else {
    // Handle layout transformation - check if already transformed to NHWC
    num_channels = (node_unit.Domain() == kMSInternalNHWCDomain) ? input_shape.back() : input_shape[1];
  }

  TensorInfo scale_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], scale_info));
  const std::vector<uint32_t>& scale_shape = scale_info.shape;
  if (scale_shape.size() != 1 || scale_shape[0] != num_channels) {
    return MAKE_EP_FAIL(("QNN GroupNorm input 1 (scale/gamma) must have 1D shape [" + std::to_string(num_channels) + "].").c_str());
  }

  TensorInfo bias_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info));
  const std::vector<uint32_t>& bias_shape = bias_info.shape;
  if (bias_shape.size() != 1 || bias_shape[0] != num_channels) {
    return MAKE_EP_FAIL(("QNN GroupNorm input 2 (bias/beta) must have 1D shape [" + std::to_string(num_channels) + "].").c_str());
  }

  const float epsilon = node_helper.Get("epsilon", 1e-05f);
  if (epsilon <= 0.0f) {
    return MAKE_EP_FAIL("QNN GroupNorm epsilon must be greater than 0.0");
  }

  // Support both "num_groups" (ONNX GroupNormalization) and "groups" (com.microsoft.GroupNorm)
  int64_t num_groups;
  if (node_unit.OpType() == "GroupNormalization") {
    num_groups = node_helper.Get("num_groups", static_cast<int64_t>(1));
  } else {
    num_groups = node_helper.Get("groups", static_cast<int64_t>(1));
  }
  if (num_groups <= 0) {
    return MAKE_EP_FAIL("QNN GroupNorm num_groups/groups must be greater than 0");
  }

  if (num_channels % static_cast<uint32_t>(num_groups) != 0) {
    return MAKE_EP_FAIL("QNN GroupNorm requires num_channels to be divisible by num_groups/groups");
  }

  // Check activation attribute for com.microsoft.GroupNorm
  if (node_unit.OpType() == "GroupNorm") {
    const int64_t activation = node_helper.Get("activation", static_cast<int64_t>(0));
    if (activation != 0 && activation != 1) {
      return MAKE_EP_FAIL("QNN GroupNorm only supports activation=0 (None) or activation=1 (SiLU)");
    }
  }

  // Continue Op validation if it's NHWC transformed
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  return Ort::Status();
}

Ort::Status GroupNormOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnit& node_unit,
                                              const Ort::Logger& logger,
                                              std::vector<std::string>& input_names,
                                              bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));  // Input 0
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[1], logger, input_names));  // Scale
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, input_names));  // Bias

  return Ort::Status();
}

Ort::Status GroupNormOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            std::vector<std::string>&& input_names,
                                                            const Ort::Logger& logger,
                                                            bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  const auto& inputs = node_unit.Inputs();
  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));

  TensorInfo scale_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], scale_info));

  TensorInfo bias_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info));

  // Check if we need to cast scale and bias to match input dtype
  std::string scale_input_name = input_names[1];
  std::string bias_input_name = input_names[2];

  if (scale_info.qnn_data_type != input_info.qnn_data_type) {
    // Create Cast node for scale
    std::string casted_scale_name = utils::GetUniqueName(node_unit.Name() + "_scale_cast");
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(casted_scale_name,
                                                  scale_input_name,
                                                  casted_scale_name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  input_info.qnn_data_type,
                                                  QnnQuantParamsWrapper(),
                                                  std::move(scale_info.shape),
                                                  do_op_validation));

    scale_input_name = casted_scale_name;
  }

  if (bias_info.qnn_data_type != input_info.qnn_data_type) {
    // Create Cast node for bias
    std::string casted_bias_name = utils::GetUniqueName(node_unit.Name() + "_bias_cast");
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(casted_bias_name,
                                                  bias_input_name,
                                                  casted_bias_name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  input_info.qnn_data_type,
                                                  QnnQuantParamsWrapper(),
                                                  std::move(bias_info.shape),
                                                  do_op_validation));

    bias_input_name = casted_bias_name;
  }

  // Update input_names with potentially casted scale and bias
  std::vector<std::string> group_norm_input_names = {input_names[0], scale_input_name, bias_input_name};

  const float epsilon = node_helper.Get("epsilon", 1e-05f);
  Qnn_Scalar_t epsilon_param = QNN_SCALAR_INIT;
  epsilon_param.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_param.floatValue = epsilon;
  QnnParamWrapper epsilon_param_wrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        QNN_OP_GROUP_NORM_PARAM_EPSILON,
                                        epsilon_param);
  param_tensor_names.push_back(epsilon_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(epsilon_param_wrapper));

  // Support both "num_groups" (ONNX GroupNormalization) and "groups" (com.microsoft.GroupNorm)
  int64_t num_groups;
  if (node_unit.OpType() == "GroupNormalization") {
    num_groups = node_helper.Get("num_groups", static_cast<int64_t>(1));
  } else {
    num_groups = node_helper.Get("groups", static_cast<int64_t>(1));
  }
  Qnn_Scalar_t num_groups_param = QNN_SCALAR_INIT;
  num_groups_param.dataType = QNN_DATATYPE_UINT_32;
  num_groups_param.uint32Value = static_cast<uint32_t>(num_groups);
  QnnParamWrapper num_groups_param_wrapper(node_unit.Index(),
                                           node_unit.Name(),
                                           QNN_OP_GROUP_NORM_PARAM_GROUP,
                                           num_groups_param);
  param_tensor_names.push_back(num_groups_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(num_groups_param_wrapper));

  // Check if we need to add SiLU activation (activation=1)
  const int64_t activation = node_helper.Get("activation", static_cast<int64_t>(0));

  if (activation == 0) {
    // No activation, just process outputs normally
    return ProcessOutputs(qnn_model_wrapper, node_unit,
                          std::move(group_norm_input_names),
                          std::move(param_tensor_names),
                          logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
  }

  // activation == 1: Add SiLU activation (x * sigmoid(x))
  const auto& outputs = node_unit.Outputs();
  const std::string& final_output_name = outputs[0].name;

  // Create intermediate output for GroupNorm
  std::string group_norm_output_name = utils::GetUniqueName(node_unit.Name() + "_group_norm_out");
  QnnTensorWrapper group_norm_output_tensor(group_norm_output_name,
                                            QNN_TENSOR_TYPE_NATIVE,
                                            input_info.qnn_data_type,
                                            QnnQuantParamsWrapper(),
                                            std::vector<uint32_t>(input_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(group_norm_output_tensor)),
                "Failed to add group_norm_output tensor.");

  // Create GroupNorm node
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit.Name() + "_group_norm"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                GetQnnOpType(node_unit.OpType()),
                                                std::move(group_norm_input_names),
                                                {group_norm_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to create GroupNorm node.");

  // Create Sigmoid output tensor
  std::string sigmoid_output_name = utils::GetUniqueName(node_unit.Name() + "_sigmoid_out");
  QnnTensorWrapper sigmoid_output_tensor(sigmoid_output_name,
                                         QNN_TENSOR_TYPE_NATIVE,
                                         input_info.qnn_data_type,
                                         QnnQuantParamsWrapper(),
                                         std::vector<uint32_t>(input_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sigmoid_output_tensor)),
                "Failed to add sigmoid_output tensor.");

  // Create Sigmoid node
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit.Name() + "_sigmoid"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_SIGMOID,
                                                {group_norm_output_name},
                                                {sigmoid_output_name},
                                                {},
                                                do_op_validation),
                "Failed to create Sigmoid node.");

  // Create final output tensor
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.IsGraphOutput(final_output_name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper final_output_tensor(final_output_name,
                                       tensor_type,
                                       input_info.qnn_data_type,
                                       input_info.quant_param.Copy(),
                                       std::vector<uint32_t>(input_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)),
                "Failed to add final output tensor.");

  // Create ElementWiseMul node for x * sigmoid(x)
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit.Name() + "_silu_mul"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                {group_norm_output_name, sigmoid_output_name},
                                                {final_output_name},
                                                {},
                                                do_op_validation),
                "Failed to create SiLU multiply node.");

  return Ort::Status();
}

void CreateGroupNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GroupNormOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
