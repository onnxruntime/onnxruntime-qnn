// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

class GroupNormOpBuilder : public BaseOpBuilder {
 public:
  GroupNormOpBuilder() : BaseOpBuilder("GroupNormOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GroupNormOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status GroupNormOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const NodeUnit& node_unit,
                                         const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // Check input type is float for CPU. Can't use Qnn Op validation API since it's before layout transformation
  ORT_RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].node_arg.Type()));

  ORT_RETURN_IF(outputs.size() > 1, "QNN GroupNorm only support 1 output.");

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0");
  const size_t input_rank = input_shape.size();

  if (input_rank <= 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN GroupNorm only supports input ranks greater than 2.");
  }

  // Handle layout transformation - check if already transformed to NHWC
  const uint32_t num_channels = (node_unit.Domain() == kMSInternalNHWCDomain) ? input_shape.back() : input_shape[1];

  std::vector<uint32_t> scale_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].node_arg, scale_shape), "Cannot get shape of input 1 (scale)");
  if (scale_shape.size() != 1 || scale_shape[0] != num_channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN GroupNorm input 1 (scale) must have 1D shape [channel].");
  }

  std::vector<uint32_t> bias_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[2].node_arg, bias_shape), "Cannot get shape of input 2 (bias)");
  if (bias_shape.size() != 1 || bias_shape[0] != num_channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN GroupNorm input 2 (bias) must have 1D shape [channel].");
  }

  NodeAttrHelper node_helper(node_unit);
  const float epsilon = node_helper.Get("epsilon", 1e-05f);
  if (epsilon <= 0.0f) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN GroupNorm epsilon must be greater than 0.0");
  }

  const int64_t num_groups = node_helper.Get("num_groups", static_cast<int64_t>(1));
  if (num_groups <= 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN GroupNorm num_groups must be greater than 0");
  }

  if (num_channels % static_cast<uint32_t>(num_groups) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN GroupNorm requires num_channels to be divisible by num_groups");
  }

  // Continue Op validation if it's NHWC transformed
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  return Status::OK();
}

Status GroupNormOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const NodeUnit& node_unit,
                                         const logging::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();

  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));  // Input 0
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[1], logger, input_names));  // Scale
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, input_names));  // Bias

  return Status::OK();
}

Status GroupNormOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const NodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const logging::Logger& logger,
                                                       bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

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

  const int64_t num_groups = node_helper.Get("num_groups", static_cast<int64_t>(1));
  Qnn_Scalar_t num_groups_param = QNN_SCALAR_INIT;
  num_groups_param.dataType = QNN_DATATYPE_UINT_32;
  num_groups_param.uint32Value = static_cast<uint32_t>(num_groups);
  QnnParamWrapper num_groups_param_wrapper(node_unit.Index(),
                                           node_unit.Name(),
                                           QNN_OP_GROUP_NORM_PARAM_GROUP,
                                           num_groups_param);
  param_tensor_names.push_back(num_groups_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(num_groups_param_wrapper));

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateGroupNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GroupNormOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
