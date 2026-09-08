// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

class GroupNormalizationOpBuilder : public BaseOpBuilder {
 public:
  GroupNormalizationOpBuilder() : BaseOpBuilder("GroupNormalizationOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GroupNormalizationOpBuilder);

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

Ort::Status GroupNormalizationOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
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

  // Handle layout transformation - check if already transformed to NHWC
  const uint32_t num_channels = (node_unit.Domain() == kMSInternalNHWCDomain) ? input_shape.back() : input_shape[1];

  TensorInfo scale_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], scale_info));
  const std::vector<uint32_t>& scale_shape = scale_info.shape;
  if (scale_shape.size() != 1 || scale_shape[0] != num_channels) {
    return MAKE_EP_FAIL("QNN GroupNorm input 1 (scale) must have 1D shape [channel].");
  }

  TensorInfo bias_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info));
  const std::vector<uint32_t>& bias_shape = bias_info.shape;
  if (bias_shape.size() != 1 || bias_shape[0] != num_channels) {
    return MAKE_EP_FAIL("QNN GroupNorm input 2 (bias) must have 1D shape [channel].");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const float epsilon = node_helper.Get("epsilon", 1e-05f);
  if (epsilon <= 0.0f) {
    return MAKE_EP_FAIL("QNN GroupNorm epsilon must be greater than 0.0");
  }

  const int64_t num_groups = node_helper.Get("num_groups", static_cast<int64_t>(1));
  if (num_groups <= 0) {
    return MAKE_EP_FAIL("QNN GroupNorm num_groups must be greater than 0");
  }

  if (num_channels % static_cast<uint32_t>(num_groups) != 0) {
    return MAKE_EP_FAIL("QNN GroupNorm requires num_channels to be divisible by num_groups");
  }

  // Continue Op validation if it's NHWC transformed
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  return Ort::Status();
}

Ort::Status GroupNormalizationOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
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

Ort::Status GroupNormalizationOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                     const OrtNodeUnit& node_unit,
                                                                     std::vector<std::string>&& input_names,
                                                                     const Ort::Logger& logger,
                                                                     bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  const float epsilon = node_helper.Get("epsilon", 1e-05f);
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), epsilon,
                                      QNN_OP_GROUP_NORM_PARAM_EPSILON, param_tensor_names));

  const int64_t num_groups = node_helper.Get("num_groups", static_cast<int64_t>(1));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(num_groups),
                                         QNN_OP_GROUP_NORM_PARAM_GROUP, param_tensor_names));

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateGroupNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GroupNormalizationOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
