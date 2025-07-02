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

Status GroupNormOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const NodeUnit& node_unit,
                                         const logging::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  const auto input_count = inputs.size();
  constexpr size_t X_IDX = 0;
  constexpr size_t SCALE_IDX = 1;
  constexpr size_t BIAS_IDX = 2;

  // Input[0] (X, required)
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[X_IDX], logger, input_names));

  // Input[1] (scale, required)
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[SCALE_IDX], logger, input_names));

  // Input[2] (bias, optional)
  const bool has_bias_input = input_count > BIAS_IDX && inputs[BIAS_IDX].node_arg.Exists();
  if (has_bias_input) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[BIAS_IDX], logger, input_names));
  }

  return Status::OK();
}

Status GroupNormOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const NodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const logging::Logger& logger,
                                                       bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  const float epsilon = node_helper.Get("epsilon", 1e-05f);  // Default is 1e-05 according to ONNX spec.
  Qnn_Scalar_t epsilon_param = QNN_SCALAR_INIT;
  epsilon_param.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_param.floatValue = epsilon;
  QnnParamWrapper epsilon_param_wrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        QNN_OP_GROUP_NORM_PARAM_EPSILON,
                                        epsilon_param);
  param_tensor_names.push_back(epsilon_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(epsilon_param_wrapper));

  const int64_t num_groups = node_helper.Get("num_groups", 1);  // Default is 1 according to ONNX spec.
  Qnn_Scalar_t num_groups_param = QNN_SCALAR_INIT;
  num_groups_param.dataType = QNN_DATATYPE_UINT_32;
  num_groups_param.uint32Value = static_cast<uint32_t>(num_groups);
  QnnParamWrapper num_groups_param_wrapper(node_unit.Index(),
                                           node_unit.Name(),
                                           QNN_OP_GROUP_NORM_PARAM_GROUP,
                                           num_groups_param);
  param_tensor_names.push_back(num_groups_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(num_groups_param_wrapper));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Status::OK();
}

void CreateGroupNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GroupNormOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
