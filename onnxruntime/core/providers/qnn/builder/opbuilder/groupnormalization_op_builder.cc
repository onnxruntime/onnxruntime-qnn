// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

/**
 * Translates ONNX GroupNormalization (NCHW) into a QNN GroupNorm node (NHWC).
 *
 * QNN's GroupNorm requires channel-last layout, while ONNX GroupNormalization is channel-first.
 * ORT's layout transformer can perform this NCHW->NHWC conversion automatically, but it may clone
 * a GroupNormalization node when its NCHW input feeds multiple consumers (e.g. one layout-sensitive,
 * one not). The cloned node's output tensor does not always inherit a resolved shape, which later
 * fails QNN EP's own shape lookup ("Cannot get shape") and strands the node with no EP able to run
 * it (since QNN EP requested the layout change but then rejects the resulting node).
 *
 * To avoid depending on ORT's layout-transform/node-duplication path, QnnEp::ShouldConvertDataLayoutForOp
 * suppresses the automatic layout transform for GroupNormalization, and this builder performs the
 * NCHW<->NHWC conversion internally (same strategy QLinearConvOpBuilder uses for QLinearConv and
 * DQConvIntegerFusion uses for ConvInteger):
 *
 *   X (NCHW) --Transpose--> X_nhwc --GroupNorm(scale, bias)--> Y_nhwc --Transpose--> Y (NCHW)
 */
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

 private:
  Ort::Status CreateOrValidate(QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit,
                               const Ort::Logger& logger,
                               bool do_op_validation) const ORT_MUST_USE_RESULT;
};

Ort::Status GroupNormalizationOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& logger) const {
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

  // GroupNormalizationOpBuilder always consumes/produces the ONNX (NCHW) domain: layout conversion
  // is suppressed via ShouldConvertDataLayoutForOp, so node_unit.Domain() is never kMSInternalNHWCDomain.
  const uint32_t num_channels = input_shape[1];

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

  // Validate the full QNN graph (internal transposes + GroupNorm) via the QNN validation API.
  return CreateOrValidate(qnn_model_wrapper, node_unit, logger, /*do_op_validation=*/true);
}

Ort::Status GroupNormalizationOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& logger,
                                                       std::vector<std::string>& input_names,
                                                       bool do_op_validation) const {
  // All graph construction happens in ProcessAttributesAndOutputs via CreateOrValidate, which
  // needs the node's attributes (epsilon, num_groups) alongside its inputs.
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(input_names);
  ORT_UNUSED_PARAMETER(do_op_validation);
  return Ort::Status();
}

Ort::Status GroupNormalizationOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                     const OrtNodeUnit& node_unit,
                                                                     std::vector<std::string>&& input_names,
                                                                     const Ort::Logger& logger,
                                                                     bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(input_names);
  return CreateOrValidate(qnn_model_wrapper, node_unit, logger, do_op_validation);
}

Ort::Status GroupNormalizationOpBuilder::CreateOrValidate(QnnModelWrapper& qnn_model_wrapper,
                                                          const OrtNodeUnit& node_unit,
                                                          const Ort::Logger& logger,
                                                          bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  TensorInfo input_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
  const std::vector<uint32_t> nchw_shape = input_info.shape;
  const size_t rank = nchw_shape.size();

  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));
  RETURN_IF_NOT(output_info.shape.size() == rank, "GroupNorm input/output rank mismatch");

  const std::vector<uint32_t> cf_to_cl = utils::ChannelFirstToLastPerm(rank);
  const std::vector<uint32_t> cl_to_cf = utils::ChannelLastToFirstPerm(rank);
  const std::vector<uint32_t> nhwc_in_shape = utils::ApplyPermToShape(nchw_shape, cf_to_cl);
  const std::vector<uint32_t> nhwc_out_shape = utils::ApplyPermToShape(output_info.shape, cf_to_cl);

  const std::string node_base = utils::UniqueNameGenerator().New(node_unit);
  const std::string x_name = inputs[0].name;
  const std::string& output_name = outputs[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  // Step 1: Transpose input NCHW -> NHWC.
  const std::string x_nhwc_name = node_base + "_x_nhwc";
  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), x_name, x_nhwc_name,
                                                     nchw_shape, cf_to_cl, nhwc_in_shape,
                                                     input_info.qnn_data_type, QnnQuantParamsWrapper(),
                                                     do_op_validation,
                                                     /*is_for_input=*/qnn_model_wrapper.IsGraphInput(x_name),
                                                     /*is_for_output=*/false));

  // Step 2: scale/bias inputs are already 1D [channel]; no layout conversion needed.
  std::vector<std::string> group_norm_input_names = {x_nhwc_name};
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[1], logger, group_norm_input_names));
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, group_norm_input_names));

  // Step 3: epsilon / num_groups params.
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;
  const float epsilon = node_helper.Get("epsilon", 1e-05f);
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), epsilon,
                                      QNN_OP_GROUP_NORM_PARAM_EPSILON, param_tensor_names));

  const int64_t num_groups = node_helper.Get("num_groups", static_cast<int64_t>(1));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(num_groups),
                                         QNN_OP_GROUP_NORM_PARAM_GROUP, param_tensor_names));

  // Step 4: GroupNorm node, producing an NHWC output.
  const std::string group_norm_out_name = node_base + "_y_nhwc";
  if (do_op_validation) {
    QnnTensorWrapper x_nhwc_tensor(x_nhwc_name, QNN_TENSOR_TYPE_NATIVE, input_info.qnn_data_type,
                                   QnnQuantParamsWrapper(), std::vector<uint32_t>(nhwc_in_shape));
    QnnTensorWrapper scale_tensor;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[1], scale_tensor));
    QnnTensorWrapper bias_tensor;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[2], bias_tensor));
    QnnTensorWrapper group_norm_out_tensor(group_norm_out_name, QNN_TENSOR_TYPE_NATIVE, output_info.qnn_data_type,
                                           QnnQuantParamsWrapper(), std::vector<uint32_t>(nhwc_out_shape));

    Qnn_Scalar_t epsilon_scalar = QNN_SCALAR_INIT;
    epsilon_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    epsilon_scalar.floatValue = epsilon;
    QnnParamWrapper epsilon_param(node_unit.Index(), group_norm_out_name, QNN_OP_GROUP_NORM_PARAM_EPSILON,
                                  epsilon_scalar);

    Qnn_Scalar_t group_scalar = QNN_SCALAR_INIT;
    group_scalar.dataType = QNN_DATATYPE_UINT_32;
    group_scalar.uint32Value = static_cast<uint32_t>(num_groups);
    QnnParamWrapper group_param(node_unit.Index(), group_norm_out_name, QNN_OP_GROUP_NORM_PARAM_GROUP,
                                group_scalar);

    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(group_norm_out_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_GROUP_NORM,
                                                      {x_nhwc_tensor.GetQnnTensor(), scale_tensor.GetQnnTensor(),
                                                       bias_tensor.GetQnnTensor()},
                                                      {group_norm_out_tensor.GetQnnTensor()},
                                                      {epsilon_param.GetQnnParam(), group_param.GetQnnParam()}));
  } else {
    QnnTensorWrapper group_norm_out_tensor(group_norm_out_name, QNN_TENSOR_TYPE_NATIVE, output_info.qnn_data_type,
                                           QnnQuantParamsWrapper(), std::vector<uint32_t>(nhwc_out_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(group_norm_out_tensor)),
                  "Failed to add GroupNorm output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(group_norm_out_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_GROUP_NORM,
                                                  std::move(group_norm_input_names),
                                                  {group_norm_out_name},
                                                  std::move(param_tensor_names),
                                                  /*do_op_validation=*/false),
                  "Failed to create GroupNorm node.");
  }

  // Step 5: Transpose output NHWC -> NCHW.
  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), group_norm_out_name, output_name,
                                                     nhwc_out_shape, cl_to_cf, output_info.shape,
                                                     output_info.qnn_data_type, QnnQuantParamsWrapper(),
                                                     do_op_validation,
                                                     /*is_for_input=*/false,
                                                     /*is_for_output=*/is_graph_output));

  return Ort::Status();
}

void CreateGroupNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GroupNormalizationOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
