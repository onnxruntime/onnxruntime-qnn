// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class RoiAlignOpBuilder : public BaseOpBuilder {
 public:
  RoiAlignOpBuilder() : BaseOpBuilder("RoiAlignOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoiAlignOpBuilder);

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

Status RoiAlignOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  NodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }
                                    
  return Status::OK();
}

Status RoiAlignOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {

  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(input_names);
  ORT_UNUSED_PARAMETER(do_op_validation);

  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));                                                  
  const std::string& input_name = input_names[0];

  std::vector<std::string> param_tensor_names;

  NodeAttrHelper node_helper(node_unit);

  // Add QNN_OP_ROI_ALIGN_PARAM_ALIGNED
  // Qnn_Scalar_t qnn_roi_align_aligned = QNN_SCALAR_INIT;
  // qnn_roi_align_aligned.dataType = QNN_DATATYPE_BOOL_8;
  // qnn_roi_align_aligned.bool8Value = static_cast<uint8_t>(0);

  // QnnParamWrapper qnn_roi_align_aligned_param(node_unit.Index(), node_unit.Name(),
  //                                         QNN_OP_ROI_ALIGN_PARAM_ALIGNED, qnn_roi_align_aligned);

  // param_tensor_names.push_back(qnn_roi_align_aligned_param.GetParamTensorName());
  // qnn_model_wrapper.AddParamWrapper(std::move(qnn_roi_align_aligned_param));

  // Add QNN_OP_ROI_ALIGN_PARAM_ALLOW_INVALID_ROI
  // Qnn_Scalar_t qnn_roi_align_allow_invalid_roi = QNN_SCALAR_INIT;
  // qnn_roi_align_allow_invalid_roi.dataType = QNN_DATATYPE_BOOL_8;
  // qnn_roi_align_allow_invalid_roi.bool8Value = static_cast<uint8_t>(0);

  // QnnParamWrapper qnn_roi_align_allow_invalid_roi_param(node_unit.Index(), node_unit.Name(),
  //                                         QNN_OP_ROI_ALIGN_PARAM_ALLOW_INVALID_ROI, qnn_roi_align_allow_invalid_roi);

  // param_tensor_names.push_back(qnn_roi_align_allow_invalid_roi_param.GetParamTensorName());
  // qnn_model_wrapper.AddParamWrapper(std::move(qnn_roi_align_allow_invalid_roi_param));

  // QNN_OP_ROI_ALIGN_PARAM_IMG_SIZE_RATIO
  std::vector<uint32_t> img_size_ratio_shape{static_cast<uint32_t>(2)};
  std::vector<float> img_size_ratio{16.0f, 16.0f};

  QnnParamWrapper roi_align_param_img_size_ratio(node_unit.Index(),
                                 node_unit.Name(),
                                 QNN_OP_ROI_ALIGN_PARAM_IMG_SIZE_RATIO,
                                 std::move(img_size_ratio_shape),
                                 std::move(img_size_ratio));

  param_tensor_names.push_back(roi_align_param_img_size_ratio.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(roi_align_param_img_size_ratio));


  // Transpose feature: [N C H W] to [N H W C]
  const std::string& transposed_feature_name = utils::GetUniqueName(node_unit, "_transposed_feature");
  std::vector<uint32_t> transpose_perm{0, 2, 3, 1};
  std::vector<uint32_t> transpose_output_shape = input_info.shape;
  transpose_output_shape[1] = input_info.shape[2];
  transpose_output_shape[2] = input_info.shape[3];
  transpose_output_shape[3] = input_info.shape[1];
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                          input_name,
                                                          transposed_feature_name,
                                                          input_info.shape,
                                                          transpose_perm,
                                                          transpose_output_shape,
                                                          input_info.qnn_data_type,
                                                          input_info.quant_param,
                                                          do_op_validation,
                                                          false,
                                                          false));

  std::vector<std::string> roi_align_input_names{transposed_feature_name};
  roi_align_input_names.push_back(input_names[1]);
  roi_align_input_names.push_back(input_names[2]);

  // 
  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
  const bool& is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
  std::vector<uint32_t> output_shape = output_info.shape;
  output_shape[1] = output_info.shape[2];
  output_shape[2] = output_info.shape[3];
  output_shape[3] = output_info.shape[1];
  Qnn_TensorType_t op_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper roialign_output(org_output_name,
                                op_output_tensor_type,
                                input_info.qnn_data_type,
                                QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(roialign_output)), "Failed to add tensor.");
  
  const std::string& roialign_name = utils::GetUniqueName(node_unit, "_roialign");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(roialign_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_ROI_ALIGN,
                                                    std::move(roi_align_input_names),
                                                    {org_output_name},
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to add Roi align node.");

  return Status::OK();
}

void CreateRoiAlignOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RoiAlignOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
