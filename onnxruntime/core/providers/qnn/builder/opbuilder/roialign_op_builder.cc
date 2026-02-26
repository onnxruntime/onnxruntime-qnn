// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "QnnOpDef.h"

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class RoiAlignOpBuilder : public BaseOpBuilder {
 public:
  RoiAlignOpBuilder() : BaseOpBuilder("RoiAlignOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoiAlignOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const final ORT_MUST_USE_RESULT;

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

Ort::Status RoiAlignOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const Ort::Logger& logger) const {
  OrtNodeAttrHelper node_helper(node_unit);

  // Sanity checks on RoiAlign onnx attrs
  // HTP supports only align=False which corresponds to coordinate_transformation_mode="output_half_pixel"
  std::string coordinate_transformation_mode = node_helper.Get("coordinate_transformation_mode", "half_pixel");  // ONNX default "half_pixel"
  RETURN_IF_NOT(coordinate_transformation_mode == "output_half_pixel", "HTP only supports coordinate_transformation_mode=output_half_pixel.");

  // HTP supports only average pooling
  std::string mode = node_helper.Get("mode", "avg");  // ONNX default "avg"
  RETURN_IF_NOT(mode == "avg", "HTP only supports avg pooling mode.");

  // HTP doesn't support sampling_ratio = 0(adaptive mode)
  int sampling_ratio = node_helper.Get("sampling_ratio", 0);
  RETURN_IF_NOT(sampling_ratio != 0, "HTP doesn't support sampling ratio = 0.");

  // Sanity check on output_width output_height
  // Output tensor size [N, C, H, W]
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  RETURN_IF_NOT(4 == static_cast<int>(output_info.shape.size()), "Roialign requires 4d output (num_rois, C, output_height, output_width).");
  int output_height = node_helper.Get("output_height", 1);
  RETURN_IF_NOT(output_height == static_cast<int>(output_info.shape[2]), "Expect output_height == output_tensor.shape[2]");
  int output_width = node_helper.Get("output_width", 1);
  RETURN_IF_NOT(output_width == static_cast<int>(output_info.shape[3]), "Expect output_width == output_tensor.shape[3]");

  // RoiAlignOp are sensitive with data layout, requires NHWC data layout.
  // Continue RoiAlign if NHWC format.
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  return Ort::Status();
}

Ort::Status RoiAlignOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const Ort::Logger& logger,
                                             std::vector<std::string>& input_names,
                                             bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }

  return Ort::Status();
}

Ort::Status RoiAlignOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                           const OrtNodeUnit& node_unit,
                                                           std::vector<std::string>&& input_names,
                                                           const Ort::Logger& logger,
                                                           bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;
  std::vector<uint32_t> img_size_ratio_shape{static_cast<uint32_t>(2)};

  // Convert onnx::spatial_scale to htp::img_size_ratio
  float spatial_scale = node_helper.Get("spatial_scale", 1.0f);
  RETURN_IF(spatial_scale == 0, "Roialign got invalid spatial_scale=0");
  std::vector<float> img_size_ratio{1 / spatial_scale, 1 / spatial_scale};
  QnnParamWrapper roi_align_param_img_size_ratio = createQnnParamWrapper<float>(
      node_unit.Index(),
      node_unit.Name(),
      QNN_OP_ROI_ALIGN_PARAM_IMG_SIZE_RATIO,
      std::move(img_size_ratio_shape),
      std::move(img_size_ratio));
  param_tensor_names.push_back(roi_align_param_img_size_ratio.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(roi_align_param_img_size_ratio));

  return ProcessOutputs(qnn_model_wrapper,
                        node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, QNN_OP_ROI_ALIGN);
}

void CreateRoiAlignOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RoiAlignOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime