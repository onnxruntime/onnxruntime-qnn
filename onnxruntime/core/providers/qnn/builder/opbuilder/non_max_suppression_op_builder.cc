// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "QnnOpDef.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// NonMaxSuppression op builder.
//
// ONNX NonMaxSuppression has 2 mandatory inputs (boxes, scores) and 3 optional scalar inputs
// (max_output_boxes_per_class, iou_threshold, score_threshold) which map to QNN params.
//
// QNN NMS output is padded to [batch * num_classes * max_boxes_selected, 3] as INT_32.
// ONNX output is unpadded [num_selected, 3] as INT_64.
// For graph outputs, a Cast (INT_32 -> INT_64) is appended to satisfy the ONNX INT_64 requirement.
//
// Backend support:
//   CPU  — float32 inputs; supported.
//   HTP  — quantized (QDQ) inputs; supported. The INT_32->INT_64 Cast for graph outputs works
//           on HTP (same pattern as NonZero op builder).
//   GPU  — not listed in QNN GPU backend supported ops; rejected.
class NonMaxSuppressionOpBuilder : public BaseOpBuilder {
 public:
  NonMaxSuppressionOpBuilder() : BaseOpBuilder("NonMaxSuppressionOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NonMaxSuppressionOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  // Return INT_32 for output 0: QNN native NMS output type.
  // The base pipeline will insert INT_32->INT_64 Cast for graph outputs automatically
  // via ProcessAttributesAndOutputs below (we handle it manually, so this override is a
  // hint for any type-check paths in the base class).
  Qnn_DataType_t GetSupportedOutputDataType(size_t index, Qnn_DataType_t qnn_data_type) const override {
    if (index == 0 && qnn_data_type == QNN_DATATYPE_INT_64) {
      return QNN_DATATYPE_INT_32;
    }
    return qnn_data_type;
  }

  // QNN GPU backend does not list NonMaxSuppression in its supported ops.
  Ort::Status CheckGpuDataTypes(const std::vector<Qnn_DataType_t>,
                                const std::vector<Qnn_DataType_t>) const override {
    return MAKE_EP_FAIL("NonMaxSuppression is not supported on QNN GPU backend.");
  }

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

Ort::Status NonMaxSuppressionOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      const Ort::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();

  // Validate boxes input: must be rank 3 with last dim == 4.
  TensorInfo boxes_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], boxes_info));
  RETURN_IF(boxes_info.shape.size() != 3, "NonMaxSuppression: boxes must be rank 3.");
  RETURN_IF(boxes_info.shape[2] != 4, "NonMaxSuppression: boxes last dimension must be 4.");

  // Validate scores input: must be rank 3.
  TensorInfo scores_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], scores_info));
  RETURN_IF(scores_info.shape.size() != 3, "NonMaxSuppression: scores must be rank 3.");

  // Validate optional scalar inputs [2..4]: if present, must be constant initializers.
  for (size_t i = 2; i <= 4; ++i) {
    if (inputs.size() > i && inputs[i].Exists()) {
      RETURN_IF(!qnn_model_wrapper.IsConstantInput(inputs[i].name),
                "NonMaxSuppression: optional scalar inputs (max_output_boxes_per_class, "
                "iou_threshold, score_threshold) must be constant initializers.");
    }
  }

  // Gate on max_output_boxes_per_class. In ONNX it is optional and defaults to 0, where 0
  // (or absent) legally means "no boxes selected". QNN pads out[0] to
  // [batch * num_classes * max_boxes_selected, 3], so a value of 0 yields a degenerate
  // [0, 3] tensor that QNN rejects at graphAddNode. Decline the node here so it falls back
  // to the CPU EP instead of hard-failing at compose.
  if (inputs.size() > 2 && inputs[2].Exists()) {
    std::vector<uint8_t> buf;
    const auto* t = qnn_model_wrapper.GetConstantTensor(inputs[2].name);
    RETURN_IF(t == nullptr, "NonMaxSuppression: failed to get constant tensor for max_output_boxes_per_class.");
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(t, buf));
    int64_t max_output_boxes_per_class = *reinterpret_cast<const int64_t*>(buf.data());
    RETURN_IF(max_output_boxes_per_class <= 0,
              "NonMaxSuppression: max_output_boxes_per_class must be > 0 for QNN "
              "(0 or absent selects no boxes and yields an invalid [0, 3] output).");
  } else {
    // max_output_boxes_per_class absent → defaults to 0 → no boxes selected.
    return MAKE_EP_FAIL(
        "NonMaxSuppression: max_output_boxes_per_class must be provided "
        "and > 0 for QNN (absent defaults to 0, which selects no boxes).");
  }

  // Validate center_point_box attribute.
  // QNN's NMS op only supports the diagonal-corners box format (center_point_box == 0).
  // Although the QNN op-def schema lists a center_point_box param (so validateOpConfig accepts
  // it), QNN's own ONNX converter rejects center_point_box != 0 outright and never emits the
  // param. The QnnCpu kernel rejects the node at graphAddNode (error 6000) when the param is
  // present at all, so we omit it entirely (see ProcessAttributesAndOutputs) and decline any
  // model that actually requests center-point format.
  int64_t center_point_box = OrtNodeAttrHelper(node_unit).Get("center_point_box", int64_t{0});
  RETURN_IF(center_point_box != 0,
            "NonMaxSuppression: QNN only supports center_point_box == 0 (diagonal corners).");

  return BaseOpBuilder::IsOpSupported(qnn_model_wrapper, node_unit, logger);
}

Ort::Status NonMaxSuppressionOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      const Ort::Logger& logger,
                                                      std::vector<std::string>& input_names,
                                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  // Only add inputs[0] (boxes) and inputs[1] (scores) as QNN graph inputs.
  // inputs[2..4] (max_output_boxes_per_class, iou_threshold, score_threshold)
  // are extracted as constant scalar QNN params in ProcessAttributesAndOutputs.
  const auto& inputs = node_unit.Inputs();
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[1], logger, input_names));

  return Ort::Status();
}

Ort::Status NonMaxSuppressionOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                    const OrtNodeUnit& node_unit,
                                                                    std::vector<std::string>&& input_names,
                                                                    const Ort::Logger& logger,
                                                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // 1. Extract shapes from mandatory inputs
  TensorInfo boxes_info = {};
  TensorInfo scores_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], boxes_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], scores_info));

  uint32_t batch = boxes_info.shape[0];
  uint32_t num_classes = scores_info.shape[1];

  // 2. Extract scalar params from constant initializers
  // max_output_boxes_per_class (ONNX int64 → QNN uint32)
  uint32_t max_boxes_selected = 0;
  if (inputs.size() > 2 && inputs[2].Exists()) {
    std::vector<uint8_t> buf;
    const auto* t = qnn_model_wrapper.GetConstantTensor(inputs[2].name);
    RETURN_IF(t == nullptr, "NonMaxSuppression: failed to get constant tensor for max_output_boxes_per_class.");
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(t, buf));
    int64_t val = *reinterpret_cast<const int64_t*>(buf.data());
    max_boxes_selected = SafeInt<uint32_t>(val);
  }

  // iou_threshold (ONNX float, default 0.0f)
  float iou_threshold = 0.0f;
  if (inputs.size() > 3 && inputs[3].Exists()) {
    std::vector<uint8_t> buf;
    const auto* t = qnn_model_wrapper.GetConstantTensor(inputs[3].name);
    RETURN_IF(t == nullptr, "NonMaxSuppression: failed to get constant tensor for iou_threshold.");
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(t, buf));
    iou_threshold = *reinterpret_cast<const float*>(buf.data());
  }

  // score_threshold (ONNX float, default 0.0f)
  float score_threshold = 0.0f;
  if (inputs.size() > 4 && inputs[4].Exists()) {
    std::vector<uint8_t> buf;
    const auto* t = qnn_model_wrapper.GetConstantTensor(inputs[4].name);
    RETURN_IF(t == nullptr, "NonMaxSuppression: failed to get constant tensor for score_threshold.");
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(t, buf));
    score_threshold = *reinterpret_cast<const float*>(buf.data());
  }

  // 3. Build QNN scalar params
  // Note: center_point_box is intentionally NOT emitted. QNN's NMS op only supports the
  // diagonal-corners format.
  std::vector<std::string> param_tensor_names;

  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                      iou_threshold,
                                      QNN_OP_NON_MAX_SUPPRESSION_PARAM_IOU_THRESHOLD,
                                      param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                      score_threshold,
                                      QNN_OP_NON_MAX_SUPPRESSION_PARAM_SCORE_THRESHOLD,
                                      param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         max_boxes_selected,
                                         QNN_OP_NON_MAX_SUPPRESSION_PARAM_MAX_BOXES_SELECTED,
                                         param_tensor_names));

  // 4. Compute fixed padded output shape
  // QNN pads the output to [batch * num_classes * max_boxes_selected, 3].
  SafeInt<uint32_t> max_selected = SafeInt<uint32_t>(batch) * num_classes * max_boxes_selected;
  std::vector<uint32_t> qnn_output_shape = {static_cast<uint32_t>(max_selected), 3u};

  const std::string& output_name = outputs[0].name;
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  // 5. Wire QNN out[1] (valid count) to a NATIVE UINT_32 sink tensor
  // QNN NMS out[1] is optional per the spec, but some backends (e.g. HTP) require
  // all op outputs to be connected to a tensor even when the caller doesn't read them.
  // Omitting it causes QNN_COMMON_ERROR_MEM_ALLOC (error 1002) at graphFinalize.
  const std::string valid_count_name = output_name + "_valid_count";
  QnnTensorWrapper valid_count_tensor(valid_count_name,
                                      QNN_TENSOR_TYPE_NATIVE,
                                      QNN_DATATYPE_UINT_32,
                                      QnnQuantParamsWrapper(),
                                      std::vector<uint32_t>{batch});
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(valid_count_tensor)),
                "NonMaxSuppression: failed to add valid count tensor.");

  // 6. Build output tensor(s) and QNN node
  // graph_output=true:  NMS -> NATIVE INT_32 -> Cast -> APP_READ INT_64
  // graph_output=false: NMS -> NATIVE INT_32 directly (no Cast)
  if (is_graph_output) {
    // Intermediate INT_32 NMS output tensor.
    const std::string nms_out_name = output_name + "_nms_int32";
    QnnTensorWrapper nms_out_tensor(nms_out_name,
                                    QNN_TENSOR_TYPE_NATIVE,
                                    QNN_DATATYPE_INT_32,
                                    QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(qnn_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(nms_out_tensor)),
                  "NonMaxSuppression: failed to add NMS INT_32 output tensor.");

    // Create the NMS QNN node with both outputs.
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_NON_MAX_SUPPRESSION,
                                                  std::move(input_names),
                                                  {nms_out_name, valid_count_name},
                                                  std::move(param_tensor_names),
                                                  do_op_validation),
                  "NonMaxSuppression: failed to create QNN node.");

    // Cast INT_32 → INT_64 graph output tensor.
    QnnTensorWrapper graph_out_tensor(output_name,
                                      QNN_TENSOR_TYPE_APP_READ,
                                      QNN_DATATYPE_INT_64,
                                      QnnQuantParamsWrapper(),
                                      std::vector<uint32_t>(qnn_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(graph_out_tensor)),
                  "NonMaxSuppression: failed to add INT_64 graph output tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(output_name + "_cast_node",
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_CAST,
                                                  {nms_out_name},
                                                  {output_name},
                                                  {},
                                                  do_op_validation),
                  "NonMaxSuppression: failed to create INT_32->INT_64 Cast node.");
  } else {
    // Non-graph-output: use output_name directly as NATIVE INT_32 NMS output.
    QnnTensorWrapper native_out_tensor(output_name,
                                       QNN_TENSOR_TYPE_NATIVE,
                                       QNN_DATATYPE_INT_32,
                                       QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>(qnn_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(native_out_tensor)),
                  "NonMaxSuppression: failed to add NATIVE INT_32 output tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_NON_MAX_SUPPRESSION,
                                                  std::move(input_names),
                                                  {output_name, valid_count_name},
                                                  std::move(param_tensor_names),
                                                  do_op_validation),
                  "NonMaxSuppression: failed to create QNN node.");
  }

  return Ort::Status();
}

void CreateNonMaxSuppressionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<NonMaxSuppressionOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
