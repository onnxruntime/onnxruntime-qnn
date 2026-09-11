// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "QnnOpDef.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/qdq_constant_folding.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Upper bound on the number of pooled bins (num_rois * pooled_h * pooled_w). The decomposition
// emits O(bins) QNN nodes, so reject very large configurations and let them fall back to the
// ORT CPU EP rather than exploding the QNN graph.
constexpr int64_t kMaxMaxRoiPoolBins = 4096;

// Reads the constant rois [num_rois, 5] = [batch_index, x1, y1, x2, y2] and returns the
// floating-point ROI corner coordinates (still in input-image space, before spatial_scale).
// Handles both a plain fp32 initializer and a QDQ-folded 8-bit quantized constant.
Ort::Status ReadRoisAsFloat(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnitIODef& rois_def,
                            uint32_t num_rois,
                            /*out*/ std::vector<float>& rois_flat /* num_rois*5 */) {
  TensorInfo rois_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(rois_def, rois_info));

  std::vector<uint8_t> rois_bytes;
  RETURN_IF_ERROR(GetEffectivelyConstantTensorBytes(qnn_model_wrapper, rois_def.name, rois_bytes));

  const size_t num_elems = static_cast<size_t>(num_rois) * 5;
  rois_flat.resize(num_elems);

  if (rois_info.qnn_data_type == QNN_DATATYPE_FLOAT_32) {
    RETURN_IF_NOT(rois_bytes.size() == num_elems * sizeof(float), "MaxRoiPool rois initializer size mismatch.");
    const float* rois = reinterpret_cast<const float*>(rois_bytes.data());
    std::copy(rois, rois + num_elems, rois_flat.begin());
  } else if (rois_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_8 ||
             rois_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_8 ||
             rois_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16 ||
             rois_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16) {
    RETURN_IF_NOT(rois_info.quant_param.IsPerTensor(/*include_bw*/ true),
                  "MaxRoiPool requires per-tensor quantized rois.");
    float scale = 0.0f;
    int32_t offset = 0;
    RETURN_IF_ERROR(rois_info.quant_param.GetPerTensorScaleOffset(scale, offset));
    const bool is_16bit = (rois_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16 ||
                           rois_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16);
    const bool is_signed = (rois_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_8 ||
                            rois_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16);
    RETURN_IF_NOT(rois_bytes.size() == num_elems * (is_16bit ? 2u : 1u),
                  "MaxRoiPool rois initializer size mismatch.");
    for (size_t i = 0; i < num_elems; ++i) {
      double q;
      if (is_16bit) {
        const uint16_t raw = reinterpret_cast<const uint16_t*>(rois_bytes.data())[i];
        q = is_signed ? static_cast<double>(static_cast<int16_t>(raw)) : static_cast<double>(raw);
      } else {
        q = is_signed ? static_cast<double>(static_cast<int8_t>(rois_bytes[i]))
                      : static_cast<double>(rois_bytes[i]);
      }
      rois_flat[i] = static_cast<float>(utils::Dequantize(offset, scale, q));
    }
  } else {
    return MAKE_EP_FAIL("MaxRoiPool only supports float32 or 8/16-bit quantized rois.");
  }
  return Ort::Status();
}

}  // namespace

// Translates ONNX MaxRoiPool by decomposing it into QNN primitives that are supported on all
// backends (CPU, HTP, GPU); QNN's native RoiPooling op only exists in the CPU/DSP op packages.
//
// ONNX MaxRoiPool(X[N,C,H,W], rois[num_rois,5]) pools each ROI into a pooled_h x pooled_w grid
// using adaptive (possibly overlapping, non-uniform) bins:
//   hstart = y1 + floor(i*roi_h/ph), hend = y1 + ceil((i+1)*roi_h/ph)   (and similarly for w)
// where corners are roundf(coord * spatial_scale) and roi_h = y2-y1+1, roi_w = x2-x1+1.
//
// The feature map arrives in NHWC layout. Each bin is realized exactly as
//   StridedSlice X[:, hstart:hend, wstart:wend, :] -> ReduceMax over {H,W} keepdims -> [1,1,1,C]
// (empty bins emit a static zero tensor, matching ONNX). Per ROI the ph*pw bin results are
// concatenated and reshaped to [1, ph, pw, C]; the per-ROI tensors are concatenated along the
// batch axis to form the [num_rois, ph, pw, C] NHWC output.
//
// The rois must be a constant initializer so the bin geometry can be computed at build time.
class MaxRoiPoolOpBuilder : public BaseOpBuilder {
 public:
  MaxRoiPoolOpBuilder() : BaseOpBuilder("MaxRoiPoolOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MaxRoiPoolOpBuilder);

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

Ort::Status MaxRoiPoolOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
                                               const Ort::Logger& logger) const {
  // MaxRoiPool is sensitive to data layout and requires NHWC. Continue once converted.
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  OrtNodeAttrHelper node_helper(node_unit);

  // ROIs (input[1]) must be a constant so the bin geometry can be computed at build time.
  // A non-constant rois (graph input) is rejected here, before the layout transform, for a clean
  // CPU-EP fallback. A QDQ'd constant rois is not yet folded at GetCapability time, so the full
  // constant check is deferred to ProcessInputs.
  RETURN_IF(qnn_model_wrapper.IsGraphInput(node_unit.Inputs()[1].name),
            "MaxRoiPool requires rois to be a constant initializer.");

  TensorInfo rois_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], rois_info));
  RETURN_IF_NOT(2 == static_cast<int>(rois_info.shape.size()) && 5 == static_cast<int>(rois_info.shape[1]),
                "MaxRoiPool requires rois of shape [num_rois, 5].");
  const int64_t num_rois = static_cast<int64_t>(rois_info.shape[0]);

  // Output tensor size [num_rois, C, pooled_h, pooled_w].
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  RETURN_IF_NOT(4 == static_cast<int>(output_info.shape.size()),
                "MaxRoiPool requires 4d output (num_rois, C, pooled_h, pooled_w).");

  // pooled_shape is required and must match the output spatial dims.
  std::vector<int64_t> pooled_shape = node_helper.Get("pooled_shape", std::vector<int64_t>{});
  RETURN_IF_NOT(pooled_shape.size() == 2, "MaxRoiPool requires the pooled_shape attribute with 2 values.");
  RETURN_IF_NOT(pooled_shape[0] == static_cast<int64_t>(output_info.shape[2]),
                "Expect pooled_shape[0] == output_tensor.shape[2]");
  RETURN_IF_NOT(pooled_shape[1] == static_cast<int64_t>(output_info.shape[3]),
                "Expect pooled_shape[1] == output_tensor.shape[3]");

  float spatial_scale = node_helper.Get("spatial_scale", 1.0f);
  RETURN_IF(spatial_scale <= 0, "MaxRoiPool got invalid spatial_scale <= 0");

  // The decomposition emits O(num_rois * ph * pw) QNN nodes; bound the graph size.
  const int64_t num_bins = num_rois * pooled_shape[0] * pooled_shape[1];
  RETURN_IF_NOT(num_bins <= kMaxMaxRoiPoolBins,
                "MaxRoiPool decomposition exceeds the supported bin count (num_rois * pooled_h * pooled_w).");

  return Ort::Status();
}

Ort::Status MaxRoiPoolOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
                                               const Ort::Logger& logger,
                                               std::vector<std::string>& input_names,
                                               bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();

  // input[0]: feature map (X). The layout transformer converts it to NHWC. Only X is wired into
  // the QNN graph; the rois are consumed at build time to compute bin geometry (handled in
  // ProcessAttributesAndOutputs), so input[1] is intentionally not processed here.
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  RETURN_IF_NOT(qnn_model_wrapper.IsEffectivelyConstantInput(inputs[1].name),
                "MaxRoiPool requires rois to be a constant initializer.");

  return Ort::Status();
}

Ort::Status MaxRoiPoolOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                             const OrtNodeUnit& node_unit,
                                                             std::vector<std::string>&& input_names,
                                                             const Ort::Logger& logger,
                                                             bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);

  const std::string& x_name = input_names[0];
  const auto& inputs = node_unit.Inputs();

  // X tensor info (NHWC: [N, H, W, C]).
  TensorInfo x_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], x_info));
  RETURN_IF_NOT(x_info.shape.size() == 4, "MaxRoiPool expects a 4D feature map.");
  const uint32_t in_h = x_info.shape[1];
  const uint32_t in_w = x_info.shape[2];
  const uint32_t channels = x_info.shape[3];

  // Output tensor info (NHWC: [num_rois, ph, pw, C]).
  TensorInfo out_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], out_info));
  const std::string& output_name = node_unit.Outputs()[0].name;
  const uint32_t num_rois = out_info.shape[0];
  const uint32_t pooled_h = out_info.shape[1];
  const uint32_t pooled_w = out_info.shape[2];

  const Qnn_DataType_t dtype = out_info.qnn_data_type;
  const float spatial_scale = node_helper.Get("spatial_scale", 1.0f);

  // Read the constant rois corners (input-image space).
  std::vector<float> rois_flat;
  RETURN_IF_ERROR(ReadRoisAsFloat(qnn_model_wrapper, inputs[1], num_rois, rois_flat));

  const std::string name_base = (node_unit.Name().empty() ? node_unit.OpType() : node_unit.Name()) +
                                std::to_string(node_unit.Index());
  auto local_name = [&name_base](std::string_view suffix) { return name_base + std::string(suffix); };

  // Reusable zero tensor for empty bins (ONNX fills them with 0.0). Created lazily.
  std::string zero_bin_name;
  auto ensure_zero_bin = [&qnn_model_wrapper, &out_info, dtype, channels,
                          &zero_bin_name, &local_name]() -> Ort::Status {
    if (!zero_bin_name.empty()) {
      return Ort::Status();
    }
    zero_bin_name = local_name("_zero_bin");
    const size_t num_bytes = qnn::utils::GetQnnTensorDataSizeInBytes(static_cast<size_t>(channels), dtype);
    std::vector<uint8_t> zero_bytes(num_bytes, 0);

    // For a quantized output, the bytes must encode 0.0 under its scale/offset, not all-zero.
    if (out_info.quant_param.IsQuantized()) {
      RETURN_IF_NOT(out_info.quant_param.IsPerTensor(/*include_bw*/ true),
                    "MaxRoiPool requires a per-tensor quantized output.");
      float scale = 0.0f;
      int32_t offset = 0;
      RETURN_IF_ERROR(out_info.quant_param.GetPerTensorScaleOffset(scale, offset));
      int quant_value = 0;
      RETURN_IF_ERROR(utils::Quantize(0.0, scale, offset, dtype, quant_value));
      switch (dtype) {
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8: {
          std::fill_n(zero_bytes.data(), channels, static_cast<uint8_t>(quant_value));
          break;
        }
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16: {
          std::fill_n(reinterpret_cast<uint16_t*>(zero_bytes.data()), channels,
                      static_cast<uint16_t>(quant_value));
          break;
        }
        default:
          return MAKE_EP_FAIL("MaxRoiPool: unsupported quantized output element type for zero-bin.");
      }
    }
    QnnTensorWrapper zero_tensor(zero_bin_name, QNN_TENSOR_TYPE_STATIC, dtype,
                                 out_info.quant_param.Copy(), std::vector<uint32_t>{1u, 1u, 1u, channels},
                                 std::move(zero_bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_tensor)),
                  "Failed to add MaxRoiPool zero-bin tensor.");
    return Ort::Status();
  };

  // Emit a StridedSlice + ReduceMax for one bin region and return its [1,1,1,C] output name.
  // batch_idx selects the image in the (NHWC) feature map; tag is a unique label for naming.
  auto emit_bin = [&qnn_model_wrapper, &node_unit, &x_info, &out_info, dtype, channels,
                   &x_name, do_op_validation, &local_name, &ensure_zero_bin, &zero_bin_name](
                      uint32_t batch_idx, const std::string& tag,
                      uint32_t hstart, uint32_t hend, uint32_t wstart, uint32_t wend,
                      /*out*/ std::string& bin_out_name) -> Ort::Status {
    if (hend <= hstart || wend <= wstart) {
      RETURN_IF_ERROR(ensure_zero_bin());
      bin_out_name = zero_bin_name;
      return Ort::Status();
    }

    const std::string suffix = tag + "_h" + std::to_string(hstart) + "_w" + std::to_string(wstart);

    // StridedSlice is a byte-copy, so it stays in X's quant domain; the ReduceMax below requantizes
    // to the output domain.
    const std::string slice_out = local_name("_slice" + suffix);
    const uint32_t bh = hend - hstart;
    const uint32_t bw = wend - wstart;
    std::vector<uint32_t> slice_shape{1u, bh, bw, channels};
    QnnTensorWrapper slice_tensor(slice_out, QNN_TENSOR_TYPE_NATIVE, x_info.qnn_data_type,
                                  x_info.quant_param.Copy(), std::vector<uint32_t>(slice_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(slice_tensor)),
                  "Failed to add MaxRoiPool slice tensor.");

    std::vector<uint32_t> ranges_dims{4u, 3u};
    std::vector<uint32_t> ranges_data{
        batch_idx, batch_idx + 1u, 1u,
        hstart, hend, 1u,
        wstart, wend, 1u,
        0u, channels, 1u};
    QnnParamWrapper ranges_param(node_unit.Index(), slice_out, QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                 std::move(ranges_dims), std::move(ranges_data), /*is_signed*/ true);
    std::vector<std::string> slice_params{ranges_param.GetParamTensorName()};
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(ranges_param)),
                  "Failed to add MaxRoiPool StridedSlice ranges param.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(local_name("_slice_node" + suffix),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE,
                                                  {x_name}, {slice_out}, std::move(slice_params),
                                                  do_op_validation),
                  "Failed to add MaxRoiPool StridedSlice node.");

    // ReduceMax over H,W (axes 1,2) keepdims -> [1,1,1,C].
    bin_out_name = local_name("_rmax" + suffix);
    QnnTensorWrapper rmax_tensor(bin_out_name, QNN_TENSOR_TYPE_NATIVE, dtype,
                                 out_info.quant_param.Copy(), std::vector<uint32_t>{1u, 1u, 1u, channels});
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(rmax_tensor)),
                  "Failed to add MaxRoiPool ReduceMax tensor.");
    std::vector<uint32_t> axes_data{1u, 2u};
    QnnParamWrapper axes_param = createQnnParamWrapper<uint32_t>(
        node_unit.Index(), bin_out_name, QNN_OP_REDUCE_MAX_PARAM_AXES,
        std::vector<uint32_t>{2u}, std::move(axes_data));
    std::vector<std::string> rmax_params{axes_param.GetParamTensorName()};
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(axes_param)),
                  "Failed to add MaxRoiPool ReduceMax axes param.");
    RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), bin_out_name, true,
                                       QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS, rmax_params));
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(local_name("_rmax_node" + suffix),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_REDUCE_MAX,
                                                  {slice_out}, {bin_out_name}, std::move(rmax_params),
                                                  do_op_validation),
                  "Failed to add MaxRoiPool ReduceMax node.");
    return Ort::Status();
  };

  // Concatenate a list of tensors along an axis, producing a new NATIVE output. A single-element
  // list is passed through with a Reshape (QNN Concat requires >= 2 inputs).
  auto emit_concat = [&qnn_model_wrapper, &node_unit, &out_info, dtype, do_op_validation, &local_name](
                         const std::vector<std::string>& parts, uint32_t axis,
                         const std::vector<uint32_t>& part_shape, const std::vector<uint32_t>& out_shape,
                         const std::string& name_suffix, /*out*/ std::string& concat_out) -> Ort::Status {
    concat_out = local_name(name_suffix);
    if (parts.size() == 1) {
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
          parts[0], concat_out, std::vector<uint32_t>(part_shape), std::vector<uint32_t>(out_shape),
          dtype, out_info.quant_param.Copy(), out_info.quant_param.Copy(),
          do_op_validation, /*is_for_input*/ false, /*is_for_output*/ false));
      return Ort::Status();
    }
    QnnTensorWrapper concat_tensor(concat_out, QNN_TENSOR_TYPE_NATIVE, dtype,
                                   out_info.quant_param.Copy(), std::vector<uint32_t>(out_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(concat_tensor)),
                  "Failed to add MaxRoiPool concat tensor.");
    std::vector<std::string> concat_params;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), concat_out, axis,
                                           QNN_OP_CONCAT_PARAM_AXIS, concat_params));
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(local_name(name_suffix + "_node"),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                                  std::vector<std::string>(parts), {concat_out},
                                                  std::move(concat_params), do_op_validation),
                  "Failed to add MaxRoiPool Concat node.");
    return Ort::Status();
  };

  // Build each ROI's [1, ph, pw, C] tile, then concat all ROIs along the batch axis.
  std::vector<std::string> roi_tile_names;
  roi_tile_names.reserve(num_rois);

  for (uint32_t r = 0; r < num_rois; ++r) {
    const float* roi = &rois_flat[static_cast<size_t>(r) * 5];
    // roi = [batch_index, x1, y1, x2, y2]. batch_index selects the image in the feature map.
    const int32_t batch_index = static_cast<int32_t>(std::lround(roi[0]));
    RETURN_IF(batch_index < 0 || static_cast<uint32_t>(batch_index) >= x_info.shape[0],
              "MaxRoiPool rois batch_index is out of range.");
    const uint32_t batch_idx = static_cast<uint32_t>(batch_index);
    const int32_t x1 = static_cast<int32_t>(std::lround(roi[1] * spatial_scale));
    const int32_t y1 = static_cast<int32_t>(std::lround(roi[2] * spatial_scale));
    const int32_t x2 = static_cast<int32_t>(std::lround(roi[3] * spatial_scale));
    const int32_t y2 = static_cast<int32_t>(std::lround(roi[4] * spatial_scale));
    const int32_t roi_h = std::max(y2 - y1 + 1, 1);
    const int32_t roi_w = std::max(x2 - x1 + 1, 1);

    const std::string roi_tag = "_r" + std::to_string(r);
    std::vector<std::string> bin_names;
    bin_names.reserve(static_cast<size_t>(pooled_h) * pooled_w);

    for (uint32_t ph = 0; ph < pooled_h; ++ph) {
      // Adaptive bin bounds along H (ONNX uses floor for start, ceil for end).
      int32_t hstart = y1 + static_cast<int32_t>(std::floor(static_cast<float>(ph) * roi_h / pooled_h));
      int32_t hend = y1 + static_cast<int32_t>(std::ceil(static_cast<float>(ph + 1) * roi_h / pooled_h));
      hstart = std::min(std::max(hstart, 0), static_cast<int32_t>(in_h));
      hend = std::min(std::max(hend, 0), static_cast<int32_t>(in_h));

      for (uint32_t pw = 0; pw < pooled_w; ++pw) {
        int32_t wstart = x1 + static_cast<int32_t>(std::floor(static_cast<float>(pw) * roi_w / pooled_w));
        int32_t wend = x1 + static_cast<int32_t>(std::ceil(static_cast<float>(pw + 1) * roi_w / pooled_w));
        wstart = std::min(std::max(wstart, 0), static_cast<int32_t>(in_w));
        wend = std::min(std::max(wend, 0), static_cast<int32_t>(in_w));

        std::string bin_out;
        RETURN_IF_ERROR(emit_bin(batch_idx, roi_tag + "_p" + std::to_string(ph) + std::to_string(pw),
                                 static_cast<uint32_t>(hstart), static_cast<uint32_t>(hend),
                                 static_cast<uint32_t>(wstart), static_cast<uint32_t>(wend), bin_out));
        bin_names.push_back(std::move(bin_out));
      }
    }

    // Concat the ph*pw bin tensors along axis 1 -> [1, ph*pw, 1, C], then reshape -> [1, ph, pw, C].
    std::string roi_concat;
    RETURN_IF_ERROR(emit_concat(bin_names, /*axis*/ 1u,
                                std::vector<uint32_t>{1u, 1u, 1u, channels},
                                std::vector<uint32_t>{1u, pooled_h * pooled_w, 1u, channels},
                                "_roi_concat_r" + std::to_string(r), roi_concat));

    std::string roi_tile = local_name("_roi_tile_r" + std::to_string(r));
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        roi_concat, roi_tile,
        std::vector<uint32_t>{1u, pooled_h * pooled_w, 1u, channels},
        std::vector<uint32_t>{1u, pooled_h, pooled_w, channels},
        dtype, out_info.quant_param.Copy(), out_info.quant_param.Copy(),
        do_op_validation, /*is_for_input*/ false, /*is_for_output*/ false));
    roi_tile_names.push_back(std::move(roi_tile));
  }

  // Final output: concat per-ROI tiles along the batch axis -> [num_rois, ph, pw, C].
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  const Qnn_TensorType_t out_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  if (roi_tile_names.size() == 1) {
    // Single ROI: the tile is already [1, ph, pw, C]; reshape it into the graph output tensor
    // (QNN Concat requires >= 2 inputs).
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        roi_tile_names[0], output_name,
        std::vector<uint32_t>{1u, pooled_h, pooled_w, channels},
        std::vector<uint32_t>{num_rois, pooled_h, pooled_w, channels},
        dtype, out_info.quant_param.Copy(), out_info.quant_param.Copy(),
        do_op_validation, /*is_for_input*/ false, /*is_for_output*/ is_graph_output));
    return Ort::Status();
  }

  QnnTensorWrapper out_tensor(output_name, out_type, dtype, out_info.quant_param.Copy(),
                              std::vector<uint32_t>{num_rois, pooled_h, pooled_w, channels});
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_tensor)),
                "Failed to add MaxRoiPool output tensor.");
  std::vector<std::string> out_concat_params;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), output_name, 0u,
                                         QNN_OP_CONCAT_PARAM_AXIS, out_concat_params));
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(local_name("_out_concat"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                                std::move(roi_tile_names), {output_name},
                                                std::move(out_concat_params), do_op_validation),
                "Failed to add MaxRoiPool output Concat node.");

  return Ort::Status();
}

void CreateMaxRoiPoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MaxRoiPoolOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
