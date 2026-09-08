// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <gsl/gsl>

#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Input indices for QLinearConv (opset 10).
//   0: x              4: w_scale
//   1: x_scale        5: w_zero_point
//   2: x_zero_point   6: y_scale
//   3: w              7: y_zero_point
//                     8: B (optional, int32)
static constexpr size_t kIdxX = 0;
static constexpr size_t kIdxXScale = 1;
static constexpr size_t kIdxXZeroPoint = 2;
static constexpr size_t kIdxW = 3;
static constexpr size_t kIdxWScale = 4;
static constexpr size_t kIdxWZeroPoint = 5;
static constexpr size_t kIdxYScale = 6;
static constexpr size_t kIdxYZeroPoint = 7;
static constexpr size_t kIdxBias = 8;

/**
 * Translates ONNX QLinearConv (opset 10) into a QNN Conv2d / DepthWiseConv2d / Conv3d node.
 *
 * QLinearConv carries quantization parameters as explicit op inputs (x/w/y scale and zero_point)
 * rather than as QDQ node metadata, and its activation is a bare quantized integer tensor.
 *
 * QNN's Conv ops require channel-last layout (NHWC for 2D, NDHWC for 3D), while ONNX Conv is
 * channel-first (NCHW). Because the activation carries no quant metadata of its own, we cannot let
 * ORT's layout transformer insert a Transpose on it (that Transpose would be an unquantized
 * uint8/int8 node, which HTP rejects). Instead ShouldConvertDataLayoutForOp suppresses the layout
 * transform for QLinearConv, and this builder performs the layout conversion internally:
 *
 *   x (NCHW, quant_x) --Transpose--> x_nhwc (quant_x)  --\
 *   w (OIHW)          --transpose bytes--> w_hwio (quant_w) --Conv2d--> y_nhwc (quant_y)
 *   B (int32, quant_bias) ----------------------------/        |
 *                                                              Transpose
 *                                                              v
 *                                                       y (NCHW, quant_y)
 *
 * Every QNN tensor gets its quant params attached, so HTP sees UFIXED_POINT_8 / SFIXED_POINT_8
 * throughout (the same strategy DQConvIntegerFusion uses for ConvInteger).
 */
class QLinearConvOpBuilder : public BaseOpBuilder {
 public:
  QLinearConvOpBuilder() : BaseOpBuilder("QLinearConvOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QLinearConvOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  // The whole QNN subgraph (input Transpose -> Conv -> output Transpose) is emitted in ProcessInputs
  // because the internal Transpose nodes tie input and output handling together. The base
  // AddToModelBuilder is final, so we cannot override it; instead ProcessAttributesAndOutputs is a
  // no-op and all work happens in ProcessInputs (which receives do_op_validation).
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
                               bool do_op_validation) const ORT_MUST_USE_RESULT;

  // Reads a scalar float32 scale initializer (QLinearConv x/y scales are always float32 per spec).
  static Ort::Status ReadScalarScale(const QnnModelWrapper& qnn_model_wrapper,
                                     const OrtNodeUnitIODef& scale_input,
                                     float& out_scale);

  // Builds a per-tensor QnnQuantParamsWrapper for x or y (scalar scale + scalar zero_point).
  static Ort::Status BuildPerTensorQuantParam(const QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnitIODef& scale_input,
                                              const OrtNodeUnitIODef& zp_input,
                                              QnnQuantParamsWrapper& out_quant_param);

  // Builds the weight QnnQuantParamsWrapper. w_scale/w_zp may be scalar (per-tensor) or 1-D of
  // size M (per-output-channel, axis 0 in OIHW). num_output_channels is the weight's dim 0 (M).
  static Ort::Status BuildWeightQuantParam(const QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnitIODef& scale_input,
                                           const OrtNodeUnitIODef& zp_input,
                                           uint32_t num_output_channels,
                                           QnnQuantParamsWrapper& out_quant_param);

  // Builds the int32 bias QnnQuantParamsWrapper: scale = x_scale * w_scale, offset = 0.
  // Per-channel when the weight is per-channel (one bias scale per output channel).
  static Ort::Status BuildBiasQuantParam(const QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         QnnQuantParamsWrapper& out_quant_param);
};

// ---------------------------------------------------------------------------
// Quant-param helpers
// ---------------------------------------------------------------------------

Ort::Status QLinearConvOpBuilder::ReadScalarScale(const QnnModelWrapper& qnn_model_wrapper,
                                                  const OrtNodeUnitIODef& scale_input,
                                                  float& out_scale) {
  RETURN_IF(!scale_input.Exists(), "QLinearConv: scale input does not exist.");
  RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(scale_input.name),
            "QLinearConv: scale must be a compile-time constant (initializer).");
  const OrtValueInfo* scale_tensor = qnn_model_wrapper.GetConstantTensor(scale_input.name);
  RETURN_IF(scale_tensor == nullptr, "QLinearConv: could not retrieve scale initializer.");

  std::vector<float> scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(scale_tensor, scales));
  RETURN_IF(scales.size() != 1, "QLinearConv: x_scale/y_scale must be scalar (per-tensor).");
  out_scale = scales[0];
  return Ort::Status();
}

Ort::Status QLinearConvOpBuilder::BuildPerTensorQuantParam(const QnnModelWrapper& qnn_model_wrapper,
                                                           const OrtNodeUnitIODef& scale_input,
                                                           const OrtNodeUnitIODef& zp_input,
                                                           QnnQuantParamsWrapper& out_quant_param) {
  float scale = 0.0f;
  RETURN_IF_ERROR(ReadScalarScale(qnn_model_wrapper, scale_input, scale));

  int32_t offset = 0;
  if (zp_input.Exists() && !zp_input.name.empty()) {
    RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(zp_input.name),
              "QLinearConv: zero_point must be a compile-time constant (initializer).");
    const OrtValueInfo* zp_tensor = qnn_model_wrapper.GetConstantTensor(zp_input.name);
    RETURN_IF(zp_tensor == nullptr, "QLinearConv: could not retrieve zero_point initializer.");
    std::vector<int32_t> zero_points;
    ONNXTensorElementDataType zp_onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(zp_tensor, zero_points, zp_onnx_type));
    RETURN_IF(zero_points.size() != 1, "QLinearConv: x_zero_point/y_zero_point must be scalar.");
    // UnpackZeroPoints already returns the negated QNN offset; pass through directly.
    offset = zero_points[0];
  }

  out_quant_param = QnnQuantParamsWrapper::PerTensor(scale, offset);
  return Ort::Status();
}

Ort::Status QLinearConvOpBuilder::BuildWeightQuantParam(const QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnitIODef& scale_input,
                                                        const OrtNodeUnitIODef& zp_input,
                                                        uint32_t num_output_channels,
                                                        QnnQuantParamsWrapper& out_quant_param) {
  RETURN_IF(!scale_input.Exists(), "QLinearConv: w_scale input does not exist.");
  RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(scale_input.name),
            "QLinearConv: w_scale must be a compile-time constant (initializer).");
  const OrtValueInfo* scale_tensor = qnn_model_wrapper.GetConstantTensor(scale_input.name);
  RETURN_IF(scale_tensor == nullptr, "QLinearConv: could not retrieve w_scale initializer.");

  std::vector<float> scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(scale_tensor, scales));

  // Read zero-points (already negated by UnpackZeroPoints). Default to all-zero if absent.
  std::vector<int32_t> offsets;
  if (zp_input.Exists() && !zp_input.name.empty()) {
    RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(zp_input.name),
              "QLinearConv: w_zero_point must be a compile-time constant (initializer).");
    const OrtValueInfo* zp_tensor = qnn_model_wrapper.GetConstantTensor(zp_input.name);
    RETURN_IF(zp_tensor == nullptr, "QLinearConv: could not retrieve w_zero_point initializer.");
    ONNXTensorElementDataType zp_onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(zp_tensor, offsets, zp_onnx_type));
  }

  if (scales.size() == 1) {
    // Per-tensor weight quantization.
    const int32_t offset = offsets.empty() ? 0 : offsets[0];
    out_quant_param = QnnQuantParamsWrapper::PerTensor(scales[0], offset);
  } else {
    // Per-channel weight quantization on output-channel axis (axis 0 in OIHW).
    RETURN_IF(scales.size() != static_cast<size_t>(num_output_channels),
              "QLinearConv: per-channel w_scale size must equal the number of output channels (M).");
    if (offsets.empty()) {
      offsets.assign(scales.size(), 0);
    }
    RETURN_IF(offsets.size() != scales.size(),
              "QLinearConv: w_zero_point size must match w_scale size for per-channel quantization.");
    out_quant_param = QnnQuantParamsWrapper::PerChannel(gsl::span<const float>(scales),
                                                        gsl::span<const int32_t>(offsets),
                                                        /*axis=*/0);
  }
  return Ort::Status();
}

Ort::Status QLinearConvOpBuilder::BuildBiasQuantParam(const QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      QnnQuantParamsWrapper& out_quant_param) {
  const auto& inputs = node_unit.Inputs();

  float x_scale = 0.0f;
  RETURN_IF_ERROR(ReadScalarScale(qnn_model_wrapper, inputs[kIdxXScale], x_scale));

  const OrtValueInfo* w_scale_tensor = qnn_model_wrapper.GetConstantTensor(inputs[kIdxWScale].name);
  RETURN_IF(w_scale_tensor == nullptr, "QLinearConv: could not retrieve w_scale initializer for bias.");
  std::vector<float> w_scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(w_scale_tensor, w_scales));

  // Bias quant: scale = x_scale * w_scale, zero_point = 0 (per ONNX QLinearConv spec).
  if (w_scales.size() == 1) {
    out_quant_param = QnnQuantParamsWrapper::PerTensor(x_scale * w_scales[0], 0);
  } else {
    std::vector<float> bias_scales(w_scales.size());
    for (size_t i = 0; i < w_scales.size(); ++i) {
      bias_scales[i] = x_scale * w_scales[i];
    }
    std::vector<int32_t> bias_offsets(w_scales.size(), 0);
    out_quant_param = QnnQuantParamsWrapper::PerChannel(gsl::span<const float>(bias_scales),
                                                        gsl::span<const int32_t>(bias_offsets),
                                                        /*axis=*/0);
  }
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// IsOpSupported
// ---------------------------------------------------------------------------

Ort::Status QLinearConvOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const Ort::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();
  RETURN_IF(inputs.size() < 8, "QLinearConv must have at least 8 inputs.");

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[kIdxX].shape, input_shape), "Cannot get shape");
  RETURN_IF(input_shape.size() != 3 && input_shape.size() != 4 && input_shape.size() != 5,
            "QNN QLinearConv only supports 1D (rank 3), 2D (rank 4), or 3D (rank 5) inputs.");

  ONNXTensorElementDataType input_data_type = inputs[kIdxX].type;
  std::string error_msg = "QNN EP: Data type " + std::to_string(static_cast<int>(input_data_type)) +
                          " is not supported for QLinearConv operator in CPU backend.";
  RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, input_data_type, error_msg));

  OrtNodeAttrHelper node_helper(node_unit);
  auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER" && auto_pad != "VALID",
            ("QNN QLinearConv does not support 'auto_pad' value: " + auto_pad).c_str());

  // All scale/zp inputs must be compile-time constants.
  const std::array<size_t, 6> const_indices = {kIdxXScale, kIdxXZeroPoint, kIdxWScale,
                                               kIdxWZeroPoint, kIdxYScale, kIdxYZeroPoint};
  for (size_t idx : const_indices) {
    RETURN_IF(idx >= inputs.size() || !inputs[idx].Exists(),
              "QLinearConv: required scale/zero_point input is missing.");
    RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(inputs[idx].name),
              "QLinearConv: scale/zero_point inputs must be compile-time constants.");
  }

  // x/y scale and zero_point must be scalar (per-tensor).
  for (size_t idx : {kIdxXScale, kIdxXZeroPoint, kIdxYScale, kIdxYZeroPoint}) {
    if (inputs[idx].shape.has_value()) {
      const auto& shape = inputs[idx].shape.value();
      const int64_t num_elems = std::accumulate(shape.begin(), shape.end(),
                                                static_cast<int64_t>(1), std::multiplies<int64_t>());
      RETURN_IF(num_elems != 1, "QLinearConv: x/y scale and zero_point must be per-tensor (scalar).");
    }
  }

  // Per-channel weight requires a constant (initializer) weight.
  std::vector<uint32_t> weight_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[kIdxW].shape, weight_shape), "Cannot get weight shape");
  if (inputs[kIdxWScale].shape.has_value()) {
    const auto& w_scale_shape = inputs[kIdxWScale].shape.value();
    const int64_t w_scale_elems = std::accumulate(w_scale_shape.begin(), w_scale_shape.end(),
                                                  static_cast<int64_t>(1), std::multiplies<int64_t>());
    if (w_scale_elems > 1) {
      RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(inputs[kIdxW].name),
                "QLinearConv: per-channel weight quantization requires a constant weight initializer.");
      RETURN_IF(w_scale_elems != static_cast<int64_t>(weight_shape[0]),
                "QLinearConv: per-channel w_scale size must equal the number of output channels (M).");
    }
  }

  // Validate the full QNN graph (internal transposes + Conv) via the QNN validation API.
  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, /*do_op_validation=*/true);
}

// ---------------------------------------------------------------------------
// ProcessInputs (emits the whole subgraph) / ProcessAttributesAndOutputs (no-op)
// ---------------------------------------------------------------------------

Ort::Status QLinearConvOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const Ort::Logger& logger,
                                                std::vector<std::string>& input_names,
                                                bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(input_names);
  return CreateOrValidate(qnn_model_wrapper, node_unit, do_op_validation);
}

Ort::Status QLinearConvOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                              const OrtNodeUnit& node_unit,
                                                              std::vector<std::string>&& input_names,
                                                              const Ort::Logger& logger,
                                                              bool do_op_validation) const {
  // All graph construction happens in ProcessInputs; nothing to do here.
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(input_names);
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(do_op_validation);
  return Ort::Status();
}

Ort::Status QLinearConvOpBuilder::CreateOrValidate(QnnModelWrapper& qnn_model_wrapper,
                                                   const OrtNodeUnit& node_unit,
                                                   bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  const size_t num_inputs = inputs.size();

  std::vector<uint32_t> x_shape_nchw;  // ONNX activation shape (NCHW / NCDHW / NCW)
  std::vector<uint32_t> y_shape_nchw;  // ONNX output shape
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[kIdxX].shape, x_shape_nchw), "Cannot get x shape");
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].shape, y_shape_nchw), "Cannot get output shape");

  const size_t onnx_rank = x_shape_nchw.size();
  const bool is_1d = (onnx_rank == 3);
  const bool is_3d = (onnx_rank == 5);

  // For 1D conv we operate as 2D by inserting H=1: NCW -> NC1W, weight [M,C,W] -> [M,C,1,W].
  // conv_rank is the spatial+2 rank used for the QNN Conv (always 4 for 1D/2D, 5 for 3D).
  const size_t conv_rank = is_3d ? 5 : 4;

  const std::string node_base = utils::UniqueNameGenerator().New(node_unit);

  // -------------------------------------------------------------------------
  // Activation x: build quant params, reshape (1D), then Transpose NCHW->NHWC.
  // -------------------------------------------------------------------------
  QnnQuantParamsWrapper quant_x;
  RETURN_IF_ERROR(BuildPerTensorQuantParam(qnn_model_wrapper, inputs[kIdxXScale], inputs[kIdxXZeroPoint], quant_x));
  Qnn_DataType_t qnn_dtype_x = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxX].type, qnn_dtype_x));

  const std::string& x_name = inputs[kIdxX].name;

  // Channel-first shape used as the Transpose input (rank 4 or 5). For 1D, insert H=1.
  std::vector<uint32_t> x_cf_shape;  // NCHW or NCDHW
  if (is_1d) {
    x_cf_shape = {x_shape_nchw[0], x_shape_nchw[1], 1, x_shape_nchw[2]};
  } else {
    x_cf_shape = x_shape_nchw;
  }

  // Ensure the (possibly reshaped) NCHW activation tensor exists in the QNN graph.
  std::string x_cf_name = x_name;
  if (is_1d) {
    // Reshape NCW -> NC1W. For a graph input/dynamic tensor add a Reshape node; the source tensor
    // must exist first.
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(x_name)) {
      QnnTensorWrapper x_src;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[kIdxX], x_src));
      // MakeTensorWrapper does not attach our explicit quant params; rebuild with quant_x.
      Qnn_TensorType_t src_type = qnn_model_wrapper.GetTensorType(x_name);
      std::vector<uint8_t> src_bytes;
      if (qnn_model_wrapper.IsEffectivelyConstantInput(x_name)) {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(
            qnn_model_wrapper.GetConstantTensor(x_name), src_bytes));
      }
      QnnTensorWrapper x_q(x_name, src_type, qnn_dtype_x, quant_x.Copy(),
                           std::vector<uint32_t>(x_shape_nchw), std::move(src_bytes));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x_q)), "Failed to add x tensor.");
    }
    x_cf_name = node_base + "_x_reshaped";
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(x_name, x_cf_name, x_shape_nchw, x_cf_shape,
                                                     qnn_dtype_x, quant_x, do_op_validation,
                                                     qnn_model_wrapper.IsGraphInput(x_name)));
  } else {
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(x_name)) {
      Qnn_TensorType_t src_type = qnn_model_wrapper.GetTensorType(x_name);
      std::vector<uint8_t> src_bytes;
      if (qnn_model_wrapper.IsEffectivelyConstantInput(x_name)) {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(
            qnn_model_wrapper.GetConstantTensor(x_name), src_bytes));
      }
      QnnTensorWrapper x_q(x_name, src_type, qnn_dtype_x, quant_x.Copy(),
                           std::vector<uint32_t>(x_cf_shape), std::move(src_bytes));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x_q)), "Failed to add x tensor.");
    }
  }

  // Transpose NCHW->NHWC (or NCDHW->NDHWC). Quant params attached on both ends.
  const std::vector<uint32_t> cf_to_cl = utils::ChannelFirstToLastPerm(conv_rank);
  const std::vector<uint32_t> x_cl_shape = utils::ApplyPermToShape(x_cf_shape, cf_to_cl);
  const std::string x_cl_name = node_base + "_x_nhwc";
  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), x_cf_name, x_cl_name,
                                                     x_cf_shape, cf_to_cl, x_cl_shape,
                                                     qnn_dtype_x, quant_x,
                                                     do_op_validation,
                                                     /*is_for_input=*/false, /*is_for_output=*/false));

  // -------------------------------------------------------------------------
  // Weight w: OIHW -> HWIO (transpose bytes for initializer). Per-channel axis follows.
  // -------------------------------------------------------------------------
  TensorInfo w_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kIdxW], w_info));
  RETURN_IF(!w_info.is_initializer, "QLinearConv: weight must be a constant initializer.");
  const uint32_t num_output_channels = w_info.shape[0];

  QnnQuantParamsWrapper quant_w;
  RETURN_IF_ERROR(BuildWeightQuantParam(qnn_model_wrapper, inputs[kIdxWScale], inputs[kIdxWZeroPoint],
                                        num_output_channels, quant_w));
  Qnn_DataType_t qnn_dtype_w = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxW].type, qnn_dtype_w));

  // Weight working shape: [M,C,kH,kW] for 2D ([M,C,1,kW] for 1D), [M,C,kD,kH,kW] for 3D.
  std::vector<uint32_t> w_work_shape;
  if (is_1d) {
    w_work_shape = {w_info.shape[0], w_info.shape[1], 1, w_info.shape[2]};
  } else {
    w_work_shape = w_info.shape;
  }

  std::vector<uint32_t> hwcn_shape(w_work_shape.size());
  RETURN_IF_ERROR(utils::NchwShapeToHwcn<uint32_t>(w_work_shape, hwcn_shape));

  std::vector<uint8_t> w_bytes;
  if (is_1d) {
    // Transpose a [M,C,1,kW] view of the [M,C,kW] initializer to HWCN.
    std::vector<int64_t> w_work_i64(w_work_shape.begin(), w_work_shape.end());
    std::vector<uint8_t> orig_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(w_info.initializer_tensor, orig_bytes));
    w_bytes.resize(orig_bytes.size());

    const size_t elem_byte_size = utils::GetElementSizeByType(inputs[kIdxW].type);
    RETURN_IF(elem_byte_size == 0, "QLinearConv 1D: cannot get weight element byte size.");
    RETURN_IF_ERROR(utils::TransposeFromNchwToHwcn(std::move(w_work_i64), elem_byte_size,
                                                   orig_bytes, w_bytes, /*is_3d=*/false));
  } else {
    RETURN_IF_ERROR(utils::TransposeFromNchwToHwcn(qnn_model_wrapper, w_info.initializer_tensor,
                                                   w_bytes, is_3d));
  }

  // Move the per-channel quant axis through the OIHW->HWIO transpose.
  if (quant_w.IsPerChannel()) {
    const std::vector<size_t>& perm = is_3d ? nchw2hwcn_perm_3d : nchw2hwcn_perm;
    std::vector<size_t> perm_inv(perm.size());
    RETURN_IF_ERROR(utils::InvertPerm<size_t>(perm, perm_inv));
    RETURN_IF_ERROR(quant_w.HandleTranspose<size_t>(perm_inv));
  }

  const std::string w_hwcn_name = node_base + "_w_hwcn";
  {
    QnnTensorWrapper w_tensor(w_hwcn_name, QNN_TENSOR_TYPE_STATIC, qnn_dtype_w, quant_w.Copy(),
                              std::vector<uint32_t>(hwcn_shape), std::vector<uint8_t>(w_bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(w_tensor)), "Failed to add weight tensor.");
  }

  // -------------------------------------------------------------------------
  // Bias (optional, int32): scale = x_scale*w_scale, zp = 0. Shape [M], layout-invariant.
  // -------------------------------------------------------------------------
  std::string bias_name;
  const bool has_bias = num_inputs > kIdxBias && inputs[kIdxBias].Exists();
  if (has_bias) {
    bias_name = inputs[kIdxBias].name;
    TensorInfo bias_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kIdxBias], bias_info));
    RETURN_IF(!bias_info.is_initializer, "QLinearConv: bias must be a constant initializer.");

    QnnQuantParamsWrapper quant_bias;
    RETURN_IF_ERROR(BuildBiasQuantParam(qnn_model_wrapper, node_unit, quant_bias));

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(bias_name)) {
      std::vector<uint8_t> bias_bytes;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor, bias_bytes));
      // The bias is plain int32 in ONNX, but QNN requires SFIXED_POINT_32 (quantized int32).
      // bias_info.qnn_data_type is INT_32 (no quant on the ONNX node); force the quantized dtype.
      Qnn_DataType_t qnn_dtype_bias = QNN_DATATYPE_UNDEFINED;
      RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxBias].type, qnn_dtype_bias));
      QnnTensorWrapper bias_tensor(bias_name, QNN_TENSOR_TYPE_STATIC, qnn_dtype_bias,
                                   std::move(quant_bias), std::vector<uint32_t>(bias_info.shape),
                                   std::move(bias_bytes));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias tensor.");
    }
  }

  // -------------------------------------------------------------------------
  // Conv attributes: kernel_shape, dilations, strides, pads, group.
  // -------------------------------------------------------------------------
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_names;

  // kernel_shape (spatial dims of the weight). For 1D, prepend H=1.
  std::vector<uint32_t> kernel_shape = node_helper.Get("kernel_shape", std::vector<uint32_t>{});
  if (kernel_shape.empty()) {
    kernel_shape.assign(w_info.shape.begin() + 2, w_info.shape.end());
  }
  if (is_1d) {
    kernel_shape.insert(kernel_shape.begin(), 1);
  }
  const size_t spatial_rank = kernel_shape.size();  // 2 for 1D/2D, 3 for 3D

  // Dilations.
  std::vector<uint32_t> dilations = node_helper.Get("dilations", std::vector<uint32_t>(spatial_rank, 1));
  if (is_1d && dilations.size() == 1) {
    dilations.insert(dilations.begin(), 1);
  }
  {
    QnnParamWrapper p(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_DILATION,
                      {SafeInt<uint32_t>(dilations.size())}, std::vector<uint32_t>(dilations));
    param_names.push_back(p.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(p));
  }

  // Strides.
  std::vector<uint32_t> strides = node_helper.Get("strides", std::vector<uint32_t>(spatial_rank, 1));
  if (is_1d && strides.size() == 1) {
    strides.insert(strides.begin(), 1);
  }
  {
    QnnParamWrapper p(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_STRIDE,
                      {SafeInt<uint32_t>(strides.size())}, std::vector<uint32_t>(strides));
    param_names.push_back(p.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(p));
  }

  // Pads / auto_pad. Channel-last spatial dims of x and y for SAME computation.
  {
    std::vector<uint32_t> pads = node_helper.Get("pads", std::vector<uint32_t>(spatial_rank * 2, 0));
    if (is_1d && pads.size() == 2) {
      // NCW pads [w_begin, w_end] -> 2D [h_begin=0, w_begin, h_end=0, w_end].
      pads = {0, pads[0], 0, pads[1]};
    }
    auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
    if (auto_pad != "NOTSET" && auto_pad != "VALID") {
      auto pad_type = qnn::StringToAutoPadType(auto_pad);
      // Channel-last spatial dims: x_cl_shape is [N, <spatial>, C].
      std::vector<uint32_t> in_spatial(x_cl_shape.begin() + 1, x_cl_shape.end() - 1);
      std::vector<uint32_t> y_cl_shape;  // derive output spatial from ONNX y (NCHW) -> channel-last
      {
        std::vector<uint32_t> y_cf = is_1d
                                         ? std::vector<uint32_t>{y_shape_nchw[0], y_shape_nchw[1], 1, y_shape_nchw[2]}
                                         : y_shape_nchw;
        y_cl_shape = utils::ApplyPermToShape(y_cf, utils::ChannelFirstToLastPerm(conv_rank));
      }
      std::vector<uint32_t> out_spatial(y_cl_shape.begin() + 1, y_cl_shape.end() - 1);
      for (size_t dim = 0; dim < spatial_rank; ++dim) {
        int64_t pad_head = pads[dim];
        int64_t pad_tail = pads[spatial_rank + dim];
        RETURN_IF_ERROR(qnn::ComputePad(in_spatial[dim], strides[dim], kernel_shape[dim],
                                        dilations[dim], pad_type, pad_head, pad_tail));
        pads[dim] = gsl::narrow<uint32_t>(pad_head);
        pads[spatial_rank + dim] = gsl::narrow<uint32_t>(pad_tail);
      }
    }
    ReArrangePads(pads);  // [b0,b1,...,e0,e1,...] -> [[b0,e0],[b1,e1],...]
    uint32_t pad_dims = gsl::narrow<uint32_t>(pads.size() / 2);
    QnnParamWrapper p(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_PAD_AMOUNT,
                      {pad_dims, 2}, std::move(pads));
    param_names.push_back(p.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(p));
  }

  const uint32_t group = node_helper.Get("group", static_cast<uint32_t>(1));
  const uint32_t num_in_channels = x_shape_nchw[1];
  const uint32_t num_out_channels = y_shape_nchw[1];

  // DepthWiseConv2d only exists for 2D (incl. reshaped-1D); 3D always uses Conv3d with a group param.
  const bool is_depthwise = (!is_3d) && (num_in_channels == num_out_channels) && (group == num_out_channels);

  if (!is_depthwise) {
    Qnn_Scalar_t group_scalar = QNN_SCALAR_INIT;
    group_scalar.dataType = QNN_DATATYPE_UINT_32;
    group_scalar.uint32Value = group;
    QnnParamWrapper p(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_GROUP, group_scalar);
    param_names.push_back(p.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(p));
  }

  const char* qnn_op = is_3d ? QNN_OP_CONV_3D : (is_depthwise ? QNN_OP_DEPTH_WISE_CONV_2D : QNN_OP_CONV_2D);

  // -------------------------------------------------------------------------
  // Conv output (channel-last) + Transpose back to NCHW (+ Reshape for 1D).
  // -------------------------------------------------------------------------
  QnnQuantParamsWrapper quant_y;
  RETURN_IF_ERROR(BuildPerTensorQuantParam(qnn_model_wrapper, inputs[kIdxYScale], inputs[kIdxYZeroPoint], quant_y));
  Qnn_DataType_t qnn_dtype_y = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, outputs[0].type, qnn_dtype_y));

  // Channel-first output shape used after transpose-back (NCHW or NC1W for 1D, NCDHW for 3D).
  std::vector<uint32_t> y_cf_shape = is_1d
                                         ? std::vector<uint32_t>{y_shape_nchw[0], y_shape_nchw[1], 1, y_shape_nchw[2]}
                                         : y_shape_nchw;
  const std::vector<uint32_t> y_cl_shape = utils::ApplyPermToShape(y_cf_shape, utils::ChannelFirstToLastPerm(conv_rank));

  const std::string y_cl_name = node_base + "_y_nhwc";
  {
    QnnTensorWrapper y_cl_tensor(y_cl_name, QNN_TENSOR_TYPE_NATIVE, qnn_dtype_y, quant_y.Copy(),
                                 std::vector<uint32_t>(y_cl_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(y_cl_tensor)), "Failed to add Conv output tensor.");
  }

  std::vector<std::string> conv_inputs = {x_cl_name, w_hwcn_name};
  if (has_bias) {
    conv_inputs.push_back(bias_name);
  }
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_base, QNN_OP_PACKAGE_NAME_QTI_AISW, qnn_op,
                                                std::vector<std::string>(conv_inputs), {y_cl_name},
                                                std::vector<std::string>(param_names), do_op_validation),
                "Failed to create QLinearConv node.");

  // Transpose channel-last Conv output back to channel-first.
  const std::vector<uint32_t> cl_to_cf = utils::ChannelLastToFirstPerm(conv_rank);
  const std::string& y_name = outputs[0].name;
  const bool y_is_graph_output = qnn_model_wrapper.IsGraphOutput(y_name);

  if (is_1d) {
    // Transpose NHWC->NC1W (intermediate), then Reshape NC1W->NCW (final ONNX output).
    const std::string y_cf_name = node_base + "_y_nchw";
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), y_cl_name, y_cf_name,
                                                       y_cl_shape, cl_to_cf, y_cf_shape,
                                                       qnn_dtype_y, quant_y,
                                                       do_op_validation,
                                                       /*is_for_input=*/false, /*is_for_output=*/false));
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(y_cf_name, y_name, y_cf_shape, y_shape_nchw,
                                                     qnn_dtype_y, quant_y, do_op_validation,
                                                     /*is_for_input=*/false, /*is_for_output=*/y_is_graph_output));
  } else {
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), y_cl_name, y_name,
                                                       y_cl_shape, cl_to_cf, y_cf_shape,
                                                       qnn_dtype_y, quant_y,
                                                       do_op_validation,
                                                       /*is_for_input=*/false, /*is_for_output=*/y_is_graph_output));
  }

  return Ort::Status();
}

void CreateQLinearConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QLinearConvOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
