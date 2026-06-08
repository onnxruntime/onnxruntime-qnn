// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <gsl/gsl>

#include <cstring>
#include <functional>
#include <numeric>
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
 * rather than as QDQ node metadata. Since QNN encodes quant params in tensor metadata, we read the
 * scale/zp initializers, build QnnQuantParamsWrapper objects, and attach them to the QNN tensor
 * wrappers for x, w, bias, and y. Shape/attribute handling (layout, weight OIHW->HWIO transpose,
 * pads/strides/dilations/group, depthwise detection, 1D reshape) mirrors ConvOpBuilder.
 *
 * The activation (input 0) arrives in NHWC layout because QLinearConv is opted into ORT core's
 * layout transformer (see ShouldConvertDataLayoutForOpImpl). The weight (input 3) stays NCHW/OIHW
 * and is transposed to HWIO here.
 */
class QLinearConvOpBuilder : public BaseOpBuilder {
 public:
  QLinearConvOpBuilder() : BaseOpBuilder("QLinearConvOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QLinearConvOpBuilder);

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
  Ort::Status ProcessConv1DInputs(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const Ort::Logger& logger,
                                  std::vector<std::string>& input_names,
                                  bool do_op_validation) const ORT_MUST_USE_RESULT;
  Ort::Status ProcessConv2D3DInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const Ort::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const ORT_MUST_USE_RESULT;

  Ort::Status GetInputChannelNumber(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    uint32_t& input_channel_number) const;

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

  out_quant_param = QnnQuantParamsWrapper(scale, offset);
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
    out_quant_param = QnnQuantParamsWrapper(scales[0], offset);
  } else {
    // Per-channel weight quantization on output-channel axis (axis 0 in OIHW).
    RETURN_IF(scales.size() != static_cast<size_t>(num_output_channels),
              "QLinearConv: per-channel w_scale size must equal the number of output channels (M).");
    if (offsets.empty()) {
      offsets.assign(scales.size(), 0);
    }
    RETURN_IF(offsets.size() != scales.size(),
              "QLinearConv: w_zero_point size must match w_scale size for per-channel quantization.");
    out_quant_param = QnnQuantParamsWrapper(gsl::span<const float>(scales),
                                            gsl::span<const int32_t>(offsets),
                                            /*axis=*/0, /*is_int4=*/false);
  }
  return Ort::Status();
}

Ort::Status QLinearConvOpBuilder::BuildBiasQuantParam(const QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      QnnQuantParamsWrapper& out_quant_param) {
  const auto& inputs = node_unit.Inputs();

  // x_scale: scalar.
  float x_scale = 0.0f;
  RETURN_IF_ERROR(ReadScalarScale(qnn_model_wrapper, inputs[kIdxXScale], x_scale));

  // w_scale: scalar or per-channel.
  const OrtValueInfo* w_scale_tensor = qnn_model_wrapper.GetConstantTensor(inputs[kIdxWScale].name);
  RETURN_IF(w_scale_tensor == nullptr, "QLinearConv: could not retrieve w_scale initializer for bias.");
  std::vector<float> w_scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(w_scale_tensor, w_scales));

  // Bias quant: scale = x_scale * w_scale, zero_point = 0 (per ONNX QLinearConv spec).
  if (w_scales.size() == 1) {
    out_quant_param = QnnQuantParamsWrapper(x_scale * w_scales[0], 0);
  } else {
    std::vector<float> bias_scales(w_scales.size());
    for (size_t i = 0; i < w_scales.size(); ++i) {
      bias_scales[i] = x_scale * w_scales[i];
    }
    std::vector<int32_t> bias_offsets(w_scales.size(), 0);
    out_quant_param = QnnQuantParamsWrapper(gsl::span<const float>(bias_scales),
                                            gsl::span<const int32_t>(bias_offsets),
                                            /*axis=*/0, /*is_int4=*/false);
  }
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// IsOpSupported
// ---------------------------------------------------------------------------

Ort::Status QLinearConvOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const Ort::Logger& logger) const {
  // Use QNN's validation API once the layout transformer has produced the NHWC node.
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

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

  return Ort::Status();
}

// ---------------------------------------------------------------------------
// ProcessInputs
// ---------------------------------------------------------------------------

Ort::Status QLinearConvOpBuilder::GetInputChannelNumber(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        uint32_t& input_channel_number) const {
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[kIdxX].shape, input_shape), "Cannot get shape");
  // Activation is NHWC after layout transform: channels are the last dim.
  input_channel_number = input_shape.back();
  return Ort::Status();
}

Ort::Status QLinearConvOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const Ort::Logger& logger,
                                                std::vector<std::string>& input_names,
                                                bool do_op_validation) const {
  std::vector<uint32_t> input0_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[kIdxX].shape, input0_shape),
                "QNN EP: Cannot get shape for QLinearConv input x");

  if (input0_shape.size() == 3) {
    return ProcessConv1DInputs(qnn_model_wrapper, node_unit, logger, input_names, do_op_validation);
  } else if (input0_shape.size() == 4 || input0_shape.size() == 5) {
    return ProcessConv2D3DInputs(qnn_model_wrapper, node_unit, logger, input_names, do_op_validation);
  }
  return MAKE_EP_FAIL("QNN QLinearConv only supports 1D (rank 3), 2D (rank 4), or 3D (rank 5) inputs.");
}

Ort::Status QLinearConvOpBuilder::ProcessConv2D3DInputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        const Ort::Logger& logger,
                                                        std::vector<std::string>& input_names,
                                                        bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();

  //
  // Input 0: activation x (NHWC). Build per-tensor quant params and attach to tensor wrapper.
  //
  {
    const std::string& x_name = inputs[kIdxX].name;
    QnnQuantParamsWrapper quant_x;
    RETURN_IF_ERROR(BuildPerTensorQuantParam(qnn_model_wrapper, inputs[kIdxXScale],
                                             inputs[kIdxXZeroPoint], quant_x));

    Qnn_DataType_t qnn_dtype_x = QNN_DATATYPE_UNDEFINED;
    RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxX].type, qnn_dtype_x));

    std::vector<uint32_t> x_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[kIdxX].shape, x_shape), "Cannot get x shape");

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(x_name)) {
      std::vector<uint8_t> unpacked;
      if (qnn_model_wrapper.IsEffectivelyConstantInput(x_name)) {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(
            qnn_model_wrapper.GetConstantTensor(x_name), unpacked));
      }
      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(x_name);
      QnnTensorWrapper x_tensor(x_name, tensor_type, qnn_dtype_x, std::move(quant_x),
                                std::move(x_shape), std::move(unpacked));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x_tensor)), "Failed to add x tensor.");
    }
    input_names.push_back(x_name);
  }

  //
  // Input 3: weight w (NCHW/OIHW). Build quant params (per-tensor or per-channel), transpose OIHW->HWIO.
  //
  {
    const std::string& w_name = inputs[kIdxW].name;
    TensorInfo w_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kIdxW], w_info));
    const bool is_3d = (w_info.shape.size() == 5);
    const uint32_t num_output_channels = w_info.shape[0];

    QnnQuantParamsWrapper quant_w;
    RETURN_IF_ERROR(BuildWeightQuantParam(qnn_model_wrapper, inputs[kIdxWScale], inputs[kIdxWZeroPoint],
                                          num_output_channels, quant_w));

    Qnn_DataType_t qnn_dtype_w = QNN_DATATYPE_UNDEFINED;
    RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxW].type, qnn_dtype_w));

    std::vector<uint32_t> hwcn_shape(w_info.shape.size());
    RETURN_IF_ERROR(utils::NchwShapeToHwcn<uint32_t>(w_info.shape, hwcn_shape));

    const std::string actual_w_name = w_info.is_initializer
                                          ? w_name
                                          : utils::UniqueNameGenerator().New(w_name, "_transpose");

    std::vector<uint8_t> unpacked;
    if (w_info.is_initializer) {
      RETURN_IF_ERROR(utils::TransposeFromNchwToHwcn(qnn_model_wrapper, w_info.initializer_tensor,
                                                     unpacked, is_3d));
      // Move the per-channel quant axis through the OIHW->HWIO transpose.
      if (quant_w.IsPerChannel()) {
        const std::vector<size_t>& perm = is_3d ? nchw2hwcn_perm_3d : nchw2hwcn_perm;
        std::vector<size_t> perm_inv(perm.size());
        RETURN_IF_ERROR(utils::InvertPerm<size_t>(perm, perm_inv));
        RETURN_IF_ERROR(quant_w.HandleTranspose<size_t>(perm_inv));
      }
    } else {
      // Dynamic weight: only per-tensor is supported (validated in IsOpSupported).
      RETURN_IF(quant_w.IsPerChannel(), "QLinearConv: dynamic weight only supports per-tensor quantization.");
      if (!qnn_model_wrapper.IsQnnTensorWrapperExist(w_name)) {
        QnnTensorWrapper weight_src;
        RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[kIdxW], weight_src));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_src)), "Failed to add weight tensor.");
      }
      RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(node_unit.Index(),
                                                              w_name,
                                                              actual_w_name,
                                                              w_info.shape,
                                                              hwcn_shape,
                                                              qnn_dtype_w,
                                                              quant_w,
                                                              do_op_validation,
                                                              qnn_model_wrapper.IsGraphInput(w_name),
                                                              false,
                                                              is_3d));
    }

    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(actual_w_name);
    QnnTensorWrapper w_tensor(actual_w_name, tensor_type, qnn_dtype_w, std::move(quant_w),
                              std::move(hwcn_shape), std::move(unpacked));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(w_tensor)), "Failed to add weight tensor.");
    input_names.push_back(actual_w_name);
  }

  //
  // Input 8: bias (optional, int32). Build quant params (scale = x_scale * w_scale, zp = 0).
  //
  const bool has_bias = num_inputs > kIdxBias && inputs[kIdxBias].Exists();
  if (has_bias) {
    const std::string& bias_name = inputs[kIdxBias].name;
    TensorInfo bias_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kIdxBias], bias_info));
    RETURN_IF(!bias_info.is_initializer, "QLinearConv: bias must be a constant initializer.");

    QnnQuantParamsWrapper quant_bias;
    RETURN_IF_ERROR(BuildBiasQuantParam(qnn_model_wrapper, node_unit, quant_bias));

    std::vector<uint8_t> unpacked;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor, unpacked));

    QnnTensorWrapper bias_tensor(bias_name, QNN_TENSOR_TYPE_STATIC, bias_info.qnn_data_type,
                                 std::move(quant_bias), std::vector<uint32_t>(bias_info.shape),
                                 std::move(unpacked));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias tensor.");
    input_names.push_back(bias_name);
  }

  return Ort::Status();
}

Ort::Status QLinearConvOpBuilder::ProcessConv1DInputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      const Ort::Logger& logger,
                                                      std::vector<std::string>& input_names,
                                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();
  ORT_UNUSED_PARAMETER(logger);

  //
  // Input 0: activation x. Reshape NHWC-1D [N,W,C] -> [N,1,W,C] (insert H=1).
  //
  {
    const std::string& x_name = inputs[kIdxX].name;
    QnnQuantParamsWrapper quant_x;
    RETURN_IF_ERROR(BuildPerTensorQuantParam(qnn_model_wrapper, inputs[kIdxXScale],
                                             inputs[kIdxXZeroPoint], quant_x));

    Qnn_DataType_t qnn_dtype_x = QNN_DATATYPE_UNDEFINED;
    RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxX].type, qnn_dtype_x));

    std::vector<uint32_t> x_shape_1d;  // [N, W, C] (NHWC, layout-transformed)
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[kIdxX].shape, x_shape_1d), "Cannot get x shape");
    std::vector<uint32_t> x_shape_2d = {x_shape_1d[0], 1, x_shape_1d[1], x_shape_1d[2]};

    const std::string conv_x_name = qnn_model_wrapper.IsEffectivelyConstantInput(x_name)
                                        ? x_name
                                        : utils::UniqueNameGenerator().New(x_name, "_reshape");

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(conv_x_name)) {
      std::vector<uint8_t> unpacked;
      if (qnn_model_wrapper.IsEffectivelyConstantInput(x_name)) {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(
            qnn_model_wrapper.GetConstantTensor(x_name), unpacked));
      } else {
        RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(x_name, conv_x_name, x_shape_1d, x_shape_2d,
                                                         qnn_dtype_x, quant_x, do_op_validation,
                                                         qnn_model_wrapper.IsGraphInput(x_name)));
      }
      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(conv_x_name);
      QnnTensorWrapper x_tensor(conv_x_name, tensor_type, qnn_dtype_x, std::move(quant_x),
                                std::move(x_shape_2d), std::move(unpacked));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x_tensor)), "Failed to add x tensor.");
    }
    input_names.push_back(conv_x_name);
  }

  //
  // Input 3: weight. Reshape [M,C,W] -> [M,C,1,W], then transpose to HWIO.
  //
  {
    const std::string& w_name = inputs[kIdxW].name;
    TensorInfo w_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kIdxW], w_info));
    RETURN_IF(w_info.shape.size() != 3, "QLinearConv 1D: weight must be rank-3 [M, C, W].");
    const uint32_t num_output_channels = w_info.shape[0];

    QnnQuantParamsWrapper quant_w;
    RETURN_IF_ERROR(BuildWeightQuantParam(qnn_model_wrapper, inputs[kIdxWScale], inputs[kIdxWZeroPoint],
                                          num_output_channels, quant_w));

    Qnn_DataType_t qnn_dtype_w = QNN_DATATYPE_UNDEFINED;
    RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxW].type, qnn_dtype_w));

    // [M, C, W] -> [M, C, 1, W]
    std::vector<uint32_t> shape_2d = {w_info.shape[0], w_info.shape[1], 1, w_info.shape[2]};
    std::vector<uint32_t> hwcn_shape(4);
    RETURN_IF_ERROR(utils::NchwShapeToHwcn<uint32_t>(shape_2d, hwcn_shape));

    const std::string actual_w_name = w_info.is_initializer
                                          ? w_name
                                          : utils::UniqueNameGenerator().New(w_name, "_transpose");

    std::vector<uint8_t> unpacked;
    if (w_info.is_initializer) {
      // Reshape the quant axis if per-channel, then transpose a [M,C,1,W] view to HWIO.
      if (quant_w.IsPerChannel()) {
        RETURN_IF_ERROR(quant_w.HandleUnsqueeze<uint32_t>(w_info.shape, shape_2d));
      }
      std::vector<int64_t> shape_2d_i64(shape_2d.begin(), shape_2d.end());

      std::vector<uint8_t> original_bytes;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(w_info.initializer_tensor, original_bytes));
      unpacked.resize(original_bytes.size());

      const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
      const OrtTypeInfo* type_info = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(
          static_cast<const OrtValueInfo*>(w_info.initializer_tensor), &type_info));
      const OrtTensorTypeAndShapeInfo* type_shape = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &type_shape));
      ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(type_shape, &elem_type));
      const size_t elem_byte_size = utils::GetElementSizeByType(elem_type);
      RETURN_IF(elem_byte_size == 0,
                ("QLinearConv 1D: can't get element byte size for weight " + w_name).c_str());

      RETURN_IF_ERROR(utils::TransposeFromNchwToHwcn(std::move(shape_2d_i64), elem_byte_size,
                                                     original_bytes, unpacked, /*is_3d=*/false));

      if (quant_w.IsPerChannel()) {
        std::vector<size_t> perm_inv(nchw2hwcn_perm.size());
        RETURN_IF_ERROR(utils::InvertPerm<size_t>(nchw2hwcn_perm, perm_inv));
        RETURN_IF_ERROR(quant_w.HandleTranspose<size_t>(perm_inv));
      }
    } else {
      RETURN_IF(quant_w.IsPerChannel(), "QLinearConv: dynamic weight only supports per-tensor quantization.");
      if (!qnn_model_wrapper.IsQnnTensorWrapperExist(w_name)) {
        QnnTensorWrapper weight_src;
        RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[kIdxW], weight_src));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_src)), "Failed to add weight tensor.");
      }
      const std::string reshape_output = utils::UniqueNameGenerator().New(w_name, "_reshape");
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(w_name, reshape_output, w_info.shape, shape_2d,
                                                       qnn_dtype_w, quant_w, do_op_validation,
                                                       qnn_model_wrapper.IsGraphInput(w_name)));
      RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(node_unit.Index(), reshape_output,
                                                              actual_w_name, shape_2d, hwcn_shape,
                                                              qnn_dtype_w, quant_w, do_op_validation,
                                                              false, false, /*is_3d=*/false));
    }

    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(actual_w_name);
    QnnTensorWrapper w_tensor(actual_w_name, tensor_type, qnn_dtype_w, std::move(quant_w),
                              std::move(hwcn_shape), std::move(unpacked));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(w_tensor)), "Failed to add weight tensor.");
    input_names.push_back(actual_w_name);
  }

  //
  // Input 8: bias (optional). Shape [M] is layout-invariant; no reshape needed.
  //
  const bool has_bias = num_inputs > kIdxBias && inputs[kIdxBias].Exists();
  if (has_bias) {
    const std::string& bias_name = inputs[kIdxBias].name;
    TensorInfo bias_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kIdxBias], bias_info));
    RETURN_IF(!bias_info.is_initializer, "QLinearConv: bias must be a constant initializer.");

    QnnQuantParamsWrapper quant_bias;
    RETURN_IF_ERROR(BuildBiasQuantParam(qnn_model_wrapper, node_unit, quant_bias));

    std::vector<uint8_t> unpacked;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor, unpacked));

    QnnTensorWrapper bias_tensor(bias_name, QNN_TENSOR_TYPE_STATIC, bias_info.qnn_data_type,
                                 std::move(quant_bias), std::vector<uint32_t>(bias_info.shape),
                                 std::move(unpacked));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias tensor.");
    input_names.push_back(bias_name);
  }

  return Ort::Status();
}

// ---------------------------------------------------------------------------
// ProcessAttributesAndOutputs
// ---------------------------------------------------------------------------

Ort::Status QLinearConvOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                              const OrtNodeUnit& node_unit,
                                                              std::vector<std::string>&& input_names,
                                                              const Ort::Logger& logger,
                                                              bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].shape, output_shape), "Cannot get output shape");
  const bool is_1d_conv = output_shape.size() == 3;
  const bool is_3d_conv = output_shape.size() == 5;

  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  std::vector<uint32_t> input_0_shape;  // NHW[D]C
  std::vector<uint32_t> input_1_shape;  // NCHW[D]
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[kIdxX].shape, input_0_shape), "Cannot get x shape");
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[kIdxW].shape, input_1_shape), "Cannot get w shape");

  // Kernel shape (infer from weight spatial dims if absent).
  std::vector<uint32_t> kernel_shape = node_helper.Get("kernel_shape", std::vector<uint32_t>{});
  if (kernel_shape.empty()) {
    kernel_shape.assign(input_1_shape.begin() + 2, input_1_shape.end());
  }
  if (is_1d_conv) {
    kernel_shape.insert(kernel_shape.begin(), 1);  // insert H=1
  }

  // Dilations.
  std::vector<uint32_t> dilations;
  dilations.assign(kernel_shape.size(), 1);
  dilations = node_helper.Get("dilations", dilations);
  if (dilations.size() == 1) {
    const uint32_t width_dilation = dilations[0];
    dilations.resize(2);
    dilations[0] = 1;  // H == 1
    dilations[1] = width_dilation;
  }
  {
    QnnParamWrapper dilation_paramwrapper(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_DILATION,
                                          {SafeInt<uint32_t>(dilations.size())}, std::vector<uint32_t>(dilations));
    param_tensor_names.push_back(dilation_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(dilation_paramwrapper));
  }

  // Strides.
  std::vector<uint32_t> strides;
  strides.assign(kernel_shape.size(), 1);
  strides = node_helper.Get("strides", strides);
  if (strides.size() == 1) {
    const uint32_t width_stride = strides[0];
    strides.resize(2);
    strides[0] = 1;  // H
    strides[1] = width_stride;
  }
  {
    QnnParamWrapper stride_paramwrapper(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_STRIDE,
                                        {SafeInt<uint32_t>(strides.size())}, std::vector<uint32_t>(strides));
    param_tensor_names.push_back(stride_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(stride_paramwrapper));
  }

  // Pads / auto_pad.
  {
    std::vector<uint32_t> pads;
    pads.assign(kernel_shape.size() * 2, 0);
    pads = node_helper.Get("pads", pads);
    auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
    RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER" && auto_pad != "VALID",
              ("QNN QLinearConv does not support 'auto_pad' value: " + auto_pad).c_str());

    if (auto_pad != "NOTSET" && auto_pad != "VALID") {
      auto pad_type = qnn::StringToAutoPadType(auto_pad);
      std::vector<uint32_t> input_dims(input_0_shape.begin() + 1, input_0_shape.end() - 1);  // NHWC -> H,W[,D]
      std::vector<uint32_t> output_dims(output_shape.begin() + 1, output_shape.end() - 1);
      if (is_1d_conv) {
        input_dims.insert(input_dims.begin(), 1);
        output_dims.insert(output_dims.begin(), 1);
      }
      size_t rank = input_dims.size();
      for (size_t dim = 0; dim < rank; ++dim) {
        int64_t pad_head = pads[dim];
        int64_t pad_tail = pads[rank + dim];
        RETURN_IF_ERROR(qnn::ComputePad(input_dims[dim], strides[dim], kernel_shape[dim],
                                        dilations[dim], pad_type, pad_head, pad_tail));
        pads[dim] = gsl::narrow<uint32_t>(pad_head);
        pads[rank + dim] = gsl::narrow<uint32_t>(pad_tail);
      }
    } else if (pads.size() == 2) {
      // 1D NOTSET/VALID: set H pad to 0.
      const uint32_t width_pad_begin = pads[0];
      const uint32_t width_pad_end = pads[1];
      pads.resize(4);
      pads[0] = 0;  // H begin
      pads[1] = width_pad_begin;
      pads[2] = 0;  // H end
      pads[3] = width_pad_end;
    }

    ReArrangePads(pads);
    uint32_t pad_size = gsl::narrow<uint32_t>(pads.size() / 2);
    QnnParamWrapper pad_paramwrapper(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_PAD_AMOUNT,
                                     {pad_size, 2}, std::move(pads));
    param_tensor_names.push_back(pad_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(pad_paramwrapper));
  }

  const uint32_t group = node_helper.Get("group", static_cast<uint32_t>(1));
  const uint32_t num_output_channels = output_shape.back();
  uint32_t num_input_channels = 0;
  RETURN_IF_ERROR(GetInputChannelNumber(qnn_model_wrapper, node_unit, num_input_channels));

  // DepthWiseConv2d exists only for 2D (and reshaped-1D); there is no DepthWiseConv3d.
  const bool is_depthwise_conv2d = (!is_3d_conv) &&
                                   (num_input_channels == num_output_channels) &&
                                   (group == num_output_channels);

  if (!is_depthwise_conv2d) {
    Qnn_Scalar_t group_scalar = QNN_SCALAR_INIT;
    group_scalar.dataType = QNN_DATATYPE_UINT_32;
    group_scalar.uint32Value = group;
    QnnParamWrapper group_paramwrapper(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_GROUP, group_scalar);
    param_tensor_names.push_back(group_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(group_paramwrapper));
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("Using DepthWiseConv2d for QLinearConv node " + node_unit.Name()).c_str());
  }

  std::string qnn_op_type;
  if (is_3d_conv) {
    qnn_op_type = QNN_OP_CONV_3D;
  } else {
    qnn_op_type = is_depthwise_conv2d ? QNN_OP_DEPTH_WISE_CONV_2D : QNN_OP_CONV_2D;
  }

  // Output quant params (per-tensor) from y_scale / y_zero_point.
  QnnQuantParamsWrapper quant_y;
  RETURN_IF_ERROR(BuildPerTensorQuantParam(qnn_model_wrapper, inputs[kIdxYScale], inputs[kIdxYZeroPoint], quant_y));

  Qnn_DataType_t qnn_dtype_y = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, outputs[0].type, qnn_dtype_y));

  const auto& output_name = outputs[0].name;

  if (is_1d_conv) {
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
    std::vector<uint32_t> output_shape_2d = {output_shape[0], 1, output_shape[1], output_shape[2]};
    const std::string conv_output_name = utils::UniqueNameGenerator().New(output_name, "_conv");
    QnnTensorWrapper conv_out(conv_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_dtype_y,
                              quant_y.Copy(), std::vector<uint32_t>(output_shape_2d));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(conv_out)), "Failed to add conv output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, qnn_op_type,
                                                  std::move(input_names), {conv_output_name},
                                                  std::move(param_tensor_names), do_op_validation),
                  "Failed to add QLinearConv node.");
    // Reshape 2D conv output back to 1D.
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(conv_output_name, output_name, output_shape_2d,
                                                     output_shape, qnn_dtype_y, quant_y,
                                                     do_op_validation, false, is_graph_output));
  } else {
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
    Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper output_tensor(output_name, tensor_type, qnn_dtype_y,
                                   std::move(quant_y), std::move(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, qnn_op_type,
                                                  std::move(input_names), {output_name},
                                                  std::move(param_tensor_names), do_op_validation),
                  "Failed to add QLinearConv node.");
  }

  return Ort::Status();
}

void CreateQLinearConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QLinearConvOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
