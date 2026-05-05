// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/opbuilder/qdq_constant_folding.h"

#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <SafeInt.hpp>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Fetch raw bytes of a constant input tensor. Works whether the tensor is a real
// graph initializer or a previously folded constant tensor we placed in the
// model wrapper.
Ort::Status GetConstantTensorBytes(QnnModelWrapper& qnn_model_wrapper,
                                   const std::string& tensor_name,
                                   /*out*/ std::vector<uint8_t>& bytes) {
  if (qnn_model_wrapper.IsConstantInput(tensor_name)) {
    const OrtValueInfo* init = qnn_model_wrapper.GetConstantTensor(tensor_name);
    RETURN_IF(init == nullptr, "Constant initializer not found for tensor.");
    return qnn_model_wrapper.UnpackInitializerData(init, bytes);
  }
  if (qnn_model_wrapper.IsFoldedConstant(tensor_name) &&
      qnn_model_wrapper.IsQnnTensorWrapperExist(tensor_name)) {
    const QnnTensorWrapper& wrapper = qnn_model_wrapper.GetQnnTensorWrapper(tensor_name);
    const Qnn_ClientBuffer_t& buf = GetQnnTensorClientBuf(wrapper.GetQnnTensor());
    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(buf.data);
    bytes.assign(data_ptr, data_ptr + buf.dataSize);
    return Ort::Status();
  }
  return MAKE_EP_FAIL("Tensor is not a constant initializer or folded constant.");
}

// Constant-fold a standalone DequantizeLinear whose input is effectively constant.
// Computes the dequantized fp32 data at compile time and emits a STATIC fp32
// tensor for the DQ output. The DQ op itself is not added to the QNN graph.
//
// Note: only fp32 DQ output is supported. ONNX opset >= 21 also allows fp16
// (and other) DQ outputs; those are rejected here so the caller falls back to
// emitting a normal QNN op.
Ort::Status FoldConstantDequantizeLinear(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit) {
  const auto& input_def = node_unit.Inputs()[0];
  const auto& output_def = node_unit.Outputs()[0];

  RETURN_IF(!input_def.quant_param.has_value(), "DQ input has no quant param.");

  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_info));
  RETURN_IF(output_info.qnn_data_type != QNN_DATATYPE_FLOAT_32,
            "Folded DequantizeLinear only supports float32 output.");

  std::vector<uint8_t> quant_bytes;
  RETURN_IF_ERROR(GetConstantTensorBytes(qnn_model_wrapper, input_def.name, quant_bytes));

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, input_info));

  std::vector<float> scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(input_def.quant_param->scale, scales));

  std::vector<int32_t> offsets;
  if (input_def.quant_param->zero_point != nullptr) {
    ONNXTensorElementDataType zp_type;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(input_def.quant_param->zero_point, offsets, zp_type));
  } else {
    offsets.assign(scales.size(), 0);
  }

  // Use SafeInt to detect a malicious model whose shape product overflows size_t.
  // The downstream allocation would fail anyway, but explicit bounds turn silent
  // wrap into an immediate, attributable error.
  SafeInt<size_t> safe_num_elems = 1;
  for (uint32_t d : input_info.shape) {
    safe_num_elems *= d;
  }
  const size_t num_elems = safe_num_elems;
  std::vector<float> fp32_data(num_elems);

  std::optional<int64_t> axis = std::nullopt;
  bool is_per_chan = false;
  int64_t per_chan_axis = 0;
  RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(input_def, is_per_chan, per_chan_axis));
  if (is_per_chan) {
    axis = per_chan_axis;
  }

  RETURN_IF_ERROR(utils::DequantizePerChannel(
      gsl::make_span(quant_bytes), gsl::make_span(input_info.shape),
      gsl::make_span(scales), gsl::make_span(offsets),
      gsl::make_span(fp32_data), input_info.qnn_data_type, axis));

  std::vector<uint8_t> output_bytes(fp32_data.size() * sizeof(float));
  std::memcpy(output_bytes.data(), fp32_data.data(), output_bytes.size());

  QnnTensorWrapper out_wrapper(output_def.name,
                               QNN_TENSOR_TYPE_STATIC,
                               QNN_DATATYPE_FLOAT_32,
                               QnnQuantParamsWrapper(),
                               std::vector<uint32_t>(output_info.shape),
                               std::move(output_bytes));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_wrapper)),
                "Failed to add folded DequantizeLinear output tensor.");
  qnn_model_wrapper.MarkTensorAsFoldedConstant(output_def.name);
  return Ort::Status();
}

// Constant-fold a standalone QuantizeLinear whose input is effectively constant.
// Computes the quantized data at compile time and emits a STATIC quantized
// tensor for the Q output. The Q op itself is not added to the QNN graph.
//
// Note: only fp32 input is supported. fp16 sources would require an additional
// fp16 -> fp32 conversion step before quantization; not needed by current
// targets.
Ort::Status FoldConstantQuantizeLinear(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit) {
  const auto& input_def = node_unit.Inputs()[0];
  const auto& output_def = node_unit.Outputs()[0];

  RETURN_IF(!output_def.quant_param.has_value(), "Q output has no quant param.");

  std::vector<uint8_t> input_bytes;
  RETURN_IF_ERROR(GetConstantTensorBytes(qnn_model_wrapper, input_def.name, input_bytes));

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, input_info));
  RETURN_IF(input_info.qnn_data_type != QNN_DATATYPE_FLOAT_32,
            "Folded QuantizeLinear only supports float32 input.");

  SafeInt<size_t> safe_num_elems = 1;
  for (uint32_t d : input_info.shape) {
    safe_num_elems *= d;
  }
  const size_t num_elems = safe_num_elems;
  RETURN_IF(input_bytes.size() != SafeInt<size_t>(num_elems) * sizeof(float),
            "QuantizeLinear input byte size mismatch with shape.");
  gsl::span<const float> fp32_input(reinterpret_cast<const float*>(input_bytes.data()), num_elems);

  std::vector<float> scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(output_def.quant_param->scale, scales));

  std::vector<int32_t> offsets;
  if (output_def.quant_param->zero_point != nullptr) {
    ONNXTensorElementDataType zp_type;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(output_def.quant_param->zero_point, offsets, zp_type));
  } else {
    offsets.assign(scales.size(), 0);
  }

  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_info));

  std::optional<int64_t> axis = std::nullopt;
  bool is_per_chan = false;
  int64_t per_chan_axis = 0;
  RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(output_def, is_per_chan, per_chan_axis));
  if (is_per_chan) {
    axis = per_chan_axis;
  }

  // GetQnnTensorDataSizeInBytes correctly accounts for sub-byte dtypes (e.g. int4),
  // unlike GetElementSizeByType which rounds up to a byte per element.
  const size_t total_bytes = utils::GetQnnTensorDataSizeInBytes(num_elems, output_info.qnn_data_type);
  std::vector<uint8_t> quant_bytes(total_bytes);

  RETURN_IF_ERROR(utils::QuantizeData(
      fp32_input, gsl::make_span(input_info.shape),
      gsl::make_span(scales), gsl::make_span(offsets),
      gsl::make_span(quant_bytes), output_info.qnn_data_type, axis));

  QnnTensorWrapper out_wrapper(output_def.name,
                               QNN_TENSOR_TYPE_STATIC,
                               output_info.qnn_data_type,
                               std::move(output_info.quant_param),
                               std::vector<uint32_t>(output_info.shape),
                               std::move(quant_bytes));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_wrapper)),
                "Failed to add folded QuantizeLinear output tensor.");
  qnn_model_wrapper.MarkTensorAsFoldedConstant(output_def.name);
  return Ort::Status();
}

}  // namespace

bool CanFoldConstantQdq(const QnnModelWrapper& qnn_model_wrapper,
                        const OrtNodeUnit& node_unit) {
  // Only standalone Q/DQ. QDQGroup-typed units have a non-Q/DQ target node and would
  // already be handled by the corresponding op builder.
  if (node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return false;
  }
  const std::string& op_type = node_unit.OpType();
  if (op_type != "DequantizeLinear" && op_type != "QuantizeLinear") {
    return false;
  }
  if (node_unit.Inputs().empty()) {
    return false;
  }
  return qnn_model_wrapper.IsEffectivelyConstantInput(node_unit.Inputs()[0].name);
}

Ort::Status TryFoldConstantQDQ(QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit) {
  const std::string& op_type = node_unit.OpType();
  if (op_type == "DequantizeLinear") {
    return FoldConstantDequantizeLinear(qnn_model_wrapper, node_unit);
  }
  if (op_type == "QuantizeLinear") {
    return FoldConstantQuantizeLinear(qnn_model_wrapper, node_unit);
  }
  return MAKE_EP_FAIL("TryFoldConstantQDQ called on a non-Q/DQ node.");
}

}  // namespace qnn
}  // namespace onnxruntime
