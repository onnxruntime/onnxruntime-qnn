// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/opbuilder/qdq_constant_folding.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <SafeInt.hpp>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

Ort::Status GetEffectivelyConstantTensorBytes(QnnModelWrapper& qnn_model_wrapper,
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

namespace {

// SafeInt guards against overflow from an adversarial shape before allocation.
Ort::Status ComputeNumElements(gsl::span<const uint32_t> shape, /*out*/ size_t& num_elems) {
  SafeInt<size_t> safe_num_elems = 1;
  for (uint32_t d : shape) {
    safe_num_elems *= d;
  }
  num_elems = safe_num_elems;
  return Ort::Status();
}

Ort::Status UnpackQuantParams(QnnModelWrapper& qnn_model_wrapper,
                              const OrtNodeUnitIODef::QuantParam& quant_param,
                              /*out*/ std::vector<float>& scales,
                              /*out*/ std::vector<int32_t>& offsets) {
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(quant_param.scale, scales));
  if (quant_param.zero_point != nullptr) {
    ONNXTensorElementDataType zp_type;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(quant_param.zero_point, offsets, zp_type));
  } else {
    // ONNX treats a missing zero_point as zero; match that for both per-tensor and per-channel.
    offsets.assign(scales.size(), 0);
  }
  return Ort::Status();
}

}  // namespace

bool CanFoldInitializerPerChannelDequantize(const QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit) {
  // QDQGroup units are owned by the target op builder. This fallback is only for a
  // standalone per-channel DQ that QNN cannot represent directly.
  if (node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return false;
  }
  if (node_unit.OpType() != "DequantizeLinear" || node_unit.Inputs().empty()) {
    return false;
  }
  const OrtNodeUnitIODef& input_def = node_unit.Inputs()[0];
  if (!qnn_model_wrapper.IsConstantInput(input_def.name)) {
    return false;
  }
  bool is_per_channel = false;
  int64_t axis = 0;
  if (!qnn_model_wrapper.IsPerChannelQuantized(input_def, is_per_channel, axis).IsOK()) {
    return false;
  }
  return is_per_channel;
}

Ort::Status FoldInitializerPerChannelDequantize(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit) {
  const auto& input_def = node_unit.Inputs()[0];
  const auto& output_def = node_unit.Outputs()[0];

  RETURN_IF(!input_def.quant_param.has_value(), "DQ input has no quant param.");

  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_info));
  RETURN_IF(output_info.qnn_data_type != QNN_DATATYPE_FLOAT_32,
            "Folded DequantizeLinear only supports float32 output.");

  std::vector<uint8_t> quant_bytes;
  RETURN_IF_ERROR(GetEffectivelyConstantTensorBytes(qnn_model_wrapper, input_def.name, quant_bytes));

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, input_info));

  std::vector<float> scales;
  std::vector<int32_t> offsets;
  RETURN_IF_ERROR(UnpackQuantParams(qnn_model_wrapper, *input_def.quant_param, scales, offsets));

  size_t num_elems = 0;
  RETURN_IF_ERROR(ComputeNumElements(gsl::make_span(input_info.shape), num_elems));
  std::vector<float> fp32_data(num_elems);

  // CanFoldInitializerPerChannelDequantize already established this is per-channel.
  bool is_per_channel = false;
  int64_t per_channel_axis = 0;
  RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(input_def, is_per_channel, per_channel_axis));

  RETURN_IF_ERROR(utils::DequantizePerChannel(
      gsl::make_span(quant_bytes), gsl::make_span(input_info.shape),
      gsl::make_span(scales), gsl::make_span(offsets),
      gsl::make_span(fp32_data), input_info.qnn_data_type, per_channel_axis));

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
  // Mark as folded so downstream consumers recognize it as compile-time data.
  qnn_model_wrapper.MarkTensorAsFoldedConstant(output_def.name);
  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
