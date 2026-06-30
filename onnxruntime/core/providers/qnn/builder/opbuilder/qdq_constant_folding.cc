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

Ort::Status ResolvePerChannelAxis(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnitIODef& io_def,
                                  /*out*/ std::optional<int64_t>& axis) {
  bool is_per_chan = false;
  int64_t per_chan_axis = 0;
  RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(io_def, is_per_chan, per_chan_axis));
  axis = is_per_chan ? std::optional<int64_t>(per_chan_axis) : std::nullopt;
  return Ort::Status();
}

// Marks the tensor as folded so downstream Q/DQ hops keep folding through the chain.
Ort::Status RegisterFoldedStaticTensor(QnnModelWrapper& qnn_model_wrapper,
                                       const std::string& name,
                                       Qnn_DataType_t data_type,
                                       QnnQuantParamsWrapper quant_param,
                                       std::vector<uint32_t> shape,
                                       std::vector<uint8_t> data,
                                       const char* failure_msg) {
  QnnTensorWrapper out_wrapper(name,
                               QNN_TENSOR_TYPE_STATIC,
                               data_type,
                               std::move(quant_param),
                               std::move(shape),
                               std::move(data));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_wrapper)), failure_msg);
  qnn_model_wrapper.MarkTensorAsFoldedConstant(name);
  return Ort::Status();
}

// Only fp32 DQ output is supported; opset >= 21 fp16/bf16 outputs fall back to the normal op.
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
  RETURN_IF_ERROR(GetEffectivelyConstantTensorBytes(qnn_model_wrapper, input_def.name, quant_bytes));

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, input_info));

  std::vector<float> scales;
  std::vector<int32_t> offsets;
  RETURN_IF_ERROR(UnpackQuantParams(qnn_model_wrapper, *input_def.quant_param, scales, offsets));

  size_t num_elems = 0;
  RETURN_IF_ERROR(ComputeNumElements(gsl::make_span(input_info.shape), num_elems));
  std::vector<float> fp32_data(num_elems);

  std::optional<int64_t> axis;
  RETURN_IF_ERROR(ResolvePerChannelAxis(qnn_model_wrapper, input_def, axis));

  RETURN_IF_ERROR(utils::DequantizePerChannel(
      gsl::make_span(quant_bytes), gsl::make_span(input_info.shape),
      gsl::make_span(scales), gsl::make_span(offsets),
      gsl::make_span(fp32_data), input_info.qnn_data_type, axis));

  std::vector<uint8_t> output_bytes(fp32_data.size() * sizeof(float));
  std::memcpy(output_bytes.data(), fp32_data.data(), output_bytes.size());

  return RegisterFoldedStaticTensor(qnn_model_wrapper, output_def.name,
                                    QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(output_info.shape),
                                    std::move(output_bytes),
                                    "Failed to add folded DequantizeLinear output tensor.");
}

// Only fp32 Q input is supported; fp16/bf16 sources fall back to the normal op.
Ort::Status FoldConstantQuantizeLinear(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit) {
  const auto& input_def = node_unit.Inputs()[0];
  const auto& output_def = node_unit.Outputs()[0];

  RETURN_IF(!output_def.quant_param.has_value(), "Q output has no quant param.");

  std::vector<uint8_t> input_bytes;
  RETURN_IF_ERROR(GetEffectivelyConstantTensorBytes(qnn_model_wrapper, input_def.name, input_bytes));

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, input_info));
  RETURN_IF(input_info.qnn_data_type != QNN_DATATYPE_FLOAT_32,
            "Folded QuantizeLinear only supports float32 input.");

  size_t num_elems = 0;
  RETURN_IF_ERROR(ComputeNumElements(gsl::make_span(input_info.shape), num_elems));
  RETURN_IF(input_bytes.size() != SafeInt<size_t>(num_elems) * sizeof(float),
            "QuantizeLinear input byte size mismatch with shape.");
  gsl::span<const float> fp32_input(reinterpret_cast<const float*>(input_bytes.data()), num_elems);

  std::vector<float> scales;
  std::vector<int32_t> offsets;
  RETURN_IF_ERROR(UnpackQuantParams(qnn_model_wrapper, *output_def.quant_param, scales, offsets));

  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_info));

  std::optional<int64_t> axis;
  RETURN_IF_ERROR(ResolvePerChannelAxis(qnn_model_wrapper, output_def, axis));

  // Needed over GetElementSizeByType to correctly size sub-byte dtypes (e.g. int4).
  const size_t total_bytes = utils::GetQnnTensorDataSizeInBytes(num_elems, output_info.qnn_data_type);
  std::vector<uint8_t> quant_bytes(total_bytes);

  RETURN_IF_ERROR(utils::QuantizeData(
      fp32_input, gsl::make_span(input_info.shape),
      gsl::make_span(scales), gsl::make_span(offsets),
      gsl::make_span(quant_bytes), output_info.qnn_data_type, axis));

  return RegisterFoldedStaticTensor(qnn_model_wrapper, output_def.name,
                                    output_info.qnn_data_type,
                                    std::move(output_info.quant_param),
                                    std::vector<uint32_t>(output_info.shape),
                                    std::move(quant_bytes),
                                    "Failed to add folded QuantizeLinear output tensor.");
}

}  // namespace

bool CanFoldConstantQdq(const QnnModelWrapper& qnn_model_wrapper,
                        const OrtNodeUnit& node_unit) {
  // QDQGroup units are owned by the target op builder; only standalone Q/DQ are foldable here.
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
