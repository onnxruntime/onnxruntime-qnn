// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/dq_integer_op_fusion_utils.h"

#include <cstring>
#include <gsl/gsl>
#include <string>
#include <string_view>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

// Reads a FLOAT initializer (referenced by `iodef`) into a typed float vector.
Ort::Status ReadFloatInitializer(const QnnModelWrapper& qmw,
                                 const OrtNodeUnitIODef& iodef,
                                 std::vector<float>& out) {
  const std::string& name = iodef.name;
  const OrtValueInfo* info = qmw.GetConstantTensor(name);
  RETURN_IF_NOT(info != nullptr, ("Constant tensor not found: " + name).c_str());

  RETURN_IF_NOT(iodef.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                ("Expected FLOAT initializer for " + name).c_str());

  std::vector<uint8_t> bytes;
  RETURN_IF_ERROR(qmw.UnpackInitializerData(info, bytes));
  RETURN_IF_NOT(bytes.size() % sizeof(float) == 0, "Unexpected byte count for float initializer");

  out.resize(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return Ort::Status();
}

// Reads a zero-point initializer (INT8 or UINT8) as int32 values. Empty `out` if `zp_iodef` is
// null / non-existent (caller treats empty as zero zero-point).
Ort::Status ReadZeroPointAsInt32(const QnnModelWrapper& qmw,
                                 const OrtNodeUnitIODef* zp_iodef,
                                 std::vector<int32_t>& out) {
  out.clear();
  if (zp_iodef == nullptr || !zp_iodef->Exists()) {
    return Ort::Status();
  }
  const std::string& name = zp_iodef->name;
  const OrtValueInfo* info = qmw.GetConstantTensor(name);
  RETURN_IF_NOT(info != nullptr, ("Constant tensor not found: " + name).c_str());

  const ONNXTensorElementDataType elem_type = zp_iodef->type;
  RETURN_IF_NOT(elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
                    elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                ("Expected INT8 or UINT8 zero-point for " + name).c_str());

  std::vector<uint8_t> bytes;
  RETURN_IF_ERROR(qmw.UnpackInitializerData(info, bytes));

  out.resize(bytes.size());
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
    const int8_t* src = reinterpret_cast<const int8_t*>(bytes.data());
    for (size_t i = 0; i < bytes.size(); ++i) {
      out[i] = static_cast<int32_t>(src[i]);
    }
  } else {
    for (size_t i = 0; i < bytes.size(); ++i) {
      out[i] = static_cast<int32_t>(bytes[i]);
    }
  }
  return Ort::Status();
}

}  // namespace

DqlLookupResult FindParentDql(
    const OrtNodeUnit& integer_op,
    const OrtNodeUnitIODef& a_q_input,
    std::string_view fusion_type_tag,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  DqlLookupResult result;

  const Ort::ConstNode integer_op_node(&integer_op.GetNode());
  const OrtNode* dql_node_raw = nullptr;
  for (const Ort::ConstValueInfo& input_info : integer_op_node.GetInputs()) {
    if (input_info.GetName() != a_q_input.name) {
      continue;
    }
    const Ort::ConstNode parent = input_info.GetProducerNode().node;
    dql_node_raw = static_cast<const OrtNode*>(parent);
    break;
  }
  if (dql_node_raw == nullptr) {
    return result;
  }

  const auto dql_it = node_to_node_unit.find(dql_node_raw);
  if (dql_it == node_to_node_unit.end()) {
    return result;
  }

  const auto claim_it = qnn_node_group_map.find(dql_it->second);
  if (claim_it != qnn_node_group_map.end()) {
    if (claim_it->second->Type() != fusion_type_tag) {
      return result;  // claimed by a different fusion type: cannot share
    }
    result.already_claimed_by_sibling = true;
  }

  result.dql = dql_it->second;
  return result;
}

bool ConsumersAreAllOfType(
    const Ort::ConstValueInfo& value_info,
    std::string_view expected_op_type,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit) {
  if (value_info.IsGraphOutput()) {
    return false;
  }
  for (const auto& c : value_info.GetConsumers()) {
    if (c.node == nullptr) return false;
    const auto it = node_to_node_unit.find(c.node);
    if (it == node_to_node_unit.end()) return false;
    const OrtNodeUnit* nu = it->second;
    if (nu->OpType() != expected_op_type || nu->UnitType() != OrtNodeUnit::Type::SingleNode) {
      return false;
    }
  }
  return true;
}

bool ConsumersAreAllParallelMuls(
    const Ort::ConstValueInfo& value_info,
    const QnnModelWrapper& qmw,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit) {
  if (value_info.IsGraphOutput()) {
    return false;
  }
  const std::string a_scale_name(value_info.GetName());
  for (const auto& c : value_info.GetConsumers()) {
    if (c.node == nullptr) return false;
    const auto it = node_to_node_unit.find(c.node);
    if (it == node_to_node_unit.end()) return false;
    const OrtNodeUnit* nu = it->second;
    if (nu->OpType() != kOpMul || nu->UnitType() != OrtNodeUnit::Type::SingleNode) return false;
    if (nu->Inputs().size() != 2 || nu->Outputs().size() != 1) return false;
    const auto& mul_inputs = nu->Inputs();
    const std::string& other_name = (mul_inputs[0].name == a_scale_name) ? mul_inputs[1].name
                                                                         : mul_inputs[0].name;
    if (!qmw.IsConstantInput(other_name)) return false;
  }
  return true;
}

Ort::Status BuildWeightQuantParams(const QnnModelWrapper& qmw,
                                   const OrtNodeUnitIODef& b_scale_iodef,
                                   const OrtNodeUnitIODef* b_zp_iodef,
                                   uint32_t out_channels,
                                   int32_t per_channel_axis,
                                   QnnQuantParamsWrapper& out_params) {
  std::vector<float> scales;
  RETURN_IF_ERROR(ReadFloatInitializer(qmw, b_scale_iodef, scales));
  RETURN_IF_NOT(!scales.empty(), "B_scale has zero elements");

  std::vector<int32_t> offsets;
  RETURN_IF_ERROR(ReadZeroPointAsInt32(qmw, b_zp_iodef, offsets));
  for (int32_t& v : offsets) v = -v;

  if (scales.size() == 1) {
    const int32_t offset = offsets.empty() ? 0 : offsets[0];
    out_params = QnnQuantParamsWrapper::PerTensor(scales[0], offset);
    return Ort::Status();
  }

  RETURN_IF_NOT(static_cast<uint32_t>(scales.size()) == out_channels,
                "Per-channel B_scale length must equal weight out_channels");
  if (offsets.empty()) {
    offsets.assign(scales.size(), 0);
  } else if (offsets.size() == 1) {
    const int32_t offset = offsets[0];
    offsets.assign(scales.size(), offset);
  } else {
    RETURN_IF_NOT(offsets.size() == scales.size(),
                  "B_zp length must equal B_scale length for per-channel");
  }

  out_params = QnnQuantParamsWrapper::PerChannel(gsl::span<const float>(scales),
                                                 gsl::span<const int32_t>(offsets),
                                                 per_channel_axis);
  return Ort::Status();
}

Ort::Status PreDequantizePerChannelWeight(const QnnModelWrapper& qmw,
                                          const OrtNodeUnitIODef& b_scale_iodef,
                                          const OrtNodeUnitIODef* b_zp_iodef,
                                          bool is_signed_weight,
                                          uint32_t out_channels,
                                          const std::vector<uint8_t>& quant_bytes,
                                          std::vector<uint8_t>& out_float_bytes) {
  std::vector<float> scales;
  RETURN_IF_ERROR(ReadFloatInitializer(qmw, b_scale_iodef, scales));
  RETURN_IF_NOT(scales.size() == static_cast<size_t>(out_channels),
                "Per-channel B_scale length mismatch");

  std::vector<int32_t> zps_onnx;
  RETURN_IF_ERROR(ReadZeroPointAsInt32(qmw, b_zp_iodef, zps_onnx));
  if (zps_onnx.empty()) {
    zps_onnx.assign(scales.size(), 0);
  } else if (zps_onnx.size() == 1) {
    const int32_t zp = zps_onnx[0];
    zps_onnx.assign(scales.size(), zp);
  } else {
    RETURN_IF_NOT(zps_onnx.size() == scales.size(), "Per-channel B_zp length mismatch");
  }

  const size_t num_elems = quant_bytes.size();
  const size_t n = static_cast<size_t>(out_channels);
  RETURN_IF_NOT(n > 0 && num_elems % n == 0,
                "Weight byte count not divisible by out_channels");

  // Dequantize into a typed float buffer first to avoid uint8_t-to-float aliasing issues,
  // then memcpy out to the byte buffer that QnnTensorWrapper expects. Per-channel scales / zps
  // are applied along the LAST axis of the byte buffer, which holds for both Conv's HWCN
  // weight and MatMul's [K,N] weight.
  std::vector<float> floats(num_elems);
  if (is_signed_weight) {
    const int8_t* src = reinterpret_cast<const int8_t*>(quant_bytes.data());
    for (size_t i = 0; i < num_elems; ++i) {
      const size_t c = i % n;
      floats[i] = scales[c] * static_cast<float>(static_cast<int32_t>(src[i]) - zps_onnx[c]);
    }
  } else {
    const uint8_t* src = quant_bytes.data();
    for (size_t i = 0; i < num_elems; ++i) {
      const size_t c = i % n;
      floats[i] = scales[c] * static_cast<float>(static_cast<int32_t>(src[i]) - zps_onnx[c]);
    }
  }

  out_float_bytes.resize(num_elems * sizeof(float));
  std::memcpy(out_float_bytes.data(), floats.data(), out_float_bytes.size());
  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
