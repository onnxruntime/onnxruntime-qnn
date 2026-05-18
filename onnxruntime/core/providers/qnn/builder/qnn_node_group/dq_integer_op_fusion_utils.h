// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// Op-type string constants shared by the DQ-IntegerOp fusion family.
constexpr char kOpDynamicQuantizeLinear[] = "DynamicQuantizeLinear";
constexpr char kOpCast[] = "Cast";
constexpr char kOpMul[] = "Mul";
constexpr char kOpAdd[] = "Add";

// Result of walking up an IntegerOp's a_q input to its producing DynamicQuantizeLinear.
struct DqlLookupResult {
  const OrtNodeUnit* dql = nullptr;         // matched DQL NodeUnit, nullptr if not found
  bool already_claimed_by_sibling = false;  // true iff DQL is already claimed by a sibling fusion of `fusion_type_tag`
};

// Walks up `integer_op`'s a_q input to find the producer DynamicQuantizeLinear NodeUnit.
// Tolerates DQL being claimed by a sibling fusion of the same `fusion_type_tag` (multi-IntegerOp-
// shared-DQL case): only the first sibling actually claims DQL; later siblings detect the
// existing claim and skip the double-claim. Returns dql=nullptr if DQL is claimed by a different
// fusion type (the fusion cannot share DQL across heterogeneous fusion types).
DqlLookupResult FindParentDql(
    const OrtNodeUnit& integer_op,
    const OrtNodeUnitIODef& a_q_input,
    std::string_view fusion_type_tag,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map);

// True if every consumer of `value_info` is a SingleNode of `expected_op_type` and `value_info`
// is not itself a graph output. Used on DQL's a_q / a_zp outputs (with op_type = ConvInteger or
// MatMulInteger).
bool ConsumersAreAllOfType(
    const Ort::ConstValueInfo& value_info,
    std::string_view expected_op_type,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit);

// True if every consumer of `value_info` looks like a parallel_Mul: 2-input/1-output Mul
// SingleNode whose other input is a constant initializer. Used on DQL's a_scale output.
bool ConsumersAreAllParallelMuls(
    const Ort::ConstValueInfo& value_info,
    const QnnModelWrapper& qmw,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit);

// Builds the weight quant params (int8 or uint8) for an `out_channels`-wide weight. ONNX
// zero-point is negated to match QNN's offset convention (QNN: x = scale * (q - offset);
// offset = -ONNX_zp).
//
// `per_channel_axis` is the axis carried in the per-channel QnnQuantParamsWrapper. The
// per-channel emission path pre-dequantizes to float offline and does not consume these quant
// params, so the axis value is informational only (e.g. axis=0 for Conv [O,I,H,W] weights,
// axis=1 for MatMul [K,N] weights).
Ort::Status BuildWeightQuantParams(const QnnModelWrapper& qmw,
                                   const OrtNodeUnitIODef& b_scale_iodef,
                                   const OrtNodeUnitIODef* b_zp_iodef,
                                   uint32_t out_channels,
                                   int32_t per_channel_axis,
                                   QnnQuantParamsWrapper& out_params);

// Pre-dequantizes per-channel int8 / uint8 weight bytes to float32 bytes. Per-channel scales /
// zps are applied along the LAST axis of the byte buffer (used by both Conv's HWCN weight and
// MatMul's [K,N] weight, where the output-channel dimension is last in both cases).
Ort::Status PreDequantizePerChannelWeight(const QnnModelWrapper& qmw,
                                          const OrtNodeUnitIODef& b_scale_iodef,
                                          const OrtNodeUnitIODef* b_zp_iodef,
                                          bool is_signed_weight,
                                          uint32_t out_channels,
                                          const std::vector<uint8_t>& quant_bytes,
                                          std::vector<uint8_t>& out_float_bytes);

}  // namespace qnn
}  // namespace onnxruntime
