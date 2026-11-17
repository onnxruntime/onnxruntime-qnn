// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>
#include <string_view>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Fuses the two-node pattern
///   SequenceConstruct(t₀..tₙ₋₁) → seq → SequenceLength(seq) → int64 scalar N
/// into a QNN subgraph that produces the compile-time-known constant N while
/// satisfying QNN HTP's requirement that every registered input tensor flows
/// unbroken to a graph output.
///
/// Rationale: QNN's type system is tensor-only; a standalone SequenceLength op
/// builder cannot be registered because the sequence-typed input edge cannot be
/// expressed as a QnnTensor. When the upstream is SequenceConstruct, N is fully
/// determined at compile time from the input count. ORT's constant-fold pass
/// does not traverse sequence-typed edges, so this pattern reaches the EP
/// unfolded. HTP requires every APP_WRITE input to be consumed by at least one
/// op that flows to a graph output (orphaned tensors fail at graphExecute with
/// QNN_COMMON_ERROR_GENERAL Code 1100), so element tensors must be wired
/// through the graph even though their values are irrelevant.
///
/// Emitted QNN subgraph (per-element ReduceSum + scalar chain; 2N+3 nodes total):
///   ReduceSum(elem_i, all axes, keepdims=true) → sum_kd_i   (NATIVE, {1,...,1}, per element)
///   EWAdd(sum_kd_{i-1}, sum_kd_i) → chain_i   (NATIVE, {1,...,1}, N-1 ops; absent if N=1)
///   Reshape(accum, {1})            → float_sum (NATIVE, {1})
///   EWMul(float_sum, STATIC_ZERO)  → zero_float (NATIVE, {1}, ≈ 0.0f)
///   Cast(zero_float, float→int64)  → zero_int64 (NATIVE, {1}, = 0)
///   EWAdd(zero_int64, STATIC_N)    → length     (APP_READ or NATIVE, {1}, = N)
///
/// Output is always registered as shape {1}. ONNX declares the output as scalar
/// {} but QNN HTP rejects rank-0 output tensors (error 3110). The byte count is
/// identical (8 bytes for int64) so ORT's buffer copy is correct either way.
///
/// Constraints (TryFusion returns nullptr if any is violated):
///   - SequenceConstruct must have ≥ 1 input (N ≥ 1).
///   - All element inputs must share the same fully-static shape.
///   - Element rank must be ≥ 1 (scalar {} elements → CPU fallback).
///   - All element dimensions must be positive (zero-dim → CPU fallback).
///   - Element count must fit in uint32_t.
///   - Element dtype must be FP32 or FP16 (other types → CPU fallback).
///     Quantized types are also rejected because the zero-point encoding
///     cannot be represented by raw zero bytes.
///   - The sequence output of SequenceConstruct must not be a graph output.
///   - The sequence output must have exactly one consumer, which must be
///     SequenceLength.
///   - SequenceLength must be a SingleNode (not QDQ-wrapped).
///
/// Known limitation: element values containing NaN or ∞ propagate through the
/// zero-multiply chain (0 × NaN = NaN under IEEE 754) and may produce undefined
/// int64 output via Cast. QNN HTP typically operates in FP16 space internally
/// and may flush such values, but the behavior is device-dependent.
/// </summary>
class SequenceLengthFusion : public IQnnNodeGroup {
 public:
  SequenceLengthFusion(const OrtNodeUnit& seq_construct_unit,
                       const OrtNodeUnit& seq_length_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SequenceLengthFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "SequenceLengthFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& seq_construct_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 2> node_units_;  // [0] = SequenceConstruct, [1] = SequenceLength
};

}  // namespace qnn
}  // namespace onnxruntime
