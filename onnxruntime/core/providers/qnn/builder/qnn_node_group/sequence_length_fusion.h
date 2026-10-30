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
///   SequenceConstruct(t0..tN-1) -> seq -> SequenceLength(seq) -> int64 scalar N
/// into a single QNN_OP_RESHAPE that moves a compile-time int64 constant to the
/// graph output (shape {1}, not {} — QNN HTP rejects rank-0 output tensors).
///
/// Rationale: The QNN EP type system is tensor-only; a standalone SequenceLength
/// op builder cannot be registered because the sequence-typed input edge cannot
/// be expressed as a QnnTensor. When the upstream is SequenceConstruct, N is
/// fully determined at compile time from the input count, so the fusion emits
/// only a static constant and a trivial Reshape — no element-tensor processing.
/// ORT's constant-fold pass does not traverse sequence-typed edges, so this
/// pattern reaches the EP unfolded.
///
/// Constraints (TryFusion returns nullptr if any is violated):
///   - The sequence output of SequenceConstruct must have exactly one
///     consumer, which must be SequenceLength.
///   - The sequence output must not itself be a graph output.
///   - SequenceLength must be a SingleNode (not QDQ-wrapped).
///   - N >= 1.
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
