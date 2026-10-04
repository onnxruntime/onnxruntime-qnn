// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Fuses the two-node pattern
///   SequenceConstruct(t0..tN-1) -> seq -> SequenceAt(seq, pos) -> out
/// into a sequence of QNN primitives that consume only tensor-typed edges:
///   Reshape x N  →  Concat(axis=0)  →  Gather(axis=0)
///
/// Rationale: The QNN EP type system is tensor-only; a standalone SequenceAt
/// op builder cannot be registered because the sequence-typed input edge
/// cannot be expressed as a QnnTensor. The fusion avoids the sequence edge
/// entirely by absorbing SequenceConstruct and SequenceAt together.
///
/// Constraints (TryFusion returns nullptr if any is violated):
///   - All N inputs of SequenceConstruct must be plain tensors with identical
///     static shapes and an element type representable in QNN (no string /
///     complex64 / complex128).
///   - The sequence output of SequenceConstruct must have exactly one
///     consumer, which must be SequenceAt.
///   - The position input of SequenceAt must be a scalar of dtype int32 or
///     int64.
///   - SequenceAt must be a SingleNode (not QDQ-wrapped).
/// </summary>
class SequenceAtFusion : public IQnnNodeGroup {
 public:
  SequenceAtFusion(const OrtNodeUnit& seq_construct_unit,
                   const OrtNodeUnit& seq_at_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SequenceAtFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "SequenceAtFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& seq_construct_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 2> node_units_;  // [0] = SequenceConstruct, [1] = SequenceAt
};

}  // namespace qnn
}  // namespace onnxruntime
