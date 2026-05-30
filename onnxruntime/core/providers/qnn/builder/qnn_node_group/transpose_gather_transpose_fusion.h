// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <gsl/gsl>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// Fuses Transpose -> Gather -> Transpose into a single Gather when the two
/// Transposes cancel out around the Gather.
///
/// Pattern:
///   x : rank N
///   t1 = Transpose(x, perm=P1)               // rank N
///   g  = Gather(t1, indices, axis=A)         // rank N - 1 + K, K = indices rank
///   y  = Transpose(g, perm=P2)               // rank N - 1 + K
///
/// Fused:
///   y  = Gather(x, indices, axis=P1[A])
///
/// The fusion is valid iff P2 is the unique permutation such that applying T2
/// to (T1 then Gather) reproduces gathering directly on x at axis P1[A] -- i.e.,
/// the data axes are returned to their original order and the indices block is
/// kept contiguous in its original order.
class TransposeGatherTransposeFusion : public IQnnNodeGroup {
 public:
  TransposeGatherTransposeFusion(gsl::span<const OrtNodeUnit* const> node_units,
                                 int32_t fused_axis)
      : fused_axis_(fused_axis) {
    if (node_units.size() != 3) {
      ORT_CXX_API_THROW("TransposeGatherTransposeFusion expects exactly 3 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = node_units[0];  // Transpose1
    node_units_[1] = node_units[1];  // Gather
    node_units_[2] = node_units[2];  // Transpose2
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TransposeGatherTransposeFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[1]; }
  std::string_view Type() const override { return "TransposeGatherTransposeFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& transpose1_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;  // [Transpose1, Gather, Transpose2]
  int32_t fused_axis_;                            // Axis on the Gather input (= P1[A])
};

}  // namespace qnn
}  // namespace onnxruntime
