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

/// <summary>
/// Represents a fusion of pattern: Reshape -> Transpose -> Reshape where intermediate tensors are rank-5.
/// QNN HTP rejects some rank-5 Transpose perms; this fusion converts them to rank-4 by merging an
/// adjacent pair of input dims that the Transpose moves together (i.e., their perm values are
/// consecutive and stored at consecutive perm positions).
/// Pattern: Tensor(t0) -> Reshape(R1) -> Tensor(t1) -> Transpose(T1) -> Tensor(t2) -> Reshape(R2) -> Tensor(t3)
/// Conditions:
/// - Rank(t1) == Rank(t2) == 5
/// - There exists a position `p` in the rank-5 perm where perm[p+1] == perm[p] + 1
///   (i.e., two adjacent input dims are moved as a contiguous block by the Transpose).
/// Example: [1, 133, 133, 128] -> Reshape([19, 7, 19, 7, 128]) -> Transpose([0, 2, 1, 3, 4]) -> Reshape
///   becomes [1, 133, 133, 128] -> Reshape([19, 7, 19, 896]) -> Transpose([0, 2, 1, 3]) -> Reshape.
/// </summary>
class Rank5ToRank4Fusion : public IQnnNodeGroup {
 public:
  explicit Rank5ToRank4Fusion(gsl::span<const OrtNodeUnit* const> node_units,
                              size_t merge_perm_index)
      : merge_perm_index_(merge_perm_index) {
    if (node_units.size() != 3) {
      ORT_CXX_API_THROW("Pattern expects exactly 3 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = node_units[0];
    node_units_[1] = node_units[1];
    node_units_[2] = node_units[2];
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Rank5ToRank4Fusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "Rank5ToRank4Fusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid Reshape -> Transpose -> Reshape
  /// pattern with rank-5 intermediate tensors that can be reduced to rank-4.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reshape1_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;  // Reshape1, Transpose, Reshape2
  size_t merge_perm_index_;                       // Position p in the rank-5 perm such that perm[p+1] == perm[p] + 1
};

}  // namespace qnn
}  // namespace onnxruntime
