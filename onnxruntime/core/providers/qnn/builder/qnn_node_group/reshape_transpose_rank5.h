// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
/// How a rank-6 intermediate is reduced to rank 5.
/// </summary>
enum class RankReduction {
  Squeeze,       ///< Drop the unit dim at index, which the transpose leaves in place.
  MergeLeading,  ///< Merge input dims 0 and 1, which must travel together.
};

/// <summary>
/// Represents a fusion of pattern: Reshape -> Transpose -> Reshape where intermediate tensors are rank-6.
/// QNN doesn't support rank-6 Reshape and Transpose operators, so this fusion converts them to rank-5
/// by dropping one dimension: a unit dimension the transpose leaves in place, or - when no such
/// dimension exists - by merging input dims 0 and 1, which must travel together
/// (perm[0] == 0 && perm[1] == 1).
///
/// Both reductions are pure view operations - an axis removal, or a merge of two adjacent axes
/// that the permutation keeps adjacent - so they preserve element order exactly. Validity
/// therefore depends only on which axes survive, never on the permutation's semantics. A perm
/// that happens to be a SpaceToDepth decomposition is not a reason to bail out here.
///
/// Priority over SpaceToDepthFusion is provided by fusions-table order (SpaceToDepthFusion is
/// tried first in qnn_node_group.cc), not by a list of permutations in this file. When
/// SpaceToDepthFusion declines a pattern -- it refuses bare RTR before Layout Transform, see
/// spacetodepth_fusion.cc - this fallback claims it rather than leaving it on the CPU.
///
/// Pattern: Tensor(t0) -> Reshape(R1) -> Tensor(t1) -> Transpose(T1) -> Tensor(t2) -> Reshape(R2) -> Tensor(t3)
/// Conditions: Rank(t1) == Rank(t2) == 6 with constant Reshape shapes, and either
/// a fixed unit dim (t1[i] == t2[i] == 1, perm[i] == i) or perm[0] == 0 && perm[1] == 1.
/// </summary>
class Rank6ToRank5Fusion : public IQnnNodeGroup {
 public:
  explicit Rank6ToRank5Fusion(gsl::span<const OrtNodeUnit* const> node_units,
                              RankReduction kind,
                              size_t index)
      : kind_(kind), index_(index) {
    if (node_units.size() != 3) {
      ORT_CXX_API_THROW("Pattern expects exactly 3 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = node_units[0];
    node_units_[1] = node_units[1];
    node_units_[2] = node_units[2];
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Rank6ToRank5Fusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "Rank6ToRank5Fusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid Reshape -> Transpose -> Reshape
  /// pattern with rank-6 intermediate tensors.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reshape1_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;  // Reshape1, Transpose, Reshape2
  RankReduction kind_;
  size_t index_;  // Squeeze: unit dim to drop. MergeLeading: unused, always 0.
};

}  // namespace qnn
}  // namespace onnxruntime
