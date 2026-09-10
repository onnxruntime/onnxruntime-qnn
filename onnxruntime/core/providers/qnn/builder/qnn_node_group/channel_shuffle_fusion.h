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
/// Represents a fusion of pattern:  Transpose -> ChannelShuffle (Reshape -> Transpose -> Reshape) -> Transpose
/// or the no-boundary-transpose variant: Reshape -> Transpose -> Reshape -> Transpose (when ORT's
/// TransposeOptimizer absorbed the leading Transpose into the Reshape shape).
/// </summary>
class ChannelShuffleFusion : public IQnnNodeGroup {
 public:
  // Full 5-node pattern: T_head, Reshape1, T_mid, Reshape2, T_tail
  explicit ChannelShuffleFusion(gsl::span<const OrtNodeUnit* const> node_units) {
    if (node_units.size() != 5) {
      ORT_CXX_API_THROW("Pattern expect exactly 5 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = node_units[0];
    node_units_[1] = node_units[1];
    node_units_[2] = node_units[2];
    node_units_[3] = node_units[3];
    node_units_[4] = node_units[4];
    has_head_transpose_ = true;
  }
  // 4-node pattern (no head transpose): Reshape1, T_mid, Reshape2, T_tail
  explicit ChannelShuffleFusion(gsl::span<const OrtNodeUnit* const> node_units, bool /*no_head_transpose_tag*/) {
    if (node_units.size() != 4) {
      ORT_CXX_API_THROW("No-head-transpose pattern expects exactly 4 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = nullptr;  // no T_head
    node_units_[1] = node_units[0];  // Reshape1
    node_units_[2] = node_units[1];  // T_mid
    node_units_[3] = node_units[2];  // Reshape2
    node_units_[4] = node_units[3];  // T_tail
    has_head_transpose_ = false;
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ChannelShuffleFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return has_head_transpose_ ? node_units_[0] : node_units_[1]; }
  std::string_view Type() const override { return "ChannelShuffleFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit (Transpose) is part of a channel shuffle pattern.
  /// Pattern: Transpose -> Reshape -> Transpose -> Reshape -> Transpose
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& transpose_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit (Reshape) is part of a channel shuffle pattern.
  /// Pattern: Reshape -> Transpose -> Reshape -> Transpose (bare variant without leading Transpose)
  /// Used when ORT's TransposeOptimizer absorbed the leading Transpose into the Reshape shape.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusionFromReshape(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reshape_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 5> node_units_;
  bool has_head_transpose_ = true;
};

}  // namespace qnn
}  // namespace onnxruntime
