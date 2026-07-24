// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <gsl/gsl>
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
/// Fuses a [Reshape -> Transpose] pair whose combined effect is a data-preserving identity
/// into a single Reshape with matching input/output shape.
///
/// Pattern:  t0 -> Reshape -> t1 -> Transpose(perm) -> t2
/// Conditions:
///   - Shape(t0) == Shape(t2)
///   - Transpose preserves memory order of t1's non-unit axes.
///
/// Scope (direction): Reshape->Transpose only. The mirror direction is out of scope;
/// open a follow-up if a real-world scenario materialises.
///
/// Scope (QDQ): All NodeUnits must be of type SingleNode. A QDQ-wrapped chain
/// (DQ->Reshape->Q->DQ->Transpose->Q) is out of scope because unequal scales across the
/// pair would drop a rescale if collapsed to an identity Reshape.
/// </summary>
class IdentityReshapeTransposeFusion : public IQnnNodeGroup {
 public:
  explicit IdentityReshapeTransposeFusion(gsl::span<const OrtNodeUnit* const> node_units) {
    if (node_units.size() != 2) {
      ORT_CXX_API_THROW("Pattern expects exactly 2 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = node_units[0];
    node_units_[1] = node_units[1];
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(IdentityReshapeTransposeFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "IdentityReshapeTransposeFusion"; }

  /// <summary>
  /// Attempts to match the pattern starting from a Reshape NodeUnit.
  /// Returns a fusion object on success, nullptr otherwise.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reshape_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 2> node_units_;  // [Reshape, Transpose]
};

}  // namespace qnn
}  // namespace onnxruntime
