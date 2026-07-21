// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
/// into a single identity Reshape (input shape == output shape).
///
/// Pattern:  t0 -> Reshape -> t1 -> Transpose(perm) -> t2
///
/// Conditions for the fusion to fire:
///   1. Shape(t0) == Shape(t2)  (same shape before Reshape and after Transpose).
///   2. The Transpose is memory-order-preserving relative to t1:
///      the non-unit axes of t1 appear in the same relative order in the Transpose output.
///      i.e., for the axes j of t1 where t1[j] > 1, taken in ascending j, the positions
///      k such that perm[k] == j must be strictly increasing.
///
/// When both hold, the Reshape and Transpose together leave the underlying memory buffer
/// unchanged (they are collectively a no-op). We collapse them into a single Reshape whose
/// input and output shapes are equal, which QNN backends treat as pure metadata (no data
/// movement) — matching the existing convention in TransposeReshapeTransposeFusion of
/// emitting a Reshape when the combined layout-op sequence reduces to a pure reshape.
///
/// Motivation: for inputs whose channel dim is 1 (e.g. grayscale [1,H,W,1]), ORT layout
/// passes can insert this pair when adapting between ONNX Conv (NCHW) and HTP native
/// (NHWC) conventions; the ops become identity but are still executed as an 8-9 MB
/// memory shuffle per inference.
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
