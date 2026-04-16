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

/// Fuses the pattern: Transpose -> Reshape -> Transpose into a single Reshape
/// when the combined transformation is equivalent to just reshaping (dimension merging only).
class TransposeReshapeTransposeFusion : public IQnnNodeGroup {
 public:
  TransposeReshapeTransposeFusion(gsl::span<const OrtNodeUnit* const> node_units,
                                  std::vector<int64_t> fused_output_shape)
      : fused_output_shape_(std::move(fused_output_shape)) {
    if (node_units.size() != 3) {
      ORT_CXX_API_THROW("TransposeReshapeTransposeFusion expects exactly 3 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = node_units[0];  // First Transpose
    node_units_[1] = node_units[1];  // Reshape
    node_units_[2] = node_units[2];  // Second Transpose
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TransposeReshapeTransposeFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "TransposeReshapeTransposeFusion"; }

  /// </summary>
  /// <param name="qnn_model_wrapper">The QNN model wrapper for graph access.</param>
  /// <param name="transpose1_node_unit">The first Transpose node unit (starting point).</param>
  /// <param name="node_to_node_unit">Maps Node* to NodeUnit*.</param>
  /// <param name="node_unit_to_qnn_node_group">Maps NodeUnit* to existing IQnnNodeGroup*.</param>
  /// <param name="logger">Logger for diagnostics.</param>
  /// <returns>A TransposeReshapeTransposeFusion if pattern matches, nullptr otherwise.</returns>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& transpose1_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;  // [Transpose1, Reshape, Transpose2]
  std::vector<int64_t> fused_output_shape_;       // Shape for the fused Reshape op
};

}  // namespace qnn
}  // namespace onnxruntime
