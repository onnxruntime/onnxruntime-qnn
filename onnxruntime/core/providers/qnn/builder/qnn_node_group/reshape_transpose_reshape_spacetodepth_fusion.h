// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <gsl/gsl>
#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of pattern: Reshape -> Transpose -> Reshape that can be replaced by SpaceToDepth.
/// </summary>
class ReshapeTransposeReshapeSpaceToDepthFusion : public IQnnNodeGroup {
 public:
  ReshapeTransposeReshapeSpaceToDepthFusion(gsl::span<const OrtNodeUnit* const> node_units,
                                            uint32_t block_height,
                                            uint32_t block_width,
                                            uint32_t mode,
                                            bool use_nhwc_fallback)
      : block_height_(block_height),
        block_width_(block_width),
        mode_(mode),
        use_nhwc_fallback_(use_nhwc_fallback) {
    if (node_units.size() != 3) {
      ORT_CXX_API_THROW("Pattern expects exactly 3 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = node_units[0];
    node_units_[1] = node_units[1];
    node_units_[2] = node_units[2];
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReshapeTransposeReshapeSpaceToDepthFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "ReshapeTransposeReshapeSpaceToDepthFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a Reshape -> Transpose -> Reshape
  /// pattern that can be replaced by SpaceToDepth. Returns a fusion if the pattern matches, or nullptr.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reshape_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;  // Reshape1, Transpose, Reshape2
  uint32_t block_height_ = 0;
  uint32_t block_width_ = 0;
  uint32_t mode_ = 0;
  bool use_nhwc_fallback_ = false;
};

}  // namespace qnn
}  // namespace onnxruntime
