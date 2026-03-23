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
/// Represents a fusion of either:
/// 1) Reshape -> Transpose -> Reshape
/// 2) Transpose -> (Reshape -> Transpose -> Reshape)
/// 3) (Reshape -> Transpose -> Reshape) -> Transpose
/// 4) Transpose -> (Reshape -> Transpose -> Reshape) -> Transpose
/// that can be replaced by SpaceToDepth.
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
    if (node_units.size() < 3 || node_units.size() > 5) {
      ORT_CXX_API_THROW("S2D Pattern expects 3 to 5 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_.reserve(node_units.size());
    for (const OrtNodeUnit* node_unit : node_units) {
      node_units_.push_back(node_unit);
    }

    // Target remains the first Reshape node regardless of optional wrap transposes.
    const bool has_head_transpose = node_units_.size() >= 4 &&
                                    node_units_[0] != nullptr &&
                                    node_units_[0]->OpType() == "Transpose";
    target_node_unit_ = has_head_transpose ? node_units_[1] : node_units_[0];
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReshapeTransposeReshapeSpaceToDepthFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return target_node_unit_; }
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
  std::vector<const OrtNodeUnit*> node_units_;  // 3-node core pattern or 5-node wrapped pattern
  const OrtNodeUnit* target_node_unit_ = nullptr;
  uint32_t block_height_ = 0;
  uint32_t block_width_ = 0;
  uint32_t mode_ = 0;
  bool use_nhwc_fallback_ = false;
};

}  // namespace qnn
}  // namespace onnxruntime
