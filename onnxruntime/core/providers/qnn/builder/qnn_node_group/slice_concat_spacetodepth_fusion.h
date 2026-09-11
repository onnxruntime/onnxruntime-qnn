// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// Strided Slice + Concat tiling (pixel-unshuffle by hand), either parallel
// (4 multi-axis Slices off one input) or cascaded (2 H-slices feeding 4
// W-slices), fused to QNN SpaceToDepth (DCR) + channel Gather. Canonical DCR
// phase order needs no Gather; any other phase order is restored with a Gather
// over the 4C channels. Bare S2D alone would silently reorder channels here.
class SliceConcatSpaceToDepthFusion : public IQnnNodeGroup {
 public:
  static constexpr uint32_t kBlockHeight = 2;
  static constexpr uint32_t kBlockWidth = 2;
  // Pattern size varies by tiling form: 5 units for parallel tiling
  // (4 Slices + Concat) up to 7 for cascaded tiling (6 Slices + Concat).
  // The Concat unit is always last, which makes it the sort target.
  static constexpr size_t kMinGroupSize = 5;

  SliceConcatSpaceToDepthFusion(gsl::span<const OrtNodeUnit* const> tiled_node_units,
                                const OrtNodeUnit& root_input_owner,
                                std::array<int64_t, 4> dcr_phase_permutation,
                                uint32_t input_channels)
      : root_input_owner_(&root_input_owner),
        phase_permutation_(dcr_phase_permutation),
        channel_count_(input_channels) {
    node_units_.reserve(tiled_node_units.size());
    for (const OrtNodeUnit* node_unit : tiled_node_units) {
      node_units_.push_back(node_unit);
    }
    concat_node_unit_ = node_units_.empty() ? nullptr : node_units_.back();
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SliceConcatSpaceToDepthFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return concat_node_unit_; }
  std::string_view Type() const override { return "SliceConcatSpaceToDepthFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& concat_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::vector<const OrtNodeUnit*> node_units_;
  const OrtNodeUnit* concat_node_unit_ = nullptr;
  // Slice whose input is the shared root tensor; node_units_ order is not meaningful.
  const OrtNodeUnit* root_input_owner_ = nullptr;
  // Canonical DCR phase index per Concat input; {0,1,2,3} needs no Gather.
  std::array<int64_t, 4> phase_permutation_{0, 1, 2, 3};
  uint32_t channel_count_ = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
