// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <gsl/gsl>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// The generic Rank6ToRank5Fusion already reduces these chains rank-6 -> rank-5 (the batch
// unit dim is dropped), but leaves a rank-5 Transpose over a large (~ws*ws*grid*C) tensor,
// which is a slow path on HTP. This fusion instead restructures the whole chain into rank-4
// ops (two rank-4 Transposes plus Reshapes), so it must be registered BEFORE
// Rank6ToRank5Fusion to claim these chains first.
//
// Partition (in_rank=4 -> mid rank-6 -> out_rank=3), keyed on the first Reshape:
//   x[B,H,W,C]
//     Reshape   -> [B, Gh, ws, Gw, ws, C]                    (rank-6)
//     Transpose -> [B, Gh, Gw, ws, ws, C]  perm=[0,1,3,2,4,5](rank-6)
//     Reshape   -> [B*Gh*Gw, ws*ws, C]                       (rank-3)
//   rewritten (H=Gh*ws, W=Gw*ws) as:
//     Reshape   -> [B, H, Gw, ws*C]
//     Transpose -> [B, Gw, H, ws*C]        perm=[0,2,1,3]
//     Reshape   -> [B, Gw, Gh, ws*ws*C]
//     Transpose -> [B, Gh, Gw, ws*ws*C]    perm=[0,2,1,3]
//     Reshape   -> [B*Gh*Gw, ws*ws, C]
//
// Reverse (in_rank=3 -> mid rank-6 -> out_rank=4), the exact inverse, keyed on the first
// Reshape:
//   x[B*Gh*Gw, ws*ws, C]
//     Reshape   -> [B, Gh, Gw, ws, ws, C]                    (rank-6)
//     Transpose -> [B, Gh, ws, Gw, ws, C]  perm=[0,1,3,2,4,5](rank-6)
//     Reshape   -> [B, H, W, C]                              (rank-4)
//   rewritten as:
//     Reshape   -> [B, Gh, Gw, ws*ws*C]
//     Transpose -> [B, Gw, Gh, ws*ws*C]    perm=[0,2,1,3]
//     Reshape   -> [B, Gw, H, ws*C]
//     Transpose -> [B, H, Gw, ws*C]        perm=[0,2,1,3]
//     Reshape   -> [B, H, W, C]
//
// The rewrite reuses the final output tensor name, so downstream consumers are undisturbed.
// It claims exactly the 3 chain nodes (Reshape, Transpose, Reshape). On non-NPU backends, or
// when the pattern does not match / fails QNN validation, TryFusion returns nullptr.
class WindowPartitionReverseFusion final : public IQnnNodeGroup {
 public:
  enum class Direction { kPartition,
                         kReverse };

  // Geometry resolved during matching and needed for emission.
  struct Params {
    Direction direction = Direction::kPartition;
    std::string input_name;   // first Reshape data input
    std::string output_name;  // final Reshape output (reused so downstream is undisturbed)
    uint32_t batch = 0;       // B
    uint32_t gh = 0;          // H / ws
    uint32_t gw = 0;          // W / ws
    uint32_t ws = 0;          // window size (assumed square)
    uint32_t channels = 0;    // C
  };

  WindowPartitionReverseFusion(gsl::span<const OrtNodeUnit* const> node_units, Params params);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(WindowPartitionReverseFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  // Serialized as a JSON key in the framework op trace (summary.fusion_count[<Type()>]).
  // Renaming is a breaking change for trace consumers.
  static constexpr std::string_view kType = "WindowPartitionReverseFusion";
  std::string_view Type() const override { return kType; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reshape_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;  // Reshape, Transpose, Reshape
  Params params_;
};

}  // namespace qnn
}  // namespace onnxruntime
