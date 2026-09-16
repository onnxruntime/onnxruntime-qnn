// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <gsl/gsl>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// Moves a Swin attention output projection after window reverse, optional cyclic unshift,
// and depadding. The group also emits the window reverse as rank-4 operations.
//
//   Cast -> MatMul -> Cast -> Add(bias) -> window_reverse -> [roll] -> depad -> Reshape
// becomes
//   rank4_window_reverse -> [roll] -> depad -> Cast -> FC -> Cast -> Add(bias) -> Reshape
//
// This is valid because all operations crossed by the projection only permute or discard
// complete tokens and leave the channel axis unchanged. Running after depadding avoids
// projecting padded tokens and removes the format-conversion island before window reverse.
class SwinAttentionProjectionFusion final : public IQnnNodeGroup {
 public:
  struct Params {
    std::string chain_input_name;
    std::string reverse_output_name;
    std::string depad_output_name;
    std::string final_output_name;
    const OrtNodeUnitIODef* chain_input_def = nullptr;
    const OrtNodeUnitIODef* weight_def = nullptr;
    const OrtNodeUnitIODef* bias_def = nullptr;
    const OrtNodeUnit* reverse_anchor = nullptr;
    Qnn_DataType_t matmul_dtype{};
    Qnn_DataType_t output_dtype{};
    uint32_t batch = 0;
    uint32_t gh = 0;
    uint32_t gw = 0;
    uint32_t window_size = 0;
    uint32_t channels = 0;
    std::vector<uint32_t> depad_shape;
    std::vector<uint32_t> final_shape;
    // Original Slice/Concat nodes after window reverse, in topological order. Re-emitted
    // through their normal op builders after the rank-4 reverse writes reverse_output_name.
    std::vector<const OrtNodeUnit*> post_reverse_nodes;
  };

  SwinAttentionProjectionFusion(gsl::span<const OrtNodeUnit* const> node_units, Params params);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SwinAttentionProjectionFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  static constexpr std::string_view kType = "SwinAttentionProjectionFusion";
  std::string_view Type() const override { return kType; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& matmul_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::vector<const OrtNodeUnit*> node_units_;
  Params params_;
};

}  // namespace qnn
}  // namespace onnxruntime
