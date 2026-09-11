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

// Rewrites the rank-5 "packed-QKV split" of a decomposed attention block into rank-<=4
// QNN ops for the NPU (HTP/DSP) backend. rank-5 tensors are supported on HTP but perform
// poorly; this fusion removes them from the hot attention path.
//
// Recognized ONNX pattern (Swin / GroundingDINO style), keyed on the head Reshape:
//
//   qkv[N,S,3*n*hs]
//     - Reshape -> [N,S,3,n,hs]              (rank-5, slow on HTP)
//     - Transpose(perm=[2,0,3,1,4]) -> [3,N,n,S,hs]   (rank-5, slow on HTP)
//     - Gather(axis=0, idx=0) = Q -> [N,n,S,hs]
//     - Gather(axis=0, idx=1) = K -> [N,n,S,hs]
//     - Gather(axis=0, idx=2) = V -> [N,n,S,hs]
//   ...downstream SDPA math (Mul/Transpose/MatMul/Add/Softmax/MatMul) is left untouched.
//
// This node group claims ONLY the 5 "split" nodes (head Reshape, head Transpose, and the
// three Gathers). It re-emits them, per branch, as:
//
//   StridedSlice(qkv, last-axis block) -> [N,S,n*hs]   (rank-3)
//     -> Reshape -> [N,S,n,hs]                          (rank-4)
//     -> Transpose(perm=[0,2,1,3]) -> [N,n,S,hs]        (rank-4)  == original Gather output
//
// The three emitted branch outputs reuse the original Gather output tensor names, so every
// downstream consumer keeps working with no further changes. The remaining attention nodes
// are matched only to confirm the block really is packed-QKV attention; they are not
// claimed and continue through their normal per-op lowering.
//
// On non-NPU backends, or when the pattern does not match, TryFusion returns nullptr and
// the individual nodes fall back to their default per-op lowering.
class QkvSplitAttentionFusion final : public IQnnNodeGroup {
 public:
  // Parameters resolved during matching and needed for emission.
  struct SplitParams {
    // Packed-QKV producer tensor (head Reshape's data input) and its shape [N, S, 3*n*hs].
    std::string qkv_input_name;
    uint32_t n_rows = 0;     // N   (product of leading/batch dims, i.e. num windows * batch)
    uint32_t seq = 0;        // S   (tokens per row)
    uint32_t num_heads = 0;  // n
    uint32_t head_size = 0;  // hs
    // Original Gather output tensor names, indexed by QKV role (0=Q, 1=K, 2=V). Emission
    // reuses these names so downstream consumers are undisturbed.
    std::array<std::string, 3> gather_out_names;
  };

  QkvSplitAttentionFusion(gsl::span<const OrtNodeUnit* const> claimed_node_units, SplitParams params);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QkvSplitAttentionFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  // Target node is the head Reshape (first claimed node), the entry point of the split.
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  // Serialized as a JSON key in the framework op trace (summary.fusion_count[<Type()>]).
  // Renaming is a breaking change for trace consumers.
  static constexpr std::string_view kType = "QkvSplitAttentionFusion";
  std::string_view Type() const override { return kType; }

  // Traverses the graph starting from the head Reshape NodeUnit. Returns a
  // QkvSplitAttentionFusion (claiming the 5 split nodes) if a valid packed-QKV attention
  // block is found on an NPU backend and the rewrite validates on QNN, otherwise nullptr.
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reshape_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  // The 5 claimed NodeUnits: [head Reshape, head Transpose, Gather_Q, Gather_K, Gather_V].
  std::array<const OrtNodeUnit*, 5> node_units_;
  SplitParams params_;
};

}  // namespace qnn
}  // namespace onnxruntime
