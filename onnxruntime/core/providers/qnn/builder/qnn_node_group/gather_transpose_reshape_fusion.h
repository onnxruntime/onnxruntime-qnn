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

/// @brief Describes how the Gather indices are structured.
enum class GatherIndicesPattern {
  kRowMajor,  ///< indices[i,j] = i*cols + j  (C-order split of dim d4 into [idx0, idx1])
  kColMajor,  ///< indices[i,j] = j*rows + i  (Fortran-order split, equivalent to reshape then swap)
};

/// Fuses the pattern: Gather(axis=4) -> Transpose -> Reshape
/// ORIGINAL                              FUSED
/// ─────────────────────────────────     ──────────────────────────────────────────
/// in: [d0, d1, d2, d3, d4]  rank-5     in: [d0, d1, d2, d3, d4]  rank-5
///          │                                        │
///     Gather(axis=4)                           Reshape
///     indices[idx0,idx1]                  (merge d0,d1,d2 + split d4)
///          │                                        │
/// [d0,d1,d2,d3,idx0,idx1]  rank-6 ❌   [d0*d1*d2, d3, idx0, idx1]   rank-4 ✅
///          │                                        │
///     Transpose(perm)                    (col-major only) Transpose([0,1,3,2])
///          │                                        │  swap idx1,idx0 → idx0,idx1
/// [d0,d1,d2,?,?,?]         rank-6 ❌   [d0*d1*d2, d3, idx0, idx1]   rank-4 ✅
///          │                                        │
///       Reshape                               Transpose(new_perm_4d)
///          │                                        │
///   out: [...]                    rank<6 ✅   [?, ?, ?, ?]          rank-4 ✅
///                                                   │
///                                                Reshape
///                                                   │
///                                       out:      [...]  rank<6 ✅  (same as original)

class GatherTransposeReshapeFusion : public IQnnNodeGroup {
 public:
  explicit GatherTransposeReshapeFusion(gsl::span<const OrtNodeUnit* const> node_units,
                                        GatherIndicesPattern index_pattern,
                                        int64_t idx0,
                                        int64_t idx1,
                                        std::vector<uint32_t> new_perm_4d)
      : index_pattern_(index_pattern),
        idx0_(idx0),
        idx1_(idx1),
        new_perm_4d_(std::move(new_perm_4d)) {
    if (node_units.size() != 3) {
      ORT_CXX_API_THROW("GatherTransposeReshapeFusion expects exactly 3 NodeUnits.", ORT_EP_FAIL);
    }
    node_units_[0] = node_units[0];  // Gather
    node_units_[1] = node_units[1];  // Transpose
    node_units_[2] = node_units[2];  // Reshape
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(GatherTransposeReshapeFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "GatherTransposeReshapeFusion"; }

  /// Traverses the graph starting from a Gather NodeUnit to check whether it is the
  /// head of a valid Gather -> Transpose -> Reshape pattern with rank-6 intermediate
  /// tensors. Returns a GatherTransposeReshapeFusion if the pattern matches, or nullptr.
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& gather_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;  ///< [Gather, Transpose, Reshape]
  GatherIndicesPattern index_pattern_;            ///< Row-major or col-major indices
  int64_t idx0_;                                  ///< First dimension of the indices tensor
  int64_t idx1_;                                  ///< Second dimension of the indices tensor
  std::vector<uint32_t> new_perm_4d_;             ///< Adjusted 4D permutation for the fused Transpose
};

}  // namespace qnn
}  // namespace onnxruntime
