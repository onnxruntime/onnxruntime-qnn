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

// Fuses Reshape -> [Q -> DQ]? -> Transpose when the Reshape is structurally equivalent
// to a Transpose (input and output have the same rank; non-1 dims appear in the same
// relative order; size-1 dims may move freely). Composes the derived Reshape-perm with
// the Transpose's perm, then emits ONE QNN node:
//
//   * composed perm is identity -> single noop Reshape (input/output shapes coincide)
//   * otherwise                 -> single Transpose(perm = fused) from the Reshape's
//                                  outer input tensor to the Transpose's outer output
//                                  tensor
//
// The upstream ORT ReshapeTransposeFusion pass in core/optimizer/ handles the same
// pattern on the pre-partition graph, but for QNN EP models the offending pairs are
// commonly introduced INSIDE the QNN partition by the layout transformer (Conv
// NCHW->NHWC) during partitioning. Level-2 fusion can no longer see them, so we handle
// the same rewrite here inside qnn_node_group, where visibility is restricted to what
// QNN EP will actually compile.
class ReshapeTransposeFusion : public IQnnNodeGroup {
 public:
  ReshapeTransposeFusion(const OrtNodeUnit& reshape_node_unit,
                         const OrtNodeUnit& transpose_node_unit,
                         std::vector<int64_t> fused_perm)
      : node_units_{&reshape_node_unit, &transpose_node_unit},
        fused_perm_(std::move(fused_perm)) {}
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReshapeTransposeFusion);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[0]; }
  std::string_view Type() const override { return "ReshapeTransposeFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reshape_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 2> node_units_;  // {Reshape, Transpose}
  std::vector<int64_t> fused_perm_;               // reshape_perm[transpose_perm[i]] for each i
};

}  // namespace qnn
}  // namespace onnxruntime
