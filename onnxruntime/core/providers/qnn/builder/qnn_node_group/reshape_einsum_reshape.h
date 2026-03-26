// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>
#include <unordered_map>

#include <gsl/span>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// Define a specific replacement for Reshape->Einsum->Reshape pattern and transform into a functionally equivalent
// sequence of Reshape, DepthToSpace, and Transpose. The pattern is expected as:
//   - Input: [NxHxW, BxBxC]
//   - Reshape: [N, H, W, B, B, C]
//   - Einsum (nhwpqc->nchpwq): [N, C, H, B, W, B]
//     - Equivalent to Transpose.
//   - Reshape: [N, C, HxB, WxB]
// Since there are unsupported 6D shapes in the above pattern, it will replaced by:
//   - Input: [NxHxW, BxBxC]
//   - Reshape: [N, H, W, BxBxC]
//   - DepthToSpace (DCR, block size B): [N, HxB, WxB, C]
//   - Transpose: [N, C, HxB, WxB]
class ReshapeEinsumReshapeNodeGroup : public IQnnNodeGroup {
 public:
  ReshapeEinsumReshapeNodeGroup(const OrtNodeUnit* pre_reshape_node_unit,
                                const OrtNodeUnit* einsum_node_unit,
                                const OrtNodeUnit* post_reshape_node_unit)
      : node_units_{pre_reshape_node_unit, einsum_node_unit, post_reshape_node_unit} {}
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReshapeEinsumReshapeNodeGroup);

  Ort::Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;

  Ort::Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const override;

  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override { return node_units_; }

  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[1]; }

  std::string_view Type() const override { return "ReshapeEinsumReshapeNodeGroup"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& einsum_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
