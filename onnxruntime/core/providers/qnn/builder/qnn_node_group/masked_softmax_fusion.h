// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"

namespace onnxruntime::qnn {

struct MaskedSoftmaxPatternAMatch {
  const OrtNodeUnit* softmax = nullptr;
  const OrtNodeUnit* gated_add = nullptr;
  const OrtNodeUnit* masked_add = nullptr;
  const OrtNodeUnit* score = nullptr;
  const OrtNodeUnit* additive_mask = nullptr;
  const OrtNodeUnit* mask_sub = nullptr;
  const OrtNodeUnitIODef* attention_mask = nullptr;
  const OrtNodeUnit* gate = nullptr;
  bool claims_legacy_mask_chain = false;
};

// Lowers the legacy PSL attention tail to an HTP-compatible form.
//
// Original PSL graph:
//   Div(score) + Mul(mask_bias) -> Add(old_masked)
//   Add(old_masked) + Mul_4(gate) -> Add_3 -> Softmax
//
// Emitted masked-softmax graph:
//                              +-> ReduceMin(axis=3, keepdims=1) -> Add(fill, -25.0)
//                              |                                      |
//   Div(score) + Mul_4(gate) -> Add(gated, uint16) ---------------------+
//                              |                                         |
//   attention_mask (nonzero=valid) -> NotEqual(mask, 0) -> cond
//       -> Select(cond, gated, fill) -> Softmax(axis=3)
//
// All data-path temporaries are uint16 affine tensors. Where input order is
// fixed as (condition, kept_gated_value, masked_fill_value).
//
// The old mask-bias Mul, inverse-mask Sub, mask Add, gated Add and Softmax
// NodeUnits are claimed by this group. The legacy mask path is therefore not
// emitted independently; the original mask is used directly with NotEqual.
// Select input order is intentionally (condition, kept_gated_value,
// masked_fill_value) for the HTP masked-softmax pattern recognizer.
class MaskedSoftmaxPatternAFusion final : public IQnnNodeGroup {
 public:
  explicit MaskedSoftmaxPatternAFusion(const MaskedSoftmaxPatternAMatch& match);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(MaskedSoftmaxPatternAFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[1]; }
  std::string_view Type() const override { return "MaskedSoftmaxPatternAFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& softmax_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  MaskedSoftmaxPatternAMatch match_;
  // old mask Add, old gated Add, old Softmax, plus the legacy mask-bias Mul
  // and inverse-mask Sub that are replaced by the direct-mask path.
  std::vector<const OrtNodeUnit*> node_units_;
};

}  // namespace onnxruntime::qnn
