// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"

namespace onnxruntime::qnn {

// Recognized QDQ attention tail:
//
//   attention_mask -> Sub(1, mask) -> Mul(mask, -M) -----+
//   score ---------> Div ----------------------------------> Add -> Add(gate) -> Softmax
//
// The lowering keeps this topology and changes the static mask value to -100.
class AttnMaskValueReductionFusion final : public IQnnNodeGroup {
 public:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(AttnMaskValueReductionFusion);
  struct Match;
  explicit AttnMaskValueReductionFusion(std::unique_ptr<Match> match);
  ~AttnMaskValueReductionFusion();

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "AttnMaskValueReductionFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& softmax_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::unique_ptr<Match> match_;
};

}  // namespace onnxruntime::qnn
