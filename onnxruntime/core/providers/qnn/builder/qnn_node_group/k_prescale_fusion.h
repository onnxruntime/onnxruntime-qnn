// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

// Q/DQ wrappers omitted:
//   X -> MatMul(static W) -> Reshape -> Transpose -> MatMul(Q, K) -> Div(c)
// Fold /c into W and K's encodings, then use Div's encoding for the QK output.
// Intermediate K tensors must be unshared; bias and cache branches are not folded.
class KPrescaleFusion : public IQnnNodeGroup {
 public:
  explicit KPrescaleFusion(std::array<const OrtNodeUnit*, 5> node_units) : node_units_(node_units) {}
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(KPrescaleFusion);

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override { return node_units_; }
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[3]; }
  std::string_view Type() const override { return "KPrescaleFusion"; }

 private:
  std::array<const OrtNodeUnit*, 5> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
