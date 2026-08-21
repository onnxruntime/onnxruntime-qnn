// Copyright (c) Qualcomm Technologies, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of the RMSNorm pattern:
///   Mul(x,x) -> ReduceMean -> Add(epsilon) -> Sqrt -> Div(x, sqrt) -> Mul(gamma) [-> Add(beta)]
///
/// This is translated into a single QNN RmsNorm operator (QNN_OP_RMS_NORM).
/// The pattern corresponds to: y = (x / sqrt(mean(x^2) + epsilon)) * gamma + beta
/// </summary>
class RmsNormFusion : public IQnnNodeGroup {
 public:
  RmsNormFusion(std::vector<const NodeUnit*> node_units);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(RmsNormFusion);

  Status IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  Status AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const override;
  gsl::span<const NodeUnit* const> GetNodeUnits() const override;
  const NodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "RmsNormFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting Mul NodeUnit is part of a valid
  /// RMSNorm pattern. The entry point is a Mul where both inputs are the same tensor (x * x).
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const NodeUnit& mul_node_unit,
      const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
      const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const logging::Logger& logger);

 private:
  std::vector<const NodeUnit*> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
