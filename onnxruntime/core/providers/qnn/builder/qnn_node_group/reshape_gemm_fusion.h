// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Unified fusion class for Reshape-Gemm patterns:
/// - 2-node: Reshape -> Gemm
/// - 3-node: Reshape -> Gemm -> Reshape
/// - 4-node: Reshape -> Gemm -> Reshape -> Reshape
///
/// All patterns fuse to QNN FullyConnected (+ optional output Reshape).
/// </summary>
class ReshapeGemmFusionGroup : public IQnnNodeGroup {
 public:
  explicit ReshapeGemmFusionGroup(std::vector<const OrtNodeUnit*> node_units) noexcept;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReshapeGemmFusionGroup);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "ReshapeGemmFusionGroup"; }

  // 2-node fusion: Reshape -> Gemm
  static std::unique_ptr<IQnnNodeGroup> TryFusion2(
      QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

  // 3-node fusion: Reshape -> Gemm -> Reshape
  static std::unique_ptr<IQnnNodeGroup> TryFusion3(
      QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

  // 4-node fusion: Reshape -> Gemm -> Reshape -> Reshape
  static std::unique_ptr<IQnnNodeGroup> TryFusion4(
      QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw, const Ort::Logger& logger, bool validate) const;

  std::vector<const OrtNodeUnit*> node_units_;
};

// Backward-compatible aliases for existing registration code
using ReshapeGemmFusion = ReshapeGemmFusionGroup;
using ReshapeGemmReshapeFusion = ReshapeGemmFusionGroup;
using ReshapeGemmReshapeReshapeFusion = ReshapeGemmFusionGroup;

}  // namespace qnn
}  // namespace onnxruntime
