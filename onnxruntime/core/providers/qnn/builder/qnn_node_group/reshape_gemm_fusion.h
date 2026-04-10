// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of a Reshape->Gemm sequence to a single Gemm node.
/// Ideally Reshape->Gemm->Reshape should be fused to a single Gemm node with keep_dims set to True,
/// but on some devices the OpConfig validation will fail when keep_dims to True (it says expected value is 0),
/// so we still need to keep the 2nd Reshape node.
/// </summary>
class ReshapeGemmFusion : public IQnnNodeGroup {
 public:
  ReshapeGemmFusion(const OrtNodeUnit& reshape_node_unit, const OrtNodeUnit& gemm_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReshapeGemmFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "ReshapeGemmFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 2> node_units_;
};

/// <summary>
/// Represents a fusion of a Reshape->Gemm->Reshape sequence.
/// Fuses to FullyConnected + Reshape (FC takes ND input, outputs 2D, then Reshape restores ND).
/// </summary>
class ReshapeGemmReshapeFusion : public IQnnNodeGroup {
 public:
  ReshapeGemmReshapeFusion(const OrtNodeUnit& input_reshape_node_unit,
                           const OrtNodeUnit& gemm_node_unit,
                           const OrtNodeUnit& output_reshape_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReshapeGemmReshapeFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "ReshapeGemmReshapeFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;
};

/// <summary>
/// Represents a fusion of Reshape->Gemm->Reshape->Reshape into FC->Reshape.
/// Pattern: ND input -> Reshape0 (ND->2D) -> Gemm (2D->2D) -> Reshape1 -> Reshape2 -> output
/// Fused:   ND input -> FullyConnected (ND->2D) -> Reshape2 -> output
/// All 4 nodes are claimed.
/// </summary>
class ReshapeGemmReshapeReshapeFusion : public IQnnNodeGroup {
 public:
  ReshapeGemmReshapeReshapeFusion(const OrtNodeUnit& input_reshape_node_unit,
                                  const OrtNodeUnit& gemm_node_unit,
                                  const OrtNodeUnit& output_reshape1_node_unit,
                                  const OrtNodeUnit& output_reshape2_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReshapeGemmReshapeReshapeFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "ReshapeGemmReshapeReshapeFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 4> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
