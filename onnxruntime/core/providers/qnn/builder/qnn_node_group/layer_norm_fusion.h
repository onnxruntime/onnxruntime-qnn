// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Fuses a decomposed LayerNorm pattern into a single QNN LayerNorm operator.
///
///                    +--------------------------------------------+
///                    |                                            |
///                    v                                            |
///   [x] --> ReduceMean --> Sub --> Pow(2) --> ReduceMean --> Add(eps) --> Sqrt --> Div --> Mul(gamma) --> Add(beta) ==>
///                                  |                                               ^
///                                  |                                               |
///                                  +-----------------------------------------------+
///
/// The affine tail is optional. Tried in order of preference:
///   - Full:       ... --> Div --> Mul(gamma) --> Add(beta)  ==>  LayerNorm(x, gamma, beta)
///   - Gamma only: ... --> Div --> Mul(gamma)                ==>  LayerNorm(x, gamma, zeros)
///   - Core only:  ... --> Div                               ==>  LayerNorm(x, ones,  zeros)
/// Core-only fusion covers the case where the normalized tensor feeds several consumers (e.g. it is
/// shared by multiple heads that each apply their own gamma/beta) or where gamma/beta have a shape
/// QNN LayerNorm cannot express. Ops after the fused part are left untouched.
///
/// Scalar constants (Pow exponent, epsilon) may be float32 or float16 initializers.
///
/// QNN HTP LayerNorm normalizes only the last axis. A single trailing axis (axes=[-1]) maps directly
/// to one LayerNorm. Several trailing axes are flattened into one:
///   x [..., d1, d2] --> Reshape [P, d1*d2] --> LayerNorm(axis 1) --> Reshape [..., d1, d2]
/// with gamma/beta collapsed to {d1*d2}.
/// All NodeUnits must be of type SingleNode.
/// </summary>
class LayerNormFusion : public IQnnNodeGroup {
 public:
  /// An empty gamma_input_name / beta_input_name means a ones / zeros tensor is synthesized.
  /// The fused output is node_units.back()->Outputs()[0].
  LayerNormFusion(std::vector<const OrtNodeUnit*>&& node_units,
                  const OrtNodeUnit* target_node_unit,
                  float epsilon,
                  std::vector<uint32_t> axes,
                  std::string gamma_input_name,
                  std::string beta_input_name);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(LayerNormFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "LayerNormFusion"; }

  // One entry point per affine-tail variant. They are registered longest-first in the fusion table
  // (qnn_node_group.cc), so the first variant that matches and validates on QNN wins.
  static std::unique_ptr<IQnnNodeGroup> TryFusionFull(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reduce_mean_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);
  static std::unique_ptr<IQnnNodeGroup> TryFusionGammaOnly(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reduce_mean_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);
  static std::unique_ptr<IQnnNodeGroup> TryFusionCoreOnly(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reduce_mean_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  enum class Tail { kFull,
                    kGammaOnly,
                    kCoreOnly };

  // Matches the decomposed core plus, for kFull / kGammaOnly, the required affine tail; returns
  // nullptr if `tail` is not present or the resulting LayerNorm does not validate on QNN.
  static std::unique_ptr<IQnnNodeGroup> TryFusionImpl(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reduce_mean_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger,
      Tail tail);

  std::vector<const OrtNodeUnit*> node_units_;
  const OrtNodeUnit* target_node_unit_;
  float epsilon_;
  std::vector<uint32_t> axes_;
  std::string gamma_input_name_;
  std::string beta_input_name_;
};

}  // namespace qnn
}  // namespace onnxruntime
