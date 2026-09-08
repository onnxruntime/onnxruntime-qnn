// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

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
/// Fuses the tanh-based GELU approximation into a single QNN Gelu op.
///
/// Pattern (entry point is Tanh; ORT lowers Pow(x,3) to Mul(Mul(x,x),x) before QNN EP sees the graph):
///
///        +-> Mul(x,x) -> Mul(x²,x) -> Mul(0.044715) --+
///        |                   ^                          v
///   [x] -+-------------------+------------------------> Add --> Mul(sqrt(2/pi)) --> Tanh --> Add(1) --> Mul --> Mul(0.5) ==>
///        |                                                                                              ^
///        +----------------------------------------------------------------------------------------------+
///
/// Equation: x * 0.5 * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// All contained NodeUnits must be SingleNode (non-QDQ). QDQ support is out of scope for this
/// PR: fusing across QDQ boundaries requires DQ/Q unwrap logic (see GeluFusion) that adds
/// significant complexity. Quantized tanh-GELU will silently fall back to individual ops.
/// </summary>
class TanhGeluFusion : public IQnnNodeGroup {
 public:
  TanhGeluFusion(std::vector<const OrtNodeUnit*>&& node_units,
                 const OrtNodeUnit* target_node_unit,
                 OrtNodeUnitIODef gelu_root_input,
                 OrtNodeUnitIODef gelu_final_output);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TanhGeluFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "TanhGeluFusion"; }

  /// <summary>
  /// Tries to match the tanh-GELU pattern starting from the given Tanh NodeUnit.
  /// Returns a TanhGeluFusion on success, nullptr otherwise.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& tanh_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::vector<const OrtNodeUnit*> node_units_;
  const OrtNodeUnit* target_node_unit_;
  OrtNodeUnitIODef gelu_root_input_;
  OrtNodeUnitIODef gelu_final_output_;
};

}  // namespace qnn
}  // namespace onnxruntime
