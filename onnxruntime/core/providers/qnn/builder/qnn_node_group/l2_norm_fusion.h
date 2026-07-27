// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
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
/// Fuses the L2-normalize pattern  Div(x, Add(ReduceL2(x), eps))  =>  QNN_OP_L2_NORM(x, axis, epsilon).
///
/// QNN has no single ReduceL2 op, so the ReduceL2 op builder lowers it to a
/// ElementWiseMultiply(x*x) -> ReduceSum -> Sqrt sequence. In fp16 the ReduceSum of squares
/// overflows once the L2 norm exceeds the fp16 max -- e.g. reducing many channels of moderately
/// large activations -- producing inf/NaN and corrupting the normalize. HTP's native
/// L2Norm op computes the same normalize without this overflow. This fusion detects the
/// ReduceL2 -> Add(eps) -> Div subgraph and emits QNN_OP_L2_NORM so the normalize runs on that
/// fp16-safe kernel instead of the overflow-prone decomposition
/// </summary>
class L2NormFusion : public IQnnNodeGroup {
 public:
  L2NormFusion(const OrtNodeUnit& reducel2_node_unit,
               const OrtNodeUnit& add_node_unit,
               const OrtNodeUnit& div_node_unit)
      : node_units_{&reducel2_node_unit, &add_node_unit, &div_node_unit} {
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(L2NormFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_units_[2]; }
  std::string_view Type() const override { return "L2NormFusion"; }

  /// <summary>
  /// Traverses the graph from a starting ReduceL2 NodeUnit to check for a valid
  /// ReduceL2 -> Add(eps) -> Div(x, .) normalize sequence. Returns a IQnnNodeGroup if matched.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reducel2_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 3> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
