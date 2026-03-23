// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

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
/// Represents a fusion of a decomposed LayerNorm pattern into a single QNN LayerNorm operator.
///
/// Handles two patterns:
///
///   Pattern A (Textbook):
///
///                    +--------------------------------------------+
///                    |                                            |
///                    v                                            |
///   [x] --> ReduceMean --> Sub --> Pow(2) --> ReduceMean --> Add(eps) --> Sqrt --> Div --> Mul(gamma) --> Add(beta) ==>
///                                  |                                               ^
///                                  |                                               |
///                                  +-----------------------------------------------+
///
///   Pattern B (Transpose-wrapped):
///   Same as Pattern A but with a Transpose(perm=0,2,1) inserted between Sub and Pow.
///   The Transpose is absorbed by the fusion; QNN always sees axis=-1.
///
///                    +--------------------------------------------+
///                    |                                            |
///                    v                                            |
///   [x] --> ReduceMean --> Sub --> Transpose --> Pow(2) --> ReduceMean --> Add(eps) --> Sqrt --> Div --> Mul(gamma) --> Add(beta) ==>
///                                  |                                                              ^
///                                  |                                                              |
///                                  +--------------------------------------------------------------+
///
/// Both patterns are translated into a single QNN LayerNorm operator.
/// The contained NodeUnits must be of type SingleNode.
/// </summary>
class LayerNormFusion : public IQnnNodeGroup {
 public:
  LayerNormFusion(std::vector<const OrtNodeUnit*>&& node_units,
                  const OrtNodeUnit* target_node_unit,
                  float epsilon,
                  std::vector<uint32_t> axes,
                  bool has_transpose);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(LayerNormFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "LayerNormFusion"; }

  /// <summary>
  /// Traverses the graph to check if the given starting NodeUnit is part of a valid decomposed
  /// LayerNorm pattern (Pattern A or Pattern B). If so, returns an IQnnNodeGroup containing all
  /// the NodeUnits in the pattern.
  /// </summary>
  /// <param name="qnn_model_wrapper">Used for validation and graph traversal</param>
  /// <param name="reduce_mean_node_unit">ReduceMean NodeUnit that could start the sequence</param>
  /// <param name="node_to_node_unit">Maps a Node to a NodeUnit</param>
  /// <param name="node_unit_to_qnn_node_group">Maps a NodeUnit to a IQnnNodeGroup</param>
  /// <param name="logger"></param>
  /// <returns>A valid IQnnNodeGroup on success or an empty std::unique_ptr otherwise</returns>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reduce_mean_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::vector<const OrtNodeUnit*> node_units_;
  const OrtNodeUnit* target_node_unit_;
  float epsilon_;
  std::vector<uint32_t> axes_;  // QNN axes (normalized to last-dim for HTP)
  bool has_transpose_;
};

}  // namespace qnn
}  // namespace onnxruntime
