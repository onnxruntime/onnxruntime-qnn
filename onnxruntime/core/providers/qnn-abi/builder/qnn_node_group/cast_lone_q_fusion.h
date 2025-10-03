// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a fusion of a Cast -> Q sequence that converts from one data type to a quantized type.
/// This is translated into a QNN Convert operator. The Cast and Q are standalone NodeUnits that are not
/// part of a QDQ node unit. The Cast input must not come from a DQ node.
/// Pattern: Non-DQ Node -> Cast -> Q => Non-DQ Node -> Convert
/// </summary>
class CastLoneQFusion : public IQnnNodeGroup {
 public:
  CastLoneQFusion(const OrtNodeUnit& cast_node_unit, const OrtNodeUnit& q_node_unit);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(CastLoneQFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "CastLoneQFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid Cast -> Q sequence.
  /// If so, returns a IQnnNodeGroup that contains the Cast and Q NodeUnits.
  /// </summary>
  /// <param name="qnn_model_wrapper">Used for validation and traverse/query the graph</param>
  /// <param name="cast_node_unit">Cast node unit that could start the Cast -> Q sequence</param>
  /// <param name="node_to_node_unit">Maps a Node to a NodeUnit.</param>
  /// <param name="node_unit_to_qnn_node_group">Maps a NodeUnit to a IQnnNodeGroup.</param>
  /// <param name="logger"></param>
  /// <returns>A valid IQnnNodeGroup on success or an empty std::unique_ptr otherwise</returns>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& cast_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 2> node_units_;
};

}  // namespace qnn
}  // namespace onnxruntime
