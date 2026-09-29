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
/// Represents a fusion of the Gelu pattern expanded into ONNX operators.
/// This fusion handles four pattern variants categorized by Erf's child node type:
///
/// ErfAdd Patterns (Erf -> Add):
///   Pattern 1:
///                +-------Mul(0.5)---------------------+
///                |                                    |
///                |                                    v
///             [root] --> Div -----> Erf  --> Add --> Mul ==>
///                       (B=1.4142...)        (1)
///
///   Pattern 2:
///                +------------------------------------+
///                |                                    |
///                |                                    v
///             [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
///                       (B=1.4142...)        (1)  (root)   (0.5)
///
///   Pattern 4:
///                +--------------------------------------------+
///                |                                            |
///                |                                            v
///             [root] --> Div -----> Erf  --> Add --> Mul --> Mul ==>
///                       (B=1.4142...)        (1)    (0.5)  (root)
///
/// ErfMul Pattern (Erf -> Mul):
///   Pattern 3:
///                +-------------------------------------------+
///                |                                           |
///                |                                           v
///             [root] --> Div -----> Erf --> Mul --> Add --> Mul ==>
///                       (B=1.4142...)      (0.5)   (0.5)
///
/// All patterns are translated into a QNN Gelu operator.
/// The contained NodeUnits can be of type SingleNode or QDQGroup (with Q-DQ nodes).
/// The second inputs to Div, Add, and Mul Node Units should be constant.
/// </summary>
class GeluFusion : public IQnnNodeGroup {
 public:
  /// <param name="node_units">All NodeUnits in the GELU pattern (Div/Mul, Erf, Add, Mul, etc.)</param>
  /// <param name="target_node_unit">The Erf NodeUnit (used as the primary node for this fusion)</param>
  /// <param name="validation_root_input">GELU pattern root input for QNN validation (quantized if outer DQ exists, else float)</param>
  /// <param name="validation_final_output">GELU pattern final output for QNN validation (quantized if outer Q exists, else float)</param>
  /// <param name="gelu_root_input">Root input to fused Gelu operator (float, from DQ output or pattern root)</param>
  /// <param name="gelu_final_output">Final output from fused Gelu operator (float, before Q input or pattern end)</param>
  GeluFusion(std::vector<const OrtNodeUnit*>&& node_units,
             const OrtNodeUnit* target_node_unit,
             OrtNodeUnitIODef validation_root_input,
             OrtNodeUnitIODef validation_final_output,
             OrtNodeUnitIODef gelu_root_input,
             OrtNodeUnitIODef gelu_final_output);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(GeluFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "GeluFusion"; }

  /// <summary>
  /// Traverses graph to check if the given starting NodeUnit is part of a valid Gelu pattern.
  /// If so, returns a IQnnNodeGroup that contains all the NodeUnits in the pattern.
  /// </summary>
  /// <param name="qnn_model_wrapper">Used for validation and traverse/query the graph</param>
  /// <param name="erf_node_unit">Erf node unit that could be part of the sequence</param>
  /// <param name="node_to_node_unit">Maps a Node to a NodeUnit.</param>
  /// <param name="node_unit_to_qnn_node_group">Maps a NodeUnit to a IQnnNodeGroup.</param>
  /// <param name="logger"></param>
  /// <returns>A valid IQnnNodeGroup on success or an empty std::unique_ptr otherwise</returns>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& erf_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::vector<const OrtNodeUnit*> node_units_;
  const OrtNodeUnit* target_node_unit_;
  OrtNodeUnitIODef validation_root_input_;
  OrtNodeUnitIODef validation_final_output_;
  OrtNodeUnitIODef gelu_root_input_;
  OrtNodeUnitIODef gelu_final_output_;
};

}  // namespace qnn
}  // namespace onnxruntime
