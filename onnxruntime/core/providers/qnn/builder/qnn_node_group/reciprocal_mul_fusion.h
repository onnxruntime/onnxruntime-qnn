// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// =============================================================================
// ReciprocalMulFusion -- header
// =============================================================================
//
// Declares the IQnnNodeGroup subclass that fuses the two-node ONNX sub-graph
//
//   [denominator] --> Reciprocal --+
//                                  v
//   [numerator]  ----------------> Mul --> [output]
//
// into a single QNN ElementWiseDivide node:
//
//   [numerator]   --> ElementWiseDivide --> [output]
//   [denominator] --+
//
// See reciprocal_mul_fusion.cc for the full implementation and design notes.
// =============================================================================

#pragma once

#include <array>
#include <memory>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Fuses a Reciprocal -> Mul sub-graph into a single QNN ElementWiseDivide node.
///
/// Background
/// ----------
/// The QNN HTP/DSP backend does not expose a native Reciprocal operator.
/// Attempting to lower a standalone Reciprocal node causes the QNN EP to fall
/// back to CPU execution for that sub-graph, which defeats the purpose of
/// running on the accelerator.  The mathematical identity
///
///   Mul(a, Reciprocal(b))  ==  Div(a, b)
///
/// lets us replace the unsupported pair with a single, natively-supported
/// ElementWiseDivide node, keeping the entire computation on the accelerator.
///
/// Matched ONNX patterns
/// ---------------------
/// FP32 / FP16 (SingleNode):
///
///   [denominator] --> Reciprocal --+
///                                  v
///   [numerator]  ----------------> Mul --> [output]
///
/// Quantized (QDQGroup):
///
///   [denominator] --> DQ --> Reciprocal --> Q --+
///                                               v
///   [numerator]  --> DQ -----------------------> Mul --> Q --> [output]
///
/// Emitted QNN graph (both cases)
/// --------------------------------
///
///   [numerator]   --> ElementWiseDivide --> [output]
///   [denominator] --+
///
/// The intermediate tensor(s) produced by Reciprocal (and the surrounding
/// Q/DQ nodes for quantized models) are never registered in the QNN graph;
/// they are completely absorbed by the fusion.
///
/// Constraints
/// -----------
///   - The Reciprocal NodeUnit may be of type SingleNode or QDQGroup.
///   - The Reciprocal output must have exactly one consumer (the Mul node).
///   - The Reciprocal output must not be a graph-level output.
///   - The Mul NodeUnit must not already belong to another IQnnNodeGroup.
///   - The Mul must have exactly 2 inputs, one of which is the Reciprocal
///     output (or its downstream DQ output for QDQ groups).  The other input
///     becomes the numerator of the Div.
///   - The fused ElementWiseDivide node must pass QNN capability validation.
/// </summary>
class ReciprocalMulFusion : public IQnnNodeGroup {
 public:
  /// Constructs the fusion from the two already-validated NodeUnits.
  /// Callers should use TryFusion() rather than constructing directly.
  ReciprocalMulFusion(const OrtNodeUnit& reciprocal_node_unit, const OrtNodeUnit& mul_node_unit,
                      bool recip_is_mul_input0);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReciprocalMulFusion);

  // -- IQnnNodeGroup interface -----------------------------------------------

  /// Performs a dry-run QNN capability check without modifying the model.
  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;

  /// Registers tensors and creates the fused ElementWiseDivide QNN node.
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;

  /// Returns the two NodeUnits owned by this fusion: [Reciprocal, Mul].
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;

  /// Returns the Mul NodeUnit as the topological target.
  ///
  /// The Mul is the convergence point where both the numerator path and the
  /// Reciprocal path meet, making it the correct target for topological
  /// ordering of IQnnNodeGroups (see IQnnNodeGroup::GetTargetNodeUnit()).
  const OrtNodeUnit* GetTargetNodeUnit() const override;

  std::string_view Type() const override { return "ReciprocalMulFusion"; }

  // -- Factory ---------------------------------------------------------------

  /// <summary>
  /// Attempts to match the Reciprocal -> Mul pattern starting at
  /// <paramref name="reciprocal_node_unit"/>.
  ///
  /// Returns a fully constructed ReciprocalMulFusion on success, or
  /// nullptr if the pattern does not match or QNN validation fails.
  /// </summary>
  /// <param name="qnn_model_wrapper">Graph wrapper used for traversal and QNN validation.</param>
  /// <param name="reciprocal_node_unit">Candidate entry node (must be Reciprocal).</param>
  /// <param name="node_to_node_unit">Maps every OrtNode* to its owning OrtNodeUnit*.</param>
  /// <param name="node_unit_to_qnn_node_group">
  ///   Maps every OrtNodeUnit* that has already been claimed by an IQnnNodeGroup.
  ///   Used to prevent double-claiming nodes.
  /// </param>
  /// <param name="logger">Logger for diagnostic messages.</param>
  /// <returns>Unique pointer to the fusion, or nullptr.</returns>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reciprocal_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  // Stores pointers to the two constituent NodeUnits in graph order:
  //   [0] = Reciprocal  (producer of the intermediate 1/x tensor)
  //   [1] = Mul         (consumer; becomes the fused Div node)
  std::array<const OrtNodeUnit*, 2> node_units_;

  // True  => the Reciprocal output feeds Mul input[0]  (recip_out is numerator slot)
  // False => the Reciprocal output feeds Mul input[1]  (recip_out is denominator slot)
  //
  // Resolved once in TryFusion (Step 3) and cached here so that
  // CreateOrValidateOnQnn can consume it directly without repeating the
  // Q -> DQ graph traversal that was already performed during fusion matching.
  bool recip_is_mul_input0_{false};
};

}  // namespace qnn
}  // namespace onnxruntime
