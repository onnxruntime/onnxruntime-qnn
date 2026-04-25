// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// =============================================================================
// ReciprocalMulFusion
// =============================================================================
//
// Fuses the two-node ONNX sub-graph
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
// Motivation
// ----------
// The QNN HTP/DSP backend does not expose a native Reciprocal operator.
// Attempting to lower a standalone Reciprocal node causes the QNN EP to fall
// back to CPU execution for that sub-graph, which defeats the purpose of
// running on the accelerator.  The mathematical identity
//
//   Mul(a, Reciprocal(b))  ==  Div(a, b)
//
// lets us replace the unsupported pair with a single, natively-supported
// ElementWiseDivide node, keeping the entire computation on the accelerator.
//
// The intermediate tensor produced by Reciprocal (the "1/b" value) is never
// registered in the QNN graph; it is completely absorbed by the fusion.
//
// Tensor role mapping
// -------------------
//   ONNX input  : denominator  (Reciprocal's input)
//   ONNX input  : numerator    (the other Mul input)
//   ONNX output : result       (Mul's output, unchanged)
//
//   QNN Div input[0]  = numerator
//   QNN Div input[1]  = denominator
//   QNN Div output[0] = result
//
// =============================================================================

#include "core/providers/qnn/builder/qnn_node_group/reciprocal_mul_fusion.h"

#include <array>
#include <cassert>
#include <gsl/gsl>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// =============================================================================
// File-local helpers
// =============================================================================

// Convenience macros that forward to the shared CreateOrValidateOnQnn helper
// with the `validate` flag pre-set.  This mirrors the pattern used throughout
// the qnn_node_group folder (e.g. gelu_fusion.cc, hardsigmoid_mul_fusion.cc).
//
//   validate=true  => dry-run capability check; does NOT modify the model wrapper.
//   validate=false => build path; registers tensors and creates the QNN node.
#define ValidateOnQnn(qnn_model_wrapper, reciprocal_node_unit, mul_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (reciprocal_node_unit), (mul_node_unit), /*validate=*/true)
#define CreateOnQnn(qnn_model_wrapper, reciprocal_node_unit, mul_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (reciprocal_node_unit), (mul_node_unit), /*validate=*/false)

// Forward declaration so the macros above can reference the function before
// its full definition appears at the bottom of this translation unit.
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& reciprocal_node_unit,
                                         const OrtNodeUnit& mul_node_unit,
                                         bool validate);

// =============================================================================
// ReciprocalMulFusion::TryFusion
// =============================================================================
//
// Entry point called by the graph-traversal loop in qnn_node_group.cc for
// every NodeUnit whose op-type is "Reciprocal".
//
// The function walks the graph in a strictly forward (producer -> consumer)
// direction:
//
//   1. Verify the entry node is a standalone Reciprocal (not inside a QDQ
//      group, which would be handled by a different fusion path).
//   2. Confirm the Reciprocal has exactly one consumer and that consumer is
//      a standalone Mul node that has not already been claimed.
//   3. Confirm the Mul actually consumes the Reciprocal output (sanity check
//      against malformed graphs where GetOnlyChildOfType might return a Mul
//      that is connected via a different edge).
//   4. Perform a QNN dry-run validation to ensure the backend can handle the
//      resulting ElementWiseDivide node.
//   5. Construct and return the ReciprocalMulFusion object.
//
std::unique_ptr<IQnnNodeGroup> ReciprocalMulFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reciprocal_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // -- Step 1: Gate on op-type and node-unit kind ---------------------------
  //
  // Only fuse standalone (SingleNode) Reciprocal units.  A Reciprocal that
  // is already wrapped inside a QDQ group (DQ -> Reciprocal -> Q) is handled
  // by a separate quantization-aware path and must not be touched here.
  if (reciprocal_node_unit.OpType() != "Reciprocal" ||
      reciprocal_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // -- Step 2: Reciprocal must have at least one input ----------------------
  //
  // ONNX Reciprocal is a unary op (output = 1 / input).  Guard against a
  // malformed graph that somehow has no inputs.
  const auto& recip_inputs = reciprocal_node_unit.Inputs();
  if (recip_inputs.empty()) {
    return nullptr;
  }

  // -- Step 3: Locate the single Mul consumer of the Reciprocal output ------
  //
  // GetOnlyChildOfType performs all of the following checks atomically:
  //   (a) The Reciprocal node has exactly one output tensor.
  //   (b) That output tensor is NOT a graph-level output (i.e. it is an
  //       internal intermediate value that can be safely removed).
  //   (c) The output tensor has exactly one consumer node.
  //   (d) That consumer is a SingleNode whose op-type is "Mul".
  //   (e) The Mul NodeUnit has not already been claimed by another
  //       IQnnNodeGroup (prevents double-fusion).
  //
  // If any condition fails, nullptr is returned and we bail out.
  const std::array<std::string_view, 1> child_op_types{"Mul"};
  const OrtNodeUnit* mul_node_unit =
      GetOnlyChildOfType(qnn_model_wrapper, reciprocal_node_unit, child_op_types,
                         node_to_node_unit, node_unit_to_qnn_node_group);
  if (mul_node_unit == nullptr) {
    return nullptr;
  }

  // -- Step 4: Mul must have exactly 2 inputs --------------------------------
  //
  // ONNX Mul is a binary op.  One input must be the Reciprocal output
  // (the denominator path); the other is the numerator.
  const auto& mul_inputs = mul_node_unit->Inputs();
  if (mul_inputs.size() < 2) {
    return nullptr;
  }

  // -- Step 5: Verify the Reciprocal output is actually wired into the Mul --
  //
  // GetOnlyChildOfType guarantees the Mul is the sole consumer of the
  // Reciprocal output, but it does not verify *which* input slot of the Mul
  // carries that value.  We do that here as a defence-in-depth check.
  //
  // ONNX Mul is commutative, so the Reciprocal result may appear in either
  // input[0] or input[1].
  const auto& recip_outputs = reciprocal_node_unit.Outputs();
  if (recip_outputs.empty()) {
    return nullptr;
  }

  const std::string& recip_output_name = recip_outputs[0].name;
  const bool recip_is_mul_input0 = (mul_inputs[0].name == recip_output_name);
  const bool recip_is_mul_input1 = (mul_inputs[1].name == recip_output_name);

  if (!recip_is_mul_input0 && !recip_is_mul_input1) {
    // The Mul does not actually consume the Reciprocal output.  This can
    // happen if the graph is malformed or if GetOnlyChildOfType returned a
    // Mul that is connected via a different edge.  Bail out safely.
    return nullptr;
  }

  // -- Step 6: QNN capability dry-run ----------------------------------------
  //
  // Ask the QNN backend whether it can handle an ElementWiseDivide node
  // with the tensor types and shapes inferred from the ONNX graph.  This
  // call does NOT modify the QnnModelWrapper's internal state; it is a
  // pure read-only capability query.
  //
  // If the backend rejects the node (e.g. unsupported data type or rank),
  // we return nullptr so the two nodes fall back to individual handling.
  if (Ort::Status status = ValidateOnQnn(qnn_model_wrapper, reciprocal_node_unit, *mul_node_unit);
      !status.IsOK()) {
    return nullptr;
  }

  // -- Step 7: Commit to the fusion ------------------------------------------
  //
  // All checks passed.  Construct the fusion object.  The actual QNN node
  // will be created later when AddToModelBuilder() is called.
  return std::make_unique<ReciprocalMulFusion>(reciprocal_node_unit, *mul_node_unit);
}

// =============================================================================
// ReciprocalMulFusion constructor
// =============================================================================

ReciprocalMulFusion::ReciprocalMulFusion(const OrtNodeUnit& reciprocal_node_unit,
                                         const OrtNodeUnit& mul_node_unit)
    : node_units_{&reciprocal_node_unit, &mul_node_unit} {
}

// =============================================================================
// IQnnNodeGroup interface
// =============================================================================

// IsSupported
// -----------
// Called during the graph partitioning phase to determine whether this fusion
// can be offloaded to QNN.  Delegates to the shared validate path which
// performs a QNN dry-run without modifying the model wrapper.
Ort::Status ReciprocalMulFusion::IsSupported(QnnModelWrapper& qmw,
                                             const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

// AddToModelBuilder
// -----------------
// Called during the model-building phase to register tensors and emit the
// fused QNN ElementWiseDivide node into the QNN graph.
Ort::Status ReciprocalMulFusion::AddToModelBuilder(QnnModelWrapper& qmw,
                                                   const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

// GetNodeUnits
// ------------
// Returns the two NodeUnits owned by this fusion in graph order:
//   [0] Reciprocal  -- the producer of the intermediate 1/x tensor
//   [1] Mul         -- the consumer; becomes the fused Div node
gsl::span<const OrtNodeUnit* const> ReciprocalMulFusion::GetNodeUnits() const {
  return node_units_;
}

// GetTargetNodeUnit
// -----------------
// Returns the Mul NodeUnit as the topological "target" of this fusion.
//
// The target is defined as the first node where ALL input paths of the
// IQnnNodeGroup converge (see IQnnNodeGroup::GetTargetNodeUnit() docs).
// In this fusion:
//
//   [denominator] --> Reciprocal --+
//                                  v
//   [numerator]  ----------------> Mul  <-- convergence point
//
// Both the numerator path and the Reciprocal path converge at the Mul node,
// making it the correct target for topological ordering of IQnnNodeGroups.
const OrtNodeUnit* ReciprocalMulFusion::GetTargetNodeUnit() const {
  return node_units_[1];  // Mul is the convergence point
}

// =============================================================================
// CreateOrValidateOnQnn
// =============================================================================
//
// Shared implementation for both the dry-run (validate=true) and build
// (validate=false) paths.
//
// Mathematical mapping
// --------------------
//   ONNX:  output = Mul(numerator, Reciprocal(denominator))
//   QNN:   output = ElementWiseDivide(numerator, denominator)
//
// Tensor roles
// ------------
//   input[0]  = numerator   -- the Mul input that is NOT the Reciprocal output
//   input[1]  = denominator -- the Reciprocal's single input
//   output[0] = result      -- the Mul's output (unchanged by the fusion)
//
// The intermediate tensor produced by Reciprocal ("recip_output") is
// intentionally NOT registered in the QNN graph; it is absorbed by the fusion.
//
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& reciprocal_node_unit,
                                         const OrtNodeUnit& mul_node_unit,
                                         bool validate) {
  assert(reciprocal_node_unit.OpType() == "Reciprocal");
  assert(mul_node_unit.OpType() == "Mul");

  // -- Resolve tensor roles --------------------------------------------------
  //
  // denominator: the single input fed into Reciprocal (the value being
  //              inverted).  This becomes input[1] of the Div node.
  const OrtNodeUnitIODef& denominator_def = reciprocal_node_unit.Inputs()[0];

  // Identify which Mul input slot carries the Reciprocal output so we can
  // determine the numerator slot.  ONNX Mul is commutative, so either slot
  // is valid.
  const std::string& recip_output_name = reciprocal_node_unit.Outputs()[0].name;
  const auto& mul_inputs = mul_node_unit.Inputs();
  const bool recip_is_input0 = (mul_inputs[0].name == recip_output_name);

  // numerator: whichever Mul input is NOT the Reciprocal output.
  //            This becomes input[0] of the Div node.
  const OrtNodeUnitIODef& numerator_def = recip_is_input0 ? mul_inputs[1] : mul_inputs[0];

  // result: the Mul's output tensor becomes the Div output unchanged.
  const OrtNodeUnitIODef& output_def = mul_node_unit.Outputs()[0];

  // -- Build QNN tensor descriptors ------------------------------------------
  //
  // MakeTensorWrapper reads the tensor's shape, element data-type, and
  // quantisation parameters from the ONNX graph and produces a
  // Qnn_Tensor_t descriptor that can be passed to the QNN API.
  QnnTensorWrapper numerator_tensor;
  QnnTensorWrapper denominator_tensor;
  QnnTensorWrapper output_tensor;

  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(numerator_def, numerator_tensor));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(denominator_def, denominator_tensor));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  // Use the Reciprocal node's unique name as the fused node name.  This
  // keeps the QNN graph node name stable and traceable back to the original
  // ONNX graph for debugging and profiling purposes.
  const std::string node_name = utils::UniqueNameGenerator().New(reciprocal_node_unit);

  if (validate) {
    // -- Dry-run: capability query only ---------------------------------------
    //
    // ValidateQnnNode queries the QNN backend for support without touching
    // the model wrapper's internal tensor/node tables.  A failure here means
    // the backend cannot handle this Div configuration (e.g. unsupported
    // data type or tensor rank), so we return the error to the caller which
    // will then fall back to individual node handling.
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(
        node_name,
        QNN_OP_PACKAGE_NAME_QTI_AISW,
        QNN_OP_ELEMENT_WISE_DIVIDE,
        /*input_tensors=*/{numerator_tensor.GetQnnTensor(), denominator_tensor.GetQnnTensor()},
        /*output_tensors=*/{output_tensor.GetQnnTensor()},
        /*params=*/{}));
  } else {
    // -- Build path: register tensors, then create the QNN node ---------------
    //
    // Tensor registration policy
    // --------------------------
    // Graph inputs and initializers may already be registered by an earlier
    // node that shares the same tensor.  IsQnnTensorWrapperExist() guards
    // against double-registration, which would corrupt the internal tables.
    //
    // The intermediate Reciprocal output tensor (recip_output_name) is
    // intentionally NEVER registered here.  It does not exist in the QNN
    // graph; the fusion replaces it with a direct edge from the denominator
    // to the Div node.

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(numerator_def.name)) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(numerator_tensor)),
                    "ReciprocalMulFusion: failed to add numerator tensor wrapper.");
    }

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(denominator_def.name)) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(denominator_tensor)),
                    "ReciprocalMulFusion: failed to add denominator tensor wrapper.");
    }

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(output_def.name)) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)),
                    "ReciprocalMulFusion: failed to add output tensor wrapper.");
    }

    // Create the fused QNN ElementWiseDivide node.
    //
    // Input ordering matters for division (non-commutative):
    //   input[0] = numerator   (the value being divided)
    //   input[1] = denominator (the divisor, originally fed into Reciprocal)
    //
    // This preserves the semantics of the original ONNX sub-graph:
    //   Mul(a, Reciprocal(b))  ==  Div(a, b)  ==  a / b
    RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(
            node_name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            QNN_OP_ELEMENT_WISE_DIVIDE,
            /*input_names=*/{numerator_def.name, denominator_def.name},
            /*output_names=*/{output_def.name},
            /*param_tensor_names=*/{},
            /*do_op_validation=*/validate),
        "ReciprocalMulFusion: failed to create fused ElementWiseDivide node.");
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
