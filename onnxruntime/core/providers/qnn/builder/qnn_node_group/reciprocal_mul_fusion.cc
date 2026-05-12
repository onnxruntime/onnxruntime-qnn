// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

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
// QDQ support
// -----------
// Both SingleNode and QDQGroup Reciprocal units are handled.  In quantized
// models the ORT graph partitioner wraps the Reciprocal in a QDQ group:
//
//   [denominator] --> DQ --> Reciprocal --> Q --+
//                                               v
//   [numerator]  --------------------------------> (DQ ->) Mul --> [output]
//
// GetChildNodeUnitAllowQdq is used to locate the downstream Mul, skipping
// the Q -> DQ boundary that separates the two logical nodes.  The
// OrtNodeUnit::Inputs() / Outputs() accessors already return the logical
// (dequantized) tensor names for QDQ groups, so CreateOrValidateOnQnn
// requires no changes to handle both cases.
//
// Tensor role mapping
// -------------------
//   ONNX input  : denominator  (Reciprocal's logical input  -- DQ output for QDQ)
//   ONNX input  : numerator    (the other Mul logical input -- DQ output for QDQ)
//   ONNX output : result       (Mul's logical output        -- Q  input  for QDQ)
//
//   QNN Div input[0]  = numerator
//   QNN Div input[1]  = denominator
//   QNN Div output[0] = result
//
// =============================================================================

#include "core/providers/qnn/builder/qnn_node_group/reciprocal_mul_fusion.h"

#include <array>
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
#define ValidateOnQnn(qnn_model_wrapper, reciprocal_node_unit, mul_node_unit, recip_is_mul_input0) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (reciprocal_node_unit), (mul_node_unit), (recip_is_mul_input0), /*validate=*/true)
#define CreateOnQnn(qnn_model_wrapper, reciprocal_node_unit, mul_node_unit, recip_is_mul_input0) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (reciprocal_node_unit), (mul_node_unit), (recip_is_mul_input0), /*validate=*/false)

// Forward declaration so the use sites of the macros above can be parsed before
// the full definition appears at the bottom of this translation unit.
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& reciprocal_node_unit,
                                         const OrtNodeUnit& mul_node_unit,
                                         bool recip_is_mul_input0,
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
//   1. Verify the entry node is a Reciprocal (SingleNode or QDQGroup).
//   2. Confirm the Reciprocal has exactly one consumer and that consumer is
//      a Mul node (SingleNode or QDQGroup) that has not already been claimed.
//      GetChildNodeUnitAllowQdq handles all of the following atomically:
//        (a) For QDQ Reciprocal: follows the Q node's output, then skips the
//            downstream DQ node to reach the true consumer.
//        (b) That output is NOT a graph-level output.
//        (c) That output has exactly one consumer node.
//        (d) That consumer's op-type is "Mul".
//        (e) The Mul NodeUnit has not already been claimed by another group.
//   3. Confirm the Mul actually consumes the Reciprocal output (sanity check
//      against malformed graphs where the lookup might return a Mul that is
//      connected via a different edge).
//   4. Perform a QNN dry-run validation to ensure the backend can handle the
//      resulting ElementWiseDivide node.
//   5. Construct and return the ReciprocalMulFusion object.
//
// Note: explicit input/output count guards for Reciprocal (unary) and Mul
// (binary) are intentionally absent — ONNX spec compliance is assumed per
// the QNN EP review checklist [T06].  GetChildNodeUnitAllowQdq (Step 2) and
// ValidateQnnNode (Step 4) already catch any malformed graphs.
std::unique_ptr<IQnnNodeGroup> ReciprocalMulFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reciprocal_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // -- Step 1: Gate on op-type -----------------------------------------------
  //
  // Accept both standalone (SingleNode) and QDQ-wrapped (QDQGroup) Reciprocal
  // units.  In quantized models the ORT graph partitioner wraps the Reciprocal
  // in a QDQ group (DQ -> Reciprocal -> Q); we must handle that case to keep
  // the entire computation on the QNN accelerator.
  if (reciprocal_node_unit.OpType() != "Reciprocal") {
    return nullptr;
  }

  // -- Step 2: Locate the single Mul consumer of the Reciprocal output ------
  //
  // GetChildNodeUnitAllowQdq performs all of the following checks atomically:
  //   (a) For a QDQGroup Reciprocal: follows the Q node's output rather than
  //       the target node's output, then skips the downstream DQ node to
  //       reach the true consumer (the Mul or its DQ wrapper).
  //   (b) That output tensor is NOT a graph-level output.
  //   (c) That output has exactly one consumer node.
  //   (d) That consumer's op-type is "Mul" (SingleNode or QDQGroup).
  //   (e) The Mul NodeUnit has not already been claimed by another
  //       IQnnNodeGroup (prevents double-fusion).
  //
  // If any condition fails, nullptr is returned and we bail out.
  const OrtNodeUnit* mul_node_unit =
      GetChildNodeUnitAllowQdq(qnn_model_wrapper, reciprocal_node_unit, "Mul",
                               node_to_node_unit, node_unit_to_qnn_node_group);
  if (mul_node_unit == nullptr) {
    return nullptr;
  }

  // -- Step 3: Verify the Reciprocal output is actually wired into the Mul --
  //
  // GetChildNodeUnitAllowQdq guarantees the Mul is the sole consumer of the
  // Reciprocal output, but it does not verify *which* input slot of the Mul
  // carries that value.  We do that here as a defence-in-depth check.
  //
  // For a QDQ-wrapped Reciprocal the logical output name exposed by
  // OrtNodeUnit::Outputs()[0] is the Q node's output (the quantized tensor),
  // while the Mul's logical input name (OrtNodeUnit::Inputs()[i]) is the
  // downstream DQ node's output (the dequantized tensor).  These two names
  // differ, so we cannot compare them directly.  Instead we rely on
  // GetChildNodeUnitAllowQdq having already confirmed the topological
  // connection and skip the name-equality check for QDQ Reciprocal units.
  //
  // For SingleNode Reciprocal units the names are directly comparable.
  //
  // ONNX Mul is commutative, so the Reciprocal result may appear in either
  // input[0] or input[1].
  const auto& mul_inputs = mul_node_unit->Inputs();
  bool recip_is_mul_input0 = false;
  bool recip_is_mul_input1 = false;

  if (reciprocal_node_unit.UnitType() == OrtNodeUnit::Type::SingleNode) {
    // For a bare Reciprocal the output name is the intermediate tensor name
    // that directly appears as one of the Mul's input names.
    const std::string& recip_output_name = reciprocal_node_unit.Outputs()[0].name;
    recip_is_mul_input0 = (mul_inputs[0].name == recip_output_name);
    recip_is_mul_input1 = (mul_inputs[1].name == recip_output_name);

    if (!recip_is_mul_input0 && !recip_is_mul_input1) {
      // The Mul does not actually consume the Reciprocal output.  This can
      // happen if the graph is malformed or if GetChildNodeUnitAllowQdq
      // returned a Mul that is connected via a different edge.  Bail out.
      return nullptr;
    }

    if (recip_is_mul_input0 && recip_is_mul_input1) {
      // Defence-in-depth: Mul(1/b, 1/b) = 1/b² ≠ Div(anything, b), so
      // fusing would change semantics.  In practice this branch is
      // unreachable: GetChildNodeUnitAllowQdq's single-consumer guard
      // already prevents the Reciprocal output from feeding both Mul
      // input slots simultaneously (that would require the same tensor
      // to be its own sole consumer twice).  The check is kept here
      // only as a belt-and-suspenders safeguard against future refactors.
      return nullptr;
    }
  } else {
    // QDQGroup: GetChildNodeUnitAllowQdq already verified the topological
    // connection (Q -> DQ boundary traversal).  We still need to determine
    // which Mul input slot carries the Reciprocal's dequantized output so
    // that CreateOrValidateOnQnn can identify the numerator correctly.
    //
    // The Reciprocal QDQ group's logical output (Outputs()[0]) is the Q
    // node's output tensor.  The downstream DQ node dequantizes that tensor
    // and its output is what appears in the Mul's Inputs() list.  We locate
    // the DQ output name by following the Q node's single consumer.
    const OrtNode* q_node = reciprocal_node_unit.GetQNodes().empty()
                                ? nullptr
                                : reciprocal_node_unit.GetQNodes()[0];
    if (q_node == nullptr) {
      return nullptr;
    }

    // The Q node has one output; its single consumer is the DQ node whose
    // output feeds the Mul.  Retrieve that DQ output name.
    const std::vector<Ort::ConstValueInfo> q_outputs = Ort::ConstNode(q_node).GetOutputs();
    if (q_outputs.size() != 1) {
      return nullptr;
    }
    const std::vector<Ort::ValueInfoConsumerProducerInfo> dq_consumers = q_outputs[0].GetConsumers();
    if (dq_consumers.size() != 1 || dq_consumers[0].node == nullptr) {
      return nullptr;
    }
    const std::vector<Ort::ConstValueInfo> dq_outputs =
        Ort::ConstNode(dq_consumers[0].node).GetOutputs();
    if (dq_outputs.size() != 1) {
      return nullptr;
    }
    const std::string dq_output_name = dq_outputs[0].GetName();

    recip_is_mul_input0 = (mul_inputs[0].name == dq_output_name);
    recip_is_mul_input1 = (mul_inputs[1].name == dq_output_name);

    if (!recip_is_mul_input0 && !recip_is_mul_input1) {
      return nullptr;
    }
    if (recip_is_mul_input0 && recip_is_mul_input1) {
      // Defence-in-depth: same reasoning as the SingleNode branch above.
      // GetChildNodeUnitAllowQdq's single-consumer guard makes this
      // unreachable in practice; kept for belt-and-suspenders safety.
      return nullptr;
    }
  }

  // -- Step 4: QNN capability dry-run ----------------------------------------
  //
  // Ask the QNN backend whether it can handle an ElementWiseDivide node
  // with the tensor types and shapes inferred from the ONNX graph.  This
  // call does NOT modify the QnnModelWrapper's internal state; it is a
  // pure read-only capability query.
  //
  // If the backend rejects the node (e.g. unsupported data type or rank),
  // we return nullptr so the two nodes fall back to individual handling.
  if (Ort::Status status = ValidateOnQnn(qnn_model_wrapper, reciprocal_node_unit, *mul_node_unit, recip_is_mul_input0);
      !status.IsOK()) {
    return nullptr;
  }

  // -- Step 5: Commit to the fusion ------------------------------------------
  //
  // All checks passed.  Construct the fusion object, caching recip_is_mul_input0
  // so that CreateOrValidateOnQnn does not need to repeat the Q -> DQ traversal
  // during the build phase.  The actual QNN node will be created later when
  // AddToModelBuilder() is called.
  return std::make_unique<ReciprocalMulFusion>(reciprocal_node_unit, *mul_node_unit, recip_is_mul_input0);
}

// =============================================================================
// ReciprocalMulFusion constructor
// =============================================================================

ReciprocalMulFusion::ReciprocalMulFusion(const OrtNodeUnit& reciprocal_node_unit,
                                         const OrtNodeUnit& mul_node_unit,
                                         bool recip_is_mul_input0)
    : node_units_{&reciprocal_node_unit, &mul_node_unit},
      recip_is_mul_input0_{recip_is_mul_input0} {
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
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1], recip_is_mul_input0_);
}

// AddToModelBuilder
// -----------------
// Called during the model-building phase to register tensors and emit the
// fused QNN ElementWiseDivide node into the QNN graph.
Ort::Status ReciprocalMulFusion::AddToModelBuilder(QnnModelWrapper& qmw,
                                                   const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1], recip_is_mul_input0_);
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
// Contract (qnn_node_group.h lines 37-38):
//   "The target should be the first NodeUnit where all input paths
//    (of the IQnnNodeGroup) converge."
//
// In this fusion the two input paths are independent until they meet at Mul:
//
//   [denominator] --> Reciprocal --+
//                                  v
//   [numerator]  ----------------> Mul  <-- convergence point
//
// The numerator arrives directly; the denominator travels through Reciprocal
// first.  Neither path is a subset of the other, so the earliest node where
// BOTH are available is Mul.  Mul is therefore the correct target.
//
// Contrast with HardSigmoidMulFusion, which returns node_units_[0]
// (HardSigmoid) as its target.  That fusion shares a single root tensor x
// for both branches:
//
//   [x] --> HardSigmoid --+
//    |                     v
//    +-------------------> Mul
//
// Because x is already present before HardSigmoid executes, HardSigmoid
// itself is the first point where all inputs of the group are available,
// making it the convergence node — not the downstream Mul.
const OrtNodeUnit* ReciprocalMulFusion::GetTargetNodeUnit() const {
  return node_units_[1];  // Mul is the convergence point; see comment above
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
//   input[1]  = denominator -- the Reciprocal's logical input
//                              (DQ output for QDQ groups)
//   output[0] = result      -- the Mul's logical output
//                              (Q input for QDQ groups)
//
// For both SingleNode and QDQGroup Reciprocal units,
// OrtNodeUnit::Inputs()[0] returns the logical (dequantized) input tensor
// and OrtNodeUnit::Outputs()[0] returns the logical output tensor.  The
// intermediate Q/DQ tensors are never registered in the QNN graph.
//
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& reciprocal_node_unit,
                                         const OrtNodeUnit& mul_node_unit,
                                         bool recip_is_mul_input0,
                                         bool validate) {
  RETURN_IF_NOT(reciprocal_node_unit.OpType() == "Reciprocal",
                ("ReciprocalMulFusion: expected Reciprocal op, got " + reciprocal_node_unit.OpType()).c_str());
  RETURN_IF_NOT(mul_node_unit.OpType() == "Mul",
                ("ReciprocalMulFusion: expected Mul op, got " + mul_node_unit.OpType()).c_str());

  // -- Resolve tensor roles --------------------------------------------------
  //
  // denominator: the logical input fed into Reciprocal (the value being
  //              inverted).  For a QDQGroup this is the DQ node's output
  //              (the dequantized tensor); OrtNodeUnit::Inputs()[0] returns
  //              this name directly.  This becomes input[1] of the Div node.
  const OrtNodeUnitIODef& denominator_def = reciprocal_node_unit.Inputs()[0];

  // recip_is_mul_input0 was resolved once in TryFusion (Step 3) and cached
  // on the fusion object.  It tells us which Mul input slot carries the
  // Reciprocal's output so we can identify the numerator without repeating
  // the Q -> DQ graph traversal here.
  const auto& mul_inputs = mul_node_unit.Inputs();

  // numerator: whichever Mul input is NOT the Reciprocal output.
  //            This becomes input[0] of the Div node.
  const OrtNodeUnitIODef& numerator_def = recip_is_mul_input0 ? mul_inputs[1] : mul_inputs[0];

  // result: the Mul's logical output tensor becomes the Div output unchanged.
  const OrtNodeUnitIODef& output_def = mul_node_unit.Outputs()[0];

  // Use the Reciprocal node's unique name as the fused node name.  This
  // keeps the QNN graph node name stable and traceable back to the original
  // ONNX graph for debugging and profiling purposes.
  const std::string node_name = utils::UniqueNameGenerator().New(reciprocal_node_unit);

  if (validate) {
    // -- Dry-run: capability query only ---------------------------------------
    //
    // Build temporary tensor descriptors solely to satisfy the ValidateQnnNode
    // signature.  MakeTensorWrapper reads the tensor's shape, element
    // data-type, and quantisation parameters from the ONNX graph.  These
    // descriptors are intentionally local to this block: ValidateQnnNode does
    // NOT modify the model wrapper's internal tables, so the wrappers are
    // discarded after the call returns.
    //
    // A failure here means the backend cannot handle this Div configuration
    // (e.g. unsupported data type or tensor rank), so we return the error to
    // the caller which will then fall back to individual node handling.
    QnnTensorWrapper numerator_tensor;
    QnnTensorWrapper denominator_tensor;
    QnnTensorWrapper output_tensor;

    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(numerator_def, numerator_tensor));
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(denominator_def, denominator_tensor));
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

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
    // node that shares the same tensor (e.g. the denominator is a graph input
    // also consumed by another op, or the numerator is shared across a
    // LayerNorm-like pattern).  IsQnnTensorWrapperExist() guards against
    // double-registration, which would corrupt the internal tables.
    //
    // Crucially, MakeTensorWrapper is called only when the tensor is NOT yet
    // registered.  Calling it unconditionally and then discarding the result
    // wastes a GetTensorInfo + shape resolution + quant-param extraction +
    // vector allocation for every already-registered tensor.
    //
    // The intermediate Reciprocal output tensor is intentionally NEVER
    // registered here.  It does not exist in the QNN graph; the fusion
    // replaces it with a direct edge from the denominator to the Div node.

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(numerator_def.name)) {
      QnnTensorWrapper numerator_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(numerator_def, numerator_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(numerator_tensor)),
                    "ReciprocalMulFusion: failed to add numerator tensor wrapper.");
    }

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(denominator_def.name)) {
      QnnTensorWrapper denominator_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(denominator_def, denominator_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(denominator_tensor)),
                    "ReciprocalMulFusion: failed to add denominator tensor wrapper.");
    }

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(output_def.name)) {
      QnnTensorWrapper output_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));
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
