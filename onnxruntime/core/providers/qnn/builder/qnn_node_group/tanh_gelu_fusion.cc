// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/tanh_gelu_fusion.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gsl/gsl>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// GELU approximation constants.
constexpr float kGeluCubicCoeff = 0.044715f;       // coefficient on x³ term
constexpr float kGeluSqrt2OverPi = 0.7978845608f;  // sqrt(2/pi)
constexpr float kGeluOne = 1.0f;
constexpr float kGeluHalf = 0.5f;

// Returns true if the first output of `node_unit` has exactly one consumer in the graph.
// Used to guard backward-walk fusions: if an intermediate is consumed by more than one op,
// fusing it would leave those other consumers with a dangling input.
bool HasSingleOutputConsumer(const OrtNodeUnit& node_unit) {
  const Ort::ConstNode node(&node_unit.GetNode());
  const auto outputs = node.GetOutputs();
  if (outputs.empty()) return false;
  return outputs[0].GetConsumers().size() == 1;
}

// Returns true if `node_unit` is a standalone (non-QDQ) SingleNode with the given op type.
// QDQ support is intentionally out of scope for this fusion: in a quantized tanh-GELU model
// every Mul/Add/Tanh is a QDQGroup, and the DQ/Q unwrap logic needed to fuse across QDQ
// boundaries (see GeluFusion::TryFusion) adds significant complexity. That work is left as a
// follow-up; for now only float32/float16 non-QDQ graphs are fused.
bool IsSingleNode(const OrtNodeUnit* node_unit, std::string_view op_type) {
  return node_unit != nullptr &&
         node_unit->UnitType() == OrtNodeUnit::Type::SingleNode &&
         node_unit->OpType() == op_type;
}

// For a commutative 2-input node, returns a pointer to the non-constant input when exactly
// one input is a scalar constant matching `constant_val`. Returns nullptr otherwise.
// Requiring exactly one match avoids false-positives when both inputs happen to equal
// the constant (e.g., Mul(0.5, 0.5)) or neither matches.
// The caller must keep `inputs` alive for the duration of any use of the returned pointer.
const OrtNodeUnitIODef* NonConstInput(const QnnModelWrapper& qmw,
                                      const std::vector<OrtNodeUnitIODef>& inputs,
                                      float constant_val) {
  if (inputs.size() < 2) return nullptr;
  const bool is0 = IsScalarConstantApprox(qmw, inputs[0].name, constant_val);
  const bool is1 = IsScalarConstantApprox(qmw, inputs[1].name, constant_val);
  if (is1 && !is0) return &inputs[0];  // input[1] is the constant → return input[0]
  if (is0 && !is1) return &inputs[1];  // input[0] is the constant → return input[1]
  return nullptr;                      // both or neither match — ambiguous, reject
}

// Returns true if both inputs of a 2-input node share the same name (i.e., Mul(x, x)).
bool BothInputsAreNamed(const std::vector<OrtNodeUnitIODef>& inputs, const std::string& name) {
  return inputs.size() >= 2 && inputs[0].name == name && inputs[1].name == name;
}

struct CubicSubtreeMatch {
  const OrtNodeUnit* mul_cubic_coeff = nullptr;  // Mul(x³, kGeluCubicCoeff)
  const OrtNodeUnit* mul_x3 = nullptr;           // Mul(x², x)  — or Mul(x, x²) when swapped
  const OrtNodeUnit* mul_x2 = nullptr;           // Mul(x, x)
  std::string root_name;                         // name of the x tensor; empty signals no match

  bool IsValid() const { return !root_name.empty(); }
};

// Tries to match the cubic sub-expression  0.044715 * x³  rooted at
// add_inner.inputs[branch_idx], verifying that add_inner.inputs[1-branch_idx] is the
// same root tensor x.  The `for i` loop in TryFusion calls this twice (branch_idx=0,1)
// to handle both orderings of the Add(x, 0.044715*x³) node.
//
// Expected sub-graph (ignoring commutative orderings within each Mul):
//
//   add_inner.inputs[branch_idx]
//         |
//         v
//   Mul(kGeluCubicCoeff, x³)       ← mul_cubic_coeff
//                         |
//                         v
//                   Mul(x², x)     ← mul_x3  (either input may be x or x²)
//                       |
//                       v
//                   Mul(x, x)      ← mul_x2
//
// Returns a populated CubicSubtreeMatch (IsValid()==true) on success, empty otherwise.
CubicSubtreeMatch TryMatchCubicSubtree(
    QnnModelWrapper& qmw,
    const OrtNodeUnit& add_inner,
    int branch_idx,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  const auto& add_inner_inputs = add_inner.Inputs();

  // Step 1: add_inner.inputs[branch_idx] must come from Mul(kGeluCubicCoeff, x³).
  const OrtNodeUnit* mul_cubic_coeff = GetParentOfInput(qmw, add_inner, add_inner_inputs[branch_idx],
                                                        node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(mul_cubic_coeff, "Mul") || !HasSingleOutputConsumer(*mul_cubic_coeff)) return {};

  // Step 2: identify which input of mul_cubic_coeff is the constant and which feeds x³.
  const OrtNodeUnitIODef* x3_def = NonConstInput(qmw, mul_cubic_coeff->Inputs(), kGeluCubicCoeff);
  if (!x3_def) return {};

  // Step 3: x³ must come from Mul(x², x) — either input ordering.
  const OrtNodeUnit* mul_x3 = GetParentOfInputByName(qmw, *mul_cubic_coeff, x3_def->name,
                                                     node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(mul_x3, "Mul") || !HasSingleOutputConsumer(*mul_x3)) return {};
  const auto& x3_inputs = mul_x3->Inputs();
  if (x3_inputs.size() < 2) return {};

  // Step 4: `for j` — try both slots of mul_x3 as the candidate root x.
  //   j=0: x3_inputs[0] is x,  x3_inputs[1] comes from Mul(x,x)
  //   j=1: x3_inputs[1] is x,  x3_inputs[0] comes from Mul(x,x)
  for (int j = 0; j < 2; ++j) {
    const std::string& root_candidate = x3_inputs[j].name;

    // The other input of mul_x3 must come from Mul(root_candidate, root_candidate) = x².
    const OrtNodeUnit* mul_x2 = GetParentOfInputByName(qmw, *mul_x3, x3_inputs[1 - j].name,
                                                       node_to_node_unit, node_unit_to_qnn_node_group);
    if (!IsSingleNode(mul_x2, "Mul") || !HasSingleOutputConsumer(*mul_x2)) continue;
    if (!BothInputsAreNamed(mul_x2->Inputs(), root_candidate)) continue;

    // Step 5: the straight-through input of add_inner (the non-cubic arm) must also be root x.
    if (add_inner_inputs[1 - branch_idx].name != root_candidate) continue;

    return CubicSubtreeMatch{mul_cubic_coeff, mul_x3, mul_x2, root_candidate};
  }
  return {};
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> TanhGeluFusion::TryFusion(
    QnnModelWrapper& qmw,
    const OrtNodeUnit& tanh_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& /*logger*/) {
  if (tanh_node_unit.OpType() != "Tanh" ||
      tanh_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  //        +-> Mul(x,x) --+
  //        |              v
  //   [x] -+-----------> Mul(x²,x) --> Mul(0.044715) --+
  //        |                                            v
  //        +-------------------------------------------> Add --> Mul(sqrt(2/pi)) --> Tanh
  //
  // ... then forward from Tanh:
  //
  //   Tanh --> Add(1) --> Mul(x) --> Mul(0.5) ==>
  //                          ^
  //                   (x fan-out)

  const auto& tanh_inputs = tanh_node_unit.Inputs();
  if (tanh_inputs.empty()) return nullptr;

  // ---- Backward walk: Tanh ← Mul(sqrt2pi) ← Add(x, 0.044715*x³) ----

  // tanh_inputs[0] must come from Mul(sqrt(2/pi), Add_inner).
  const OrtNodeUnit* mul_coeff = GetParentOfInput(qmw, tanh_node_unit, tanh_inputs[0],
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(mul_coeff, "Mul") || !HasSingleOutputConsumer(*mul_coeff)) return nullptr;

  // One input of mul_coeff is the sqrt(2/pi) scalar; the other feeds from add_inner.
  const OrtNodeUnitIODef* add_inner_out = NonConstInput(qmw, mul_coeff->Inputs(), kGeluSqrt2OverPi);
  if (!add_inner_out) return nullptr;

  const OrtNodeUnit* add_inner = GetParentOfInputByName(qmw, *mul_coeff, add_inner_out->name,
                                                        node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(add_inner, "Add") || !HasSingleOutputConsumer(*add_inner)) return nullptr;
  if (add_inner->Inputs().size() < 2) return nullptr;

  // Try add_inner with the cubic arm in slot 0 (i=0) then slot 1 (i=1).
  CubicSubtreeMatch cubic;
  for (int i = 0; i < 2 && !cubic.IsValid(); ++i) {
    cubic = TryMatchCubicSubtree(qmw, *add_inner, i, node_to_node_unit, node_unit_to_qnn_node_group);
  }
  if (!cubic.IsValid()) return nullptr;

  // ---- Forward walk: Tanh → Add(1) → Mul(x) → Mul(0.5) ----

  const auto& tanh_outputs = tanh_node_unit.Outputs();
  if (tanh_outputs.empty()) return nullptr;

  // tanh_out → Add(1) — shifts the Tanh range from [-1,1] to [0,2].
  const OrtNodeUnit* add_one = GetOnlyChildOfOutput(qmw, tanh_node_unit, tanh_outputs[0],
                                                    node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(add_one, "Add")) return nullptr;
  const auto& add_one_inputs = add_one->Inputs();
  if (add_one_inputs.size() < 2) return nullptr;
  if (!IsScalarConstantApprox(qmw, add_one_inputs[0].name, kGeluOne) &&
      !IsScalarConstantApprox(qmw, add_one_inputs[1].name, kGeluOne)) {
    return nullptr;
  }
  const auto& add_one_outputs = add_one->Outputs();
  if (add_one_outputs.empty()) return nullptr;

  // add_one_out → Mul(x) — multiplies (1 + tanh(...)) by the root input x.
  // Confirms the same x flows into the tail as the one identified in the cubic sub-tree.
  const OrtNodeUnit* mul_x = GetOnlyChildOfOutput(qmw, *add_one, add_one_outputs[0],
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(mul_x, "Mul")) return nullptr;
  const auto& mul_x_inputs = mul_x->Inputs();
  if (mul_x_inputs.size() < 2) return nullptr;
  if (mul_x_inputs[0].name != cubic.root_name && mul_x_inputs[1].name != cubic.root_name) {
    return nullptr;
  }
  const auto& mul_x_outputs = mul_x->Outputs();
  if (mul_x_outputs.empty()) return nullptr;

  // mul_x_out → Mul(0.5) — applies the final ½ factor.
  const OrtNodeUnit* mul_half = GetOnlyChildOfOutput(qmw, *mul_x, mul_x_outputs[0],
                                                     node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(mul_half, "Mul")) return nullptr;
  const auto& mul_half_inputs = mul_half->Inputs();
  if (mul_half_inputs.size() < 2) return nullptr;
  if (!IsScalarConstantApprox(qmw, mul_half_inputs[0].name, kGeluHalf) &&
      !IsScalarConstantApprox(qmw, mul_half_inputs[1].name, kGeluHalf)) {
    return nullptr;
  }
  const auto& mul_half_outputs = mul_half->Outputs();
  if (mul_half_outputs.empty()) return nullptr;

  // Re-derive the root IODef from add_inner so we carry its full type/shape metadata
  // into the fused node, not just the tensor name string from cubic.root_name.
  OrtNodeUnitIODef root_input;
  for (const auto& inp : add_inner->Inputs()) {
    if (inp.name == cubic.root_name) {
      root_input = inp;
      break;
    }
  }
  OrtNodeUnitIODef final_output = mul_half_outputs[0];

  // Validate QNN ElementWiseNeuron(Gelu) accepts these tensor types.
  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;
  if (!qmw.MakeTensorWrapper(root_input, input_tensor).IsOK()) return nullptr;
  if (!qmw.MakeTensorWrapper(final_output, output_tensor).IsOK()) return nullptr;
  const std::string node_name = utils::UniqueNameGenerator().New(tanh_node_unit);
  Qnn_Scalar_t neuron_op = QNN_SCALAR_INIT;
  neuron_op.dataType = QNN_DATATYPE_UINT_32;
  neuron_op.uint32Value = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU;
  QnnParamWrapper op_param(tanh_node_unit.Index(), node_name,
                           QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION, neuron_op);
  if (!qmw.ValidateQnnNode(node_name,
                           QNN_OP_PACKAGE_NAME_QTI_AISW,
                           QNN_OP_ELEMENT_WISE_NEURON,
                           {input_tensor.GetQnnTensor()},
                           {output_tensor.GetQnnTensor()},
                           {op_param.GetQnnParam()})
           .IsOK()) {
    return nullptr;
  }

  std::vector<const OrtNodeUnit*> node_units = {
      cubic.mul_x2, cubic.mul_x3, cubic.mul_cubic_coeff, add_inner, mul_coeff,
      &tanh_node_unit, add_one, mul_x, mul_half};

  return std::make_unique<TanhGeluFusion>(std::move(node_units),
                                          &tanh_node_unit,
                                          std::move(root_input),
                                          std::move(final_output));
}

TanhGeluFusion::TanhGeluFusion(std::vector<const OrtNodeUnit*>&& node_units,
                               const OrtNodeUnit* target_node_unit,
                               OrtNodeUnitIODef gelu_root_input,
                               OrtNodeUnitIODef gelu_final_output)
    : node_units_(std::move(node_units)),
      target_node_unit_(target_node_unit),
      gelu_root_input_(std::move(gelu_root_input)),
      gelu_final_output_(std::move(gelu_final_output)) {
}

Ort::Status TanhGeluFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& /*logger*/) const {
  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(gelu_root_input_, input_tensor));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(gelu_final_output_, output_tensor));
  const std::string node_name = utils::UniqueNameGenerator().New(*target_node_unit_);
  Qnn_Scalar_t neuron_op = QNN_SCALAR_INIT;
  neuron_op.dataType = QNN_DATATYPE_UINT_32;
  neuron_op.uint32Value = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU;
  QnnParamWrapper op_param(target_node_unit_->Index(), node_name,
                           QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION, neuron_op);
  return qmw.ValidateQnnNode(node_name,
                             QNN_OP_PACKAGE_NAME_QTI_AISW,
                             QNN_OP_ELEMENT_WISE_NEURON,
                             {input_tensor.GetQnnTensor()},
                             {output_tensor.GetQnnTensor()},
                             {op_param.GetQnnParam()});
}

Ort::Status TanhGeluFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& /*logger*/) const {
  if (!qmw.IsQnnTensorWrapperExist(gelu_root_input_.name)) {
    QnnTensorWrapper input_tensor;
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(gelu_root_input_, input_tensor));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(input_tensor)), "Failed to add TanhGelu input tensor.");
  }
  if (!qmw.IsQnnTensorWrapperExist(gelu_final_output_.name)) {
    QnnTensorWrapper output_tensor;
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(gelu_final_output_, output_tensor));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(output_tensor)), "Failed to add TanhGelu output tensor.");
  }

  const std::string node_name = utils::UniqueNameGenerator().New(*target_node_unit_);
  // QNN GELU uses the exact-erf definition; the tanh approximation differs by at most ~4.7e-4
  // over [-10, 10], which is within the test tolerances (2e-3 to 6e-3) and acceptable for HTP.
  Qnn_Scalar_t neuron_op = QNN_SCALAR_INIT;
  neuron_op.dataType = QNN_DATATYPE_UINT_32;
  neuron_op.uint32Value = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU;
  QnnParamWrapper op_param(target_node_unit_->Index(), node_name,
                           QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION, neuron_op);
  const std::string op_param_name = op_param.GetParamTensorName();
  RETURN_IF_NOT(qmw.AddParamWrapper(std::move(op_param)), "Failed to add TanhGelu operation param.");
  RETURN_IF_NOT(qmw.CreateQnnNode(node_name,
                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                  QNN_OP_ELEMENT_WISE_NEURON,
                                  {gelu_root_input_.name},
                                  {gelu_final_output_.name},
                                  {op_param_name},
                                  /*do_op_validation=*/false),
                "Failed to add fused TanhGelu node.");
  return Ort::Status();
}

gsl::span<const OrtNodeUnit* const> TanhGeluFusion::GetNodeUnits() const {
  return gsl::make_span(node_units_);
}

const OrtNodeUnit* TanhGeluFusion::GetTargetNodeUnit() const {
  return target_node_unit_;
}

}  // namespace qnn
}  // namespace onnxruntime
