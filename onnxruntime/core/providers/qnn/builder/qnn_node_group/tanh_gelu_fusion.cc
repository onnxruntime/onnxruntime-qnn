// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/tanh_gelu_fusion.h"

#include <algorithm>
#include <cassert>
#include <cmath>
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

// Returns true if the named input is a scalar constant approximately equal to `expected`.
static bool IsScalarConstantApprox(const QnnModelWrapper& qmw,
                                   const std::string& input_name,
                                   float expected,
                                   float tol = 1e-3f) {
  if (!qmw.IsConstantInput(input_name)) {
    return false;
  }
  const OrtValueInfo* vi = qmw.GetConstantTensor(input_name);
  if (!vi) {
    return false;
  }
  Ort::ConstValueInfo ort_vi(vi);
  Ort::ConstValue ort_val;
  if (!ort_vi.GetInitializer(ort_val).IsOK()) {
    return false;
  }
  auto type_info = ort_vi.TypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  const size_t count = tensor_info.GetElementCount();
  if (count != 1) {
    return false;
  }
  const auto elem_type = tensor_info.GetElementType();
  float val = 0.0f;
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float* data = ort_val.GetTensorData<float>();
    if (!data) return false;
    val = *data;
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const Ort::Float16_t* data = ort_val.GetTensorData<Ort::Float16_t>();
    if (!data) return false;
    val = data->ToFloat();
  } else {
    return false;
  }
  return std::abs(val - expected) <= tol;
}

// Returns true if `node_unit` is a standalone (non-QDQ) SingleNode with the given op type.
static bool IsSingleNode(const OrtNodeUnit* node_unit, std::string_view op_type) {
  return node_unit != nullptr &&
         node_unit->UnitType() == OrtNodeUnit::Type::SingleNode &&
         node_unit->OpType() == op_type;
}

std::unique_ptr<IQnnNodeGroup> TanhGeluFusion::TryFusion(
    QnnModelWrapper& qmw,
    const OrtNodeUnit& tanh_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& /*logger*/) {
  // Entry point: must be a standalone Tanh node.
  if (tanh_node_unit.OpType() != "Tanh" ||
      tanh_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // ---- Walk backwards through the pattern ----
  //
  // Pattern (x³ via two Muls, as produced by ORT optimizers):
  //
  //  [x] --+-> Mul(x,x) -> Mul(x²,x) -> Mul(0.044715) -> Add --+-> Mul(sqrt2pi) -> Tanh
  //        |                                                     |
  //        +-----------------------------------------------------+

  const auto& tanh_inputs = tanh_node_unit.Inputs();
  if (tanh_inputs.empty()) {
    return nullptr;
  }

  // Tanh <- Mul_coeff  (multiplies by sqrt(2/pi) ≈ 0.7978845608)
  const OrtNodeUnit* mul_coeff = GetParentOfInput(qmw, tanh_node_unit, tanh_inputs[0],
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(mul_coeff, "Mul")) {
    return nullptr;
  }

  const auto& mul_coeff_inputs = mul_coeff->Inputs();
  if (mul_coeff_inputs.size() < 2) {
    return nullptr;
  }

  // One input of Mul_coeff must be the sqrt(2/pi) constant; the other feeds from Add_inner.
  constexpr float kSqrt2OverPi = 0.7978845608f;
  bool coeff_is_input1 = IsScalarConstantApprox(qmw, mul_coeff_inputs[1].name, kSqrt2OverPi);
  bool coeff_is_input0 = IsScalarConstantApprox(qmw, mul_coeff_inputs[0].name, kSqrt2OverPi);
  if (!coeff_is_input0 && !coeff_is_input1) {
    return nullptr;
  }
  const OrtNodeUnitIODef& add_inner_output_def = coeff_is_input1 ? mul_coeff_inputs[0] : mul_coeff_inputs[1];

  // Mul_coeff <- Add_inner  (adds x and 0.044715*x^3)
  const OrtNodeUnit* add_inner = GetParentOfInputByName(qmw, *mul_coeff, add_inner_output_def.name,
                                                        node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(add_inner, "Add")) {
    return nullptr;
  }

  const auto& add_inner_inputs = add_inner->Inputs();
  if (add_inner_inputs.size() < 2) {
    return nullptr;
  }

  // One input of Add_inner is [x] (root); the other is Mul(0.044715, x³).
  // x³ is computed as Mul(Mul(x,x), x) — no Pow node.
  // Try both input orderings.
  constexpr float k0044715 = 0.044715f;
  const OrtNodeUnit* mul_0044715 = nullptr;  // Mul(0.044715, x³)
  const OrtNodeUnit* mul_x2 = nullptr;  // Mul(x, x)  — produces x²
  const OrtNodeUnit* mul_x3 = nullptr;  // Mul(x², x) — produces x³
  std::string root_name;

  for (int i = 0; i < 2; ++i) {
    const OrtNodeUnit* candidate = GetParentOfInput(qmw, *add_inner, add_inner_inputs[i],
                                                    node_to_node_unit, node_unit_to_qnn_node_group);
    if (!IsSingleNode(candidate, "Mul")) {
      continue;
    }
    const auto& ci = candidate->Inputs();
    if (ci.size() < 2) continue;
    if (!IsScalarConstantApprox(qmw, ci[0].name, k0044715) &&
        !IsScalarConstantApprox(qmw, ci[1].name, k0044715)) {
      continue;
    }
    // candidate is Mul(0.044715, x³); find x³ = Mul(x²,x)
    bool c_const_is1 = IsScalarConstantApprox(qmw, ci[1].name, k0044715);
    const OrtNodeUnitIODef& x3_def = c_const_is1 ? ci[0] : ci[1];
    const OrtNodeUnit* x3_node = GetParentOfInputByName(qmw, *candidate, x3_def.name,
                                                        node_to_node_unit, node_unit_to_qnn_node_group);
    if (!IsSingleNode(x3_node, "Mul")) {
      continue;
    }
    // x3_node = Mul(x², x): one input must be x, the other is Mul(x,x).
    const auto& x3i = x3_node->Inputs();
    if (x3i.size() < 2) continue;

    // Try to identify which input of x3_node is x and which is x².
    for (int j = 0; j < 2; ++j) {
      const std::string& candidate_root = x3i[j].name;
      const OrtNodeUnit* sq_candidate = GetParentOfInputByName(qmw, *x3_node, x3i[1 - j].name,
                                                               node_to_node_unit, node_unit_to_qnn_node_group);
      if (!IsSingleNode(sq_candidate, "Mul")) continue;
      // sq_candidate = Mul(x,x): both inputs must be candidate_root.
      const auto& sqi = sq_candidate->Inputs();
      if (sqi.size() < 2) continue;
      if (sqi[0].name != candidate_root && sqi[1].name != candidate_root) continue;

      // All checks passed — record the match.
      mul_0044715 = candidate;
      mul_x3 = x3_node;
      mul_x2 = sq_candidate;
      root_name = candidate_root;
      // Also verify the other Add_inner input is root [x].
      if (add_inner_inputs[1 - i].name != root_name) {
        // Reset — different root
        mul_0044715 = nullptr;
        mul_x3 = nullptr;
        mul_x2 = nullptr;
        root_name.clear();
        continue;
      }
      break;
    }
    if (mul_0044715 != nullptr) break;
  }

  if (mul_0044715 == nullptr || root_name.empty()) {
    return nullptr;
  }

  // ---- Walk forwards from Tanh ----
  //
  // Tanh -> Add(1) -> Mul(x)[Mul_4] -> Mul(0.5)[Mul_5]  [final output]
  // (ORT emits Add_1 → Mul(Add_1_out, x) → Mul(result, 0.5))

  const auto& tanh_outputs = tanh_node_unit.Outputs();
  if (tanh_outputs.empty()) {
    return nullptr;
  }

  const OrtNodeUnit* add_one = GetOnlyChildOfOutput(qmw, tanh_node_unit, tanh_outputs[0],
                                                    node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(add_one, "Add")) {
    return nullptr;
  }

  // Add(1): one input must be constant 1.0.
  const auto& add_one_inputs = add_one->Inputs();
  if (add_one_inputs.size() < 2) {
    return nullptr;
  }
  if (!IsScalarConstantApprox(qmw, add_one_inputs[0].name, 1.0f) &&
      !IsScalarConstantApprox(qmw, add_one_inputs[1].name, 1.0f)) {
    return nullptr;
  }

  const auto& add_one_outputs = add_one->Outputs();
  if (add_one_outputs.empty()) {
    return nullptr;
  }

  // Add_1 -> Mul(x) — multiplies (1+Tanh) by root [x]
  const OrtNodeUnit* mul_x = GetOnlyChildOfOutput(qmw, *add_one, add_one_outputs[0],
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(mul_x, "Mul")) {
    return nullptr;
  }
  const auto& mul_x_inputs = mul_x->Inputs();
  if (mul_x_inputs.size() < 2) {
    return nullptr;
  }
  if (mul_x_inputs[0].name != root_name && mul_x_inputs[1].name != root_name) {
    return nullptr;
  }

  const auto& mul_x_outputs = mul_x->Outputs();
  if (mul_x_outputs.empty()) {
    return nullptr;
  }

  // Mul(x) -> Mul(0.5) — final scaling
  const OrtNodeUnit* mul_half = GetOnlyChildOfOutput(qmw, *mul_x, mul_x_outputs[0],
                                                     node_to_node_unit, node_unit_to_qnn_node_group);
  if (!IsSingleNode(mul_half, "Mul")) {
    return nullptr;
  }
  const auto& mul_half_inputs = mul_half->Inputs();
  if (mul_half_inputs.size() < 2) {
    return nullptr;
  }
  if (!IsScalarConstantApprox(qmw, mul_half_inputs[0].name, 0.5f) &&
      !IsScalarConstantApprox(qmw, mul_half_inputs[1].name, 0.5f)) {
    return nullptr;
  }

  const auto& mul_half_outputs = mul_half->Outputs();
  if (mul_half_outputs.empty()) {
    return nullptr;
  }

  // Collect root IODef from add_inner's inputs.
  OrtNodeUnitIODef root_input;
  for (const auto& inp : add_inner_inputs) {
    if (inp.name == root_name) {
      root_input = inp;
      break;
    }
  }
  OrtNodeUnitIODef final_output = mul_half_outputs[0];

  // Validate QNN Gelu accepts these tensor types.
  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;
  if (!qmw.MakeTensorWrapper(root_input, input_tensor).IsOK()) {
    return nullptr;
  }
  if (!qmw.MakeTensorWrapper(final_output, output_tensor).IsOK()) {
    return nullptr;
  }
  const std::string node_name = utils::UniqueNameGenerator().New(tanh_node_unit);
  if (!qmw.ValidateQnnNode(node_name,
                           QNN_OP_PACKAGE_NAME_QTI_AISW,
                           QNN_OP_GELU,
                           {input_tensor.GetQnnTensor()},
                           {output_tensor.GetQnnTensor()},
                           {})
           .IsOK()) {
    return nullptr;
  }

  std::vector<const OrtNodeUnit*> node_units = {
      mul_x2, mul_x3, mul_0044715, add_inner, mul_coeff,
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
  return qmw.ValidateQnnNode(node_name,
                             QNN_OP_PACKAGE_NAME_QTI_AISW,
                             QNN_OP_GELU,
                             {input_tensor.GetQnnTensor()},
                             {output_tensor.GetQnnTensor()},
                             {});
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
  RETURN_IF_NOT(qmw.CreateQnnNode(node_name,
                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                  QNN_OP_GELU,
                                  {gelu_root_input_.name},
                                  {gelu_final_output_.name},
                                  {},
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
