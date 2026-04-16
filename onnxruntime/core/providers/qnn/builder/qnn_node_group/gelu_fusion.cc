// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/gelu_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, node_units, root_input, final_output) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (root_input), (final_output), true)
#define CreateOnQnn(qnn_model_wrapper, node_units, root_input, final_output) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (root_input), (final_output), false)

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         gsl::span<const OrtNodeUnit* const> node_units,
                                         const OrtNodeUnitIODef& root_input,
                                         const OrtNodeUnitIODef& final_output,
                                         bool validate);

namespace {

struct GeluPatternMatchResult {
  std::vector<const OrtNodeUnit*> node_units;
  const OrtNodeUnit* final_mul_node_unit = nullptr;  // traces location of MUL with skip connection with root.
};

struct GeluPatternMatchContext {
  QnnModelWrapper& qnn_model_wrapper;
  const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit;
  const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group;
  const std::string& root_input_name;
};

/* Checks if the given NodeUnit has an input tensor with the specified name. */
bool HasInputWithName(const OrtNodeUnit& node_unit, std::string_view input_name) {
  const auto& inputs = node_unit.Inputs();
  return std::any_of(inputs.begin(), inputs.end(), [&input_name](const OrtNodeUnitIODef& input) {
    return input.name == input_name;
  });
}

const OrtNodeUnit* GetProducerForInput(const OrtNodeUnit& consumer_node_unit,
                                       size_t input_index,
                                       const GeluPatternMatchContext& ctx) {
  const auto& inputs = consumer_node_unit.Inputs();
  if (input_index >= inputs.size()) {
    return nullptr;
  }

  return GetParentOfInput(ctx.qnn_model_wrapper,
                          consumer_node_unit,
                          inputs[input_index],
                          ctx.node_to_node_unit,
                          ctx.node_unit_to_qnn_node_group);
}

bool TryMatchErfAddPattern1(
    const OrtNodeUnit* div_node_unit,
    const OrtNodeUnit& erf_node_unit,
    const OrtNodeUnit* add_node_unit,
    const OrtNodeUnit* mul_after_add_node_unit,
    const GeluPatternMatchContext& ctx,
    GeluPatternMatchResult& result) {
  // ErfAdd Pattern 1:
  //               +-------Mul(0.5)---------------------+
  //               |                                    |
  //               |                                    v
  //            [root] --> Div -----> Erf  --> Add --> Mul ==>
  //                      (B=1.4142...)        (1)
  //
  // At this stage: "mul_after_add_node_unit" is the final Mul after Add.
  // We now verify its non-Add input comes from Mul(root, const=0.5).
  const auto& mul_inputs = mul_after_add_node_unit->Inputs();
  if (mul_inputs.size() < 2) {
    return false;
  }

  for (size_t i = 0; i < mul_inputs.size(); ++i) {
    const OrtNodeUnit* producer = GetProducerForInput(*mul_after_add_node_unit,
                                                      i,
                                                      ctx);
    if (producer == nullptr || producer->OpType() != "Mul") {
      continue;
    }

    if (!HasInputWithName(*producer, ctx.root_input_name)) {
      continue;
    }

    result.node_units = {div_node_unit, &erf_node_unit, add_node_unit, producer, mul_after_add_node_unit};
    result.final_mul_node_unit = mul_after_add_node_unit;
    return true;
  }

  return false;
}

bool TryMatchErfAddPattern2(
    const OrtNodeUnit* div_node_unit,
    const OrtNodeUnit& erf_node_unit,
    const OrtNodeUnit* add_node_unit,
    const OrtNodeUnit* mul_after_add_node_unit,
    const GeluPatternMatchContext& ctx,
    GeluPatternMatchResult& result) {
  // ErfAdd Pattern 2:
  //               +------------------------------------+
  //               |                                    |
  //               |                                    v
  //            [root] --> Div -----> Erf  --> Add --> Mul --> Mul ==>
  //                      (B=1.4142...)        (1)            (0.5)
  //
  // At this stage: "mul_after_add_node_unit" is the first Mul after Add, and it must
  // already consume root. Then its child Mul is the final output node.
  if (!HasInputWithName(*mul_after_add_node_unit, ctx.root_input_name)) {
    return false;
  }

  const auto& mul_outputs = mul_after_add_node_unit->Outputs();
  if (mul_outputs.empty()) {
    return false;
  }

  const OrtNodeUnit* final_mul_node_unit = GetOnlyChildOfOutput(ctx.qnn_model_wrapper,
                                                                *mul_after_add_node_unit,
                                                                mul_outputs[0],
                                                                ctx.node_to_node_unit,
                                                                ctx.node_unit_to_qnn_node_group);
  if (final_mul_node_unit == nullptr || final_mul_node_unit->OpType() != "Mul") {
    return false;
  }

  result.node_units = {div_node_unit, &erf_node_unit, add_node_unit, mul_after_add_node_unit, final_mul_node_unit};
  result.final_mul_node_unit = final_mul_node_unit;
  return true;
}

bool TryMatchErfMulPattern(
    const OrtNodeUnit* div_node_unit,
    const OrtNodeUnit& erf_node_unit,
    const GeluPatternMatchContext& ctx,
    GeluPatternMatchResult& result) {
  // ErfMul Pattern (Pattern 3):
  //               +-------------------------------------------+
  //               |                                           |
  //               |                                           v
  //            [root] --> Div -----> Erf --> Mul --> Add --> Mul ==>
  //                      (B=1.4142...)      (0.5)   (0.5)
  const auto& erf_outputs = erf_node_unit.Outputs();
  if (erf_outputs.empty()) {
    return false;
  }

  // Erf should have a Mul child
  const OrtNodeUnit* mul_after_erf_node_unit = GetOnlyChildOfOutput(ctx.qnn_model_wrapper,
                                                                    erf_node_unit,
                                                                    erf_outputs[0],
                                                                    ctx.node_to_node_unit,
                                                                    ctx.node_unit_to_qnn_node_group);
  if (mul_after_erf_node_unit == nullptr || mul_after_erf_node_unit->OpType() != "Mul") {
    return false;
  }

  // This Mul should NOT consume root (it multiplies by constant 0.5)
  if (HasInputWithName(*mul_after_erf_node_unit, ctx.root_input_name)) {
    return false;
  }

  // Mul must have an Add child
  const auto& mul_outputs = mul_after_erf_node_unit->Outputs();
  if (mul_outputs.empty()) {
    return false;
  }

  const OrtNodeUnit* add_node_unit = GetOnlyChildOfOutput(ctx.qnn_model_wrapper,
                                                          *mul_after_erf_node_unit,
                                                          mul_outputs[0],
                                                          ctx.node_to_node_unit,
                                                          ctx.node_unit_to_qnn_node_group);
  if (add_node_unit == nullptr || add_node_unit->OpType() != "Add") {
    return false;
  }

  // Add must have a Mul child (final node)
  const auto& add_outputs = add_node_unit->Outputs();
  if (add_outputs.empty()) {
    return false;
  }

  const OrtNodeUnit* final_mul_node_unit = GetOnlyChildOfOutput(ctx.qnn_model_wrapper,
                                                                *add_node_unit,
                                                                add_outputs[0],
                                                                ctx.node_to_node_unit,
                                                                ctx.node_unit_to_qnn_node_group);
  if (final_mul_node_unit == nullptr || final_mul_node_unit->OpType() != "Mul") {
    return false;
  }

  // Final Mul must consume root (skip connection)
  if (!HasInputWithName(*final_mul_node_unit, ctx.root_input_name)) {
    return false;
  }

  result.node_units = {div_node_unit, &erf_node_unit, mul_after_erf_node_unit, add_node_unit, final_mul_node_unit};
  result.final_mul_node_unit = final_mul_node_unit;
  return true;
}

bool TryMatchErfAddPatterns(const OrtNodeUnit* div_node_unit,
                            const OrtNodeUnit& erf_node_unit,
                            const OrtNodeUnit* add_node_unit,
                            const OrtNodeUnit* mul_after_add_node_unit,
                            const GeluPatternMatchContext& ctx,
                            GeluPatternMatchResult& result) {
  return TryMatchErfAddPattern1(div_node_unit,
                                erf_node_unit,
                                add_node_unit,
                                mul_after_add_node_unit,
                                ctx,
                                result) ||
         TryMatchErfAddPattern2(div_node_unit,
                                erf_node_unit,
                                add_node_unit,
                                mul_after_add_node_unit,
                                ctx,
                                result);
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> GeluFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& erf_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  // Looking for an Erf node (can be SingleNode or QDQGroup).
  if (erf_node_unit.OpType() != "Erf") {
    return nullptr;
  }

  // Erf must have a Div parent on its input
  const auto& erf_inputs = erf_node_unit.Inputs();
  if (erf_inputs.empty()) {
    return nullptr;
  }

  const OrtNodeUnit* div_node_unit = GetParentOfInput(qnn_model_wrapper, erf_node_unit, erf_inputs[0],
                                                      node_to_node_unit, node_unit_to_qnn_node_group);
  if (div_node_unit == nullptr || div_node_unit->OpType() != "Div") {
    return nullptr;
  }
  const auto& div_inputs = div_node_unit->Inputs();
  if (div_inputs.size() < 2) {
    return nullptr;
  }

  // Determine which GELU pattern variant by checking Erf's child node type
  const auto& erf_outputs = erf_node_unit.Outputs();
  if (erf_outputs.empty()) {
    return nullptr;
  }

  const OrtNodeUnit* erf_child_node_unit = GetOnlyChildOfOutput(qnn_model_wrapper, erf_node_unit, erf_outputs[0],
                                                                node_to_node_unit, node_unit_to_qnn_node_group);
  if (erf_child_node_unit == nullptr) {
    return nullptr;
  }

  const std::string& root_input_name = div_inputs[0].name;
  const GeluPatternMatchContext match_ctx{qnn_model_wrapper,
                                          node_to_node_unit,
                                          node_unit_to_qnn_node_group,
                                          root_input_name};

  GeluPatternMatchResult pattern_match;
  bool is_match = false;

  if (erf_child_node_unit->OpType() == "Mul") {
    // ErfMul Pattern3: Erf -> Mul -> Add -> Mul
    // Structure: x * (Erf(x / sqrt2) * 0.5 + 0.5)
    is_match = TryMatchErfMulPattern(div_node_unit,
                                     erf_node_unit,
                                     match_ctx,
                                     pattern_match);
  } else if (erf_child_node_unit->OpType() == "Add") {
    // ErfAdd Patterns 1 or 2: Erf -> Add -> Mul [-> Mul]
    // Structure: x * 0.5 * (Erf(x / sqrt2) + 1) [with variations in Mul ordering]
    const OrtNodeUnit* add_node_unit = erf_child_node_unit;

    const auto& add_inputs = add_node_unit->Inputs();
    if (add_inputs.size() < 2) {
      return nullptr;
    }

    const auto& add_outputs = add_node_unit->Outputs();
    if (add_outputs.empty()) {
      return nullptr;
    }

    const OrtNodeUnit* mul_node_unit = GetOnlyChildOfOutput(qnn_model_wrapper, *add_node_unit, add_outputs[0],
                                                            node_to_node_unit, node_unit_to_qnn_node_group);
    if (mul_node_unit == nullptr || mul_node_unit->OpType() != "Mul") {
      return nullptr;
    }

    is_match = TryMatchErfAddPatterns(div_node_unit,
                                      erf_node_unit,
                                      add_node_unit,
                                      mul_node_unit,
                                      match_ctx,
                                      pattern_match);
  }

  if (!is_match) {
    return nullptr;
  }

  // Validate on QNN
  const OrtNodeUnitIODef& root_input = div_inputs[0];
  if (pattern_match.final_mul_node_unit == nullptr || pattern_match.final_mul_node_unit->Outputs().empty()) {
    return nullptr;
  }
  const OrtNodeUnitIODef& final_output = pattern_match.final_mul_node_unit->Outputs()[0];

  Ort::Status status = ValidateOnQnn(qnn_model_wrapper, pattern_match.node_units, root_input, final_output);
  if (!status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<GeluFusion>(std::move(pattern_match.node_units), &erf_node_unit);
}

GeluFusion::GeluFusion(std::vector<const OrtNodeUnit*>&& node_units, const OrtNodeUnit* target_node_unit)
    : node_units_(std::move(node_units)), target_node_unit_(target_node_unit) {
}

Ort::Status GeluFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const OrtNodeUnitIODef& root_input = node_units_[0]->Inputs()[0];
  const OrtNodeUnitIODef& final_output = node_units_.back()->Outputs()[0];
  return ValidateOnQnn(qmw, node_units_, root_input, final_output);
}

Ort::Status GeluFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const OrtNodeUnitIODef& root_input = node_units_[0]->Inputs()[0];
  const OrtNodeUnitIODef& final_output = node_units_.back()->Outputs()[0];
  return CreateOnQnn(qmw, node_units_, root_input, final_output);
}

gsl::span<const OrtNodeUnit* const> GeluFusion::GetNodeUnits() const {
  return gsl::make_span(node_units_);
}

const OrtNodeUnit* GeluFusion::GetTargetNodeUnit() const {
  return target_node_unit_;
}

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         gsl::span<const OrtNodeUnit* const> node_units,
                                         const OrtNodeUnitIODef& root_input,
                                         const OrtNodeUnitIODef& final_output,
                                         bool validate) {
  assert(node_units.size() >= 4);
  const auto& node_name = utils::UniqueNameGenerator().New(*node_units[0]);

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(root_input, input_tensor));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(final_output, output_tensor));

  if (validate) {
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_GELU,
                                                      {input_tensor.GetQnnTensor()},
                                                      {output_tensor.GetQnnTensor()},
                                                      {}));
  } else {
    // Only add tensor wrappers if they don't already exist
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(root_input.name)) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    }
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(final_output.name)) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    }
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_GELU,
                                                  {root_input.name},
                                                  {final_output.name},
                                                  {},
                                                  validate),
                  "Failed to add fused Gelu node.");
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
