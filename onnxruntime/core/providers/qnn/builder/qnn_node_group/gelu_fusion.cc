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

// Quantized GELU subgraphs can route the same logical root tensor through generated
// duplicated-value names such as `.../duplicated` or `.../duplicated_token_*`.
std::string_view CanonicalizeRootTensorName(std::string_view tensor_name) {
  constexpr std::string_view kDuplicatedSuffix = "/duplicated";
  constexpr std::string_view kGeneratedTokenSuffix = "_token_";

  // 1. Fast path: most tensor names are not duplicated aliases.
  const size_t duplicated_pos = tensor_name.rfind(kDuplicatedSuffix);
  if (duplicated_pos == std::string_view::npos) {
    return tensor_name;
  }

  // 2. Inspect the suffix after `/duplicated`.
  const size_t suffix_pos = duplicated_pos + kDuplicatedSuffix.size();
  const std::string_view trailing_suffix = tensor_name.substr(suffix_pos);

  // 3. Keep the name unchanged unless the remainder is empty or a generated token,
  // so we do not collapse unrelated user-defined names that merely contain the same text.
  const bool has_generated_token_suffix =
      trailing_suffix.size() >= kGeneratedTokenSuffix.size() &&
      trailing_suffix.substr(0, kGeneratedTokenSuffix.size()) == kGeneratedTokenSuffix;
  if (!trailing_suffix.empty() && !has_generated_token_suffix) {
    return tensor_name;
  }

  // 4. Strip the duplicated suffix so equivalent roots compare equal.
  return tensor_name.substr(0, duplicated_pos);
}

std::string_view GetNodeUnitTypeString(const OrtNodeUnit& node_unit) {
  return node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup ? "QDQGroup" : "SingleNode";
}

std::string GetNodeUnitDescription(const OrtNodeUnit& node_unit) {
  return "'" + node_unit.Name() + "' (op='" + node_unit.OpType() + "', unit_type='" +
         std::string(GetNodeUnitTypeString(node_unit)) + "')";
}

struct GeluPatternMatchResult {
  std::vector<const OrtNodeUnit*> node_units;
  const OrtNodeUnit* final_mul_node_unit = nullptr;  // traces location of MUL with skip connection with root.
};

struct GeluPatternMatchContext {
  QnnModelWrapper& qnn_model_wrapper;
  const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit;
  const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group;
  const Ort::Logger& logger;
  std::string_view root_input_name;
};

// Checks if a NodeUnit consumes the GELU pattern's root tensor (or an equivalent duplicate).
// When ORT creates QDQGroups with shared parent DQ nodes, it duplicates the tensor name with
// suffixes like "/duplicated" or "/duplicated_token_*" to maintain unique tensor-to-QDQ mappings.
// This canonicalizes both the input_name and the NodeUnit's inputs before comparing.
bool HasInputWithEquivalentName(const OrtNodeUnit& node_unit, std::string_view input_name) {
  const std::string_view canonical_input_name = CanonicalizeRootTensorName(input_name);
  const auto& inputs = node_unit.Inputs();

  return std::any_of(inputs.begin(), inputs.end(), [&canonical_input_name](const OrtNodeUnitIODef& input) {
    return CanonicalizeRootTensorName(input.name) == canonical_input_name;
  });
}

bool IsStandaloneQdqNodeUnit(const OrtNodeUnit* node_unit) {
  return node_unit != nullptr &&
         node_unit->UnitType() == OrtNodeUnit::Type::SingleNode &&
         (node_unit->OpType() == QUANTIZE_LINEAR || node_unit->OpType() == DEQUANTIZE_LINEAR);
}

bool WarnAndFailOnStandaloneQdq(const OrtNodeUnit* node_unit,
                                const OrtNodeUnit& /*erf_node_unit*/,
                                const Ort::Logger& logger,
                                std::string_view /*traversal_name*/) {
  if (!IsStandaloneQdqNodeUnit(node_unit)) {
    return false;
  }

  ORT_CXX_LOG(logger,
              ORT_LOGGING_LEVEL_WARNING,
              ("[GeluFusion] Mixed GELU topology detected. Culprit node unit " +
               GetNodeUnitDescription(*node_unit) +
               ". GELU fusion expects pattern nodes to be all SingleNodeUnits or QDQGroups"
               ", not partially grouped patterns with SingleNodeUnits.")
                  .c_str());
  return true;
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
  //            [root] --> Div/Mul --> Erf  --> Add --> Mul ==>
  //                    (sqrt(2) or 1/sqrt(2))  (1)
  //
  // At this stage: "mul_after_add_node_unit" is the final Mul after Add.
  // We now verify its non-Add input comes from Mul(root, Mul const=0.5).
  const auto& mul_inputs = mul_after_add_node_unit->Inputs();

  for (size_t i = 0; i < mul_inputs.size(); ++i) {
    const OrtNodeUnit* producer = GetParentOfInput(ctx.qnn_model_wrapper,
                                                   *mul_after_add_node_unit,
                                                   mul_inputs[i],
                                                   ctx.node_to_node_unit,
                                                   ctx.node_unit_to_qnn_node_group);
    if (WarnAndFailOnStandaloneQdq(producer, erf_node_unit, ctx.logger, "GetParentOfInput")) {
      return false;
    }

    if (producer == nullptr || producer->OpType() != "Mul") {
      continue;
    }

    if (!HasInputWithEquivalentName(*producer, ctx.root_input_name)) {
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
  //            [root] --> Div/Mul --> Erf  --> Add --> Mul --> Mul ==>
  //                    (sqrt(2) or 1/sqrt(2))  (1)            (0.5)
  //
  // At this stage: "mul_after_add_node_unit" is the first Mul after Add, and it must
  // already consume root. Then its child Mul is the final output node.
  if (!HasInputWithEquivalentName(*mul_after_add_node_unit, ctx.root_input_name)) {
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
  if (WarnAndFailOnStandaloneQdq(final_mul_node_unit, erf_node_unit, ctx.logger, "GetOnlyChildOfOutput")) {
    return false;
  }

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
  //            [root] --> Div/Mul --> Erf --> Mul --> Add --> Mul ==>
  //                  (sqrt(2) or 1/sqrt(2))  (0.5)   (0.5)
  // In this pattern, the Mul right after Erf must not consume root and have a Add and Mul seq after it.
  const auto& erf_outputs = erf_node_unit.Outputs();
  if (erf_outputs.empty()) {
    return false;
  }

  const OrtNodeUnit* mul_after_erf_node_unit = GetOnlyChildOfOutput(ctx.qnn_model_wrapper,
                                                                    erf_node_unit,
                                                                    erf_outputs[0],
                                                                    ctx.node_to_node_unit,
                                                                    ctx.node_unit_to_qnn_node_group);
  if (WarnAndFailOnStandaloneQdq(mul_after_erf_node_unit, erf_node_unit, ctx.logger, "GetOnlyChildOfOutput")) {
    return false;
  }

  if (mul_after_erf_node_unit == nullptr || mul_after_erf_node_unit->OpType() != "Mul") {
    return false;
  }

  if (HasInputWithEquivalentName(*mul_after_erf_node_unit, ctx.root_input_name)) {
    return false;
  }

  const auto& mul_outputs = mul_after_erf_node_unit->Outputs();
  if (mul_outputs.empty()) {
    return false;
  }

  const OrtNodeUnit* add_node_unit = GetOnlyChildOfOutput(ctx.qnn_model_wrapper,
                                                          *mul_after_erf_node_unit,
                                                          mul_outputs[0],
                                                          ctx.node_to_node_unit,
                                                          ctx.node_unit_to_qnn_node_group);
  if (WarnAndFailOnStandaloneQdq(add_node_unit, erf_node_unit, ctx.logger, "GetOnlyChildOfOutput")) {
    return false;
  }

  if (add_node_unit == nullptr || add_node_unit->OpType() != "Add") {
    return false;
  }

  const auto& add_outputs = add_node_unit->Outputs();
  if (add_outputs.empty()) {
    return false;
  }

  const OrtNodeUnit* final_mul_node_unit = GetOnlyChildOfOutput(ctx.qnn_model_wrapper,
                                                                *add_node_unit,
                                                                add_outputs[0],
                                                                ctx.node_to_node_unit,
                                                                ctx.node_unit_to_qnn_node_group);
  if (WarnAndFailOnStandaloneQdq(final_mul_node_unit, erf_node_unit, ctx.logger, "GetOnlyChildOfOutput")) {
    return false;
  }

  if (final_mul_node_unit == nullptr || final_mul_node_unit->OpType() != "Mul") {
    return false;
  }

  if (!HasInputWithEquivalentName(*final_mul_node_unit, ctx.root_input_name)) {
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
  // Looking for an Erf node (can be SingleNode or QDQGroup).
  if (erf_node_unit.OpType() != "Erf") {
    return nullptr;
  }

  // Reject fusion if QDQ nodes are found inside the GELU topology to possibly avoid accuracy variation.
  if (erf_node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup) {
    return nullptr;
  }

  const auto& erf_inputs = erf_node_unit.Inputs();
  if (erf_inputs.empty()) {
    return nullptr;
  }

  const OrtNodeUnit* div_node_unit = GetParentOfInput(qnn_model_wrapper,
                                                      erf_node_unit,
                                                      erf_inputs[0],
                                                      node_to_node_unit,
                                                      node_unit_to_qnn_node_group);
  if (WarnAndFailOnStandaloneQdq(div_node_unit, erf_node_unit, logger, "GetParentOfInput")) {
    return nullptr;
  }

  // optimizer may replace Div with Mul.
  if (div_node_unit == nullptr || (div_node_unit->OpType() != "Div" && div_node_unit->OpType() != "Mul")) {
    return nullptr;
  }

  const auto& div_inputs = div_node_unit->Inputs();

  const GeluPatternMatchContext match_ctx{qnn_model_wrapper,
                                          node_to_node_unit,
                                          node_unit_to_qnn_node_group,
                                          logger,
                                          div_inputs[0].name};

  GeluPatternMatchResult pattern_match;
  bool is_match = false;

  const auto& erf_outputs = erf_node_unit.Outputs();
  if (erf_outputs.empty()) {
    return nullptr;
  }

  const OrtNodeUnit* erf_child_node_unit = GetOnlyChildOfOutput(qnn_model_wrapper,
                                                                erf_node_unit,
                                                                erf_outputs[0],
                                                                node_to_node_unit,
                                                                node_unit_to_qnn_node_group);
  if (WarnAndFailOnStandaloneQdq(erf_child_node_unit, erf_node_unit, logger, "GetOnlyChildOfOutput")) {
    return nullptr;
  }

  if (erf_child_node_unit == nullptr) {
    return nullptr;
  }

  if (erf_child_node_unit->OpType() == "Mul") {
    is_match = TryMatchErfMulPattern(div_node_unit,
                                     erf_node_unit,
                                     match_ctx,
                                     pattern_match);
  } else if (erf_child_node_unit->OpType() == "Add") {
    const OrtNodeUnit* add_node_unit = erf_child_node_unit;
    const auto& add_outputs = add_node_unit->Outputs();
    if (add_outputs.empty()) {
      return nullptr;
    }

    const OrtNodeUnit* mul_node_unit = GetOnlyChildOfOutput(qnn_model_wrapper,
                                                            *add_node_unit,
                                                            add_outputs[0],
                                                            node_to_node_unit,
                                                            node_unit_to_qnn_node_group);
    if (WarnAndFailOnStandaloneQdq(mul_node_unit, erf_node_unit, logger, "GetOnlyChildOfOutput")) {
      return nullptr;
    }

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

  // Build-time GELU should remain float-in/float-out so surrounding DQ/Q nodes can be lowered separately.
  if (div_inputs.empty()) {
    return nullptr;
  }

  // Copy IODef values to avoid dangling references from temporary vectors returned by Inputs()/Outputs().
  OrtNodeUnitIODef root_input = div_inputs[0];
  OrtNodeUnitIODef validate_root_input = root_input;

  const OrtNodeUnit* root_input_parent = GetParentOfInput(qnn_model_wrapper,
                                                          *div_node_unit,
                                                          div_inputs[0],
                                                          node_to_node_unit,
                                                          node_unit_to_qnn_node_group);
  if (root_input_parent != nullptr &&
      root_input_parent->OpType() == DEQUANTIZE_LINEAR) {
    const auto& parent_inputs = root_input_parent->Inputs();
    if (parent_inputs.empty()) {
      return nullptr;
    }
    validate_root_input = parent_inputs[0];
  }

  if (pattern_match.final_mul_node_unit == nullptr) {
    return nullptr;
  }

  const auto& final_mul_outputs = pattern_match.final_mul_node_unit->Outputs();
  if (final_mul_outputs.empty()) {
    return nullptr;
  }

  OrtNodeUnitIODef final_output = final_mul_outputs[0];
  OrtNodeUnitIODef validate_final_output = final_output;

  const OrtNodeUnit* final_output_child = GetOnlyChildOfOutput(qnn_model_wrapper,
                                                               *pattern_match.final_mul_node_unit,
                                                               final_mul_outputs[0],
                                                               node_to_node_unit,
                                                               node_unit_to_qnn_node_group);
  if (final_output_child != nullptr &&
      final_output_child->OpType() == QUANTIZE_LINEAR) {
    const auto& child_outputs = final_output_child->Outputs();
    if (child_outputs.empty()) {
      return nullptr;
    }
    validate_final_output = child_outputs[0];
  }

  Ort::Status status = ValidateOnQnn(qnn_model_wrapper,
                                     pattern_match.node_units,
                                     validate_root_input,
                                     validate_final_output);
  if (!status.IsOK()) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_WARNING,
                ("[GeluFusion] ValidateOnQnn failed for GELU pattern (target Erf='" + erf_node_unit.Name() +
                 "'): " + std::string(status.GetErrorMessage()))
                    .c_str());
    return nullptr;
  }

  return std::make_unique<GeluFusion>(std::move(pattern_match.node_units),
                                      &erf_node_unit,
                                      validate_root_input,
                                      validate_final_output,
                                      root_input,
                                      final_output);
}

GeluFusion::GeluFusion(std::vector<const OrtNodeUnit*>&& node_units,
                       const OrtNodeUnit* target_node_unit,
                       OrtNodeUnitIODef validation_root_input,
                       OrtNodeUnitIODef validation_final_output,
                       OrtNodeUnitIODef gelu_root_input,
                       OrtNodeUnitIODef gelu_final_output)
    : node_units_(std::move(node_units)),
      target_node_unit_(target_node_unit),
      validation_root_input_(std::move(validation_root_input)),
      validation_final_output_(std::move(validation_final_output)),
      gelu_root_input_(std::move(gelu_root_input)),
      gelu_final_output_(std::move(gelu_final_output)) {
}

Ort::Status GeluFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, node_units_, validation_root_input_, validation_final_output_);
}

Ort::Status GeluFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, node_units_, gelu_root_input_, gelu_final_output_);
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
