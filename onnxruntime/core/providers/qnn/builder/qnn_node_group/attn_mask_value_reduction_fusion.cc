// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/attn_mask_value_reduction_fusion.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime::qnn {

constexpr double kReplacementMaskMagnitude = 100.0;

struct AttnMaskValueReductionFusion::Match {
  const OrtNodeUnit* softmax = nullptr;
  const OrtNodeUnit* gated_add = nullptr;
  const OrtNodeUnit* masked_add = nullptr;
  const OrtNodeUnit* mask_mul = nullptr;
  const OrtNodeUnitIODef* mask_bias = nullptr;
  double original_mask_magnitude = 0.0;
  bool owns_shared_mask_chain = false;
  std::vector<const OrtNodeUnit*> node_units;
};

namespace {

bool IsBinary(const OrtNodeUnit* node, const char* op_type) {
  return node != nullptr && node->OpType() == op_type && node->Inputs().size() == 2 && node->Outputs().size() == 1;
}

bool IsRank4(const OrtNodeUnitIODef& io) {
  return io.shape.has_value() && io.shape->size() == 4;
}

bool SameShape(const OrtNodeUnitIODef& lhs, const OrtNodeUnitIODef& rhs) {
  return lhs.shape.has_value() && rhs.shape.has_value() && *lhs.shape == *rhs.shape;
}

bool MatchAttnMaskValueReductionPattern(
    const QnnModelWrapper& qmw, const OrtNodeUnit& softmax,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    AttnMaskValueReductionFusion::Match& match) {
  static const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*> no_claims;
  match = {};
  if (softmax.OpType() != "Softmax" || softmax.Inputs().size() != 1 || softmax.Outputs().size() != 1 ||
      !IsRank4(softmax.Inputs()[0]) || !SameShape(softmax.Inputs()[0], softmax.Outputs()[0])) {
    return false;
  }
  const OrtNodeAttrHelper attrs(softmax);
  const int64_t axis = attrs.Get("axis", static_cast<int64_t>(-1));
  if (axis != -1 && axis != 3) return false;

  const OrtNodeUnit* gated_add = GetParentOfInput(qmw, softmax, softmax.Inputs()[0], node_to_node_unit,
                                                  node_unit_to_qnn_node_group);
  if (!IsBinary(gated_add, "Add")) return false;
  const OrtNodeUnit* lhs = GetParentOfInput(qmw, *gated_add, gated_add->Inputs()[0], node_to_node_unit,
                                            node_unit_to_qnn_node_group);
  const OrtNodeUnit* rhs = GetParentOfInput(qmw, *gated_add, gated_add->Inputs()[1], node_to_node_unit,
                                            node_unit_to_qnn_node_group);
  const OrtNodeUnit* masked_add = nullptr;
  const OrtNodeUnit* gate = nullptr;
  if (IsBinary(lhs, "Add") && IsBinary(rhs, "Mul")) {
    masked_add = lhs;
    gate = rhs;
  } else if (IsBinary(lhs, "Mul") && IsBinary(rhs, "Add")) {
    masked_add = rhs;
    gate = lhs;
  }
  if (masked_add == nullptr || gate == nullptr || !IsRank4(gate->Outputs()[0]) ||
      !SameShape(gate->Outputs()[0], softmax.Inputs()[0])) {
    return false;
  }

  const OrtNodeUnit* masked_lhs =
      GetParentOfInput(qmw, *masked_add, masked_add->Inputs()[0], node_to_node_unit, no_claims);
  const OrtNodeUnit* masked_rhs =
      GetParentOfInput(qmw, *masked_add, masked_add->Inputs()[1], node_to_node_unit, no_claims);
  const OrtNodeUnit* score = nullptr;
  const OrtNodeUnit* mask_mul = nullptr;
  if (IsBinary(masked_lhs, "Div") && IsBinary(masked_rhs, "Mul")) {
    score = masked_lhs;
    mask_mul = masked_rhs;
  } else if (IsBinary(masked_lhs, "Mul") && IsBinary(masked_rhs, "Div")) {
    score = masked_rhs;
    mask_mul = masked_lhs;
  }
  if (score == nullptr || mask_mul == nullptr || !IsRank4(score->Outputs()[0]) ||
      !SameShape(score->Outputs()[0], softmax.Inputs()[0])) {
    return false;
  }

  const OrtNodeUnit* mul_lhs = GetParentOfInput(qmw, *mask_mul, mask_mul->Inputs()[0], node_to_node_unit, no_claims);
  const OrtNodeUnit* mul_rhs = GetParentOfInput(qmw, *mask_mul, mask_mul->Inputs()[1], node_to_node_unit, no_claims);
  const OrtNodeUnit* mask_sub = IsBinary(mul_lhs, "Sub") ? mul_lhs : (IsBinary(mul_rhs, "Sub") ? mul_rhs : nullptr);
  if (mask_sub == nullptr || !IsRank4(mask_sub->Outputs()[0]) ||
      !IsStaticQdqInputWithValue(qmw, *mask_sub, 0, node_to_node_unit, 1.0, /*require_scalar=*/true) ||
      qmw.IsConstantInput(mask_sub->Inputs()[1].name) || !IsRank4(mask_sub->Inputs()[1])) {
    return false;
  }
  const size_t mask_bias_index = mul_lhs == mask_sub ? 1 : 0;
  const OrtNodeUnitIODef& mask_bias_input = mask_mul->Inputs()[mask_bias_index];
  const auto mask_bias_value =
      GetStaticQdqInputValue(qmw, *mask_mul, mask_bias_index, node_to_node_unit, /*require_scalar=*/false);
  if (!mask_bias_value.has_value() || !std::isfinite(*mask_bias_value) || *mask_bias_value >= 0.0) {
    return false;
  }
  const double original_mask_magnitude = -*mask_bias_value;

  const OrtNodeUnitIODef& attention_mask = mask_sub->Inputs()[1];
  if (!SameShape(mask_sub->Outputs()[0], attention_mask) || attention_mask.shape->at(0) != softmax.Inputs()[0].shape->at(0) ||
      attention_mask.shape->at(3) != softmax.Inputs()[0].shape->at(3)) {
    return false;
  }
  TensorInfo mask_info = {};
  float mask_scale = 0.0f;
  int32_t mask_offset = 0;
  if (!qmw.GetTensorInfo(attention_mask, mask_info).IsOK() || mask_info.qnn_data_type != QNN_DATATYPE_UFIXED_POINT_16 ||
      !mask_info.quant_param.IsPerTensor() || !mask_info.quant_param.GetPerTensorScaleOffset(mask_scale, mask_offset).IsOK() ||
      mask_scale <= 0.0f) {
    return false;
  }

  const bool owns_shared_mask_chain = node_unit_to_qnn_node_group.find(mask_mul) == node_unit_to_qnn_node_group.end();
  match = {&softmax, gated_add, masked_add, mask_mul, &mask_bias_input, original_mask_magnitude, owns_shared_mask_chain, {}};
  return true;
}

Ort::Status DeriveReducedMaskValueQuantParams(const QnnModelWrapper& qmw, const OrtNodeUnitIODef& tensor,
                                              double original_mask_magnitude, QnnQuantParamsWrapper& replacement) {
  QnnQuantParamsWrapper original;
  // Use the QDQ encoding, not a previously installed override from a shared mask chain.
  RETURN_IF_ERROR(original.Init(qmw, tensor));
  float old_scale = 0.0f;
  int32_t old_offset = 0;
  RETURN_IF_NOT(original.IsPerTensor() &&
                    original.GetPerTensorScaleOffset(old_scale, old_offset).IsOK() &&
                    old_scale > 0.0f,
                "AttnMaskValueReduction fusion requires a positive per-tensor output encoding.");

  float replacement_scale = 0.0f;
  int32_t replacement_offset = 0;
  const double replacement_min = static_cast<double>(old_offset) * old_scale +
                                 (original_mask_magnitude - kReplacementMaskMagnitude);
  RETURN_IF_NOT(DeriveUInt16EncodingWithMin(old_scale, old_offset, replacement_min,
                                            replacement_scale, replacement_offset),
                "Invalid reduced mask value output quantization range.");
  replacement = QnnQuantParamsWrapper::PerTensor(replacement_scale, replacement_offset);
  return Ort::Status();
}

Ort::Status ValidateUInt16Tensor(const QnnModelWrapper& qmw, const OrtNodeUnitIODef& tensor) {
  TensorInfo tensor_info = {};
  RETURN_IF_ERROR(qmw.GetTensorInfo(tensor, tensor_info));
  RETURN_IF_NOT(tensor_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16,
                "AttnMaskValueReduction fusion supports uint16 tensors only.");
  return Ort::Status();
}

Ort::Status CreateAttnMaskValueReductionGraph(QnnModelWrapper& qmw,
                                              const AttnMaskValueReductionFusion::Match& match,
                                              bool validate_only,
                                              const Ort::Logger& logger) {
  RETURN_IF_NOT(match.mask_bias != nullptr, "AttnMaskValueReduction fusion has no mask-bias input.");
  RETURN_IF_NOT(match.original_mask_magnitude > 0.0, "AttnMaskValueReduction fusion requires a negative mask value.");
  for (const OrtNodeUnitIODef* tensor : {match.mask_bias, &match.mask_mul->Outputs()[0],
                                         &match.masked_add->Outputs()[0], &match.gated_add->Outputs()[0]}) {
    RETURN_IF_ERROR(ValidateUInt16Tensor(qmw, *tensor));
  }

  float mask_scale = 0.0f;
  int32_t mask_offset = 0;
  QnnQuantParamsWrapper original_mask_quant;
  RETURN_IF_ERROR(original_mask_quant.Init(qmw, *match.mask_bias));
  RETURN_IF_NOT(original_mask_quant.IsPerTensor() &&
                    original_mask_quant.GetPerTensorScaleOffset(mask_scale, mask_offset).IsOK() &&
                    mask_scale > 0.0f,
                "AttnMaskValueReduction fusion requires a per-tensor mask-bias encoding.");
  const QnnQuantParamsWrapper replacement_mask_quant =
      QnnQuantParamsWrapper::PerTensor(
          static_cast<float>(static_cast<double>(mask_scale) * kReplacementMaskMagnitude /
                             match.original_mask_magnitude),
          mask_offset);
  QnnQuantParamsWrapper mask_mul_quant;
  QnnQuantParamsWrapper masked_add_quant;
  QnnQuantParamsWrapper gated_add_quant;
  RETURN_IF_ERROR(DeriveReducedMaskValueQuantParams(qmw, match.mask_mul->Outputs()[0],
                                                    match.original_mask_magnitude, mask_mul_quant));
  RETURN_IF_ERROR(DeriveReducedMaskValueQuantParams(qmw, match.masked_add->Outputs()[0],
                                                    match.original_mask_magnitude, masked_add_quant));
  RETURN_IF_ERROR(DeriveReducedMaskValueQuantParams(qmw, match.gated_add->Outputs()[0],
                                                    match.original_mask_magnitude, gated_add_quant));

  if (!validate_only) {
    // Keep the payload and topology; change encodings only.
    if (match.owns_shared_mask_chain) {
      qmw.SetQuantParamOverride(match.mask_bias->name, replacement_mask_quant);
      qmw.SetQuantParamOverride(match.mask_mul->Outputs()[0].name, mask_mul_quant);
    }
    qmw.SetQuantParamOverride(match.masked_add->Outputs()[0].name, masked_add_quant);
    qmw.SetQuantParamOverride(match.gated_add->Outputs()[0].name, gated_add_quant);
  }

  std::vector<const OrtNodeUnit*> nodes_to_build;
  if (match.owns_shared_mask_chain) {
    nodes_to_build.push_back(match.mask_mul);
  }
  nodes_to_build.insert(nodes_to_build.end(), {match.masked_add, match.gated_add, match.softmax});
  for (const OrtNodeUnit* node : nodes_to_build) {
    const auto* builder = GetOpBuilder(node->OpType());
    RETURN_IF_NOT(builder != nullptr, ("Missing QNN OpBuilder for " + node->OpType()).c_str());
    RETURN_IF_ERROR(builder->AddToModelBuilder(qmw, *node, logger, validate_only));
  }
  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> AttnMaskValueReductionFusion::TryFusion(
    QnnModelWrapper& qmw, const OrtNodeUnit& softmax_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& claimed,
    [[maybe_unused]] const Ort::Logger& logger) {
  auto match = std::make_unique<Match>();
  if (!MatchAttnMaskValueReductionPattern(qmw, softmax_node_unit, node_to_node_unit, claimed, *match)) return nullptr;
  return std::make_unique<AttnMaskValueReductionFusion>(std::move(match));
}

AttnMaskValueReductionFusion::AttnMaskValueReductionFusion(std::unique_ptr<Match> match) : match_(std::move(match)) {
  match_->node_units = {match_->masked_add, match_->gated_add, match_->softmax};
  if (match_->owns_shared_mask_chain) {
    match_->node_units.push_back(match_->mask_mul);
  }
}

AttnMaskValueReductionFusion::~AttnMaskValueReductionFusion() = default;

Ort::Status AttnMaskValueReductionFusion::IsSupported(QnnModelWrapper& qmw,
                                                      const Ort::Logger& logger) const {
  return CreateAttnMaskValueReductionGraph(qmw, *match_, true, logger);
}

Ort::Status AttnMaskValueReductionFusion::AddToModelBuilder(QnnModelWrapper& qmw,
                                                            const Ort::Logger& logger) const {
  return CreateAttnMaskValueReductionGraph(qmw, *match_, false, logger);
}

gsl::span<const OrtNodeUnit* const> AttnMaskValueReductionFusion::GetNodeUnits() const {
  return match_->node_units;
}

const OrtNodeUnit* AttnMaskValueReductionFusion::GetTargetNodeUnit() const {
  return match_->softmax;
}

}  // namespace onnxruntime::qnn
