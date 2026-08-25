// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/masked_softmax_fusion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime::qnn {
namespace {

const OrtNodeUnit* ParentForInput(const QnnModelWrapper& qmw, const OrtNodeUnit& child, size_t index,
                                  const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_map,
                                  const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& claimed) {
  return index < child.Inputs().size() ? GetParentOfInput(qmw, child, child.Inputs()[index], node_map, claimed) : nullptr;
}

const OrtNodeUnit* ParentForInputIgnoringClaims(
    const QnnModelWrapper& qmw, const OrtNodeUnit& child, size_t index,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_map) {
  static const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*> no_claims;
  return index < child.Inputs().size()
             ? GetParentOfInput(qmw, child, child.Inputs()[index], node_map, no_claims)
             : nullptr;
}

bool IsBinaryAdd(const OrtNodeUnit* node) { return node != nullptr && node->OpType() == "Add" && node->Inputs().size() == 2; }
bool IsRank4(const OrtNodeUnitIODef& io) { return io.shape.has_value() && io.shape->size() == 4; }
bool SameShape(const OrtNodeUnitIODef& lhs, const OrtNodeUnitIODef& rhs) {
  return lhs.shape.has_value() && rhs.shape.has_value() && *lhs.shape == *rhs.shape;
}

bool MatchPatternA(const QnnModelWrapper& qmw, const OrtNodeUnit& softmax,
                   const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_map,
                   const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& claimed,
                   MaskedSoftmaxPatternAMatch& match) {
  match = {};
  if (softmax.OpType() != "Softmax" || softmax.Inputs().size() != 1 || softmax.Outputs().size() != 1 ||
      !IsRank4(softmax.Inputs()[0]) || !SameShape(softmax.Inputs()[0], softmax.Outputs()[0])) return false;
  const OrtNodeAttrHelper attrs(softmax);
  const int64_t axis = attrs.Get("axis", static_cast<int64_t>(-1));
  if (axis != -1 && axis != 3) return false;
  const OrtNodeUnit* gated_add = ParentForInput(qmw, softmax, 0, node_map, claimed);
  if (!IsBinaryAdd(gated_add)) return false;
  const OrtNodeUnit* lhs = ParentForInput(qmw, *gated_add, 0, node_map, claimed);
  const OrtNodeUnit* rhs = ParentForInput(qmw, *gated_add, 1, node_map, claimed);
  if (lhs == nullptr || rhs == nullptr) return false;
  const OrtNodeUnit* masked_add = nullptr;
  const OrtNodeUnit* gate = nullptr;
  if (lhs->OpType() == "Add" && rhs->OpType() == "Mul") {
    masked_add = lhs;
    gate = rhs;
  } else if (lhs->OpType() == "Mul" && rhs->OpType() == "Add") {
    masked_add = rhs;
    gate = lhs;
  }
  if (!IsBinaryAdd(masked_add) || gate == nullptr || gate->Outputs().size() != 1 ||
      !IsRank4(gate->Outputs()[0]) || !SameShape(gate->Outputs()[0], softmax.Inputs()[0])) return false;
  // The PSL mask-bias chain is shared by all 12 layers. Once the first
  // PatternA group claims it, later layers still need to recognize the same
  // source graph in order to build their own direct-mask pattern.
  const OrtNodeUnit* mask_lhs = ParentForInputIgnoringClaims(qmw, *masked_add, 0, node_map);
  const OrtNodeUnit* mask_rhs = ParentForInputIgnoringClaims(qmw, *masked_add, 1, node_map);
  if (mask_lhs == nullptr || mask_rhs == nullptr) return false;
  const OrtNodeUnit* score = nullptr;
  const OrtNodeUnit* additive_mask = nullptr;
  if (mask_lhs->OpType() == "Div" && mask_rhs->OpType() == "Mul") {
    score = mask_lhs;
    additive_mask = mask_rhs;
  } else if (mask_lhs->OpType() == "Mul" && mask_rhs->OpType() == "Div") {
    score = mask_rhs;
    additive_mask = mask_lhs;
  }
  if (score == nullptr || additive_mask == nullptr || score->Outputs().size() != 1 ||
      !IsRank4(score->Outputs()[0]) || !SameShape(score->Outputs()[0], softmax.Inputs()[0])) return false;
  const OrtNodeUnit* additive_mask_lhs = ParentForInputIgnoringClaims(qmw, *additive_mask, 0, node_map);
  const OrtNodeUnit* additive_mask_rhs = ParentForInputIgnoringClaims(qmw, *additive_mask, 1, node_map);
  const OrtNodeUnit* mask_sub = nullptr;
  if (additive_mask_lhs != nullptr && additive_mask_lhs->OpType() == "Sub") {
    mask_sub = additive_mask_lhs;
  } else if (additive_mask_rhs != nullptr && additive_mask_rhs->OpType() == "Sub") {
    mask_sub = additive_mask_rhs;
  }
  if (mask_sub == nullptr || mask_sub->Outputs().size() != 1 || !IsRank4(mask_sub->Outputs()[0]) ||
      mask_sub->Inputs().size() != 2 ||
      !IsStaticQdqInputWithValue(qmw, *mask_sub, 0, node_map, 1.0, /*require_scalar=*/true) ||
      qmw.IsConstantInput(mask_sub->Inputs()[1].name) || !IsRank4(mask_sub->Inputs()[1])) {
    return false;
  }
  const OrtNodeUnitIODef* attention_mask = &mask_sub->Inputs()[1];
  if (!attention_mask->shape.has_value() || attention_mask->shape->size() != 4 ||
      !softmax.Inputs()[0].shape.has_value() || softmax.Inputs()[0].shape->size() != 4 ||
      (*attention_mask->shape)[0] != (*softmax.Inputs()[0].shape)[0] ||
      (*attention_mask->shape)[3] != (*softmax.Inputs()[0].shape)[3]) {
    return false;
  }
  TensorInfo attention_mask_info = {};
  float mask_scale = 0.0f;
  int32_t mask_offset = 0;
  if (!qmw.GetTensorInfo(*attention_mask, attention_mask_info).IsOK() ||
      attention_mask_info.qnn_data_type != QNN_DATATYPE_UFIXED_POINT_16 ||
      !attention_mask_info.quant_param.IsQuantized() || !attention_mask_info.quant_param.IsPerTensor() ||
      !attention_mask_info.quant_param.GetPerTensorScaleOffset(mask_scale, mask_offset).IsOK() ||
      mask_scale <= 0.0f) {
    return false;
  }
  match = {&softmax, gated_add, masked_add, score, additive_mask, mask_sub, attention_mask, gate,
           claimed.find(mask_sub) == claimed.end() && claimed.find(additive_mask) == claimed.end()};
  return true;
}

constexpr uint32_t kUInt16QMax = std::numeric_limits<uint16_t>::max();
constexpr float kFillValue = -25.0f;

struct QuantRange {
  float min;
  float max;
  float scale;
  int32_t offset;
};

Ort::Status GetQuantRange(const QnnTensorWrapper& tensor, QuantRange& range) {
  const auto& params = tensor.GetQnnQuantParams();
  RETURN_IF_NOT(params.IsPerTensor(), "Pattern-A requires per-tensor activation encodings.");
  RETURN_IF_ERROR(params.GetPerTensorScaleOffset(range.scale, range.offset));
  RETURN_IF_NOT(range.scale > 0.0f, "Pattern-A requires a positive activation scale.");
  range.min = static_cast<float>(range.offset) * range.scale;
  range.max = (static_cast<float>(kUInt16QMax) + static_cast<float>(range.offset)) * range.scale;
  return Ort::Status();
}

QuantRange MakeRange(float min, float max) {
  const float scale = (max - min) / static_cast<float>(kUInt16QMax);
  const int32_t zero_point = static_cast<int32_t>(std::clamp(
      std::lround(-min / scale), 0l, static_cast<long>(kUInt16QMax)));
  return {min, max, scale, -zero_point};
}

uint16_t Quantize(float value, const QuantRange& range) {
  const long code = std::lround(value / range.scale - static_cast<float>(range.offset));
  return static_cast<uint16_t>(std::clamp(code, 0l, static_cast<long>(kUInt16QMax)));
}

std::vector<uint8_t> UInt16Bytes(uint16_t value, size_t count) {
  std::vector<uint8_t> bytes(count * sizeof(value));
  for (size_t i = 0; i < count; ++i) {
    std::memcpy(bytes.data() + i * sizeof(value), &value, sizeof(value));
  }
  return bytes;
}

size_t ElementCount(gsl::span<const uint32_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
}

Ort::Status AddBinaryOp(QnnModelWrapper& qmw,
                        const OrtNodeUnit& owner,
                        const std::string& name,
                        const std::string& input0,
                        const std::string& input1,
                        const std::string& output,
                        uint32_t operation,
                        bool do_op_validation) {
  std::vector<std::string> params;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qmw, owner.Index(), name, operation,
                                         QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, params));
  RETURN_IF_NOT(qmw.CreateQnnNode(name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_BINARY,
                                  {input0, input1}, {output}, std::move(params), do_op_validation),
                "Failed to create Pattern-A ElementWiseBinary node.");
  return Ort::Status();
}

Ort::Status AddStaticUInt16(QnnModelWrapper& qmw,
                            const std::string& name,
                            uint16_t value,
                            const QnnTensorWrapper& encoding_source,
                            const std::vector<uint32_t>& shape) {
  QnnTensorWrapper tensor(name, QNN_TENSOR_TYPE_STATIC, encoding_source.GetTensorDataType(),
                          encoding_source.GetQnnQuantParams().Copy(), std::vector<uint32_t>(shape),
                          UInt16Bytes(value, ElementCount(shape)));
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tensor)), "Failed to add Pattern-A static uint16 tensor.");
  return Ort::Status();
}

Ort::Status AddNative(QnnModelWrapper& qmw,
                      const std::string& name,
                      Qnn_DataType_t data_type,
                      QnnQuantParamsWrapper&& quant_params,
                      std::vector<uint32_t>&& shape) {
  QnnTensorWrapper tensor(name, QNN_TENSOR_TYPE_NATIVE, data_type, std::move(quant_params), std::move(shape));
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tensor)), "Failed to add Pattern-A native tensor.");
  return Ort::Status();
}

Ort::Status AddReduceMin(QnnModelWrapper& qmw,
                         const OrtNodeUnit& owner,
                         const std::string& name,
                         const std::string& input,
                         const std::string& output,
                         bool do_op_validation) {
  std::vector<std::string> params;
  QnnParamWrapper axes(owner.Index(), name, QNN_OP_REDUCE_MIN_PARAM_AXES,
                       std::vector<uint32_t>{1}, std::vector<uint32_t>{3});
  params.push_back(axes.GetParamTensorName());
  RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axes)), "Failed to add Pattern-A ReduceMin axes.");
  RETURN_IF_ERROR(AddQnnScalar<bool>(qmw, owner.Index(), name, true,
                                     QNN_OP_REDUCE_MIN_PARAM_KEEP_DIMS, params));
  RETURN_IF_NOT(qmw.CreateQnnNode(name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_REDUCE_MIN,
                                  {input}, {output}, std::move(params), do_op_validation),
                "Failed to create Pattern-A ReduceMin node.");
  return Ort::Status();
}

Ort::Status CreatePatternA(QnnModelWrapper& qmw, const MaskedSoftmaxPatternAMatch& match, bool validate_only) {
  const OrtNodeUnitIODef& score_def = match.score->Outputs()[0];
  const OrtNodeUnitIODef& gate_def = match.gate->Outputs()[0];
  const OrtNodeUnitIODef& mask_def = *match.attention_mask;
  const OrtNodeUnitIODef& softmax_output_def = match.softmax->Outputs()[0];

  QnnTensorWrapper score;
  QnnTensorWrapper gate;
  QnnTensorWrapper softmax_output;
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(score_def, score));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(gate_def, gate));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(softmax_output_def, softmax_output));

  QuantRange score_range;
  QuantRange gate_range;
  RETURN_IF_ERROR(GetQuantRange(score, score_range));
  RETURN_IF_ERROR(GetQuantRange(gate, gate_range));
  RETURN_IF_NOT(score.GetTensorDataType() == QNN_DATATYPE_UFIXED_POINT_16 &&
                    gate.GetTensorDataType() == QNN_DATATYPE_UFIXED_POINT_16,
                "Pattern-A currently supports uint16 score and gate tensors only.");

  QuantRange gated_range = MakeRange(score_range.min + gate_range.min, score_range.max + gate_range.max);
  QuantRange fill_range = MakeRange(gated_range.min + kFillValue, gated_range.max + kFillValue);
  QuantRange masked_range = MakeRange(fill_range.min, gated_range.max);

  const std::string prefix = utils::UniqueNameGenerator().New(*match.softmax, "_patterna");
  const std::string gated = prefix + "_gated";
  const std::string zero = prefix + "_zero";
  const std::string cond = prefix + "_cond";
  const std::string rmin = prefix + "_rmin";
  const std::string fill_const = prefix + "_fill_const";
  const std::string fill = prefix + "_fill";
  const std::string masked = prefix + "_masked";
  const std::string add_gated = prefix + "_GatedAdd";
  const std::string not_equal = prefix + "_NotEqual";
  const std::string reduce = prefix + "_ReduceMin";
  const std::string add_fill = prefix + "_FillAdd";
  const std::string where = prefix + "_Where";
  const std::string softmax = prefix + "_Softmax";

  const std::vector<uint32_t> score_shape = score.GetTensorDims();
  QnnTensorWrapper attention_mask;
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(mask_def, attention_mask));
  RETURN_IF_NOT(attention_mask.GetTensorDataType() == QNN_DATATYPE_UFIXED_POINT_16,
                "Pattern-A currently supports a uint16 attention mask tensor.");
  const std::vector<uint32_t> mask_shape = attention_mask.GetTensorDims();
  RETURN_IF_NOT(score_shape.size() == 4 && mask_shape.size() == 4 && score_shape[0] == mask_shape[0] &&
                    score_shape[3] == mask_shape[3],
                "Pattern-A score/mask shapes are incompatible.");
  const std::vector<uint32_t> rmin_shape{score_shape[0], score_shape[1], score_shape[2], 1};
  RETURN_IF_ERROR(AddNative(qmw, gated, score.GetTensorDataType(),
                            QnnQuantParamsWrapper::PerTensor(gated_range.scale, gated_range.offset),
                            std::vector<uint32_t>(score_shape)));
  RETURN_IF_ERROR(AddNative(qmw, cond, QNN_DATATYPE_BOOL_8, QnnQuantParamsWrapper(),
                            std::vector<uint32_t>(mask_shape)));
  RETURN_IF_ERROR(AddNative(qmw, rmin, score.GetTensorDataType(),
                            QnnQuantParamsWrapper::PerTensor(gated_range.scale, gated_range.offset),
                            std::vector<uint32_t>(rmin_shape)));
  RETURN_IF_ERROR(AddNative(qmw, fill, score.GetTensorDataType(),
                            QnnQuantParamsWrapper::PerTensor(fill_range.scale, fill_range.offset),
                            std::vector<uint32_t>(rmin_shape)));
  RETURN_IF_ERROR(AddNative(qmw, masked, score.GetTensorDataType(),
                            QnnQuantParamsWrapper::PerTensor(masked_range.scale, masked_range.offset),
                            std::vector<uint32_t>(score_shape)));
  QuantRange mask_range;
  RETURN_IF_ERROR(GetQuantRange(attention_mask, mask_range));
  RETURN_IF_ERROR(AddStaticUInt16(qmw, zero, Quantize(0.0f, mask_range), attention_mask, {1}));
  const QnnTensorWrapper gated_encoding(gated, QNN_TENSOR_TYPE_NATIVE, score.GetTensorDataType(),
                                        QnnQuantParamsWrapper::PerTensor(gated_range.scale, gated_range.offset),
                                        std::vector<uint32_t>(score_shape));
  RETURN_IF_ERROR(AddStaticUInt16(qmw, fill_const, Quantize(kFillValue, gated_range), gated_encoding, {1}));

  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(score)), "Failed to add Pattern-A score tensor.");
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(gate)), "Failed to add Pattern-A gate tensor.");
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(attention_mask)), "Failed to add Pattern-A attention mask tensor.");
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(softmax_output)), "Failed to add Pattern-A Softmax output tensor.");
  RETURN_IF_ERROR(AddBinaryOp(qmw, *match.softmax, add_gated, score_def.name, gate_def.name, gated,
                              QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, validate_only));
  RETURN_IF_ERROR(AddBinaryOp(qmw, *match.softmax, not_equal, mask_def.name, zero, cond,
                              QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL, false));
  RETURN_IF_ERROR(AddReduceMin(qmw, *match.softmax, reduce, gated, rmin, validate_only));
  RETURN_IF_ERROR(AddBinaryOp(qmw, *match.softmax, add_fill, rmin, fill_const, fill,
                              QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, validate_only));
  RETURN_IF_NOT(qmw.CreateQnnNode(where, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_SELECT,
                                  {cond, gated, fill}, {masked}, {}, validate_only),
                "Failed to create Pattern-A Where node.");
  std::vector<std::string> softmax_params;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qmw, match.softmax->Index(), softmax, 3,
                                         QNN_OP_SOFTMAX_PARAM_AXIS, softmax_params));
  RETURN_IF_NOT(qmw.CreateQnnNode(softmax, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_SOFTMAX,
                                  {masked}, {softmax_output_def.name}, std::move(softmax_params), validate_only),
                "Failed to create Pattern-A Softmax node.");
  return Ort::Status();
}

}  // namespace

MaskedSoftmaxPatternAFusion::MaskedSoftmaxPatternAFusion(const MaskedSoftmaxPatternAMatch& match)
    : match_(match), node_units_{match.masked_add, match.gated_add, match.softmax} {
  if (match.claims_legacy_mask_chain) {
    node_units_.push_back(match.additive_mask);
    node_units_.push_back(match.mask_sub);
  }
}

std::unique_ptr<IQnnNodeGroup> MaskedSoftmaxPatternAFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& softmax_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  MaskedSoftmaxPatternAMatch match;
  if (!MatchPatternA(qnn_model_wrapper, softmax_node_unit, node_to_node_unit,
                     node_unit_to_qnn_node_group, match)) {
    return nullptr;
  }
  return std::make_unique<MaskedSoftmaxPatternAFusion>(match);
}

gsl::span<const OrtNodeUnit* const> MaskedSoftmaxPatternAFusion::GetNodeUnits() const {
  return node_units_;
}

Ort::Status MaskedSoftmaxPatternAFusion::IsSupported(QnnModelWrapper& qmw,
                                                     [[maybe_unused]] const Ort::Logger& logger) const {
  return CreatePatternA(qmw, match_, true);
}

Ort::Status MaskedSoftmaxPatternAFusion::AddToModelBuilder(QnnModelWrapper& qmw,
                                                           [[maybe_unused]] const Ort::Logger& logger) const {
  return CreatePatternA(qmw, match_, false);
}

}  // namespace onnxruntime::qnn
