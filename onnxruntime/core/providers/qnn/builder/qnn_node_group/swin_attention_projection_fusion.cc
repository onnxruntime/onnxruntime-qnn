// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/swin_attention_projection_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "QnnOpDef.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

using Params = SwinAttentionProjectionFusion::Params;
using NodeMap = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using GroupMap = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

constexpr std::array<int64_t, 6> kWindowPerm{0, 1, 3, 2, 4, 5};

struct Match {
  Params params;
  std::vector<const OrtNodeUnit*> claimed;
  const OrtNodeUnit* reverse_reshape = nullptr;
};

std::vector<const OrtNodeUnit*> Children(const OrtNodeUnit& parent,
                                         const NodeMap& node_map,
                                         const GroupMap& group_map) {
  std::vector<const OrtNodeUnit*> result;
  const auto outputs = Ort::ConstNode(&parent.GetNode()).GetOutputs();
  if (outputs.size() != 1 || outputs[0].IsGraphOutput()) return {};
  for (const auto& consumer : outputs[0].GetConsumers()) {
    if (consumer.node == nullptr) continue;
    const auto it = node_map.find(consumer.node);
    if (it == node_map.end() || group_map.count(it->second) != 0 ||
        it->second->UnitType() != OrtNodeUnit::Type::SingleNode) {
      return {};
    }
    if (std::find(result.begin(), result.end(), it->second) == result.end()) {
      result.push_back(it->second);
    }
  }
  return result;
}

const OrtNodeUnit* OnlyChild(const OrtNodeUnit& parent, std::string_view type,
                             const NodeMap& node_map, const GroupMap& group_map) {
  const auto children = Children(parent, node_map, group_map);
  return children.size() == 1 && children[0]->OpType() == type ? children[0] : nullptr;
}

bool Shape(const QnnModelWrapper& qmw, const OrtNodeUnitIODef& def,
           std::vector<uint32_t>& shape) {
  return qmw.GetOnnxShape(def.shape, shape);
}

bool IsTokenShape(const std::vector<uint32_t>& shape, uint32_t channels) {
  return !shape.empty() && shape.back() == channels &&
         std::all_of(shape.begin(), shape.end(), [](uint32_t d) { return d != 0; });
}

std::optional<Match> MatchPattern(const QnnModelWrapper& qmw,
                                  const OrtNodeUnit& mm,
                                  const NodeMap& node_map,
                                  const GroupMap& group_map) {
  if (mm.OpType() != "MatMul" || mm.UnitType() != OrtNodeUnit::Type::SingleNode ||
      mm.Inputs().size() != 2 || !qmw.IsConstantInput(mm.Inputs()[1].name)) {
    return std::nullopt;
  }

  Match m;
  m.claimed.push_back(&mm);

  const OrtNodeUnit* post_cast = OnlyChild(mm, "Cast", node_map, group_map);
  if (post_cast == nullptr) return std::nullopt;
  const OrtNodeUnit* add = OnlyChild(*post_cast, "Add", node_map, group_map);
  if (add == nullptr || add->Inputs().size() != 2) return std::nullopt;
  m.claimed.push_back(post_cast);
  m.claimed.push_back(add);

  const OrtNodeUnitIODef* bias = nullptr;
  for (const auto& input : add->Inputs()) {
    if (qmw.IsConstantInput(input.name)) {
      if (bias != nullptr) return std::nullopt;
      bias = &input;
    }
  }
  if (bias == nullptr) return std::nullopt;

  const OrtNodeUnit* r1 = OnlyChild(*add, "Reshape", node_map, group_map);
  const OrtNodeUnit* transpose = r1 == nullptr ? nullptr : OnlyChild(*r1, "Transpose", node_map, group_map);
  const OrtNodeUnit* r2 = transpose == nullptr ? nullptr : OnlyChild(*transpose, "Reshape", node_map, group_map);
  if (r1 == nullptr || transpose == nullptr || r2 == nullptr) return std::nullopt;

  OrtNodeAttrHelper transpose_attrs(*transpose);
  if (transpose_attrs.Get("perm", std::vector<int64_t>{}) !=
      std::vector<int64_t>(kWindowPerm.begin(), kWindowPerm.end())) {
    return std::nullopt;
  }

  std::vector<uint32_t> chain_in, rank6, reverse_out;
  if (!Shape(qmw, r1->Inputs()[0], chain_in) || !Shape(qmw, r1->Outputs()[0], rank6) ||
      !Shape(qmw, r2->Outputs()[0], reverse_out) || rank6.size() != 6 ||
      chain_in.size() != 3 || reverse_out.size() != 4) {
    return std::nullopt;
  }

  Params& p = m.params;
  p.batch = rank6[0];
  p.gh = rank6[1];
  p.gw = rank6[2];
  p.window_size = rank6[3];
  p.channels = rank6[5];
  if (p.window_size == 0 || rank6[4] != p.window_size || p.channels == 0 ||
      chain_in != std::vector<uint32_t>{p.batch * p.gh * p.gw,
                                        p.window_size * p.window_size, p.channels} ||
      reverse_out != std::vector<uint32_t>{p.batch, p.gh * p.window_size,
                                           p.gw * p.window_size, p.channels}) {
    return std::nullopt;
  }

  std::vector<uint32_t> weight_shape, bias_shape, mm_input_shape;
  if (!Shape(qmw, mm.Inputs()[1], weight_shape) ||
      weight_shape != std::vector<uint32_t>{p.channels, p.channels} ||
      !Shape(qmw, *bias, bias_shape) || bias_shape.empty() || bias_shape.back() != p.channels ||
      !Shape(qmw, mm.Inputs()[0], mm_input_shape) || !IsTokenShape(mm_input_shape, p.channels)) {
    return std::nullopt;
  }
  for (size_t i = 0; i + 1 < bias_shape.size(); ++i) {
    if (bias_shape[i] != 1) return std::nullopt;
  }

  const OrtNodeUnit* pre_cast = GetParentOfType(qmw, mm, std::array<std::string_view, 1>{"Cast"},
                                                node_map, group_map);
  if (pre_cast == nullptr || OnlyChild(*pre_cast, "MatMul", node_map, group_map) != &mm) {
    return std::nullopt;
  }
  p.chain_input_name = pre_cast->Inputs()[0].name;
  p.chain_input_def = &pre_cast->Inputs()[0];
  m.claimed.insert(m.claimed.begin(), pre_cast);

  TensorInfo chain_input_info{}, mm_input_info{}, weight_info{}, mm_info{};
  TensorInfo post_cast_info{}, bias_info{}, out_info{};
  if (!qmw.GetTensorInfo(pre_cast->Inputs()[0], chain_input_info).IsOK() ||
      !qmw.GetTensorInfo(mm.Inputs()[0], mm_input_info).IsOK() ||
      !qmw.GetTensorInfo(mm.Inputs()[1], weight_info).IsOK() ||
      !qmw.GetTensorInfo(mm.Outputs()[0], mm_info).IsOK() ||
      !qmw.GetTensorInfo(post_cast->Outputs()[0], post_cast_info).IsOK() ||
      !qmw.GetTensorInfo(*bias, bias_info).IsOK() ||
      !qmw.GetTensorInfo(add->Outputs()[0], out_info).IsOK()) {
    return std::nullopt;
  }

  // Match the mixed-precision island used by the Swin projection exactly. This also
  // guarantees that the emitted FC and bias Add have homogeneous operand dtypes.
  if (chain_input_info.qnn_data_type != QNN_DATATYPE_FLOAT_16 ||
      mm_input_info.qnn_data_type != QNN_DATATYPE_FLOAT_32 ||
      weight_info.qnn_data_type != QNN_DATATYPE_FLOAT_32 ||
      mm_info.qnn_data_type != QNN_DATATYPE_FLOAT_32 ||
      post_cast_info.qnn_data_type != QNN_DATATYPE_FLOAT_16 ||
      bias_info.qnn_data_type != QNN_DATATYPE_FLOAT_16 ||
      out_info.qnn_data_type != QNN_DATATYPE_FLOAT_16 ||
      chain_input_info.quant_param.IsQuantized() || mm_info.quant_param.IsQuantized() ||
      out_info.quant_param.IsQuantized()) {
    return std::nullopt;
  }
  p.matmul_dtype = QNN_DATATYPE_FLOAT_32;
  p.output_dtype = QNN_DATATYPE_FLOAT_16;
  p.weight_def = &mm.Inputs()[1];
  p.bias_def = bias;
  p.reverse_anchor = r1;
  p.reverse_output_name = r2->Outputs()[0].name;
  m.reverse_reshape = r1;
  m.claimed.insert(m.claimed.end(), {r1, transpose, r2});

  // Walk the Slice/Concat DAG after window reverse. The unique terminal must be a depad
  // Slice followed by the final token Reshape. Branching is allowed only inside this DAG.
  std::vector<const OrtNodeUnit*> work = Children(*r2, node_map, group_map);
  std::unordered_set<const OrtNodeUnit*> seen;
  const OrtNodeUnit* final_reshape = nullptr;
  const OrtNodeUnit* depad = nullptr;
  while (!work.empty()) {
    const OrtNodeUnit* node = work.back();
    work.pop_back();
    if (!seen.insert(node).second) continue;
    if (node->OpType() != "Slice" && node->OpType() != "Concat") return std::nullopt;

    std::vector<uint32_t> out_shape;
    if (!Shape(qmw, node->Outputs()[0], out_shape) || !IsTokenShape(out_shape, p.channels) ||
        qmw.IsGraphOutput(node->Outputs()[0].name)) {
      return std::nullopt;
    }

    const auto children = Children(*node, node_map, group_map);
    if (children.size() == 1 && children[0]->OpType() == "Reshape") {
      if (node->OpType() != "Slice" || final_reshape != nullptr) return std::nullopt;
      depad = node;
      final_reshape = children[0];
      p.depad_shape = out_shape;
      continue;
    }
    if (children.empty()) return std::nullopt;
    work.insert(work.end(), children.begin(), children.end());
  }
  if (depad == nullptr || final_reshape == nullptr ||
      !Shape(qmw, final_reshape->Outputs()[0], p.final_shape) ||
      !IsTokenShape(p.final_shape, p.channels)) {
    return std::nullopt;
  }

  p.depad_output_name = depad->Outputs()[0].name;
  p.final_output_name = final_reshape->Outputs()[0].name;
  p.post_reverse_nodes.assign(seen.begin(), seen.end());
  std::sort(p.post_reverse_nodes.begin(), p.post_reverse_nodes.end(),
            [](const OrtNodeUnit* a, const OrtNodeUnit* b) { return a->Index() < b->Index(); });
  m.claimed.insert(m.claimed.end(), p.post_reverse_nodes.begin(), p.post_reverse_nodes.end());
  m.claimed.push_back(final_reshape);

  // Every crossed node must preserve complete tokens. The depad must actually reduce the
  // spatial/token count; otherwise there is no reason to perform this more invasive fusion.
  uint64_t before = 1, after = 1;
  for (size_t i = 0; i + 1 < reverse_out.size(); ++i) before *= reverse_out[i];
  for (size_t i = 0; i + 1 < p.depad_shape.size(); ++i) after *= p.depad_shape[i];
  if (after >= before) return std::nullopt;

  return m;
}

Ort::Status EmitReverse(QnnModelWrapper& qmw, const OrtNodeUnit& anchor,
                        const Params& p, bool validate) {
  RETURN_IF_NOT(p.chain_input_def != nullptr, "Swin fusion has no chain input definition.");
  if (!qmw.IsQnnTensorWrapperExist(p.chain_input_name)) {
    QnnTensorWrapper wrapper;
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(*p.chain_input_def, wrapper));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(wrapper)), "Failed to add Swin fusion input.");
  }

  const uint32_t B = p.batch, Gh = p.gh, Gw = p.gw, ws = p.window_size, C = p.channels;
  const uint32_t H = Gh * ws, W = Gw * ws;
  const size_t index = anchor.Index();
  auto name = [&p](std::string_view suffix) {
    return utils::UniqueNameGenerator().New(p.final_output_name, std::string(suffix));
  };
  const std::vector<uint32_t> perm{0, 2, 1, 3};
  const QnnQuantParamsWrapper qp;

  const std::string r1 = name("_swin_wr_r1");
  RETURN_IF_ERROR(qmw.AddReshapeNode(p.chain_input_name, r1,
                                     {B * Gh * Gw, ws * ws, C}, {B, Gh, Gw, ws * ws * C},
                                     p.output_dtype, qp, validate, false, false));
  const std::string t1 = name("_swin_wr_t1");
  RETURN_IF_ERROR(qmw.AddTransposeNode(index, r1, t1, {B, Gh, Gw, ws * ws * C}, perm,
                                       {B, Gw, Gh, ws * ws * C}, p.output_dtype, qp,
                                       validate, false, false));
  const std::string r2 = name("_swin_wr_r2");
  RETURN_IF_ERROR(qmw.AddReshapeNode(t1, r2, {B, Gw, Gh, ws * ws * C},
                                     {B, Gw, H, ws * C}, p.output_dtype, qp,
                                     validate, false, false));
  const std::string t2 = name("_swin_wr_t2");
  RETURN_IF_ERROR(qmw.AddTransposeNode(index, r2, t2, {B, Gw, H, ws * C}, perm,
                                       {B, H, Gw, ws * C}, p.output_dtype, qp,
                                       validate, false, false));
  return qmw.AddReshapeNode(t2, p.reverse_output_name, {B, H, Gw, ws * C},
                            {B, H, W, C}, p.output_dtype, qp, validate, false, false);
}

Ort::Status EmitProjection(QnnModelWrapper& qmw, const Params& p,
                           const Ort::Logger& logger, bool validate) {
  auto name = [&p](std::string_view suffix) {
    return utils::UniqueNameGenerator().New(p.final_output_name, std::string(suffix));
  };
  uint32_t tokens = 1;
  for (size_t i = 0; i + 1 < p.depad_shape.size(); ++i) tokens *= p.depad_shape[i];
  const std::vector<uint32_t> flat_shape{tokens, p.channels};
  const QnnQuantParamsWrapper qp;

  const std::string flat = name("_swin_proj_flat");
  RETURN_IF_ERROR(qmw.AddReshapeNode(p.depad_output_name, flat, p.depad_shape, flat_shape,
                                     p.output_dtype, qp, validate, false, false));

  std::vector<uint32_t> weight_shape;
  RETURN_IF_NOT(qmw.GetOnnxShape(p.weight_def->shape, weight_shape), "Missing projection weight shape.");
  const OrtValueInfo* weight = qmw.GetConstantTensor(p.weight_def->name);
  RETURN_IF_NOT(weight != nullptr, "Projection weight is not constant.");
  std::vector<uint8_t> weight_data;
  RETURN_IF_ERROR(utils::TwoDimensionTranspose(qmw, weight_shape, weight, weight_data, logger, validate));
  std::reverse(weight_shape.begin(), weight_shape.end());
  const std::string weight_name = name("_swin_proj_weight");
  if (!qmw.IsQnnTensorWrapperExist(weight_name)) {
    QnnTensorWrapper wrapper(weight_name, QNN_TENSOR_TYPE_STATIC, p.matmul_dtype,
                             QnnQuantParamsWrapper(), std::move(weight_shape),
                             std::move(weight_data));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(wrapper)), "Failed to add projection weight.");
  }

  const std::string fc_input = name("_swin_proj_input");
  RETURN_IF_ERROR(qmw.AddCastNode(name("_swin_proj_precast"), flat, fc_input,
                                  QNN_TENSOR_TYPE_NATIVE, p.matmul_dtype,
                                  QnnQuantParamsWrapper(), std::vector<uint32_t>(flat_shape),
                                  validate));

  const std::string zero_bias = name("_swin_proj_zero_bias");
  if (!qmw.IsQnnTensorWrapperExist(zero_bias)) {
    const size_t element_size = p.matmul_dtype == QNN_DATATYPE_FLOAT_32 ? 4u : 2u;
    QnnTensorWrapper wrapper(zero_bias, QNN_TENSOR_TYPE_STATIC, p.matmul_dtype,
                             QnnQuantParamsWrapper(), std::vector<uint32_t>{p.channels},
                             std::vector<uint8_t>(p.channels * element_size, 0));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(wrapper)), "Failed to add zero FC bias.");
  }

  const std::string fc_output = name("_swin_proj_fc");
  QnnTensorWrapper fc_wrapper(fc_output, QNN_TENSOR_TYPE_NATIVE, p.matmul_dtype,
                              QnnQuantParamsWrapper(), std::vector<uint32_t>(flat_shape),
                              std::vector<uint8_t>());
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(fc_wrapper)), "Failed to add FC output.");
  RETURN_IF_NOT(qmw.CreateQnnNode(name("_swin_proj_fc_node"), QNN_OP_PACKAGE_NAME_QTI_AISW,
                                  QNN_OP_FULLY_CONNECTED, {fc_input, weight_name, zero_bias},
                                  {fc_output}, {}, validate), "Failed to create projection FC.");

  const std::string cast_output = name("_swin_proj_postcast");
  RETURN_IF_ERROR(qmw.AddCastNode(name("_swin_proj_postcast_node"), fc_output, cast_output,
                                  QNN_TENSOR_TYPE_NATIVE, p.output_dtype,
                                  QnnQuantParamsWrapper(), std::vector<uint32_t>(flat_shape),
                                  validate));

  if (!qmw.IsQnnTensorWrapperExist(p.bias_def->name)) {
    QnnTensorWrapper bias_wrapper;
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(*p.bias_def, bias_wrapper));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(bias_wrapper)), "Failed to add projection bias.");
  }
  const std::string add_output = name("_swin_proj_add");
  QnnTensorWrapper add_wrapper(add_output, QNN_TENSOR_TYPE_NATIVE, p.output_dtype,
                               QnnQuantParamsWrapper(), std::vector<uint32_t>(flat_shape),
                               std::vector<uint8_t>());
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(add_wrapper)), "Failed to add projection Add output.");
  RETURN_IF_NOT(qmw.CreateQnnNode(name("_swin_proj_add_node"), QNN_OP_PACKAGE_NAME_QTI_AISW,
                                  QNN_OP_ELEMENT_WISE_ADD, {cast_output, p.bias_def->name},
                                  {add_output}, {}, validate), "Failed to create projection bias Add.");

  return qmw.AddReshapeNode(add_output, p.final_output_name, flat_shape, p.final_shape,
                            p.output_dtype, qp, validate, false,
                            qmw.IsGraphOutput(p.final_output_name));
}

Ort::Status Emit(QnnModelWrapper& qmw, const OrtNodeUnit& reverse_anchor,
                 const Params& p, const Ort::Logger& logger, bool validate) {
  RETURN_IF_ERROR(EmitReverse(qmw, reverse_anchor, p, validate));
  for (const OrtNodeUnit* node : p.post_reverse_nodes) {
    const IOpBuilder* builder = GetOpBuilder(node->OpType());
    RETURN_IF_NOT(builder != nullptr, "Missing Slice/Concat builder for Swin fusion.");
    RETURN_IF_ERROR(builder->AddToModelBuilder(qmw, *node, logger, validate));
  }
  return EmitProjection(qmw, p, logger, validate);
}

}  // namespace

SwinAttentionProjectionFusion::SwinAttentionProjectionFusion(
    gsl::span<const OrtNodeUnit* const> node_units, Params params)
    : node_units_(node_units.begin(), node_units.end()), params_(std::move(params)) {}

std::unique_ptr<IQnnNodeGroup> SwinAttentionProjectionFusion::TryFusion(
    QnnModelWrapper& qmw, const OrtNodeUnit& matmul, const NodeMap& node_map,
    const GroupMap& group_map, const Ort::Logger& logger) {
  if (!IsNpuBackend(qmw.GetQnnBackendType())) return nullptr;
  auto match = MatchPattern(qmw, matmul, node_map, group_map);
  if (!match.has_value()) return nullptr;
  if (!Emit(qmw, *match->reverse_reshape, match->params, logger, true).IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("SwinAttentionProjectionFusion: QNN validation failed at '" + matmul.Name() + "'.").c_str());
    return nullptr;
  }
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
              ("SwinAttentionProjectionFusion: moved projection after depad at '" + matmul.Name() + "'.").c_str());
  return std::make_unique<SwinAttentionProjectionFusion>(
      gsl::span<const OrtNodeUnit* const>{match->claimed.data(), match->claimed.size()},
      std::move(match->params));
}

gsl::span<const OrtNodeUnit* const> SwinAttentionProjectionFusion::GetNodeUnits() const {
  return {node_units_.data(), node_units_.size()};
}

Ort::Status SwinAttentionProjectionFusion::IsSupported(QnnModelWrapper& qmw,
                                                        const Ort::Logger& logger) const {
  RETURN_IF_NOT(params_.reverse_anchor != nullptr, "Swin fusion has no reverse anchor.");
  return Emit(qmw, *params_.reverse_anchor, params_, logger, true);
}

Ort::Status SwinAttentionProjectionFusion::AddToModelBuilder(QnnModelWrapper& qmw,
                                                              const Ort::Logger& logger) const {
  RETURN_IF_NOT(params_.reverse_anchor != nullptr, "Swin fusion has no reverse anchor.");
  return Emit(qmw, *params_.reverse_anchor, params_, logger, false);
}

}  // namespace qnn
}  // namespace onnxruntime
