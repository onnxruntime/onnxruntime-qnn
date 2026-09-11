// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/qkv_split_attention_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "QnnOpDef.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kOpReshape[] = "Reshape";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kOpGather[] = "Gather";
constexpr char kOpMul[] = "Mul";
constexpr char kOpMatMul[] = "MatMul";
constexpr char kOpAdd[] = "Add";
constexpr char kOpSoftmax[] = "Softmax";

// ---------------------------------------------------------------------------
// Matched sub-graph, resolved during MatchPattern. All pointers are non-owning
// references into the graph's NodeUnits.
// ---------------------------------------------------------------------------
struct MatchedPattern {
  const OrtNodeUnit* head_reshape = nullptr;    // Reshape([B,S,3,n,hs])
  const OrtNodeUnit* head_transpose = nullptr;  // Transpose after the head reshape

  const OrtNodeUnit* gather_q = nullptr;  // Gather(indices=0)
  const OrtNodeUnit* gather_k = nullptr;  // Gather(indices=1)
  const OrtNodeUnit* gather_v = nullptr;  // Gather(indices=2)

  const OrtNodeUnit* q_scale_mul = nullptr;  // Mul(Q, 1/sqrt(hs))
  const OrtNodeUnit* k_transpose = nullptr;  // Transpose on K before Q*K^T
  const OrtNodeUnit* qk_matmul = nullptr;    // MatMul(Q, K^T)

  // Interior mask-add / reshape chain between qk_matmul and softmax. Kept generic:
  // the exact count of Add/Reshape nodes may vary (the reference graph has
  // Add -> Reshape -> Add -> Reshape).
  std::vector<const OrtNodeUnit*> interior;

  const OrtNodeUnit* softmax = nullptr;    // Softmax
  const OrtNodeUnit* av_matmul = nullptr;  // MatMul(softmax, V)

  // Resolved emission parameters (packed-qkv input name + dims + gather output names).
  QkvSplitAttentionFusion::SplitParams params;
};

// Return the single output value-info name of a node unit's target node.
const std::string& FirstOutputName(const OrtNodeUnit& nu) {
  return nu.Outputs()[0].name;
}

// Read the (int) 'indices' scalar of a Gather node unit whose indices input is a
// constant scalar initializer. Returns nullopt if indices is not a constant scalar.
std::optional<int64_t> GetGatherScalarIndex(const QnnModelWrapper& qmw, const OrtNodeUnit& gather) {
  const auto& inputs = gather.Inputs();
  if (inputs.size() < 2) {
    return std::nullopt;
  }
  const std::string& idx_name = inputs[1].name;
  if (!qmw.IsConstantInput(idx_name)) {
    return std::nullopt;
  }
  const OrtValueInfo* vi = qmw.GetConstantTensor(idx_name);
  if (vi == nullptr) {
    return std::nullopt;
  }
  std::vector<uint8_t> bytes;
  if (!qmw.UnpackInitializerData(vi, bytes).IsOK()) {
    return std::nullopt;
  }
  // indices may be int64 or int32 scalar.
  if (bytes.size() == sizeof(int64_t)) {
    int64_t v = 0;
    std::memcpy(&v, bytes.data(), sizeof(int64_t));
    return v;
  }
  if (bytes.size() == sizeof(int32_t)) {
    int32_t v = 0;
    std::memcpy(&v, bytes.data(), sizeof(int32_t));
    return static_cast<int64_t>(v);
  }
  return std::nullopt;
}

// Enumerate all distinct child NodeUnits that consume the single output of `parent`.
// Unlike GetOnlyChildOfType this permits fan-out (the head Transpose feeds 3 Gathers).
// Each returned child must be a standalone SingleNode NodeUnit that has not already been
// claimed by another IQnnNodeGroup.
std::vector<const OrtNodeUnit*> GetAllChildren(
    const OrtNodeUnit& parent,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  std::vector<const OrtNodeUnit*> children;
  const Ort::ConstNode parent_node(&parent.GetNode());
  const std::vector<Ort::ConstValueInfo> outputs = parent_node.GetOutputs();
  if (outputs.size() != 1) {
    return children;
  }
  for (const Ort::ConstValueInfo& out : outputs) {
    if (out.IsGraphOutput()) {
      return {};  // feeding a graph output disqualifies fusion
    }
  }
  for (const Ort::ValueInfoConsumerProducerInfo& consumer : outputs[0].GetConsumers()) {
    if (consumer.node == nullptr) {
      continue;
    }
    const auto it = node_to_node_unit.find(consumer.node);
    if (it == node_to_node_unit.end()) {
      return {};
    }
    const OrtNodeUnit* child = it->second;
    if (node_unit_to_qnn_node_group.count(child) != 0) {
      return {};
    }
    if (child->UnitType() != OrtNodeUnit::Type::SingleNode) {
      return {};
    }
    // Deduplicate (a node could appear twice if it consumes the output on two inputs).
    if (std::find(children.begin(), children.end(), child) == children.end()) {
      children.push_back(child);
    }
  }
  return children;
}

// ---------------------------------------------------------------------------
// Pattern matcher. Returns a populated MatchedPattern on success, else nullopt.
// ---------------------------------------------------------------------------
std::optional<MatchedPattern> MatchPattern(
    const QnnModelWrapper& qmw,
    const OrtNodeUnit& reshape_nu,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  if (reshape_nu.OpType() != kOpReshape ||
      reshape_nu.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  MatchedPattern m;
  m.head_reshape = &reshape_nu;

  // Head Reshape must reshape the packed qkv into rank-5 [N, S, 3, n, hs]. The stacked-QKV
  // axis (index 2) must equal 3; the trailing [n, hs] are heads/head_size.
  std::vector<uint32_t> reshape_out_shape;
  if (!qmw.GetOnnxShape(reshape_nu.Outputs()[0].shape, reshape_out_shape)) {
    return std::nullopt;
  }
  if (reshape_out_shape.size() != 5 || reshape_out_shape[2] != 3) {
    return std::nullopt;
  }

  // The packed-qkv producer must have a statically known rank-3 shape [N, S, 3*n*hs] whose
  // last dim equals 3 * n * hs. This is the tensor we slice from.
  const OrtNodeUnitIODef& qkv_in_def = reshape_nu.Inputs()[0];
  std::vector<uint32_t> qkv_in_shape;
  if (!qmw.GetOnnxShape(qkv_in_def.shape, qkv_in_shape) || qkv_in_shape.size() != 3) {
    return std::nullopt;
  }
  const uint32_t n_rows = qkv_in_shape[0];
  const uint32_t seq = qkv_in_shape[1];
  const uint32_t packed = qkv_in_shape[2];
  const uint32_t num_heads = reshape_out_shape[3];
  const uint32_t head_size = reshape_out_shape[4];
  if (num_heads == 0 || head_size == 0 || seq == 0 || n_rows == 0) {
    return std::nullopt;
  }
  // Consistency: reshape leading dims collapse to N and S, and packed == 3 * n * hs.
  // The product is computed in int64_t so an overflowing n*hs cannot alias a valid packed.
  const int64_t hidden64 = static_cast<int64_t>(num_heads) * static_cast<int64_t>(head_size);
  if (reshape_out_shape[0] != n_rows || reshape_out_shape[1] != seq ||
      static_cast<int64_t>(packed) != 3 * hidden64) {
    return std::nullopt;
  }
  // Slice bounds are emitted as int32_t ranges; reject geometry that cannot be represented.
  if (3 * hidden64 > std::numeric_limits<int32_t>::max()) {
    return std::nullopt;
  }
  // The rank-5 head Reshape output is absorbed by the rewrite and never re-emitted, so it
  // must not be a graph output -- otherwise the tensor would vanish from the QNN graph.
  // The head Transpose output is covered by the IsGraphOutput check in GetAllChildren.
  if (qmw.IsGraphOutput(reshape_nu.Outputs()[0].name)) {
    return std::nullopt;
  }

  m.params.qkv_input_name = qkv_in_def.name;
  m.params.n_rows = n_rows;
  m.params.seq = seq;
  m.params.num_heads = num_heads;
  m.params.head_size = head_size;

  // Head Reshape -> single Transpose child.
  const std::array<std::string_view, 1> transpose_type{kOpTranspose};
  const OrtNodeUnit* transpose = GetOnlyChildOfType(qmw, reshape_nu, transpose_type,
                                                    node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose == nullptr) {
    return std::nullopt;
  }
  m.head_transpose = transpose;
  // The head Transpose must move the stacked-3 axis to the front: perm == [2,0,3,1,4],
  // producing [3, N, n, S, hs]. This is what makes each scalar Gather(axis=0) yield the
  // per-role [N, n, S, hs] tensor.
  {
    OrtNodeAttrHelper transpose_helper(*transpose);
    const std::vector<int64_t> perm =
        transpose_helper.Get("perm", std::vector<int64_t>{});
    const std::vector<int64_t> expected_perm{2, 0, 3, 1, 4};
    if (perm != expected_perm) {
      return std::nullopt;
    }
  }

  // Transpose -> exactly 3 Gather children (the QKV split).
  std::vector<const OrtNodeUnit*> children =
      GetAllChildren(*transpose, node_to_node_unit, node_unit_to_qnn_node_group);
  if (children.size() != 3) {
    return std::nullopt;
  }
  for (const OrtNodeUnit* child : children) {
    if (child->OpType() != kOpGather) {
      return std::nullopt;
    }
    OrtNodeAttrHelper gather_helper(*child);
    if (gather_helper.Get("axis", static_cast<int64_t>(0)) != 0) {
      return std::nullopt;  // must gather along the stacked-3 (front) axis
    }
    std::optional<int64_t> idx = GetGatherScalarIndex(qmw, *child);
    if (!idx.has_value()) {
      return std::nullopt;
    }
    // Each Gather output must be rank-4 [N, n, S, hs] (the scalar index removes axis 0).
    std::vector<uint32_t> gather_out_shape;
    if (!qmw.GetOnnxShape(child->Outputs()[0].shape, gather_out_shape) ||
        gather_out_shape.size() != 4 ||
        gather_out_shape[0] != n_rows || gather_out_shape[1] != num_heads ||
        gather_out_shape[2] != seq || gather_out_shape[3] != head_size) {
      return std::nullopt;
    }
    switch (idx.value()) {
      case 0:
        m.gather_q = child;
        break;
      case 1:
        m.gather_k = child;
        break;
      case 2:
        m.gather_v = child;
        break;
      default:
        return std::nullopt;
    }
  }
  if (m.gather_q == nullptr || m.gather_k == nullptr || m.gather_v == nullptr) {
    return std::nullopt;
  }
  m.params.gather_out_names[0] = FirstOutputName(*m.gather_q);
  m.params.gather_out_names[1] = FirstOutputName(*m.gather_k);
  m.params.gather_out_names[2] = FirstOutputName(*m.gather_v);

  // Q branch: Gather(0) -> Mul(scalar scale).
  const std::array<std::string_view, 1> mul_type{kOpMul};
  const OrtNodeUnit* q_mul = GetOnlyChildOfType(qmw, *m.gather_q, mul_type,
                                                node_to_node_unit, node_unit_to_qnn_node_group);
  if (q_mul == nullptr) {
    return std::nullopt;
  }
  // Require a positive scalar constant on one of Mul's inputs (the 1/sqrt(head_size) scale).
  {
    bool has_scale = false;
    for (const auto& in : q_mul->Inputs()) {
      std::optional<float> val = GetScalarConstantValue(qmw, in.name);
      if (val.has_value() && val.value() > 0.0f) {
        has_scale = true;
        break;
      }
    }
    if (!has_scale) {
      return std::nullopt;
    }
  }
  m.q_scale_mul = q_mul;

  // K branch: Gather(1) -> Transpose.
  const OrtNodeUnit* k_transpose = GetOnlyChildOfType(qmw, *m.gather_k, transpose_type,
                                                      node_to_node_unit, node_unit_to_qnn_node_group);
  if (k_transpose == nullptr) {
    return std::nullopt;
  }
  m.k_transpose = k_transpose;

  // Q*K^T MatMul: the Mul output and the K Transpose output must converge on the same MatMul.
  const std::array<std::string_view, 1> matmul_type{kOpMatMul};
  const OrtNodeUnit* qk_matmul = GetOnlyChildOfType(qmw, *q_mul, matmul_type,
                                                    node_to_node_unit, node_unit_to_qnn_node_group);
  if (qk_matmul == nullptr) {
    return std::nullopt;
  }
  // Confirm k_transpose also feeds qk_matmul.
  const OrtNodeUnit* k_child = GetOnlyChildOfType(qmw, *k_transpose, matmul_type,
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
  if (k_child != qk_matmul) {
    return std::nullopt;
  }
  m.qk_matmul = qk_matmul;

  // Interior chain: walk single-child Add/Reshape nodes until reaching Softmax.
  const OrtNodeUnit* cursor = qk_matmul;
  constexpr int kMaxInterior = 8;  // guard against runaway loops
  const std::array<std::string_view, 3> interior_types{kOpAdd, kOpReshape, kOpSoftmax};
  const OrtNodeUnit* softmax = nullptr;
  for (int i = 0; i < kMaxInterior; ++i) {
    const OrtNodeUnit* next = GetOnlyChildOfType(qmw, *cursor, interior_types,
                                                 node_to_node_unit, node_unit_to_qnn_node_group);
    if (next == nullptr) {
      return std::nullopt;
    }
    if (next->OpType() == kOpSoftmax) {
      softmax = next;
      break;
    }
    // Add or Reshape: record and continue.
    m.interior.push_back(next);
    cursor = next;
  }
  if (softmax == nullptr) {
    return std::nullopt;
  }
  m.softmax = softmax;

  // Softmax -> MatMul(., V). V branch (Gather(2)) must also feed this MatMul.
  const OrtNodeUnit* av_matmul = GetOnlyChildOfType(qmw, *softmax, matmul_type,
                                                    node_to_node_unit, node_unit_to_qnn_node_group);
  if (av_matmul == nullptr) {
    return std::nullopt;
  }
  // Verify V (Gather(2)) feeds av_matmul as its other input.
  {
    const std::string& v_out = FirstOutputName(*m.gather_v);
    const auto& av_inputs = av_matmul->Inputs();
    const std::string& sm_out = FirstOutputName(*softmax);
    bool v_feeds = false;
    bool sm_feeds = false;
    for (const auto& in : av_inputs) {
      if (in.name == v_out) v_feeds = true;
      if (in.name == sm_out) sm_feeds = true;
    }
    if (!v_feeds || !sm_feeds) {
      return std::nullopt;
    }
  }
  m.av_matmul = av_matmul;

  return m;
}

// ---------------------------------------------------------------------------
// Emission helpers
// ---------------------------------------------------------------------------

// Emit a QNN StridedSlice that selects [begin, end) along the last axis of a rank-3 tensor
// [N, S, packed], producing [N, S, end-begin]. Mirrors MaxRoiPoolOpBuilder's ranges-only
// StridedSlice (no begin/end/shrink masks needed).
Ort::Status EmitLastAxisSlice(QnnModelWrapper& qmw,
                              size_t node_index,
                              const std::string& input_name,
                              const std::string& output_name,
                              const std::vector<uint32_t>& input_shape,
                              int32_t begin,
                              int32_t end,
                              Qnn_DataType_t data_type,
                              const QnnQuantParamsWrapper& quant_param,
                              bool do_op_validation) {
  const uint32_t n_rows = input_shape[0];
  const uint32_t seq = input_shape[1];
  const std::vector<uint32_t> output_shape{n_rows, seq, static_cast<uint32_t>(end - begin)};

  const std::string node_name = utils::UniqueNameGenerator().New(output_name, QNN_OP_STRIDED_SLICE);

  // ranges: [start, stop, step] per input dim (rank-3).
  std::vector<uint32_t> ranges_data{
      0u, n_rows, 1u,
      0u, seq, 1u,
      static_cast<uint32_t>(begin), static_cast<uint32_t>(end), 1u};
  QnnParamWrapper ranges_param(node_index, node_name, QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                               {3u, 3u}, std::move(ranges_data), /*is_signed=*/true);
  std::vector<std::string> param_names{ranges_param.GetParamTensorName()};
  RETURN_IF_NOT(qmw.AddParamWrapper(std::move(ranges_param)), "Failed to add StridedSlice ranges param.");

  QnnTensorWrapper output_tensor(output_name, QNN_TENSOR_TYPE_NATIVE, data_type,
                                 quant_param.Copy(), std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(output_tensor)), "Failed to add StridedSlice output tensor.");

  RETURN_IF_NOT(qmw.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE,
                                  {input_name}, {output_name}, std::move(param_names), do_op_validation),
                "Failed to create StridedSlice node.");
  return Ort::Status();
}

// Emit one QKV branch: slice the packed qkv on the last axis, Reshape to [N,S,n,hs], then
// Transpose(perm=[0,2,1,3]) to [N,n,S,hs] under the original Gather output name.
Ort::Status EmitBranch(QnnModelWrapper& qmw,
                       const OrtNodeUnit& head_reshape,
                       const QkvSplitAttentionFusion::SplitParams& p,
                       int role,  // 0=Q, 1=K, 2=V
                       Qnn_DataType_t data_type,
                       const QnnQuantParamsWrapper& quant_param,
                       bool do_op_validation) {
  const uint32_t hidden = p.num_heads * p.head_size;  // n*hs (== each slice width)
  const std::vector<uint32_t> qkv_shape{p.n_rows, p.seq, 3u * hidden};
  const int32_t begin = static_cast<int32_t>(role) * static_cast<int32_t>(hidden);
  const int32_t end = begin + static_cast<int32_t>(hidden);

  const std::string& gather_out = p.gather_out_names[role];
  const std::string slice_out = utils::UniqueNameGenerator().New(gather_out, "_slice");
  const std::string reshape_out = utils::UniqueNameGenerator().New(gather_out, "_reshape");

  // 1) StridedSlice: [N,S,3*hidden] -> [N,S,hidden]
  RETURN_IF_ERROR(EmitLastAxisSlice(qmw, head_reshape.Index(), p.qkv_input_name, slice_out,
                                    qkv_shape, begin, end, data_type, quant_param, do_op_validation));

  // 2) Reshape: [N,S,hidden] -> [N,S,n,hs]
  RETURN_IF_ERROR(qmw.AddReshapeNode(slice_out, reshape_out,
                                     {p.n_rows, p.seq, hidden},
                                     {p.n_rows, p.seq, p.num_heads, p.head_size},
                                     data_type, quant_param, do_op_validation,
                                     /*is_for_input=*/false, /*is_for_output=*/false));

  // 3) Transpose(perm=[0,2,1,3]): [N,S,n,hs] -> [N,n,S,hs] under the original Gather name.
  RETURN_IF_ERROR(qmw.AddTransposeNode(head_reshape.Index(), reshape_out, gather_out,
                                       {p.n_rows, p.seq, p.num_heads, p.head_size},
                                       {0u, 2u, 1u, 3u},
                                       {p.n_rows, p.num_heads, p.seq, p.head_size},
                                       data_type, quant_param, do_op_validation,
                                       /*is_for_input=*/false, /*is_for_output=*/false));
  return Ort::Status();
}

// Shared implementation for IsSupported (validate=true) and AddToModelBuilder (validate=false).
Ort::Status EmitOrValidate(QnnModelWrapper& qmw,
                           const OrtNodeUnit& head_reshape,
                           const QkvSplitAttentionFusion::SplitParams& p,
                           bool do_op_validation) {
  // Derive the QNN data type / quant params from the packed-qkv producer tensor so the
  // emitted tensors preserve the original precision (e.g. fp16).
  const OrtNodeUnitIODef& qkv_in_def = head_reshape.Inputs()[0];
  TensorInfo qkv_info = {};
  RETURN_IF_ERROR(qmw.GetTensorInfo(qkv_in_def, qkv_info));

  // Ensure the packed-qkv input tensor exists in the QNN graph (it is produced by an
  // upstream node, but may not have been registered yet during validation).
  if (!qmw.IsQnnTensorWrapperExist(p.qkv_input_name)) {
    QnnTensorWrapper qkv_tensor;
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(qkv_in_def, qkv_tensor));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(qkv_tensor)), "Failed to add packed-qkv input tensor.");
  }

  for (int role = 0; role < 3; ++role) {
    RETURN_IF_ERROR(EmitBranch(qmw, head_reshape, p, role, qkv_info.qnn_data_type,
                               qkv_info.quant_param, do_op_validation));
  }
  return Ort::Status();
}

}  // namespace

// ---------------------------------------------------------------------------
// QkvSplitAttentionFusion members
// ---------------------------------------------------------------------------
QkvSplitAttentionFusion::QkvSplitAttentionFusion(gsl::span<const OrtNodeUnit* const> claimed_node_units,
                                                 SplitParams params)
    : params_(std::move(params)) {
  if (claimed_node_units.size() != node_units_.size()) {
    ORT_CXX_API_THROW("QkvSplitAttentionFusion requires exactly 5 claimed NodeUnits.", ORT_EP_FAIL);
  }
  std::copy(claimed_node_units.begin(), claimed_node_units.end(), node_units_.begin());
}

std::unique_ptr<IQnnNodeGroup> QkvSplitAttentionFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  // NPU-only fusion. rank-5 tensors are supported on HTP but perform poorly; this fusion
  // rewrites the rank-5 Reshape -> Transpose -> 3x Gather "QKV split" into rank-<=4 QNN
  // ops, leaving the downstream SDPA math untouched. On non-NPU backends we defer to the
  // existing per-op lowering.
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    return nullptr;
  }

  std::optional<MatchedPattern> matched =
      MatchPattern(qnn_model_wrapper, reshape_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!matched.has_value()) {
    return nullptr;
  }

  // Validate the rewrite on QNN before committing. If the emitted ops don't validate on the
  // target backend, decline the fusion so the original nodes lower normally.
  if (!EmitOrValidate(qnn_model_wrapper, *matched->head_reshape, matched->params,
                      /*do_op_validation=*/true)
           .IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("QkvSplitAttentionFusion: match at Reshape '" + reshape_node_unit.Name() +
                 "' failed QNN validation; leaving unfused.")
                    .c_str());
    return nullptr;
  }

  // Claim only the 5 split nodes: [head Reshape, head Transpose, Gather_Q, Gather_K, Gather_V].
  const std::array<const OrtNodeUnit*, 5> claimed{
      matched->head_reshape, matched->head_transpose,
      matched->gather_q, matched->gather_k, matched->gather_v};

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
              ("QkvSplitAttentionFusion: rewriting rank-5 QKV split at Reshape '" +
               reshape_node_unit.Name() + "' to rank-<=4 ops.")
                  .c_str());
  return std::make_unique<QkvSplitAttentionFusion>(
      gsl::span<const OrtNodeUnit* const>{claimed.data(), claimed.size()}, matched->params);
}

gsl::span<const OrtNodeUnit* const> QkvSplitAttentionFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status QkvSplitAttentionFusion::IsSupported(QnnModelWrapper& qnn_model_wrapper,
                                                 const Ort::Logger& /*logger*/) const {
  return EmitOrValidate(qnn_model_wrapper, *node_units_[0], params_, /*do_op_validation=*/true);
}

Ort::Status QkvSplitAttentionFusion::AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                                       const Ort::Logger& /*logger*/) const {
  return EmitOrValidate(qnn_model_wrapper, *node_units_[0], params_, /*do_op_validation=*/false);
}

}  // namespace qnn
}  // namespace onnxruntime
