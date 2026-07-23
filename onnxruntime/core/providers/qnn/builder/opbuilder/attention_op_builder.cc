// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "QnnOpDef.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Decomposition builder for ai.onnx::Attention (opset 23/24).
//
// 4D inputs [B, n, S, hs] (BNSH layout, no reshape needed)
// 3D inputs [B, S, n*hs] (BSH layout, reshape + transpose to BNSH)
// GQA/MQA, KV cache (past/present), softcap, qk_matmul_output
//
// The SDK version guard matches GroupQueryAttentionOpBuilder so that this class
// is compiled away when building against an SDK without the required op-set
// version metadata.
#if !(QNN_OPSET_VERSION_MAJOR < 2 || (QNN_OPSET_VERSION_MAJOR == 2 && QNN_OPSET_VERSION_MINOR <= 11))

class AttentionOpBuilder : public BaseOpBuilder {
 public:
  AttentionOpBuilder() : BaseOpBuilder("AttentionOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AttentionOpBuilder);

 protected:
  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

// ---------------------------------------------------------------------------
// IsOpSupported
// ---------------------------------------------------------------------------
Ort::Status AttentionOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnit& node_unit,
                                              const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  const size_t num_inputs = inputs.size();
  const size_t num_outputs = outputs.size();

  OrtNodeAttrHelper node_helper(node_unit);

  // ---- Primary output must exist ----
  RETURN_IF_NOT(num_outputs > 0 && outputs[0].Exists(),
                "Required output Y (output[0]) not provided");

  // ---- KV cache: past_key and present_key must both be present or both absent ----
  const bool has_past_key = (num_inputs > 4 && inputs[4].Exists());
  const bool has_past_value = (num_inputs > 5 && inputs[5].Exists());
  const bool has_present_key = (num_outputs > 1 && outputs[1].Exists());
  const bool has_present_value = (num_outputs > 2 && outputs[2].Exists());

  RETURN_IF(has_past_key != has_past_value,
            "Attention: past_key and past_value must both be present or both absent");
  RETURN_IF(has_present_key != has_present_value,
            "Attention: present_key and present_value must both be present or both absent");
  RETURN_IF(has_past_key && !has_present_key,
            "Attention: past_key present but present_key output not provided");

  // ---- nonpad_kv_seqlen (input[6], opset 24) ----
  const bool has_nonpad_kv_seqlen = (num_inputs > 6 && inputs[6].Exists());
  RETURN_IF(has_nonpad_kv_seqlen && has_past_key,
            "Attention: nonpad_kv_seqlen and past_key are mutually exclusive per ONNX spec");

  // ---- qk_matmul_output_mode ----
  const int64_t qk_mode = node_helper.Get("qk_matmul_output_mode", static_cast<int64_t>(0));
  RETURN_IF(qk_mode < 0 || qk_mode > 3,
            "Attention: qk_matmul_output_mode must be in [0,3]");
  // If mode != 0 but output[3] is not provided, we just ignore (not an error).
  // If output[3] is provided but mode == 0, it's mode-0 (post QK matmul).

  // ---- scale must be positive if provided ----
  if (node_helper.HasAttr("scale")) {
    const float scale_check = node_helper.Get("scale", 1.0f);
    RETURN_IF(scale_check <= 0.0f, "scale attribute must be positive");
  }

  // ---- Validate Q shape and determine input layout ----
  TensorInfo q_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], q_info));
  const size_t q_rank = q_info.shape.size();
  RETURN_IF(q_rank != 3 && q_rank != 4,
            "Attention: Q input must be rank 3 ([B,S_q,n_q*hs]) or rank 4 ([B,n_q,S_q,hs])");

  // ---- Reject dynamic (0-dim) shapes on Q, K, V ----
  for (uint32_t d : q_info.shape) {
    RETURN_IF(d == 0,
              "Attention: Q input contains a dynamic (0) dimension; only static shapes are supported");
  }

  TensorInfo k_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], k_info));
  for (uint32_t d : k_info.shape) {
    RETURN_IF(d == 0,
              "Attention: K input contains a dynamic (0) dimension; only static shapes are supported");
  }

  TensorInfo v_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], v_info));
  for (uint32_t d : v_info.shape) {
    RETURN_IF(d == 0,
              "Attention: V input contains a dynamic (0) dimension; only static shapes are supported");
  }

  // ---- For 3D inputs, q_num_heads and kv_num_heads attrs are required ----
  if (q_rank == 3) {
    RETURN_IF_NOT(node_helper.HasAttr("q_num_heads"),
                  "Attention: q_num_heads attribute required for 3D (BSH) inputs");
    RETURN_IF_NOT(node_helper.HasAttr("kv_num_heads"),
                  "Attention: kv_num_heads attribute required for 3D (BSH) inputs");
  }

  // ---- GQA divisibility check ----
  {
    uint32_t q_nh = 0;
    uint32_t kv_nh = 0;
    if (q_rank == 4) {
      q_nh = q_info.shape[1];
      kv_nh = k_info.shape[1];
    } else {
      const auto opt_q = node_helper.GetInt64("q_num_heads");
      const auto opt_kv = node_helper.GetInt64("kv_num_heads");
      if (opt_q.has_value()) q_nh = static_cast<uint32_t>(opt_q.value());
      if (opt_kv.has_value()) kv_nh = static_cast<uint32_t>(opt_kv.value());
    }
    RETURN_IF(q_nh != 0 && kv_nh != 0 && q_nh != kv_nh && q_nh % kv_nh != 0,
              "Attention: GQA requires q_num_heads to be divisible by kv_num_heads");
  }

  // ---- KV cache: require static S_past ----
  if (has_past_key) {
    TensorInfo past_k_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[4], past_k_info));
    RETURN_IF(past_k_info.shape.size() != 4,
              "Attention: past_key must be rank 4 ([B,n,S_past,hs])");
    for (uint32_t d : past_k_info.shape) {
      RETURN_IF(d == 0,
                "Attention: past_key contains a dynamic (0) dimension; only static shapes are supported");
    }
  }

  // ---- Full validation: build decomposed nodes with do_op_validation=true ----
  std::vector<std::string> input_names;
  RETURN_IF_ERROR(ProcessInputs(qnn_model_wrapper, node_unit, logger, input_names, true));
  RETURN_IF_ERROR(
      ProcessAttributesAndOutputs(qnn_model_wrapper, node_unit, std::move(input_names), logger, true));
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// ProcessInputs — register Q, K, V, attn_mask, past_key, past_value,
//                 nonpad_kv_seqlen
// ---------------------------------------------------------------------------
Ort::Status AttentionOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnit& node_unit,
                                              const Ort::Logger& logger,
                                              std::vector<std::string>& input_names,
                                              bool /*do_op_validation*/) const {
  const auto& onnx_inputs = node_unit.Inputs();

  // ---- GPU native GQA path: determine routing ----
  {
    TensorInfo q_info{}, k_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[0], q_info));
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[1], k_info));
    const size_t q_rank = q_info.shape.size();
    OrtNodeAttrHelper node_helper(node_unit);
    uint32_t n_q = 0, n_kv = 0;
    if (q_rank == 4) {
      n_q = q_info.shape[1];
      n_kv = k_info.shape[1];
    } else {
      const auto opt_q = node_helper.GetInt64("q_num_heads");
      const auto opt_kv = node_helper.GetInt64("kv_num_heads");
      if (opt_q.has_value()) n_q = static_cast<uint32_t>(opt_q.value());
      if (opt_kv.has_value()) n_kv = static_cast<uint32_t>(opt_kv.value());
    }
    const int64_t is_causal = node_helper.Get("is_causal", static_cast<int64_t>(0));
    const float softcap = node_helper.Get("softcap", 0.0f);
    const bool has_attn_mask = (onnx_inputs.size() > 3 && onnx_inputs[3].Exists());
    const bool has_qk_output = (node_unit.Outputs().size() > 3 &&
                                node_unit.Outputs()[3].Exists());
    if (ShouldUseNativeGQA(qnn_model_wrapper.GetQnnBackendType(), q_rank,
                           n_q, n_kv, is_causal, softcap, has_attn_mask, has_qk_output)) {
      return ProcessInputsNativeGQA(qnn_model_wrapper, node_unit, logger, input_names);
    }
  }

  // ---- Decomposition path ----
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[0], logger, input_names));  // Q
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[1], logger, input_names));  // K
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[2], logger, input_names));  // V

  // input[3] = attn_mask (optional)
  if (onnx_inputs.size() > 3 && onnx_inputs[3].Exists()) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[3], logger, input_names));
  }
  // input[4] = past_key (optional, KV cache)
  if (onnx_inputs.size() > 4 && onnx_inputs[4].Exists()) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[4], logger, input_names));
  }
  // input[5] = past_value (optional, KV cache)
  if (onnx_inputs.size() > 5 && onnx_inputs[5].Exists()) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[5], logger, input_names));
  }
  // input[6] = nonpad_kv_seqlen (optional, opset 24)
  if (onnx_inputs.size() > 6 && onnx_inputs[6].Exists()) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[6], logger, input_names));
  }

  return Ort::Status();
}

// ---------------------------------------------------------------------------
// GPU native GQA path
//
// When backend=GPU, input layout is 3D BSH, the node is true GQA
// (n_q > n_kv, divisible), is_causal=1, and no features QNN GQA cannot
// express (softcap/attn_mask/qk_output), a single
// QNN_OP_GROUP_QUERY_ATTENTION node is emitted instead of the decomposed
// graph.  All other cases (MHA, non-causal, softcap, HTP, 4D inputs) fall
// through to decomposition unchanged.
// ---------------------------------------------------------------------------

static bool ShouldUseNativeGQA(QnnBackendType backend,
                               size_t q_rank,
                               uint32_t n_q, uint32_t n_kv,
                               int64_t is_causal,
                               float softcap,
                               bool has_attn_mask,
                               bool has_qk_output) {
  return IsGpuBackend(backend) &&
         q_rank == 3 &&                    // QNN GQA uses BSH; 4D BNSH → decomposition
         n_q > n_kv && n_q % n_kv == 0 &&  // true GQA per https://arxiv.org/pdf/2305.13245
         is_causal == 1 &&                 // QNN GQA is always causal — no is_causal param
         softcap == 0.0f &&                // no softcap param in QNN GQA
         !has_attn_mask &&                 // no additive mask input in QNN GQA
         !has_qk_output;                   // no per-stage debug output in QNN GQA
}

// Synthesize seqlens_k and total_sequence_length that QNN GQA requires but
// ONNX Attention omits.  Both values are derived from static input shapes
// already validated in IsOpSupported.
//
//   seqlens_k         INT32 [B]   = S_past + S_k − 1  (spec: total_seq_len − 1)
//   total_seq_len     INT32 0D    = S_past + S_k
static Ort::Status AddNativeGQASyntheticInputs(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
                                               uint32_t B, uint32_t S_past, uint32_t S_k,
                                               std::string& seqlens_k_name,
                                               std::string& total_seq_len_name) {
  const int32_t seqlens_val = static_cast<int32_t>(S_past + S_k) - 1;
  const int32_t total_val = static_cast<int32_t>(S_past + S_k);

  seqlens_k_name = utils::UniqueNameGenerator().New(node_unit, "_gqa_seqlens_k");
  {
    std::vector<uint8_t> bytes(static_cast<size_t>(B) * sizeof(int32_t));
    int32_t* p = reinterpret_cast<int32_t*>(bytes.data());
    for (uint32_t b = 0; b < B; ++b) p[b] = seqlens_val;
    QnnTensorWrapper t(seqlens_k_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32,
                       QnnQuantParamsWrapper{}, {B}, std::move(bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(t)),
                  ("Failed to add seqlens_k: " + seqlens_k_name).c_str());
  }

  total_seq_len_name = utils::UniqueNameGenerator().New(node_unit, "_gqa_total_seq_len");
  {
    std::vector<uint8_t> bytes(sizeof(int32_t));
    *reinterpret_cast<int32_t*>(bytes.data()) = total_val;
    // 0D shape (empty dims vector) — QNN requires a scalar, same override used
    // by GroupQueryAttentionOpBuilder for com.microsoft::GroupQueryAttention.
    QnnTensorWrapper t(total_seq_len_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32,
                       QnnQuantParamsWrapper{}, {}, std::move(bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(t)),
                  ("Failed to add total_seq_len: " + total_seq_len_name).c_str());
  }
  return Ort::Status();
}

// Build input_names in the 10-slot QNN GQA order.
// Q/K/V (3D BSH) and past_key/past_value (4D BNSH) are passed straight
// through — no reshape needed.  Slots 7-9 (rotary, position) are null.
static Ort::Status ProcessInputsNativeGQA(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const Ort::Logger& logger,
                                          std::vector<std::string>& input_names) {
  const auto& onnx_inputs = node_unit.Inputs();
  const bool has_past_key = (onnx_inputs.size() > 4 && onnx_inputs[4].Exists());

  TensorInfo k_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[1], k_info));
  const uint32_t B = k_info.shape[0];
  const uint32_t S_k = k_info.shape[1];

  uint32_t S_past = 0;
  if (has_past_key) {
    TensorInfo past_k_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[4], past_k_info));
    S_past = past_k_info.shape[2];  // [B, n_kv, S_past, hs]
  }

  auto AddNull = [&](const char* suffix) -> Ort::Status {
    const std::string name = utils::UniqueNameGenerator().New(node_unit, suffix);
    input_names.push_back(name);
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(QnnTensorWrapper::MakeNull(name)),
                  ("Failed to add null tensor: " + name).c_str());
    return Ort::Status();
  };

  // [0] query  [1] seqlens_k  [2] total_seq_len  [3] key  [4] value
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[0], logger, input_names));

  std::string seqlens_k_name, total_seq_name;
  RETURN_IF_ERROR(AddNativeGQASyntheticInputs(qnn_model_wrapper, node_unit,
                                              B, S_past, S_k,
                                              seqlens_k_name, total_seq_name));
  input_names.push_back(seqlens_k_name);
  input_names.push_back(total_seq_name);

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[1], logger, input_names));
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[2], logger, input_names));

  // [5] past_key  [6] past_value  — 4D BNSH, passed straight through (matches
  //   QNN GQA cache format).  Null-padded when no KV cache is present.
  if (has_past_key) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[4], logger, input_names));
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[5], logger, input_names));
  } else {
    RETURN_IF_ERROR(AddNull("_null_past_key"));
    RETURN_IF_ERROR(AddNull("_null_past_value"));
  }

  // [7] cos_cache  [8] sin_cache  [9] position_ids — no rotary in ai.onnx::Attention
  RETURN_IF_ERROR(AddNull("_null_cos"));
  RETURN_IF_ERROR(AddNull("_null_sin"));
  RETURN_IF_ERROR(AddNull("_null_pos"));

  return Ort::Status();
}

// Emit a single QNN_OP_GROUP_QUERY_ATTENTION node (GPU native GQA path).
static Ort::Status EmitNativeGQANode(QnnModelWrapper& qnn_model_wrapper,
                                     const OrtNodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     bool do_op_validation) {
  const auto& onnx_inputs = node_unit.Inputs();
  const auto& onnx_outputs = node_unit.Outputs();
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_names;

  // NUM_HEADS
  const auto opt_q = node_helper.GetInt64("q_num_heads");
  RETURN_IF_NOT(opt_q.has_value(), "q_num_heads required for native GQA path");
  const uint32_t num_heads_u32 = SafeInt<uint32_t>(opt_q.value());
  RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                               num_heads_u32,
                               QNN_OP_GROUP_QUERY_ATTENTION_PARAM_NUM_HEADS, param_names));

  // KV_NUM_HEADS
  const auto opt_kv = node_helper.GetInt64("kv_num_heads");
  RETURN_IF_NOT(opt_kv.has_value(), "kv_num_heads required for native GQA path");
  const uint32_t kv_num_heads_u32 = SafeInt<uint32_t>(opt_kv.value());
  RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                               kv_num_heads_u32,
                               QNN_OP_GROUP_QUERY_ATTENTION_PARAM_KV_NUM_HEADS, param_names));

  // DO_ROTARY = 0  (ai.onnx::Attention has no rotary embeddings)
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         0u,
                                         QNN_OP_GROUP_QUERY_ATTENTION_PARAM_DO_ROTARY,
                                         param_names));

  // SCALE — from attribute or 1/sqrt(head_size); Q is 3D [B, S, n*hs]
  TensorInfo q_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[0], q_info));
  const uint32_t head_size = q_info.shape[2] / num_heads_u32;
  const float scale_default = 1.0f / std::sqrt(static_cast<float>(head_size));
  const float scale = node_helper.Get("scale", scale_default);
  RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                               scale,
                               QNN_OP_GROUP_QUERY_ATTENTION_PARAM_SCALE, param_names));

  // Outputs: Y (mandatory), present_key, present_value (optional, up to slot 2).
  // qk_matmul_output (slot 3) is never reached — rejected by ShouldUseNativeGQA.
  std::vector<std::string> output_names;
  const size_t n_outs = std::min(onnx_outputs.size(), size_t{3});
  for (size_t i = 0; i < n_outs; ++i) {
    if (onnx_outputs[i].Exists()) {
      const std::string& name = onnx_outputs[i].name;
      output_names.push_back(name);
      TensorInfo out_info{};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_outputs[i], out_info));
      const bool is_graph_out = qnn_model_wrapper.IsGraphOutput(name);
      QnnTensorWrapper wrapper(name,
                               is_graph_out ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                               out_info.qnn_data_type,
                               std::move(out_info.quant_param),
                               std::move(out_info.shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(wrapper)),
                    ("Failed to add output: " + name).c_str());
    } else {
      const std::string null_name = utils::UniqueNameGenerator().New(node_unit, "_null_out");
      output_names.push_back(null_name);
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(QnnTensorWrapper::MakeNull(null_name)),
                    "Failed to add null output.");
    }
  }

  const std::string node_name = utils::UniqueNameGenerator().New(node_unit);
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_GROUP_QUERY_ATTENTION,
                                                std::move(input_names),
                                                std::move(output_names),
                                                std::move(param_names),
                                                do_op_validation),
                "Failed to create QNN_OP_GROUP_QUERY_ATTENTION node.");
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Helper: emit an ElementWiseBinary (MUL or ADD or DIV) node.
// ---------------------------------------------------------------------------
static Ort::Status AddBinaryOpNode(QnnModelWrapper& qnn_model_wrapper,
                                   const OrtNodeUnit& node_unit,
                                   uint32_t operation,
                                   const std::string& lhs_name,
                                   const std::string& rhs_name,
                                   const std::string& out_name,
                                   const std::vector<uint32_t>& out_shape,
                                   Qnn_DataType_t dtype,
                                   const QnnQuantParamsWrapper& quant_param,
                                   bool is_graph_output,
                                   bool do_op_validation) {
  const Qnn_TensorType_t tensor_type =
      is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper out_tensor(out_name, tensor_type, dtype, quant_param.Copy(),
                              std::vector<uint32_t>(out_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_tensor)),
                ("Failed to add output tensor: " + out_name).c_str());

  const std::string node_name = utils::UniqueNameGenerator().New(node_unit, "_ewb");

  std::vector<std::string> param_names;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper,
                                         node_unit.Index(),
                                         node_name,
                                         operation,
                                         QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
                                         param_names));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_BINARY,
                                                {lhs_name, rhs_name},
                                                {out_name},
                                                std::move(param_names),
                                                do_op_validation),
                "Failed to create ElementWiseBinary node.");
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Helper: emit a MatMul node (with optional transpose_in1).
// ---------------------------------------------------------------------------
static Ort::Status AddMatMulNode(QnnModelWrapper& qnn_model_wrapper,
                                 const OrtNodeUnit& node_unit,
                                 const std::string& lhs_name,
                                 const std::string& rhs_name,
                                 const std::string& out_name,
                                 const std::vector<uint32_t>& out_shape,
                                 Qnn_DataType_t dtype,
                                 const QnnQuantParamsWrapper& quant_param,
                                 bool transpose_in1,
                                 bool do_op_validation) {
  QnnTensorWrapper out_tensor(out_name, QNN_TENSOR_TYPE_NATIVE, dtype, quant_param.Copy(),
                              std::vector<uint32_t>(out_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_tensor)),
                ("Failed to add MatMul output tensor: " + out_name).c_str());

  const std::string node_name = utils::UniqueNameGenerator().New(node_unit, "_matmul");

  std::vector<std::string> param_names;
  if (transpose_in1) {
    RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper,
                                       node_unit.Index(),
                                       node_name,
                                       true,
                                       QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                                       param_names));
  }

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_MAT_MUL,
                                                {lhs_name, rhs_name},
                                                {out_name},
                                                std::move(param_names),
                                                do_op_validation),
                "Failed to create MatMul node.");
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Helper: emit a Softmax node (axis param).
// ---------------------------------------------------------------------------
static Ort::Status AddSoftmaxNode(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const std::string& in_name,
                                  const std::string& out_name,
                                  const std::vector<uint32_t>& shape,
                                  Qnn_DataType_t dtype,
                                  const QnnQuantParamsWrapper& quant_param,
                                  uint32_t axis,
                                  bool do_op_validation) {
  QnnTensorWrapper out_tensor(out_name, QNN_TENSOR_TYPE_NATIVE, dtype, quant_param.Copy(),
                              std::vector<uint32_t>(shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_tensor)),
                ("Failed to add Softmax output tensor: " + out_name).c_str());

  const std::string node_name = utils::UniqueNameGenerator().New(node_unit, "_softmax");

  std::vector<std::string> param_names;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper,
                                         node_unit.Index(),
                                         node_name,
                                         axis,
                                         QNN_OP_SOFTMAX_PARAM_AXIS,
                                         param_names));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_SOFTMAX,
                                                {in_name},
                                                {out_name},
                                                std::move(param_names),
                                                do_op_validation),
                "Failed to create Softmax node.");
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Helper: create a static scalar tensor (fp32 or fp16) for softcap arithmetic.
// ---------------------------------------------------------------------------
static Ort::Status AddScalarTensor(QnnModelWrapper& qnn_model_wrapper,
                                   const std::string& name,
                                   float value,
                                   Qnn_DataType_t dtype) {
  std::vector<uint8_t> bytes;
  if (dtype == QNN_DATATYPE_FLOAT_16) {
    const Ort::Float16_t fp16(value);
    bytes.resize(sizeof(uint16_t));
    const uint16_t raw = fp16.val;
    std::memcpy(bytes.data(), &raw, sizeof(uint16_t));
  } else {
    bytes.resize(sizeof(float));
    std::memcpy(bytes.data(), &value, sizeof(float));
  }
  QnnTensorWrapper t(name, QNN_TENSOR_TYPE_STATIC, dtype, QnnQuantParamsWrapper{},
                     {1u}, std::move(bytes));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(t)),
                ("Failed to add scalar tensor: " + name).c_str());
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Helper: Softcap node.
//   out = softcap * tanh(scores / softcap)
//   Steps: Div(scores, sc) → Tanh → Mul(result, sc)
// ---------------------------------------------------------------------------
static Ort::Status AddSoftcapNode(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const std::string& in_name,
                                  const std::string& out_name,
                                  const std::vector<uint32_t>& shape,
                                  Qnn_DataType_t dtype,
                                  const QnnQuantParamsWrapper& quant_param,
                                  float softcap_val,
                                  bool do_op_validation) {
  // Static scalar for softcap value.
  const std::string sc_name = utils::UniqueNameGenerator().New(node_unit, "_softcap_scalar");
  RETURN_IF_ERROR(AddScalarTensor(qnn_model_wrapper, sc_name, softcap_val, dtype));

  // Div(scores, softcap) -> x
  const std::string div_out = utils::UniqueNameGenerator().New(node_unit, "_softcap_div");
  RETURN_IF_ERROR(AddBinaryOpNode(qnn_model_wrapper, node_unit,
                                  QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE,
                                  in_name, sc_name, div_out,
                                  shape, dtype, quant_param,
                                  /*is_graph_output=*/false, do_op_validation));

  // Tanh(x) -> t  [QNN_OP_ELEMENT_WISE_NEURON with TANH param]
  const std::string tanh_out = utils::UniqueNameGenerator().New(node_unit, "_softcap_tanh");
  {
    QnnTensorWrapper tanh_tensor(tanh_out, QNN_TENSOR_TYPE_NATIVE, dtype, quant_param.Copy(),
                                 std::vector<uint32_t>(shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tanh_tensor)),
                  ("Failed to add softcap Tanh output: " + tanh_out).c_str());
    const std::string tanh_node = utils::UniqueNameGenerator().New(node_unit, "_tanh");

    Qnn_Scalar_t tanh_op_scalar = QNN_SCALAR_INIT;
    tanh_op_scalar.dataType = QNN_DATATYPE_UINT_32;
    tanh_op_scalar.uint32Value = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH;
    QnnParamWrapper tanh_op_param(node_unit.Index(), tanh_node,
                                  QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
                                  tanh_op_scalar);
    std::vector<std::string> tanh_param_names = {tanh_op_param.GetParamTensorName()};
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(tanh_op_param)),
                  "Failed to add softcap Tanh operation param.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(tanh_node,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_NEURON,
                                                  {div_out},
                                                  {tanh_out},
                                                  std::move(tanh_param_names),
                                                  do_op_validation),
                  "Failed to create softcap Tanh node.");
  }

  // Mul(t, softcap) -> out
  RETURN_IF_ERROR(AddBinaryOpNode(qnn_model_wrapper, node_unit,
                                  QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY,
                                  tanh_out, sc_name, out_name,
                                  shape, dtype, quant_param,
                                  /*is_graph_output=*/false, do_op_validation));
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Helper: GQA head expansion.
//   in_shape = [B, n_kv, S, hs]  (K or V already in BNSH after 3D→BNSH transform)
//   out_shape = [B, n_q, S, hs]
//
//   Three-step expansion (correct floor-division semantics):
//     1. Reshape in → [B, n_kv, 1, S, hs]       (insert new dim at axis 2)
//     2. Tile [1,1,head_ratio,1,1] → [B, n_kv, head_ratio, S, hs]
//     3. Reshape → [B, n_q, S, hs]
//
// ---------------------------------------------------------------------------
static Ort::Status AddGQAExpandNode(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const std::string& in_name,
                                    const std::string& out_name,
                                    const std::vector<uint32_t>& in_shape,   // [B, n_kv, S, hs]
                                    const std::vector<uint32_t>& out_shape,  // [B, n_q,  S, hs]
                                    Qnn_DataType_t dtype,
                                    const QnnQuantParamsWrapper& quant_param,
                                    uint32_t head_ratio,
                                    bool /*do_op_validation*/) {
  // 4D-only GQA expansion that avoids 5D tensors (HTP finalization fails with 5D).
  //
  // Goal: produce K_expanded[b, kv*head_ratio+r, s, h] = K[b, kv, s, h]
  //       = floor-division [K0,K0,...,K1,K1,...] matching ONNX spec.
  //
  // Steps (all 4D):
  //   [B, n_kv, S, hs]
  //   → Reshape [B, 1, n_kv, S*hs]          (insert unit dim, merge S+hs)
  //   → Tile [1, head_ratio, 1, 1]           (block-repeat on size-1 dim → safe copies)
  //   → [B, head_ratio, n_kv, S*hs]
  //   → Transpose (0,2,1,3) → [B, n_kv, head_ratio, S*hs]
  //   → Reshape [B, n_q, S, hs]              (C-order: kv*head_ratio+r → floor-div ✓)

  const uint32_t B = in_shape[0];
  const uint32_t n_kv = in_shape[1];
  const uint32_t S = in_shape[2];
  const uint32_t hs = in_shape[3];
  const uint32_t Shs = S * hs;  // merged dim

  // Step 1: Reshape [B, n_kv, S, hs] → [B, 1, n_kv, S*hs]
  const std::string r1_name = utils::UniqueNameGenerator().New(node_unit, "_gqa_r1");
  const std::vector<uint32_t> r1_shape = {B, 1u, n_kv, Shs};
  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(in_name, r1_name,
                                                   in_shape, r1_shape,
                                                   dtype, quant_param,
                                                   /*do_op_validation=*/false,
                                                   /*is_for_input=*/false));

  // Step 2: Tile [1, head_ratio, 1, 1] → [B, head_ratio, n_kv, S*hs]
  // Block-repeat on size-1 dim 1 is always correct regardless of Tile semantics.
  const std::string tiled_name = utils::UniqueNameGenerator().New(node_unit, "_gqa_tile");
  const std::vector<uint32_t> tiled_shape = {B, head_ratio, n_kv, Shs};
  {
    QnnTensorWrapper tiled_tensor(tiled_name, QNN_TENSOR_TYPE_NATIVE, dtype, quant_param.Copy(),
                                  std::vector<uint32_t>(tiled_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tiled_tensor)),
                  ("Failed to add GQA Tile output tensor: " + tiled_name).c_str());

    const std::string tile_node = utils::UniqueNameGenerator().New(node_unit, "_gqa_tilenode");
    std::vector<uint32_t> mult_data = {1u, head_ratio, 1u, 1u};
    QnnParamWrapper mult_param(node_unit.Index(), tile_node, QNN_OP_TILE_PARAM_MULTIPLES,
                               {4u}, std::move(mult_data));
    std::vector<std::string> tile_params;
    tile_params.push_back(mult_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(mult_param)),
                  "Failed to add GQA Tile multiples param.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(tile_node,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_TILE,
                                                  {r1_name},
                                                  {tiled_name},
                                                  std::move(tile_params),
                                                  /*do_op_validation=*/false),
                  "Failed to create GQA Tile node.");
  }

  // Step 3: Transpose (0,2,1,3) → [B, n_kv, head_ratio, S*hs]
  const std::string tr_name = utils::UniqueNameGenerator().New(node_unit, "_gqa_tr");
  const std::vector<uint32_t> tr_shape = {B, n_kv, head_ratio, Shs};
  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                     tiled_name, tr_name,
                                                     tiled_shape,
                                                     {0u, 2u, 1u, 3u},
                                                     tr_shape,
                                                     dtype, quant_param,
                                                     /*do_op_validation=*/false,
                                                     /*is_for_input=*/false));

  // Step 4: Reshape [B, n_kv, head_ratio, S*hs] → [B, n_q, S, hs]
  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(tr_name, out_name,
                                                   tr_shape, out_shape,
                                                   dtype, quant_param,
                                                   /*do_op_validation=*/false,
                                                   /*is_for_input=*/false));
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Helper: KV concat (past || current along the sequence axis).
//   past_shape = [B, n, S_past, hs]
//   cur_shape  = [B, n, S_cur,  hs]
//   out_shape  = [B, n, S_past+S_cur, hs]
// ---------------------------------------------------------------------------
static Ort::Status AddKVConcatNode(QnnModelWrapper& qnn_model_wrapper,
                                   const OrtNodeUnit& node_unit,
                                   const std::string& past_name,
                                   const std::string& cur_name,
                                   const std::string& out_name,
                                   const std::vector<uint32_t>& out_shape,
                                   Qnn_DataType_t dtype,
                                   const QnnQuantParamsWrapper& quant_param,
                                   bool is_graph_output,
                                   bool do_op_validation) {
  const Qnn_TensorType_t tensor_type =
      is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper out_tensor(out_name, tensor_type, dtype, quant_param.Copy(),
                              std::vector<uint32_t>(out_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_tensor)),
                ("Failed to add KV concat output tensor: " + out_name).c_str());

  const std::string node_name = utils::UniqueNameGenerator().New(node_unit, "_kv_concat");
  std::vector<std::string> param_names;
  // axis = 2 (the sequence dimension in BNSH layout).
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper,
                                         node_unit.Index(),
                                         node_name,
                                         2u,
                                         QNN_OP_CONCAT_PARAM_AXIS,
                                         param_names));
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_CONCAT,
                                                {past_name, cur_name},
                                                {out_name},
                                                std::move(param_names),
                                                do_op_validation),
                "Failed to create KV concat node.");
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Helper: register an existing intermediate tensor as an APP_READ output
//   by routing it through a no-op Reshape.
// ---------------------------------------------------------------------------
static Ort::Status RegisterIntermediateAsOutput(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const std::string& src_name,
                                                const std::string& out_name,
                                                const std::vector<uint32_t>& shape,
                                                Qnn_DataType_t dtype,
                                                const QnnQuantParamsWrapper& quant_param,
                                                bool do_op_validation) {
  QnnTensorWrapper out_tensor(out_name, QNN_TENSOR_TYPE_APP_READ, dtype, quant_param.Copy(),
                              std::vector<uint32_t>(shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_tensor)),
                ("Failed to add output tensor: " + out_name).c_str());
  const std::string node_name = utils::UniqueNameGenerator().New(node_unit, "_out_reshape");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {src_name},
                                                {out_name},
                                                {},
                                                do_op_validation),
                "Failed to create output identity Reshape node.");
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// ProcessAttributesAndOutputs — emit the full decomposed attention graph
// ---------------------------------------------------------------------------
Ort::Status AttentionOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            std::vector<std::string>&& input_names,
                                                            const Ort::Logger& /*logger*/,
                                                            bool do_op_validation) const {
  const auto& onnx_inputs = node_unit.Inputs();
  const auto& onnx_outputs = node_unit.Outputs();

  OrtNodeAttrHelper node_helper(node_unit);

  // ---- Gather input tensor info ----
  TensorInfo q_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[0], q_info));
  TensorInfo k_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[1], k_info));
  TensorInfo v_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[2], v_info));

  const Qnn_DataType_t dtype = q_info.qnn_data_type;
  const QnnQuantParamsWrapper& q_quant = q_info.quant_param;
  const QnnQuantParamsWrapper& k_quant = k_info.quant_param;
  const QnnQuantParamsWrapper& v_quant = v_info.quant_param;

  const size_t q_rank = q_info.shape.size();
  const bool is_4d = (q_rank == 4);

  // ---- Resolve head counts and sequence/head dimensions ----
  // 3D: Q=[B,S_q,n_q*hs]  K=[B,S_k,n_kv*hs]  V=[B,S_k,n_kv*v_hs]
  // 4D: Q=[B,n_q,S_q,hs]  K=[B,n_kv,S_k,hs]  V=[B,n_kv,S_k,v_hs]
  const uint32_t B = q_info.shape[0];
  uint32_t n_q = 0, n_kv = 0, S_q = 0, S_k = 0, hs = 0, v_hs = 0;

  if (is_4d) {
    n_q = q_info.shape[1];
    S_q = q_info.shape[2];
    hs = q_info.shape[3];
    n_kv = k_info.shape[1];
    S_k = k_info.shape[2];
    v_hs = v_info.shape[3];
  } else {
    const auto opt_q_num_heads = node_helper.GetInt64("q_num_heads");
    const auto opt_kv_num_heads = node_helper.GetInt64("kv_num_heads");
    RETURN_IF_NOT(opt_q_num_heads.has_value() && opt_kv_num_heads.has_value(),
                  "q_num_heads and kv_num_heads are required for 3D Attention inputs");
    n_q = static_cast<uint32_t>(opt_q_num_heads.value());
    n_kv = static_cast<uint32_t>(opt_kv_num_heads.value());
    S_q = q_info.shape[1];
    S_k = k_info.shape[1];
    RETURN_IF(n_q == 0, "q_num_heads must be > 0");
    RETURN_IF(n_kv == 0, "kv_num_heads must be > 0");
    RETURN_IF(q_info.shape[2] % n_q != 0, "Q hidden dim must be divisible by q_num_heads");
    RETURN_IF(k_info.shape[2] % n_kv != 0, "K hidden dim must be divisible by kv_num_heads");
    hs = q_info.shape[2] / n_q;
    v_hs = v_info.shape[2] / n_kv;
  }

  // ---- Feature flags ----
  const bool is_gqa = (n_q != n_kv);
  const uint32_t head_ratio = is_gqa ? (n_q / n_kv) : 1u;

  const bool has_past_key = (onnx_inputs.size() > 4 && onnx_inputs[4].Exists());
  const bool has_qk_output = (onnx_outputs.size() > 3 && onnx_outputs[3].Exists());
  const int64_t qk_mode = node_helper.Get("qk_matmul_output_mode", static_cast<int64_t>(0));
  const float softcap = node_helper.Get("softcap", 0.0f);

  // ---- KV cache: resolve past seq dimension ----
  uint32_t S_past = 0;
  if (has_past_key) {
    TensorInfo past_k_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[4], past_k_info));
    S_past = past_k_info.shape[2];  // [B, n, S_past, hs]
  }

  // ---- Build input_names index helpers ----
  // The order in input_names matches the order ProcessInputs registered them:
  //   [0]=Q [1]=K [2]=V [3]=attn_mask (if present) [4]=past_key (if present) ...
  const bool has_attn_mask = (onnx_inputs.size() > 3 && onnx_inputs[3].Exists());
  const std::string& q_in = input_names[0];
  const std::string& k_in = input_names[1];
  const std::string& v_in = input_names[2];
  // Offset into input_names for optional inputs (computed by walking present flags).
  size_t opt_idx = 3;
  const std::string attn_mask_in = has_attn_mask ? input_names[opt_idx++] : std::string{};
  const std::string past_key_in = has_past_key ? input_names[opt_idx++] : std::string{};
  const std::string past_value_in = has_past_key ? input_names[opt_idx++] : std::string{};

  std::string q_cur = q_in;
  std::string k_cur = k_in;
  std::string v_cur = v_in;

  // ---- Step 1: Create a scalar initializer for sqrt(scale) ----
  const float scale_default = 1.0f / std::sqrt(static_cast<float>(hs));
  const float scale_attr = node_helper.Get("scale", scale_default);
  const float sqrt_scale = std::sqrt(scale_attr);

  const int64_t is_causal = node_helper.Get("is_causal", static_cast<int64_t>(0));

  // ---- GPU native GQA path ----
  if (ShouldUseNativeGQA(qnn_model_wrapper.GetQnnBackendType(), q_rank,
                         n_q, n_kv, is_causal, softcap, has_attn_mask, has_qk_output)) {
    return EmitNativeGQANode(qnn_model_wrapper, node_unit,
                             std::move(input_names), do_op_validation);
  }

  // ---- Decomposition path ----
  const std::string sqrt_scale_name = utils::UniqueNameGenerator().New(node_unit, "_sqrt_scale");
  {
    std::vector<uint8_t> scale_bytes;
    if (dtype == QNN_DATATYPE_FLOAT_16) {
      const Ort::Float16_t fp16_val(sqrt_scale);
      scale_bytes.resize(sizeof(uint16_t));
      const uint16_t raw = fp16_val.val;
      std::memcpy(scale_bytes.data(), &raw, sizeof(uint16_t));
    } else {
      scale_bytes.resize(sizeof(float));
      std::memcpy(scale_bytes.data(), &sqrt_scale, sizeof(float));
    }
    QnnTensorWrapper scale_tensor(sqrt_scale_name,
                                  QNN_TENSOR_TYPE_STATIC,
                                  dtype,
                                  QnnQuantParamsWrapper{},
                                  {1u},
                                  std::move(scale_bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_tensor)),
                  "Failed to add sqrt_scale tensor.");
  }

  // ---- Steps 2-3: Scale Q (always) and K (deferred when KV cache is active) ----
  // For KV cache: present_key = Concat(past_key_raw, K_raw) — store UNSCALED K.
  // Scaling of the full K_present happens AFTER the concat so that past and
  // current keys are treated uniformly.  Without cache, scale K immediately.
  const std::string q_scaled = utils::UniqueNameGenerator().New(node_unit, "_q_scaled");
  RETURN_IF_ERROR(AddBinaryOpNode(qnn_model_wrapper, node_unit,
                                  QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY,
                                  q_cur, sqrt_scale_name, q_scaled,
                                  q_info.shape, dtype, q_quant,
                                  /*is_graph_output=*/false, do_op_validation));
  q_cur = q_scaled;

  if (!has_past_key) {
    // No KV cache: scale K now (standard path).
    const std::string k_scaled = utils::UniqueNameGenerator().New(node_unit, "_k_scaled");
    RETURN_IF_ERROR(AddBinaryOpNode(qnn_model_wrapper, node_unit,
                                    QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY,
                                    k_cur, sqrt_scale_name, k_scaled,
                                    k_info.shape, dtype, k_quant,
                                    /*is_graph_output=*/false, do_op_validation));
    k_cur = k_scaled;
  }
  // If has_past_key: k_cur is still the raw (unscaled) K here.
  // Scaling of k_present happens below, after the Concat.

  // ---- Steps 4-7 (3D only): Reshape + Transpose Q and K into [B, n, S, hs] ----
  if (!is_4d) {
    // Reshape Q: [B, S_q, n_q*hs] -> [B, S_q, n_q, hs]
    const std::string q_reshaped = utils::UniqueNameGenerator().New(node_unit, "_q_reshaped");
    const std::vector<uint32_t> q_reshaped_shape = {B, S_q, n_q, hs};
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(q_cur, q_reshaped,
                                                     {B, S_q, n_q * hs},
                                                     q_reshaped_shape,
                                                     dtype, q_quant,
                                                     do_op_validation,
                                                     /*is_for_input=*/false));
    // Transpose Q: (0,2,1,3) -> [B, n_q, S_q, hs]
    const std::string q_transposed = utils::UniqueNameGenerator().New(node_unit, "_q_transposed");
    const std::vector<uint32_t> q_transposed_shape = {B, n_q, S_q, hs};
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                       q_reshaped, q_transposed,
                                                       q_reshaped_shape,
                                                       {0u, 2u, 1u, 3u},
                                                       q_transposed_shape,
                                                       dtype, q_quant,
                                                       do_op_validation,
                                                       /*is_for_input=*/false));
    q_cur = q_transposed;

    // Reshape K: [B, S_k, n_kv*hs] -> [B, S_k, n_kv, hs]
    const std::string k_reshaped = utils::UniqueNameGenerator().New(node_unit, "_k_reshaped");
    const std::vector<uint32_t> k_reshaped_shape = {B, S_k, n_kv, hs};
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(k_cur, k_reshaped,
                                                     {B, S_k, n_kv * hs},
                                                     k_reshaped_shape,
                                                     dtype, k_quant,
                                                     do_op_validation,
                                                     /*is_for_input=*/false));
    // Transpose K: (0,2,1,3) -> [B, n_kv, S_k, hs]
    const std::string k_transposed = utils::UniqueNameGenerator().New(node_unit, "_k_transposed");
    const std::vector<uint32_t> k_transposed_shape = {B, n_kv, S_k, hs};
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                       k_reshaped, k_transposed,
                                                       k_reshaped_shape,
                                                       {0u, 2u, 1u, 3u},
                                                       k_transposed_shape,
                                                       dtype, k_quant,
                                                       do_op_validation,
                                                       /*is_for_input=*/false));
    k_cur = k_transposed;
  }
  // k_cur is now [B, n_kv, S_k, hs] in BNSH layout.

  // ---- GQA expansion of K (if GQA) ----
  if (is_gqa) {
    const std::vector<uint32_t> k_in_shape = {B, n_kv, S_k, hs};
    const std::vector<uint32_t> k_out_shape = {B, n_q, S_k, hs};
    const std::string k_expanded = utils::UniqueNameGenerator().New(node_unit, "_k_gqa");
    RETURN_IF_ERROR(AddGQAExpandNode(qnn_model_wrapper, node_unit,
                                     k_cur, k_expanded,
                                     k_in_shape, k_out_shape,
                                     dtype, k_quant,
                                     head_ratio,
                                     do_op_validation));
    k_cur = k_expanded;
  }

  // ---- KV cache concat for K ----
  std::string k_present_name;
  if (has_past_key) {
    const uint32_t S_k_total = S_past + S_k;
    const std::vector<uint32_t> k_present_shape = {B, n_q, S_k_total, hs};
    // If present_key is a graph output, emit it as APP_READ directly from concat.
    const bool pk_is_graph_out = (onnx_outputs.size() > 1 &&
                                  onnx_outputs[1].Exists() &&
                                  qnn_model_wrapper.IsGraphOutput(onnx_outputs[1].name));
    k_present_name = pk_is_graph_out
                         ? onnx_outputs[1].name
                         : utils::UniqueNameGenerator().New(node_unit, "_k_present");
    RETURN_IF_ERROR(AddKVConcatNode(qnn_model_wrapper, node_unit,
                                    past_key_in, k_cur, k_present_name,
                                    k_present_shape,
                                    dtype, k_quant,
                                    pk_is_graph_out,
                                    do_op_validation));
    k_cur = k_present_name;
    S_k = S_k_total;  // Update S_k to reflect the full sequence after concat.

    // Scale the full K_present (past + current) by sqrt(scale) for attention.
    // This ensures past and current keys are treated uniformly.
    const std::string k_present_scaled = utils::UniqueNameGenerator().New(node_unit, "_k_present_scaled");
    RETURN_IF_ERROR(AddBinaryOpNode(qnn_model_wrapper, node_unit,
                                    QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY,
                                    k_cur, sqrt_scale_name, k_present_scaled,
                                    k_present_shape, dtype, k_quant,
                                    /*is_graph_output=*/false, do_op_validation));
    k_cur = k_present_scaled;
  }

  // ---- Steps 12-13 (3D only): Reshape + Transpose V into [B, n_kv, S_k_cur, v_hs] ----
  // NOTE: S_k used here is the original S_k of V (before KV concat).
  // We do V transform BEFORE KV concat so V uses original S_k dimensions.
  const uint32_t S_k_orig = has_past_key ? (S_k - S_past) : S_k;
  std::string v_cur_4d = v_cur;

  if (!is_4d) {
    // Reshape V: [B, S_k_orig, n_kv*v_hs] -> [B, S_k_orig, n_kv, v_hs]
    const std::string v_reshaped = utils::UniqueNameGenerator().New(node_unit, "_v_reshaped");
    const std::vector<uint32_t> v_reshaped_shape = {B, S_k_orig, n_kv, v_hs};
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(v_cur, v_reshaped,
                                                     {B, S_k_orig, n_kv * v_hs},
                                                     v_reshaped_shape,
                                                     dtype, v_quant,
                                                     do_op_validation,
                                                     /*is_for_input=*/false));
    // Transpose V: (0,2,1,3) -> [B, n_kv, S_k_orig, v_hs]
    const std::string v_transposed = utils::UniqueNameGenerator().New(node_unit, "_v_transposed");
    const std::vector<uint32_t> v_transposed_shape = {B, n_kv, S_k_orig, v_hs};
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                       v_reshaped, v_transposed,
                                                       v_reshaped_shape,
                                                       {0u, 2u, 1u, 3u},
                                                       v_transposed_shape,
                                                       dtype, v_quant,
                                                       do_op_validation,
                                                       /*is_for_input=*/false));
    v_cur_4d = v_transposed;
  }
  // v_cur_4d is now [B, n_kv, S_k_orig, v_hs] in BNSH layout.

  // ---- GQA expansion of V (if GQA) ----
  if (is_gqa) {
    const std::vector<uint32_t> v_in_shape = {B, n_kv, S_k_orig, v_hs};
    const std::vector<uint32_t> v_out_shape = {B, n_q, S_k_orig, v_hs};
    const std::string v_expanded = utils::UniqueNameGenerator().New(node_unit, "_v_gqa");
    RETURN_IF_ERROR(AddGQAExpandNode(qnn_model_wrapper, node_unit,
                                     v_cur_4d, v_expanded,
                                     v_in_shape, v_out_shape,
                                     dtype, v_quant,
                                     head_ratio,
                                     do_op_validation));
    v_cur_4d = v_expanded;
  }

  // ---- KV cache concat for V ----
  std::string v_present_name;
  if (has_past_key) {
    const uint32_t S_k_total = S_k;  // already updated above.
    const std::vector<uint32_t> v_present_shape = {B, n_q, S_k_total, v_hs};
    const bool pv_is_graph_out = (onnx_outputs.size() > 2 &&
                                  onnx_outputs[2].Exists() &&
                                  qnn_model_wrapper.IsGraphOutput(onnx_outputs[2].name));
    v_present_name = pv_is_graph_out
                         ? onnx_outputs[2].name
                         : utils::UniqueNameGenerator().New(node_unit, "_v_present");
    RETURN_IF_ERROR(AddKVConcatNode(qnn_model_wrapper, node_unit,
                                    past_value_in, v_cur_4d, v_present_name,
                                    v_present_shape,
                                    dtype, v_quant,
                                    pv_is_graph_out,
                                    do_op_validation));
    v_cur_4d = v_present_name;
  }

  // ---- Step 8: MatMul Q * K^T (transpose_in1) -> [B, n_q, S_q, S_k] ----
  const std::string qk_out = utils::UniqueNameGenerator().New(node_unit, "_qk_out");
  const std::vector<uint32_t> qk_shape = {B, n_q, S_q, S_k};
  RETURN_IF_ERROR(AddMatMulNode(qnn_model_wrapper, node_unit,
                                q_cur, k_cur, qk_out,
                                qk_shape, dtype, q_quant,
                                /*transpose_in1=*/true, do_op_validation));
  std::string scores_cur = qk_out;

  // ---- qk_matmul_output mode 0: raw post-QK scores (pre-mask, pre-softcap) ----
  // Per ONNX spec the pipeline is: QK → masks → softcap → softmax
  // mode 0 = raw QK, mode 1 = post-mask pre-softcap,
  // mode 2 = post-softcap,  mode 3 = post-softmax.
  std::string qk_captured;  // The intermediate captured for qk_matmul_output.
  if (has_qk_output && qk_mode == 0) {
    qk_captured = scores_cur;
  }

  // ---- Masks applied BEFORE softcap (per ONNX spec) ----
  // Applying mask after softcap would be wrong: softcap saturates large negatives
  // to -softcap (not -inf), so the mask must set large negatives first so softcap
  // can compress the full range uniformly.

  // ---- Causal mask (is_causal=1): ADD static lower-triangular mask ----
  // With KV cache: offset = S_past so that row i attends to positions <= i+S_past.
  if (is_causal != 0) {
    const uint32_t offset = S_past;  // 0 for no KV cache path.
    const std::string causal_mask_name = utils::UniqueNameGenerator().New(node_unit, "_causal_mask");
    {
      std::vector<uint32_t> mask_shape = {B, n_q, S_q, S_k};
      const size_t total = static_cast<size_t>(B) * static_cast<size_t>(n_q) *
                           static_cast<size_t>(S_q) * static_cast<size_t>(S_k);
      std::vector<uint8_t> mask_bytes;

      if (dtype == QNN_DATATYPE_FLOAT_16) {
        const Ort::Float16_t fp16_large_neg(-1e4f);
        const uint16_t neg_raw = fp16_large_neg.val;
        mask_bytes.resize(total * sizeof(uint16_t));
        uint16_t* mask_ptr = reinterpret_cast<uint16_t*>(mask_bytes.data());
        for (uint32_t b = 0; b < B; ++b) {
          for (uint32_t h = 0; h < n_q; ++h) {
            for (uint32_t i = 0; i < S_q; ++i) {
              for (uint32_t j = 0; j < S_k; ++j) {
                const size_t idx = ((static_cast<size_t>(b) * n_q + h) * S_q + i) * S_k + j;
                mask_ptr[idx] = (j <= i + offset) ? static_cast<uint16_t>(0u) : neg_raw;
              }
            }
          }
        }
      } else {
        constexpr float large_neg = -1e9f;
        mask_bytes.resize(total * sizeof(float));
        float* mask_ptr = reinterpret_cast<float*>(mask_bytes.data());
        for (uint32_t b = 0; b < B; ++b) {
          for (uint32_t h = 0; h < n_q; ++h) {
            for (uint32_t i = 0; i < S_q; ++i) {
              for (uint32_t j = 0; j < S_k; ++j) {
                const size_t idx = ((static_cast<size_t>(b) * n_q + h) * S_q + i) * S_k + j;
                mask_ptr[idx] = (j <= i + offset) ? 0.0f : large_neg;
              }
            }
          }
        }
      }

      QnnTensorWrapper mask_tensor(causal_mask_name,
                                   QNN_TENSOR_TYPE_STATIC,
                                   dtype,
                                   QnnQuantParamsWrapper{},
                                   std::move(mask_shape),
                                   std::move(mask_bytes));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mask_tensor)),
                    "Failed to add causal mask tensor.");
    }

    const std::string masked_out = utils::UniqueNameGenerator().New(node_unit, "_causal_masked");
    RETURN_IF_ERROR(AddBinaryOpNode(qnn_model_wrapper, node_unit,
                                    QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD,
                                    scores_cur, causal_mask_name, masked_out,
                                    qk_shape, dtype, q_quant,
                                    /*is_graph_output=*/false, do_op_validation));
    scores_cur = masked_out;
  }

  // ---- User attention mask: ADD ----
  if (has_attn_mask) {
    const std::string attn_masked_out =
        utils::UniqueNameGenerator().New(node_unit, "_attn_masked");
    RETURN_IF_ERROR(AddBinaryOpNode(qnn_model_wrapper, node_unit,
                                    QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD,
                                    scores_cur, attn_mask_in, attn_masked_out,
                                    qk_shape, dtype, q_quant,
                                    /*is_graph_output=*/false, do_op_validation));
    scores_cur = attn_masked_out;
  }

  // ---- qk_matmul_output mode 1: post-mask, pre-softcap ----
  if (has_qk_output && qk_mode == 1) {
    qk_captured = scores_cur;
  }

  // ---- Softcap (applied AFTER masks per ONNX spec) ----
  if (softcap != 0.0f) {
    const std::string sc_out = utils::UniqueNameGenerator().New(node_unit, "_scores_softcap");
    RETURN_IF_ERROR(AddSoftcapNode(qnn_model_wrapper, node_unit,
                                   scores_cur, sc_out,
                                   qk_shape, dtype, q_quant,
                                   softcap,
                                   do_op_validation));
    scores_cur = sc_out;
  }

  // ---- qk_matmul_output mode 2: post-softcap ----
  if (has_qk_output && qk_mode == 2) {
    qk_captured = scores_cur;
  }

  // ---- Step 11: Softmax (axis=3) -> [B, n_q, S_q, S_k] ----
  const std::string softmax_out = utils::UniqueNameGenerator().New(node_unit, "_softmax_out");
  RETURN_IF_ERROR(AddSoftmaxNode(qnn_model_wrapper, node_unit,
                                 scores_cur, softmax_out,
                                 qk_shape, dtype, q_quant,
                                 /*axis=*/3u, do_op_validation));
  const std::string& attn_weights = softmax_out;

  // ---- qk_matmul_output mode 3: post-softmax (attn_weights) ----
  if (has_qk_output && qk_mode == 3) {
    qk_captured = attn_weights;
  }

  // ---- Step 14: MatMul attn_weights * V -> [B, n_q, S_q, v_hs] ----
  const std::string y_pre_transpose =
      utils::UniqueNameGenerator().New(node_unit, "_y_pre_transpose");
  const std::vector<uint32_t> y_pre_shape = {B, n_q, S_q, v_hs};
  RETURN_IF_ERROR(AddMatMulNode(qnn_model_wrapper, node_unit,
                                attn_weights, v_cur_4d, y_pre_transpose,
                                y_pre_shape, dtype, q_quant,
                                /*transpose_in1=*/false, do_op_validation));

  // ---- Steps 15-16 (3D outputs): Transpose + Reshape back to [B, S_q, n_q*v_hs] ----
  const std::string& final_output_name = onnx_outputs[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  TensorInfo y_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_outputs[0], y_info));

  if (!is_4d) {
    // Transpose Y: (0,2,1,3) -> [B, S_q, n_q, v_hs]
    const std::string y_transposed = utils::UniqueNameGenerator().New(node_unit, "_y_transposed");
    const std::vector<uint32_t> y_transposed_shape = {B, S_q, n_q, v_hs};
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                       y_pre_transpose, y_transposed,
                                                       y_pre_shape,
                                                       {0u, 2u, 1u, 3u},
                                                       y_transposed_shape,
                                                       dtype, q_quant,
                                                       do_op_validation,
                                                       /*is_for_input=*/false));
    // Reshape Y: [B, S_q, n_q, v_hs] -> [B, S_q, n_q*v_hs]
    const std::vector<uint32_t> y_final_shape = {B, S_q, n_q * v_hs};
    const Qnn_TensorType_t final_type =
        is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper final_tensor(final_output_name,
                                  final_type,
                                  y_info.qnn_data_type,
                                  y_info.quant_param.Copy(),
                                  std::vector<uint32_t>(y_final_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_tensor)),
                  "Failed to add final Attention output tensor.");

    const std::string reshape_node = utils::UniqueNameGenerator().New(node_unit, "_y_reshape");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(reshape_node,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_RESHAPE,
                                                  {y_transposed},
                                                  {final_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to create final Reshape node.");
  } else {
    // 4D: y_pre_transpose is [B, n_q, S_q, v_hs] — already the correct output layout.
    const Qnn_TensorType_t final_type =
        is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper final_tensor(final_output_name,
                                  final_type,
                                  y_info.qnn_data_type,
                                  y_info.quant_param.Copy(),
                                  std::vector<uint32_t>(y_pre_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_tensor)),
                  "Failed to add final Attention output tensor (4D).");

    const std::string rename_node = utils::UniqueNameGenerator().New(node_unit, "_y_rename");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(rename_node,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_RESHAPE,
                                                  {y_pre_transpose},
                                                  {final_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to create 4D output identity Reshape node.");
  }

  // ---- Register KV cache outputs (if not already APP_READ) ----
  // When has_past_key is true and present_key/present_value are expected outputs,
  // the concat nodes above already produced them. If they are graph outputs they
  // were created as APP_READ. If they are not graph outputs but the ONNX node
  // declares them as outputs we still need to register them.
  if (has_past_key) {
    // present_key = output[1]
    if (onnx_outputs.size() > 1 && onnx_outputs[1].Exists()) {
      const bool is_go = qnn_model_wrapper.IsGraphOutput(onnx_outputs[1].name);
      if (!is_go && onnx_outputs[1].name != k_present_name) {
        // Need to expose it. The concat output already has the right shape.
        RETURN_IF_ERROR(RegisterIntermediateAsOutput(qnn_model_wrapper, node_unit,
                                                     k_present_name,
                                                     onnx_outputs[1].name,
                                                     {B, n_q, S_k, hs},
                                                     dtype, k_quant,
                                                     do_op_validation));
      }
    }
    // present_value = output[2]
    if (onnx_outputs.size() > 2 && onnx_outputs[2].Exists()) {
      const bool is_go = qnn_model_wrapper.IsGraphOutput(onnx_outputs[2].name);
      if (!is_go && onnx_outputs[2].name != v_present_name) {
        RETURN_IF_ERROR(RegisterIntermediateAsOutput(qnn_model_wrapper, node_unit,
                                                     v_present_name,
                                                     onnx_outputs[2].name,
                                                     {B, n_q, S_k, v_hs},
                                                     dtype, v_quant,
                                                     do_op_validation));
      }
    }
  }

  // ---- Register qk_matmul_output (output[3]) ----
  if (has_qk_output && !qk_captured.empty()) {
    TensorInfo qk_out_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_outputs[3], qk_out_info));
    RETURN_IF_ERROR(RegisterIntermediateAsOutput(qnn_model_wrapper, node_unit,
                                                 qk_captured,
                                                 onnx_outputs[3].name,
                                                 qk_shape,
                                                 dtype, q_quant,
                                                 do_op_validation));
  }

  return Ort::Status();
}

void CreateAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<AttentionOpBuilder>());
}

#else  // SDK version guard

void CreateAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  ORT_UNUSED_PARAMETER(op_type);
  ORT_UNUSED_PARAMETER(op_registrations);
}

#endif  // !(QNN_OPSET_VERSION_MAJOR < 2 || (QNN_OPSET_VERSION_MAJOR == 2 && QNN_OPSET_VERSION_MINOR <= 11))

}  // namespace qnn
}  // namespace onnxruntime
