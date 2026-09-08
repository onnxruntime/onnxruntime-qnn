// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ---------------------------------------------------------------------------
// Helper: build a minimal float32 Attention test model (Q, K, V only).
// ---------------------------------------------------------------------------
static GetTestModelFn BuildAttentionTestCase(
    const std::vector<TestInputDef<float>>& q_k_v_defs,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [q_k_v_defs, attrs](ModelTestBuilder& builder) {
    ASSERT_EQ(q_k_v_defs.size(), 3u);

    const std::string q_name = "attention_Q";
    const std::string k_name = "attention_K";
    const std::string v_name = "attention_V";

    MakeTestInput<float>(builder, q_name, q_k_v_defs[0]);
    MakeTestInput<float>(builder, k_name, q_k_v_defs[1]);
    MakeTestInput<float>(builder, v_name, q_k_v_defs[2]);

    builder.MakeOutput("attention_Y");
    builder.AddNode("attention_node", "Attention",
                    {q_name, k_name, v_name}, {"attention_Y"},
                    kOnnxDomain, attrs);
  };
}

// ---------------------------------------------------------------------------
// Helper: build an Attention model with an additive float attn_mask (input[3]).
// ---------------------------------------------------------------------------
static GetTestModelFn BuildAttentionTestCaseWithMask(
    const std::vector<TestInputDef<float>>& q_k_v_defs,
    const TestInputDef<float>& mask_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [q_k_v_defs, mask_def, attrs](ModelTestBuilder& builder) {
    ASSERT_EQ(q_k_v_defs.size(), 3u);

    const std::string q_name = "attention_Q";
    const std::string k_name = "attention_K";
    const std::string v_name = "attention_V";
    const std::string mask_name = "attention_mask";

    MakeTestInput<float>(builder, q_name, q_k_v_defs[0]);
    MakeTestInput<float>(builder, k_name, q_k_v_defs[1]);
    MakeTestInput<float>(builder, v_name, q_k_v_defs[2]);
    MakeTestInput<float>(builder, mask_name, mask_def);

    builder.MakeOutput("attention_Y");
    builder.AddNode("attention_node", "Attention",
                    {q_name, k_name, v_name, mask_name}, {"attention_Y"},
                    kOnnxDomain, attrs);
  };
}

// ---------------------------------------------------------------------------
// Helper: Build an Attention model with KV cache (past_key/value as
// static initializers) and present_key/value as outputs.
//   Q / K / V are dynamic inputs.
//   past_key, past_value are static initializers (is_initializer=true).
//   Outputs: Y, present_key, present_value.
// ---------------------------------------------------------------------------
static GetTestModelFn BuildAttentionTestCaseKV(
    const TestInputDef<float>& q_def,
    const TestInputDef<float>& k_def,
    const TestInputDef<float>& v_def,
    const TestInputDef<float>& past_key_def,
    const TestInputDef<float>& past_value_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [q_def, k_def, v_def, past_key_def, past_value_def, attrs](ModelTestBuilder& builder) {
    const std::string q_name = "attention_Q";
    const std::string k_name = "attention_K";
    const std::string v_name = "attention_V";
    const std::string past_key_name = "attention_past_key";
    const std::string past_val_name = "attention_past_value";

    MakeTestInput<float>(builder, q_name, q_def);
    MakeTestInput<float>(builder, k_name, k_def);
    MakeTestInput<float>(builder, v_name, v_def);
    MakeTestInput<float>(builder, past_key_name, past_key_def);
    MakeTestInput<float>(builder, past_val_name, past_value_def);

    builder.MakeOutput("attention_Y");
    builder.MakeOutput("attention_present_key");
    builder.MakeOutput("attention_present_value");

    // ONNX Attention: inputs 0-5, outputs 0-2 (Y, present_key, present_value).
    // Use an empty string "" for input[3] (no attn_mask) to skip that slot.
    builder.AddNode("attention_node", "Attention",
                    {q_name, k_name, v_name, "", past_key_name, past_val_name},
                    {"attention_Y", "attention_present_key", "attention_present_value"},
                    kOnnxDomain, attrs);
  };
}

// ---------------------------------------------------------------------------
// Helper: Build a 4D Attention model with qk_matmul_output (output[3]).
// ---------------------------------------------------------------------------
static GetTestModelFn BuildAttentionTestCaseDebugOutput(
    const std::vector<TestInputDef<float>>& q_k_v_defs,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [q_k_v_defs, attrs](ModelTestBuilder& builder) {
    ASSERT_EQ(q_k_v_defs.size(), 3u);

    const std::string q_name = "attention_Q";
    const std::string k_name = "attention_K";
    const std::string v_name = "attention_V";

    MakeTestInput<float>(builder, q_name, q_k_v_defs[0]);
    MakeTestInput<float>(builder, k_name, q_k_v_defs[1]);
    MakeTestInput<float>(builder, v_name, q_k_v_defs[2]);

    builder.MakeOutput("attention_Y");
    // output[1] and output[2] (present_key/value) are absent.
    // output[3] is qk_matmul_output.
    builder.MakeOutput("attention_qk_output");

    // Use empty strings for outputs[1] and outputs[2] to skip KV cache outputs.
    builder.AddNode("attention_node", "Attention",
                    {q_name, k_name, v_name},
                    {"attention_Y", "", "", "attention_qk_output"},
                    kOnnxDomain, attrs);
  };
}

// ===========================================================================
// 4D inputs (BNSH layout): no reshape/transpose needed
// ===========================================================================

// MHA 4D non-causal.
// Q/K/V [1, 4, 8, 16]: batch=1, 4 heads, seq=8, head_size=16.
TEST_F(QnnHTPBackendTests, Attention_MHA_4D_NonCausal) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 4D causal — adds static lower-triangular causal mask.
TEST_F(QnnHTPBackendTests, Attention_MHA_4D_Causal) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(1))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 4D single head — n_heads=1 edge case.
TEST_F(QnnHTPBackendTests, Attention_MHA_4D_SingleHead) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 1, 8, 32}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 1, 8, 32}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 1, 8, 32}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 4D batch_size=2 — verifies batch dimension handling.
TEST_F(QnnHTPBackendTests, Attention_MHA_4D_Batch2) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({2, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({2, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({2, 4, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 4D with explicit scale attribute.
TEST_F(QnnHTPBackendTests, Attention_MHA_4D_CustomScale) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0)),
           test::MakeAttribute("scale", 0.1f)}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// ===========================================================================
// 3D inputs (BSH layout): reshape + transpose to BNSH internally
// ===========================================================================

// MHA 3D non-causal.
// Q/K/V [1, 8, 64]: seq=8, hidden=64=4*16, head_size=16.
TEST_F(QnnHTPBackendTests, Attention_MHA_3D_NonCausal) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 3D causal — static lower-triangular mask.
TEST_F(QnnHTPBackendTests, Attention_MHA_3D_Causal) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(1))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 3D single head.
TEST_F(QnnHTPBackendTests, Attention_MHA_3D_SingleHead) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(1)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(1)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 3D batch_size=2.
TEST_F(QnnHTPBackendTests, Attention_MHA_3D_Batch2) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({2, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({2, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({2, 8, 64}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 3D with explicit additive attn_mask [1, 4, 8, 8].
// Exercises the attn_mask ADD node in the decomposition.
TEST_F(QnnHTPBackendTests, Attention_MHA_3D_AttnMask) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  // Mask shape: [B=1, n_heads=4, S_q=8, S_k=8] — additive bias.
  const std::vector<int64_t> mask_shape = {1, 4, 8, 8};

  RunQnnModelTest(
      BuildAttentionTestCaseWithMask(
          {TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f)},
          TestInputDef<float>(mask_shape, false, -0.5f, 0.0f),  // negative bias
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 3D causal + attn_mask — both mask paths active simultaneously.
TEST_F(QnnHTPBackendTests, Attention_MHA_3D_Causal_AttnMask) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  const std::vector<int64_t> mask_shape = {1, 4, 8, 8};

  RunQnnModelTest(
      BuildAttentionTestCaseWithMask(
          {TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f)},
          TestInputDef<float>(mask_shape, false, -0.5f, 0.0f),
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(1))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MHA 3D with custom scale attribute.
TEST_F(QnnHTPBackendTests, Attention_MHA_3D_CustomScale) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(0)),
           test::MakeAttribute("scale", 0.1f)}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// ===========================================================================
// GQA/MQA: q_num_heads != kv_num_heads
// ===========================================================================

// GQA 3D non-causal — n_q=4, n_kv=2, head_ratio=2.
// Q[1,8,32] (4 heads * head_size=8), K/V[1,8,16] (2 heads * head_size=8).
TEST_F(QnnHTPBackendTests, Attention_GQA_3D_NonCausal) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),   // Q: n_q=4, hs=8
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f),   // K: n_kv=2, hs=8
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f)},  // V: n_kv=2, hs=8
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(2)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// GQA 3D causal with causal mask.
TEST_F(QnnHTPBackendTests, Attention_GQA_3D_Causal) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(2)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(1))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// GQA 4D — Q[1,4,8,16], K/V[1,2,8,16], head_ratio=2.
TEST_F(QnnHTPBackendTests, Attention_GQA_4D) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),   // Q: [B,n_q,S,hs]
           TestInputDef<float>({1, 2, 8, 16}, false, -1.0f, 1.0f),   // K: [B,n_kv,S,hs]
           TestInputDef<float>({1, 2, 8, 16}, false, -1.0f, 1.0f)},  // V: [B,n_kv,S,hs]
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// MQA 3D — n_q=4, n_kv=1 (Multi-Query Attention).
// Q[1,8,32] (4 heads * 8), K/V[1,8,8] (1 head * 8).
TEST_F(QnnHTPBackendTests, Attention_MQA_3D) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),  // Q: n_q=4, hs=8
           TestInputDef<float>({1, 8, 8}, false, -1.0f, 1.0f),   // K: n_kv=1, hs=8
           TestInputDef<float>({1, 8, 8}, false, -1.0f, 1.0f)},  // V: n_kv=1, hs=8
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(1)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// ===========================================================================
// Softcap: scores = softcap * tanh(scores / softcap)
// Gated to real ARM64 hardware
// ===========================================================================
#if defined(__aarch64__) || defined(_M_ARM64)

// 4D MHA with softcap=10.0, non-causal.
TEST_F(QnnHTPBackendTests, Attention_Softcap_4D) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0)),
           test::MakeAttribute("softcap", 10.0f)}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// 3D MHA with softcap=50.0, causal.
// Q/K/V [1, 8, 64]: n_heads=4, head_size=16.
TEST_F(QnnHTPBackendTests, Attention_Softcap_3D_Causal) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(1)),
           test::MakeAttribute("softcap", 50.0f)}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

// ===========================================================================
// KV cache: past_key/value as static initializers,
//           present_key/value as graph outputs.
// ===========================================================================

// 4D MHA with KV cache.
//   Q [1,4,8,16], K [1,4,8,16], V [1,4,8,16]
//   past_key=[1,4,4,16] (S_past=4 initializer), past_value=[1,4,4,16]
//   present_key=[1,4,12,16], present_value=[1,4,12,16]
TEST_F(QnnHTPBackendTests, Attention_KVCache_4D) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCaseKV(
          TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),  // Q
          TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),  // K
          TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),  // V
          TestInputDef<float>({1, 4, 4, 16}, true, -1.0f, 1.0f),   // past_key  (initializer)
          TestInputDef<float>({1, 4, 4, 16}, true, -1.0f, 1.0f),   // past_value (initializer)
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// ===========================================================================
// qk_matmul_output: debug outputs at different stages
// ===========================================================================

// qk_matmul_output_mode=0 (post-QK matmul, before softcap/mask/softmax).
// Q/K/V [1,4,8,16]: qk_output shape should be [1,4,8,8].
TEST_F(QnnHTPBackendTests, Attention_DebugOutput_Mode0) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCaseDebugOutput(
          {TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0)),
           test::MakeAttribute("qk_matmul_output_mode", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// qk_matmul_output_mode=3 (post-softmax / attn_weights).
// qk_output shape should be [1,4,8,8] (same as mode 0 but different values).
TEST_F(QnnHTPBackendTests, Attention_DebugOutput_Mode3) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCaseDebugOutput(
          {TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 4, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0)),
           test::MakeAttribute("qk_matmul_output_mode", static_cast<int64_t>(3))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ===========================================================================
// GPU Attention tests
//
// Two categories:
//   Native   — GPU + true GQA (n_q > n_kv, divisible) + is_causal=1
//               + no softcap/attn_mask/qk_output → QNN_OP_GROUP_QUERY_ATTENTION
//   Decompose — any condition that disqualifies native GQA → decomposition
//               (same graph as HTP, but runs on Adreno)
//
// Gated to _M_ARM64: Adreno GPU is only available on Snapdragon X hardware.
// ===========================================================================
#if defined(_M_ARM64)

// ---------------------------------------------------------------------------
// Native GQA path (GPU + causal + no softcap/attn_mask/qk_output)
// ---------------------------------------------------------------------------

// GQA 3D with KV cache, causal — native QNN_OP_GROUP_QUERY_ATTENTION.
// DISABLED: unpacked QKV (separate K/V inputs) is not supported by the GPU backend
// in QAIRT 2.48; expected to be supported in QAIRT 2.50. Re-enable when the SDK
// is uplevelled.
TEST_F(QnnGPUBackendTests, DISABLED_Attention_GPU_GQA_3D_Native_KVCache) {
  ProviderOptions opts;
  opts["backend_type"] = "gpu";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCaseKV(
          TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),    // Q
          TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f),    // K
          TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f),    // V
          TestInputDef<float>({1, 2, 4, 8}, false, -1.0f, 1.0f),  // past_key  (dynamic APP_WRITE)
          TestInputDef<float>({1, 2, 4, 8}, false, -1.0f, 1.0f),  // past_value (dynamic APP_WRITE)
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(2)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(1))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// GQA 3D, head_ratio=2, causal, no KV cache → native QNN_OP_GROUP_QUERY_ATTENTION.
// Q [1,8,32] (4 heads × 8), K/V [1,8,16] (2 heads × 8).
TEST_F(QnnGPUBackendTests, Attention_GPU_GQA_3D_Native) {
  ProviderOptions opts;
  opts["backend_type"] = "gpu";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(2)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(1))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// MQA 3D (kv_num_heads=1), causal, no KV cache → native path.
TEST_F(QnnGPUBackendTests, Attention_GPU_MQA_3D_Native) {
  ProviderOptions opts;
  opts["backend_type"] = "gpu";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 8}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 8}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(1)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(1))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// MHA 3D (n_q == n_kv), causal, no KV cache → native path (MHA is supported).
TEST_F(QnnGPUBackendTests, Attention_GPU_MHA_3D_Native) {
  ProviderOptions opts;
  opts["backend_type"] = "gpu";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(1))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// GQA 4D BNSH, causal, no KV cache → native (EmitNativeGQANode inserts Transpose+Reshape).
TEST_F(QnnGPUBackendTests, Attention_GPU_GQA_4D_Native) {
  ProviderOptions opts;
  opts["backend_type"] = "gpu";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 4, 8, 8}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 2, 8, 8}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 2, 8, 8}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(1))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// ---------------------------------------------------------------------------
// Decomposition path on GPU — each test disqualifies one native condition.
// ---------------------------------------------------------------------------

// GQA 3D with softcap — no softcap param in QNN GQA → decomposition.
TEST_F(QnnGPUBackendTests, Attention_GPU_GQA_3D_Decompose_Softcap) {
  ProviderOptions opts;
  opts["backend_type"] = "gpu";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(2)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(1)),
           test::MakeAttribute("softcap", 5.0f)}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// GQA 4D BNSH inputs, is_causal=0 → decomposition (causal required for native path).
// Q [1,4,8,8] (n_q=4), K/V [1,2,8,8] (n_kv=2).
TEST_F(QnnGPUBackendTests, Attention_GPU_GQA_4D_Decompose) {
  ProviderOptions opts;
  opts["backend_type"] = "gpu";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 4, 8, 8}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 2, 8, 8}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 2, 8, 8}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// GQA 3D with explicit attn_mask — no additive mask in QNN GQA → decomposition.
// mask shape [8,8] broadcast to [1,4,8,8].
TEST_F(QnnGPUBackendTests, Attention_GPU_GQA_3D_Decompose_AttnMask) {
  ProviderOptions opts;
  opts["backend_type"] = "gpu";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCaseWithMask(
          {TestInputDef<float>({1, 8, 32}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 16}, false, -1.0f, 1.0f)},
          TestInputDef<float>({8, 8}, false, -0.5f, 0.0f),  // additive float mask
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(2)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// MHA 3D non-causal (is_causal=0) — causal required for native → decomposition.
TEST_F(QnnGPUBackendTests, Attention_GPU_MHA_3D_Decompose_NonCausal) {
  ProviderOptions opts;
  opts["backend_type"] = "gpu";
  opts["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildAttentionTestCase(
          {TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f),
           TestInputDef<float>({1, 8, 64}, false, -1.0f, 1.0f)},
          {test::MakeAttribute("q_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("kv_num_heads", static_cast<int64_t>(4)),
           test::MakeAttribute("is_causal", static_cast<int64_t>(0))}),
      opts, /*opset_version=*/24,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

#endif  // defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
