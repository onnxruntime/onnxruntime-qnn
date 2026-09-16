// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <gsl/gsl>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

// Geometry for the decomposed packed-QKV window-attention block that
// QkvSplitAttentionFusion targets. Kept tiny: N=2 rows/windows, S=4 tokens,
// n=2 heads, hs=3 head size => packed = 3 * n * hs = 18.
constexpr int64_t kN = 2;
constexpr int64_t kS = 4;
constexpr int64_t kHeads = 2;
constexpr int64_t kHeadSize = 3;
constexpr int64_t kPacked = 3 * kHeads * kHeadSize;

// Builds the decomposed packed-QKV attention block:
//
//   qkv[N,S,3*n*hs]
//     Reshape -> [N,S,3,n,hs]
//     Transpose(perm=[2,0,3,1,4]) -> [3,N,n,S,hs]
//     Gather(axis=0, idx=0/1/2) = Q/K/V -> [N,n,S,hs]
//   Q * scale ; K^T(perm=[0,1,3,2]) ; MatMul(Q,K^T) -> [N,n,S,S]
//   + mask ; Softmax ; MatMul(., V) -> [N,n,S,hs]
//
// `single_mask == true`  emits MatMul -> Add(mask) -> Softmax (non-shifted block).
// `single_mask == false` emits MatMul -> Add -> Reshape -> Add -> Reshape -> Softmax
//                        (the shifted-window double-mask variant).
//
// All downstream shapes are derived from `head_perm` so the graph stays ONNX-valid for any
// permutation (the negative test passes a non-window perm, which changes the score shape).
GetTestModelFn BuildQkvSplitAttentionTestCase(bool single_mask,
                                              const std::vector<int64_t>& head_perm = {2, 0, 3, 1, 4},
                                              int64_t packed_override = kPacked) {
  return [single_mask, head_perm, packed_override](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("qkv_split_attention_graph");

    // Shapes implied by head_perm: transpose_out = permute([N,S,3,n,hs]); each Gather(axis=0)
    // drops the leading axis, so Q/K/V share g = permuted[1:]. The QK product (and hence the
    // mask) is [g0, g1, g2, g2].
    const std::vector<int64_t> base5{kN, kS, 3, kHeads, kHeadSize};
    std::vector<int64_t> permuted(base5.size());
    for (size_t i = 0; i < head_perm.size(); ++i) {
      permuted[i] = base5[static_cast<size_t>(head_perm[i])];
    }
    const std::vector<int64_t> g(permuted.begin() + 1, permuted.end());
    const std::vector<int64_t> score_shape{g[0], g[1], g[2], g[2]};
    const std::vector<int64_t> score_shape5{1, g[0], g[1], g[2], g[2]};

    const auto qkv_def = TestInputDef<float>({kN, kS, packed_override}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "qkv", qkv_def);

    // Reshape qkv -> [N,S,3,n,hs]
    builder.Make1DInitializer<int64_t>("reshape_shape", {kN, kS, 3, kHeads, kHeadSize});
    builder.AddNode("Reshape", "Reshape", {"qkv", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    builder.AddNode("Transpose", "Transpose", {"reshape_out"}, {"transpose_out"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", head_perm)});

    // Three scalar Gathers along axis 0 (indices 0/1/2) -> Q/K/V each [N,n,S,hs].
    builder.MakeScalarInitializer<int64_t>("idx0", 0);
    builder.MakeScalarInitializer<int64_t>("idx1", 1);
    builder.MakeScalarInitializer<int64_t>("idx2", 2);
    builder.AddNode("GatherQ", "Gather", {"transpose_out", "idx0"}, {"q"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))});
    builder.AddNode("GatherK", "Gather", {"transpose_out", "idx1"}, {"k"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))});
    builder.AddNode("GatherV", "Gather", {"transpose_out", "idx2"}, {"v"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))});

    // Q * scale (1/sqrt(hs)).
    builder.MakeScalarInitializer<float>("scale", 1.0f / std::sqrt(static_cast<float>(kHeadSize)));
    builder.AddNode("MulScale", "Mul", {"q", "scale"}, {"q_scaled"}, kOnnxDomain);

    // K^T: transpose last two dims -> [N,n,hs,S].
    builder.AddNode("TransposeK", "Transpose", {"k"}, {"k_t"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", {0, 1, 3, 2})});

    builder.AddNode("QK", "MatMul", {"q_scaled", "k_t"}, {"qk"}, kOnnxDomain);

    std::string softmax_in;
    if (single_mask) {
      const auto mask_def = TestInputDef<float>(score_shape, false, -0.5f, 0.5f);
      MakeTestInput<float>(builder, "mask", mask_def);
      builder.AddNode("AddMask", "Add", {"qk", "mask"}, {"qk_masked"}, kOnnxDomain);
      softmax_in = "qk_masked";
    } else {
      const auto mask1_def = TestInputDef<float>(score_shape, false, -0.5f, 0.5f);
      MakeTestInput<float>(builder, "mask1", mask1_def);
      builder.AddNode("AddMask1", "Add", {"qk", "mask1"}, {"qk_m1"}, kOnnxDomain);

      builder.Make1DInitializer<int64_t>("rshape5", score_shape5);
      builder.AddNode("Reshape5", "Reshape", {"qk_m1", "rshape5"}, {"qk_r5"}, kOnnxDomain);

      const auto mask2_def = TestInputDef<float>(score_shape5, false, -0.5f, 0.5f);
      MakeTestInput<float>(builder, "mask2", mask2_def);
      builder.AddNode("AddMask2", "Add", {"qk_r5", "mask2"}, {"qk_m2"}, kOnnxDomain);

      builder.Make1DInitializer<int64_t>("rshape4", score_shape);
      builder.AddNode("Reshape4", "Reshape", {"qk_m2", "rshape4"}, {"qk_r4"}, kOnnxDomain);
      softmax_in = "qk_r4";
    }

    builder.AddNode("Softmax", "Softmax", {softmax_in}, {"attn"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(-1))});

    builder.AddNode("AV", "MatMul", {"attn", "v"}, {"output"}, kOnnxDomain);
    builder.MakeOutput("output");
  };
}

ProviderOptions GetHtpProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

ProviderOptions GetCpuProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  return provider_options;
}

}  // namespace

// ---------------------------------------------------------------------------
// CPU backend: the fusion is NPU-only, so the pattern must lower per-op and
// still produce correct numerics. Not inside an ARM64 guard: the CPU backend
// is exercised on x86 hosts too.
// ---------------------------------------------------------------------------
TEST_F(QnnCPUBackendTests, QkvSplitAttentionFusion_Cpu_NotFusedStillCorrect) {
  const std::filesystem::path json_dir = "QkvSplitAttentionFusion_Cpu_NotFused";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetCpuProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildQkvSplitAttentionTestCase(/*single_mask=*/true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-4f)});

  // NPU-only fusion => the three Gathers must survive on the CPU backend.
  AssertOpInQnnGraph(json_dir, "Gather", 3);
  AssertOpInQnnGraph(json_dir, "StridedSlice", 0);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Single-mask (non-shifted) window attention block.
TEST_F(QnnHTPBackendTests, QkvSplitAttentionFusion_SingleMask) {
  const std::filesystem::path json_dir = "QkvSplitAttentionFusion_SingleMask";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetHtpProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildQkvSplitAttentionTestCase(/*single_mask=*/true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Structural proof the fusion fired: the three Gathers are replaced by three
  // StridedSlices. Asserting node assignment alone would pass even if the
  // fusion declined, because the per-op path also runs entirely on QNN.
  AssertOpInQnnGraph(json_dir, "StridedSlice", 3);
  AssertOpInQnnGraph(json_dir, "Gather", 0);
}

// Shifted-window variant with the Add -> Reshape -> Add -> Reshape mask chain
// between the QK MatMul and Softmax.
TEST_F(QnnHTPBackendTests, QkvSplitAttentionFusion_DoubleMask) {
  const std::filesystem::path json_dir = "QkvSplitAttentionFusion_DoubleMask";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetHtpProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildQkvSplitAttentionTestCase(/*single_mask=*/false),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "StridedSlice", 3);
  AssertOpInQnnGraph(json_dir, "Gather", 0);
}

// NEGATIVE: a head Transpose perm that is not the packed-QKV split perm must
// not fuse. [2,0,1,3,4] keeps the 3-way axis leading (so the Gathers still
// index it) but permutes the remaining axes differently, which would corrupt
// the Q/K/V layout if the fusion claimed it.
TEST_F(QnnHTPBackendTests, QkvSplitAttentionFusion_WrongPerm_DoesNotFuse) {
  const std::filesystem::path json_dir = "QkvSplitAttentionFusion_WrongPerm";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetHtpProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildQkvSplitAttentionTestCase(/*single_mask=*/true,
                                                 /*head_perm=*/{2, 0, 1, 3, 4}),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Fusion must decline: Gathers survive, no StridedSlice emitted.
  AssertOpInQnnGraph(json_dir, "Gather", 3);
  AssertOpInQnnGraph(json_dir, "StridedSlice", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
