// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

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

// Small window geometry: B=1, H=W=24, ws=12 => Gh=Gw=2, nW=4, ws*ws=144, C=8.
constexpr int64_t kB = 1;
constexpr int64_t kH = 24;
constexpr int64_t kW = 24;
constexpr int64_t kC = 8;
constexpr int64_t kWs = 12;
constexpr int64_t kGh = kH / kWs;
constexpr int64_t kGw = kW / kWs;

// Window PARTITION: [B,H,W,C] -> rank-6 Reshape/Transpose -> [B*nW, ws*ws, C].
// A trailing Add consumes the rank-3 result, mirroring how the real graph feeds
// the partitioned tensor into attention.
GetTestModelFn BuildWindowPartitionTestCase(const std::vector<int64_t>& perm = {0, 1, 3, 2, 4, 5}) {
  return [perm](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("window_partition_graph");

    const auto x_def = TestInputDef<float>({kB, kH, kW, kC}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "x", x_def);

    builder.Make1DInitializer<int64_t>("rs1", {kB, kGh, kWs, kGw, kWs, kC});
    builder.AddNode("R1", "Reshape", {"x", "rs1"}, {"r1"}, kOnnxDomain);
    builder.AddNode("T1", "Transpose", {"r1"}, {"t1"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", perm)});
    builder.Make1DInitializer<int64_t>("rs2", {kB * kGh * kGw, kWs * kWs, kC});
    builder.AddNode("R2", "Reshape", {"t1", "rs2"}, {"windows"}, kOnnxDomain);

    const auto bias_def = TestInputDef<float>({kC}, false, -0.5f, 0.5f);
    MakeTestInput<float>(builder, "bias", bias_def);
    builder.AddNode("Add", "Add", {"windows", "bias"}, {"output"}, kOnnxDomain);
    builder.MakeOutput("output");
  };
}

// Window REVERSE: [B*nW, ws*ws, C] -> rank-6 Reshape/Transpose -> [B,H,W,C].
GetTestModelFn BuildWindowReverseTestCase(const std::vector<int64_t>& perm = {0, 1, 3, 2, 4, 5}) {
  return [perm](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("window_reverse_graph");

    const auto x_def = TestInputDef<float>({kB * kGh * kGw, kWs * kWs, kC}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "x", x_def);

    builder.Make1DInitializer<int64_t>("rs1", {kB, kGh, kGw, kWs, kWs, kC});
    builder.AddNode("R1", "Reshape", {"x", "rs1"}, {"r1"}, kOnnxDomain);
    builder.AddNode("T1", "Transpose", {"r1"}, {"t1"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", perm)});
    builder.Make1DInitializer<int64_t>("rs2", {kB, kH, kW, kC});
    builder.AddNode("R2", "Reshape", {"t1", "rs2"}, {"img"}, kOnnxDomain);

    const auto bias_def = TestInputDef<float>({kC}, false, -0.5f, 0.5f);
    MakeTestInput<float>(builder, "bias", bias_def);
    builder.AddNode("Add", "Add", {"img", "bias"}, {"output"}, kOnnxDomain);
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
// CPU backend: the fusion is NPU-only. Not inside an ARM64 guard so it runs on
// x86 hosts as well.
// ---------------------------------------------------------------------------
TEST_F(QnnCPUBackendTests, WindowPartitionFusion_Cpu_NotFusedStillCorrect) {
  const std::filesystem::path json_dir = "WindowPartitionFusion_Cpu_NotFused";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetCpuProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildWindowPartitionTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-4f)});

  // NPU-only fusion => the original 2 Reshapes + 1 Transpose survive.
  AssertOpInQnnGraph(json_dir, "Transpose", 1);
  AssertOpInQnnGraph(json_dir, "Reshape", 2);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// The rewrite emits 3 Reshapes + 2 rank-4 Transposes, replacing the rank-6
// Reshape/Transpose/Reshape chain.
TEST_F(QnnHTPBackendTests, WindowPartitionFusion_Partition) {
  const std::filesystem::path json_dir = "WindowPartitionFusion_Partition";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetHtpProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildWindowPartitionTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});

  // Structural proof the fusion fired (2 Transposes instead of 1, 3 Reshapes
  // instead of 2). Node assignment alone cannot distinguish this from the
  // unfused path, nor from the generic Rank6ToRank5Fusion rewrite.
  AssertOpInQnnGraph(json_dir, "Transpose", 2);
  AssertOpInQnnGraph(json_dir, "Reshape", 3);
}

TEST_F(QnnHTPBackendTests, WindowPartitionFusion_Reverse) {
  const std::filesystem::path json_dir = "WindowPartitionFusion_Reverse";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetHtpProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildWindowReverseTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});

  AssertOpInQnnGraph(json_dir, "Transpose", 2);
  AssertOpInQnnGraph(json_dir, "Reshape", 3);
}

// NEGATIVE (guards the layout-correctness condition): perm [0,1,2,3,4,5] is the
// identity, and [0,2,1,3,4,5] / [0,1,3,2,5,4] are genuine permutations that are
// NOT the window axis-swap. None may be rewritten as a window partition -- doing
// so would silently produce a different element ordering. The fusion must
// decline and leave the chain to the generic rank-6 handling.
TEST_F(QnnHTPBackendTests, WindowPartitionFusion_GenuinePerm_DoesNotFuse) {
  const std::filesystem::path json_dir = "WindowPartitionFusion_GenuinePerm";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetHtpProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  // Swaps axes 4 and 5 in addition to 2/3 -- not a window partition.
  RunQnnModelTest(BuildWindowPartitionTestCase(/*perm=*/{0, 1, 3, 2, 5, 4}),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});

  // Must NOT be the 3-Reshape/2-Transpose window rewrite. Rank6ToRank5Fusion
  // still reduces the chain, so exactly 1 Transpose and 2 Reshapes remain.
  AssertOpInQnnGraph(json_dir, "Transpose", 1);
  AssertOpInQnnGraph(json_dir, "Reshape", 2);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
