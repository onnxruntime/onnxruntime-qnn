// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>

#include <gsl/util>
#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Add -> Reshape -> Transpose -> Add. Wraps the Reshape/Transpose in Adds so both
// nodes end up strictly inside a QNN partition (graph inputs/outputs go through the
// Adds, so the fusion pattern is never a graph boundary).
GetTestModelFn BuildReshapeTransposeFloatCase(const std::vector<int64_t>& input_shape,
                                              const std::vector<int64_t>& reshape_shape,
                                              const std::vector<int64_t>& transpose_perm) {
  return [input_shape, reshape_shape, transpose_perm](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_transpose_fusion_graph");

    MakeTestInput<float>(builder, "input", TestInputDef<float>(input_shape, false, -1.0f, 1.0f));

    builder.MakeScalarInitializer<float>("add_const1", 0.0f);
    builder.AddNode("pre_add", "Add", {"input", "add_const1"}, {"pre_add_out"}, kOnnxDomain);

    builder.Make1DInitializer<int64_t>("reshape_shape", reshape_shape);
    builder.AddNode("reshape", "Reshape", {"pre_add_out", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    builder.AddNode("transpose", "Transpose", {"reshape_out"}, {"transpose_out"},
                    kOnnxDomain, {builder.MakeIntsAttribute("perm", transpose_perm)});

    builder.MakeScalarInitializer<float>("add_const2", 0.0f);
    builder.AddNode("post_add", "Add", {"transpose_out", "add_const2"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Wraps Reshape and Transpose in a QDQ node unit each: Add -> Q -> DQ -> Reshape -> Q_r -> DQ ->
// Transpose -> Q_t -> DQ -> Add. All Q/DQ pairs share the same scale/zero-point so the QNN
// Reshape/Transpose builders (which require aligned input/output quant params) can lower the
// graph, and the fusion's HaveMatchingIntermediateEncoding check passes.
GetTestModelFn BuildReshapeTransposeQdqCase(const std::vector<int64_t>& input_shape,
                                            const std::vector<int64_t>& reshape_shape,
                                            const std::vector<int64_t>& transpose_perm) {
  return [input_shape, reshape_shape, transpose_perm](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_transpose_fusion_qdq_graph");

    constexpr float kScale = 1.0f / 128.0f;
    constexpr uint8_t kZeroPoint = 128;

    MakeTestInput<float>(builder, "input", TestInputDef<float>(input_shape, false, -1.0f, 1.0f));

    builder.MakeScalarInitializer<float>("add_const1", 0.0f);
    builder.AddNode("pre_add", "Add", {"input", "add_const1"}, {"pre_add_out"}, kOnnxDomain);

    // pre_add_out -> Q -> DQ -> reshape_in
    const std::string reshape_in =
        AddQDQNodePair<uint8_t>(builder, "qdq_reshape_in", "pre_add_out", kScale, kZeroPoint);

    builder.Make1DInitializer<int64_t>("reshape_shape", reshape_shape);
    builder.AddNode("reshape", "Reshape", {reshape_in, "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    // reshape_out -> Q_r -> DQ -> transpose_in
    const std::string transpose_in =
        AddQDQNodePair<uint8_t>(builder, "qdq_reshape_out", "reshape_out", kScale, kZeroPoint);

    builder.AddNode("transpose", "Transpose", {transpose_in}, {"transpose_out"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", transpose_perm)});

    // transpose_out -> Q_t -> DQ -> post_add_in
    const std::string post_add_in =
        AddQDQNodePair<uint8_t>(builder, "qdq_transpose_out", "transpose_out", kScale, kZeroPoint);

    builder.MakeScalarInitializer<float>("add_const2", 0.0f);
    builder.AddNode("post_add", "Add", {post_add_in, "add_const2"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

// RAII wrapper: create a fresh dump dir, ask QNN EP to write its JSON graph into it,
// and remove it at scope exit. Returns the dir path.
std::filesystem::path MakeDumpDir(ProviderOptions& provider_options, const std::string& name) {
  const std::filesystem::path dir = name;
  std::filesystem::remove_all(dir);
  EXPECT_TRUE(std::filesystem::create_directory(dir));
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = dir.string();
  return dir;
}

}  // namespace

// Composed perm is identity: fusion collapses to a single noop Reshape and drops the
// Transpose. Reshape [3,4,1,1] -> [3,1,4,1] is Transpose-equivalent with derived
// perm=[0,2,1,3]; a following Transpose(perm=[0,2,1,3]) gives fused=[0,1,2,3]. The
// noop Reshape passes QNN op validation (both endpoints have identical shape
// [3,4,1,1]).
TEST_F(QnnHTPBackendTests, ReshapeTransposeFusion_IdentityComposedPerm_NoopReshape) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options = GetProviderOptions();
  const auto dir = MakeDumpDir(provider_options, "ReshapeTransposeFusion_Identity");
  auto cleanup = gsl::finally([&dir]() { std::filesystem::remove_all(dir); });

  RunQnnModelTest(BuildReshapeTransposeFloatCase(/*input_shape=*/{3, 4, 1, 1},
                                                 /*reshape_shape=*/{3, 1, 4, 1},
                                                 /*transpose_perm=*/{0, 2, 1, 3}),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       ElementwiseAbsoluteVerifier(1e-2f)});

  // Fusion produced a single noop Reshape and zero Transposes.
  AssertOpInQnnGraph(dir, "Transpose", 0);
  AssertOpInQnnGraph(dir, "Reshape", 1);
}

// Composed perm is non-identity but the Reshape is Transpose-equivalent: fusion
// collapses to a single Transpose(fused_perm) and drops the Reshape entirely.
// Reshape-perm=[0,3,1,2], Transpose perm=[0,1,3,2], fused=[0,3,2,1].
TEST_F(QnnHTPBackendTests, ReshapeTransposeFusion_NonIdentityComposedPerm_SingleTranspose) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options = GetProviderOptions();
  const auto dir = MakeDumpDir(provider_options, "ReshapeTransposeFusion_NonIdentity");
  auto cleanup = gsl::finally([&dir]() { std::filesystem::remove_all(dir); });

  RunQnnModelTest(BuildReshapeTransposeFloatCase(/*input_shape=*/{1, 4, 4, 1},
                                                 /*reshape_shape=*/{1, 1, 4, 4},
                                                 /*transpose_perm=*/{0, 1, 3, 2}),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       ElementwiseAbsoluteVerifier(1e-2f)});

  // Fusion produced exactly one Transpose (with the fused perm) and zero Reshapes.
  AssertOpInQnnGraph(dir, "Transpose", 1);
  AssertOpInQnnGraph(dir, "Reshape", 0);
}

// Reshape is NOT structurally a Transpose (ranks differ). Fusion must decline and both
// original nodes must remain in the compiled QNN graph.
TEST_F(QnnHTPBackendTests, ReshapeTransposeFusion_NonTransposeEquivalentReshape_NotFused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options = GetProviderOptions();
  const auto dir = MakeDumpDir(provider_options, "ReshapeTransposeFusion_NotFused");
  auto cleanup = gsl::finally([&dir]() { std::filesystem::remove_all(dir); });

  // Rank changes from 3 to 2 -- IsReshapePermutable returns false.
  RunQnnModelTest(BuildReshapeTransposeFloatCase(/*input_shape=*/{1, 4, 4},
                                                 /*reshape_shape=*/{1, 16},
                                                 /*transpose_perm=*/{1, 0}),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(dir, "Reshape", 1);
  AssertOpInQnnGraph(dir, "Transpose", 1);
}

// Two non-1 dims have equal size and sit at non-adjacent positions in the input.
// ComputeReshapePerm walks the output left-to-right and picks the first still-unclaimed
// input dim that matches, so [2,1,2,1] -> [2,2,1,1] resolves to reshape_perm=[0,2,1,3]
// (first output '2' -> input dim 0, second output '2' -> input dim 2). Composed with
// Transpose perm=[0,2,1,3] this gives identity, so fusion must fire and emit a single
// noop Reshape.
//
// If the greedy instead picked the second input '2' for the first output '2', the
// derived perm would be [2,0,1,3] and the composed perm [2,1,0,3] (non-identity) --
// same tensor shapes but a genuinely different data mapping. This test locks in the
// greedy invariant: reordering equal non-1 dims across size-1 dims is legal only if the
// pick preserves the relative order of the non-1 dims.
TEST_F(QnnHTPBackendTests, ReshapeTransposeFusion_EqualNonUnitDims_Identity) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options = GetProviderOptions();
  const auto dir = MakeDumpDir(provider_options, "ReshapeTransposeFusion_EqualDims");
  auto cleanup = gsl::finally([&dir]() { std::filesystem::remove_all(dir); });

  RunQnnModelTest(BuildReshapeTransposeFloatCase(/*input_shape=*/{2, 1, 2, 1},
                                                 /*reshape_shape=*/{2, 2, 1, 1},
                                                 /*transpose_perm=*/{0, 2, 1, 3}),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(dir, "Transpose", 0);
  AssertOpInQnnGraph(dir, "Reshape", 1);
}

// QDQ Reshape -> QDQ Transpose with matching intermediate encoding.
// Exercises HaveMatchingIntermediateEncoding on the quantized path: fusion should collapse
// the pair to a single Transpose in the QNN graph, same as the float case.
TEST_F(QnnHTPBackendTests, ReshapeTransposeFusion_Qdq_MatchingEncoding_SingleTranspose) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options = GetProviderOptions();
  provider_options["offload_graph_io_quantization"] = "0";
  const auto dir = MakeDumpDir(provider_options, "ReshapeTransposeFusion_Qdq_Match");
  auto cleanup = gsl::finally([&dir]() { std::filesystem::remove_all(dir); });

  RunQnnModelTest(BuildReshapeTransposeQdqCase(/*input_shape=*/{1, 4, 4, 1},
                                               /*reshape_shape=*/{1, 1, 4, 4},
                                               /*transpose_perm=*/{0, 1, 3, 2}),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       ElementwiseAbsoluteVerifier(1e-2f)});

  // Fusion fired: exactly one Transpose and zero Reshapes remain in the QNN graph.
  AssertOpInQnnGraph(dir, "Transpose", 1);
  AssertOpInQnnGraph(dir, "Reshape", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
