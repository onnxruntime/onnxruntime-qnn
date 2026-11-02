// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Build float test: Add -> Reshape(rank-6) -> Transpose -> Reshape -> Add
// Uses smaller dimensions for testing
GetTestModelFn BuildRank6ToRank5FloatTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("rank6_to_rank5_fusion_float_graph");

    // input
    const auto input_def = TestInputDef<float>({256, 64}, false, -10.0f, 10.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // input + 1 -> add1_out
    builder.MakeScalarInitializer<float>("add_const1", 1.0f);
    builder.AddNode("Add1",
                    "Add",
                    {"input", "add_const1"},
                    {"add1_out"},
                    kOnnxDomain);

    // Reshape: (256, 64) -> (1, 4, 4, 4, 4, 64)
    builder.Make1DInitializer<int64_t>("reshape1_shape", {1, 4, 4, 4, 4, 64});
    builder.AddNode("Reshape1",
                    "Reshape",
                    {"add1_out", "reshape1_shape"},
                    {"reshape1_out"},
                    kOnnxDomain);

    // Transpose: perm [0, 2, 1, 3, 4, 5]
    builder.AddNode("Transpose",
                    "Transpose",
                    {"reshape1_out"},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", {0, 2, 1, 3, 4, 5})});

    // Reshape: (1, 4, 4, 4, 4, 64) -> (1, 256, 64)
    builder.Make1DInitializer<int64_t>("reshape2_shape", {1, 256, 64});
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape"},
                    {"reshape2_out"},
                    kOnnxDomain);

    // reshape2_out + 1 -> output
    builder.MakeScalarInitializer<float>("add_const2", 1.0f);
    builder.AddNode("Add2",
                    "Add",
                    {"reshape2_out", "add_const2"},
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Build float test: Input -> Reshape(rank-6) -> Transpose -> Reshape
//                         -> Add
GetTestModelFn BuildRank6ToRank5FloatMultiConsumerTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("rank6_to_rank5_fusion_float_multi_consumer_graph");

    auto input_def = TestInputDef<float>({256, 64}, false, -10.0f, 10.0f);
    MakeTestInput<float>(builder, "input", input_def);

    builder.MakeScalarInitializer<float>("add_const", 1.0f);
    builder.AddNode("add", "Add", {"input", "add_const"}, {"add_out"}, kOnnxDomain);
    builder.MakeOutput("add_out");

    // Reshape: (256, 64) -> (1, 4, 4, 4, 4, 64)
    builder.Make1DInitializer<int64_t>("reshape1_shape", {1, 4, 4, 4, 4, 64});
    builder.AddNode("reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"});

    // Transpose: perm [0, 2, 1, 3, 4, 5]
    builder.AddNode("transpose",
                    "Transpose",
                    {"reshape1_out"},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", {0, 2, 1, 3, 4, 5})});

    // Reshape: (1, 4, 4, 4, 4, 64) -> (1, 256, 64)
    builder.Make1DInitializer<int64_t>("reshape2_shape", {1, 256, 64});
    builder.AddNode("reshape2", "Reshape", {"transpose_out", "reshape2_shape"}, {"reshape2_out"}, kOnnxDomain);
    builder.MakeOutput("reshape2_out");
  };
}

// Build float test for merge mode: no unit dim exists, but input dims 0 and 1
// travel together (perm[0] == 0 && perm[1] == 1), so they are merged.
// Add -> Reshape([2,8,2,4,2,4]) -> Transpose([0,1,4,2,5,3]) -> Reshape -> Add
GetTestModelFn BuildRank6ToRank5MergeTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("rank6_to_rank5_fusion_merge_graph");

    const auto input_def = TestInputDef<float>({2, 8, 8, 8}, false, -10.0f, 10.0f);
    MakeTestInput<float>(builder, "input", input_def);

    builder.MakeScalarInitializer<float>("add_const1", 1.0f);
    builder.AddNode("Add1",
                    "Add",
                    {"input", "add_const1"},
                    {"add1_out"},
                    kOnnxDomain);

    // Reshape: (2, 8, 8, 8) -> (2, 8, 2, 4, 2, 4)
    builder.Make1DInitializer<int64_t>("reshape1_shape", {2, 8, 2, 4, 2, 4});
    builder.AddNode("Reshape1",
                    "Reshape",
                    {"add1_out", "reshape1_shape"},
                    {"reshape1_out"},
                    kOnnxDomain);

    // Transpose: perm [0, 1, 4, 2, 5, 3]
    builder.AddNode("Transpose",
                    "Transpose",
                    {"reshape1_out"},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", {0, 1, 4, 2, 5, 3})});

    // Reshape: (2, 8, 2, 2, 4, 4) -> (16, 2, 2, 16)
    builder.Make1DInitializer<int64_t>("reshape2_shape", {16, 2, 2, 16});
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape"},
                    {"reshape2_out"},
                    kOnnxDomain);

    builder.MakeScalarInitializer<float>("add_const2", 1.0f);
    builder.AddNode("Add2",
                    "Add",
                    {"reshape2_out", "add_const2"},
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Build float test for the SpaceToDepth-perm fallback: DCR perm without
// SpaceToDepth-consistent shapes, batch 1. Squeezes unit dim 0.
// Add -> Reshape([1,8,2,4,2,4]) -> Transpose(DCR) -> Reshape -> Add
GetTestModelFn BuildRank6ToRank5S2dFallbackTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("rank6_to_rank5_fusion_s2d_fallback_graph");

    const auto input_def = TestInputDef<float>({1, 8, 8, 8}, false, -10.0f, 10.0f);
    MakeTestInput<float>(builder, "input", input_def);

    builder.MakeScalarInitializer<float>("add_const1", 1.0f);
    builder.AddNode("Add1",
                    "Add",
                    {"input", "add_const1"},
                    {"add1_out"},
                    kOnnxDomain);

    // Reshape: (1, 8, 8, 8) -> (1, 8, 2, 4, 2, 4)
    builder.Make1DInitializer<int64_t>("reshape1_shape", {1, 8, 2, 4, 2, 4});
    builder.AddNode("Reshape1",
                    "Reshape",
                    {"add1_out", "reshape1_shape"},
                    {"reshape1_out"},
                    kOnnxDomain);

    // Transpose: SpaceToDepth DCR perm, but the output shape below is not
    // SpaceToDepth-consistent ([1,128,2,2] would be), so SpaceToDepthFusion
    // rejects and this fusion squeezes dim 0 instead.
    builder.AddNode("Transpose",
                    "Transpose",
                    {"reshape1_out"},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", {0, 3, 5, 1, 2, 4})});

    // Reshape: (1, 4, 4, 8, 2, 2) -> (1, 32, 4, 4)
    builder.Make1DInitializer<int64_t>("reshape2_shape", {1, 32, 4, 4});
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape"},
                    {"reshape2_out"},
                    kOnnxDomain);

    builder.MakeScalarInitializer<float>("add_const2", 1.0f);
    builder.AddNode("Add2",
                    "Add",
                    {"reshape2_out", "add_const2"},
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

TEST_F(QnnHTPBackendTests, Rank6ToRank5Fusion_Float) {
  ProviderOptions provider_options = GetProviderOptions();
  RunQnnModelTest(BuildRank6ToRank5FloatTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

TEST_F(QnnHTPBackendTests, Rank6ToRank5Fusion_Float_MultiConsumer) {
  ProviderOptions provider_options = GetProviderOptions();
  RunQnnModelTest(BuildRank6ToRank5FloatMultiConsumerTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

TEST_F(QnnHTPBackendTests, Rank6ToRank5Fusion_MergeFirstTwoDims) {
  ProviderOptions provider_options = GetProviderOptions();
  RunQnnModelTest(BuildRank6ToRank5MergeTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

TEST_F(QnnHTPBackendTests, Rank6ToRank5Fusion_S2dPermFallback) {
  ProviderOptions provider_options = GetProviderOptions();
  RunQnnModelTest(BuildRank6ToRank5S2dFallbackTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
