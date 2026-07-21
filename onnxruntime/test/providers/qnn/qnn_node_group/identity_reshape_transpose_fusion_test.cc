// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

// Builds the graph:  Input -> Reshape -> Transpose -> Output
// The pattern collapses to a single identity Reshape when:
//   - Shape(Reshape input) == Shape(Transpose output), and
//   - The Transpose preserves memory order relative to the Reshape output
//     (non-unit axes of the Reshape output map to output positions in the same
//      relative order).
GetTestModelFn BuildIdentityReshapeTransposeTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& reshape_shape,
    const std::vector<int64_t>& perm) {
  return [input_def, reshape_shape, perm](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    builder.Make1DInitializer<int64_t>("reshape_shape", reshape_shape);
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"});

    builder.MakeOutput("output");
    builder.AddNode("transpose", "Transpose", {"reshape_out"}, {"output"}, "",
                    {test::MakeAttribute("perm", perm)});
  };
}

// Same pattern feeding into a Conv, mirroring the AISW-192362 customer scenario
// (Reshape -> Transpose -> Conv). Conv is layout-sensitive so ORT's TransposeOptimizer
// leaves the Transpose in place, which is the situation the fusion is designed to catch.
//   Input -> Reshape -> Transpose -> Conv -> Output
GetTestModelFn BuildIdentityReshapeTransposeFeedingConvTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& reshape_shape,
    const std::vector<int64_t>& perm,
    const std::vector<int64_t>& conv_weight_shape) {
  return [input_def, reshape_shape, perm, conv_weight_shape](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    builder.Make1DInitializer<int64_t>("reshape_shape", reshape_shape);
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"});

    builder.AddNode("transpose", "Transpose", {"reshape_out"}, {"transpose_out"}, "",
                    {test::MakeAttribute("perm", perm)});

    builder.MakeInitializer<float>("conv_weight", conv_weight_shape, -0.5f, 0.5f);
    builder.MakeOutput("output");
    builder.AddNode("conv", "Conv", {"transpose_out", "conv_weight"}, {"output"}, kOnnxDomain);
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// AISW-192362 shape (scaled down for CI): input has a channel dim of 1, and the
// Reshape+Transpose pair collapses to an identity because permuting a size-1
// axis does not reorder memory.
//   Input:      [1, H, W, 1]
//   Reshape ->  [1, 1, H, W]
//   Transpose (perm=[0,2,3,1]) -> [1, H, W, 1]
TEST_F(QnnHTPBackendTests, IdentityReshapeTransposeFusion_ChannelOne_Basic) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "IdentityReshapeTransposeFusion_ChannelOne_Basic";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({1, 8, 6, 1}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildIdentityReshapeTransposeTestCase(input_def,
                                                        /*reshape_shape=*/{1, 1, 8, 6},
                                                        /*perm=*/{0, 2, 3, 1}),
                  provider_options,
                  13,  // opset
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-4f)});

  // Fused into a single identity Reshape; original Transpose is gone.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Same pattern feeding into a Conv (matches the AISW-192362 customer scenario).
TEST_F(QnnHTPBackendTests, IdentityReshapeTransposeFusion_ChannelOne_FeedingConv) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "IdentityReshapeTransposeFusion_ChannelOne_FeedingConv";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({1, 8, 6, 1}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildIdentityReshapeTransposeFeedingConvTestCase(input_def,
                                                                   /*reshape_shape=*/{1, 1, 8, 6},
                                                                   /*perm=*/{0, 2, 3, 1},
                                                                   /*conv_weight_shape=*/{4, 8, 1, 1}),
                  provider_options,
                  13,  // opset
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});

  // Verify the fusion fired: the user's ONNX Transpose (named "transpose") is gone.
  // We do NOT assert Transpose count == 0 because QNN EP's Conv op-builder inserts
  // its own layout-adapter Transposes (perm=[0,3,1,2], NHWC<->NCHW) around Conv,
  // which are unrelated to this fusion.
  AssertNodeNotInQnnGraph(json_qnn_graph_dir, "transpose");
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Conv2d", 1);
}

// Multiple unit dimensions:  [2, 1, 3, 1] -> Reshape [1, 2, 1, 3] -> Transpose(perm=[1,0,3,2]) -> [2, 1, 3, 1]
// Non-unit axes of the Reshape output are {1, 3} (sizes 2 and 3). perm[k]==1 at k=0,
// perm[k]==3 at k=2 -> 0 < 2, strictly increasing -> identity.
TEST_F(QnnHTPBackendTests, IdentityReshapeTransposeFusion_MultipleUnitDims) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "IdentityReshapeTransposeFusion_MultipleUnitDims";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({2, 1, 3, 1}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildIdentityReshapeTransposeTestCase(input_def,
                                                        /*reshape_shape=*/{1, 2, 1, 3},
                                                        /*perm=*/{1, 0, 3, 2}),
                  provider_options,
                  13,  // opset
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-4f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Negative: Reshape input shape != Transpose output shape.
// Fusion must NOT fire — the Transpose is left in the compiled graph.
//   Input [2, 3, 4]  ->  Reshape [6, 4]  ->  Transpose(perm=[1,0]) -> [4, 6]
// Shapes differ (t0=[2,3,4], t2=[4,6]), so condition 1 (shape equality) fails.
TEST_F(QnnHTPBackendTests, IdentityReshapeTransposeFusion_NotIdentity_ShapeDiffers) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "IdentityReshapeTransposeFusion_NotIdentity_ShapeDiffers";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({2, 3, 4}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildIdentityReshapeTransposeTestCase(input_def,
                                                        /*reshape_shape=*/{6, 4},
                                                        /*perm=*/{1, 0}),
                  provider_options,
                  13,  // opset
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-4f)});

  // Fusion should NOT have fired: the Transpose still appears in the compiled graph.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 1);
}

// Negative: Transpose reorders two non-unit axes (memory not preserved).
//   Input [2, 3, 4]  ->  Reshape [2, 3, 4]  ->  Transpose(perm=[0,2,1]) -> [2, 4, 3]
// Even though Reshape is identity, the Transpose swaps two non-unit axes, so
// condition 2 (memory-order-preserving) fails and fusion must not fire.
// Additionally t0=[2,3,4] and t2=[2,4,3] differ, so condition 1 also fails.
TEST_F(QnnHTPBackendTests, IdentityReshapeTransposeFusion_NotIdentity_NonUnitReorder) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "IdentityReshapeTransposeFusion_NotIdentity_NonUnitReorder";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({2, 3, 4}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildIdentityReshapeTransposeTestCase(input_def,
                                                        /*reshape_shape=*/{2, 3, 4},
                                                        /*perm=*/{0, 2, 1}),
                  provider_options,
                  13,  // opset
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-4f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
