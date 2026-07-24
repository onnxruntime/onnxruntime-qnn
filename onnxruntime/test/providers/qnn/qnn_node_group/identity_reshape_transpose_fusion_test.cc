// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

// Builds:  Input -> Reshape -> Transpose -> Output
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

// Builds:  Input NHWC -> Reshape (to NCHW) -> Conv NCHW -> Output.
// ORT's TransformLayoutForEP inserts an NCHW<->NHWC adapter Transpose between the Reshape
// and the Conv; that inserted Transpose is what pairs with the user's Reshape.
GetTestModelFn BuildIdentityReshapeTransposeFeedingConvTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& reshape_shape,
    const std::vector<int64_t>& conv_weight_shape) {
  return [input_def, reshape_shape, conv_weight_shape](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    builder.Make1DInitializer<int64_t>("reshape_shape", reshape_shape);
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"});

    builder.MakeInitializer<float>("conv_weight", conv_weight_shape, -0.5f, 0.5f);
    builder.MakeOutput("output");
    builder.AddNode("conv", "Conv", {"reshape_out", "conv_weight"}, {"output"}, kOnnxDomain);
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Channel-1 identity pair: permuting a size-1 axis does not reorder memory.
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

// User graph Reshape -> Conv; ORT inserts the middle Transpose (see builder).
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
                                                                   /*conv_weight_shape=*/{4, 1, 1, 1}),
                  provider_options,
                  13,  // opset
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});

  // Fusion consumed the (user Reshape, ORT-inserted head adapter Transpose) pair; only
  // the Conv output adapter Transpose remains. Without fusion Transpose count would be 2.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Conv2d", 1);
}

// Multiple unit dimensions in the Reshape output; fusion still fires because only
// unit axes are reordered.
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

// Negative: shape(t0) != shape(t2) — fusion must not fire.
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
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Fusion should NOT have fired: the Transpose still appears in the compiled graph.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 1);
}

// Negative: Transpose swaps two non-unit axes — fusion must not fire.
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
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
