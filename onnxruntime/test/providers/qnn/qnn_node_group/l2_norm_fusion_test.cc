// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

// Builds the L2-normalize subgraph:  Div(x, Add(ReduceL2(x, axis), eps)).
// This is x / (||x||_2 + eps) along `axis`. The QNN EP should fuse it into a single QNN L2Norm op.
// Without the fusion, the EP decomposes ReduceL2 into Mul(x*x) -> ReduceSum -> Sqrt; in fp16 the
// ReduceSum of squares overflows once the norm exceeds the fp16 max, corrupting the result.
GetTestModelFn BuildL2NormalizeTestCase(const TestInputDef<float>& input_def,
                                        int64_t axis,
                                        float eps = 1e-7f) {
  return [=, &input_def](ModelTestBuilder& builder) -> void {
    MakeTestInput<float>(builder, "input", input_def);
    builder.MakeInitializer<int64_t>("axes", {1}, {axis});
    builder.MakeScalarInitializer<float>("eps", eps);

    builder.AddNode("reducel2", "ReduceL2", {"input", "axes"}, {"l2_out"}, "",
                    {builder.MakeScalarAttribute("keepdims", int64_t{1})});
    builder.AddNode("add_eps", "Add", {"l2_out", "eps"}, {"denom"});
    builder.AddNode("div", "Div", {"input", "denom"}, {"output"});
    builder.MakeOutput("output");
  };
}

// ReduceL2 whose output does NOT feed the Add(eps)->Div normalize pattern (here it feeds the Div as
// the numerator instead). The fusion must NOT fire.
GetTestModelFn BuildReduceL2NoFusionTestCase(const TestInputDef<float>& input_def, int64_t axis) {
  return [=, &input_def](ModelTestBuilder& builder) -> void {
    MakeTestInput<float>(builder, "input", input_def);
    builder.MakeInitializer<int64_t>("axes", {1}, {axis});
    builder.AddNode("reducel2", "ReduceL2", {"input", "axes"}, {"l2_out"}, "",
                    {builder.MakeScalarAttribute("keepdims", int64_t{1})});
    // l2_out used directly as a numerator (not the eps/normalize shape) -> not the fused pattern.
    builder.AddNode("add_self", "Add", {"l2_out", "l2_out"}, {"output"});
    builder.MakeOutput("output");
  };
}

ProviderOptions GetHtpFp16ProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_htp_fp16_precision"] = "1";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Fusion fires on the standard L2-normalize pattern: graph has a single L2Norm and none of the
// decomposed ReduceL2 ops (Mul/ReduceSum/Sqrt) nor the Add/Div elementwise ops remain.
TEST_F(QnnHTPBackendTests, L2NormFusion_Basic) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "L2NormFusion_Basic";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetHtpFp16ProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Normalize along the channel axis (axis=1) of an NCHW-style input.
  auto input_def = TestInputDef<float>({1, 8, 4, 4}, false, GetFloatDataInRange(-3.0f, 3.0f, 128));
  RunQnnModelTest(BuildL2NormalizeTestCase(input_def, /*axis=*/1),
                  provider_options,
                  /*opset_version=*/18,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "L2Norm", 1);
  // Fused: no leftover ReduceSum / elementwise (Mul, Add, Div, Sqrt) ops from the decomposition.
  AssertOpInQnnGraph(json_qnn_graph_dir, "ReduceSum", 0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseBinary", 0);
}

// Regression for fp16 overflow in the decomposed L2 norm: with large activations the sum of squares
// exceeds the fp16 max (|x| ~ 180 -> x^2 ~ 32k, summed over many channels >> 65504 -> inf). The fused
// L2Norm must compute the norm on the fp16-safe kernel and match the CPU reference.
TEST_F(QnnHTPBackendTests, L2NormFusion_LargeValues_NoFp16Overflow) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "L2NormFusion_LargeValues";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetHtpFp16ProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // 128 channels of values up to ~180: per-element square ~32k, channel-sum overflows fp16 if the
  // square is materialized (the un-fused decomposition). The fused L2Norm avoids the overflow.
  auto input_def = TestInputDef<float>({1, 128, 5, 5}, false, GetFloatDataInRange(-180.0f, 180.0f, 128 * 25));
  RunQnnModelTest(BuildL2NormalizeTestCase(input_def, /*axis=*/1),
                  provider_options,
                  /*opset_version=*/18,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "L2Norm", 1);
}

// Negative: a ReduceL2 not in the normalize pattern must NOT be fused (no L2Norm emitted).
TEST_F(QnnHTPBackendTests, L2NormFusion_NoFusionWhenNotNormalize) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "L2NormFusion_NoFusion";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetHtpFp16ProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({1, 8, 4, 4}, false, GetFloatDataInRange(-3.0f, 3.0f, 128));
  RunQnnModelTest(BuildReduceL2NoFusionTestCase(input_def, /*axis=*/1),
                  provider_options,
                  /*opset_version=*/18,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "L2Norm", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
