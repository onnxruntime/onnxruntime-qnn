// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Builds: root -> HardSigmoid -> Mul(<ordered inputs>) -> output, which the QNN EP
// recognizes as HardSwish(x) = x * HardSigmoid(x) and fuses into a single
// QNN_OP_ELEMENT_WISE_NEURON (HardSwish) node.
//
// `hardsigmoid_first` controls the Mul input ordering:
//   true  -> Mul(hsig_out, input)   (HardSigmoid output is Inputs()[0])
//   false -> Mul(input, hsig_out)   (HardSigmoid output is Inputs()[1])
// Both orderings are mathematically equivalent and must fuse identically.
template <typename FloatType>
GetTestModelFn BuildHardSigmoidMulTestCase(const TestInputDef<FloatType>& input_def, bool hardsigmoid_first) {
  return [input_def, hardsigmoid_first](ModelTestBuilder& builder) -> void {
    MakeTestInput<FloatType>(builder, "input", input_def);

    // HardSigmoid uses QNN's required alpha=1/6, beta=0.5 so the fusion is eligible.
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("alpha", 1.0f / 6.0f));
    attrs.push_back(MakeAttribute("beta", 0.5f));
    builder.AddNode("HardSigmoid", "HardSigmoid", {"input"}, {"hsig_out"}, kOnnxDomain, attrs);

    if (hardsigmoid_first) {
      builder.AddNode("Mul", "Mul", {"hsig_out", "input"}, {"output"}, kOnnxDomain);
    } else {
      builder.AddNode("Mul", "Mul", {"input", "hsig_out"}, {"output"}, kOnnxDomain);
    }
    builder.MakeOutput("output");
  };
}

TestInputDef<float> MakeHardSigmoidAccuracyInputDef() {
  std::vector<float> input_data = {-8.0f, -2.0f, 0.0f, 0.5f, 0.9f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, 0.0f, 0.2f, 0.4f, 0.8f, 2.1f, 4.3f, 7.0f};
  return TestInputDef<float>({2, 2, 2, 2}, false, input_data);
}

ProviderOptions GetHtpProviderOptions(const std::filesystem::path& json_qnn_graph_dir,
                                      bool enable_htp_fp16_precision = true) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  if (enable_htp_fp16_precision) {
    provider_options["enable_htp_fp16_precision"] = "1";
  }
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  return provider_options;
}

void AssertHardSwishFusion(const std::filesystem::path& json_qnn_graph_dir) {
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseMultiply", /*count=*/0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron", /*count=*/1);
}

// Runs the model and asserts the HardSigmoid+Mul pair fused into a single HardSwish:
// the standalone Mul (ElementWiseMultiply) must be gone, replaced by one
// ElementWiseNeuron (HardSwish).
void RunAndAssertFused(const TestInputDef<float>& input_def, bool hardsigmoid_first,
                       const std::filesystem::path& json_qnn_graph_dir) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetHtpProviderOptions(json_qnn_graph_dir);

  RunQnnModelTest(BuildHardSigmoidMulTestCase(input_def, hardsigmoid_first),
                  provider_options,
                  /*opset_version=*/18,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       // fp16 (QNN) vs fp32 (CPU EP).
                                       ElementwiseAbsoluteVerifier(0.01f)});

  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertHardSwishFusion(json_qnn_graph_dir);
}

}  // namespace

// Test FP32 fusion of HardSigmoid into HardSwish on the HTP backend with
// enable_htp_fp16_precision enabled.
TEST_F(QnnHTPBackendTests, HardSigmoidFusedIntoHardSwish_FP32_as_FP16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = MakeHardSigmoidAccuracyInputDef();
  RunAndAssertFused(input_def, /*hardsigmoid_first=*/true,
                    "HardSigmoidFusedIntoHardSwish_FP32_as_FP16");
}

// Test FP16 fusion of HardSigmoid into HardSwish on the HTP backend.
TEST_F(QnnHTPBackendTests, HardSigmoidFusedIntoHardSwish_FP16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "HardSigmoidFusedIntoHardSwish_FP16";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options =
      GetHtpProviderOptions(json_qnn_graph_dir, /*enable_htp_fp16_precision=*/false);
  // TestFp16ModelAccuracy uses relative error, which is undefined at an exact-zero reference.
  // Keep this FP16-specific input away from zero; the FP32-as-FP16 test above uses an absolute
  // verifier and still covers exact-zero inputs.
  auto input_def = TestInputDef<float>({2, 2, 2, 2}, false,
                                       {-8.0f, -2.0f, -0.1f, 0.5f, 0.9f, 1.1f, 3.3f, 8.0f,
                                        -7.0f, 0.1f, 0.2f, 0.4f, 0.8f, 2.1f, 4.3f, 7.0f});
  auto input_fp16_def = ConvertToFP16InputDef(input_def);

  auto model_fp32_fn = BuildHardSigmoidMulTestCase(input_def, /*hardsigmoid_first=*/true);
  auto model_fp16_fn = BuildHardSigmoidMulTestCase(input_fp16_def, /*hardsigmoid_first=*/true);

  TestFp16ModelAccuracy(model_fp32_fn,
                        model_fp16_fn,
                        provider_options,
                        /*opset_version=*/18,
                        ExpectedEPNodeAssignment::All,
                        0.005f);

  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertHardSwishFusion(json_qnn_graph_dir);
}

// HardSigmoid -> Mul(input, hsig_out): HardSigmoid output is the SECOND Mul input.
// This is the ordering the original same_root_input check already handled.
TEST_F(QnnHTPBackendTests, HardSigmoidMulFusion_NormalOrder_Fuses) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 2, 4}, false, GetFloatDataInRange(-5.0f, 5.0f, 16));
  RunAndAssertFused(input_def, /*hardsigmoid_first=*/false, "HardSigmoidMulFusion_NormalOrder");
}

// HardSigmoid -> Mul(hsig_out, input): HardSigmoid output is the FIRST Mul input.
// Reproduces the copy-paste bug: the same_root_input check compared Mul.Inputs()[0]
// against the HardSigmoid input on both sides of the ||, so this (valid) ordering
// failed to match and the pattern was NOT fused. This asserts the regression path
// directly against the generated QNN graph.
TEST_F(QnnHTPBackendTests, HardSigmoidMulFusion_ReversedOrder_Fuses) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 2, 4}, false, GetFloatDataInRange(-5.0f, 5.0f, 16));
  RunAndAssertFused(input_def, /*hardsigmoid_first=*/true, "HardSigmoidMulFusion_ReversedOrder");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
