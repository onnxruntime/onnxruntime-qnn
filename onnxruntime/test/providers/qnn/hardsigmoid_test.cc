// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <optional>
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ============================================================
// HardSigmoid tests
// ============================================================

TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_QU8) {
  RunQdqModelOnHtp<uint8_t>("HardSigmoid",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {utils::MakeAttribute("alpha", 0.1f),
                             utils::MakeAttribute("beta", 0.4f)},
                            21,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_QU16) {
  RunQdqModelOnHtp<uint16_t>("HardSigmoid",
                             {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                             {},
                             21,
                             ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_QDQ_Supported) {
  RunQdqModelOnHtp<uint8_t>("HardSigmoid",
                            {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f)},
                            {},
                            19,
                            ExpectedEPNodeAssignment::All);
}

// Enables running f32 ops using fp16 precision.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_FP32_as_FP16) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 16);

  // Rank 3, default alpha/beta.
  // Tolerance: comparing fp16 (QNN) with fp32 (CPU EP).
  RunF32ModelOnHtp<float>("HardSigmoid",
                          {TestInputDef<float>({1, 2, 8}, false, input_data)},
                          {},
                          21,
                          ExpectedEPNodeAssignment::All,
                          kOnnxDomain,
                          0.004f,
                          true);  // enable_htp_fp16_precision

  // Rank 4, non-default alpha and beta.
  RunF32ModelOnHtp<float>("HardSigmoid",
                          {TestInputDef<float>({1, 2, 2, 4}, false, input_data)},
                          {utils::MakeAttribute("alpha", 0.1f),
                           utils::MakeAttribute("beta", 0.4f)},
                          21,
                          ExpectedEPNodeAssignment::All,
                          kOnnxDomain,
                          0.004f,
                          true);  // enable_htp_fp16_precision
}

TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_FP16) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 16);

  RunF16ModelOnHtp("HardSigmoid",
                   {TestInputDef<float>({1, 2, 8}, false, input_data)},
                   {},
                   21,
                   ExpectedEPNodeAssignment::All,
                   kOnnxDomain);

  // Rank 4, non-default alpha and beta.
  RunF16ModelOnHtp("HardSigmoid",
                   {TestInputDef<float>({1, 2, 2, 4}, false, input_data)},
                   {utils::MakeAttribute("alpha", 0.1f),
                    utils::MakeAttribute("beta", 0.4f)},
                   21,
                   ExpectedEPNodeAssignment::All,
                   kOnnxDomain);
}

// ============================================================
// HardSigmoid fused into HardSwish tests
// ============================================================

// Returns a function that creates the model `X * HardSigmoid(X)`, which can be
// potentially fused into a single HardSwish(X) operator.
template <typename FloatType>
static GetTestModelFn BuildHardSigmoidFusionTestCase(TestInputDef<FloatType>& input_def,
                                                     std::optional<float> alpha,
                                                     std::optional<float> beta) {
  return [input_def, alpha, beta](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput<FloatType>(builder, input_def);

    NodeArg* hs_output = builder.MakeIntermediate();
    Node& hs_node = builder.AddNode("HardSigmoid", {input}, {hs_output});

    if (alpha.has_value()) {
      hs_node.AddAttribute("alpha", alpha.value());
    }
    if (beta.has_value()) {
      hs_node.AddAttribute("beta", beta.value());
    }

    auto* output = builder.MakeOutput();
    builder.AddNode("Mul", {hs_output, input}, {output});
  };
}

// Test FP32 fusion of HardSigmoid into HardSwish on HTP with enable_htp_fp16_precision.
TEST_F(QnnHTPBackendTests, HardSigmoidFusedIntoHardSwish_FP32_as_FP16) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
#if defined(_WIN32)
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "Test requires HTP FP16 support (arch > V68).";
  }
#endif
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  provider_options["enable_htp_fp16_precision"] = "1";

  std::vector<float> input_data = {-8.0f, -2.0f, 0.0f, 0.5f, 0.9f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, 0.0f, 0.2f, 0.4f, 0.8f, 2.1f, 4.3f, 7.0f};
  auto input_def = TestInputDef<float>({2, 2, 2, 2}, false, input_data);
  constexpr float alpha = 1.0f / 6.0f;
  constexpr float beta = 0.5f;
  auto model_fn = BuildHardSigmoidFusionTestCase<float>(input_def, alpha, beta);

  RunQnnModelTest(model_fn,
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  0.01f);  // abs err: comparing fp16 (QNN) vs fp32 (CPU EP).
}

TEST_F(QnnHTPBackendTests, HardSigmoidFusedIntoHardSwish_FP16) {
#if defined(_WIN32)
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "Test requires HTP FP16 support (arch > V68).";
  }
#endif
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  std::vector<float> input_data = {-8.0f, -2.0f, 0.0f, 0.5f, 0.9f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, 0.0f, 0.2f, 0.4f, 0.8f, 2.1f, 4.3f, 7.0f};
  auto input_def = TestInputDef<float>({2, 2, 2, 2}, false, input_data);
  auto input_fp16_def = ConvertToFP16InputDef(input_def);

  constexpr float alpha = 1.0f / 6.0f;
  constexpr float beta = 0.5f;
  auto model_fp32_fn = BuildHardSigmoidFusionTestCase<float>(input_def, alpha, beta);
  auto model_fp16_fn = BuildHardSigmoidFusionTestCase<MLFloat16>(input_fp16_def, alpha, beta);

  TestFp16ModelAccuracy(model_fp32_fn,
                        model_fp16_fn,
                        provider_options,
                        18,
                        ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
