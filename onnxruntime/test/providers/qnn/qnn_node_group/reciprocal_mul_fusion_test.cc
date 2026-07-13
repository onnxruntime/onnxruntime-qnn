// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Tests for ReciprocalMulFusion: validates fusion of Reciprocal->Mul into ElementWiseBinary (DIVIDE).

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Builds Reciprocal->Mul pattern (commute controls Mul input order).
GetTestModelFn BuildReciprocalMulTestCase(const TestInputDef<float>& numerator_def,
                                          const TestInputDef<float>& denominator_def,
                                          bool commute = false) {
  return [numerator_def, denominator_def, commute](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reciprocal_mul_fusion_graph");

    MakeTestInput<float>(builder, "numerator", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    // denominator -> Reciprocal -> recip_out
    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {"denominator"},
                    {"recip_out"},
                    kOnnxDomain);

    // Mul(numerator, recip_out)  or  Mul(recip_out, numerator)
    std::vector<std::string> mul_inputs = commute
                                              ? std::vector<std::string>{"recip_out", "numerator"}
                                              : std::vector<std::string>{"numerator", "recip_out"};

    builder.AddNode("Mul_node",
                    "Mul",
                    mul_inputs,
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// FP16 variant of fusion pattern.
GetTestModelFn BuildReciprocalMulFP16TestCase(const TestInputDef<float>& numerator_def,
                                              const TestInputDef<float>& denominator_def,
                                              bool commute = false) {
  const TestInputDef<Ort::Float16_t> num_fp16_def = ConvertToFP16InputDef(numerator_def);
  const TestInputDef<Ort::Float16_t> den_fp16_def = ConvertToFP16InputDef(denominator_def);

  return [num_fp16_def, den_fp16_def, commute](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reciprocal_mul_fp16_fusion_graph");

    MakeTestInput<Ort::Float16_t>(builder, "numerator", num_fp16_def);
    MakeTestInput<Ort::Float16_t>(builder, "denominator", den_fp16_def);

    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {"denominator"},
                    {"recip_out"},
                    kOnnxDomain);

    std::vector<std::string> mul_inputs = commute
                                              ? std::vector<std::string>{"recip_out", "numerator"}
                                              : std::vector<std::string>{"numerator", "recip_out"};

    builder.AddNode("Mul_node",
                    "Mul",
                    mul_inputs,
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}
// No-fusion case: QDQ-wrapped Reciprocal with two Mul consumers.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQReciprocalMulNoFusionTestCase(
    const TestInputDef<float>& numerator_def,
    const TestInputDef<float>& denominator_def,
    bool use_contrib_qdq = false) {
  return [numerator_def, denominator_def, use_contrib_qdq](
             ModelTestBuilder& builder,
             std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    builder.graph_->set_name("qdq_reciprocal_qdq_wrapped_no_fusion_graph");

    MakeTestInput<float>(builder, "numerator_a", numerator_def);
    MakeTestInput<float>(builder, "numerator_b", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    const QuantParams<QuantType> num_qparams = GetTestInputQuantParams<QuantType>(numerator_def);
    const QuantParams<QuantType> den_qparams = GetTestInputQuantParams<QuantType>(denominator_def);

    const std::string num_a_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_num_a", "numerator_a", num_qparams.scale, num_qparams.zero_point, use_contrib_qdq);
    const std::string num_b_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_num_b", "numerator_b", num_qparams.scale, num_qparams.zero_point, use_contrib_qdq);
    const std::string den_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_den", "denominator", den_qparams.scale, den_qparams.zero_point, use_contrib_qdq);

    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {den_qdq},
                    {"recip_out"},
                    kOnnxDomain);

    const QuantParams<QuantType> recip_qparams = GetTestInputQuantParams<QuantType>(denominator_def);
    const std::string recip_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_recip", "recip_out",
        recip_qparams.scale, recip_qparams.zero_point, use_contrib_qdq);

    builder.AddNode("Mul_A",
                    "Mul",
                    {num_a_qdq, recip_qdq},
                    {"mul_out_a"},
                    kOnnxDomain);

    builder.AddNode("Mul_B",
                    "Mul",
                    {num_b_qdq, recip_qdq},
                    {"mul_out_b"},
                    kOnnxDomain);

    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "qdq_out_a", "mul_out_a",
        output_qparams[0].scale, output_qparams[0].zero_point, use_contrib_qdq);
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "qdq_out_b", "mul_out_b",
        output_qparams[1].scale, output_qparams[1].zero_point, use_contrib_qdq);
  };
}

// No-fusion case: Reciprocal with two Mul consumers.
GetTestModelFn BuildReciprocalTwoConsumersTestCase(const TestInputDef<float>& numerator_def,
                                                   const TestInputDef<float>& denominator_def) {
  return [numerator_def, denominator_def](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reciprocal_two_consumers_graph");

    MakeTestInput<float>(builder, "numerator_a", numerator_def);
    MakeTestInput<float>(builder, "numerator_b", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {"denominator"},
                    {"recip_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_A",
                    "Mul",
                    {"numerator_a", "recip_out"},
                    {"out_a"},
                    kOnnxDomain);

    builder.AddNode("Mul_B",
                    "Mul",
                    {"numerator_b", "recip_out"},
                    {"out_b"},
                    kOnnxDomain);

    builder.MakeOutput("out_a");
    builder.MakeOutput("out_b");
  };
}

// No-fusion case: Reciprocal output is a graph output.
GetTestModelFn BuildReciprocalOutputIsGraphOutputTestCase(const TestInputDef<float>& numerator_def,
                                                          const TestInputDef<float>& denominator_def) {
  return [numerator_def, denominator_def](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reciprocal_output_is_graph_output_graph");

    MakeTestInput<float>(builder, "numerator", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {"denominator"},
                    {"recip_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_node",
                    "Mul",
                    {"numerator", "recip_out"},
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("recip_out");
    builder.MakeOutput("output");
  };
}

// No-fusion case: Both Mul inputs are the same Reciprocal output.
GetTestModelFn BuildReciprocalBothMulInputsSameTestCase(const TestInputDef<float>& denominator_def) {
  return [denominator_def](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reciprocal_both_mul_inputs_same_graph");

    MakeTestInput<float>(builder, "denominator", denominator_def);

    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {"denominator"},
                    {"recip_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_node",
                    "Mul",
                    {"recip_out", "recip_out"},
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return provider_options;
}

}  // namespace

TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_Float32_4D_StandardOrder) {
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "FP32 HTP test skipped on architecture <= 68";
  }

  const std::filesystem::path json_qnn_graph_dir = "ReciprocalMulFusion_Float32_4D_StandardOrder";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  RunQnnModelTest(BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/false),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseBinary", 1);
}

TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_Float32_4D_CommutedOrder) {
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "FP32 HTP test skipped on architecture <= 68";
  }

  const std::filesystem::path json_qnn_graph_dir = "ReciprocalMulFusion_Float32_4D_CommutedOrder";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  RunQnnModelTest(BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/true),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseBinary", 1);
}
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_FP16) {
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "uint16 QDQ requires HTP arch > v68";
  }

  const std::filesystem::path json_qnn_graph_dir = "ReciprocalMulFusion_FP16";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  const auto fp32_model_fn = BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/false);
  const auto fp16_model_fn = BuildReciprocalMulFP16TestCase(numerator_def, denominator_def, /*commute=*/false);

  TestFp16ModelAccuracy(fp32_model_fn,
                        fp16_model_fn,
                        provider_options,
                        /*opset_version=*/13,
                        /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                        /*tolerance=*/0.004f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseBinary", /*count=*/1);
}

TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_ReciprocalOutputIsGraphOutput_NoFusion) {
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "FP32 HTP test skipped on architecture <= 68";
  }

  const std::filesystem::path json_qnn_graph_dir = "ReciprocalMulFusion_ReciprocalOutputIsGraphOutput_NoFusion";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  RunQnnModelTest(BuildReciprocalOutputIsGraphOutputTestCase(numerator_def, denominator_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseBinary", /*count=*/2);
}

TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_QDQWrappedReciprocal_TwoConsumers_NoFusion) {
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "QDQ test skipped on HTP architecture <= 68";
  }

  const std::filesystem::path json_qnn_graph_dir = "ReciprocalMulFusion_QDQWrappedReciprocal_TwoConsumers_NoFusion";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  TestQDQModelAccuracy(
      BuildReciprocalTwoConsumersTestCase(numerator_def, denominator_def),
      BuildQDQReciprocalMulNoFusionTestCase<uint8_t>(numerator_def, denominator_def),
      provider_options,
      /*opset_version=*/13,
      /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseBinary", /*count=*/3);
}

TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_BothMulInputsSame_NoFusion) {
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "FP32 HTP test skipped on architecture <= 68";
  }

  const std::filesystem::path json_qnn_graph_dir = "ReciprocalMulFusion_BothMulInputsSame_NoFusion";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  RunQnnModelTest(BuildReciprocalBothMulInputsSameTestCase(denominator_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/3e-3f);

  // Should NOT fuse: fusion expects pattern a * (1/b) = a/b, but Mul(1/b, 1/b) = 1/b²
  // is a different semantic pattern (squaring the reciprocal).
  // Expect: 1 ElementWiseBinary for Reciprocal (1/b), 1 ElementWiseBinary for Mul (multiply)
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseBinary", /*count=*/2);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
