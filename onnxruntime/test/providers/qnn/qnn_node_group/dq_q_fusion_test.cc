// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cstdint>
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

constexpr float kBetaScale = 0.0078125f;
constexpr uint8_t kBetaZeroPoint = 128;

// Fan-out bias: DQ feeds a requant chain and a Mul.
//   beta -> Q -> DQ -> Q(requant) -> DQ -> Add(x, .)
//                  \-> Mul(x, .)
GetTestModelFn BuildConstantDQQTestCase(float requant_scale) {
  return [requant_scale](ModelTestBuilder& builder) {
    const TestInputDef<float> input_def({1, 16, 4, 4}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "x", input_def);
    const std::string x_dq = AddQDQNodePair<uint8_t>(builder, "x_qdq", "x", 1.0f / 128.0f, 128);

    builder.MakeInitializer<float>("beta", {1, 16, 1, 1}, -0.9f, 0.9f);
    const std::string beta_dq = AddQDQNodePair<uint8_t>(builder, "beta_qdq", "beta", kBetaScale, kBetaZeroPoint);
    const std::string bias = AddQDQNodePair<uint8_t>(builder, "bias_qdq", beta_dq, requant_scale, kBetaZeroPoint);

    builder.AddNode("add", "Add", {x_dq, bias}, {"add_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "add_qdq", "add_out", 1.0f / 64.0f, 128);
    builder.AddNode("mul", "Mul", {x_dq, beta_dq}, {"mul_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "mul_qdq", "mul_out", 1.0f / 128.0f, 128);
  };
}

// A transposed MatMul weight behind a requantizing DQ -> Q; ORT's transpose optimizer does not push a
// Transpose through MatMul, so only the EP can fold it.
//   w -> Q -> DQ -> Q(requant) -> Transpose -> DQ -> MatMul(x, .)
GetTestModelFn BuildConstantDQQTransposeTestCase(float requant_scale) {
  return [requant_scale](ModelTestBuilder& builder) {
    const TestInputDef<float> input_def({4, 3}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "x", input_def);
    const std::string x_dq = AddQDQNodePair<uint8_t>(builder, "x_qdq", "x", 1.0f / 128.0f, 128);

    builder.MakeInitializer<float>("w", {2, 3}, -0.9f, 0.9f);
    const std::string w_dq = AddQDQNodePair<uint8_t>(builder, "w_qdq", "w", kBetaScale, kBetaZeroPoint);
    builder.AddQuantizeLinearNode<uint8_t>("w_requant_q", w_dq, requant_scale, kBetaZeroPoint, "w_requant");
    builder.AddNode("w_transpose", "Transpose", {"w_requant"}, {"w_t"}, "",
                    {builder.MakeIntsAttribute("perm", {1, 0})});
    builder.AddDequantizeLinearNode<uint8_t>("w_t_dq", "w_t", requant_scale, kBetaZeroPoint, "w_t_dq_out");

    builder.AddNode("matmul", "MatMul", {x_dq, "w_t_dq_out"}, {"matmul_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "matmul_qdq", "matmul_out", 1.0f / 64.0f, 128);
  };
}

// A [1, N] Gemm bias behind a requantizing DQ -> Q. A uint8 bias keeps the Gemm out of a QDQ group, so it
// receives the folded [1, N] bias directly.
//   c -> Q -> DQ -> Q(requant) -> DQ -> Gemm(a, b, .)
GetTestModelFn BuildConstantDQQGemmBiasTestCase(float requant_scale) {
  return [requant_scale](ModelTestBuilder& builder) {
    const TestInputDef<float> input_def({2, 8}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "a", input_def);
    const std::string a_dq = AddQDQNodePair<uint8_t>(builder, "a_qdq", "a", 1.0f / 128.0f, 128);

    builder.MakeInitializer<uint8_t>("b", {8, 4}, 0, 255);
    builder.AddDequantizeLinearNode<uint8_t>("b_dq_node", "b", 1.0f / 128.0f, 128, "b_dq");

    builder.MakeInitializer<float>("c", {1, 4}, -0.9f, 0.9f);
    const std::string c_dq = AddQDQNodePair<uint8_t>(builder, "c_qdq", "c", kBetaScale, kBetaZeroPoint);
    const std::string bias = AddQDQNodePair<uint8_t>(builder, "bias_qdq", c_dq, requant_scale, kBetaZeroPoint);

    builder.AddNode("gemm", "Gemm", {a_dq, "b_dq", bias}, {"gemm_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "gemm_qdq", "gemm_out", 1.0f / 64.0f, 128);
  };
}

// A runtime DQ -> Q pair after Add. The scale nudge prevents ORT's QDQ optimizer
// from removing the pair before QNN EP composes it as a Convert.
template <typename QuantType>
GetTestModelFn BuildRuntimeDQQTestCase(const TestInputDef<float>& input0_def,
                                       const TestInputDef<float>& input1_def,
                                       const QuantParams<QuantType>& output_qparams) {
  return [input0_def, input1_def, output_qparams](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input0", input0_def);
    MakeTestInput<float>(builder, "input1", input1_def);

    const QuantParams<QuantType> input0_qparams = GetTestInputQuantParams<QuantType>(input0_def);
    const std::string input0_after_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq0", "input0", input0_qparams.scale, input0_qparams.zero_point);

    const QuantParams<QuantType> input1_qparams = GetTestInputQuantParams<QuantType>(input1_def);
    const std::string input1_after_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq1", "input1", input1_qparams.scale, input1_qparams.zero_point);

    builder.AddNode("add", "Add", {input0_after_qdq, input1_after_qdq}, {"add_out"});

    QuantParams<QuantType> add_output_qparams = output_qparams;
    add_output_qparams.scale *= 1.01f;
    const std::string add_output_after_qdq =
        AddQDQNodePair<QuantType>(builder, "add_qdq", "add_out",
                                  add_output_qparams.scale, add_output_qparams.zero_point);

    builder.AddQuantizeLinearNode<QuantType>("final_q", add_output_after_qdq,
                                             output_qparams.scale, output_qparams.zero_point, "output");
    builder.MakeOutput("output");
  };
}

// The DQ output has two consumers, so GetOnlyChildOfType must reject the
// DQ -> Q fusion and leave the standalone Dequantize and Quantize nodes intact.
GetTestModelFn BuildRuntimeDQQExtraConsumerTestCase() {
  return [](ModelTestBuilder& builder) {
    MakeTestInput<uint8_t>(builder, "input",
                           TestInputDef<uint8_t>({1, 2, 2, 2}, false,
                                                 {0, 32, 64, 96, 128, 160, 192, 255}));

    builder.AddDequantizeLinearNode<uint8_t>("shared_dq", "input", 0.1f, 128, "shared_dq_out");
    builder.AddQuantizeLinearNode<uint8_t>("requant_q", "shared_dq_out", 0.2f, 120, "requant_output");
    builder.MakeOutput("requant_output");

    builder.AddNode("relu", "Relu", {"shared_dq_out"}, {"relu_output"});
    builder.MakeOutput("relu_output");
  };
}

void RunDQQTest(const GetTestModelFn& build_test_case,
                const std::filesystem::path& json_dir,
                const EPVerificationParams& verification_params =
                    EPVerificationParams{ExpectedEPNodeAssignment::All}) {
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(build_test_case, provider_options, /*opset_version=*/21, verification_params);
}

}  // namespace

TEST_F(QnnHTPBackendTests, DQQFusion_RuntimeU8_EmitsConvert) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "DQQFusion_RuntimeU8_EmitsConvert";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  const std::vector<float> input_data = {-8.0f, -6.0f, -2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  const TestInputDef<float> input0_def({1, 2, 2, 2}, false, input_data);
  const TestInputDef<float> input1_def({1, 2, 2, 2}, false, input_data);
  RunDQQTest(BuildRuntimeDQQTestCase<uint8_t>(input0_def, input1_def, {1.0f, 128}), json_dir);
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpWithNameInQnnGraph(json_dir, "Convert", "add_qdq_dq");
}

TEST_F(QnnHTPBackendTests, DQQFusion_RuntimeU16_EmitsConvert) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "DQQFusion_RuntimeU16_EmitsConvert";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  const std::vector<float> input_data = {-8.0f, -6.0f, -2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  const TestInputDef<float> input0_def({1, 2, 2, 2}, false, input_data);
  const TestInputDef<float> input1_def({1, 2, 2, 2}, false, input_data);
  RunDQQTest(BuildRuntimeDQQTestCase<uint16_t>(input0_def, input1_def, {1.0f, 32768}), json_dir);
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpWithNameInQnnGraph(json_dir, "Convert", "add_qdq_dq");
}

TEST_F(QnnHTPBackendTests, DQQFusion_ExtraConsumer_DoesNotFuse) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "DQQFusion_ExtraConsumer_DoesNotFuse";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunDQQTest(BuildRuntimeDQQExtraConsumerTestCase(), json_dir,
             EPVerificationParams{ExpectedEPNodeAssignment::All,
                                  ElementwiseAbsoluteVerifier{1e-2f}});
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpWithNameInQnnGraph(json_dir, "Dequantize", "shared_dq");
}

// A runtime Convert here would turn the constant bias into an activation.
TEST_F(QnnHTPBackendTests, DQQFusion_ConstantInput_FoldsToStaticTensor) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "DQQFusion_ConstantInput_FoldsToStaticTensor";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunDQQTest(BuildConstantDQQTestCase(kBetaScale), json_dir,
             EPVerificationParams{ExpectedEPNodeAssignment::All,
                                  ElementwiseAbsoluteVerifier{1.0f / 64.0f + 1e-5f}});
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpInQnnGraph(json_dir, "Convert", 0);
}

TEST_F(QnnHTPBackendTests, DQQFusion_ConstantInputRequantized_FoldsToStaticTensor) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "DQQFusion_ConstantInputRequantized_FoldsToStaticTensor";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunDQQTest(BuildConstantDQQTestCase(2 * kBetaScale), json_dir,
             EPVerificationParams{ExpectedEPNodeAssignment::All,
                                  ElementwiseAbsoluteVerifier{1.0f / 64.0f + 1e-5f}});
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpInQnnGraph(json_dir, "Convert", 0);
}

TEST_F(QnnHTPBackendTests, DQQFusion_ConstantInputTransposed_FoldsTranspose) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "DQQFusion_ConstantInputTransposed_FoldsTranspose";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunDQQTest(BuildConstantDQQTransposeTestCase(2 * kBetaScale), json_dir,
             EPVerificationParams{ExpectedEPNodeAssignment::All,
                                  ElementwiseAbsoluteVerifier{1.0f / 64.0f + 1e-5f}});
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpInQnnGraph(json_dir, "Convert", 0);
  AssertOpInQnnGraph(json_dir, "Transpose", 0);
}

TEST_F(QnnHTPBackendTests, DQQFusion_ConstantGemmBias_FoldsToStaticTensor) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "DQQFusion_ConstantGemmBias_FoldsToStaticTensor";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunDQQTest(BuildConstantDQQGemmBiasTestCase(2 * kBetaScale), json_dir,
             EPVerificationParams{ExpectedEPNodeAssignment::All,
                                  ElementwiseAbsoluteVerifier{1.0f / 64.0f + 1e-5f}});
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpInQnnGraph(json_dir, "Convert", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
