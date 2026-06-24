// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

// Builds the tanh-GELU approximation pattern:
//
//  [x] --+-> Mul(x,x) -> Mul(x²,x) -> Mul(0.044715) -> Add --+-> Mul(sqrt2pi) -> Tanh -> Add(1) -> Mul(x) -> Mul(0.5) ==>
//        |                                                     |
//        +-----------------------------------------------------+
//
// This matches what ORT's optimizer produces (Pow(x,3) is lowered to Mul(Mul(x,x),x)).
GetTestModelFn BuildTanhGeluTestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    constexpr float k0044715 = 0.044715f;
    constexpr float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/pi)
    constexpr float kOne = 1.0f;
    constexpr float kHalf = 0.5f;

    builder.graph_->set_name("tanh_gelu_graph");

    MakeTestInput<float>(builder, "input", input_def);

    // x² = Mul(x, x)
    builder.AddNode("Mul_x2", "Mul", {"input", "input"}, {"x2_out"}, kOnnxDomain);

    // x³ = Mul(x², x)
    builder.AddNode("Mul_x3", "Mul", {"x2_out", "input"}, {"x3_out"}, kOnnxDomain);

    // 0.044715 * x³
    builder.MakeScalarInitializer<float>("c0044715", k0044715);
    builder.AddNode("Mul_0044715", "Mul", {"x3_out", "c0044715"}, {"mul_0044715_out"}, kOnnxDomain);

    // x + 0.044715*x³
    builder.AddNode("Add_inner", "Add", {"input", "mul_0044715_out"}, {"add_inner_out"}, kOnnxDomain);

    // sqrt(2/pi) * (x + 0.044715*x³)
    builder.MakeScalarInitializer<float>("sqrt2pi", kSqrt2OverPi);
    builder.AddNode("Mul_coeff", "Mul", {"add_inner_out", "sqrt2pi"}, {"mul_coeff_out"}, kOnnxDomain);

    // Tanh
    builder.AddNode("Tanh", "Tanh", {"mul_coeff_out"}, {"tanh_out"}, kOnnxDomain);

    // 1 + Tanh(...)
    builder.MakeScalarInitializer<float>("one", kOne);
    builder.AddNode("Add_one", "Add", {"tanh_out", "one"}, {"add_one_out"}, kOnnxDomain);

    // x * (1 + Tanh(...))
    builder.AddNode("Mul_x", "Mul", {"input", "add_one_out"}, {"mul_x_out"}, kOnnxDomain);

    // 0.5 * x * (1 + Tanh(...))
    builder.MakeScalarInitializer<float>("half", kHalf);
    builder.AddNode("Mul_half", "Mul", {"mul_x_out", "half"}, {"output"}, kOnnxDomain);

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

// Basic pattern: verifies fusion fires and produces a single QNN Gelu node.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_Float32_4D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  const std::filesystem::path json_qnn_graph_dir = "TanhGeluFusion_Float32_4D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildTanhGeluTestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/6e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Typical transformer hidden-size shape {1, 128, 768}.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_Float32_3D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 128, 768}, false, -1.5f, 1.5f);

  const std::filesystem::path json_qnn_graph_dir = "TanhGeluFusion_Float32_3D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildTanhGeluTestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// 2D shape (e.g., after flatten in a linear layer).
TEST_F(QnnHTPBackendTests, TanhGeluFusion_Float32_2D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({32, 512}, false, -1.5f, 1.5f);

  const std::filesystem::path json_qnn_graph_dir = "TanhGeluFusion_Float32_2D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildTanhGeluTestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/6e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Negative test: a broken pattern (wrong coefficient) must NOT fuse.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_WrongCoeff_ShouldNotFuse) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  const std::filesystem::path json_qnn_graph_dir = "TanhGeluFusion_WrongCoeff";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Build pattern with wrong coefficient (0.1 instead of 0.044715).
  GetTestModelFn bad_model = [&input_def](ModelTestBuilder& builder) -> void {
    constexpr float kWrongCoeff = 0.1f;  // Should be 0.044715
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kOne = 1.0f;
    constexpr float kHalf = 0.5f;

    builder.graph_->set_name("tanh_gelu_wrong_coeff_graph");
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("Mul_x2", "Mul", {"input", "input"}, {"x2_out"}, kOnnxDomain);
    builder.AddNode("Mul_x3", "Mul", {"x2_out", "input"}, {"x3_out"}, kOnnxDomain);
    builder.MakeScalarInitializer<float>("wrong_coeff", kWrongCoeff);
    builder.AddNode("Mul_wrong", "Mul", {"x3_out", "wrong_coeff"}, {"mul_wrong_out"}, kOnnxDomain);
    builder.AddNode("Add_inner", "Add", {"input", "mul_wrong_out"}, {"add_inner_out"}, kOnnxDomain);
    builder.MakeScalarInitializer<float>("sqrt2pi", kSqrt2OverPi);
    builder.AddNode("Mul_coeff", "Mul", {"add_inner_out", "sqrt2pi"}, {"mul_coeff_out"}, kOnnxDomain);
    builder.AddNode("Tanh", "Tanh", {"mul_coeff_out"}, {"tanh_out"}, kOnnxDomain);
    builder.MakeScalarInitializer<float>("one", kOne);
    builder.AddNode("Add_one", "Add", {"tanh_out", "one"}, {"add_one_out"}, kOnnxDomain);
    builder.AddNode("Mul_x", "Mul", {"input", "add_one_out"}, {"mul_x_out"}, kOnnxDomain);
    builder.MakeScalarInitializer<float>("half", kHalf);
    builder.AddNode("Mul_half", "Mul", {"mul_x_out", "half"}, {"output"}, kOnnxDomain);
    builder.MakeOutput("output");
  };

  RunQnnModelTest(bad_model,
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/6e-3f);

  // No Gelu should appear in the QNN graph.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
