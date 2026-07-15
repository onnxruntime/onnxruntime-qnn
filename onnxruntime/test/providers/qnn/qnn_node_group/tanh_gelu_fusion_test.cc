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

namespace {

// Builds the canonical tanh-GELU approximation pattern (float32):
//
//  [x] --+-> Mul(x,x) -> Mul(x²,x) -> Mul(0.044715) --+
//        |                                              v
//        +--------------------------------------------> Add -> Mul(sqrt2pi) -> Tanh -> Add(1) -> Mul(x) -> Mul(0.5) ==>
//
// Equation: 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
GetTestModelFn BuildTanhGeluTestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    constexpr float k0044715 = 0.044715f;
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kOne = 1.0f;
    constexpr float kHalf = 0.5f;

    builder.graph_->set_name("tanh_gelu_graph");
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("Mul_x2", "Mul", {"input", "input"}, {"x2_out"}, kOnnxDomain);
    builder.AddNode("Mul_x3", "Mul", {"x2_out", "input"}, {"x3_out"}, kOnnxDomain);

    builder.MakeScalarInitializer<float>("c0044715", k0044715);
    builder.AddNode("Mul_0044715", "Mul", {"x3_out", "c0044715"}, {"mul_0044715_out"}, kOnnxDomain);

    // Canonical Add: Add(input, 0.044715*x³)  — x is input[0], cubic branch is input[1]
    builder.AddNode("Add_inner", "Add", {"input", "mul_0044715_out"}, {"add_inner_out"}, kOnnxDomain);

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
}

// Same as BuildTanhGeluTestCase but with commutative-input orderings swapped at two places
// that exercise the matcher's `for i` (add_inner branch index) and `for j` (mul_x3 child order):
//
//  add_inner   : Add(0.044715*x³, x)  — cubic branch is input[0] instead of input[1]  (for i loop)
//  mul_x3      : Mul(x, x²)           — x is input[0] instead of input[1]              (for j loop)
GetTestModelFn BuildTanhGeluTestCaseSwappedInputs(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    constexpr float k0044715 = 0.044715f;
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kOne = 1.0f;
    constexpr float kHalf = 0.5f;

    builder.graph_->set_name("tanh_gelu_swapped_graph");
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("Mul_x2", "Mul", {"input", "input"}, {"x2_out"}, kOnnxDomain);

    // for j: Mul(x, x²) — x in slot 0, x² in slot 1 (swapped vs canonical Mul(x²,x))
    builder.AddNode("Mul_x3", "Mul", {"input", "x2_out"}, {"x3_out"}, kOnnxDomain);

    builder.MakeScalarInitializer<float>("c0044715", k0044715);
    builder.AddNode("Mul_0044715", "Mul", {"x3_out", "c0044715"}, {"mul_0044715_out"}, kOnnxDomain);

    // for i: Add(0.044715*x³, x) — cubic branch is input[0] (swapped vs canonical Add(x, 0.044715*x³))
    builder.AddNode("Add_inner", "Add", {"mul_0044715_out", "input"}, {"add_inner_out"}, kOnnxDomain);

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
}

// Builds the tanh-GELU pattern using fp16 scalar constants.
// GetScalarConstantValue supports FLOAT16 element type, so fusion must also fire for fp16 models.
GetTestModelFn BuildTanhGeluTestCaseFp16(const TestInputDef<float>& float_input_def) {
  return [float_input_def](ModelTestBuilder& builder) -> void {
    const Ort::Float16_t k0044715 = static_cast<Ort::Float16_t>(0.044715f);
    const Ort::Float16_t kSqrt2OverPi = static_cast<Ort::Float16_t>(0.7978845608f);
    const Ort::Float16_t kOne = static_cast<Ort::Float16_t>(1.0f);
    const Ort::Float16_t kHalf = static_cast<Ort::Float16_t>(0.5f);

    builder.graph_->set_name("tanh_gelu_fp16_graph");
    const TestInputDef<Ort::Float16_t> input_def = ConvertToFP16InputDef(float_input_def);
    MakeTestInput<Ort::Float16_t>(builder, "input", input_def);

    builder.AddNode("Mul_x2", "Mul", {"input", "input"}, {"x2_out"}, kOnnxDomain);
    builder.AddNode("Mul_x3", "Mul", {"x2_out", "input"}, {"x3_out"}, kOnnxDomain);

    builder.MakeScalarInitializer<Ort::Float16_t>("c0044715", k0044715);
    builder.AddNode("Mul_0044715", "Mul", {"x3_out", "c0044715"}, {"mul_0044715_out"}, kOnnxDomain);

    builder.AddNode("Add_inner", "Add", {"input", "mul_0044715_out"}, {"add_inner_out"}, kOnnxDomain);

    builder.MakeScalarInitializer<Ort::Float16_t>("sqrt2pi", kSqrt2OverPi);
    builder.AddNode("Mul_coeff", "Mul", {"add_inner_out", "sqrt2pi"}, {"mul_coeff_out"}, kOnnxDomain);

    builder.AddNode("Tanh", "Tanh", {"mul_coeff_out"}, {"tanh_out"}, kOnnxDomain);

    builder.MakeScalarInitializer<Ort::Float16_t>("one", kOne);
    builder.AddNode("Add_one", "Add", {"tanh_out", "one"}, {"add_one_out"}, kOnnxDomain);

    builder.AddNode("Mul_x", "Mul", {"input", "add_one_out"}, {"mul_x_out"}, kOnnxDomain);

    builder.MakeScalarInitializer<Ort::Float16_t>("half", kHalf);
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

// Runs a positive fusion test: verifies fusion fires (one ElementWiseNeuron in the QNN graph)
// and that QNN output matches CPU EP within abs_err.
void RunTanhGeluFusionTest(const std::string& test_name,
                           const GetTestModelFn& build_fn,
                           float abs_err) {
  const std::filesystem::path json_qnn_graph_dir = test_name;
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(build_fn,
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(abs_err)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
// ---- Positive tests --------------------------------------------------------

// 4D input: typical batch * channel * H * W layout.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_Float32_4D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTanhGeluFusionTest("TanhGeluFusion_Float32_4D",
                        BuildTanhGeluTestCase(TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f)),
                        6e-3f);
}

// 3D input: typical transformer hidden-size shape {batch, seq, hidden}.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_Float32_3D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTanhGeluFusionTest("TanhGeluFusion_Float32_3D",
                        BuildTanhGeluTestCase(TestInputDef<float>({1, 128, 768}, false, -1.5f, 1.5f)),
                        2e-3f);
}

// 2D input: typical post-flatten shape in a linear layer.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_Float32_2D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTanhGeluFusionTest("TanhGeluFusion_Float32_2D",
                        BuildTanhGeluTestCase(TestInputDef<float>({32, 512}, false, -1.5f, 1.5f)),
                        6e-3f);
}

// Commutative-input-ordering test:
//   add_inner  = Add(0.044715*x³, x)   — cubic arm in slot 0  → exercises `for i` in TryMatchCubicSubtree
//   mul_x3     = Mul(x, x²)            — x in slot 0          → exercises `for j` inside TryMatchCubicSubtree
TEST_F(QnnHTPBackendTests, TanhGeluFusion_Float32_SwappedInputs) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTanhGeluFusionTest("TanhGeluFusion_Float32_SwappedInputs",
                        BuildTanhGeluTestCaseSwappedInputs(TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f)),
                        6e-3f);
}

// fp16 test: scalar constants are FLOAT16 initializers.
// GetScalarConstantValue handles FLOAT16 via Ort::Float16_t::ToFloat(), so fusion must fire.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_Float16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTanhGeluFusionTest("TanhGeluFusion_Float16",
                        BuildTanhGeluTestCaseFp16(TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f)),
                        6e-3f);
}

// Peak substitution error: input range [-3, 3] covers |x|≈2 where the divergence between the
// tanh approximation and the exact-erf GELU is largest (~4.7e-4). Verifies the fused QNN GELU
// output still matches CPU tanh-GELU within the documented tolerance.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_Float32_PeakSubstitutionError) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTanhGeluFusionTest("TanhGeluFusion_Float32_PeakSubstitutionError",
                        BuildTanhGeluTestCase(TestInputDef<float>({1, 2, 3, 4}, false, -3.0f, 3.0f)),
                        6e-3f);
}

// ---- Negative tests --------------------------------------------------------
// Fusion is rejected but Mul/Add/Tanh are individually supported on HTP, so
// individual ops still run on QNN EP → ExpectedEPNodeAssignment::Some.

// Wrong cubic coefficient (0.1 instead of 0.044715) — matcher must reject.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_WrongCoeff_ShouldNotFuse) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  GetTestModelFn bad_model = [&input_def](ModelTestBuilder& builder) -> void {
    constexpr float kWrongCoeff = 0.1f;
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
                  GetProviderOptions(),
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(6e-3f)});
}

// Shared backward intermediate: x² is also consumed by an extra Relu outside the pattern.
// HasSingleOutputConsumer rejects it, so the matcher must reject the full pattern.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_SharedIntermediate_ShouldNotFuse) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  GetTestModelFn shared_model = [&input_def](ModelTestBuilder& builder) -> void {
    constexpr float k0044715 = 0.044715f;
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kOne = 1.0f;
    constexpr float kHalf = 0.5f;

    builder.graph_->set_name("tanh_gelu_shared_graph");
    MakeTestInput<float>(builder, "input", input_def);

    // x² is shared: consumed by both Mul(x²,x) and the extra Relu below.
    builder.AddNode("Mul_x2", "Mul", {"input", "input"}, {"x2_out"}, kOnnxDomain);

    // Extra consumer of x² — prevents fusion.
    builder.AddNode("Relu_extra", "Relu", {"x2_out"}, {"extra_out"}, kOnnxDomain);
    builder.MakeOutput("extra_out");

    builder.AddNode("Mul_x3", "Mul", {"x2_out", "input"}, {"x3_out"}, kOnnxDomain);
    builder.MakeScalarInitializer<float>("c0044715", k0044715);
    builder.AddNode("Mul_0044715", "Mul", {"x3_out", "c0044715"}, {"mul_0044715_out"}, kOnnxDomain);
    builder.AddNode("Add_inner", "Add", {"input", "mul_0044715_out"}, {"add_inner_out"}, kOnnxDomain);
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

  RunQnnModelTest(shared_model,
                  GetProviderOptions(),
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(6e-3f)});
}

// Shared forward intermediate: add_one_out has a second consumer (an extra Relu) outside the pattern.
// GetOnlyChildOfOutput rejects multi-consumer outputs, so the forward walk fails.
TEST_F(QnnHTPBackendTests, TanhGeluFusion_SharedForwardIntermediate_ShouldNotFuse) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  GetTestModelFn forward_shared_model = [&input_def](ModelTestBuilder& builder) -> void {
    constexpr float k0044715 = 0.044715f;
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kOne = 1.0f;
    constexpr float kHalf = 0.5f;

    builder.graph_->set_name("tanh_gelu_forward_shared_graph");
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("Mul_x2", "Mul", {"input", "input"}, {"x2_out"}, kOnnxDomain);
    builder.AddNode("Mul_x3", "Mul", {"x2_out", "input"}, {"x3_out"}, kOnnxDomain);
    builder.MakeScalarInitializer<float>("c0044715", k0044715);
    builder.AddNode("Mul_0044715", "Mul", {"x3_out", "c0044715"}, {"mul_0044715_out"}, kOnnxDomain);
    builder.AddNode("Add_inner", "Add", {"input", "mul_0044715_out"}, {"add_inner_out"}, kOnnxDomain);
    builder.MakeScalarInitializer<float>("sqrt2pi", kSqrt2OverPi);
    builder.AddNode("Mul_coeff", "Mul", {"add_inner_out", "sqrt2pi"}, {"mul_coeff_out"}, kOnnxDomain);
    builder.AddNode("Tanh", "Tanh", {"mul_coeff_out"}, {"tanh_out"}, kOnnxDomain);
    builder.MakeScalarInitializer<float>("one", kOne);
    builder.AddNode("Add_one", "Add", {"tanh_out", "one"}, {"add_one_out"}, kOnnxDomain);

    // Extra consumer of add_one_out — prevents fusion of the forward tail.
    builder.AddNode("Relu_extra", "Relu", {"add_one_out"}, {"extra_out"}, kOnnxDomain);
    builder.MakeOutput("extra_out");

    builder.AddNode("Mul_x", "Mul", {"input", "add_one_out"}, {"mul_x_out"}, kOnnxDomain);
    builder.MakeScalarInitializer<float>("half", kHalf);
    builder.AddNode("Mul_half", "Mul", {"mul_x_out", "half"}, {"output"}, kOnnxDomain);
    builder.MakeOutput("output");
  };

  RunQnnModelTest(forward_shared_model,
                  GetProviderOptions(),
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(6e-3f)});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
