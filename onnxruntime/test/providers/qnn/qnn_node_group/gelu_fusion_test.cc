// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

void ResetQnnGraphDir(const std::filesystem::path& json_qnn_graph_dir) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
}

// Helper function to build GELU Pattern 1: root -> Mul -> Div/Mul -> Erf -> Add -> Mul
// Pattern 1:
//                   +-------Mul(0.5)---------------------+
//                   |                                    |
//                   |                                    v
//                [root] --> Div/Mul -----> Erf  --> Add --> Mul ==>
//                      (1.4142... or 1/1.4142...)   (1)
GetTestModelFn BuildGeluPattern1TestCase(const TestInputDef<float>& input_def, bool use_mul = false) {
  return [input_def, use_mul](ModelTestBuilder& builder) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float inv_sqrt_2 = 0.7071067690849304f;  // 1 / sqrt(2)
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name(use_mul ? "gelu_pattern1_mul_graph" : "gelu_pattern1_graph");

    // input
    MakeTestInput<float>(builder, "input", input_def);

    // root -> Mul(0.5) -> mul_half_out
    builder.MakeScalarInitializer<float>("half", half);
    builder.AddNode("Mul_half",
                    "Mul",
                    {"input", "half"},
                    {"mul_half_out"},
                    kOnnxDomain);

    // root -> Div(sqrt2) or Mul(1/sqrt2) -> norm_out
    if (use_mul) {
      builder.MakeScalarInitializer<float>("inv_sqrt2", inv_sqrt_2);
      builder.AddNode("Mul_inv_sqrt2",
                      "Mul",
                      {"input", "inv_sqrt2"},
                      {"norm_out"},
                      kOnnxDomain);
    } else {
      builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
      builder.AddNode("Div_sqrt2",
                      "Div",
                      {"input", "sqrt2"},
                      {"norm_out"},
                      kOnnxDomain);
    }

    // norm_out -> Erf -> erf_out
    builder.AddNode("Erf",
                    "Erf",
                    {"norm_out"},
                    {"erf_out"},
                    kOnnxDomain);

    // erf_out -> Add(1.0) -> add_out
    builder.MakeScalarInitializer<float>("one", one);
    builder.AddNode("Add_one",
                    "Add",
                    {"erf_out", "one"},
                    {"add_out"},
                    kOnnxDomain);

    // add_out * mul_half_out -> output
    builder.AddNode("Mul_out",
                    "Mul",
                    {"add_out", "mul_half_out"},
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Helper function to build GELU Pattern 2: Mul(0.5) after the main sequence
// Pattern 2:
//                   +------------------------------------+
//                   |                                    |
//                   |                                    v
//                [root] --> Div/Mul -----> Erf  --> Add --> Mul -->Mul ==>
//                          (1.4142... or 1/1.4142...)   (1)            (0.5)
GetTestModelFn BuildGeluPattern2TestCase(const TestInputDef<float>& input_def, bool use_mul = false) {
  return [input_def, use_mul](ModelTestBuilder& builder) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float inv_sqrt_2 = 0.7071067690849304f;  // 1 / sqrt(2)
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name(use_mul ? "gelu_pattern2_mul_graph" : "gelu_pattern2_graph");

    // input
    MakeTestInput<float>(builder, "input", input_def);

    // root -> Div(sqrt2) or Mul(1/sqrt2) -> norm_out
    if (use_mul) {
      builder.MakeScalarInitializer<float>("inv_sqrt2", inv_sqrt_2);
      builder.AddNode("Mul_inv_sqrt2",
                      "Mul",
                      {"input", "inv_sqrt2"},
                      {"norm_out"},
                      kOnnxDomain);
    } else {
      builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
      builder.AddNode("Div_sqrt2",
                      "Div",
                      {"input", "sqrt2"},
                      {"norm_out"},
                      kOnnxDomain);
    }

    // norm_out -> Erf -> erf_out
    builder.AddNode("Erf",
                    "Erf",
                    {"norm_out"},
                    {"erf_out"},
                    kOnnxDomain);

    // erf_out -> Add(1.0) -> add_out
    builder.MakeScalarInitializer<float>("one", one);
    builder.AddNode("Add_one",
                    "Add",
                    {"erf_out", "one"},
                    {"add_out"},
                    kOnnxDomain);

    // root * add_out -> mul_out
    builder.AddNode("Mul_input",
                    "Mul",
                    {"input", "add_out"},
                    {"mul_out"},
                    kOnnxDomain);

    // mul_out * 0.5 -> output
    builder.MakeScalarInitializer<float>("half", half);
    builder.AddNode("Mul_half",
                    "Mul",
                    {"mul_out", "half"},
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Helper function to build GELU Pattern 3 (ErfMul Pattern)
// Pattern 3:
//                   +-------------------------------------------+
//                   |                                           |
//                   |                                           v
//                [root] --> Div/Mul -----> Erf --> Mul --> Add --> Mul ==>
//                          (1.4142... or 1/1.4142...)  (0.5)   (0.5)
GetTestModelFn BuildGeluPattern3TestCase(const TestInputDef<float>& input_def, bool use_mul = false) {
  return [input_def, use_mul](ModelTestBuilder& builder) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float inv_sqrt_2 = 0.7071067690849304f;  // 1 / sqrt(2)
    constexpr float half = 0.5f;

    builder.graph_->set_name(use_mul ? "gelu_pattern3_mul_graph" : "gelu_pattern3_graph");

    // input
    MakeTestInput<float>(builder, "input", input_def);

    // root -> Div(sqrt2) or Mul(1/sqrt2) -> norm_out
    if (use_mul) {
      builder.MakeScalarInitializer<float>("inv_sqrt2", inv_sqrt_2);
      builder.AddNode("Mul_inv_sqrt2",
                      "Mul",
                      {"input", "inv_sqrt2"},
                      {"norm_out"},
                      kOnnxDomain);
    } else {
      builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
      builder.AddNode("Div_sqrt2",
                      "Div",
                      {"input", "sqrt2"},
                      {"norm_out"},
                      kOnnxDomain);
    }

    // norm_out -> Erf -> erf_out
    builder.AddNode("Erf",
                    "Erf",
                    {"norm_out"},
                    {"erf_out"},
                    kOnnxDomain);

    // erf_out * 0.5 -> mul_out
    builder.MakeScalarInitializer<float>("half", half);
    builder.AddNode("Mul_half",
                    "Mul",
                    {"erf_out", "half"},
                    {"mul_out"},
                    kOnnxDomain);

    // mul_out + 0.5 -> add_out
    builder.MakeScalarInitializer<float>("half2", half);
    builder.AddNode("Add_half",
                    "Add",
                    {"mul_out", "half2"},
                    {"add_out"},
                    kOnnxDomain);

    // root * add_out -> output
    builder.AddNode("Mul_out",
                    "Mul",
                    {"input", "add_out"},
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Helper function to build QDQ GELU Pattern 1
// QDQ is only at pattern boundaries (input and output), not around internal operators
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGeluPattern1TestCase(const TestInputDef<float>& input_def,
                                                          bool use_mul = false) {
  return [input_def, use_mul](ModelTestBuilder& builder,
                              std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float inv_sqrt_2 = 0.7071067690849304f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name(use_mul ? "qdq_gelu_pattern1_mul_graph" : "qdq_gelu_pattern1_graph");

    // input
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input", input_qparams.scale, input_qparams.zero_point);

    // Constants: float initializers (no QDQ to keep pattern operators as SingleNode)
    builder.MakeScalarInitializer<float>("one", one);
    builder.MakeScalarInitializer<float>("half", half);

    // GELU Pattern 1:
    // input -> Div/Mul(sqrt2 or 1/sqrt2) -> Erf -> Add(one) -> Mul(with (input * half))
    std::string norm_out = "div_out";
    if (use_mul) {
      builder.MakeScalarInitializer<float>("inv_sqrt2", inv_sqrt_2);
      builder.AddNode("Mul_inv_sqrt2",
                      "Mul",
                      {input_qdq, "inv_sqrt2"},
                      {norm_out},
                      kOnnxDomain);
    } else {
      builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
      builder.AddNode("Div_sqrt2",
                      "Div",
                      {input_qdq, "sqrt2"},
                      {norm_out},
                      kOnnxDomain);
    }

    // Erf operates on float (no QDQ around it)
    builder.AddNode("Erf",
                    "Erf",
                    {norm_out},
                    {"erf_out"},
                    kOnnxDomain);

    builder.AddNode("Add_one",
                    "Add",
                    {"erf_out", "one"},
                    {"add_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_half",
                    "Mul",
                    {input_qdq, "half"},
                    {"mul_half_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_out",
                    "Mul",
                    {"add_out", "mul_half_out"},
                    {"Y"},
                    kOnnxDomain);

    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "Y",
                                                     output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

// Helper function to build QDQ GELU Pattern 2
// QDQ is only at pattern boundaries (input and output), not around internal operators
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGeluPattern2TestCase(const TestInputDef<float>& input_def,
                                                          bool use_mul = false) {
  return [input_def, use_mul](ModelTestBuilder& builder,
                              std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float inv_sqrt_2 = 0.7071067690849304f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name(use_mul ? "qdq_gelu_pattern2_mul_graph" : "qdq_gelu_pattern2_graph");

    // input
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input", input_qparams.scale, input_qparams.zero_point);

    // Constants: float initializers (no QDQ to keep pattern operators as SingleNode)
    builder.MakeScalarInitializer<float>("one", one);
    builder.MakeScalarInitializer<float>("half", half);

    // GELU Pattern 2:
    // input -> Div/Mul(sqrt2 or 1/sqrt2) -> Erf -> Add(one) -> Mul(with input) -> Mul(half)
    std::string norm_out = "div_out";
    if (use_mul) {
      builder.MakeScalarInitializer<float>("inv_sqrt2", inv_sqrt_2);
      builder.AddNode("Mul_inv_sqrt2",
                      "Mul",
                      {input_qdq, "inv_sqrt2"},
                      {norm_out},
                      kOnnxDomain);
    } else {
      builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
      builder.AddNode("Div_sqrt2",
                      "Div",
                      {input_qdq, "sqrt2"},
                      {norm_out},
                      kOnnxDomain);
    }

    // Erf operates on float (no QDQ around it)
    builder.AddNode("Erf",
                    "Erf",
                    {norm_out},
                    {"erf_out"},
                    kOnnxDomain);

    builder.AddNode("Add_one",
                    "Add",
                    {"erf_out", "one"},
                    {"add_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_input",
                    "Mul",
                    {input_qdq, "add_out"},
                    {"mul_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_half",
                    "Mul",
                    {"mul_out", "half"},
                    {"Y"},
                    kOnnxDomain);

    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "Y",
                                                     output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

// Helper function to build QDQ GELU Pattern 3 (ErfMul Pattern)
// QDQ is only at pattern boundaries (input and output), not around internal operators
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGeluPattern3TestCase(const TestInputDef<float>& input_def,
                                                          bool use_mul = false) {
  return [input_def, use_mul](ModelTestBuilder& builder,
                              std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float inv_sqrt_2 = 0.7071067690849304f;
    constexpr float half = 0.5f;

    builder.graph_->set_name(use_mul ? "qdq_gelu_pattern3_mul_graph" : "qdq_gelu_pattern3_graph");

    // input
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input", input_qparams.scale, input_qparams.zero_point);

    // Constants: float initializers (no QDQ to keep pattern operators as SingleNode)
    builder.MakeScalarInitializer<float>("half", half);
    builder.MakeScalarInitializer<float>("half2", half);

    // input -> Div/Mul(sqrt2 or 1/sqrt2) -> Erf -> Mul(0.5) -> Add(0.5) -> Mul(input)
    std::string norm_out = "div_out";
    if (use_mul) {
      builder.MakeScalarInitializer<float>("inv_sqrt_2", inv_sqrt_2);
      builder.AddNode("Mul_inv_sqrt2",
                      "Mul",
                      {input_qdq, "inv_sqrt_2"},
                      {norm_out},
                      kOnnxDomain);
    } else {
      builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
      builder.AddNode("Div_sqrt2",
                      "Div",
                      {input_qdq, "sqrt2"},
                      {norm_out},
                      kOnnxDomain);
    }

    // Erf operates on float (no QDQ around it)
    builder.AddNode("Erf",
                    "Erf",
                    {norm_out},
                    {"erf_out"},
                    kOnnxDomain);

    // ErfMul Pattern: Mul(erf_out, 0.5) -> Add(0.5) -> Mul(input)
    builder.AddNode("Mul_half",
                    "Mul",
                    {"erf_out", "half"},
                    {"mul_out"},
                    kOnnxDomain);

    builder.AddNode("Add_half",
                    "Add",
                    {"mul_out", "half2"},
                    {"add_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_out",
                    "Mul",
                    {input_qdq, "add_out"},
                    {"Y"},
                    kOnnxDomain);

    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "Y",
                                                     output_qparams[0].scale, output_qparams[0].zero_point);
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

// Test GELU Pattern 1 with float32 model (for baseline comparison)
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_Float32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern1_Float32";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Test with Div
  RunQnnModelTest(BuildGeluPattern1TestCase(input_def, false),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  // Test with Mul (Div replaced by Mul)
  ResetQnnGraphDir(json_qnn_graph_dir);
  RunQnnModelTest(BuildGeluPattern1TestCase(input_def, true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 2 with float32 model (for baseline comparison)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_Float32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern2_Float32";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Test with Div
  RunQnnModelTest(BuildGeluPattern2TestCase(input_def, false),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  // Test with Mul (Div replaced by Mul)
  ResetQnnGraphDir(json_qnn_graph_dir);
  RunQnnModelTest(BuildGeluPattern2TestCase(input_def, true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 3 (ErfMul Pattern) with float32 model
TEST_F(QnnHTPBackendTests, GeluFusionPattern3_Float32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern3_Float32";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Test with Div
  RunQnnModelTest(BuildGeluPattern3TestCase(input_def, false),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  // Test with Mul (Div replaced by Mul)
  ResetQnnGraphDir(json_qnn_graph_dir);
  RunQnnModelTest(BuildGeluPattern3TestCase(input_def, true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 1 with larger input shape
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_LargeInput) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern1_LargeInput";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 128, 768}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 2 with larger input shape
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_LargeInput) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern2_LargeInput";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 128, 768}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 1 with 3D input
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_3D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 16, 32}, false, -1.0f, 1.0f);

  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern1_3D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Test with Div
  RunQnnModelTest(BuildGeluPattern1TestCase(input_def, false),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  // Test with Mul (Div replaced by Mul)
  ResetQnnGraphDir(json_qnn_graph_dir);
  RunQnnModelTest(BuildGeluPattern1TestCase(input_def, true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 2 with 3D input
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_3D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({1, 16, 32}, false, -1.0f, 1.0f);

  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern2_3D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Test with Div
  RunQnnModelTest(BuildGeluPattern2TestCase(input_def, false),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  // Test with Mul (Div replaced by Mul)
  ResetQnnGraphDir(json_qnn_graph_dir);
  RunQnnModelTest(BuildGeluPattern2TestCase(input_def, true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 1 with 2D input (typical for linear layers)
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_2D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({32, 512}, false, -1.5f, 1.5f);

  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern1_2D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Test with Div
  RunQnnModelTest(BuildGeluPattern1TestCase(input_def, false),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  // Test with Mul (Div replaced by Mul)
  ResetQnnGraphDir(json_qnn_graph_dir);
  RunQnnModelTest(BuildGeluPattern1TestCase(input_def, true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 2 with 2D input (typical for linear layers)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_2D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto input_def = TestInputDef<float>({32, 512}, false, -1.5f, 1.5f);

  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern2_2D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Test with Div
  RunQnnModelTest(BuildGeluPattern2TestCase(input_def, false),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  // Test with Mul (Div replaced by Mul)
  ResetQnnGraphDir(json_qnn_graph_dir);
  RunQnnModelTest(BuildGeluPattern2TestCase(input_def, true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(6e-3f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 1 with QDQ
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_QDQ_U8) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern1_QDQ_U8";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -10.0f, 10.0f);

  TestQDQModelAccuracy(BuildGeluPattern1TestCase(input_def, false),
                       BuildQDQGeluPattern1TestCase<uint8_t>(input_def, false),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  ResetQnnGraphDir(json_qnn_graph_dir);

  TestQDQModelAccuracy(BuildGeluPattern1TestCase(input_def, true),
                       BuildQDQGeluPattern1TestCase<uint8_t>(input_def, true),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 2 with QDQ
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_QDQ_U8) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern2_QDQ_U8";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -10.0f, 10.0f);

  TestQDQModelAccuracy(BuildGeluPattern2TestCase(input_def, false),
                       BuildQDQGeluPattern2TestCase<uint8_t>(input_def, false),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  ResetQnnGraphDir(json_qnn_graph_dir);

  TestQDQModelAccuracy(BuildGeluPattern2TestCase(input_def, true),
                       BuildQDQGeluPattern2TestCase<uint8_t>(input_def, true),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Test GELU Pattern 3 with QDQ
TEST_F(QnnHTPBackendTests, GeluFusionPattern3_QDQ_U8) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern3_QDQ_U8";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -10.0f, 10.0f);

  TestQDQModelAccuracy(BuildGeluPattern3TestCase(input_def, false),
                       BuildQDQGeluPattern3TestCase<uint8_t>(input_def, false),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");

  ResetQnnGraphDir(json_qnn_graph_dir);

  TestQDQModelAccuracy(BuildGeluPattern3TestCase(input_def, true),
                       BuildQDQGeluPattern3TestCase<uint8_t>(input_def, true),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron");
}

// Negative test: GELU Pattern with internal QDQ nodes should NOT fuse
// Helper function to build QDQ GELU Pattern with internal QDQ (around Erf)
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGeluPatternWithInternalQDQ(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder,
                     std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name("qdq_gelu_internal_qdq_graph");

    // input
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input", input_qparams.scale, input_qparams.zero_point);

    // Constants with QDQ
    builder.MakeScalarInitializer<float>("one", one);
    builder.MakeScalarInitializer<float>("half", half);
    builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
    const std::string one_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_one", "one", input_qparams.scale, input_qparams.zero_point);
    const std::string half_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_half", "half", input_qparams.scale, input_qparams.zero_point);
    const std::string sqrt2_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_sqrt2", "sqrt2", input_qparams.scale, input_qparams.zero_point);

    // Div node
    builder.AddNode("Div_sqrt2",
                    "Div",
                    {input_qdq, sqrt2_qdq},
                    {"div_out"},
                    kOnnxDomain);

    // INTERNAL QDQ around Erf (this makes Erf a QDQGroup and should prevent fusion)
    const std::string erf_in_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_erf_in", "div_out", input_qparams.scale, input_qparams.zero_point);

    builder.AddNode("Erf",
                    "Erf",
                    {erf_in_qdq},
                    {"erf_out"},
                    kOnnxDomain);

    const std::string erf_out_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_erf_out", "erf_out", input_qparams.scale, input_qparams.zero_point);

    builder.AddNode("Add_one",
                    "Add",
                    {erf_out_qdq, one_qdq},
                    {"add_out"},
                    kOnnxDomain);

    const std::string add_out_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_add_out", "add_out", input_qparams.scale, input_qparams.zero_point);

    builder.AddNode("Mul_half",
                    "Mul",
                    {input_qdq, half_qdq},
                    {"mul_half_out"},
                    kOnnxDomain);

    const std::string mul_half_out_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_mul_half_out", "mul_half_out",
                                  input_qparams.scale, input_qparams.zero_point);

    builder.AddNode("Mul_out",
                    "Mul",
                    {add_out_qdq, mul_half_out_qdq},
                    {"Y"},
                    kOnnxDomain);

    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "Y",
                                                     output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

TEST_F(QnnHTPBackendTests, GeluFusionPattern_QDQ_InternalQDQ_ShouldNotFuse) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern_QDQ_InternalQDQ";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -10.0f, 10.0f);

  // This pattern has internal QDQ nodes around Erf, so fusion should be rejected.
  // Expected: Individual nodes are assigned to QNN EP, but NOT fused as a GELU.
  TestQDQModelAccuracy(BuildGeluPattern1TestCase(input_def, false),
                       BuildQDQGeluPatternWithInternalQDQ<uint8_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::Some);

  // Verify that NO fused GELU op was created (count = 0)
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron", 0);
}

// Negative test: GELU Pattern with incorrect skip connection topology should NOT fuse
// Helper function to build QDQ GELU Pattern with wrong skip connection
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGeluPatternWithWrongSkipConnection(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder,
                     std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name("qdq_gelu_wrong_skip_graph");

    // input
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input", input_qparams.scale, input_qparams.zero_point);

    // Constants: float initializers
    builder.MakeScalarInitializer<float>("one", one);
    builder.MakeScalarInitializer<float>("half", half);
    builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);

    // GELU-like Pattern 1 BUT with wrong skip connection:
    // Instead of using input for skip, we use a different tensor (div_out)
    builder.AddNode("Div_sqrt2",
                    "Div",
                    {input_qdq, "sqrt2"},
                    {"div_out"},
                    kOnnxDomain);

    builder.AddNode("Erf",
                    "Erf",
                    {"div_out"},
                    {"erf_out"},
                    kOnnxDomain);

    builder.AddNode("Add_one",
                    "Add",
                    {"erf_out", "one"},
                    {"add_out"},
                    kOnnxDomain);

    // WRONG: Using div_out instead of input for the skip connection
    builder.AddNode("Mul_wrong_skip",
                    "Mul",
                    {"div_out", "half"},  // Should be input_qdq, not div_out
                    {"mul_half_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_out",
                    "Mul",
                    {"add_out", "mul_half_out"},
                    {"Y"},
                    kOnnxDomain);

    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "Y",
                                                     output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

TEST_F(QnnHTPBackendTests, GeluFusionPattern_QDQ_WrongSkipConnection_ShouldNotFuse) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern_QDQ_WrongSkip";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -10.0f, 10.0f);

  // This pattern has correct QDQ placement but wrong skip connection topology.
  // The skip connection uses div_out instead of the original input.
  // Expected: Individual nodes are assigned to QNN EP, but NOT fused as a GELU.
  TestQDQModelAccuracy(BuildGeluPattern1TestCase(input_def, false),
                       BuildQDQGeluPatternWithWrongSkipConnection<uint8_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::Some);

  // Verify that NO fused GELU op was created (count = 0)
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
