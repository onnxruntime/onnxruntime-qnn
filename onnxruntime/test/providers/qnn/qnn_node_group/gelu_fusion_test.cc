// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Helper function to build GELU Pattern 1: root -> Mul -> Div -> Erf -> Add -> Mul
// Pattern 1:
//                   +-------Mul(0.5)---------------------+
//                   |                                    |
//                   |                                    v
//                [root] --> Div -----> Erf  --> Add --> Mul ==>
//                          (B=1.4142...)        (1)
GetTestModelFn BuildGeluPattern1TestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name("gelu_pattern1_graph");

    // input
    MakeTestInput<float>(builder, "input", input_def);

    // input -> Mul(0.5) -> mul_half_out
    builder.MakeScalarInitializer<float>("half", half);
    builder.AddNode("Mul_half",
                    "Mul",
                    {"input", "half"},
                    {"mul_half_out"},
                    kOnnxDomain);

    // input -> Div(sqrt2) -> div_out
    builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
    builder.AddNode("Div_sqrt2",
                    "Div",
                    {"input", "sqrt2"},
                    {"div_out"},
                    kOnnxDomain);

    // div_out -> Erf -> erf_out
    builder.AddNode("Erf",
                    "Erf",
                    {"div_out"},
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
//                [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
//                          (B=1.4142...)        (1)            (0.5)
GetTestModelFn BuildGeluPattern2TestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name("gelu_pattern2_graph");

    // input
    MakeTestInput<float>(builder, "input", input_def);

    // input -> Div(sqrt2) -> div_out
    builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
    builder.AddNode("Div_sqrt2",
                    "Div",
                    {"input", "sqrt2"},
                    {"div_out"},
                    kOnnxDomain);

    // div_out -> Erf -> erf_out
    builder.AddNode("Erf",
                    "Erf",
                    {"div_out"},
                    {"erf_out"},
                    kOnnxDomain);

    // erf_out -> Add(1.0) -> add_out
    builder.MakeScalarInitializer<float>("one", one);
    builder.AddNode("Add_one",
                    "Add",
                    {"erf_out", "one"},
                    {"add_out"},
                    kOnnxDomain);

    // input * add_out -> mul_out
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

// Helper function to build QDQ GELU Pattern 1
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGeluPattern1TestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name("qdq_gelu_pattern1_graph");

    // input
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input", input_qparams.scale, input_qparams.zero_point);

    // Constants: add explicit QDQ after each initializer (to match expected QDQ patterns).
    builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
    builder.MakeScalarInitializer<float>("one", one);
    builder.MakeScalarInitializer<float>("half", half);

    const std::string sqrt2_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_sqrt2", "sqrt2", input_qparams.scale, input_qparams.zero_point);
    const std::string one_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_one", "one", input_qparams.scale, input_qparams.zero_point);
    const std::string half_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_half", "half", input_qparams.scale, input_qparams.zero_point);

    // GELU Pattern 1:
    // input -> Div(sqrt2) -> Erf -> Add(one) -> Mul(with (input * half))
    builder.AddNode("Div_sqrt2",
                    "Div",
                    {input_qdq, sqrt2_qdq},
                    {"div_out"},
                    kOnnxDomain);

    // Add explicit QDQ around Erf to match expected QDQ patterns:
    // div_out -> Q -> DQ -> erf_in
    const std::string erf_in_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_erf_in", "div_out", input_qparams.scale, input_qparams.zero_point);

    // erf_in -> Erf -> erf_out
    builder.AddNode("Erf",
                    "Erf",
                    {erf_in_qdq},
                    {"erf_out"},
                    kOnnxDomain);

    // erf_out -> Q -> DQ -> erf_out_qdq
    const std::string erf_out_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_erf_out", "erf_out", input_qparams.scale, input_qparams.zero_point);

    builder.AddNode("Add_one",
                    "Add",
                    {erf_out_qdq, one_qdq},
                    {"add_out"},
                    kOnnxDomain);

    // add_out -> Q -> DQ -> add_out_qdq
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

// Helper function to build QDQ GELU Pattern 2
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGeluPattern2TestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    builder.graph_->set_name("qdq_gelu_pattern2_graph");

    // input
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input", input_qparams.scale, input_qparams.zero_point);

    // Constants: add explicit QDQ after each initializer (to match expected QDQ patterns).
    builder.MakeScalarInitializer<float>("sqrt2", sqrt_2);
    builder.MakeScalarInitializer<float>("one", one);
    builder.MakeScalarInitializer<float>("half", half);

    const std::string sqrt2_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_sqrt2", "sqrt2", input_qparams.scale, input_qparams.zero_point);
    const std::string one_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_one", "one", input_qparams.scale, input_qparams.zero_point);
    const std::string half_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_half", "half", input_qparams.scale, input_qparams.zero_point);

    // GELU Pattern 2:
    // input -> Div(sqrt2) -> Erf -> Add(one) -> Mul(with input) -> Mul(half)
    builder.AddNode("Div_sqrt2",
                    "Div",
                    {input_qdq, sqrt2_qdq},
                    {"div_out"},
                    kOnnxDomain);

    // Add explicit QDQ around Erf to match expected QDQ patterns:
    // div_out -> Q -> DQ -> erf_in
    const std::string erf_in_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_erf_in", "div_out", input_qparams.scale, input_qparams.zero_point);

    // erf_in -> Erf -> erf_out
    builder.AddNode("Erf",
                    "Erf",
                    {erf_in_qdq},
                    {"erf_out"},
                    kOnnxDomain);

    // erf_out -> Q -> DQ -> erf_out_qdq
    const std::string erf_out_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_erf_out", "erf_out", input_qparams.scale, input_qparams.zero_point);

    builder.AddNode("Add_one",
                    "Add",
                    {erf_out_qdq, one_qdq},
                    {"add_out"},
                    kOnnxDomain);

    // add_out -> Q -> DQ -> add_out_qdq
    const std::string add_out_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_add_out", "add_out", input_qparams.scale, input_qparams.zero_point);

    builder.AddNode("Mul_input",
                    "Mul",
                    {input_qdq, add_out_qdq},
                    {"mul_out"},
                    kOnnxDomain);

    // mul_out -> Q -> DQ -> mul_out_qdq
    const std::string mul_out_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_mul_out", "mul_out", input_qparams.scale, input_qparams.zero_point);

    builder.AddNode("Mul_half",
                    "Mul",
                    {mul_out_qdq, half_qdq},
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
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern1_Float32";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Test GELU Pattern 2 with float32 model (for baseline comparison)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_Float32) {
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern2_Float32";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Test GELU Pattern 1 with larger input shape
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_LargeInput) {
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
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Test GELU Pattern 2 with larger input shape
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_LargeInput) {
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
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Test GELU Pattern 1 with 3D input
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_3D) {
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern1_3D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 16, 32}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Test GELU Pattern 2 with 3D input
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_3D) {
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern2_3D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 16, 32}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Test GELU Pattern 1 with 2D input (typical for linear layers)
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_2D) {
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern1_2D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({32, 512}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Test GELU Pattern 2 with 2D input (typical for linear layers)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_2D) {
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern2_2D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({32, 512}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Test GELU Pattern 1 with QDQ
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_QDQ_U8) {
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern1_QDQ_U8";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  TestQDQModelAccuracy(BuildGeluPattern1TestCase(input_def),
                       BuildQDQGeluPattern1TestCase<uint8_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

// Test GELU Pattern 2 with QDQ
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_QDQ_U8) {
  const std::filesystem::path json_qnn_graph_dir = "GeluFusionPattern2_QDQ_U8";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  TestQDQModelAccuracy(BuildGeluPattern2TestCase(input_def),
                       BuildQDQGeluPattern2TestCase<uint8_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gelu");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
