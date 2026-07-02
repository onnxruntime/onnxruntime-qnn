// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>

#include <gsl/util>
#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Build a 2-node fusion test case: Reshape -> Gemm
// Input: [batch, seq, hidden] -> Reshape to [batch*seq, hidden] -> Gemm -> [batch*seq, out]
GetTestModelFn BuildReshapeGemmTestCase(const std::vector<int64_t>& input_shape,
                                        int64_t hidden_size,
                                        int64_t output_size) {
  return [input_shape, hidden_size, output_size](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_graph");

    // Input tensor
    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Calculate flattened batch size (all dims except last)
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      batch_size *= input_shape[i];
    }

    // Reshape: [batch, seq, hidden] -> [batch*seq, hidden]
    builder.Make1DInitializer<int64_t>("reshape_shape", {batch_size, hidden_size});
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    // Gemm weight: [hidden, output] (transB=0)
    std::vector<int64_t> weight_shape = {hidden_size, output_size};
    builder.MakeInitializer<float>("weight", weight_shape, -0.5f, 0.5f);

    // Gemm bias
    builder.MakeInitializer<float>("bias", {output_size}, -0.1f, 0.1f);

    // Gemm: [batch*seq, hidden] x [hidden, output] -> [batch*seq, output]
    builder.AddNode("gemm", "Gemm", {"reshape_out", "weight", "bias"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Build a 3-node fusion test case: Reshape -> Gemm -> Reshape
// Input: [batch, seq, hidden] -> Reshape -> Gemm -> Reshape -> [batch, seq, out]
GetTestModelFn BuildReshapeGemmReshapeTestCase(const std::vector<int64_t>& input_shape,
                                               int64_t hidden_size,
                                               int64_t output_size) {
  return [input_shape, hidden_size, output_size](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_reshape_graph");

    // Input tensor
    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Calculate flattened batch size (all dims except last)
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      batch_size *= input_shape[i];
    }

    // Reshape: [batch, seq, hidden] -> [batch*seq, hidden]
    builder.Make1DInitializer<int64_t>("reshape1_shape", {batch_size, hidden_size});
    builder.AddNode("reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    // Gemm weight: [hidden, output]
    std::vector<int64_t> weight_shape = {hidden_size, output_size};
    builder.MakeInitializer<float>("weight", weight_shape, -0.5f, 0.5f);

    // Gemm bias
    builder.MakeInitializer<float>("bias", {output_size}, -0.1f, 0.1f);

    // Gemm: [batch*seq, hidden] x [hidden, output] -> [batch*seq, output]
    builder.AddNode("gemm", "Gemm", {"reshape1_out", "weight", "bias"}, {"gemm_out"}, kOnnxDomain);

    // Build output shape: same as input but with last dim = output_size
    std::vector<int64_t> output_shape_vec = input_shape;
    output_shape_vec.back() = output_size;

    // Reshape: [batch*seq, output] -> [batch, seq, output]
    builder.Make1DInitializer<int64_t>("reshape2_shape", output_shape_vec);
    builder.AddNode("reshape2", "Reshape", {"gemm_out", "reshape2_shape"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Build a 4-node fusion test case: Reshape -> Gemm -> Reshape -> Reshape
// Input: [batch, seq, hidden] -> Reshape -> Gemm -> Reshape -> Reshape -> [1, batch*seq, out]
GetTestModelFn BuildReshapeGemmReshapeReshapeTestCase(const std::vector<int64_t>& input_shape,
                                                      int64_t hidden_size,
                                                      int64_t output_size) {
  return [input_shape, hidden_size, output_size](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_reshape_reshape_graph");

    // Input tensor
    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Calculate flattened batch size (all dims except last)
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      batch_size *= input_shape[i];
    }

    // Reshape: [batch, seq, hidden] -> [batch*seq, hidden]
    builder.Make1DInitializer<int64_t>("reshape1_shape", {batch_size, hidden_size});
    builder.AddNode("reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    // Gemm weight: [hidden, output]
    std::vector<int64_t> weight_shape = {hidden_size, output_size};
    builder.MakeInitializer<float>("weight", weight_shape, -0.5f, 0.5f);

    // Gemm bias
    builder.MakeInitializer<float>("bias", {output_size}, -0.1f, 0.1f);

    // Gemm: [batch*seq, hidden] x [hidden, output] -> [batch*seq, output]
    builder.AddNode("gemm", "Gemm", {"reshape1_out", "weight", "bias"}, {"gemm_out"}, kOnnxDomain);

    // Build intermediate shape: same as input but with last dim = output_size
    std::vector<int64_t> intermediate_shape_vec = input_shape;
    intermediate_shape_vec.back() = output_size;

    // Reshape2: [batch*seq, output] -> [batch, seq, output]
    builder.Make1DInitializer<int64_t>("reshape2_shape", intermediate_shape_vec);
    builder.AddNode("reshape2", "Reshape", {"gemm_out", "reshape2_shape"}, {"reshape2_out"}, kOnnxDomain);

    // Reshape3: [batch, seq, output] -> [1, batch*seq, output]
    builder.Make1DInitializer<int64_t>("reshape3_shape", {1, batch_size, output_size});
    builder.AddNode("reshape3", "Reshape", {"reshape2_out", "reshape3_shape"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Build a test case with pattern: keep first dim, flatten last dims
// Input: [16, 1, 4, 8] -> Reshape to [16, 32] -> Gemm -> [16, 32] -> Reshape -> [16, 1, 32]
// (Smaller version of ViT attention output pattern)
GetTestModelFn BuildReshapeGemmReshapeKeepFirstDimTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_reshape_keep_first_graph");

    // Input: [16, 1, 4, 8] - smaller ViT-like attention output shape
    auto input_def = TestInputDef<float>({16, 1, 4, 8}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Reshape: [16, 1, 4, 8] -> [16, 32] (keep first dim, flatten last dims)
    builder.Make1DInitializer<int64_t>("reshape1_shape", {16, 32});
    builder.AddNode("reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    // Gemm weight: [32, 32]
    builder.MakeInitializer<float>("weight", {32, 32}, -0.5f, 0.5f);

    // Gemm bias
    builder.MakeInitializer<float>("bias", {32}, -0.1f, 0.1f);

    // Gemm: [16, 32] x [32, 32] -> [16, 32]
    builder.AddNode("gemm", "Gemm", {"reshape1_out", "weight", "bias"}, {"gemm_out"}, kOnnxDomain);

    // Reshape: [16, 32] -> [16, 1, 32]
    builder.Make1DInitializer<int64_t>("reshape2_shape", {16, 1, 32});
    builder.AddNode("reshape2", "Reshape", {"gemm_out", "reshape2_shape"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return provider_options;
}

// Build a test case with rank-5 input: Reshape -> Gemm -> Reshape
// Mirrors the proj/MatMul pattern: [3,3,14,14,384] -> [1764,384] -> Gemm -> output
// ReshapeGemmFusion must NOT fire because QNN HTP FC rejects rank-5 input.
GetTestModelFn BuildReshapeGemmReshapeRank5InputTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_reshape_rank5_graph");

    // Rank-5 input mimicking ViT-Matte attention output: [3, 3, 14, 14, 384]
    auto input_def = TestInputDef<float>({3, 3, 14, 14, 384}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Reshape: [3,3,14,14,384] -> [1764,384] (flatten first 4 dims)
    builder.Make1DInitializer<int64_t>("reshape1_shape", {1764, 384});
    builder.AddNode("reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    // Gemm weight: [384, 384], bias: [384]
    builder.MakeInitializer<float>("weight", {384, 384}, -0.5f, 0.5f);
    builder.MakeInitializer<float>("bias", {384}, -0.1f, 0.1f);

    // Gemm: [1764, 384] x [384, 384] -> [1764, 384]
    builder.AddNode("gemm", "Gemm", {"reshape1_out", "weight", "bias"}, {"gemm_out"}, kOnnxDomain);

    // Reshape output back to [3,3,14,14,384]
    builder.Make1DInitializer<int64_t>("reshape2_shape", {3, 3, 14, 14, 384});
    builder.AddNode("reshape2", "Reshape", {"gemm_out", "reshape2_shape"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

}  // namespace

// Test 2-node fusion: Reshape -> Gemm (3D input)
TEST_F(QnnHTPBackendTests, ReshapeGemmFusion_3D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmFusion_3D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeGemmTestCase({1, 32, 64}, 64, 128),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify FullyConnected is in the graph (fusion happened)
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
}

// Test 3-node fusion: Reshape -> Gemm -> Reshape (3D input)
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_3D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmReshapeFusion_3D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeGemmReshapeTestCase({1, 32, 64}, 64, 128),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify FullyConnected and one Reshape (output reshape kept)
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

// Test 3-node fusion: Reshape -> Gemm -> Reshape (4D input)
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_4D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmReshapeFusion_4D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeGemmReshapeTestCase({2, 4, 8, 32}, 32, 64),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify FullyConnected and one Reshape
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

// Test 4-node fusion: Reshape -> Gemm -> Reshape -> Reshape
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeReshapeFusion_3D) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmReshapeReshapeFusion_3D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeGemmReshapeReshapeTestCase({1, 32, 64}, 64, 128),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify FullyConnected and one Reshape (only final reshape kept)
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

// Test 3-node fusion with ViT-like pattern: keep first dim, flatten last dims
// [197, 1, 12, 64] -> [197, 768] -> Gemm -> [197, 768] -> [197, 1, 768]
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_ViTPattern) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmReshapeFusion_ViTPattern";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeGemmReshapeKeepFirstDimTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify FullyConnected and one Reshape
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

// Test with transformer-like shape (smaller for unit test)
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_Transformer) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmReshapeFusion_Transformer";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Smaller transformer-like shape: [batch, seq, hidden] with hidden=64
  RunQnnModelTest(BuildReshapeGemmReshapeTestCase({1, 16, 64}, 64, 64),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify FullyConnected and one Reshape
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

// ============================================================================
// Negative Tests - Fusion should NOT happen
// ============================================================================

// Build a test case where input Reshape has two consumers (fusion should not happen).
// Graph: Input -> Reshape -> Gemm1 (output1)
//                        \-> Gemm2 (output2)
// Claiming the shared Reshape for Gemm1 would break Gemm2, so fusion must not fire.
GetTestModelFn BuildReshapeSharedByTwoGemmsTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_shared_two_gemms_graph");

    auto input_def = TestInputDef<float>({1, 4, 8}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Shared Reshape: [1, 4, 8] -> [4, 8]
    builder.Make1DInitializer<int64_t>("reshape_shape", {4, 8});
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    // Gemm1: [4, 8] x [8, 16] -> [4, 16]
    builder.MakeInitializer<float>("weight1", {8, 16}, -0.5f, 0.5f);
    builder.MakeInitializer<float>("bias1", {16}, -0.1f, 0.1f);
    builder.AddNode("gemm1", "Gemm", {"reshape_out", "weight1", "bias1"}, {"output1"}, kOnnxDomain);

    // Gemm2: [4, 8] x [8, 32] -> [4, 32]  — shares reshape_out
    builder.MakeInitializer<float>("weight2", {8, 32}, -0.5f, 0.5f);
    builder.MakeInitializer<float>("bias2", {32}, -0.1f, 0.1f);
    builder.AddNode("gemm2", "Gemm", {"reshape_out", "weight2", "bias2"}, {"output2"}, kOnnxDomain);

    builder.MakeOutput("output1");
    builder.MakeOutput("output2");
  };
}

// Test: Fusion should NOT happen when the input Reshape has two consumers.
// If fusion incorrectly claimed the shared Reshape for one Gemm, the other Gemm
// would lose its input and the model would fail. All nodes must still run on QNN EP.
TEST_F(QnnHTPBackendTests, ReshapeGemmFusion_Negative_SharedReshape) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmFusion_Negative_SharedReshape";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeSharedByTwoGemmsTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify fusion did NOT fire: the shared Reshape must remain standalone (not consumed
  // by any fusion), and both Gemms must run as independent FullyConnected nodes.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 2);
}

// Build a 3-node shared-Reshape test case for TryFusion3 (Reshape -> Gemm -> Reshape).
// Two Reshape->Gemm->Reshape chains share the same input Reshape; fusion must not fire.
// Graph: Input -> Reshape -> Gemm1 -> Reshape1 (output1)
//                        \-> Gemm2 -> Reshape2 (output2)
GetTestModelFn BuildReshapeGemmReshapeSharedByTwoTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_reshape_shared_graph");

    auto input_def = TestInputDef<float>({1, 4, 8}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Shared Reshape: [1, 4, 8] -> [4, 8]
    builder.Make1DInitializer<int64_t>("reshape_shape", {4, 8});
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    // Gemm1: [4, 8] x [8, 16] -> [4, 16]
    builder.MakeInitializer<float>("weight1", {8, 16}, -0.5f, 0.5f);
    builder.MakeInitializer<float>("bias1", {16}, -0.1f, 0.1f);
    builder.AddNode("gemm1", "Gemm", {"reshape_out", "weight1", "bias1"}, {"gemm1_out"}, kOnnxDomain);

    // Reshape1: [4, 16] -> [1, 4, 16]
    builder.Make1DInitializer<int64_t>("reshape1_shape", {1, 4, 16});
    builder.AddNode("reshape1", "Reshape", {"gemm1_out", "reshape1_shape"}, {"output1"}, kOnnxDomain);

    // Gemm2: [4, 8] x [8, 32] -> [4, 32]  — shares reshape_out
    builder.MakeInitializer<float>("weight2", {8, 32}, -0.5f, 0.5f);
    builder.MakeInitializer<float>("bias2", {32}, -0.1f, 0.1f);
    builder.AddNode("gemm2", "Gemm", {"reshape_out", "weight2", "bias2"}, {"gemm2_out"}, kOnnxDomain);

    // Reshape2: [4, 32] -> [1, 4, 32]
    builder.Make1DInitializer<int64_t>("reshape2_shape", {1, 4, 32});
    builder.AddNode("reshape2", "Reshape", {"gemm2_out", "reshape2_shape"}, {"output2"}, kOnnxDomain);

    builder.MakeOutput("output1");
    builder.MakeOutput("output2");
  };
}

// Test: TryFusion3 (Reshape->Gemm->Reshape) must not fire when the input Reshape is shared.
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_Negative_SharedReshape) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmReshapeFusion_Negative_SharedReshape";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeGemmReshapeSharedByTwoTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify fusion did NOT fire: the shared input Reshape is standalone, both Gemms run
  // as independent FullyConnected nodes, and each output Reshape is also standalone.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 3);
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 2);
}

// Build a 4-node shared-Reshape test case for TryFusion4 (Reshape -> Gemm -> Reshape -> Reshape).
// Two chains share the same input Reshape; fusion must not fire.
// Graph: Input -> Reshape -> Gemm1 -> Reshape1a -> Reshape1b (output1)
//                        \-> Gemm2 -> Reshape2a -> Reshape2b (output2)
GetTestModelFn BuildReshapeGemmReshapeReshapeSharedByTwoTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_reshape_reshape_shared_graph");

    auto input_def = TestInputDef<float>({1, 4, 8}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Shared Reshape: [1, 4, 8] -> [4, 8]
    builder.Make1DInitializer<int64_t>("reshape_shape", {4, 8});
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    // Gemm1: [4, 8] x [8, 16] -> [4, 16]
    builder.MakeInitializer<float>("weight1", {8, 16}, -0.5f, 0.5f);
    builder.MakeInitializer<float>("bias1", {16}, -0.1f, 0.1f);
    builder.AddNode("gemm1", "Gemm", {"reshape_out", "weight1", "bias1"}, {"gemm1_out"}, kOnnxDomain);

    // Reshape1a: [4, 16] -> [2, 2, 16]  (intermediate, not a graph output)
    builder.Make1DInitializer<int64_t>("reshape1a_shape", {2, 2, 16});
    builder.AddNode("reshape1a", "Reshape", {"gemm1_out", "reshape1a_shape"}, {"reshape1a_out"}, kOnnxDomain);

    // Reshape1b: [2, 2, 16] -> [1, 4, 16]  (final output)
    builder.Make1DInitializer<int64_t>("reshape1b_shape", {1, 4, 16});
    builder.AddNode("reshape1b", "Reshape", {"reshape1a_out", "reshape1b_shape"}, {"output1"}, kOnnxDomain);

    // Gemm2: [4, 8] x [8, 32] -> [4, 32]  — shares reshape_out
    builder.MakeInitializer<float>("weight2", {8, 32}, -0.5f, 0.5f);
    builder.MakeInitializer<float>("bias2", {32}, -0.1f, 0.1f);
    builder.AddNode("gemm2", "Gemm", {"reshape_out", "weight2", "bias2"}, {"gemm2_out"}, kOnnxDomain);

    // Reshape2a: [4, 32] -> [2, 2, 32]  (intermediate, not a graph output)
    builder.Make1DInitializer<int64_t>("reshape2a_shape", {2, 2, 32});
    builder.AddNode("reshape2a", "Reshape", {"gemm2_out", "reshape2a_shape"}, {"reshape2a_out"}, kOnnxDomain);

    // Reshape2b: [2, 2, 32] -> [1, 4, 32]  (final output)
    builder.Make1DInitializer<int64_t>("reshape2b_shape", {1, 4, 32});
    builder.AddNode("reshape2b", "Reshape", {"reshape2a_out", "reshape2b_shape"}, {"output2"}, kOnnxDomain);

    builder.MakeOutput("output1");
    builder.MakeOutput("output2");
  };
}

// Test: TryFusion4 (Reshape->Gemm->Reshape->Reshape) must not fire when the input Reshape is shared.
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeReshapeFusion_Negative_SharedReshape) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmReshapeReshapeFusion_Negative_SharedReshape";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeGemmReshapeReshapeSharedByTwoTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify fusion did NOT fire: the shared input Reshape is standalone, both Gemms run
  // as independent FullyConnected nodes, and each chain's output Reshapes are also standalone.
  // Note: ORT's ReshapeFusion::FuseContiguousReshapes graph transformer merges each chain's
  // two consecutive output Reshapes (Reshape1a+1b) into one before the QNN EP sees the graph,
  // so expected count is 3: 1 (shared input) + 1 (merged chain1) + 1 (merged chain2).
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 3);
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 2);
}

// Build a test case where Gemm has transA=1 (fusion should not happen)
GetTestModelFn BuildReshapeGemmWithTransATestCase(const std::vector<int64_t>& input_shape,
                                                  int64_t hidden_size,
                                                  int64_t output_size) {
  return [input_shape, hidden_size, output_size](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_transA_graph");

    // Input tensor - transposed shape for transA=1
    std::vector<int64_t> transposed_input_shape = input_shape;
    std::swap(transposed_input_shape[transposed_input_shape.size() - 1],
              transposed_input_shape[transposed_input_shape.size() - 2]);
    auto input_def = TestInputDef<float>(transposed_input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Calculate flattened batch size
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      batch_size *= input_shape[i];
    }

    // Reshape to 2D (transposed)
    builder.Make1DInitializer<int64_t>("reshape_shape", {hidden_size, batch_size});
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    // Gemm weight: [hidden, output]
    std::vector<int64_t> weight_shape = {hidden_size, output_size};
    builder.MakeInitializer<float>("weight", weight_shape, -0.5f, 0.5f);

    // Gemm bias
    builder.MakeInitializer<float>("bias", {output_size}, -0.1f, 0.1f);

    // Gemm with transA=1: [hidden, batch*seq]^T x [hidden, output] -> [batch*seq, output]
    builder.AddNode("gemm", "Gemm", {"reshape_out", "weight", "bias"}, {"output"}, kOnnxDomain,
                    {builder.MakeScalarAttribute("transA", static_cast<int64_t>(1))});

    builder.MakeOutput("output");
  };
}

// Build a test case where Gemm has transB=1 (fusion should not happen)
GetTestModelFn BuildReshapeGemmWithTransBTestCase(const std::vector<int64_t>& input_shape,
                                                  int64_t hidden_size,
                                                  int64_t output_size) {
  return [input_shape, hidden_size, output_size](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_transB_graph");

    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      batch_size *= input_shape[i];
    }

    builder.Make1DInitializer<int64_t>("reshape_shape", {batch_size, hidden_size});
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    // Gemm weight: [output, hidden] (transposed for transB=1)
    std::vector<int64_t> weight_shape = {output_size, hidden_size};
    builder.MakeInitializer<float>("weight", weight_shape, -0.5f, 0.5f);

    builder.MakeInitializer<float>("bias", {output_size}, -0.1f, 0.1f);

    // Gemm with transB=1
    builder.AddNode("gemm", "Gemm", {"reshape_out", "weight", "bias"}, {"output"}, kOnnxDomain,
                    {builder.MakeScalarAttribute("transB", static_cast<int64_t>(1))});

    builder.MakeOutput("output");
  };
}

// Build a test case where Gemm weight is not constant (fusion should not happen)
GetTestModelFn BuildReshapeGemmDynamicWeightTestCase(const std::vector<int64_t>& input_shape,
                                                     int64_t hidden_size,
                                                     int64_t output_size) {
  return [input_shape, hidden_size, output_size](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_dynamic_weight_graph");

    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      batch_size *= input_shape[i];
    }

    builder.Make1DInitializer<int64_t>("reshape_shape", {batch_size, hidden_size});
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    // Dynamic weight (not initializer)
    std::vector<int64_t> weight_shape = {hidden_size, output_size};
    auto weight_def = TestInputDef<float>(weight_shape, false, -0.5f, 0.5f);
    MakeTestInput<float>(builder, "weight", weight_def);

    builder.MakeInitializer<float>("bias", {output_size}, -0.1f, 0.1f);

    builder.AddNode("gemm", "Gemm", {"reshape_out", "weight", "bias"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Build a test case with non-default alpha (fusion should not happen for Gemm)
GetTestModelFn BuildReshapeGemmNonDefaultAlphaTestCase(const std::vector<int64_t>& input_shape,
                                                       int64_t hidden_size,
                                                       int64_t output_size) {
  return [input_shape, hidden_size, output_size](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_gemm_alpha_graph");

    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      batch_size *= input_shape[i];
    }

    builder.Make1DInitializer<int64_t>("reshape_shape", {batch_size, hidden_size});
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    std::vector<int64_t> weight_shape = {hidden_size, output_size};
    builder.MakeInitializer<float>("weight", weight_shape, -0.5f, 0.5f);
    builder.MakeInitializer<float>("bias", {output_size}, -0.1f, 0.1f);

    // Gemm with non-default alpha
    builder.AddNode("gemm", "Gemm", {"reshape_out", "weight", "bias"}, {"output"}, kOnnxDomain,
                    {builder.MakeScalarAttribute("alpha", 0.5f)});

    builder.MakeOutput("output");
  };
}

// Test: Fusion should NOT happen when transA=1
TEST_F(QnnHTPBackendTests, ReshapeGemmFusion_Negative_TransA) {
  ProviderOptions provider_options = GetProviderOptions();

  // Model should still run, but fusion won't happen (Gemm handled separately)
  RunQnnModelTest(BuildReshapeGemmWithTransATestCase({1, 32, 64}, 64, 128),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Test: Fusion should NOT happen when transB=1
TEST_F(QnnHTPBackendTests, ReshapeGemmFusion_Negative_TransB) {
  ProviderOptions provider_options = GetProviderOptions();

  RunQnnModelTest(BuildReshapeGemmWithTransBTestCase({1, 32, 64}, 64, 128),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Test: Fusion should NOT happen when weight is dynamic
TEST_F(QnnHTPBackendTests, ReshapeGemmFusion_Negative_DynamicWeight) {
  ProviderOptions provider_options = GetProviderOptions();

  RunQnnModelTest(BuildReshapeGemmDynamicWeightTestCase({1, 32, 64}, 64, 128),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Test: Fusion should NOT happen when the input Reshape's input has rank 5.
// QNN HTP FullyConnected only supports input rank <= 4.
// Mirrors the proj/MatMul regression introduced by PR #232.
// All ops (Reshape, Gemm, Reshape) must still run on QNN EP via standalone builders.
TEST_F(QnnHTPBackendTests, ReshapeGemmFusion_Negative_Rank5Input) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeGemmFusion_Negative_Rank5Input";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // All nodes must run on QNN EP (Gemm handled standalone, not as fused FC)
  RunQnnModelTest(BuildReshapeGemmReshapeRank5InputTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  // Verify the fusion did NOT fire: one FullyConnected node and two Reshape nodes in the QNN graph
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 2);
}

// ============================================================================
// QDQ Tests - Fusion should NOT happen for QDQ-wrapped Gemm
// ============================================================================

// Build a QDQ test case: Reshape -> Q -> DQ -> Gemm -> Q -> DQ
// Fusion should NOT happen because Gemm is QDQ-wrapped
GetTestModelFn BuildQDQReshapeGemmTestCase(const std::vector<int64_t>& input_shape,
                                           int64_t hidden_size,
                                           int64_t output_size) {
  return [input_shape, hidden_size, output_size](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("qdq_reshape_gemm_graph");

    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      batch_size *= input_shape[i];
    }

    // Reshape
    builder.Make1DInitializer<int64_t>("reshape_shape", {batch_size, hidden_size});
    builder.AddNode("reshape", "Reshape", {"input", "reshape_shape"}, {"reshape_out"}, kOnnxDomain);

    // Q -> DQ on reshape output
    float scale = 0.01f;
    uint8_t zp = 128;
    builder.AddQuantizeLinearNode<uint8_t>("q1", "reshape_out", scale, zp, "q1_out", false);
    builder.AddDequantizeLinearNode<uint8_t>("dq1", "q1_out", scale, zp, "dq1_out", false);

    // Pre-quantized weight (uint8 initializer with DQ only - no Q node needed for initializers)
    std::vector<int64_t> weight_shape = {hidden_size, output_size};
    builder.MakeInitializer<uint8_t>("weight_q", weight_shape, static_cast<uint8_t>(64), static_cast<uint8_t>(192));
    builder.AddDequantizeLinearNode<uint8_t>("dq_weight", "weight_q", 0.01f, static_cast<uint8_t>(128), "dq_weight_out", false);

    // Bias (not quantized for Gemm)
    builder.MakeInitializer<float>("bias", {output_size}, -0.1f, 0.1f);

    // Gemm (QDQ-wrapped input and weight)
    builder.AddNode("gemm", "Gemm", {"dq1_out", "dq_weight_out", "bias"}, {"gemm_out"}, kOnnxDomain);

    // Q -> DQ on output
    builder.AddQuantizeLinearNode<uint8_t>("q2", "gemm_out", scale, zp, "q2_out", false);
    builder.AddDequantizeLinearNode<uint8_t>("dq2", "q2_out", scale, zp, "output", false);

    builder.MakeOutput("output");
  };
}

// Build a QDQ test case: Reshape -> Q -> DQ -> Gemm -> Q -> DQ -> Reshape
GetTestModelFn BuildQDQReshapeGemmReshapeTestCase(const std::vector<int64_t>& input_shape,
                                                  int64_t hidden_size,
                                                  int64_t output_size) {
  return [input_shape, hidden_size, output_size](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("qdq_reshape_gemm_reshape_graph");

    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
      batch_size *= input_shape[i];
    }

    // Input Reshape
    builder.Make1DInitializer<int64_t>("reshape1_shape", {batch_size, hidden_size});
    builder.AddNode("reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    // Q -> DQ
    float scale = 0.01f;
    uint8_t zp = 128;
    builder.AddQuantizeLinearNode<uint8_t>("q1", "reshape1_out", scale, zp, "q1_out", false);
    builder.AddDequantizeLinearNode<uint8_t>("dq1", "q1_out", scale, zp, "dq1_out", false);

    // Pre-quantized weight (uint8 initializer with DQ only - no Q node needed for initializers)
    std::vector<int64_t> weight_shape = {hidden_size, output_size};
    builder.MakeInitializer<uint8_t>("weight_q", weight_shape, static_cast<uint8_t>(64), static_cast<uint8_t>(192));
    builder.AddDequantizeLinearNode<uint8_t>("dq_weight", "weight_q", 0.01f, static_cast<uint8_t>(128), "dq_weight_out", false);

    // Bias
    builder.MakeInitializer<float>("bias", {output_size}, -0.1f, 0.1f);

    // Gemm
    builder.AddNode("gemm", "Gemm", {"dq1_out", "dq_weight_out", "bias"}, {"gemm_out"}, kOnnxDomain);

    // Q -> DQ on Gemm output
    builder.AddQuantizeLinearNode<uint8_t>("q2", "gemm_out", scale, zp, "q2_out", false);
    builder.AddDequantizeLinearNode<uint8_t>("dq2", "q2_out", scale, zp, "dq2_out", false);

    // Output Reshape
    std::vector<int64_t> output_shape_vec = input_shape;
    output_shape_vec.back() = output_size;
    builder.Make1DInitializer<int64_t>("reshape2_shape", output_shape_vec);
    builder.AddNode("reshape2", "Reshape", {"dq2_out", "reshape2_shape"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// Test: QDQ Reshape -> Gemm (fusion should NOT happen, QDQ Gemm handled differently)
TEST_F(QnnHTPBackendTests, ReshapeGemmFusion_QDQ_NoFusion) {
  ProviderOptions provider_options = GetProviderOptions();

  // QDQ model should run but ReshapeGemmFusion should not apply
  // (QDQ Gemm is handled by different code path)
  // Use ExpectedEPNodeAssignment::Some since not all nodes may be assigned to QNN EP
  RunQnnModelTest(BuildQDQReshapeGemmTestCase({1, 32, 64}, 64, 128),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(0.5f)});
}

// Test: QDQ Reshape -> Gemm -> Reshape (fusion should NOT happen)
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_QDQ_NoFusion) {
  ProviderOptions provider_options = GetProviderOptions();

  // Use ExpectedEPNodeAssignment::Some since not all nodes may be assigned to QNN EP
  RunQnnModelTest(BuildQDQReshapeGemmReshapeTestCase({1, 32, 64}, 64, 128),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(0.5f)});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
