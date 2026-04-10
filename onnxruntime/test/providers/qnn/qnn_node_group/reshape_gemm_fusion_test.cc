// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

}  // namespace

// Test 2-node fusion: Reshape -> Gemm (3D input)
TEST_F(QnnHTPBackendTests, ReshapeGemmFusion_3D) {
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
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  // Verify FullyConnected is in the graph (fusion happened)
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
}

// Test 3-node fusion: Reshape -> Gemm -> Reshape (3D input)
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_3D) {
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
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  // Verify FullyConnected and one Reshape (output reshape kept)
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

// Test 3-node fusion: Reshape -> Gemm -> Reshape (4D input)
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_4D) {
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
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  // Verify FullyConnected and one Reshape
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

// Test 4-node fusion: Reshape -> Gemm -> Reshape -> Reshape
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeReshapeFusion_3D) {
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
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  // Verify FullyConnected and one Reshape (only final reshape kept)
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

// Test 3-node fusion with ViT-like pattern: keep first dim, flatten last dims
// [197, 1, 12, 64] -> [197, 768] -> Gemm -> [197, 768] -> [197, 1, 768]
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_ViTPattern) {
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
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  // Verify FullyConnected and one Reshape
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

// Test with transformer-like shape (smaller for unit test)
TEST_F(QnnHTPBackendTests, ReshapeGemmReshapeFusion_Transformer) {
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
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  // Verify FullyConnected and one Reshape
  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
