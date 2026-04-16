// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

// Builds the graph: Input -> Transpose -> Reshape -> Transpose -> Output
// This pattern can be fused to a single Reshape when dimensions are only merged (not reordered).
GetTestModelFn BuildTransposeReshapeTransposeTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& perm1,
    const std::vector<int64_t>& reshape_shape,
    const std::vector<int64_t>& perm2) {
  return [input_def, perm1, reshape_shape, perm2](ModelTestBuilder& builder) {
    // Input
    MakeTestInput<float>(builder, "input", input_def);

    // Transpose1: input -> transpose1_out
    builder.AddNode("transpose1", "Transpose", {"input"}, {"transpose1_out"}, "",
                    {test::MakeAttribute("perm", perm1)});

    // Reshape: transpose1_out -> reshape_out
    builder.Make1DInitializer<int64_t>("reshape_shape", reshape_shape);
    builder.AddNode("reshape", "Reshape", {"transpose1_out", "reshape_shape"}, {"reshape_out"});

    // Transpose2: reshape_out -> output
    builder.MakeOutput("output");
    builder.AddNode("transpose2", "Transpose", {"reshape_out"}, {"output"}, "",
                    {test::MakeAttribute("perm", perm2)});
  };
}

// Builds a graph with ops before and after the fusion pattern to ensure it works in context.
// Input -> Add -> Transpose -> Reshape -> Transpose -> Add -> Output
GetTestModelFn BuildTransposeReshapeTransposeWithContextTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& perm1,
    const std::vector<int64_t>& reshape_shape,
    const std::vector<int64_t>& perm2) {
  return [input_def, perm1, reshape_shape, perm2](ModelTestBuilder& builder) {
    // Input
    MakeTestInput<float>(builder, "input", input_def);

    // Add1: input + 1 -> add1_out
    builder.MakeScalarInitializer<float>("add_const1", 1.0f);
    builder.AddNode("add1", "Add", {"input", "add_const1"}, {"add1_out"});

    // Transpose1: add1_out -> transpose1_out
    builder.AddNode("transpose1", "Transpose", {"add1_out"}, {"transpose1_out"}, "",
                    {test::MakeAttribute("perm", perm1)});

    // Reshape: transpose1_out -> reshape_out
    builder.Make1DInitializer<int64_t>("reshape_shape", reshape_shape);
    builder.AddNode("reshape", "Reshape", {"transpose1_out", "reshape_shape"}, {"reshape_out"});

    // Transpose2: reshape_out -> transpose2_out
    builder.AddNode("transpose2", "Transpose", {"reshape_out"}, {"transpose2_out"}, "",
                    {test::MakeAttribute("perm", perm2)});

    // Add2: transpose2_out + 1 -> output
    builder.MakeScalarInitializer<float>("add_const2", 1.0f);
    builder.MakeOutput("output");
    builder.AddNode("add2", "Add", {"transpose2_out", "add_const2"}, {"output"});
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Test Case 1: Basic fusable pattern
// Input: [2, 3, 4] (A=2, B=3, C=4)
// Transpose1 perm=[1, 2, 0] -> [3, 4, 2] (B, C, A)
// Reshape -> [12, 2] (B*C, A)
// Transpose2 perm=[1, 0] -> [2, 12] (A, B*C)
// This is equivalent to Reshape [2, 3, 4] -> [2, 12]
TEST_F(QnnHTPBackendTests, TransposeReshapeTransposeFusion_Basic) {
  const std::filesystem::path json_qnn_graph_dir = "TransposeReshapeTransposeFusion_Basic";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  std::vector<int64_t> input_shape = {2, 3, 4};
  auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);

  std::vector<int64_t> perm1 = {1, 2, 0};
  std::vector<int64_t> reshape_shape = {12, 2};
  std::vector<int64_t> perm2 = {1, 0};

  RunQnnModelTest(BuildTransposeReshapeTransposeTestCase(input_def, perm1, reshape_shape, perm2),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);

  // Verify fusion: should have Reshape, no Transpose
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Test Case 2: Fusable pattern with surrounding ops
// Same as Test Case 1 but with Add ops before and after
TEST_F(QnnHTPBackendTests, TransposeReshapeTransposeFusion_WithContext) {
  const std::filesystem::path json_qnn_graph_dir = "TransposeReshapeTransposeFusion_WithContext";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  std::vector<int64_t> input_shape = {2, 3, 4};
  auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);

  std::vector<int64_t> perm1 = {1, 2, 0};
  std::vector<int64_t> reshape_shape = {12, 2};
  std::vector<int64_t> perm2 = {1, 0};

  RunQnnModelTest(BuildTransposeReshapeTransposeWithContextTestCase(input_def, perm1, reshape_shape, perm2),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);

  // Verify fusion: should have Reshape (from fusion), no Transpose
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Test Case 3: 4D tensor fusion
// Input: [1, 2, 3, 4] -> Transpose [2, 3, 0, 1] -> [3, 4, 1, 2]
// Reshape -> [12, 2] (merge dims 0,1 and 2,3)
// Transpose [1, 0] -> [2, 12]
// Equivalent to Reshape [1, 2, 3, 4] -> [2, 12]
TEST_F(QnnHTPBackendTests, TransposeReshapeTransposeFusion_4D) {
  const std::filesystem::path json_qnn_graph_dir = "TransposeReshapeTransposeFusion_4D";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  std::vector<int64_t> input_shape = {1, 2, 3, 4};
  auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);

  std::vector<int64_t> perm1 = {2, 3, 0, 1};
  std::vector<int64_t> reshape_shape = {12, 2};
  std::vector<int64_t> perm2 = {1, 0};

  RunQnnModelTest(BuildTransposeReshapeTransposeTestCase(input_def, perm1, reshape_shape, perm2),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);

  // Verify fusion: should have Reshape, no Transpose
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Test Case 4: Trivial fusion (both transposes are identity)
// Input: [2, 3, 4]
// Transpose1 perm=[0, 1, 2] (identity) -> [2, 3, 4]
// Reshape -> [2, 12]
// Transpose2 perm=[0, 1] (identity) -> [2, 12]
// Equivalent to Reshape [2, 3, 4] -> [2, 12]
TEST_F(QnnHTPBackendTests, TransposeReshapeTransposeFusion_IdentityTransposes) {
  const std::filesystem::path json_qnn_graph_dir = "TransposeReshapeTransposeFusion_IdentityTransposes";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  std::vector<int64_t> input_shape = {2, 3, 4};
  auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);

  std::vector<int64_t> perm1 = {0, 1, 2};
  std::vector<int64_t> reshape_shape = {2, 12};
  std::vector<int64_t> perm2 = {0, 1};

  RunQnnModelTest(BuildTransposeReshapeTransposeTestCase(input_def, perm1, reshape_shape, perm2),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);

  // Verify fusion: should have Reshape, no Transpose
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Test Case 5: Merge first two dimensions
// Input: [2, 3, 4] (A=2, B=3, C=4)
// Transpose1 perm=[0, 1, 2] (identity) -> [2, 3, 4]
// Reshape -> [6, 4] (A*B, C)
// Transpose2 perm=[0, 1] (identity) -> [6, 4]
// Equivalent to Reshape [2, 3, 4] -> [6, 4]
TEST_F(QnnHTPBackendTests, TransposeReshapeTransposeFusion_MergeFirstTwoDims) {
  const std::filesystem::path json_qnn_graph_dir = "TransposeReshapeTransposeFusion_MergeFirstTwoDims";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  std::vector<int64_t> input_shape = {2, 3, 4};
  auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);

  std::vector<int64_t> perm1 = {0, 1, 2};
  std::vector<int64_t> reshape_shape = {6, 4};
  std::vector<int64_t> perm2 = {0, 1};

  RunQnnModelTest(BuildTransposeReshapeTransposeTestCase(input_def, perm1, reshape_shape, perm2),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);

  // Verify fusion: should have Reshape, no Transpose
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Test Case 6: Non-fusable pattern (dimensions out of order after transformation)
// Input: [2, 3, 4] (A=2, B=3, C=4)
// Transpose1 perm=[2, 0, 1] -> [4, 2, 3] (C, A, B)
// Reshape -> [4, 6] (keep C, merge A*B)
// Transpose2 perm=[0, 1] (identity) -> [4, 6]
// NOT fusable because output dim 0 is original dim 2 (C), not dim 0 (A)
// final_mapping = [[2], [0, 1]] -> order check: expect 0, got 2 -> FAIL
TEST_F(QnnHTPBackendTests, TransposeReshapeTransposeFusion_NotFusable_Reordered) {
  const std::filesystem::path json_qnn_graph_dir = "TransposeReshapeTransposeFusion_NotFusable_Reordered";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  std::vector<int64_t> input_shape = {2, 3, 4};
  auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);

  std::vector<int64_t> perm1 = {2, 0, 1};
  std::vector<int64_t> reshape_shape = {4, 6};
  std::vector<int64_t> perm2 = {0, 1};

  RunQnnModelTest(BuildTransposeReshapeTransposeTestCase(input_def, perm1, reshape_shape, perm2),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);

  // Verify NO fusion: should have Transpose ops (fusion did not happen)
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 2);
}

// Test Case 7: Larger tensor with batch dimension
// Input: [8, 16, 32] (batch=8, height=16, width=32)
// Transpose1 perm=[1, 2, 0] -> [16, 32, 8]
// Reshape -> [512, 8]
// Transpose2 perm=[1, 0] -> [8, 512]
// Equivalent to Reshape [8, 16, 32] -> [8, 512]
TEST_F(QnnHTPBackendTests, TransposeReshapeTransposeFusion_LargerTensor) {
  const std::filesystem::path json_qnn_graph_dir = "TransposeReshapeTransposeFusion_LargerTensor";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  std::vector<int64_t> input_shape = {8, 16, 32};
  auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);

  std::vector<int64_t> perm1 = {1, 2, 0};
  std::vector<int64_t> reshape_shape = {512, 8};
  std::vector<int64_t> perm2 = {1, 0};

  RunQnnModelTest(BuildTransposeReshapeTransposeTestCase(input_def, perm1, reshape_shape, perm2),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);

  // Verify fusion: should have Reshape, no Transpose
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
