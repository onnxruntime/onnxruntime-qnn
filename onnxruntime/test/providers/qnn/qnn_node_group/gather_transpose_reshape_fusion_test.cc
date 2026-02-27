// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>
#include <numeric>

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Helper to create the row-major indices
// indices[i, j] = i * idx1 + j
std::vector<int64_t> CreateRowMajorIndices(int64_t idx0, int64_t idx1) {
  std::vector<int64_t> indices(idx0 * idx1);
  for (int64_t i = 0; i < idx0; ++i) {
    for (int64_t j = 0; j < idx1; ++j) {
      indices[i * idx1 + j] = i * idx1 + j;
    }
  }
  return indices;
}

// Helper to create the col-major indices
// indices[i, j] = j * idx0 + i
std::vector<int64_t> CreateColMajorIndices(int64_t idx0, int64_t idx1) {
  std::vector<int64_t> indices(idx0 * idx1);
  for (int64_t i = 0; i < idx0; ++i) {
    for (int64_t j = 0; j < idx1; ++j) {
      indices[i * idx1 + j] = j * idx0 + i;
    }
  }
  return indices;
}

// Builds the graph: Gather(axis=4) -> Transpose -> Reshape
// This pattern typically produces a rank-6 intermediate tensor which QNN does not support.
// The fusion should replace it with rank-4 operations.
GetTestModelFn BuildGatherTransposeReshapeTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& indices_shape,
    const std::vector<int64_t>& indices_data,
    const std::vector<int64_t>& transpose_perm,
    const std::vector<int64_t>& final_reshape_shape) {
  return [input_def, indices_shape, indices_data, transpose_perm, final_reshape_shape](ModelTestBuilder& builder) {
    // Input: rank-5 [d0, d1, d2, d3, d4]
    NodeArg* input = MakeTestInput<float>(builder, input_def);

    // Constant Indices: rank-2 [idx0, idx1]
    NodeArg* indices = builder.MakeInitializer<int64_t>(indices_shape, indices_data);

    // Gather(axis=4)
    // Output shape: [d0, d1, d2, d3, idx0, idx1] (Rank 6)
    NodeArg* gather_out = builder.MakeIntermediate();
    NodeAttributes gather_attrs;
    gather_attrs["axis"] = utils::MakeAttribute("axis", int64_t(4));
    builder.AddNode("Gather", {input, indices}, {gather_out}, "", &gather_attrs);

    // Transpose
    // Output shape: Rank 6 (permuted)
    NodeArg* transpose_out = builder.MakeIntermediate();
    NodeAttributes transpose_attrs;
    transpose_attrs["perm"] = utils::MakeAttribute("perm", transpose_perm);
    builder.AddNode("Transpose", {gather_out}, {transpose_out}, "", &transpose_attrs);

    // Reshape -> Final Output
    // Output shape: final_reshape_shape (Rank < 6)
    NodeArg* shape_const = builder.Make1DInitializer<int64_t>(final_reshape_shape);
    NodeArg* output = builder.MakeOutput();
    builder.AddNode("Reshape", {transpose_out, shape_const}, {output});
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

// Test Case 1: Row-Major Indices
// Input: [1, 1, 1, 32, 64]
// Indices: [8, 8] (covering 64 elements) -> Row Major
// Gather Out: [1, 1, 1, 32, 8, 8] (Rank 6)
// Transpose Perm: [0, 1, 2, 4, 3, 5] (swap d3 and idx0) -> [1, 1, 1, 8, 32, 8]
// Reshape: [1, 8, 32, 8]
TEST_F(QnnHTPBackendTests, GatherTransposeReshape_Fusion_RowMajor) {
  const std::filesystem::path json_qnn_graph_dir = "GatherTransposeReshape_Fusion_RowMajor";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Input dimensions
  int64_t d0 = 1, d1 = 1, d2 = 1, d3 = 32, d4 = 64;
  std::vector<int64_t> input_shape = {d0, d1, d2, d3, d4};
  auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);

  // Indices dimensions (d4 = 64 = 8 * 8)
  int64_t idx0 = 8, idx1 = 8;
  std::vector<int64_t> indices_shape = {idx0, idx1};
  std::vector<int64_t> indices_data = CreateRowMajorIndices(idx0, idx1);

  // Transpose Permutation: [0, 1, 2, 4, 3, 5]
  // Permuting the last 3 dims (3,4,5) -> (4,3,5)
  // This is a valid permutation for the fusion (perm[0-2] are fixed, perm[3-5] are {3,4,5})
  std::vector<int64_t> transpose_perm = {0, 1, 2, 4, 3, 5};

  // Final Reshape
  std::vector<int64_t> final_shape = {1, 8, 32, 8};

  RunQnnModelTest(BuildGatherTransposeReshapeTestCase(input_def, indices_shape, indices_data, transpose_perm, final_shape),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);  // fp32_abs_err: HTP uses fixed-point arithmetic, allow small numerical differences

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gather", 0);
}

// Test Case 2: Col-Major Indices
// Input: [1, 1, 1, 16, 100]
// Indices: [10, 10] (covering 100 elements) -> Col Major
// Gather Out: [1, 1, 1, 16, 10, 10] (Rank 6)
// Transpose Perm: [0, 1, 2, 5, 4, 3] (reverse last 3) -> [1, 1, 1, 10, 10, 16]
// Reshape: [1, 100, 16]
TEST_F(QnnHTPBackendTests, GatherTransposeReshape_Fusion_ColMajor) {
  const std::filesystem::path json_qnn_graph_dir = "GatherTransposeReshape_Fusion_ColMajor";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Input dimensions
  int64_t d0 = 1, d1 = 1, d2 = 1, d3 = 16, d4 = 100;
  std::vector<int64_t> input_shape = {d0, d1, d2, d3, d4};
  auto input_def = TestInputDef<float>(input_shape, false, -5.0f, 5.0f);

  // Indices dimensions (d4 = 100 = 10 * 10)
  int64_t idx0 = 10, idx1 = 10;
  std::vector<int64_t> indices_shape = {idx0, idx1};
  std::vector<int64_t> indices_data = CreateColMajorIndices(idx0, idx1);

  // Transpose Permutation: [0, 1, 2, 5, 4, 3]
  // Valid permutation: perm[0-2] fixed, perm[3-5] is set {3,4,5}
  std::vector<int64_t> transpose_perm = {0, 1, 2, 5, 4, 3};

  // Final Reshape
  std::vector<int64_t> final_shape = {1, 100, 16};

  RunQnnModelTest(BuildGatherTransposeReshapeTestCase(input_def, indices_shape, indices_data, transpose_perm, final_shape),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);  // fp32_abs_err: HTP uses fixed-point arithmetic, allow small numerical differences

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gather", 0);
}

// Asymmetric indices: idx0 != idx1 (6x8 split of d4=48), row-major.
// Input: [1, 1, 1, 8, 48], Gather Out: [1, 1, 1, 8, 6, 8] (Rank 6)
// Transpose Perm: [0, 1, 2, 4, 3, 5] -> [1, 1, 1, 6, 8, 8]
// Reshape: [1, 6, 8, 8]
TEST_F(QnnHTPBackendTests, GatherTransposeReshape_Fusion_AsymmetricIndices) {
  const std::filesystem::path json_qnn_graph_dir = "GatherTransposeReshape_Fusion_AsymmetricIndices";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({1, 1, 1, 8, 48}, false, -1.0f, 1.0f);
  std::vector<int64_t> indices_data = CreateRowMajorIndices(6, 8);

  RunQnnModelTest(
      BuildGatherTransposeReshapeTestCase(input_def, {6, 8}, indices_data, {0, 1, 2, 4, 3, 5}, {1, 6, 8, 8}),
      provider_options, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gather", 0);
}

// Merged batch dims > 1: d0*d1*d2 = 2*3*4 = 24, col-major indices.
// Input: [2, 3, 4, 16, 36], Gather Out: [2, 3, 4, 16, 6, 6] (Rank 6)
// Transpose Perm: [0, 1, 2, 5, 4, 3] -> [2, 3, 4, 6, 6, 16]
// Reshape: [24, 36, 16]
TEST_F(QnnHTPBackendTests, GatherTransposeReshape_Fusion_MergedBatchDims) {
  const std::filesystem::path json_qnn_graph_dir = "GatherTransposeReshape_Fusion_MergedBatchDims";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({2, 3, 4, 16, 36}, false, -1.0f, 1.0f);
  std::vector<int64_t> indices_data = CreateColMajorIndices(6, 6);

  RunQnnModelTest(
      BuildGatherTransposeReshapeTestCase(input_def, {6, 6}, indices_data, {0, 1, 2, 5, 4, 3}, {24, 36, 16}),
      provider_options, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gather", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
