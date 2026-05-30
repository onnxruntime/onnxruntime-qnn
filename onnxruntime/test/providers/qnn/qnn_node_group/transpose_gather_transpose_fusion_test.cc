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

// Builds: Input -> Transpose(perm1) -> Gather(axis, indices) -> Transpose(perm2) -> Output
// `indices` is always a constant initializer (the fusion requires this).
GetTestModelFn BuildTransposeGatherTransposeTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& perm1,
    const std::vector<int64_t>& indices_shape,
    const std::vector<int64_t>& indices_data,
    int64_t gather_axis,
    const std::vector<int64_t>& perm2) {
  return [input_def, perm1, indices_shape, indices_data, gather_axis, perm2](
             ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("transpose1", "Transpose", {"input"}, {"transpose1_out"}, "",
                    {test::MakeAttribute("perm", perm1)});

    builder.MakeInitializer<int64_t>("indices", indices_shape, indices_data);
    builder.AddNode("gather", "Gather", {"transpose1_out", "indices"}, {"gather_out"}, "",
                    {test::MakeAttribute("axis", gather_axis)});

    builder.MakeOutput("output");
    builder.AddNode("transpose2", "Transpose", {"gather_out"}, {"output"}, "",
                    {test::MakeAttribute("perm", perm2)});
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Gather(axis=0, indices=[1] scalar-shape vec of rank 1) selects batch index 1, perm2=[0,2,3,1]
TEST_F(QnnHTPBackendTests, TransposeGatherTransposeFusion_axis0) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "TransposeGatherTransposeFusion_axis0";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({2, 8, 4, 6}, false, -1.0f, 1.0f);
  std::vector<int64_t> perm1 = {0, 3, 1, 2};  // [2,8,4,6] -> [2,6,8,4]
  std::vector<int64_t> indices_shape = {1};
  std::vector<int64_t> indices_data = {1};  // pick batch index 1
  int64_t gather_axis = 0;
  std::vector<int64_t> perm2 = {0, 2, 3, 1};  // [1,6,8,4] -> [1,8,4,6]

  RunQnnModelTest(BuildTransposeGatherTransposeTestCase(input_def, perm1, indices_shape, indices_data,
                                                        gather_axis, perm2),
                  provider_options, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gather", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Input: [3, 5, 7, 4]
// perm1 = [2, 0, 1, 3] -> t1 shape [7, 3, 5, 4]
// Gather(axis=1, indices_shape=[1]) -> shape [7, 1, 5, 4]   (rank 4, K=1)
// fused_axis = perm1[1] = 0. Equivalent fused op: Gather(x, indices, axis=0) -> [1, 5, 7, 4]
// perm2 = [1, 2, 0, 3]
TEST_F(QnnHTPBackendTests, TransposeGatherTransposeFusion_axis1) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "TransposeGatherTransposeFusion_K1InteriorAxis";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({3, 5, 7, 4}, false, -1.0f, 1.0f);
  std::vector<int64_t> perm1 = {2, 0, 1, 3};  // [3,5,7,4] -> [7,3,5,4]
  std::vector<int64_t> indices_shape = {1};
  std::vector<int64_t> indices_data = {1};
  int64_t gather_axis = 1;  // dim of size 3 in t1 = source dim 0
  std::vector<int64_t> perm2 = {1, 2, 0, 3};

  RunQnnModelTest(BuildTransposeGatherTransposeTestCase(input_def, perm1, indices_shape, indices_data,
                                                        gather_axis, perm2),
                  provider_options, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gather", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Negative gather_axis must be normalized before MatchPattern accepts it.
// Same shape setup as the PSPNet case but with gather axis = -4 (== 0).
TEST_F(QnnHTPBackendTests, TransposeGatherTransposeFusion_NegativeAxis) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "TransposeGatherTransposeFusion_NegativeAxis";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({2, 8, 4, 6}, false, -1.0f, 1.0f);
  std::vector<int64_t> perm1 = {0, 3, 1, 2};
  std::vector<int64_t> indices_shape = {1};
  std::vector<int64_t> indices_data = {0};
  int64_t gather_axis = -4;  // == 0 after normalization
  std::vector<int64_t> perm2 = {0, 2, 3, 1};

  RunQnnModelTest(BuildTransposeGatherTransposeTestCase(input_def, perm1, indices_shape, indices_data,
                                                        gather_axis, perm2),
                  provider_options, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gather", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Identity transposes on both ends: trivially cancelable. perm1 = perm2 = identity.
// IsCancelingPair must accept this; the result is just a plain Gather.
TEST_F(QnnHTPBackendTests, TransposeGatherTransposeFusion_IdentityTransposes) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "TransposeGatherTransposeFusion_IdentityTransposes";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({2, 4, 5, 6}, false, -1.0f, 1.0f);
  std::vector<int64_t> perm1 = {0, 1, 2, 3};
  std::vector<int64_t> indices_shape = {1};
  std::vector<int64_t> indices_data = {1};
  int64_t gather_axis = 1;
  std::vector<int64_t> perm2 = {0, 1, 2, 3};

  RunQnnModelTest(BuildTransposeGatherTransposeTestCase(input_def, perm1, indices_shape, indices_data,
                                                        gather_axis, perm2),
                  provider_options, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Gather", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 0);
}

// Non-canceling perm2: the second transpose does NOT undo perm1+gather, so IsCancelingPair
// must return false and the three nodes must NOT collapse. They should still go through
// QNN unfused (HTP supports each individually for these ranks/dtypes).
TEST_F(QnnHTPBackendTests, TransposeGatherTransposeFusion_NotCanceling) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "TransposeGatherTransposeFusion_NotCanceling";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({2, 8, 4, 6}, false, -1.0f, 1.0f);
  std::vector<int64_t> perm1 = {0, 3, 1, 2};
  std::vector<int64_t> indices_shape = {1};
  std::vector<int64_t> indices_data = {0};
  int64_t gather_axis = 0;
  // Correct cancel would be {0,2,3,1}; this swaps two dims so it doesn't cancel.
  std::vector<int64_t> perm2 = {0, 3, 2, 1};

  RunQnnModelTest(BuildTransposeGatherTransposeTestCase(input_def, perm1, indices_shape, indices_data,
                                                        gather_axis, perm2),
                  provider_options, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  // Both Transpose ops survive; Gather still present.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 2);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Gather", 1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
