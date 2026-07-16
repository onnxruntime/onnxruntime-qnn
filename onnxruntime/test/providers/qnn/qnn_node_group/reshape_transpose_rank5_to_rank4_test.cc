// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <vector>

#include <gsl/util>
#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Builds: Input -> Add -> Reshape(rank-5) -> Transpose -> Reshape -> Add -> Output
// The Add ops surround the fusion pattern so the Reshape/Transpose/Reshape triple is
// fully internal to the QNN subgraph and the fusion can fire.
GetTestModelFn BuildRank5ToRank4FloatTestCase(const std::vector<int64_t>& input_shape,
                                              const std::vector<int64_t>& reshape1_shape,
                                              const std::vector<int64_t>& perm,
                                              const std::vector<int64_t>& reshape2_shape) {
  return [input_shape, reshape1_shape, perm, reshape2_shape](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("rank5_to_rank4_fusion_float_graph");

    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    builder.MakeScalarInitializer<float>("add_const1", 1.0f);
    builder.AddNode("add1", "Add", {"input", "add_const1"}, {"add1_out"}, kOnnxDomain);

    builder.Make1DInitializer<int64_t>("reshape1_shape", reshape1_shape);
    builder.AddNode("reshape1", "Reshape", {"add1_out", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    builder.AddNode("transpose", "Transpose", {"reshape1_out"}, {"transpose_out"}, kOnnxDomain,
                    {test::MakeAttribute("perm", perm)});

    builder.Make1DInitializer<int64_t>("reshape2_shape", reshape2_shape);
    builder.AddNode("reshape2", "Reshape", {"transpose_out", "reshape2_shape"}, {"reshape2_out"}, kOnnxDomain);

    builder.MakeScalarInitializer<float>("add_const2", 1.0f);
    builder.AddNode("add2", "Add", {"reshape2_out", "add_const2"}, {"output"}, kOnnxDomain);
    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

// Positive: docstring example pattern.
// Input [1, 12, 12, 8] -> Reshape [3, 4, 3, 4, 8] -> Transpose perm=[0, 2, 1, 3, 4] -> Reshape.
// perm has the consecutive pair (perm[3]=3, perm[4]=4) at positions 3,4, so input dims 3 and 4
// (sizes 4 and 8) merge into a single dim of size 32, dropping the rank from 5 to 4.
TEST_F(QnnHTPBackendTests, Rank5ToRank4Fusion_Float_DocstringExample) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "Rank5ToRank4Fusion_Float_DocstringExample";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildRank5ToRank4FloatTestCase(
                      /*input_shape=*/{1, 12, 12, 8},
                      /*reshape1_shape=*/{3, 4, 3, 4, 8},
                      /*perm=*/{0, 2, 1, 3, 4},
                      /*reshape2_shape=*/{1, 12, 12, 8}),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  // After fusion the Transpose runs on rank-4 tensors. The two Reshapes survive (one before and
  // one after the Transpose). The fusion does not change op counts in the QNN graph, only ranks.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 2);
}

// Positive: consecutive pair at the beginning of perm.
// perm=[2, 3, 0, 1, 4] has perm[0]=2, perm[1]=3 consecutive at position 0, so input dims 2 and 3 merge.
// Model input/output are rank-4; the rank-5 tensors only exist between the two Reshapes.
TEST_F(QnnHTPBackendTests, Rank5ToRank4Fusion_Float_MergeAtStart) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "Rank5ToRank4Fusion_Float_MergeAtStart";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildRank5ToRank4FloatTestCase(
                      /*input_shape=*/{6, 4, 5, 6},
                      /*reshape1_shape=*/{2, 3, 4, 5, 6},
                      /*perm=*/{2, 3, 0, 1, 4},
                      /*reshape2_shape=*/{4, 5, 6, 6}),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 2);
}

// Positive: consecutive pair at perm position 0 with values that exercise the value-shift branch.
// perm=[2, 3, 4, 0, 1] -- merge at position 0 picks input dims 2 and 3; the trailing perm value 4
// is greater than merge_input_idx_b=3 and must be shifted down by one in the rank-4 perm.
TEST_F(QnnHTPBackendTests, Rank5ToRank4Fusion_Float_MergeWithValueShift) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "Rank5ToRank4Fusion_Float_MergeWithValueShift";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildRank5ToRank4FloatTestCase(
                      /*input_shape=*/{6, 4, 5, 6},
                      /*reshape1_shape=*/{2, 3, 4, 5, 6},
                      /*perm=*/{2, 3, 4, 0, 1},
                      /*reshape2_shape=*/{4, 5, 6, 6}),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 2);
}

// Negative: rank-4 intermediate tensors. The fusion requires Rank(t1) == Rank(t2) == 5,
// so a Reshape -> Transpose -> Reshape with rank-4 intermediates must not be picked up by
// Rank5ToRank4Fusion. The model still runs on QNN EP via the standalone op builders.
TEST_F(QnnHTPBackendTests, Rank5ToRank4Fusion_Float_Rank4Intermediate_NoFusion) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "Rank5ToRank4Fusion_Float_Rank4Intermediate_NoFusion";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Rank-4 reshape between a rank-2 input and rank-2 output, so the intermediate tensors
  // (t1, t2) are rank-4 -- the fusion's rank-5 condition is unsatisfied.
  auto build_rank4 = [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("rank5_to_rank4_fusion_no_fusion_rank4_graph");

    auto input_def = TestInputDef<float>({12, 8}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    builder.MakeScalarInitializer<float>("add_const1", 1.0f);
    builder.AddNode("add1", "Add", {"input", "add_const1"}, {"add1_out"}, kOnnxDomain);

    builder.Make1DInitializer<int64_t>("reshape1_shape", {3, 4, 2, 4});
    builder.AddNode("reshape1", "Reshape", {"add1_out", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    builder.AddNode("transpose", "Transpose", {"reshape1_out"}, {"transpose_out"}, kOnnxDomain,
                    {test::MakeAttribute("perm", std::vector<int64_t>{0, 2, 1, 3})});

    builder.Make1DInitializer<int64_t>("reshape2_shape", {12, 8});
    builder.AddNode("reshape2", "Reshape", {"transpose_out", "reshape2_shape"}, {"reshape2_out"}, kOnnxDomain);

    builder.MakeScalarInitializer<float>("add_const2", 1.0f);
    builder.AddNode("add2", "Add", {"reshape2_out", "add_const2"}, {"output"}, kOnnxDomain);
    builder.MakeOutput("output");
  };

  RunQnnModelTest(build_rank4, provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  // Rank5ToRank4Fusion did not fire; the rank-4 Transpose remains.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Reshape", 2);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
