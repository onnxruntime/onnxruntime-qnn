// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>

#include <gsl/gsl_util>
#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

GetTestModelFn BuildReshapeEinsumReshapeFloatTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reshape_einsum_reshape_float_graph");

    auto input_def = TestInputDef<float>({1024, 32}, false, -10.0f, 10.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Reshape: (1024, 32) -> (1, 32, 32, 2, 2, 8)
    builder.Make1DInitializer<int64_t>("reshape1_shape", {1, 32, 32, 2, 2, 8});
    builder.AddNode("reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    // Einsum: equation "nhwpqc->nchpwq" (equivalent to Transpose with perm [0, 5, 1, 3, 2, 4])
    builder.AddNode("einsum",
                    "Einsum",
                    {"reshape1_out"},
                    {"einsum_out"},
                    kOnnxDomain,
                    {test::MakeAttribute("equation", "nhwpqc->nchpwq")});

    // Reshape: (1, 8, 32, 2, 32, 2) -> (1, 8, 64, 64)
    builder.Make1DInitializer<int64_t>("reshape2_shape", {1, 8, 64, 64});
    builder.AddNode("reshape2", "Reshape", {"einsum_out", "reshape2_shape"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

GetTestModelFn BuildNonRank6ReshapeEinsumReshapeTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("non_rank6_reshape_einsum_reshape_graph");

    auto input_def = TestInputDef<float>({512, 32}, false, -10.0f, 10.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Reshape to rank 5. This is a valid standalone Einsum transpose pattern,
    // but it must not be replaced by the rank-6-only node group.
    builder.Make1DInitializer<int64_t>("reshape1_shape", {1, 32, 32, 2, 8});
    builder.AddNode("reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    builder.AddNode("einsum",
                    "Einsum",
                    {"reshape1_out"},
                    {"einsum_out"},
                    kOnnxDomain,
                    {test::MakeAttribute("equation", "nhwpc->nchpw")});

    builder.Make1DInitializer<int64_t>("reshape2_shape", {1, 8, 64, 32});
    builder.AddNode("reshape2", "Reshape", {"einsum_out", "reshape2_shape"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, ReshapeEinsumReshape_DerivesAllFixedIndexShapes) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeEinsumReshape_DerivesAllFixedIndexShapes";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeEinsumReshapeFloatTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "DepthToSpace", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Einsum", 0);
  AssertTensorShapeInQnnGraph(json_qnn_graph_dir, "reshape1_out", {1, 32, 32, 32});
  AssertTensorShapeInQnnGraph(json_qnn_graph_dir, "einsum_out", {1, 64, 64, 8});
  AssertTensorShapeInQnnGraph(json_qnn_graph_dir, "output", {1, 8, 64, 64});
}

TEST_F(QnnHTPBackendTests, ReshapeEinsumReshape_RejectsNonRank6Input) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "ReshapeEinsumReshape_RejectsNonRank6Input";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildNonRank6ReshapeEinsumReshapeTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(1e-2f)});

  // The standalone operators may be assigned to QNN, but the rank-6-only fusion must not fire.
  AssertOpInQnnGraph(json_qnn_graph_dir, "DepthToSpace", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
