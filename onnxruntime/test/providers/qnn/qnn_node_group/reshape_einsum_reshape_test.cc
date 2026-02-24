// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>

#include <gsl/gsl_util>
#include "gtest/gtest.h"

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
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

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, ReshapeEinsumReshape_Float) {
  const std::filesystem::path json_qnn_graph_dir = "ReshapeEinsumReshape_Float";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildReshapeEinsumReshapeFloatTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "DepthToSpace", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Einsum", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
