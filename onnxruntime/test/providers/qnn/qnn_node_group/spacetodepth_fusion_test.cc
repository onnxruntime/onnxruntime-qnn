// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>

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

GetTestModelFn BuildSpaceToDepthTestCase(const std::vector<int64_t>& input_shape,
                                         int64_t block_height,
                                         int64_t block_width,
                                         const std::vector<int64_t>& perm,
                                         bool use_qdq) {
  return [=](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("spacetodepth_fusion_graph");

    const auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    std::string reshape1_input = "input";
    if (use_qdq) {
      const QuantParams<uint8_t> input_qparams = GetTestInputQuantParams<uint8_t>(input_def);
      reshape1_input = AddQDQNodePair<uint8_t>(builder, "qdq_in", "input",
                                               input_qparams.scale, input_qparams.zero_point);
    }

    const int64_t n = input_shape[0];
    const int64_t c = input_shape[1];
    const int64_t h = input_shape[2];
    const int64_t w = input_shape[3];
    const int64_t h_div = h / block_height;
    const int64_t w_div = w / block_width;

    // Reshape1: NCHW -> [N, C, H/block_h, block_h, W/block_w, block_w]
    builder.Make1DInitializer<int64_t>("reshape1_shape", {n, c, h_div, block_height, w_div, block_width});
    builder.AddNode("Reshape1",
                    "Reshape",
                    {reshape1_input, "reshape1_shape"},
                    {"reshape1_out"},
                    kOnnxDomain);

    std::string transpose_input = "reshape1_out";
    if (use_qdq) {
      const QuantParams<uint8_t> input_qparams = GetTestInputQuantParams<uint8_t>(input_def);
      transpose_input = AddQDQNodePair<uint8_t>(builder, "qdq_after_reshape1", "reshape1_out",
                                                input_qparams.scale, input_qparams.zero_point);
    }

    builder.AddNode("Transpose",
                    "Transpose",
                    {transpose_input},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", perm)});

    // Reshape2: rank-6 -> [N, C*block_h*block_w, H/block_h, W/block_w]
    builder.Make1DInitializer<int64_t>("reshape2_shape", {n, c * block_height * block_width, h_div, w_div});
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape"},
                    {"reshape2_out"},
                    kOnnxDomain);

    builder.MakeOutput("reshape2_out");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

void RunSpaceToDepthFusionTest(const std::filesystem::path& json_qnn_graph_dir,
                               const std::vector<int64_t>& input_shape,
                               int64_t block_height,
                               int64_t block_width,
                               const std::vector<int64_t>& perm,
                               bool use_qdq) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildSpaceToDepthTestCase(input_shape, block_height, block_width, perm, use_qdq),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "SpaceToDepth", 1);
}

}  // namespace

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Float_DCR) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatDCR",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/false);
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Float_CRD) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatCRD",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/false);
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionQDQ",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/true);
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_UnequalBlockSize) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlock",
                            /*input_shape=*/{1, 2, 4, 6},
                            /*block_height=*/2,
                            /*block_width=*/3,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/false);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
