// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

// Declared in test_main.cc.
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

template <typename QuantType = uint8_t>
GetTestModelFn BuildSpaceToDepthTestCase(const std::vector<int64_t>& input_shape,
                                         int64_t block_height,
                                         int64_t block_width,
                                         const std::vector<int64_t>& perm,
                                         bool use_qdq,
                                         bool use_contrib_qdq,
                                         const std::vector<int64_t>& reshape1_shape = {},
                                         const std::vector<int64_t>& reshape2_shape = {}) {
  return [=](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("spacetodepth_fusion_graph");

    const auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // Add layout-sensitive Conv around RTR so wrapped fusion can trigger.
    const int64_t c = input_shape[1];
    const std::vector<int64_t> conv1_weight_shape = {c, c, 1, 1};
    builder.MakeInitializer<float>("conv1_weight", conv1_weight_shape, -2.f, 2.f);
    builder.AddNode("Conv1",
                    "Conv",
                    {"input", "conv1_weight"},
                    {"conv1_out"},
                    kOnnxDomain);

    std::string reshape1_input = "conv1_out";
    if (use_qdq) {
      const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
      reshape1_input = AddQDQNodePair<QuantType>(builder, "qdq_in", "conv1_out",
                                                 input_qparams.scale, input_qparams.zero_point,
                                                 use_contrib_qdq);
    }

    const int64_t n = input_shape[0];
    const int64_t h = input_shape[2];
    const int64_t w = input_shape[3];
    const int64_t h_div = h / block_height;
    const int64_t w_div = w / block_width;

    // reshape1_shape / reshape2_shape may each carry a single -1 that ONNX shape inference
    // resolves from the concrete input. Empty means "fully concrete".
    const std::vector<int64_t> reshape1_dims =
        reshape1_shape.empty()
            ? std::vector<int64_t>{n, c, h_div, block_height, w_div, block_width}
            : reshape1_shape;

    // Reshape1: NCHW -> [N, C, H/block_h, block_h, W/block_w, block_w]
    builder.Make1DInitializer<int64_t>("reshape1_shape", reshape1_dims);
    builder.AddNode("Reshape1",
                    "Reshape",
                    {reshape1_input, "reshape1_shape"},
                    {"reshape1_out"},
                    kOnnxDomain);

    std::string transpose_input = "reshape1_out";
    if (use_qdq) {
      const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
      transpose_input = AddQDQNodePair<QuantType>(builder, "qdq_after_reshape1", "reshape1_out",
                                                  input_qparams.scale, input_qparams.zero_point,
                                                  use_contrib_qdq);
    }

    builder.AddNode("Transpose",
                    "Transpose",
                    {transpose_input},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", perm)});

    // Reshape2: rank-6 -> [N, C*block_h*block_w, H/block_h, W/block_w]
    const std::vector<int64_t> reshape2_dims =
        reshape2_shape.empty()
            ? std::vector<int64_t>{n, c * block_height * block_width, h_div, w_div}
            : reshape2_shape;
    builder.Make1DInitializer<int64_t>("reshape2_shape", reshape2_dims);
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape"},
                    {"reshape2_out"},
                    kOnnxDomain);

    const int64_t out_c = c * block_height * block_width;
    const std::vector<int64_t> conv2_weight_shape = {out_c, out_c, 1, 1};
    builder.MakeInitializer<float>("conv2_weight", conv2_weight_shape, -2.f, 2.f);
    builder.AddNode("Conv2",
                    "Conv",
                    {"reshape2_out", "conv2_weight"},
                    {"Y"},
                    kOnnxDomain);

    builder.MakeOutput("Y");
  };
}

template <typename QuantType = uint8_t>
GetTestModelFn BuildHeadWrappedSpaceToDepthTestCase(bool use_qdq,
                                                    bool use_contrib_qdq) {
  return [=](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("spacetodepth_head_wrapped_fusion_graph");

    const int64_t block_height = 2;
    const int64_t block_width = 2;
    const std::vector<int64_t> input_shape{1, 6, 8, 8};
    const auto input_def = TestInputDef<float>(input_shape, false, -0.5f, 0.5f);
    MakeTestInput<float>(builder, "input", input_def);

    std::string conv1_input = "input";
    if (use_qdq) {
      const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
      conv1_input = AddQDQNodePair<QuantType>(builder, "qdq_in_head", "input",
                                              input_qparams.scale, input_qparams.zero_point,
                                              use_contrib_qdq);
    }

    const std::vector<int64_t> conv1_weight_shape = {12, 6, 1, 1};
    builder.MakeInitializer<float>("conv1_weight", conv1_weight_shape, -2.f, 2.f);
    builder.AddNode("Conv1",
                    "Conv",
                    {conv1_input, "conv1_weight"},
                    {"conv1_out"},
                    kOnnxDomain);

    const int64_t n = 1;
    const int64_t c = 12;
    const int64_t h = 8;
    const int64_t w = 8;
    const int64_t h_div = h / block_height;
    const int64_t w_div = w / block_width;

    builder.Make1DInitializer<int64_t>("reshape1_shape_head", {n, c, h_div, block_height, w_div, block_width});
    builder.AddNode("Reshape1",
                    "Reshape",
                    {"conv1_out", "reshape1_shape_head"},
                    {"reshape1_out"},
                    kOnnxDomain);

    builder.AddNode("TransposeCore",
                    "Transpose",
                    {"reshape1_out"},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", std::vector<int64_t>{0, 1, 3, 5, 2, 4})});

    builder.Make1DInitializer<int64_t>("reshape2_shape_head", {n, c * block_height * block_width, h_div, w_div});
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape_head"},
                    {"Y"},
                    kOnnxDomain);

    builder.MakeOutput("Y");
  };
}

template <typename QuantType = uint8_t>
GetTestModelFn BuildTailWrappedSpaceToDepthTestCase(bool use_qdq,
                                                    bool use_contrib_qdq) {
  return [=](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("spacetodepth_tail_wrapped_fusion_graph");

    const int64_t block_height = 2;
    const int64_t block_width = 2;
    const std::vector<int64_t> input_shape{1, 12, 8, 8};
    const auto input_def = TestInputDef<float>(input_shape, false, -0.5f, 0.5f);
    MakeTestInput<float>(builder, "input", input_def);

    std::string reshape1_input = "input";
    if (use_qdq) {
      const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
      reshape1_input = AddQDQNodePair<QuantType>(builder, "qdq_in_tail", "input",
                                                 input_qparams.scale, input_qparams.zero_point,
                                                 use_contrib_qdq);
    }

    const int64_t n = input_shape[0];
    const int64_t c = input_shape[1];
    const int64_t h = input_shape[2];
    const int64_t w = input_shape[3];
    const int64_t h_div = h / block_height;
    const int64_t w_div = w / block_width;

    builder.Make1DInitializer<int64_t>("reshape1_shape_tail", {n, c, h_div, block_height, w_div, block_width});
    builder.AddNode("Reshape1",
                    "Reshape",
                    {reshape1_input, "reshape1_shape_tail"},
                    {"reshape1_out"},
                    kOnnxDomain);

    std::string transpose_input = "reshape1_out";
    if (use_qdq) {
      const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
      transpose_input = AddQDQNodePair<QuantType>(builder, "qdq_after_reshape1_tail", "reshape1_out",
                                                  input_qparams.scale, input_qparams.zero_point,
                                                  use_contrib_qdq);
    }

    builder.AddNode("TransposeCore",
                    "Transpose",
                    {transpose_input},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", std::vector<int64_t>{0, 1, 3, 5, 2, 4})});

    builder.Make1DInitializer<int64_t>("reshape2_shape_tail", {n, c * block_height * block_width, h_div, w_div});
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape_tail"},
                    {"reshape2_out"},
                    kOnnxDomain);

    std::string conv_input = "reshape2_out";
    if (use_qdq) {
      const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
      conv_input = AddQDQNodePair<QuantType>(builder, "qdq_after_reshape2_tail", "reshape2_out",
                                             input_qparams.scale, input_qparams.zero_point,
                                             use_contrib_qdq);
    }

    const std::vector<int64_t> conv2_weight_shape = {c * block_height * block_width, 1, 3, 1};
    builder.MakeInitializer<float>("conv2_weight_tail", conv2_weight_shape, -2.f, 2.f);

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(test::MakeAttribute("group", static_cast<int64_t>(c * block_height * block_width)));
    attrs.push_back(test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 1}));
    builder.AddNode("Conv2",
                    "Conv",
                    {conv_input, "conv2_weight_tail"},
                    {"Y"},
                    kOnnxDomain,
                    attrs);
    builder.MakeOutput("Y");
  };
}

// Bare R->T->R chain, optionally plus an unrelated 1x1 Conv on the graph input.
// The side Conv is a layout-sensitive op that forces ORT to issue the 2nd
// GetCapability pass; the RTR itself is still "bare" for fusion purposes.
// Pass add_side_conv=false to build a truly-bare graph for tripwire coverage.
GetTestModelFn BuildBareRTRSpaceToDepthTestCase(const std::vector<int64_t>& input_shape,
                                                int64_t block_height,
                                                int64_t block_width,
                                                const std::vector<int64_t>& perm,
                                                bool add_side_conv = true) {
  return [=](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("spacetodepth_bare_rtr_graph");

    const auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    const int64_t n = input_shape[0];
    const int64_t c = input_shape[1];
    const int64_t h = input_shape[2];
    const int64_t w = input_shape[3];
    const int64_t h_div = h / block_height;
    const int64_t w_div = w / block_width;

    // Reshape1: [N, C, H, W] -> [N, C, H/bh, bh, W/bw, bw]
    builder.Make1DInitializer<int64_t>("reshape1_shape", {n, c, h_div, block_height, w_div, block_width});
    builder.AddNode("Reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);

    // Transpose: 6D permutation (CRD or DCR)
    builder.AddNode("Transpose", "Transpose", {"reshape1_out"}, {"transpose_out"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", perm)});

    // Reshape2: -> [N, C*bh*bw, H/bh, W/bw]
    builder.Make1DInitializer<int64_t>("reshape2_shape", {n, c * block_height * block_width, h_div, w_div});
    builder.AddNode("Reshape2", "Reshape", {"transpose_out", "reshape2_shape"}, {"output"}, kOnnxDomain);

    builder.MakeOutput("output");

    if (add_side_conv) {
      // Unrelated side-branch 1x1 Conv on the same input — layout-sensitive op that
      // forces ORT to issue the 2nd GetCapability pass. Not adjacent to the RTR.
      const std::vector<int64_t> side_conv_weight_shape = {c, c, 1, 1};
      builder.MakeInitializer<float>("side_conv_weight", side_conv_weight_shape, -1.0f, 1.0f);
      builder.AddNode("SideConv", "Conv", {"input", "side_conv_weight"}, {"side_output"}, kOnnxDomain);
      builder.MakeOutput("side_output");
    }
  };
}

ProviderOptions GetProviderOptions(const std::string& backend_type) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_type;
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

template <typename QuantType = uint8_t>
void RunSpaceToDepthFusionTest(const std::filesystem::path& json_qnn_graph_dir,
                               const std::vector<int64_t>& input_shape,
                               int64_t block_height,
                               int64_t block_width,
                               const std::vector<int64_t>& perm,
                               bool use_qdq,
                               bool use_contrib_qdq,
                               const std::string& backend_type,
                               float fp32_abs_err = 1e-2f,
                               const std::vector<int64_t>& reshape1_shape = {},
                               const std::vector<int64_t>& reshape2_shape = {}) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  const int uncaught_on_entry = std::uncaught_exceptions();
  auto cleanup = gsl::finally([uncaught_on_entry, json_qnn_graph_dir]() {
    if (std::uncaught_exceptions() > uncaught_on_entry) {
      return;
    }
    std::filesystem::remove_all(json_qnn_graph_dir);
  });

  ProviderOptions provider_options = GetProviderOptions(backend_type);
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildSpaceToDepthTestCase<QuantType>(input_shape, block_height, block_width, perm,
                                                       use_qdq, use_contrib_qdq, reshape1_shape, reshape2_shape),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(fp32_abs_err)},
                  /*log_severity=*/OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);

  AssertOpInQnnGraph(json_qnn_graph_dir, "SpaceToDepth", 1);
}

void RunWrappedPatternSpaceToDepthFusionTest(const std::filesystem::path& json_qnn_graph_dir,
                                             GetTestModelFn model_builder,
                                             const std::string& backend_type,
                                             float fp32_abs_err = 1e-2f) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  const int uncaught_on_entry = std::uncaught_exceptions();
  auto cleanup = gsl::finally([uncaught_on_entry, json_qnn_graph_dir]() {
    if (std::uncaught_exceptions() > uncaught_on_entry) {
      return;
    }
    std::filesystem::remove_all(json_qnn_graph_dir);
  });

  ProviderOptions provider_options = GetProviderOptions(backend_type);
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(model_builder,
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(fp32_abs_err)},
                  /*log_severity=*/OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);

  AssertOpInQnnGraph(json_qnn_graph_dir, "SpaceToDepth", 1);
}

}  // namespace

// Disabling this test as Layout Transformer for CPU BE breaks the pattern by modifying Transpose perm.
// Test Graph : Input -> Conv -> Reshape -> Transpose(perm={0,3,5,1,2,4}) -> Reshape -> Conv -> Output
// Layout Transformer modifies Transpose perm to {0,1,3,5,2,4} to {0,5,2,4,1,3} to save one Transpose op.
// LT Graph: T(NCHW->NHWC) -> Conv -> Reshape -> Transpose(perm={0,5,2,4,1,3}) -> Reshape -> T(NCHW->NHWC) Conv -> Output
TEST_F(QnnCPUBackendTests, DISABLED_SpaceToDepthFusion_Float_CRD) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatCRD_CPU",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"cpu");
}

// Disabling this test as Layout Transformer for CPU BE breaks the pattern by modifying Transpose perm, as explained above.
TEST_F(QnnCPUBackendTests, DISABLED_SpaceToDepthFusion_Float_DCR) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatDCR_CPU",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"cpu");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Wrapped4Node_Head_Float_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunWrappedPatternSpaceToDepthFusionTest("SpaceToDepthFusionWrapped4NodeHeadFloatCRD_HTP",
                                          BuildHeadWrappedSpaceToDepthTestCase<>(/*use_qdq=*/false,
                                                                                 /*use_contrib_qdq=*/false),
                                          /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Wrapped4Node_Head_QDQ_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunWrappedPatternSpaceToDepthFusionTest("SpaceToDepthFusionWrapped4NodeHeadQDQCRD_HTP",
                                          BuildHeadWrappedSpaceToDepthTestCase<>(/*use_qdq=*/true,
                                                                                 /*use_contrib_qdq=*/false),
                                          /*backend_type=*/"htp",
                                          /*fp32_abs_err=*/1.5e-2f);
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Wrapped4Node_Tail_Float_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunWrappedPatternSpaceToDepthFusionTest("SpaceToDepthFusionWrapped4NodeTailFloatCRD_HTP",
                                          BuildTailWrappedSpaceToDepthTestCase<>(/*use_qdq=*/false,
                                                                                 /*use_contrib_qdq=*/false),
                                          /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Wrapped4Node_Tail_QDQ_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunWrappedPatternSpaceToDepthFusionTest("SpaceToDepthFusionWrapped4NodeTailQDQCRD_HTP",
                                          BuildTailWrappedSpaceToDepthTestCase<>(/*use_qdq=*/true,
                                                                                 /*use_contrib_qdq=*/false),
                                          /*backend_type=*/"htp");
}

// Fails with QNN CPU graph execution failure.
// * Tracking issue: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
TEST_F(QnnCPUBackendTests, DISABLED_SpaceToDepthFusion_Float_UnequalBlockSize) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlock_CPU",
                            /*input_shape=*/{1, 2, 4, 6},
                            /*block_height=*/2,
                            /*block_width=*/3,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"cpu");
}

// Fails with QNN CPU graph execution failure.
// * Tracking issue: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
TEST_F(QnnCPUBackendTests, DISABLED_SpaceToDepthFusion_Float_UnequalBlockSize_CRD) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlockCRD_CPU",
                            /*input_shape=*/{1, 2, 4, 6},
                            /*block_height=*/2,
                            /*block_width=*/3,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"cpu");
}

// Fails with Accuracy mismatch
// * Tracking issue: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
TEST_F(QnnHTPBackendTests, DISABLED_SpaceToDepthFusion_Float_DCR) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatDCR",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_DCR) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatDCRQDQ",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/3.9e-2f);
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Float_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatCRD",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionQDQ_CRD",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/2.9e-2);
}

// Fails with Accuracy mismatch
// * Tracking issue: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
TEST_F(QnnHTPBackendTests, DISABLED_SpaceToDepthFusion_UnequalBlockSize_DCR) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlock",
                            /*input_shape=*/{1, 2, 4, 6},
                            /*block_height=*/2,
                            /*block_width=*/3,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_UnequalBlockSize_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlockCRD",
                            /*input_shape=*/{1, 2, 4, 6},
                            /*block_height=*/2,
                            /*block_width=*/3,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_UnequalBlockSize_QDQ) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlockQDQ",
                            /*input_shape=*/{1, 2, 4, 6},
                            /*block_height=*/2,
                            /*block_width=*/3,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/2.9e-2f);
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_UnequalBlockSize_QDQ_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlockQDQ_CRD",
                            /*input_shape=*/{1, 2, 4, 6},
                            /*block_height=*/2,
                            /*block_width=*/3,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/4.9e-2f);
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_U16_DCR) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest<uint16_t>("SpaceToDepthFusionQDQ_U16_DCR",
                                      /*input_shape=*/{1, 2, 4, 4},
                                      /*block_height=*/2,
                                      /*block_width=*/2,
                                      /*perm=*/{0, 3, 5, 1, 2, 4},
                                      /*use_qdq=*/true,
                                      /*use_contrib_qdq=*/true,
                                      /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_U16_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest<uint16_t>("SpaceToDepthFusionQDQ_U16_CRD",
                                      /*input_shape=*/{1, 2, 4, 4},
                                      /*block_height=*/2,
                                      /*block_width=*/2,
                                      /*perm=*/{0, 1, 3, 5, 2, 4},
                                      /*use_qdq=*/true,
                                      /*use_contrib_qdq=*/true,
                                      /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_UnequalBlockSize_QDQ_U16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest<uint16_t>("SpaceToDepthFusionUnequalBlockQDQ_U16",
                                      /*input_shape=*/{1, 2, 4, 6},
                                      /*block_height=*/2,
                                      /*block_width=*/3,
                                      /*perm=*/{0, 3, 5, 1, 2, 4},
                                      /*use_qdq=*/true,
                                      /*use_contrib_qdq=*/true,
                                      /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_UnequalBlockSize_QDQ_U16_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest<uint16_t>("SpaceToDepthFusionUnequalBlockQDQ_U16_CRD",
                                      /*input_shape=*/{1, 2, 4, 6},
                                      /*block_height=*/2,
                                      /*block_width=*/3,
                                      /*perm=*/{0, 1, 3, 5, 2, 4},
                                      /*use_qdq=*/true,
                                      /*use_contrib_qdq=*/true,
                                      /*backend_type=*/"htp");
}

// Regression: HasSpaceToDepthCoreSignature was rejecting -1 (ONNX placeholder marker) in
// the Reshape shape initializer. Shape inference resolves it from the concrete input.
TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Float_CRD_Reshape1BatchPlaceholder) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatCRD_Reshape1BatchPlaceholder",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/1e-2f,
                            /*reshape1_shape=*/{-1, 2, 2, 2, 2, 2});
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_CRD_Reshape1BatchPlaceholder) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionQDQ_CRD_Reshape1BatchPlaceholder",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/2.9e-2f,
                            /*reshape1_shape=*/{-1, 2, 2, 2, 2, 2});
}

// Regression: -1 outside the batch dim. PyTorch exports SpaceToDepth as
// reshape(N, -1, H/b, b, W/b, b) — channel is the placeholder, resolved by shape inference.
TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Float_CRD_Reshape1ChannelPlaceholder) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatCRD_Reshape1ChannelPlaceholder",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/1e-2f,
                            /*reshape1_shape=*/{1, -1, 2, 2, 2, 2});
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_CRD_Reshape1ChannelPlaceholder) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionQDQ_CRD_Reshape1ChannelPlaceholder",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/2.9e-2f,
                            /*reshape1_shape=*/{1, -1, 2, 2, 2, 2});
}

// DCR counterpart of the channel-placeholder test. QDQ only: HTP has no accurate float DCR
// SpaceToDepth kernel (the suite carries no Float DCR case for the same reason).
TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_DCR_Reshape1ChannelPlaceholder) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionQDQ_DCR_Reshape1ChannelPlaceholder",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/3.9e-2f,
                            /*reshape1_shape=*/{1, -1, 2, 2, 2, 2});
}

// -1 at the H/block_h dim: confirms the gate accepts a single -1 at any position, not just 0/1.
TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_CRD_Reshape1HeightPlaceholder) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionQDQ_CRD_Reshape1HeightPlaceholder",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/2.9e-2f,
                            /*reshape1_shape=*/{1, 2, -1, 2, 2, 2});
}

// -1 in the second (rank-4) Reshape initializer: the gate reads resolved ValueInfo for both
// reshapes, so reshape2's -1 is covered too. Guards that path explicitly.
TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_CRD_Reshape2ChannelPlaceholder) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunSpaceToDepthFusionTest("SpaceToDepthFusionQDQ_CRD_Reshape2ChannelPlaceholder",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp",
                            /*fp32_abs_err=*/2.9e-2f,
                            /*reshape1_shape=*/{},
                            /*reshape2_shape=*/{1, -1, 2, 2});
}

// RTR-only CRD on HTP backend.
TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_BareRTR_Float_CRD) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "SpaceToDepthFusion_BareRTR_Float_CRD_HTP";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions("htp");
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildBareRTRSpaceToDepthTestCase({1, 3, 4, 4}, 2, 2, {0, 1, 3, 5, 2, 4}),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "SpaceToDepth", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Conv2d", 1);
  // 4 = side-Conv NCHW<->NHWC pair (LT) + fused S2D NCHW<->NHWC pre/post pair.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 4);
}

// Tripwire: truly-bare RTR (no side layout-sensitive op) is currently unreachable
// for SpaceToDepthFusion. Pass 1 sees rank-6 R/T/R as unsupported per-node and
// returns empty capabilities; ORT elides Layout Transformer and never issues
// pass 2, so the post-LT-gated fusion never fires and QNN takes nothing. When
// ORT ever changes this (unconditional pass 2, or HTP support for rank-6 R/T/R),
// Assignment::None will fail — forcing us to update the fusion gate or delete
// this tripwire.
TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_TrulyBareRTR_Float_CRD_NotFused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  RunQnnModelTest(BuildBareRTRSpaceToDepthTestCase({1, 3, 4, 4}, 2, 2, {0, 1, 3, 5, 2, 4},
                                                   /*add_side_conv=*/false),
                  GetProviderOptions("htp"),
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// Bare-RTR DCR structural coverage; disabled — HTP's SpaceToDepth DCR kernel mismatches
// element-wise (same as pre-existing DISABLED_SpaceToDepthFusion_Float_DCR).
// Tracking: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
TEST_F(QnnHTPBackendTests, DISABLED_SpaceToDepthFusion_BareRTR_Float_DCR) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "SpaceToDepthFusion_BareRTR_Float_DCR_HTP";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions("htp");
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildBareRTRSpaceToDepthTestCase({1, 3, 4, 4}, 2, 2, {0, 3, 5, 1, 2, 4}),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "SpaceToDepth", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Conv2d", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Transpose", 4);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
