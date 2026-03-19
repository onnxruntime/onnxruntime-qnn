// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

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

constexpr std::string_view kDlcOutputDir("dlc_output");

enum class TestBackend {
  Htp,
  Ir,
};

static std::string ToBackendLibName(TestBackend backend) {
  switch (backend) {
    case TestBackend::Htp:
      return "Htp";
    case TestBackend::Ir:
      return "Ir";
  }

  return "";
}

static void AddSerializerConfigs(TestBackend serializer_backend, onnxruntime::ProviderOptions& options) {
  std::string serializer_lib = ToBackendLibName(serializer_backend);
  std::string serializer_path_key;

  switch (serializer_backend) {
    case TestBackend::Htp:
      FAIL() << "Unsupported serializer backend for DLC dump test: Htp";
      return;
    case TestBackend::Ir:
      serializer_path_key = "qnn_ir_backend_path";
      options["dump_qnn_ir_dlc"] = "1";
      options["dump_qnn_ir_dlc_dir"] = std::string{kDlcOutputDir};
      break;
  }

#if defined(_WIN32)
  options[serializer_path_key] = "Qnn" + serializer_lib + ".dll";
#else
  options[serializer_path_key] = "libQnn" + serializer_lib + ".so";
#endif
}

static Ort::Session InitNHWCResizeModel(const ORTCHAR_T* ort_model_path,
                                        TestBackend backend,
                                        RegisteredEpDeviceUniquePtr& registered_ep_device,
                                        std::optional<TestBackend> serializer_backend = std::nullopt) {
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionsConfigStrictShapeTypeInference, "1");
  so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

  onnxruntime::ProviderOptions options;
  options["offload_graph_io_quantization"] = "0";

  std::string backend_lib = ToBackendLibName(backend);

#if defined(_WIN32)
  options["backend_path"] = "Qnn" + backend_lib + ".dll";
#else
  options["backend_path"] = "libQnn" + backend_lib + ".so";
#endif

  if (serializer_backend) {
    AddSerializerConfigs(*serializer_backend, options);
  }

  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);
  return Ort::Session(*::ort_env, ort_model_path, so);
}

template <typename QuantType = uint8_t>
GetTestModelFn BuildSpaceToDepthTestCase(const std::vector<int64_t>& input_shape,
                                         int64_t block_height,
                                         int64_t block_width,
                                         const std::vector<int64_t>& perm,
                                         bool use_qdq,
                                         bool use_contrib_qdq) {
  return [=](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("spacetodepth_fusion_graph");

    const auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    std::string reshape1_input = "input";
    if (use_qdq) {
      const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
      reshape1_input = AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                                 input_qparams.scale, input_qparams.zero_point,
                                                 use_contrib_qdq);
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
    builder.Make1DInitializer<int64_t>("reshape2_shape", {n, c * block_height * block_width, h_div, w_div});
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape"},
                    {"reshape2_out"},
                    kOnnxDomain);

    builder.MakeOutput("reshape2_out");
  };
}

GetTestModelFn BuildWrappedSpaceToDepthTestCase() {
  return [=](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("spacetodepth_wrapped_fusion_graph");

    // Match ChannelShuffle test topology: Conv -> RTR -> Conv.
    const int64_t num_channels = 12;
    const int64_t block_height = 2;
    const int64_t block_width = 2;
    const std::vector<int64_t> input_shape{1, num_channels, 8, 8};
    const auto input_def = TestInputDef<float>(input_shape, false, -0.5f, 0.5f);
    MakeTestInput<float>(builder, "input", input_def);

    // Conv1 weights
    const std::vector<int64_t> conv1_weight_shape = {num_channels, num_channels / 2, 1, 1};
    builder.MakeInitializer<float>("conv1_weight", conv1_weight_shape, -2.f, 2.f);

    // Conv1: input + conv1_weight -> conv1_out
    {
      std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
      attrs.push_back(test::MakeAttribute("group", static_cast<int64_t>(2)));
      builder.AddNode("Conv1",
                      "Conv",
                      {"input", "conv1_weight"},
                      {"conv1_out"},
                      kOnnxDomain,
                      attrs);
    }

    const int64_t n = input_shape[0];
    const int64_t c = input_shape[1];
    const int64_t h = input_shape[2];
    const int64_t w = input_shape[3];
    const int64_t h_div = h / block_height;
    const int64_t w_div = w / block_width;

    // RTR for SpaceToDepth CRD decomposition.
    builder.Make1DInitializer<int64_t>("reshape1_shape_wrapped", {n, c, h_div, block_height, w_div, block_width});
    builder.AddNode("Reshape1",
                    "Reshape",
                    {"conv1_out", "reshape1_shape_wrapped"},
                    {"reshape1_out"},
                    kOnnxDomain);

    builder.AddNode("TransposeCore",
                    "Transpose",
                    {"reshape1_out"},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", std::vector<int64_t>{0, 1, 3, 5, 2, 4})});

    builder.Make1DInitializer<int64_t>("reshape2_shape_wrapped", {n, c * block_height * block_width, h_div, w_div});
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape_wrapped"},
                    {"reshape2_out"},
                    kOnnxDomain);

    // Conv2 weights
    const std::vector<int64_t> conv2_weight_shape = {c * block_height * block_width, 1, 3, 1};
    builder.MakeInitializer<float>("conv2_weight", conv2_weight_shape, -2.f, 2.f);

    // Conv2: reshape2_out + conv2_weight -> Y
    {
      std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
      attrs.push_back(test::MakeAttribute("group", static_cast<int64_t>(c * block_height * block_width)));
      attrs.push_back(test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 1}));
      builder.MakeOutput("Y");
      builder.AddNode("Conv2",
                      "Conv",
                      {"reshape2_out", "conv2_weight"},
                      {"Y"},
                      kOnnxDomain,
                      attrs);
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
                               const std::string& backend_type) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  const int uncaught_on_entry = std::uncaught_exceptions();
  auto cleanup = gsl::finally([&json_qnn_graph_dir, uncaught_on_entry]() {
    if (std::uncaught_exceptions() > uncaught_on_entry) {
      return;
    }
    // std::filesystem::remove_all(json_qnn_graph_dir);
  });

  ProviderOptions provider_options = GetProviderOptions(backend_type);
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildSpaceToDepthTestCase<QuantType>(input_shape, block_height, block_width, perm, use_qdq, use_contrib_qdq),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f,
                  /*log_severity=*/OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);

  AssertOpInQnnGraph(json_qnn_graph_dir, "SpaceToDepth", 1);
}

void RunWrappedSpaceToDepthFusionTest(const std::filesystem::path& json_qnn_graph_dir,
                                      const std::string& backend_type) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  const int uncaught_on_entry = std::uncaught_exceptions();
  auto cleanup = gsl::finally([&json_qnn_graph_dir, uncaught_on_entry]() {
    if (std::uncaught_exceptions() > uncaught_on_entry) {
      return;
    }
    // std::filesystem::remove_all(json_qnn_graph_dir);
  });

  ProviderOptions provider_options = GetProviderOptions(backend_type);
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildWrappedSpaceToDepthTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f,
                  /*log_severity=*/OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);

  // 1) Ensure SpaceToDepth is materialized exactly once.
  AssertOpInQnnGraph(json_qnn_graph_dir, "SpaceToDepth", 1);

  // 2) Ensure original RTR decomposition nodes are gone.
  AssertNodeNotInQnnGraph(json_qnn_graph_dir, "Reshape1");
  AssertNodeNotInQnnGraph(json_qnn_graph_dir, "TransposeCore");
  AssertNodeNotInQnnGraph(json_qnn_graph_dir, "Reshape2");
}

}  // namespace

TEST_F(QnnCPUBackendTests, SpaceToDepthFusion_Float_DCR) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatDCR_CPU",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"cpu");
}

TEST_F(QnnCPUBackendTests, SpaceToDepthFusion_Float_CRD) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatCRD_CPU",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/false,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"cpu");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Wrapped5Node_Float_CRD) {
  RunWrappedSpaceToDepthFusionTest("SpaceToDepthFusionWrapped5NodeFloatCRD_HTP",
                                   /*backend_type=*/"htp");
}

// Fails with QNN CPU graph execution failure.
// * Tracking issue: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
// TEST_F(QnnCPUBackendTests, SpaceToDepthFusion_Float_UnequalBlockSize) {
//   RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlock_CPU",
//                             /*input_shape=*/{1, 2, 4, 6},
//                             /*block_height=*/2,
//                             /*block_width=*/3,
//                             /*perm=*/{0, 3, 5, 1, 2, 4},
//                             /*use_qdq=*/false,
//                             /*use_contrib_qdq=*/false,
//                             /*backend_type=*/"cpu");
// }

// Fails with QNN CPU graph execution failure.
// * Tracking issue: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
// TEST_F(QnnCPUBackendTests, SpaceToDepthFusion_Float_UnequalBlockSize_CRD) {
//   RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlockCRD_CPU",
//                             /*input_shape=*/{1, 2, 4, 6},
//                             /*block_height=*/2,
//                             /*block_width=*/3,
//                             /*perm=*/{0, 1, 3, 5, 2, 4},
//                             /*use_qdq=*/false,
//                             /*use_contrib_qdq=*/false,
//                             /*backend_type=*/"cpu");
// }

// Fails with Accuracy mismatch
// * Tracking issue: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
// TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Float_DCR) {
//   RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatDCR",
//                             /*input_shape=*/{1, 2, 4, 4},
//                             /*block_height=*/2,
//                             /*block_width=*/2,
//                             /*perm=*/{0, 3, 5, 1, 2, 4},
//                             /*use_qdq=*/false,
//                             /*use_contrib_qdq=*/false,
//                             /*backend_type=*/"htp");
// }

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_DCR) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionFloatDCRQDQ",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_Float_CRD) {
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
  RunSpaceToDepthFusionTest("SpaceToDepthFusionQDQ_CRD",
                            /*input_shape=*/{1, 2, 4, 4},
                            /*block_height=*/2,
                            /*block_width=*/2,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp");
}

// Fails with Accuracy mismatch
// * Tracking issue: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
// TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_UnequalBlockSize_DCR) {
//   RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlock",
//                             /*input_shape=*/{1, 2, 4, 6},
//                             /*block_height=*/2,
//                             /*block_width=*/3,
//                             /*perm=*/{0, 3, 5, 1, 2, 4},
//                             /*use_qdq=*/false,
//                             /*use_contrib_qdq=*/false,
//                             /*backend_type=*/"htp");
// }

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_UnequalBlockSize_CRD) {
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
  RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlockQDQ",
                            /*input_shape=*/{1, 2, 4, 6},
                            /*block_height=*/2,
                            /*block_width=*/3,
                            /*perm=*/{0, 3, 5, 1, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_UnequalBlockSize_QDQ_CRD) {
  RunSpaceToDepthFusionTest("SpaceToDepthFusionUnequalBlockQDQ_CRD",
                            /*input_shape=*/{1, 2, 4, 6},
                            /*block_height=*/2,
                            /*block_width=*/3,
                            /*perm=*/{0, 1, 3, 5, 2, 4},
                            /*use_qdq=*/true,
                            /*use_contrib_qdq=*/false,
                            /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, SpaceToDepthFusion_QDQ_U16_DCR) {
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
  RunSpaceToDepthFusionTest<uint16_t>("SpaceToDepthFusionUnequalBlockQDQ_U16_CRD",
                                      /*input_shape=*/{1, 2, 4, 6},
                                      /*block_height=*/2,
                                      /*block_width=*/3,
                                      /*perm=*/{0, 1, 3, 5, 2, 4},
                                      /*use_qdq=*/true,
                                      /*use_contrib_qdq=*/true,
                                      /*backend_type=*/"htp");
}

TEST_F(QnnHTPBackendTests, TempDumpDlcTest) {
  if (IsIRBackendSupported() == BackendSupport::UNSUPPORTED) {
    GTEST_SKIP() << "QNN IR backend is not available.";
  } else if (IsIRBackendSupported() == BackendSupport::SUPPORT_ERROR) {
    FAIL() << "Failed to check if QNN IR backend is available.";
  }

  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::filesystem::path qnn_dlc_dir = kDlcOutputDir;

  // Remove pre-existing QNN IR output files. remove_all handles non-existing paths.
  std::filesystem::remove_all(qnn_dlc_dir);
  ASSERT_FALSE(std::filesystem::exists(qnn_dlc_dir));

  InitNHWCResizeModel("sr_sim.onnx",
                      TestBackend::Htp,
                      registered_ep_device,
                      TestBackend::Ir);

  ASSERT_TRUE(std::filesystem::exists(qnn_dlc_dir));
  int file_count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(qnn_dlc_dir)) {
    EXPECT_TRUE(entry.is_regular_file());
    EXPECT_EQ(entry.path().extension(), ".dlc");
    ++file_count;
  }

  EXPECT_EQ(file_count, 1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
