// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Unit tests for QnnModel (qnn_model.cc).

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "HTP/QnnHtpDevice.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/unit/qnn_fake_ort_graph.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// QnnModelMinimalTestContext
//
// Minimal context: QnnBackendManager::Create only (no SetupBackend, no real
// QNN backend library loaded). Sufficient for tests that exercise QnnModel
// methods which do not touch the QNN SDK (e.g., DeserializeGraphInfoFromBinaryInfo
// unsupported-version path, GetNodeUnit).
// ---------------------------------------------------------------------------
struct QnnModelMinimalTestContext {
  Ort::Logger logger = MakeNullLogger();
  OrtApi stub_ort_api{};
  OrtEpApi stub_ep_api{};
  OrtModelEditorApi stub_editor_api{};
  std::shared_ptr<qnn::QnnBackendManager> manager;
  std::unique_ptr<qnn::QnnModel> model;

  QnnModelMinimalTestContext() {
    qnn::QnnBackendManagerConfig cfg;
    cfg.backend_path = "libQnnHtp.so";
    cfg.profiling_level_etw = qnn::ProfilingLevel::OFF;
    cfg.profiling_level = qnn::ProfilingLevel::OFF;
    cfg.context_priority = qnn::ContextPriority::NORMAL;
    cfg.device_id = 0;
    cfg.htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
    cfg.soc_model = 0;
    cfg.skip_qnn_version_check = true;

    ApiPtrs api_ptrs{stub_ort_api, stub_ep_api, stub_editor_api};
    manager = qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
    if (!manager) return;
    model = std::make_unique<qnn::QnnModel>(manager.get(), api_ptrs);
  }

  bool IsValid() const { return model != nullptr; }
};

// ---------------------------------------------------------------------------
// QnnModelHtpTestContext
//
// Full context for tests that need a real QNN HTP backend (ComposeGraph,
// FinalizeGraphs, etc.). Loads libQnnHtp.so and runs SetupBackend.
// On x86_64 the validation path works but execution requires HTP hardware.
// Tests should call GTEST_SKIP() when IsValid() returns false.
// ---------------------------------------------------------------------------
struct QnnModelHtpTestContext {
  Ort::Logger logger = MakeNullLogger();
  OrtApi stub_ort_api{};
  OrtEpApi stub_ep_api{};
  OrtModelEditorApi stub_editor_api{};
  bool valid = false;
  std::shared_ptr<qnn::QnnBackendManager> manager;
  std::unique_ptr<qnn::QnnModel> model;

  QnnModelHtpTestContext() {
    stub_ort_api.Graph_GetNumInitializers = [](const OrtGraph*, size_t* n) noexcept -> OrtStatus* {
      *n = 0;
      return nullptr;
    };
    stub_ort_api.Graph_GetInitializers = [](const OrtGraph*, const OrtValueInfo**, size_t) noexcept -> OrtStatus* {
      return nullptr;
    };

    qnn::QnnBackendManagerConfig cfg;
    cfg.backend_path = "libQnnHtp.so";
    cfg.profiling_level_etw = qnn::ProfilingLevel::OFF;
    cfg.profiling_level = qnn::ProfilingLevel::OFF;
    cfg.context_priority = qnn::ContextPriority::NORMAL;
    cfg.device_id = 0;
    cfg.htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
    cfg.soc_model = 0;
    cfg.skip_qnn_version_check = true;

    ApiPtrs api_ptrs{stub_ort_api, stub_ep_api, stub_editor_api};
    manager = qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
    if (!manager) return;

    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
    auto status = manager->SetupBackend(false, false, false, -1, false, nullptr, dummy_map);
    if (!status.IsOK()) return;

    model = std::make_unique<qnn::QnnModel>(manager.get(), api_ptrs);
    valid = true;
  }

  bool IsValid() const { return valid; }
};

// ---------------------------------------------------------------------------
// DeserializeGraphInfoFromBinaryInfo tests
// ---------------------------------------------------------------------------

TEST(QnnUnit_ModelTest, Deserialize_UnsupportedVersion_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  QnnSystemContext_GraphInfo_t info{};
  // version = 0 is not a recognized QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_*.
  // → "Unsupported context graph info version." (line 863)
  info.version = static_cast<QnnSystemContext_GraphInfoVersion_t>(0);

  Qnn_ContextHandle_t fake_ctx = nullptr;
  EXPECT_FALSE(ctx.model->DeserializeGraphInfoFromBinaryInfo(info, fake_ctx).IsOK());
}

TEST(QnnUnit_ModelTest, Deserialize_Version1_NullInputTensors_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  QnnSystemContext_GraphInfo_t info{};
  info.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;
  info.graphInfoV1.graphName = const_cast<char*>("test_graph");
  info.graphInfoV1.numGraphInputs = 1;
  info.graphInfoV1.graphInputs = nullptr;  // null → error at line 865

  Qnn_ContextHandle_t fake_ctx = nullptr;
  EXPECT_FALSE(ctx.model->DeserializeGraphInfoFromBinaryInfo(info, fake_ctx).IsOK());
}

TEST(QnnUnit_ModelTest, Deserialize_Version1_NullOutputTensors_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  // graphInputs must be non-null to pass line 865; 0 iterations → safe with dummy ptr.
  static int g_dummy_buf = 0;
  QnnSystemContext_GraphInfo_t info{};
  info.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;
  info.graphInfoV1.graphName = const_cast<char*>("test_graph");
  info.graphInfoV1.numGraphInputs = 0;
  info.graphInfoV1.graphInputs = reinterpret_cast<Qnn_Tensor_t*>(&g_dummy_buf);
  info.graphInfoV1.numGraphOutputs = 1;
  info.graphInfoV1.graphOutputs = nullptr;  // null → error at line 866

  Qnn_ContextHandle_t fake_ctx = nullptr;
  EXPECT_FALSE(ctx.model->DeserializeGraphInfoFromBinaryInfo(info, fake_ctx).IsOK());
}

#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 18)
TEST(QnnUnit_ModelTest, Deserialize_Version2_NullInputTensors_ReturnsError) {
  // Covers the V2 conditional branch at lines 844-849.
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  QnnSystemContext_GraphInfo_t info{};
  info.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2;
  info.graphInfoV2.graphName = const_cast<char*>("test_graph");
  info.graphInfoV2.numGraphInputs = 1;
  info.graphInfoV2.graphInputs = nullptr;  // null → error at line 865

  Qnn_ContextHandle_t fake_ctx = nullptr;
  EXPECT_FALSE(ctx.model->DeserializeGraphInfoFromBinaryInfo(info, fake_ctx).IsOK());
}
#endif

#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 21)
TEST(QnnUnit_ModelTest, Deserialize_Version3_NullInputTensors_ReturnsError) {
  // Covers the V3 conditional branch at lines 853-858.
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  QnnSystemContext_GraphInfo_t info{};
  info.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3;
  info.graphInfoV3.graphName = const_cast<char*>("test_graph");
  info.graphInfoV3.numGraphInputs = 1;
  info.graphInfoV3.graphInputs = nullptr;  // null → error at line 865

  Qnn_ContextHandle_t fake_ctx = nullptr;
  EXPECT_FALSE(ctx.model->DeserializeGraphInfoFromBinaryInfo(info, fake_ctx).IsOK());
}
#endif

// ---------------------------------------------------------------------------
// SetGraphInputOutputInfo tests
//
// Exercises QnnModel::SetGraphInputOutputInfo with a fake OrtGraph + fake
// fused OrtNode. No real QNN backend needed — the function reads graph
// metadata via OrtApi stubs only.
// ---------------------------------------------------------------------------

TEST(QnnUnit_ModelTest, SetGraphInputOutputInfo_Basic_PopulatesInputsOutputs) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  // Make Ort::ConstNode / Ort::ConstValueInfo wrappers route through our stub
  // instead of the real ORT runtime, which would SIGSEGV on our fake pointers.
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  // graph: x -> Identity -> y, both float[4]
  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode inner{"identity_0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{inner}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};

  std::vector<std::string> in_names{"x"};
  std::vector<std::string> out_names{"y"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                          &in_names, &out_names, &settings};

  auto status = ctx.model->SetGraphInputOutputInfo(mc);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

// ---------------------------------------------------------------------------
// ComposeGraph early-validation error paths (no QNN backend required)
// ---------------------------------------------------------------------------

TEST(QnnUnit_ModelTest, ComposeGraph_NullInputNames_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeGraph graph{{}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};

  std::vector<std::string> out_names{"y"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                          /*onnx_input_names=*/nullptr, &out_names, &settings};

  EXPECT_FALSE(ctx.model->ComposeGraph(mc).IsOK());
}

TEST(QnnUnit_ModelTest, ComposeGraph_NullOutputNames_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeGraph graph{{}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};

  std::vector<std::string> in_names{"x"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                          &in_names, /*onnx_output_names=*/nullptr, &settings};

  EXPECT_FALSE(ctx.model->ComposeGraph(mc).IsOK());
}

TEST(QnnUnit_ModelTest, ComposeGraph_NullModelSettings_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeGraph graph{{}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};

  std::vector<std::string> in_names{"x"};
  std::vector<std::string> out_names{"y"};
  qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                          &in_names, &out_names, /*model_settings=*/nullptr};

  EXPECT_FALSE(ctx.model->ComposeGraph(mc).IsOK());
}

// ---------------------------------------------------------------------------
// ComposeGraph + LogTensorDetails (real HTP backend)
//
// These tests need a real QNN backend so that QnnModelWrapper::CreateQnnGraph
// and op-builder paths succeed. LogTensorDetails runs as part of ComposeGraph
// when json_qnn_graph_path is non-empty.
// ---------------------------------------------------------------------------

TEST(QnnUnit_ModelTest, ComposeGraph_IdentityGraph_WithJsonPath_TriggersLogTensorDetails) {
  QnnModelHtpTestContext ctx;
  if (!ctx.IsValid()) GTEST_SKIP() << "HTP backend not available on this host";
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode inner{"identity_0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{inner}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};

  std::string tmp_json = (std::filesystem::temp_directory_path() / "qnn_unit_model_test_graph.json").string();
  std::remove(tmp_json.c_str());

  std::vector<std::string> in_names{"x"};
  std::vector<std::string> out_names{"y"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                          &in_names, &out_names, &settings,
                          /*graph_configs=*/nullptr,
                          /*tensor_name_overrides=*/nullptr,
                          /*json_qnn_graph_path=*/tmp_json};

  auto status = ctx.model->ComposeGraph(mc);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // LogTensorDetails writes "<base>_tensor_log.json" alongside the graph json.
  std::string log_file = tmp_json.substr(0, tmp_json.rfind('.')) + "_tensor_log.json";
  EXPECT_TRUE(std::filesystem::exists(log_file)) << "Expected: " << log_file;

  std::remove(tmp_json.c_str());
  std::remove(log_file.c_str());
}

// Helper: run ComposeGraph with a single-Identity graph of the given element
// type so LogTensorDetails encounters that QNN data type in its switch.
static void RunComposeGraphWithDtype(ONNXTensorElementDataType elem_type) {
  QnnModelHtpTestContext ctx;
  if (!ctx.IsValid()) GTEST_SKIP() << "HTP backend not available on this host";
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo x{"x", elem_type, {1, 4}};
  FakeValueInfo y{"y", elem_type, {1, 4}};
  FakeNode inner{"identity_0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{inner}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};

  std::string tmp_json = (std::filesystem::temp_directory_path() / "qnn_unit_model_test_dtype.json").string();
  std::remove(tmp_json.c_str());

  std::vector<std::string> in_names{"x"};
  std::vector<std::string> out_names{"y"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                          &in_names, &out_names, &settings,
                          nullptr, nullptr, tmp_json};
  auto status = ctx.model->ComposeGraph(mc);
  // ComposeGraph may fail if HTP doesn't accept this dtype for Identity; the
  // failure-path LogTensorDetails datatype cases are then NOT covered. For
  // dtypes that succeed, LogTensorDetails fires and the switch case is hit.
  if (!status.IsOK()) {
    std::remove(tmp_json.c_str());
    GTEST_SKIP() << "HTP does not accept Identity for this dtype: " << status.GetErrorMessage();
  }
  std::string log_file = tmp_json.substr(0, tmp_json.rfind('.')) + "_tensor_log.json";
  std::remove(tmp_json.c_str());
  std::remove(log_file.c_str());
}

TEST(QnnUnit_ModelTest, LogTensorDetails_INT8_CoversSwitchCase) {
  RunComposeGraphWithDtype(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
}
TEST(QnnUnit_ModelTest, LogTensorDetails_INT16_CoversSwitchCase) {
  RunComposeGraphWithDtype(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
}
TEST(QnnUnit_ModelTest, LogTensorDetails_INT64_CoversSwitchCase) {
  RunComposeGraphWithDtype(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}
TEST(QnnUnit_ModelTest, LogTensorDetails_UINT16_CoversSwitchCase) {
  RunComposeGraphWithDtype(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
}
TEST(QnnUnit_ModelTest, LogTensorDetails_UINT32_CoversSwitchCase) {
  RunComposeGraphWithDtype(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
}
TEST(QnnUnit_ModelTest, LogTensorDetails_FLOAT16_CoversSwitchCase) {
  RunComposeGraphWithDtype(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
}
TEST(QnnUnit_ModelTest, LogTensorDetails_BOOL_CoversSwitchCase) {
  RunComposeGraphWithDtype(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}

// json_qnn_graph_path without "." extension — exercises the else branch at
// LogTensorDetails line 811 that appends "_tensor_log.json" directly.
TEST(QnnUnit_ModelTest, LogTensorDetails_JsonPathWithoutDot_CoversElseBranch) {
  QnnModelHtpTestContext ctx;
  if (!ctx.IsValid()) GTEST_SKIP() << "HTP backend not available on this host";
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode inner{"identity_0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{inner}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};

  std::string tmp_json = (std::filesystem::temp_directory_path() / "qnn_unit_no_ext").string();
  std::remove(tmp_json.c_str());

  std::vector<std::string> in_names{"x"};
  std::vector<std::string> out_names{"y"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                          &in_names, &out_names, &settings,
                          nullptr, nullptr, tmp_json};

  ASSERT_TRUE(ctx.model->ComposeGraph(mc).IsOK());
  std::string log_file = tmp_json + "_tensor_log.json";  // path lacks "." → just appended
  EXPECT_TRUE(std::filesystem::exists(log_file));
  std::remove(tmp_json.c_str());
  std::remove(log_file.c_str());
}

// json_qnn_graph_path that points to a non-existent directory — fopen fails,
// exercises the else branch at LogTensorDetails line 820.
TEST(QnnUnit_ModelTest, LogTensorDetails_InvalidJsonPath_CoversFileOpenFailure) {
  QnnModelHtpTestContext ctx;
  if (!ctx.IsValid()) GTEST_SKIP() << "HTP backend not available on this host";
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode inner{"identity_0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{inner}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};

  // Directory path component does not exist → fopen returns nullptr.
  std::string bad_json = "/nonexistent_dir_qnn_unit_test/graph.json";

  std::vector<std::string> in_names{"x"};
  std::vector<std::string> out_names{"y"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                          &in_names, &out_names, &settings,
                          nullptr, nullptr, bad_json};

  // ComposeGraph itself still succeeds (the JSON write failure is just a warning).
  EXPECT_TRUE(ctx.model->ComposeGraph(mc).IsOK());
}

// ---------------------------------------------------------------------------
// Sanity probe: confirm HTP SetupBackend succeeds on this host.
// Skipped silently if HTP backend unavailable (e.g., aarch64 host without SDK).
// ---------------------------------------------------------------------------
TEST(QnnUnit_ModelTest, Probe_HtpSetupBackend_WorksOnX86_64) {
  QnnModelHtpTestContext ctx;
  if (!ctx.IsValid()) {
    GTEST_SKIP() << "HTP backend not available on this host";
  }
  EXPECT_NE(ctx.model, nullptr);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
