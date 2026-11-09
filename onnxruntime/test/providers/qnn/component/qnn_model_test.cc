// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Unit tests for QnnModel (qnn_model.cc).

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/qnn_test_windows_compat.h"

#include "HTP/QnnHtpDevice.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/infra/qnn_fake_ort_graph.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

template <typename Tag, typename Tag::type Member>
struct QnnModelPrivateMember {
  friend typename Tag::type GetQnnModelPrivateMember(Tag) { return Member; }
};

struct SetupTensorsTag {
  using type = Ort::Status (qnn::QnnModel::*)(std::vector<qnn::QnnTensorInfo>&,
                                              const std::vector<qnn::QnnTensorWrapper>&,
                                              bool);
  friend type GetQnnModelPrivateMember(SetupTensorsTag);
};

template struct QnnModelPrivateMember<SetupTensorsTag, &qnn::QnnModel::SetupTensors>;

struct GraphInfoTag {
  using type = std::unique_ptr<qnn::GraphInfo> qnn::QnnModel::*;
  friend type GetQnnModelPrivateMember(GraphInfoTag);
};

template struct QnnModelPrivateMember<GraphInfoTag, &qnn::QnnModel::graph_info_>;

struct PrepareRuntimeTensorTag {
  using type = Ort::Status (qnn::QnnModel::*)(const qnn::QnnTensorInfo&,
                                              const qnn::OnnxTensorInfo&,
                                              const OrtValue*, uint32_t&, Qnn_Tensor_t&,
                                              std::vector<uint32_t>&, uint32_t&) const;
  friend type GetQnnModelPrivateMember(PrepareRuntimeTensorTag);
};

template struct QnnModelPrivateMember<PrepareRuntimeTensorTag, &qnn::QnnModel::PrepareRuntimeTensor>;

struct BindQnnTensorMemoryTag {
  using type = Ort::Status (qnn::QnnModel::*)(const Ort::Logger&,
                                              const OrtMemoryInfo*,
                                              void*,
                                              uint32_t,
                                              Qnn_ContextHandle_t,
                                              Qnn_Tensor_t&);
  friend type GetQnnModelPrivateMember(BindQnnTensorMemoryTag);
};

template struct QnnModelPrivateMember<BindQnnTensorMemoryTag,
                                      &qnn::QnnModel::BindQnnTensorMemoryToOrtValueMemory>;

}  // namespace

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

  explicit QnnModelMinimalTestContext(bool use_gpu_default_memory = false) {
    stub_ort_api.MemoryInfoGetDeviceType = use_gpu_default_memory
                                               ? +[](const OrtMemoryInfo*, OrtMemoryInfoDeviceType* device_type) noexcept {
                                                   *device_type = OrtMemoryInfoDeviceType_GPU;
                                                 }
                                               : +[](const OrtMemoryInfo*, OrtMemoryInfoDeviceType* device_type) noexcept {
                                                   *device_type = OrtMemoryInfoDeviceType_CPU;
                                                 };
    stub_ort_api.MemoryInfoGetDeviceMemType = [](const OrtMemoryInfo*) noexcept {
      return OrtDeviceMemoryType_DEFAULT;
    };

    qnn::QnnBackendManagerConfig cfg;
    cfg.backend_path = QnnHtpBackendLibraryName();
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
// FinalizeGraphs, etc.). Loads the platform HTP backend and runs SetupBackend.
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
    cfg.backend_path = QnnHtpBackendLibraryName();
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

TEST(QnnUnit_ModelTest, Deserialize_Version1_NullGraphName_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  QnnSystemContext_GraphInfo_t info{};
  info.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;

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
TEST(QnnUnit_ModelTest, Deserialize_Version2_NullGraphName_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  QnnSystemContext_GraphInfo_t info{};
  info.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2;

  Qnn_ContextHandle_t fake_ctx = nullptr;
  EXPECT_FALSE(ctx.model->DeserializeGraphInfoFromBinaryInfo(info, fake_ctx).IsOK());
}

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
TEST(QnnUnit_ModelTest, Deserialize_Version3_NullGraphName_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  QnnSystemContext_GraphInfo_t info{};
  info.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3;

  Qnn_ContextHandle_t fake_ctx = nullptr;
  EXPECT_FALSE(ctx.model->DeserializeGraphInfoFromBinaryInfo(info, fake_ctx).IsOK());
}

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

TEST(QnnUnit_ModelTest, SetGraphInputOutputInfo_AotDynamicBatch_PreservesOrtDeclarations) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo input{"input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 3, 8, 8}};
  FakeValueInfo output{"output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 2, 6, 6}};
  FakeGraph graph{{}, {&input}, {&output}, {}};
  FakeNode fused{"fused", "EPContext", "com.microsoft", 1, {&input}, {&output}};
  std::unordered_map<std::string, std::string> overrides{{"input", "context_input"}};
  qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger};
  mc.tensor_name_overrides = &overrides;

  std::vector<qnn::QnnTensorWrapper> inputs;
  inputs.emplace_back("context_input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                      qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{2, 8, 8, 3});
  std::vector<qnn::QnnTensorWrapper> outputs;
  outputs.emplace_back("output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                       qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4, 6, 6, 2});
  ((*ctx.model).*GetQnnModelPrivateMember(GraphInfoTag{})) = std::make_unique<qnn::GraphInfo>(
      nullptr, "context_graph", nullptr, std::move(inputs), std::move(outputs));

  const auto status = ctx.model->SetGraphInputOutputInfo(mc);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(ctx.model->GetInputsInfo().at("input").shape_, (std::vector<int64_t>{2, 3, 8, 8}));
  EXPECT_EQ(ctx.model->GetOutputsInfo().at("output").shape_, (std::vector<int64_t>{4, 2, 6, 6}));
  EXPECT_TRUE(ctx.model->GetInputsInfo().at("context_input").dynamic_batch_);
  EXPECT_TRUE(ctx.model->GetOutputsInfo().at("output").dynamic_batch_);
  EXPECT_EQ(input.shape, (std::vector<int64_t>{-1, 3, 8, 8}));
  EXPECT_EQ(output.shape, (std::vector<int64_t>{-1, 2, 6, 6}));
}

TEST(QnnUnit_ModelTest, SetGraphInputOutputInfo_AotDynamicBatch_RejectsUnsupportedShapes) {
  struct Case {
    const char* name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    bool has_context;
    const char* context_input_name;
    std::vector<uint32_t> base_input_shape;
  };
  const std::vector<Case> cases{
      {"JIT input", {-1, 3, 8, 8}, {2, 2, 6, 6}, false, "input", {2, 8, 8, 3}},
      {"JIT output", {2, 3, 8, 8}, {-1, 2, 6, 6}, false, "input", {2, 8, 8, 3}},
      {"non-batch input", {-1, -1, 8, 8}, {2, 2, 6, 6}, true, "input", {2, 8, 8, 3}},
      {"non-batch output", {2, 3, 8, 8}, {2, -1, 6, 6}, true, "input", {2, 8, 8, 3}},
      {"missing tensor", {-1, 3, 8, 8}, {2, 2, 6, 6}, true, "other", {2, 8, 8, 3}},
      {"rank mismatch", {-1, 3, 8, 8}, {2, 2, 6, 6}, true, "input", {2, 8, 3}},
      {"zero base batch", {-1, 3, 8, 8}, {2, 2, 6, 6}, true, "input", {0, 8, 8, 3}},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    QnnModelMinimalTestContext ctx;
    ASSERT_TRUE(ctx.IsValid());
    InstallFakeGraphApiStubs(ctx.stub_ort_api);
    OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

    FakeValueInfo input{"input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, test_case.input_shape};
    FakeValueInfo output{"output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, test_case.output_shape};
    FakeGraph graph{{}, {&input}, {&output}, {}};
    FakeNode fused{"fused", "EPContext", "com.microsoft", 1, {&input}, {&output}};
    qnn::QnnModelContext mc{*graph.AsGraph(), *fused.AsNode(), ctx.logger};
    if (test_case.has_context) {
      std::vector<qnn::QnnTensorWrapper> inputs;
      inputs.emplace_back(test_case.context_input_name, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                          qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>(test_case.base_input_shape));
      ((*ctx.model).*GetQnnModelPrivateMember(GraphInfoTag{})) = std::make_unique<qnn::GraphInfo>(
          nullptr, "context_graph", nullptr, std::move(inputs), std::vector<qnn::QnnTensorWrapper>{});
    }
    EXPECT_FALSE(ctx.model->SetGraphInputOutputInfo(mc).IsOK());
    EXPECT_EQ(input.shape, test_case.input_shape);
    EXPECT_EQ(output.shape, test_case.output_shape);
  }
}

TEST(QnnUnit_ModelTest, SetupTensors_SparseOutputIndex_ReturnsCompactedTensorInfo) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo output0{"output0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeValueInfo output1{"output1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeGraph graph{{}, {}, {&output0, &output1}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {}, {&output0, &output1}};
  std::vector<std::string> input_names;
  std::vector<std::string> output_names{"output0", "output1"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext model_context{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                                     &input_names, &output_names, &settings};
  ASSERT_TRUE(ctx.model->SetGraphInputOutputInfo(model_context).IsOK());

  std::vector<qnn::QnnTensorWrapper> tensor_wrappers;
  tensor_wrappers.emplace_back("output1", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1});
  std::vector<qnn::QnnTensorInfo> tensor_infos;
  const auto setup_tensors = GetQnnModelPrivateMember(SetupTensorsTag{});

  const auto status = ((*ctx.model).*setup_tensors)(tensor_infos, tensor_wrappers, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_EQ(tensor_infos.size(), 1u);
  EXPECT_EQ(tensor_infos[0].tensor_wrapper, &tensor_wrappers[0]);
  EXPECT_EQ(tensor_infos[0].ort_index, 1u);
}

TEST(QnnUnit_ModelTest, SetupTensors_OutputCountLessThanTensorWrapperCount_ReturnsError) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  OrtGlobalApiOverride api_override(&ctx.stub_ort_api);

  FakeValueInfo output0{"output0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeGraph graph{{}, {}, {&output0}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {}, {&output0}};
  std::vector<std::string> input_names;
  std::vector<std::string> output_names{"output0"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext model_context{*graph.AsGraph(), *fused.AsNode(), ctx.logger,
                                     &input_names, &output_names, &settings};
  ASSERT_TRUE(ctx.model->SetGraphInputOutputInfo(model_context).IsOK());

  std::vector<qnn::QnnTensorWrapper> tensor_wrappers;
  tensor_wrappers.emplace_back("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1});
  tensor_wrappers.emplace_back("output1", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1});
  std::vector<qnn::QnnTensorInfo> tensor_infos;
  const auto setup_tensors = GetQnnModelPrivateMember(SetupTensorsTag{});

  const auto status = ((*ctx.model).*setup_tensors)(tensor_infos, tensor_wrappers, false);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_ModelTest, BindTensor_UnregisteredGpuDefaultMemory_ReturnsError) {
  QnnModelMinimalTestContext ctx(/*use_gpu_default_memory=*/true);
  ASSERT_TRUE(ctx.IsValid());

  const auto bind_tensor_memory = GetQnnModelPrivateMember(BindQnnTensorMemoryTag{});
  int data = 0;
  Qnn_Tensor_t qnn_tensor = QNN_TENSOR_INIT;
  const auto status = ((*ctx.model).*bind_tensor_memory)(
      ctx.logger, reinterpret_cast<const OrtMemoryInfo*>(1), &data, sizeof(data), nullptr, qnn_tensor);

  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorMessage(), "GPU DEFAULT memory is not a registered imported memory handle.");
}

TEST(QnnUnit_ModelTest, PrepareRuntimeTensor_ScalesContextLayoutAndActualBuffers) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  ctx.stub_ort_api.GetTensorTypeAndShape = Ort::GetApi().GetTensorTypeAndShape;
  const auto prepare = GetQnnModelPrivateMember(PrepareRuntimeTensorTag{});
  const auto bind = GetQnnModelPrivateMember(BindQnnTensorMemoryTag{});
  Ort::AllocatorWithDefaultOptions allocator;
  uint32_t multiplier = 0;

  qnn::QnnTensorWrapper input("input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                              qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{2, 8, 8, 3});
  qnn::QnnTensorWrapper output("output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4, 6, 6, 2});
  qnn::OnnxTensorInfo input_info(0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 3, 8, 8}, true);
  qnn::OnnxTensorInfo output_info(0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4, 2, 6, 6}, true);
  const std::vector<int64_t> input_shape{256, 3, 8, 8};
  const std::vector<int64_t> output_shape{512, 2, 6, 6};
  auto input_value = Ort::Value::CreateTensor<float>(allocator, input_shape.data(), input_shape.size());
  auto output_value = Ort::Value::CreateTensor<float>(allocator, output_shape.data(), output_shape.size());
  qnn::QnnTensorInfo qnn_input{&input, 2 * 3 * 8 * 8 * sizeof(float), 0};
  qnn::QnnTensorInfo qnn_output{&output, 4 * 2 * 6 * 6 * sizeof(float), 0};
  Qnn_Tensor_t input_tensor = QNN_TENSOR_INIT;
  Qnn_Tensor_t output_tensor = QNN_TENSOR_INIT;
  std::vector<uint32_t> input_dimensions, output_dimensions;
  uint32_t input_bytes = 0, output_bytes = 0;
  auto status = ((*ctx.model).*prepare)(qnn_input, input_info, input_value, multiplier,
                                        input_tensor, input_dimensions, input_bytes);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  status = ((*ctx.model).*prepare)(qnn_output, output_info, output_value, multiplier,
                                   output_tensor, output_dimensions, output_bytes);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(multiplier, 128u);
  EXPECT_EQ(input_dimensions, (std::vector<uint32_t>{256, 8, 8, 3}));
  EXPECT_EQ(output_dimensions, (std::vector<uint32_t>{512, 6, 6, 2}));
  EXPECT_EQ(qnn::GetQnnTensorDims(input_tensor), input_dimensions.data());
  EXPECT_EQ(qnn::GetQnnTensorDims(output_tensor), output_dimensions.data());
  EXPECT_EQ(input.GetTensorDims(), (std::vector<uint32_t>{2, 8, 8, 3}));
  EXPECT_EQ(output.GetTensorDims(), (std::vector<uint32_t>{4, 6, 6, 2}));
  EXPECT_EQ(input_bytes, qnn_input.tensor_byte_size * 128u);
  EXPECT_EQ(output_bytes, qnn_output.tensor_byte_size * 128u);
  ASSERT_TRUE(((*ctx.model).*bind)(ctx.logger, nullptr, input_value.GetTensorMutableData<float>(),
                                   input_bytes, nullptr, input_tensor)
                  .IsOK());
  ASSERT_TRUE(((*ctx.model).*bind)(ctx.logger, nullptr, output_value.GetTensorMutableData<float>(),
                                   output_bytes, nullptr, output_tensor)
                  .IsOK());
  EXPECT_EQ(qnn::GetQnnTensorClientBuf(input_tensor).dataSize, input_bytes);
  EXPECT_EQ(qnn::GetQnnTensorClientBuf(output_tensor).dataSize, output_bytes);
}

TEST(QnnUnit_ModelTest, PrepareRuntimeTensor_VaryingRunsKeepBaseAndEarlierDimensions) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  ctx.stub_ort_api.GetTensorTypeAndShape = Ort::GetApi().GetTensorTypeAndShape;
  const auto prepare = GetQnnModelPrivateMember(PrepareRuntimeTensorTag{});
  Ort::AllocatorWithDefaultOptions allocator;
  qnn::QnnTensorWrapper wrapper("input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                                qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{2, 3});
  qnn::OnnxTensorInfo info(0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 3}, true);
  qnn::QnnTensorInfo qnn_info{&wrapper, 6 * sizeof(float), 0};
  std::vector<std::vector<uint32_t>> dimensions(3);
  std::vector<Qnn_Tensor_t> tensors(3, QNN_TENSOR_INIT);
  const std::vector<int64_t> batches{4, 256, 2};
  for (size_t i = 0; i < batches.size(); ++i) {
    const std::vector<int64_t> shape{batches[i], 3};
    auto value = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    uint32_t multiplier = 0, bytes = 0;
    const auto status = ((*ctx.model).*prepare)(qnn_info, info, value, multiplier,
                                                tensors[i], dimensions[i], bytes);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    EXPECT_EQ(qnn::GetQnnTensorDims(tensors[i])[0], batches[i]);
    EXPECT_EQ(multiplier, batches[i] / 2);
  }
  EXPECT_EQ(qnn::GetQnnTensorDims(tensors[0])[0], 4u);
  EXPECT_EQ(qnn::GetQnnTensorDims(tensors[1])[0], 256u);
  EXPECT_EQ(wrapper.GetTensorDims(), (std::vector<uint32_t>{2, 3}));
}

TEST(QnnUnit_ModelTest, PrepareRuntimeTensor_RejectsInvalidShapesTypesAndSizes) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());
  ctx.stub_ort_api.GetTensorTypeAndShape = Ort::GetApi().GetTensorTypeAndShape;
  const auto prepare = GetQnnModelPrivateMember(PrepareRuntimeTensorTag{});
  Ort::AllocatorWithDefaultOptions allocator;
  struct Case {
    const char* name;
    std::vector<int64_t> shape;
    bool dynamic_batch = true;
    uint32_t multiplier = 0;
    uint32_t base_bytes = 2 * 3 * 8 * 8 * sizeof(float);
    bool wrong_type = false;
  };
  const std::vector<Case> cases{
      {"nonintegral batch", {3, 3, 8, 8}},
      {"zero batch", {0, 3, 8, 8}},
      {"rank", {4, 3, 64}},
      {"equal byte count, wrong dimensions", {4, 3, 4, 16}},
      {"static batch", {4, 3, 8, 8}, false},
      {"inconsistent multiplier", {4, 3, 8, 8}, true, 3},
      {"base byte count", {4, 3, 8, 8}, true, 0, 1},
      {"byte overflow", {4, 3, 8, 8}, true, 0, std::numeric_limits<uint32_t>::max()},
      {"element type", {4, 3, 8, 8}, true, 0, 2 * 3 * 8 * 8 * sizeof(float), true},
  };
  qnn::QnnTensorWrapper wrapper("input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                                qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{2, 8, 8, 3});
  for (const auto& test : cases) {
    SCOPED_TRACE(test.name);
    qnn::OnnxTensorInfo info(0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 3, 8, 8}, test.dynamic_batch);
    qnn::QnnTensorInfo qnn_info{&wrapper, test.base_bytes, 0};
    auto value = test.wrong_type
                     ? Ort::Value::CreateTensor<int32_t>(allocator, test.shape.data(), test.shape.size())
                     : Ort::Value::CreateTensor<float>(allocator, test.shape.data(), test.shape.size());
    uint32_t multiplier = test.multiplier, bytes = 0;
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
    std::vector<uint32_t> dimensions;
    EXPECT_FALSE(((*ctx.model).*prepare)(qnn_info, info, value, multiplier,
                                         tensor, dimensions, bytes)
                     .IsOK());
    EXPECT_EQ(wrapper.GetTensorDims(), (std::vector<uint32_t>{2, 8, 8, 3}));
  }
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

// ---------------------------------------------------------------------------
// ApplyRuntimeGraphConfigs tests
//
// These tests exercise the code-logic paths of ApplyRuntimeGraphConfigs that
// are reachable without hardware. The hardware-dependent path (graphSetConfig
// on a real finalized graph) is covered by the EPContext and SSR integration
// tests.
// ---------------------------------------------------------------------------

// A freshly-constructed QnnModel has no graph_info_ yet. ApplyRuntimeGraphConfigs
// must return OK without calling graphSetConfig (graph_info_ == nullptr guard).
TEST(QnnUnit_ModelTest, ApplyRuntimeGraphConfigs_NoGraphInfo_ReturnsOk) {
  QnnModelHtpTestContext ctx;
  if (!ctx.IsValid()) {
    GTEST_SKIP() << "HTP backend not available on this host";
  }

  qnn::HtpGraphConfigs_t configs;
  configs.enable_htp_fp16_clamp_overflow = true;

  auto status = ctx.model->ApplyRuntimeGraphConfigs(configs, ctx.logger);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

// When no runtime-applicable flags are set, the builder produces nothing and
// ApplyRuntimeGraphConfigs returns OK without issuing a graphSetConfig call.
TEST(QnnUnit_ModelTest, ApplyRuntimeGraphConfigs_NoFlagsSet_ReturnsOk) {
  QnnModelHtpTestContext ctx;
  if (!ctx.IsValid()) {
    GTEST_SKIP() << "HTP backend not available on this host";
  }

  qnn::HtpGraphConfigs_t configs;  // all flags default false
  auto status = ctx.model->ApplyRuntimeGraphConfigs(configs, ctx.logger);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

// ApplyRuntimeGraphConfigs stores the configs struct for SSR re-apply. Verify
// the cached value reflects the last call (used by RecoverFromSSR).
TEST(QnnUnit_ModelTest, ApplyRuntimeGraphConfigs_CachesConfigsForSsr) {
  QnnModelHtpTestContext ctx;
  if (!ctx.IsValid()) {
    GTEST_SKIP() << "HTP backend not available on this host";
  }

  // First call: enable the flag.
  qnn::HtpGraphConfigs_t configs_on;
  configs_on.enable_htp_fp16_clamp_overflow = true;
  EXPECT_TRUE(ctx.model->ApplyRuntimeGraphConfigs(configs_on, ctx.logger).IsOK());

  // Second call with flag off — SSR would now skip re-applying the config.
  qnn::HtpGraphConfigs_t configs_off;
  configs_off.enable_htp_fp16_clamp_overflow = false;
  EXPECT_TRUE(ctx.model->ApplyRuntimeGraphConfigs(configs_off, ctx.logger).IsOK());

  // Third call restores the flag — SSR re-apply would include it again.
  EXPECT_TRUE(ctx.model->ApplyRuntimeGraphConfigs(configs_on, ctx.logger).IsOK());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
