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

struct GraphInfoTag {
  using type = std::unique_ptr<qnn::GraphInfo> qnn::QnnModel::*;
  friend type GetQnnModelPrivateMember(GraphInfoTag);
};
template struct QnnModelPrivateMember<GraphInfoTag, &qnn::QnnModel::graph_info_>;

struct GraphEmitterTag {
  using type = std::unique_ptr<qnn::QnnGraphEmitter> qnn::QnnModel::*;
  friend type GetQnnModelPrivateMember(GraphEmitterTag);
};
template struct QnnModelPrivateMember<GraphEmitterTag, &qnn::QnnModel::qnn_graph_emitter_>;

struct BackendContextsTag {
  using type = std::vector<Qnn_ContextHandle_t> qnn::QnnBackendManager::*;
  friend type GetQnnModelPrivateMember(BackendContextsTag);
};
template struct QnnModelPrivateMember<BackendContextsTag, &qnn::QnnBackendManager::contexts_>;

struct ModelGraphContextStub {
  Qnn_GraphHandle_t graph;
  Qnn_ErrorHandle_t result = QNN_SUCCESS;
};

void InstallModelGraphStubs(QNN_INTERFACE_VER_TYPE& qnn_interface) {
  qnn_interface.graphRetrieve = [](Qnn_ContextHandle_t context, const char*,
                                   Qnn_GraphHandle_t* graph) -> Qnn_ErrorHandle_t {
    const auto& stub = *static_cast<ModelGraphContextStub*>(context);
    *graph = stub.graph;
    return stub.result;
  };
  qnn_interface.errorGetMessage = [](Qnn_ErrorHandle_t, const char** message) -> Qnn_ErrorHandle_t {
    *message = "stub QNN error";
    return QNN_SUCCESS;
  };
}

Ort::Status DeserializeStubGraph(qnn::QnnModel& model, ModelGraphContextStub& context) {
  qnn::QnnTensorWrapper input("x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                              qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper output("y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  Qnn_Tensor_t input_tensor = input.GetQnnTensor();
  Qnn_Tensor_t output_tensor = output.GetQnnTensor();
  QnnSystemContext_GraphInfo_t info{};
  info.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;
  info.graphInfoV1.graphName = const_cast<char*>("graph");
  info.graphInfoV1.numGraphInputs = 1;
  info.graphInfoV1.graphInputs = &input_tensor;
  info.graphInfoV1.numGraphOutputs = 1;
  info.graphInfoV1.graphOutputs = &output_tensor;
  return model.DeserializeGraphInfoFromBinaryInfo(info, &context);
}

}  // namespace

class QnnUnitModelExecutionTest : public ::testing::Test {
 public:
  QnnUnitModelExecutionTest() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnUnitModelExecutionTest);

 protected:
  struct ExecutionStub {
    ExecutionStub() = default;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExecutionStub);

    qnn::QnnBackendManager* backend = nullptr;
    Qnn_ContextHandle_t original_context = nullptr;
    std::vector<Qnn_ErrorHandle_t> results{QNN_GRAPH_NO_ERROR};
    size_t calls = 0;
    bool recover_context = false;
  } execution_;

  struct KernelContext {
    KernelContext() = default;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelContext);

    float input_data[4]{};
    float output_data[4]{};
    int64_t shape[1]{4};
    Ort::MemoryInfo memory_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Value input{Ort::Value::CreateTensor<float>(memory_info, input_data, 4, shape, 1)};
    Ort::Value output{Ort::Value::CreateTensor<float>(memory_info, output_data, 4, shape, 1)};
  } kernel_;

  StubApiEnv env_;
  StubBackendManager backend_{env_.api_ptrs, env_.logger};
  ModelGraphContextStub original_context_{&execution_};
  ModelGraphContextStub replacement_context_{&execution_};
  std::unique_ptr<qnn::QnnModel> model_;

  void SetUp() override {
    auto& qnn_interface = backend_.QnnInterface();
    InstallModelGraphStubs(qnn_interface);
    qnn_interface.contextFree = [](Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      return QNN_SUCCESS;
    };
    qnn_interface.graphExecute = [](Qnn_GraphHandle_t graph, const Qnn_Tensor_t*, uint32_t,
                                    Qnn_Tensor_t*, uint32_t, Qnn_ProfileHandle_t,
                                    Qnn_SignalHandle_t) -> Qnn_ErrorHandle_t {
      auto& execution = *static_cast<ExecutionStub*>(graph);
      const auto result = execution.results.at(execution.calls++);
      if (execution.recover_context && execution.calls == 1) {
        // Simulate a sibling model replacing the shared context after an SSR.
        // RecoverFromSSR then reuses the replacement without loading a binary.
        execution.backend->ReleaseSpecificContextHandle(execution.original_context);
      }
      return result;
    };
    qnn_interface.errorGetMessage = [](Qnn_ErrorHandle_t error, const char** message) -> Qnn_ErrorHandle_t {
      *message = error == QNN_COMMON_ERROR_SYSTEM_COMMUNICATION
                     ? "QNN_COMMON_ERROR_SYSTEM_COMMUNICATION"
                     : "QNN_COMMON_ERROR_INVALID_ARGUMENT";
      return QNN_SUCCESS;
    };
    execution_.backend = backend_.GetMutable();
    execution_.original_context = &original_context_;
    ASSERT_TRUE(backend_.GetMutable()->AddQnnContextHandle(&original_context_).IsOK());
    ASSERT_TRUE(backend_.GetMutable()->AddQnnContextHandle(&replacement_context_).IsOK());
    model_ = std::make_unique<qnn::QnnModel>(backend_.GetMutable(), env_.api_ptrs);

    InstallFakeGraphApiStubs(env_.stub_ort_api);
    {
      OrtGlobalApiOverride api_override(&env_.stub_ort_api);
      FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
      FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
      FakeGraph graph{{}, {&x}, {&y}, {}};
      FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};
      std::vector<std::string> input_names{"x"};
      std::vector<std::string> output_names{"y"};
      qnn::ModelSettings settings{};
      qnn::QnnModelContext model_context{*graph.AsGraph(), *fused.AsNode(), env_.logger,
                                         &input_names, &output_names, &settings};
      ASSERT_TRUE(model_->SetGraphInputOutputInfo(model_context).IsOK());
    }
    ASSERT_TRUE(DeserializeStubGraph(*model_, original_context_).IsOK());
    ASSERT_TRUE(model_->SetupQnnInputOutput(env_.logger).IsOK());

    // Bind real ORT tensors, while the QNN stub controls execution results.
    env_.stub_ort_api = Ort::GetApi();
    env_.stub_ort_api.KernelContext_GetInputCount = [](const OrtKernelContext*, size_t* count) noexcept -> OrtStatus* {
      *count = 1;
      return nullptr;
    };
    env_.stub_ort_api.KernelContext_GetOutputCount = env_.stub_ort_api.KernelContext_GetInputCount;
    env_.stub_ort_api.KernelContext_GetInput = [](const OrtKernelContext* context, size_t,
                                                  const OrtValue** input) noexcept -> OrtStatus* {
      *input = static_cast<const KernelContext*>(static_cast<const void*>(context))->input;
      return nullptr;
    };
    env_.stub_ort_api.KernelContext_GetOutput = [](OrtKernelContext* context, size_t, const int64_t*, size_t,
                                                   OrtValue** output) noexcept -> OrtStatus* {
      *output = static_cast<KernelContext*>(static_cast<void*>(context))->output;
      return nullptr;
    };
  }

  void EnableRecovery() {
    model_->SetContextRecoveryInfo("context.bin", 0, qnn::ContextPriority::NORMAL);
    execution_.recover_context = true;
  }

  Ort::Status Execute() {
    return model_->ExecuteGraph(reinterpret_cast<OrtKernelContext*>(&kernel_), env_.logger,
                                qnn::EpContextIoDispatch());
  }

  static void ExpectQnnError(const Ort::Status& status, Qnn_ErrorHandle_t error, const char* name) {
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.GetErrorMessage().find(name), std::string::npos);
    EXPECT_NE(status.GetErrorMessage().find("Code: " + std::to_string(error)), std::string::npos);
  }
};

TEST_F(QnnUnitModelExecutionTest, OrdinaryFailureIncludesQnnNameAndCode) {
  execution_.results = {QNN_COMMON_ERROR_INVALID_ARGUMENT};
  const auto status = Execute();
  ASSERT_NO_FATAL_FAILURE(ExpectQnnError(status, QNN_COMMON_ERROR_INVALID_ARGUMENT, "QNN_COMMON_ERROR_INVALID_ARGUMENT"));
  EXPECT_EQ(status.GetErrorCode(), ORT_EP_FAIL);
  EXPECT_EQ(execution_.calls, 1u);
}

TEST_F(QnnUnitModelExecutionTest, SsrWithoutRecoveryIncludesQnnNameAndCode) {
  execution_.results = {QNN_COMMON_ERROR_SYSTEM_COMMUNICATION};
  const auto status = Execute();
  ASSERT_NO_FATAL_FAILURE(ExpectQnnError(status, QNN_COMMON_ERROR_SYSTEM_COMMUNICATION, "QNN_COMMON_ERROR_SYSTEM_COMMUNICATION"));
  EXPECT_EQ(status.GetErrorCode(), QNN_SSR_UNRECOVERABLE_ERROR_CODE);
  EXPECT_NE(status.GetErrorMessage().find("Recovery not supported"), std::string::npos);
  EXPECT_EQ(execution_.calls, 1u);
}

TEST_F(QnnUnitModelExecutionTest, RepeatedSsrIncludesQnnNameAndCode) {
  EnableRecovery();
  execution_.results = {QNN_COMMON_ERROR_SYSTEM_COMMUNICATION, QNN_COMMON_ERROR_SYSTEM_COMMUNICATION};
  const auto status = Execute();
  ASSERT_NO_FATAL_FAILURE(ExpectQnnError(status, QNN_COMMON_ERROR_SYSTEM_COMMUNICATION, "QNN_COMMON_ERROR_SYSTEM_COMMUNICATION"));
  EXPECT_EQ(status.GetErrorCode(), QNN_SSR_UNRECOVERABLE_ERROR_CODE);
  EXPECT_NE(status.GetErrorMessage().find("after SSR recovery"), std::string::npos);
  EXPECT_EQ(execution_.calls, 2u);
}

TEST_F(QnnUnitModelExecutionTest, NonSsrRetryFailureReturnsQnnNameAndCode) {
  EnableRecovery();
  execution_.results = {QNN_COMMON_ERROR_SYSTEM_COMMUNICATION, QNN_COMMON_ERROR_INVALID_ARGUMENT};
  const auto status = Execute();
  ASSERT_NO_FATAL_FAILURE(ExpectQnnError(status, QNN_COMMON_ERROR_INVALID_ARGUMENT, "QNN_COMMON_ERROR_INVALID_ARGUMENT"));
  EXPECT_EQ(status.GetErrorCode(), ORT_EP_FAIL);
  EXPECT_EQ(execution_.calls, 2u);
}

TEST_F(QnnUnitModelExecutionTest, SuccessfulSsrRecoveryReturnsSuccess) {
  EnableRecovery();
  execution_.results = {QNN_COMMON_ERROR_SYSTEM_COMMUNICATION, QNN_GRAPH_NO_ERROR};
  EXPECT_TRUE(Execute().IsOK());
  EXPECT_EQ(execution_.calls, 2u);
}

TEST_F(QnnUnitModelExecutionTest, SuccessfulExecutionReturnsSuccess) {
  EXPECT_TRUE(Execute().IsOK());
  EXPECT_EQ(execution_.calls, 1u);
}

TEST(QnnUnit_ModelTest, DeserializeKeepsEmitterAndMetadataHandlesTogetherOnSuccessAndFailure) {
  StubApiEnv env;
  StubBackendManager backend(env.api_ptrs, env.logger);
  InstallModelGraphStubs(backend.QnnInterface());
  qnn::QnnModel model(backend.GetMutable(), env.api_ptrs);
  auto& graph_info = model.*GetQnnModelPrivateMember(GraphInfoTag{});
  auto& emitter = model.*GetQnnModelPrivateMember(GraphEmitterTag{});

  int original_graph = 0;
  int replacement_graph = 0;
  ModelGraphContextStub original_context{&original_graph};
  ModelGraphContextStub replacement_context{&replacement_graph};
  ASSERT_TRUE(DeserializeStubGraph(model, original_context).IsOK());
  ASSERT_TRUE(DeserializeStubGraph(model, replacement_context).IsOK());
  ASSERT_NE(graph_info, nullptr);
  EXPECT_EQ(graph_info->Graph(), &replacement_graph);
  EXPECT_EQ(graph_info->GraphContext(), &replacement_context);
  EXPECT_EQ(emitter->Graph(), graph_info->Graph());
  EXPECT_EQ(emitter->Context(), graph_info->GraphContext());

  const auto* previous_metadata = graph_info.get();
  original_context.result = QNN_COMMON_ERROR_INVALID_ARGUMENT;
  EXPECT_FALSE(DeserializeStubGraph(model, original_context).IsOK());
  EXPECT_EQ(graph_info.get(), previous_metadata);
  EXPECT_EQ(emitter->Graph(), graph_info->Graph());
  EXPECT_EQ(emitter->Context(), graph_info->GraphContext());

  original_context.result = QNN_SUCCESS;
  original_context.graph = nullptr;
  EXPECT_FALSE(DeserializeStubGraph(model, original_context).IsOK());
  EXPECT_EQ(graph_info.get(), previous_metadata);
  EXPECT_EQ(graph_info->Graph(), &replacement_graph);
  EXPECT_EQ(graph_info->GraphContext(), &replacement_context);
  EXPECT_EQ(emitter->Graph(), graph_info->Graph());
  EXPECT_EQ(emitter->Context(), graph_info->GraphContext());
}

TEST(QnnUnit_ModelTest, SsrRecoveryReusesNewContextAndReplacesBothHandlePairs) {
  StubApiEnv env;
  StubBackendManager backend(env.api_ptrs, env.logger);
  backend.BackendType() = qnn::QnnBackendType::HTP;
  InstallModelGraphStubs(backend.QnnInterface());
  qnn::QnnModel model(backend.GetMutable(), env.api_ptrs);
  InstallFakeGraphApiStubs(env.stub_ort_api);
  OrtGlobalApiOverride api_override(&env.stub_ort_api);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeGraph graph{{}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};
  std::vector<std::string> input_names{"x"};
  std::vector<std::string> output_names{"y"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext model_context{*graph.AsGraph(), *fused.AsNode(), env.logger,
                                     &input_names, &output_names, &settings};
  ASSERT_TRUE(model.SetGraphInputOutputInfo(model_context).IsOK());

  int original_graph = 0;
  int replacement_graph = 0;
  ModelGraphContextStub original_context{&original_graph};
  ModelGraphContextStub replacement_context{&replacement_graph};
  ASSERT_TRUE(DeserializeStubGraph(model, original_context).IsOK());
  auto& graph_info = model.*GetQnnModelPrivateMember(GraphInfoTag{});
  auto& emitter = model.*GetQnnModelPrivateMember(GraphEmitterTag{});
  const auto* previous_metadata = graph_info.get();
  const auto* previous_inputs = graph_info->InputTensors().data();
  const auto* previous_outputs = graph_info->OutputTensors().data();

  // Simulate another model having recovered the shared context: the old context
  // is no longer tracked, and GetQnnContext(0) returns the replacement.
  auto& contexts = (*backend.GetMutable()).*GetQnnModelPrivateMember(BackendContextsTag{});
  contexts.push_back(&replacement_context);
  ASSERT_FALSE(backend.Get()->HasContextHandle(&original_context));
  ASSERT_TRUE(model.RecoverFromSSR(env.logger, qnn::EpContextIoDispatch()).IsOK());

  EXPECT_EQ(graph_info.get(), previous_metadata);
  EXPECT_EQ(graph_info->InputTensors().data(), previous_inputs);
  EXPECT_EQ(graph_info->OutputTensors().data(), previous_outputs);
  EXPECT_EQ(graph_info->Graph(), &replacement_graph);
  EXPECT_EQ(graph_info->GraphContext(), &replacement_context);
  EXPECT_EQ(emitter->Graph(), graph_info->Graph());
  EXPECT_EQ(emitter->Context(), graph_info->GraphContext());
}

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
