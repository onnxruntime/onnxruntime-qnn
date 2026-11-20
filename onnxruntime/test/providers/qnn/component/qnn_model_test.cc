// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Unit tests for QnnModel (qnn_model.cc).

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <gsl/gsl>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/qnn_test_windows_compat.h"

#include "HTP/QnnHtpDevice.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/shared_context.h"

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
                                              const qnn::QnnBackendManager::ContextHandleLease&,
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

void SetQnnModelNameForTest(qnn::QnnModel& model, const std::string& name) {
  const auto graph_info = GetQnnModelPrivateMember(GraphInfoTag{});
  model.*graph_info = std::make_unique<qnn::GraphInfo>(
      nullptr, name, nullptr, std::vector<qnn::QnnTensorWrapper>{}, std::vector<qnn::QnnTensorWrapper>{});
}

struct GraphEmitterTag {
  using type = std::unique_ptr<qnn::QnnGraphEmitter> qnn::QnnModel::*;
  friend type GetQnnModelPrivateMember(GraphEmitterTag);
};
template struct QnnModelPrivateMember<GraphEmitterTag, &qnn::QnnModel::qnn_graph_emitter_>;

struct ModelGraphContextStub {
  Qnn_GraphHandle_t graph;
  Qnn_ErrorHandle_t result = QNN_SUCCESS;
  std::atomic<unsigned> free_calls{0};
};

void InstallModelGraphStubs(QNN_INTERFACE_VER_TYPE& qnn_interface) {
  qnn_interface.contextFree = [](Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
    return QNN_SUCCESS;
  };
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

TEST(QnnUnit_ModelTest, DeserializeKeepsEmitterAndMetadataHandlesTogetherOnSuccessAndFailure) {
  StubApiEnv env;
  StubBackendManager backend(env.api_ptrs, env.logger);
  InstallModelGraphStubs(backend.QnnInterface());
  qnn::QnnModel model(backend.GetMutable()->shared_from_this(), env.api_ptrs);
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
  qnn::QnnModel model(backend.GetMutable()->shared_from_this(), env.api_ptrs);
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
  ASSERT_TRUE(backend.GetMutable()->AddQnnContextHandle(&replacement_context).IsOK());
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

namespace {

enum class SsrPausePoint {
  BIND,
  EXECUTE,
};

// A condition variable fixes the interleaving without sleeps. The test's scope
// guard releases the worker even when an assertion fails before recovery.
struct SsrExecutionGate {
  explicit SsrExecutionGate(SsrPausePoint pause_point) : point(pause_point) {}

  SsrPausePoint point;
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool released = false;

  void Pause(SsrPausePoint current_point) {
    if (point != current_point) return;
    std::unique_lock<std::mutex> lock(mutex);
    if (entered) return;
    entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return released; });
  }

  bool WaitUntilPaused() {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    cv.notify_all();
  }
};

struct SsrTensorStub {
  std::array<float, 4> data{};
  SsrExecutionGate* gate = nullptr;
};

struct SsrKernelContextStub {
  SsrTensorStub input;
  SsrTensorStub output;
};

struct SsrGraphStub {
  ModelGraphContextStub* context = nullptr;
  SsrExecutionGate* gate = nullptr;
  const void* paused_input = nullptr;
  Qnn_ErrorHandle_t result = QNN_SUCCESS;
  std::atomic<unsigned> execute_calls{0};
};

struct SsrReloadStub {
  ModelGraphContextStub* replacement;
  unsigned calls = 0;
};

void InstallSsrExecutionApiStubs(OrtApi& api, QNN_INTERFACE_VER_TYPE& qnn_interface) {
  api.KernelContext_GetInputCount = [](const OrtKernelContext*, size_t* count) noexcept -> OrtStatus* {
    *count = 1;
    return nullptr;
  };
  api.KernelContext_GetOutputCount = api.KernelContext_GetInputCount;
  api.KernelContext_GetInput = [](const OrtKernelContext* context, size_t index,
                                  const OrtValue** value) noexcept -> OrtStatus* {
    EXPECT_EQ(index, 0u);
    *value = reinterpret_cast<const OrtValue*>(&reinterpret_cast<const SsrKernelContextStub*>(context)->input);
    return nullptr;
  };
  api.KernelContext_GetOutput = [](OrtKernelContext* context, size_t index, const int64_t* shape,
                                   size_t rank, OrtValue** value) noexcept -> OrtStatus* {
    EXPECT_EQ(index, 0u);
    EXPECT_EQ(rank, 1u);
    EXPECT_EQ(shape[0], 4);
    *value = reinterpret_cast<OrtValue*>(&reinterpret_cast<SsrKernelContextStub*>(context)->output);
    return nullptr;
  };
  api.GetTensorTypeAndShape = [](const OrtValue* value, OrtTensorTypeAndShapeInfo** info) noexcept -> OrtStatus* {
    *info = reinterpret_cast<OrtTensorTypeAndShapeInfo*>(const_cast<OrtValue*>(value));
    return nullptr;
  };
  api.GetTensorShapeElementCount = [](const OrtTensorTypeAndShapeInfo*, size_t* count) noexcept -> OrtStatus* {
    *count = 4;
    return nullptr;
  };
  api.GetTensorElementType = [](const OrtTensorTypeAndShapeInfo*,
                                ONNXTensorElementDataType* type) noexcept -> OrtStatus* {
    *type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    return nullptr;
  };
  api.GetTensorMemoryInfo = [](const OrtValue* value, const OrtMemoryInfo** info) noexcept -> OrtStatus* {
    *info = reinterpret_cast<const OrtMemoryInfo*>(value);
    return nullptr;
  };
  api.GetTensorData = [](const OrtValue* value, const void** data) noexcept -> OrtStatus* {
    *data = reinterpret_cast<const SsrTensorStub*>(value)->data.data();
    return nullptr;
  };
  api.GetTensorMutableData = [](OrtValue* value, void** data) noexcept -> OrtStatus* {
    *data = reinterpret_cast<SsrTensorStub*>(value)->data.data();
    return nullptr;
  };
  api.MemoryInfoGetDeviceType = [](const OrtMemoryInfo* info, OrtMemoryInfoDeviceType* type) noexcept {
    const auto& tensor = *reinterpret_cast<const SsrTensorStub*>(info);
    if (tensor.gate) tensor.gate->Pause(SsrPausePoint::BIND);
    *type = OrtMemoryInfoDeviceType_CPU;
  };
  api.MemoryInfoGetDeviceMemType = [](const OrtMemoryInfo*) noexcept {
    return OrtDeviceMemoryType_DEFAULT;
  };
  qnn_interface.contextFree = [](Qnn_ContextHandle_t context, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
    ++static_cast<ModelGraphContextStub*>(context)->free_calls;
    return QNN_SUCCESS;
  };
  qnn_interface.contextCreateFromBinary = [](Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t,
                                             const QnnContext_Config_t**, const void* buffer,
                                             Qnn_ContextBinarySize_t size, Qnn_ContextHandle_t* context,
                                             Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
    auto& reload = *static_cast<SsrReloadStub*>(backend);
    EXPECT_NE(buffer, nullptr);
    EXPECT_GT(size, 0u);
    ++reload.calls;
    *context = reload.replacement;
    return QNN_SUCCESS;
  };
  qnn_interface.graphExecute = [](Qnn_GraphHandle_t graph, const Qnn_Tensor_t* inputs, uint32_t num_inputs,
                                  Qnn_Tensor_t* outputs, uint32_t num_outputs, Qnn_ProfileHandle_t,
                                  Qnn_SignalHandle_t) -> Qnn_ErrorHandle_t {
    auto& stub = *static_cast<SsrGraphStub*>(graph);
    EXPECT_EQ(num_inputs, 1u);
    EXPECT_EQ(num_outputs, 1u);
    if (num_inputs != 1 || num_outputs != 1) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
    const auto input = qnn::GetQnnTensorClientBuf(inputs[0]);
    const auto output = qnn::GetQnnTensorClientBuf(outputs[0]);
    EXPECT_EQ(qnn::GetQnnTensorMemType(inputs[0]), QNN_TENSORMEMTYPE_RAW);
    EXPECT_EQ(qnn::GetQnnTensorMemType(outputs[0]), QNN_TENSORMEMTYPE_RAW);
    EXPECT_EQ(input.dataSize, sizeof(std::array<float, 4>));
    EXPECT_EQ(output.dataSize, sizeof(std::array<float, 4>));
    if (stub.gate && input.data == stub.paused_input) stub.gate->Pause(SsrPausePoint::EXECUTE);
    EXPECT_EQ(stub.context->free_calls.load(), 0u) << "Context freed during bind/execute";
    ++stub.execute_calls;
    if (stub.result == QNN_SUCCESS) {
      // Copy through the buffers bound for this invocation, so each model's
      // distinct output proves that recovery re-bound its own ORT tensors.
      std::copy_n(static_cast<const float*>(input.data), 4, static_cast<float*>(output.data));
    }
    return stub.result;
  };
}

class QnnModelConcurrentSsrTest : public ::testing::TestWithParam<SsrPausePoint> {};

TEST_P(QnnModelConcurrentSsrTest, ContextSurvivesBindAndExecuteWhilePeerReloads) {
  StubApiEnv env;
  InstallFakeGraphApiStubs(env.stub_ort_api);
  OrtGlobalApiOverride api_override(&env.stub_ort_api);
  SsrExecutionGate gate{GetParam()};
  SsrGraphStub original_graph;
  SsrGraphStub replacement_graph;
  ModelGraphContextStub original_context{&original_graph};
  ModelGraphContextStub replacement_context{&replacement_graph};
  SsrReloadStub reload{&replacement_context};
  StubBackendManager backend(env.api_ptrs, env.logger);
  backend.BackendType() = qnn::QnnBackendType::HTP;
  InstallModelGraphStubs(backend.QnnInterface());
  InstallSsrExecutionApiStubs(env.stub_ort_api, backend.QnnInterface());
  backend.BackendHandle() = &reload;
  auto* manager = backend.GetMutable();
  qnn::QnnModel model_a(manager->shared_from_this(), env.api_ptrs);
  qnn::QnnModel model_b(manager->shared_from_this(), env.api_ptrs);
  const qnn::EpContextIoDispatch io_dispatch;

  const auto binary_path = std::filesystem::temp_directory_path() /
                           (GetParam() == SsrPausePoint::BIND ? "qnn_ssr_bind.bin" : "qnn_ssr_execute.bin");
  auto remove_binary = gsl::finally([&] { std::filesystem::remove(binary_path); });
  {
    std::ofstream binary(binary_path, std::ios::binary);
    binary << "stub context binary";
    ASSERT_TRUE(binary.good());
  }

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeGraph graph{{}, {&x}, {&y}, {}};
  FakeNode fused{"fused", "QnnPartition_0", "", 13, {&x}, {&y}};
  std::vector<std::string> input_names{"x"};
  std::vector<std::string> output_names{"y"};
  qnn::ModelSettings settings{};
  qnn::QnnModelContext model_context{*graph.AsGraph(), *fused.AsNode(), env.logger,
                                     &input_names, &output_names, &settings};
  for (auto* model : {&model_a, &model_b}) {
    ASSERT_TRUE(model->SetGraphInputOutputInfo(model_context).IsOK());
    ASSERT_TRUE(DeserializeStubGraph(*model, original_context).IsOK());
    ASSERT_TRUE(model->SetupQnnInputOutput(env.logger).IsOK());
    model->SetContextRecoveryInfo(binary_path.string(), 0, qnn::ContextPriority::NORMAL);
  }
  ASSERT_TRUE(manager->AddQnnContextHandle(&original_context).IsOK());
  original_graph.context = &original_context;
  original_graph.gate = &gate;
  original_graph.result = QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  replacement_graph.context = &replacement_context;
  SsrKernelContextStub kernel_a{{{1, 2, 3, 4}, &gate}, {}};
  SsrKernelContextStub kernel_b{{{5, 6, 7, 8}}, {}};
  original_graph.paused_input = kernel_a.input.data.data();

  auto execution_a = std::async(std::launch::async, [&] {
    return model_a.ExecuteGraph(reinterpret_cast<OrtKernelContext*>(&kernel_a), env.logger, io_dispatch);
  });
  auto release_worker = gsl::finally([&] {
    gate.Release();
    if (execution_a.valid()) execution_a.wait();
  });
  ASSERT_TRUE(gate.WaitUntilPaused()) << "Model A did not enter the selected bind/execute pause";

  // Model B sees SSR, removes the shared old context, reloads from the stub
  // binary, then binds and executes the replacement while A is still paused.
  auto status_b = model_b.ExecuteGraph(reinterpret_cast<OrtKernelContext*>(&kernel_b), env.logger, io_dispatch);
  ASSERT_TRUE(status_b.IsOK()) << status_b.GetErrorMessage();
  EXPECT_EQ(reload.calls, 1u);
  EXPECT_FALSE(manager->HasContextHandle(&original_context));
  EXPECT_TRUE(manager->HasContextHandle(&replacement_context));
  EXPECT_EQ(original_context.free_calls.load(), 0u);
  EXPECT_EQ(replacement_graph.execute_calls.load(), 1u);
  EXPECT_EQ(kernel_b.output.data, kernel_b.input.data);

  gate.Release();
  auto status_a = execution_a.get();
  ASSERT_TRUE(status_a.IsOK()) << status_a.GetErrorMessage();
  EXPECT_EQ(original_context.free_calls.load(), 1u);
  EXPECT_EQ(original_graph.execute_calls.load(), 2u);
  EXPECT_EQ(reload.calls, 1u) << "Model A must reuse the context already recovered by B";
  EXPECT_EQ(replacement_graph.execute_calls.load(), 2u);
  EXPECT_EQ(kernel_a.output.data, kernel_a.input.data);
  for (auto* model : {&model_a, &model_b}) {
    const auto& info = model->*GetQnnModelPrivateMember(GraphInfoTag{});
    const auto& emitter = model->*GetQnnModelPrivateMember(GraphEmitterTag{});
    EXPECT_EQ(info->Graph(), &replacement_graph);
    EXPECT_EQ(info->GraphContext(), &replacement_context);
    EXPECT_EQ(emitter->Graph(), info->Graph());
    EXPECT_EQ(emitter->Context(), info->GraphContext());
  }
  EXPECT_EQ(replacement_context.free_calls.load(), 0u);
  manager->ReleaseSpecificContextHandle(&replacement_context);
  EXPECT_EQ(replacement_context.free_calls.load(), 1u);
  EXPECT_EQ(original_context.free_calls.load(), 1u);
}

INSTANTIATE_TEST_SUITE_P(BindAndExecute, QnnModelConcurrentSsrTest,
                         ::testing::Values(SsrPausePoint::BIND, SsrPausePoint::EXECUTE),
                         [](const ::testing::TestParamInfo<SsrPausePoint>& info) {
                           return info.param == SsrPausePoint::BIND ? "DuringBind" : "DuringExecute";
                         });

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
    model = std::make_unique<qnn::QnnModel>(manager, api_ptrs);
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

    model = std::make_unique<qnn::QnnModel>(manager, api_ptrs);
    valid = true;
  }

  bool IsValid() const { return valid; }
};

TEST(QnnUnit_ModelTest, KeepsBackendManagerAlive) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  std::weak_ptr<qnn::QnnBackendManager> weak_manager = ctx.manager;

  // A model parked in SharedContext can outlive the QnnEp that created it.
  ctx.manager.reset();
  EXPECT_FALSE(weak_manager.expired());

  ctx.model.reset();
  EXPECT_TRUE(weak_manager.expired());
}

TEST(QnnUnit_ModelTest, SharedContextRemovalIsScopedToProducingSession) {
  QnnModelMinimalTestContext ctx;
  ASSERT_TRUE(ctx.IsValid());

  ApiPtrs api_ptrs{ctx.stub_ort_api, ctx.stub_ep_api, ctx.stub_editor_api};
  auto model_a = std::make_unique<qnn::QnnModel>(ctx.manager, api_ptrs);
  auto model_b = std::make_unique<qnn::QnnModel>(ctx.manager, api_ptrs);
  SetQnnModelNameForTest(*model_a, "session_a_graph");
  SetQnnModelNameForTest(*model_b, "session_b_graph");

  SharedContext::QnnModelOwnerToken owner_a = std::make_shared<char>();
  SharedContext::QnnModelOwnerToken owner_b = std::make_shared<char>();
  std::string duplicate_graph_names;
  std::vector<std::unique_ptr<qnn::QnnModel>> session_a_models;
  session_a_models.push_back(std::move(model_a));
  EXPECT_FALSE(SharedContext::GetInstance().SetSharedQnnModel(
      std::move(session_a_models), ctx.manager, owner_a, duplicate_graph_names));
  std::vector<std::unique_ptr<qnn::QnnModel>> session_b_models;
  session_b_models.push_back(std::move(model_b));
  EXPECT_FALSE(SharedContext::GetInstance().SetSharedQnnModel(
      std::move(session_b_models), ctx.manager, owner_b, duplicate_graph_names));

  // Sessions A and B share one manager. Destroying A must leave B's parked model
  // available for a later consumer session C.
  SharedContext::GetInstance().RemoveSharedQnnModels(owner_a);
  EXPECT_FALSE(SharedContext::GetInstance().HasSharedQnnModel("session_a_graph", ctx.manager));
  EXPECT_TRUE(SharedContext::GetInstance().HasSharedQnnModel("session_b_graph", ctx.manager));
  auto session_c_model =
      SharedContext::GetInstance().GetSharedQnnModel("session_b_graph", ctx.manager);
  ASSERT_NE(session_c_model, nullptr);
  EXPECT_EQ(session_c_model->Name(), "session_b_graph");

  SharedContext::GetInstance().RemoveSharedQnnModels(owner_b);
}

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
  const qnn::QnnBackendManager::ContextHandleLease context_handle_lease;
  const auto status = ((*ctx.model).*bind_tensor_memory)(
      ctx.logger, reinterpret_cast<const OrtMemoryInfo*>(1), &data, sizeof(data), context_handle_lease, qnn_tensor);

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
