// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Backend / wrapper test contexts for QNN EP function-level unit tests.
//
//   OpBuilderTestContext       — stub-backed wrapper context (no live backend)
//   QnnRealCpuBackendContext         — dlopen-based libQnnCpu, exposes interface + backend handle
//   QnnRealCpuBackendManagerContext  — full CPU backend via QnnBackendManager (has context handle)
//   QnnRealHtpBackendManagerContext  — full HTP backend via QnnBackendManager (has context handle)
//
// The Manager variants give a usable Qnn_ContextHandle_t suitable for
// CreateQnnGraph + ComposeQnnGraph (Path E1 JSON snapshot tests).

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"
#include "test/util/include/test/test_environment.h"

// FakeGraph / FakeNode / FakeValueInfo primitives — OpBuilderTestContext uses a
// plain int as OrtGraph* sentinel (never dereferenced), so we don't need FakeGraph
// here. Kept as an include for tests that DO need the fake primitives directly.
#include "test/providers/qnn/test_infra/qnn_fake_ort_graph.h"

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Default stubs for OpBuilderTestContext
//
// Registered in the default constructor so any test using
// OpBuilderTestContext can safely call FindInitializer / IsConstantInput
// without crashing on a null function pointer. They report 0 initializers,
// which makes IsConstantInput return false — the correct behaviour for tests
// that do not need to simulate initializer inputs.
// ---------------------------------------------------------------------------
namespace {

OrtStatus* DefaultStubGetNumInitializersZero(const OrtGraph*, size_t* count) noexcept {
  *count = 0;
  return nullptr;
}

OrtStatus* DefaultStubGetInitializersEmpty(const OrtGraph*, const OrtValueInfo**, size_t) noexcept {
  return nullptr;
}

}  // namespace

// Context for constructing a QnnModelWrapper in function-level unit tests.
//
//   ctx.input_info.indices  = {{"input0", 0}};   // declare graph inputs
//   ctx.output_info.indices = {{"output0", 0}};  // declare graph outputs
//   auto wrapper = ctx.CreateWrapper(settings);
struct OpBuilderTestContext {
  // Zero-init C API structs. All function pointers are null, which is safe for
  // tests that exercise tensor-metadata logic without invoking any ORT or EP APIs.
  OrtApi stub_ort_api{};
  OrtEpApi stub_ep_api{};
  OrtModelEditorApi stub_editor_api{};

  QNN_INTERFACE_VER_TYPE qnn_interface;
  Qnn_BackendHandle_t backend_handle;

  // Validator interface/handle — required by QnnModelWrapper ctor (added on origin/main).
  // No live validator backend in stub-only unit tests; pass zero-init lvalues so the
  // wrapper stores stable references.
  QNN_INTERFACE_VER_TYPE qnn_validator_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t validator_backend_handle = nullptr;

  qnn::GraphInputOutputInfo input_info;
  qnn::GraphInputOutputInfo output_info;

  // Null-safe logger constructed via MakeNullLogger() (declared in qnn_unit_test_utils.h,
  // which includes this header AFTER defining the helper). Cached severity is FATAL,
  // so every ORT_CXX_LOG call short-circuits without dereferencing the null OrtLogger*.
  Ort::Logger ort_logger{MakeNullLogger()};

  // Stable lvalue used as OrtGraph* argument to QnnModelWrapper. The wrapper only
  // flows this pointer through OrtApi function pointers (never dereferences it as a
  // concrete OrtGraph), so any stable address works — the int here is just for its
  // address, never read.
  int fake_graph_sentinel_{};

  OpBuilderTestContext()
      : qnn_interface(QNN_INTERFACE_VER_TYPE_INIT), backend_handle(nullptr) {
    // Default stubs: report 0 initializers so FindInitializer / IsConstantInput
    // return safely without crashing on a null function pointer.
    stub_ort_api.Graph_GetNumInitializers = DefaultStubGetNumInitializersZero;
    stub_ort_api.Graph_GetInitializers = DefaultStubGetInitializersEmpty;
  }

  std::unique_ptr<qnn::QnnModelWrapper> CreateWrapper(
      const qnn::ModelSettings& settings,
      qnn::QnnBackendType backend_type = qnn::QnnBackendType::HTP) {
    ApiPtrs api_ptrs{stub_ort_api, stub_ep_api, stub_editor_api};
    const OrtGraph& fake_graph = *reinterpret_cast<const OrtGraph*>(&fake_graph_sentinel_);
    return std::make_unique<qnn::QnnModelWrapper>(
        fake_graph,
        api_ptrs,
        ort_logger,
        qnn_interface,
        backend_handle,
        qnn_validator_interface,
        validator_backend_handle,
        input_info,
        output_info,
        backend_type,
        settings);
  }
};

// Context for tests that need a real QNN CPU backend (e.g., ValidateQnnNode).
// Loads libQnnCpu.so at construction time via dlopen and creates a live backend handle.
// Use IsValid() before calling — if the library cannot be loaded the context is a no-op
// and GTest tests should call GTEST_SKIP() rather than FAIL().
//
// Usage:
//   QnnRealCpuBackendContext backend;
//   if (!backend.IsValid()) GTEST_SKIP() << "libQnnCpu.so not available";
//   ctx.qnn_interface  = backend.qnn_interface;
//   ctx.backend_handle = backend.backend_handle;
struct QnnRealCpuBackendContext {
  QNN_INTERFACE_VER_TYPE qnn_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle = nullptr;

  QnnRealCpuBackendContext() {
#ifndef _WIN32
    lib_handle_ = ::dlopen("libQnnCpu.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib_handle_) return;

    using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
    auto get_providers = reinterpret_cast<GetProvidersFn>(
        ::dlsym(lib_handle_, "QnnInterface_getProviders"));
    if (!get_providers) return;

    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    if (get_providers(&providers, &count) != QNN_SUCCESS || count == 0 || !providers) return;

    qnn_interface = providers[0]->QNN_INTERFACE_VER_NAME;
    if (!qnn_interface.backendCreate) return;

    if (qnn_interface.backendCreate(nullptr, nullptr, &backend_handle) != QNN_BACKEND_NO_ERROR) {
      backend_handle = nullptr;
      return;
    }
    initialized_ = true;
#endif
  }

  ~QnnRealCpuBackendContext() {
#ifndef _WIN32
    if (initialized_ && qnn_interface.backendFree) {
      qnn_interface.backendFree(backend_handle);
    }
    if (lib_handle_) ::dlclose(lib_handle_);
#endif
  }

  bool IsValid() const { return initialized_; }

  // Non-copyable / non-movable — holds raw lib handle and backend handle.
  QnnRealCpuBackendContext(const QnnRealCpuBackendContext&) = delete;
  QnnRealCpuBackendContext& operator=(const QnnRealCpuBackendContext&) = delete;

 private:
  void* lib_handle_ = nullptr;
  bool initialized_ = false;
};

// Context for tests that need a real QNN CPU backend with a live context handle
// suitable for ComposeQnnGraph (Path E1 JSON snapshot tests).
//
// Drives ORT's `qnn::QnnBackendManager` to create the backend AND a usable
// context. Output is the in-memory `QnnJSONGraph` populated by
// ComposeQnnGraph(true) (read via wrapper.GetQnnJSONGraph()).
//
// Usage:
//   QnnRealCpuBackendManagerContext cpu;
//   if (!cpu.IsValid()) GTEST_SKIP() << "libQnnCpu.so not available";
//   auto wrapper = MakeSnapshotWrapperJson(ctx, cpu, {"in"}, {"out"});
//   ... drive op-builder ...
//   wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true);
//   AssertSnapshotJson(*wrapper, "MyTest", "builder/opbuilder/clip");
struct QnnRealCpuBackendManagerContext {
  QNN_INTERFACE_VER_TYPE qnn_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle = nullptr;
  Qnn_ContextHandle_t context_handle = nullptr;

  QnnRealCpuBackendManagerContext() {
#ifndef _WIN32
    qnn::QnnBackendManagerConfig cfg;
    cfg.backend_path = "libQnnCpu.so";
    cfg.profiling_level_etw = qnn::ProfilingLevel::OFF;
    cfg.profiling_level = qnn::ProfilingLevel::OFF;
    cfg.context_priority = qnn::ContextPriority::NORMAL;
    cfg.device_id = 0;
    cfg.htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
    cfg.soc_model = 0;
    cfg.skip_qnn_version_check = true;

    ApiPtrs api_ptrs{stub_ort_api_, stub_ep_api_, stub_editor_api_};
    manager_ = qnn::QnnBackendManager::Create(cfg, api_ptrs, ort_logger_);
    if (!manager_) return;

    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
    auto status = manager_->SetupBackend(/*load_from_cached_context=*/false,
                                         /*need_load_system_lib=*/false,
                                         /*share_ep_contexts=*/false,
                                         /*htp_share_resource_optimization=*/-1,
                                         /*enable_file_mapped_weights=*/false,
                                         /*rpcmem_library=*/nullptr,
                                         dummy_map);
    if (!status.IsOK()) {
      manager_.reset();
      return;
    }

    qnn_interface = manager_->GetQnnInterface();
    backend_handle = manager_->GetQnnBackendHandle();
    context_handle = manager_->GetQnnContext(0);
    initialized_ = true;
#endif
  }

  ~QnnRealCpuBackendManagerContext() = default;

  bool IsValid() const { return initialized_; }

  QnnRealCpuBackendManagerContext(const QnnRealCpuBackendManagerContext&) = delete;
  QnnRealCpuBackendManagerContext& operator=(const QnnRealCpuBackendManagerContext&) = delete;

 private:
  bool initialized_ = false;

  // Lifetimes: manager_ stores api_ptrs by reference, so these must outlive manager_.
  OrtApi stub_ort_api_{};
  OrtEpApi stub_ep_api_{};
  OrtModelEditorApi stub_editor_api_{};
  Ort::Logger ort_logger_{MakeNullLogger()};

  std::shared_ptr<qnn::QnnBackendManager> manager_;
};

// Context for tests that need a real QNN HTP backend with a live context handle
// suitable for ComposeQnnGraph. Use this when the dtype under test is HTP-only
// (FP16, U16 / U8 mixed quantization) and libQnnCpu would reject the op
// (graphAddNode rc 3110).
//
// Note: HTP triggers `ProcessBF16Conversions` for FP32 tensors, which inserts
// Convert ops and mutates the op list. Use the CPU variant for FP32 cases.
//
// Usage:
//   QnnRealHtpBackendManagerContext htp;
//   if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
//   auto wrapper = MakeSnapshotWrapperHtpJson(ctx, htp, {"in"}, {"out"});
struct QnnRealHtpBackendManagerContext {
  QNN_INTERFACE_VER_TYPE qnn_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle = nullptr;
  Qnn_ContextHandle_t context_handle = nullptr;

  QnnRealHtpBackendManagerContext() {
#ifndef _WIN32
    qnn::QnnBackendManagerConfig cfg;
    cfg.backend_path = "libQnnHtp.so";
    cfg.profiling_level_etw = qnn::ProfilingLevel::OFF;
    cfg.profiling_level = qnn::ProfilingLevel::OFF;
    cfg.context_priority = qnn::ContextPriority::NORMAL;
    cfg.device_id = 0;
    cfg.htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
    cfg.soc_model = 0;
    cfg.skip_qnn_version_check = true;

    ApiPtrs api_ptrs{stub_ort_api_, stub_ep_api_, stub_editor_api_};
    manager_ = qnn::QnnBackendManager::Create(cfg, api_ptrs, ort_logger_);
    if (!manager_) return;

    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
    auto status = manager_->SetupBackend(/*load_from_cached_context=*/false,
                                         /*need_load_system_lib=*/false,
                                         /*share_ep_contexts=*/false,
                                         /*htp_share_resource_optimization=*/-1,
                                         /*enable_file_mapped_weights=*/false,
                                         /*rpcmem_library=*/nullptr,
                                         dummy_map);
    if (!status.IsOK()) {
      manager_.reset();
      return;
    }

    qnn_interface = manager_->GetQnnInterface();
    backend_handle = manager_->GetQnnBackendHandle();
    context_handle = manager_->GetQnnContext(0);
    initialized_ = true;
#endif
  }

  ~QnnRealHtpBackendManagerContext() = default;

  bool IsValid() const { return initialized_; }

  QnnRealHtpBackendManagerContext(const QnnRealHtpBackendManagerContext&) = delete;
  QnnRealHtpBackendManagerContext& operator=(const QnnRealHtpBackendManagerContext&) = delete;

 private:
  bool initialized_ = false;

  OrtApi stub_ort_api_{};
  OrtEpApi stub_ep_api_{};
  OrtModelEditorApi stub_editor_api_{};
  Ort::Logger ort_logger_{MakeNullLogger()};

  std::shared_ptr<qnn::QnnBackendManager> manager_;
};

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
