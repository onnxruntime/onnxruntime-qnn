// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared test utilities for QNN EP function-level / component-level unit tests.
//
// Requires QNN_EP_FUNCTION_LEVEL_UT (set by cmake when ENABLE_COVERAGE=1 and the
// QNN EP is built as a SHARED library). The test binary must link against
// onnxruntime_providers_qnn so EP-internal symbols are accessible.
//
// OrtGraph is provided via GetTestOrtGraph() (defined in qnn_mock_ort_graph.cc).
// The mock lives in a separate TU that includes abi_graph_types.h without EP
// headers, avoiding macro/symbol conflicts with ort_api.h.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_FUNCTION_LEVEL_UT

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "QnnInterface.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace test {

// Context for constructing a QnnModelWrapper in function-level unit tests.
//
//   ctx.input_info.indices  = {{"input0", 0}};   // declare graph inputs
//   ctx.output_info.indices = {{"output0", 0}};  // declare graph outputs
//   auto wrapper = ctx.CreateWrapper(settings);
struct QnnModelWrapperTestContext {
  // Mostly zero-init C API structs; a minimal subset of function pointers is
  // stubbed in the constructor to support the paths exercised by these tests.
  OrtApi stub_ort_api{};
  OrtEpApi stub_ep_api{};
  OrtModelEditorApi stub_editor_api{};

  QNN_INTERFACE_VER_TYPE qnn_interface;
  Qnn_BackendHandle_t backend_handle;
  QNN_INTERFACE_VER_TYPE qnn_validator_interface;  // null interface — no validator in unit tests
  Qnn_BackendHandle_t validator_backend_handle;    // must be a stable lvalue: QnnModelWrapper stores it as a const reference

  // Book-keeping for which tensors are graph inputs / outputs.
  qnn::GraphInputOutputInfo input_info;
  qnn::GraphInputOutputInfo output_info;

  QnnModelWrapperTestContext()
      : qnn_interface(QNN_INTERFACE_VER_TYPE_INIT),
        backend_handle(nullptr),
        qnn_validator_interface(QNN_INTERFACE_VER_TYPE_INIT),
        validator_backend_handle(nullptr) {
    // Stub initializer-query APIs so that IsConstantInput() safely returns false
    // without dereferencing null function pointers.
    stub_ort_api.Graph_GetNumInitializers = [](const OrtGraph*, size_t* num) noexcept -> OrtStatus* {
      *num = 0;
      return nullptr;
    };
    stub_ort_api.Graph_GetInitializers = [](const OrtGraph*, const OrtValueInfo**, size_t count) noexcept -> OrtStatus* {
      // This stub is paired with Graph_GetNumInitializers above (which always reports 0).
      // If a future test makes Graph_GetNumInitializers return non-zero, this stub must be
      // updated to actually populate the buffer; silently returning success would leave the
      // caller's buffer uninitialized and lead to UB.
      assert(count == 0 && "stub only supports empty initializer query; update Graph_GetInitializers when changing Graph_GetNumInitializers");
      (void)count;  // silence unused-variable warning in release builds where assert is no-op
      return nullptr;
    };
  }

  std::unique_ptr<qnn::QnnModelWrapper> CreateWrapper(
      const qnn::ModelSettings& settings,
      qnn::QnnBackendType backend_type = qnn::QnnBackendType::HTP) {
    ApiPtrs api_ptrs{stub_ort_api, stub_ep_api, stub_editor_api};
    return std::make_unique<qnn::QnnModelWrapper>(
        /*ort_graph=*/nullptr,
        api_ptrs,
        /*logger=*/nullptr,
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

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_FUNCTION_LEVEL_UT
