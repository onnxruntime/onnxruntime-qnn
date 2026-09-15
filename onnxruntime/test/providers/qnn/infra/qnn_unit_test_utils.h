// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Umbrella include for QNN EP component-tier test utilities.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <stdexcept>
#include <vector>

#include "QnnInterface.h"

#include "core/providers/qnn/ort_api.h"
#include "test/providers/qnn/infra/backend_contexts.h"
#include "test/providers/qnn/infra/mock_init_registry.h"
#include "test/providers/qnn/infra/mock_node_unit.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"
#include "test/providers/qnn/infra/stub_backend_manager.h"

namespace onnxruntime {
namespace test {

inline const char* QnnHtpBackendLibraryName() {
#ifdef _WIN32
  return "QnnHtp.dll";
#else
  return "libQnnHtp.so";
#endif
}

inline const char* QnnIrBackendLibraryName() {
#ifdef _WIN32
  return "QnnIr.dll";
#else
  return "libQnnIr.so";
#endif
}

inline const char* QnnSaverBackendLibraryName() {
#ifdef _WIN32
  return "QnnSaver.dll";
#else
  return "libQnnSaver.so";
#endif
}

inline std::basic_string<ORTCHAR_T> QnnHtpBackendLibraryPath() {
#ifdef _WIN32
  return ORT_TSTR("QnnHtp.dll");
#else
  return ORT_TSTR("libQnnHtp.so");
#endif
}

// Reusable OrtApi stub tables for function-level unit tests.
struct OrtApiStubContext {
  OrtApi stub_ort_api{};
  OrtEpApi stub_ep_api{};
  OrtModelEditorApi stub_editor_api{};

  OrtApiStubContext() {
    stub_ort_api.GetExperimentalFunction = [](const char*) noexcept -> OrtExperimentalFnPtr {
      return nullptr;
    };
    stub_ort_api.Graph_GetNumInitializers = [](const OrtGraph*, size_t* num) noexcept -> OrtStatus* {
      *num = 0;
      return nullptr;
    };
    stub_ort_api.Graph_GetInitializers = [](const OrtGraph*, const OrtValueInfo**, size_t count) noexcept -> OrtStatus* {
      // Pairs with Graph_GetNumInitializers above which always reports 0. Tests
      // that need non-zero initializers must replace this stub before constructing
      // a wrapper.
      (void)count;
      return nullptr;
    };
  }

  ApiPtrs MakeApiPtrs() const {
    if (stub_ort_api.Graph_GetNumInitializers == nullptr ||
        stub_ort_api.Graph_GetInitializers == nullptr) {
      throw std::logic_error(
          "Graph_GetNumInitializers / Graph_GetInitializers stubs missing "
          "— re-add them after resetting stub_ort_api");
    }
    return ApiPtrs{stub_ort_api, stub_ep_api, stub_editor_api};
  }
};

// Context for tests that need a real QNN HTP backend handle for validation.
struct QnnRealHtpBackendContext {
  QNN_INTERFACE_VER_TYPE qnn_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle = nullptr;

  QnnRealHtpBackendContext() {
    void* lib_handle = nullptr;
    if (!OrtLoadDynamicLibrary(QnnHtpBackendLibraryPath(), /*global_symbols=*/true, &lib_handle).IsOK()) {
      return;
    }
    lib_handle_ = lib_handle;

    using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
    void* get_providers_symbol = nullptr;
    if (!OrtGetSymbolFromLibrary(lib_handle_, "QnnInterface_getProviders", &get_providers_symbol).IsOK()) {
      return;
    }
    auto get_providers = reinterpret_cast<GetProvidersFn>(get_providers_symbol);
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
  }

  ~QnnRealHtpBackendContext() {
    if (initialized_ && qnn_interface.backendFree) {
      qnn_interface.backendFree(backend_handle);
    }
    if (lib_handle_) {
      (void)OrtUnloadDynamicLibrary(lib_handle_);
    }
  }

  bool IsValid() const { return initialized_; }

  QnnRealHtpBackendContext(const QnnRealHtpBackendContext&) = delete;
  QnnRealHtpBackendContext& operator=(const QnnRealHtpBackendContext&) = delete;
  QnnRealHtpBackendContext(QnnRealHtpBackendContext&&) = delete;
  QnnRealHtpBackendContext& operator=(QnnRealHtpBackendContext&&) = delete;

 private:
  void* lib_handle_ = nullptr;
  bool initialized_ = false;
};

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
