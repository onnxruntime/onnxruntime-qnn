// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared test utilities for QNN EP function-level / component-level unit tests.
//
// Requires QNN_EP_INTERNAL_SYMBOL_ACCESS (set by cmake when the test binary is
// link-time bound to the SHARED QNN EP library — currently ENABLE_COVERAGE=1
// on Linux x86_64). The macro is a build-system gate, not a production-source
// guard: when it is off, this header and all unit/ test bodies compile to empty
// translation units, so non-coverage builds see no undefined references.
//
// Class-specific fixtures (e.g. constructing a QnnModelWrapper with a fake
// graph + null logger) live next to the test file that owns them. This header
// only collects reusable pieces: MakeNullLogger(), OrtApi stub plumbing, and
// a real HTP backend handle.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <memory>
#include <stdexcept>
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

// MakeNullLogger
//
// Constructs an Ort::Logger with logger_=nullptr and cached_severity_level_=
// ORT_LOGGING_LEVEL_FATAL, so that all ORT_CXX_LOG calls (severity < FATAL)
// short-circuit on the cached-severity check without dereferencing the null
// logger pointer. QNN EP never logs at FATAL, so LogMessage is never reached.
//
// Background: the public ORT API exposes no way to construct an Ort::Logger
// without a real OrtLogger* obtained through EP plugin loading. The
// Logger(const OrtLogger*) constructor calls Logger_GetLoggingSeverityLevel
// on the pointer, which crashes on nullptr. The Logger(nullptr_t) constructor
// is a no-op that leaves cached_severity_level_=VERBOSE, which makes every
// log statement attempt the underlying LogMessage call.
//
// This helper uses memcpy to set cached_severity_level_=FATAL after default
// construction. Well-defined for trivially copyable types per [basic.types] /
// memcpy semantics.
inline Ort::Logger MakeNullLogger() {
  static_assert(sizeof(Ort::Logger) == 2 * sizeof(void*),
                "Ort::Logger layout changed — update MakeNullLogger()");
  Ort::Logger logger{std::nullptr_t{}};
  OrtLoggingLevel fatal = ORT_LOGGING_LEVEL_FATAL;
  std::memcpy(reinterpret_cast<char*>(&logger) + sizeof(const OrtLogger*),
              &fatal, sizeof(OrtLoggingLevel));
  return logger;
}

// OrtGlobalApiOverride
//
// RAII guard that replaces the global Ort::GetApi() with a caller-supplied
// OrtApi for the duration of the scope, then restores the original on
// destruction.
//
// Why this is needed: Ort::ConstNode / Ort::ConstValueInfo / Ort::ConstGraph
// wrappers call OrtApi function pointers via the global Ort::GetApi(), not
// through api_ptrs_. Tests that pass fake OrtNode*/OrtGraph* pointers to EP
// code must override the global so that wrapper calls route through stubs
// rather than the real ORT runtime (which dereferences fake pointers and
// SIGSEGVs). Process-wide global; gtest runs tests sequentially so this is
// safe, but do not use two overrides simultaneously in the same thread.
//
// Implementation note: uses Ort::detail::Global::Api(), which is declared in
// the public onnxruntime_cxx_api.h header (not a private "core/" include).
// Ort::InitApi() — the intended public setter — is only available when
// ORT_API_MANUAL_INIT is defined; ort_api.h suppresses that macro in
// unit-test builds so all TUs agree on static initialisation. This helper
// is test-only (gated by QNN_EP_INTERNAL_SYMBOL_ACCESS) and must be
// re-verified if ORT uplevels and changes the detail::Global layout.
class OrtGlobalApiOverride {
 public:
  explicit OrtGlobalApiOverride(const OrtApi* new_api) {
    original_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    Ort::detail::Global::Api(new_api);
  }
  ~OrtGlobalApiOverride() { Ort::detail::Global::Api(original_); }

  OrtGlobalApiOverride(const OrtGlobalApiOverride&) = delete;
  OrtGlobalApiOverride& operator=(const OrtGlobalApiOverride&) = delete;
  OrtGlobalApiOverride(OrtGlobalApiOverride&&) = delete;
  OrtGlobalApiOverride& operator=(OrtGlobalApiOverride&&) = delete;

 private:
  const OrtApi* original_ = nullptr;
};

// Reusable OrtApi stub tables for function-level unit tests.
//
// Holds the three stub structs (OrtApi / OrtEpApi / OrtModelEditorApi) that any
// code interacting with ORT through ApiPtrs needs. Tests assign individual
// function-pointer members directly (e.g. ctx.stub_ort_api.GetTensorData = ...).
//
// Initializer-query stubs are installed in the constructor so that paths like
// QnnModelWrapper::IsConstantInput() safely return false on graphs with no
// initializers — the default fixture for almost every test. Tests that need
// non-zero initializers replace these two stubs before constructing the wrapper.
//
// MakeApiPtrs() returns an ApiPtrs view over the three stub tables AND verifies
// that the initializer-query stubs are still installed (a test that wholesale
// resets stub_ort_api must re-add them, otherwise QnnModelWrapper SIGSEGVs at
// the first initializer query). Throwing std::logic_error fails the test rather
// than the process; assert() would be stripped by NDEBUG (CMake RelWithDebInfo,
// the coverage build's config).
struct OrtApiStubContext {
  OrtApi stub_ort_api{};
  OrtEpApi stub_ep_api{};
  OrtModelEditorApi stub_editor_api{};

  OrtApiStubContext() {
    stub_ort_api.Graph_GetNumInitializers = [](const OrtGraph*, size_t* num) noexcept -> OrtStatus* {
      *num = 0;
      return nullptr;
    };
    stub_ort_api.Graph_GetInitializers = [](const OrtGraph*, const OrtValueInfo**, size_t count) noexcept -> OrtStatus* {
      // Pairs with Graph_GetNumInitializers above which always reports 0. Tests
      // that need non-zero initializers must replace this stub before constructing
      // a wrapper.
      // Note: ORT_ENFORCE / assert are not used here because this lambda is noexcept —
      // throwing or calling abort() from a noexcept function terminates the process
      // rather than failing the test case. The invariant is enforced by MakeApiPtrs().
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

// Context for tests that need a real QNN HTP backend (e.g., ValidateQnnNode).
// Loads libQnnHtp.so at construction time via dlopen and creates a live backend handle.
//
// libQnnHtp.so is part of the QAIRT SDK that ships with every supported CI image,
// so a missing library is a CI configuration error rather than a normal condition.
// Tests should ASSERT_TRUE(backend.IsValid()) so a missing SDK fails the test
// loudly instead of silently skipping.
//
// Note: this helper only produces a Qnn_BackendHandle_t. It does NOT create a QNN
// context/session (no contextCreate call) and does NOT create a graph. The validation
// path (backendValidateOpConfig) only needs the backend handle, so a session is
// unnecessary for the current set of unit tests. Tests that need a real QNN
// context/session, graph, or graph execution must add their own helper.
//
// On Linux x86-64 (the unit-test host), libQnnHtp.so loads successfully and supports
// graph-validation calls. Graph execution requires HTP hardware and is not exercised here.
//
// Usage:
//   QnnRealHtpBackendContext backend;
//   ASSERT_TRUE(backend.IsValid()) << "libQnnHtp.so not available";
//   ctx.qnn_interface  = backend.qnn_interface;
//   ctx.backend_handle = backend.backend_handle;
struct QnnRealHtpBackendContext {
  QNN_INTERFACE_VER_TYPE qnn_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle = nullptr;

  QnnRealHtpBackendContext() {
#ifndef _WIN32
    lib_handle_ = ::dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
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

  ~QnnRealHtpBackendContext() {
#ifndef _WIN32
    if (initialized_ && qnn_interface.backendFree) {
      qnn_interface.backendFree(backend_handle);
    }
    if (lib_handle_) ::dlclose(lib_handle_);
#endif
  }

  bool IsValid() const { return initialized_; }

  // Non-copyable / non-movable — holds raw lib handle and backend handle.
  // (User-declared dtor + deleted copy already make this class non-movable
  //  implicitly — std::move(x) won't compile. Explicit move-deletes below
  //  are for self-documentation per Rule of Five.)
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
