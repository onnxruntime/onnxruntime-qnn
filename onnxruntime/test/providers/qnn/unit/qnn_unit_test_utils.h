// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared test utilities for QNN EP function-level / component-level unit tests.
//
// Requires QNN_EP_FUNCTION_LEVEL_UT (set by cmake when ENABLE_COVERAGE=1 and the
// QNN EP is built as a SHARED library). The test binary must link against
// onnxruntime_providers_qnn so EP-internal symbols are accessible.
//
// QnnModelWrapper is constructed with ort_graph=nullptr; tests stub the
// OrtApi entry points needed for the path under test (see CreateWrapper).

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_FUNCTION_LEVEL_UT

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

  // Stable lvalues passed to the production QnnModelWrapper constructor.
  // The constructor stores pointers to these members; they must outlive any
  // wrapper created by CreateWrapper().
  //
  // null_logger_: default-constructed Ort::Logger has a null internal OrtLogger*;
  //   ORT_CXX_LOG_PTR calls IsNullLogger() before logging, so logging is safely skipped.
  // fake_graph_sentinel_: an int used as a stable address; stubs receive this pointer
  //   but never dereference it (same pattern as g_type_info_sentinel etc.).
  Ort::Logger null_logger_{MakeNullLogger()};
  int fake_graph_sentinel_{};

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
    //
    // INVARIANT: tests that wholesale reset stub_ort_api MUST re-add these two stubs
    // before calling CreateWrapper(). Otherwise QnnModelWrapper forwards the null
    // ort_graph to a zero-initialized function pointer on initializer queries
    // (FindInitializer / IsConstantInput / GetConstantTensor) and SIGSEGVs.
    // CreateWrapper() asserts both pointers are non-null at construction time as a
    // defense against this footgun.
    stub_ort_api.Graph_GetNumInitializers = [](const OrtGraph*, size_t* num) noexcept -> OrtStatus* {
      *num = 0;
      return nullptr;
    };
    stub_ort_api.Graph_GetInitializers = [](const OrtGraph*, const OrtValueInfo**, size_t count) noexcept -> OrtStatus* {
      // This stub only supports the empty-initializer case, paired with Graph_GetNumInitializers
      // above which always reports 0. Tests that need non-zero initializers must replace this
      // stub before calling CreateWrapper.
      // Note: ORT_ENFORCE / assert are not used here because this lambda is noexcept — throwing
      // or calling abort() from a noexcept function terminates the process rather than failing
      // the test case. The invariant is enforced by the companion stub above.
      (void)count;
      return nullptr;
    };
  }

  std::unique_ptr<qnn::QnnModelWrapper> CreateWrapper(
      const qnn::ModelSettings& settings,
      qnn::QnnBackendType backend_type = qnn::QnnBackendType::HTP) {
    // Fail fast at construction (rather than silently SIGSEGV at first initializer query)
    // when these stubs were cleared by a test resetting stub_ort_api wholesale.
    // See the INVARIANT comment in QnnModelWrapperTestContext() above.
    //
    // assert() is intentionally NOT used: CMake's RelWithDebInfo (the coverage build's
    // config) defines NDEBUG, which strips assert() — disabling the guard in precisely
    // the environment these tests run in. throw fires in every build configuration.
    if (stub_ort_api.Graph_GetNumInitializers == nullptr ||
        stub_ort_api.Graph_GetInitializers == nullptr) {
      throw std::logic_error(
          "Graph_GetNumInitializers / Graph_GetInitializers stubs missing "
          "— re-add them after resetting stub_ort_api");
    }
    ApiPtrs api_ptrs{stub_ort_api, stub_ep_api, stub_editor_api};
    // fake_graph_sentinel_ is a stable int member used as the graph address.
    // Stubs receive this pointer but never dereference it.
    const OrtGraph& fake_graph = *reinterpret_cast<const OrtGraph*>(&fake_graph_sentinel_);
    return std::make_unique<qnn::QnnModelWrapper>(
        fake_graph,
        api_ptrs,
        null_logger_,
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

// Context for tests that need a real QNN HTP backend (e.g., ValidateQnnNode).
// Loads libQnnHtp.so at construction time via dlopen and creates a live backend handle.
// Use IsValid() before calling — if the library cannot be loaded the context is a no-op
// and GTest tests should call GTEST_SKIP() rather than FAIL().
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
//   if (!backend.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
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

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_FUNCTION_LEVEL_UT
