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

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace test {

// MakeNullLogger
//
// Constructs an Ort::Logger whose cached severity is FATAL, so every ORT_CXX_LOG
// call short-circuits on the severity gate and never dereferences the null logger
// pointer. QNN EP never logs at FATAL, so LogMessage is unreachable.
//
// The public ORT API can't build an Ort::Logger without a real OrtLogger* from EP
// plugin loading: Logger(const OrtLogger*) crashes on nullptr, and Logger(nullptr_t)
// leaves the cached severity at VERBOSE (every log then attempts LogMessage). So we
// default-construct and memcpy FATAL into the cached-severity field.
inline Ort::Logger MakeNullLogger() {
  static_assert(sizeof(Ort::Logger) == 2 * sizeof(void*),
                "Ort::Logger layout changed — update MakeNullLogger()");
  Ort::Logger logger{std::nullptr_t{}};
  OrtLoggingLevel fatal = ORT_LOGGING_LEVEL_FATAL;
  std::memcpy(reinterpret_cast<char*>(&logger) + sizeof(const OrtLogger*),
              &fatal, sizeof(OrtLoggingLevel));
  return logger;
}

// StubApiEnv
//
// Convenience bundle of zero-initialised OrtApi / OrtEpApi / OrtModelEditorApi
// stubs plus an ApiPtrs view and a null logger. Used by tests that exercise
// EP code paths which only need the API tables for type-erasure and do not
// dispatch through them (or that pass them to functions whose code paths
// avoid every uninitialised function pointer).
//
// Non-copyable / non-movable because ApiPtrs stores references to the stub
// API tables.
struct StubApiEnv {
  OrtApi stub_ort_api{};
  OrtEpApi stub_ep_api{};
  OrtModelEditorApi stub_editor_api{};
  Ort::Logger logger{MakeNullLogger()};
  ApiPtrs api_ptrs{stub_ort_api, stub_ep_api, stub_editor_api};

  StubApiEnv() = default;
  StubApiEnv(const StubApiEnv&) = delete;
  StubApiEnv& operator=(const StubApiEnv&) = delete;
};

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

// ---------------------------------------------------------------------------
// StubBackendManager — a QnnBackendManager whose QNN interface can be stubbed
// ---------------------------------------------------------------------------
//
// Why this exists
//
// QnnModelWrapper reaches the QNN interface, backend handles, and backend type
// through a `const QnnBackendManager&` (it used to take them as separate
// constructor arguments). Those live in QnnBackendManager's private section with
// no setter, so the mock layer this suite depends on — "zero-init a
// QNN_INTERFACE_VER_TYPE and override only the function pointers the test path
// exercises" — is no longer reachable through the public API, and
// core/providers/qnn/ is not modified for testing.
//
// The accessors below hand out mutable references to those private members.
// They are deliberately declared here and *defined* in qnn_unit_test_utils.cc,
// because reaching the private members needs an explicit-instantiation +
// friend-injection trick that must exist in exactly one translation unit:
// repeating an explicit instantiation definition of the same specialization
// across translation units is IFNDR ([temp.explicit]/13), and this header is
// included by every unit/*_test.cc. Confining the machinery to the .cc keeps
// it to a single definition; see that file for how it works and why it is
// standard-sanctioned rather than a `#define private public` ODR violation.
//
// Caveats — read before extending:
//   - Test-only. Gated by QNN_EP_INTERNAL_SYMBOL_ACCESS, so it exists only in
//     the coverage build (Linux x86_64, GCC/Clang).
//   - The machinery in the .cc names QnnBackendManager's private members
//     directly. If a member is renamed or retyped, the build breaks there and
//     the tag must be updated.
//   - Adding an accessor for another private member means editing both files:
//     declare it here, define it (plus its tag and instantiation) there.
//   - Prefer the public API where one exists. QnnBackendManager::SetQnnBackendType()
//     is the public path when you hold a QNN backend *id*; BackendType() below
//     exists because the fixtures take a qnn::QnnBackendType directly (and
//     QnnBackendType::HTP_FP16 has no corresponding backend id).

// Owns a QnnBackendManager created through the public Create() factory with no
// backend library loaded (nothing is dlopen'd, SetupBackend is never called), and
// exposes the pieces QnnModelWrapper reads as mutable references so tests can
// stub them.
//
// Note that a freshly created manager reports QNN_HTP_DEVICE_ARCH_NONE from
// GetHtpArch() even when QnnBackendManagerConfig::htp_arch is set — the arch a
// caller supplies is only copied into the internal holder by
// SetupDeviceAndContext(). Use HtpArch() to simulate a set-up backend.
//
// Non-copyable / non-movable: the manager stores a reference-holding ApiPtrs and
// a pointer to the logger, so both must outlive it.
class StubBackendManager {
 public:
  StubBackendManager(const ApiPtrs& api_ptrs, const Ort::Logger& logger) {
    qnn::QnnBackendManagerConfig cfg{};  // value-init zeroes every field
    cfg.profiling_level = qnn::ProfilingLevel::OFF;
    cfg.profiling_level_etw = qnn::ProfilingLevel::OFF;
    cfg.context_priority = qnn::ContextPriority::NORMAL;
    cfg.htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
    cfg.soc_model = QNN_SOC_MODEL_UNKNOWN;
    cfg.skip_qnn_version_check = true;
    manager_ = qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StubBackendManager);

  const qnn::QnnBackendManager* Get() const { return manager_.get(); }

  // Defined in qnn_unit_test_utils.cc — see the comment block above.
  QNN_INTERFACE_VER_TYPE& QnnInterface();
  Qnn_BackendHandle_t& BackendHandle();
  QNN_INTERFACE_VER_TYPE& ValidatorInterface();
  Qnn_BackendHandle_t& ValidatorBackendHandle();
  qnn::QnnBackendType& BackendType();
  QnnHtpDevice_Arch_t& HtpArch();

 private:
  std::shared_ptr<qnn::QnnBackendManager> manager_;
};

// Context for tests that need a real QNN HTP backend (e.g., ValidateQnnNode).
// Loads libQnnHtp.so via dlopen at construction and creates a live backend handle.
//
// This helper produces ONLY a Qnn_BackendHandle_t. It does NOT create a QNN
// context/session (no contextCreate) and does NOT create a graph — the validation
// path (backendValidateOpConfig) only needs the backend handle. Tests that need a
// real context/session, graph, or graph execution must add their own helper.
//
// On Linux x86-64 (the unit-test host) libQnnHtp.so loads and supports graph
// validation; graph execution requires HTP hardware and is not exercised here.
// See the mock-strategy table in unit/README.md for when to ASSERT_TRUE(IsValid())
// vs GTEST_SKIP().
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
