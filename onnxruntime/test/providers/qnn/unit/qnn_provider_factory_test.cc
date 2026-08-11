// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for QnnEpFactory (qnn_provider_factory.cc).
//
// All tests here run without loading any real QNN backend shared library, and
// without constructing a real QnnEp instance. They exercise:
//   - Basic getters (name/vendor/version/vendor-id/is-stream-aware) and the
//     ctor's CreateMemoryInfo_V2 wiring.
//   - Trivial factory methods that never dispatch through the OrtApi tables
//     (CreateDataTransferImpl, ReleaseEpImpl(nullptr)).
//   - CreateEpFactories / ReleaseEpFactory extern-"C" entry points, focused
//     on argument-validation and version-check error paths that do NOT
//     reach the QnnEpFactory ctor's Ort::InitApi() (which would clobber the
//     global api table that gtest itself uses).
//   - GetSupportedDevicesImpl device filtering + NPU-synthesis fallback.
//   - GetHardwareDeviceIncompatibilityDetailsImpl / ValidateCompiledModel-
//     CompatibilityInfoImpl error paths that stop before std::make_unique<QnnEp>.
//
// Real-QnnEp paths (autoep CreateEpImpl clone + AddSessionConfigEntry, and
// Validate / Incompatibility temp-EP construction) are covered by the
// QnnUnit_ProviderFactoryHtpTest fixture at the end of this file; those
// tests GTEST_SKIP() when libQnnHtp.so is unavailable. This mirrors
// QnnUnit_BackendManagerHtpTest — component-level tests with a real backend
// load but no ORT session.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_provider_factory.h"

#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

// Public entry points from qnn_provider_factory.cc (extern "C" linkage,
// re-declared here rather than via a private header).
extern "C" {
OrtStatus* CreateEpFactories(const char* registration_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories);
OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
}

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Fake opaque handles.
//
// The ORT public C API treats these as opaque struct pointers; QNN EP never
// dereferences them itself — every access goes through an OrtApi/OrtEpApi
// function pointer. Handing the stubs distinguishable "tag" pointers makes
// it possible to identify individual devices in expectations without a full
// mock framework.
// ---------------------------------------------------------------------------

static OrtHardwareDevice* MakeFakeHwDevice(uintptr_t tag) {
  return reinterpret_cast<OrtHardwareDevice*>(tag);
}

// Non-null pointer used as a placeholder OrtEpDevice*, OrtSessionOptions*, etc.
// The stubs treat these as opaque tokens and never dereference them.
static constexpr uintptr_t kFakeToken = 0x1;

// ---------------------------------------------------------------------------
// FactoryStubContext
//
// A superset of OrtApiStubContext that also installs the OrtApi and OrtEpApi
// function pointers touched by qnn_provider_factory.cc. Individual tests
// override specific stubs before calling into factory code (e.g. injecting a
// device-type lookup or a CreateEpDevice that returns an error).
//
// The default behaviour is:
//   - CreateMemoryInfo_V2 hands back a fake OrtMemoryInfo* and returns OK,
//     so the QnnEpFactory ctor completes without touching real ORT internals.
//   - ReleaseMemoryInfo / ReleaseHardwareDevice / ReleaseKeyValuePairs are
//     no-ops (the fake pointers were never allocated).
//   - CreateStatus / ReleaseStatus / GetErrorMessage / GetErrorCode use a
//     small heap-allocated record so callers can inspect the error message.
//   - CreateKeyValuePairs / AddKeyValuePair produce a non-null fake pointer.
//   - HardwareDevice_Type / HardwareDevice_VendorId return the values recorded
//     in device_type_map / device_vendor_map keyed by the device pointer.
//   - CreateEpDevice returns a fake OrtEpDevice* and records the input device
//     into created_ep_devices for later verification.
// ---------------------------------------------------------------------------

struct StatusRecord {
  OrtErrorCode code;
  std::string msg;
};

class FactoryStubContext {
 public:
  OrtApi stub_ort_api{};
  OrtEpApi stub_ep_api{};
  OrtModelEditorApi stub_editor_api{};

  // Simple maps used by HardwareDevice_Type / VendorId stubs, populated
  // per test before calling into factory code.
  std::unordered_map<const OrtHardwareDevice*, OrtHardwareDeviceType> device_type_map;
  std::unordered_map<const OrtHardwareDevice*, uint32_t> device_vendor_map;

  // Ordered log of every OrtHardwareDevice* passed to CreateEpDevice.
  std::vector<const OrtHardwareDevice*> created_ep_devices;

  // If non-zero, the next call to CreateEpDevice returns an error status
  // instead of a fake pointer; the counter is decremented for each call.
  int fail_next_create_ep_device = 0;

  // Version string reported by MakeFakeApiBase()'s GetVersionString. An empty
  // string is reported back as a nullptr (exercises the "(null)" branch of the
  // parse-error message); a non-empty string is passed through verbatim.
  std::string version_string = "1.99.0";

  // If true, MakeFakeApiBase()'s GetApi returns nullptr (fallback-api failure).
  bool fail_get_api = false;

  // Count of DeviceEpIncompatibilityDetails_SetDetails invocations.
  int set_details_calls = 0;

  // Session-config plumbing (used by Group 6 HTP tests):
  //   session_config          — populated per test to pre-set entries that
  //                             HasSessionConfigEntry / GetSessionConfigEntry
  //                             report to the factory (e.g., pre-loading a
  //                             backend_path to skip the autoep branch).
  //   added_config_entries    — records the (key, value) pairs written by
  //                             the factory via AddSessionConfigEntry (used to
  //                             verify the autoep + Validate/Incompatibility
  //                             temp-EP paths wired the expected keys).
  //   clone_session_options_calls — counts CloneSessionOptions calls (verifies
  //                             the autoep branch cloned before mutating).
  //
  // Groups 1–5 never populate these (empty maps mean autoep is unreachable
  // without an NPU/GPU device), so behaviour is backwards-compatible.
  std::unordered_map<std::string, std::string> session_config;
  std::vector<std::pair<std::string, std::string>> added_config_entries;
  int clone_session_options_calls = 0;

  FactoryStubContext() {
    InstallOrtApiStubs();
    InstallOrtEpApiStubs();
  }

  ApiPtrs MakeApiPtrs() { return ApiPtrs{stub_ort_api, stub_ep_api, stub_editor_api}; }

  // The stub function pointers are captureless lambdas cast to C function
  // pointers; the "self" pointer they need is threaded through this
  // thread-local pointer because the C API has no user-data slot. Public so
  // that per-test OrtApiBase::GetApi stubs can route through it too.
  static thread_local FactoryStubContext* current_;

 private:
  void InstallOrtApiStubs() {
    stub_ort_api.CreateStatus = [](OrtErrorCode code, const char* msg) noexcept -> OrtStatus* {
      auto* rec = new StatusRecord{code, msg ? msg : ""};
      return reinterpret_cast<OrtStatus*>(rec);
    };
    stub_ort_api.ReleaseStatus = [](OrtStatus* s) noexcept {
      delete reinterpret_cast<StatusRecord*>(s);
    };
    stub_ort_api.GetErrorMessage = [](const OrtStatus* s) noexcept -> const char* {
      return reinterpret_cast<const StatusRecord*>(s)->msg.c_str();
    };
    stub_ort_api.GetErrorCode = [](const OrtStatus* s) noexcept -> OrtErrorCode {
      return reinterpret_cast<const StatusRecord*>(s)->code;
    };

    stub_ort_api.CreateMemoryInfo_V2 =
        [](const char*, OrtMemoryInfoDeviceType, uint32_t, int32_t,
           OrtDeviceMemoryType, size_t, OrtAllocatorType,
           OrtMemoryInfo** out) noexcept -> OrtStatus* {
      *out = reinterpret_cast<OrtMemoryInfo*>(kFakeToken);
      return nullptr;
    };
    stub_ort_api.ReleaseMemoryInfo = [](OrtMemoryInfo*) noexcept {};

    stub_ort_api.CreateKeyValuePairs = [](OrtKeyValuePairs** out) noexcept {
      *out = reinterpret_cast<OrtKeyValuePairs*>(kFakeToken);
    };
    stub_ort_api.AddKeyValuePair = [](OrtKeyValuePairs*, const char*, const char*) noexcept {};
    stub_ort_api.ReleaseKeyValuePairs = [](OrtKeyValuePairs*) noexcept {};

    stub_ort_api.HardwareDevice_Type =
        [](const OrtHardwareDevice* dev) noexcept -> OrtHardwareDeviceType {
      auto* self = current_;
      if (self == nullptr) return OrtHardwareDeviceType_CPU;
      auto it = self->device_type_map.find(dev);
      return it == self->device_type_map.end() ? OrtHardwareDeviceType_CPU : it->second;
    };
    stub_ort_api.HardwareDevice_VendorId =
        [](const OrtHardwareDevice* dev) noexcept -> uint32_t {
      auto* self = current_;
      if (self == nullptr) return 0;
      auto it = self->device_vendor_map.find(dev);
      return it == self->device_vendor_map.end() ? 0 : it->second;
    };

    stub_ort_api.HasSessionConfigEntry =
        [](const OrtSessionOptions*, const char* key, int* out) noexcept -> OrtStatus* {
      auto* self = current_;
      *out = 0;
      if (self == nullptr || key == nullptr) return nullptr;
      if (self->session_config.count(key)) {
        *out = 1;
        return nullptr;
      }
      for (const auto& kv : self->added_config_entries) {
        if (kv.first == key) {
          *out = 1;
          return nullptr;
        }
      }
      return nullptr;
    };
    // Reads back an entry previously added via AddSessionConfigEntry or
    // pre-populated in session_config. Missing keys report size==1 with an
    // empty NUL-terminated buffer, mirroring ORT's public API contract.
    stub_ort_api.GetSessionConfigEntry =
        [](const OrtSessionOptions*, const char* key, char* buf, size_t* sz) noexcept -> OrtStatus* {
      auto* self = current_;
      const std::string* val = nullptr;
      if (self != nullptr && key != nullptr) {
        auto it = self->session_config.find(key);
        if (it != self->session_config.end()) {
          val = &it->second;
        } else {
          for (const auto& kv : self->added_config_entries) {
            if (kv.first == key) {
              val = &kv.second;
              break;
            }
          }
        }
      }
      if (val != nullptr) {
        const size_t needed = val->size() + 1;
        if (buf != nullptr) std::memcpy(buf, val->c_str(), needed);
        *sz = needed;
      } else {
        if (buf != nullptr) buf[0] = '\0';
        *sz = 1;
      }
      return nullptr;
    };
    stub_ort_api.CreateSessionOptions = [](OrtSessionOptions** out) noexcept -> OrtStatus* {
      *out = reinterpret_cast<OrtSessionOptions*>(kFakeToken);
      return nullptr;
    };
    stub_ort_api.CloneSessionOptions =
        [](const OrtSessionOptions*, OrtSessionOptions** out) noexcept -> OrtStatus* {
      if (auto* self = current_) ++self->clone_session_options_calls;
      *out = reinterpret_cast<OrtSessionOptions*>(kFakeToken);
      return nullptr;
    };
    stub_ort_api.ReleaseSessionOptions = [](OrtSessionOptions*) noexcept {};
    stub_ort_api.AddSessionConfigEntry =
        [](OrtSessionOptions*, const char* key, const char* value) noexcept -> OrtStatus* {
      if (auto* self = current_) {
        self->added_config_entries.emplace_back(key ? key : "", value ? value : "");
      }
      return nullptr;
    };
    stub_ort_api.Logger_LogMessage =
        [](const OrtLogger*, OrtLoggingLevel, const char*, const ORTCHAR_T*, int,
           const char*) noexcept -> OrtStatus* { return nullptr; };
    // Returns FATAL so Ort::Logger(OrtLogger*)'s C++ ctor short-circuits on
    // the cached-severity check — required for QnnEp construction under
    // OrtGlobalApiOverride (Group 6 tests) so that logger dispatch never
    // dereferences the fake OrtLogger* token.
    stub_ort_api.Logger_GetLoggingSeverityLevel =
        [](const OrtLogger*, OrtLoggingLevel* out) noexcept -> OrtStatus* {
      *out = ORT_LOGGING_LEVEL_FATAL;
      return nullptr;
    };
  }

  void InstallOrtEpApiStubs() {
    stub_ep_api.CreateEpDevice =
        [](OrtEpFactory*, const OrtHardwareDevice* device,
           const OrtKeyValuePairs*, const OrtKeyValuePairs*,
           OrtEpDevice** ep_device) noexcept -> OrtStatus* {
      auto* self = current_;
      if (self && self->fail_next_create_ep_device > 0) {
        --self->fail_next_create_ep_device;
        *ep_device = nullptr;
        return reinterpret_cast<OrtStatus*>(new StatusRecord{ORT_FAIL, "stub CreateEpDevice failure"});
      }
      if (self) self->created_ep_devices.push_back(device);
      *ep_device = reinterpret_cast<OrtEpDevice*>(kFakeToken);
      return nullptr;
    };
    stub_ep_api.ReleaseEpDevice = [](OrtEpDevice*) noexcept {};
    stub_ep_api.CreateHardwareDevice =
        [](OrtHardwareDeviceType, uint32_t, uint32_t, const char*,
           const OrtKeyValuePairs*, OrtHardwareDevice** out) noexcept -> OrtStatus* {
      // 3 was chosen as a fresh tag distinct from kFakeToken so tests can
      // identify the synthesized NPU device by pointer.
      *out = reinterpret_cast<OrtHardwareDevice*>(3);
      return nullptr;
    };
    stub_ep_api.ReleaseHardwareDevice = [](OrtHardwareDevice*) noexcept {};
    stub_ep_api.EpDevice_AddAllocatorInfo =
        [](OrtEpDevice*, const OrtMemoryInfo*) noexcept -> OrtStatus* { return nullptr; };
    stub_ep_api.DeviceEpIncompatibilityDetails_SetDetails =
        [](OrtDeviceEpIncompatibilityDetails*, uint32_t, int32_t, const char*) noexcept -> OrtStatus* {
      if (auto* self = current_) ++self->set_details_calls;
      return nullptr;
    };
  }
};

thread_local FactoryStubContext* FactoryStubContext::current_ = nullptr;

// RAII installer for FactoryStubContext::current_. Stub lambdas read the
// active context through this TLS pointer since C function pointers have no
// user-data slot. Restoring the previous value on scope exit keeps the stub
// state predictable if a test happens to nest contexts.
class UseFactoryStubs {
 public:
  explicit UseFactoryStubs(FactoryStubContext& ctx) noexcept {
    prev_ = FactoryStubContext::current_;
    FactoryStubContext::current_ = &ctx;
  }
  ~UseFactoryStubs() { FactoryStubContext::current_ = prev_; }

  UseFactoryStubs(const UseFactoryStubs&) = delete;
  UseFactoryStubs& operator=(const UseFactoryStubs&) = delete;

 private:
  FactoryStubContext* prev_;
};

// Combines UseFactoryStubs with OrtGlobalApiOverride so that Ort::Logger's
// C++ ctor (invoked during QnnEp's member-initialiser list when the factory
// calls std::make_unique<QnnEp>) routes through the stubbed
// Logger_GetLoggingSeverityLevel and short-circuits at FATAL. Without the
// global override, the Ort::Logger ctor would call the real
// Logger_GetLoggingSeverityLevel on our fake OrtLogger* token and SIGSEGV.
// Used exclusively by Group 6 (QnnUnit_ProviderFactoryHtpTest) tests that
// reach QnnEp construction; other groups keep their existing stub scopes.
class UseGlobalFactoryStubs {
 public:
  explicit UseGlobalFactoryStubs(FactoryStubContext& ctx)
      : use_stubs_(ctx), global_override_(&ctx.stub_ort_api) {}

  UseGlobalFactoryStubs(const UseGlobalFactoryStubs&) = delete;
  UseGlobalFactoryStubs& operator=(const UseGlobalFactoryStubs&) = delete;

 private:
  UseFactoryStubs use_stubs_;
  OrtGlobalApiOverride global_override_;
};

// Builds a fake OrtApiBase whose GetVersionString / GetApi route through the
// active FactoryStubContext (via the thread-local current_). Used by the
// CreateEpFactories tests that must stay on error paths that return BEFORE
// Ort::InitApi() (which would clobber the global C++ API table). An empty
// version_string is reported as nullptr; fail_get_api makes GetApi return null.
static OrtApiBase MakeFakeApiBase() {
  OrtApiBase base{};
  base.GetApi = [](uint32_t) noexcept -> const OrtApi* {
    auto* self = FactoryStubContext::current_;
    if (self == nullptr || self->fail_get_api) return nullptr;
    return &self->stub_ort_api;
  };
  base.GetVersionString = []() noexcept -> const char* {
    auto* self = FactoryStubContext::current_;
    if (self == nullptr || self->version_string.empty()) return nullptr;
    return self->version_string.c_str();
  };
  return base;
}

// Base fixture: resets the process-wide default logger to nullptr at the start
// of each test so that HasDefaultLogger()==false is the default state. Tests
// that need a non-null default logger call OrtLoggingManager::SetDefaultLogger
// themselves.
class QnnUnit_ProviderFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override { OrtLoggingManager::SetDefaultLogger(nullptr); }
  void TearDown() override { OrtLoggingManager::SetDefaultLogger(nullptr); }
};

// Convenience: Qualcomm vendor ID computed the same way as in the factory
// ('Q' | 'C'<<8 | 'O'<<16 | 'M'<<24).
static constexpr uint32_t kQualcommVendorId =
    static_cast<uint32_t>('Q') |
    (static_cast<uint32_t>('C') << 8) |
    (static_cast<uint32_t>('O') << 16) |
    (static_cast<uint32_t>('M') << 24);

// ===========================================================================
// Group 1: Basic getters + ctor + trivial factory methods.
// These paths never dispatch through the OrtApi tables (except the ctor's
// CreateMemoryInfo_V2, stubbed above).
// ===========================================================================

TEST_F(QnnUnit_ProviderFactoryTest, GetName_ReturnsInjectedName) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("my_ep_name", ctx.MakeApiPtrs());
  EXPECT_STREQ(factory.GetName(&factory), "my_ep_name");
}

TEST_F(QnnUnit_ProviderFactoryTest, GetVendor_ReturnsQualcomm) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  EXPECT_STREQ(factory.GetVendor(&factory), "Qualcomm");
}

TEST_F(QnnUnit_ProviderFactoryTest, GetVendorId_MatchesQualcommAcpiId) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  EXPECT_EQ(factory.GetVendorId(&factory), kQualcommVendorId);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetVersion_ReturnsSemver) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  EXPECT_STREQ(factory.GetVersion(&factory), "0.1.0");
}

TEST_F(QnnUnit_ProviderFactoryTest, IsStreamAware_ReturnsFalse) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  EXPECT_FALSE(factory.IsStreamAware(&factory));
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateDataTransfer_SetsNullAndReturnsOk) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  OrtDataTransferImpl* transfer = reinterpret_cast<OrtDataTransferImpl*>(0xDEAD);
  EXPECT_EQ(factory.CreateDataTransfer(&factory, &transfer), nullptr);
  EXPECT_EQ(transfer, nullptr);
}

TEST_F(QnnUnit_ProviderFactoryTest, ReleaseEp_NullPointer_NoCrash) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  // Must not crash / must be a no-op — early return path in ReleaseEpImpl.
  factory.ReleaseEp(&factory, nullptr);
}

// ===========================================================================
// Group 2: CreateEpFactories — argument validation + version-check error paths.
//
// These all return BEFORE the ctor's Ort::InitApi(ort_api), so they are safe
// to drive with a fake OrtApiBase (MakeFakeApiBase). The two tests that must
// pass the version gate (max_factories / null-out-params) use the REAL
// OrtGetApiBase() so InitApi installs the genuine api table (no pollution).
// ===========================================================================

// Helper: read code/message from a StatusRecord produced by the ctx stubs.
static OrtErrorCode StubStatusCode(const FactoryStubContext& ctx, const OrtStatus* s) {
  return ctx.stub_ort_api.GetErrorCode(s);
}
static std::string StubStatusMsg(const FactoryStubContext& ctx, const OrtStatus* s) {
  return ctx.stub_ort_api.GetErrorMessage(s);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_NullBase_ReturnsNull) {
  OrtEpFactory* factories[1] = {nullptr};
  size_t num = 123;
  // ort_api_base == nullptr → early return nullptr, num_factories untouched.
  EXPECT_EQ(CreateEpFactories("ep", nullptr, nullptr, factories, 1, &num), nullptr);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_NullVersionString_ReturnsParseError) {
  FactoryStubContext ctx;
  ctx.version_string = "";  // reported as nullptr → parse fails → runtime version 0
  UseFactoryStubs use(ctx);
  OrtApiBase base = MakeFakeApiBase();

  OrtEpFactory* factories[1] = {nullptr};
  size_t num = 0;
  OrtStatus* status = CreateEpFactories("ep", &base, nullptr, factories, 1, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_NE(StubStatusMsg(ctx, status).find("could not parse"), std::string::npos);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_UnparsableVersion_ReturnsParseError) {
  FactoryStubContext ctx;
  ctx.version_string = "garbage";
  UseFactoryStubs use(ctx);
  OrtApiBase base = MakeFakeApiBase();

  OrtEpFactory* factories[1] = {nullptr};
  size_t num = 0;
  OrtStatus* status = CreateEpFactories("ep", &base, nullptr, factories, 1, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_NE(StubStatusMsg(ctx, status).find("garbage"), std::string::npos);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_NonOneMajor_ReturnsParseError) {
  FactoryStubContext ctx;
  ctx.version_string = "2.0.0";  // major != 1 → parse returns 0
  UseFactoryStubs use(ctx);
  OrtApiBase base = MakeFakeApiBase();

  OrtEpFactory* factories[1] = {nullptr};
  size_t num = 0;
  OrtStatus* status = CreateEpFactories("ep", &base, nullptr, factories, 1, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_VersionTooLow_ReturnsError) {
  FactoryStubContext ctx;
  ctx.version_string = "1.5.0";  // parses to 5, below kMinOrtApiVersion (24)
  UseFactoryStubs use(ctx);
  OrtApiBase base = MakeFakeApiBase();

  OrtEpFactory* factories[1] = {nullptr};
  size_t num = 0;
  OrtStatus* status = CreateEpFactories("ep", &base, nullptr, factories, 1, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_NE(StubStatusMsg(ctx, status).find("requires ORT"), std::string::npos);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_FallbackApiNull_ReturnsNull) {
  FactoryStubContext ctx;
  ctx.version_string = "";  // parse fails → runtime version 0
  ctx.fail_get_api = true;  // GetApi(1) returns nullptr → early return nullptr
  UseFactoryStubs use(ctx);
  OrtApiBase base = MakeFakeApiBase();

  OrtEpFactory* factories[1] = {nullptr};
  size_t num = 0;
  EXPECT_EQ(CreateEpFactories("ep", &base, nullptr, factories, 1, &num), nullptr);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_MaxFactoriesZero_ReturnsInvalidArgument) {
  // Uses the REAL api base so the version gate passes and Ort::InitApi installs
  // the genuine table. max_factories < 1 returns before GetEpApi / factory ctor.
  const OrtApi* real_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEpFactory* factories[1] = {nullptr};
  size_t num = 0;
  OrtStatus* status =
      CreateEpFactories("ep", OrtGetApiBase(), nullptr, factories, /*max_factories*/ 0, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(real_api->GetErrorCode(status), ORT_INVALID_ARGUMENT);
  real_api->ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_NullOutParams_ReturnsInvalidArgument) {
  const OrtApi* real_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  size_t num = 0;
  // factories == nullptr (max_factories>=1 so it passes the size check first).
  OrtStatus* status =
      CreateEpFactories("ep", OrtGetApiBase(), nullptr, /*factories*/ nullptr, 1, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(real_api->GetErrorCode(status), ORT_INVALID_ARGUMENT);
  real_api->ReleaseStatus(status);
}

// ===========================================================================
// Group 3: GetSupportedDevicesImpl — provided-device filtering.
//
// Tests are constructed so the NPU-synthesis fallback never fires (either an
// NPU is among the created devices, or num_ep_devices == max_ep_devices after
// the loop), keeping results deterministic on x86_64 where GetSocId() /
// HasFastRpcCdspDevice() are environment-dependent.
// ===========================================================================

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_NpuQualcomm_CreatesEpDevice) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(10);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;

  const OrtHardwareDevice* devices[] = {npu};
  OrtEpDevice* ep_devices[4] = {nullptr};
  size_t num = 0;
  // has_npu_hw_device becomes true → synthesis branch skipped regardless of SoC.
  EXPECT_EQ(factory.GetSupportedDevices(&factory, devices, 1, ep_devices, 4, &num), nullptr);
  EXPECT_EQ(num, 1u);
  ASSERT_EQ(ctx.created_ep_devices.size(), 1u);
  EXPECT_EQ(ctx.created_ep_devices[0], npu);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_GpuQualcomm_CreatesEpDevice) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* gpu = MakeFakeHwDevice(11);
  ctx.device_type_map[gpu] = OrtHardwareDeviceType_GPU;
  ctx.device_vendor_map[gpu] = kQualcommVendorId;

  const OrtHardwareDevice* devices[] = {gpu};
  OrtEpDevice* ep_devices[1] = {nullptr};
  size_t num = 0;
  // max_ep_devices == 1: after the GPU is created num == max → synthesis skipped.
  EXPECT_EQ(factory.GetSupportedDevices(&factory, devices, 1, ep_devices, 1, &num), nullptr);
  EXPECT_EQ(num, 1u);
  ASSERT_EQ(ctx.created_ep_devices.size(), 1u);
  EXPECT_EQ(ctx.created_ep_devices[0], gpu);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_CpuHost_CreatesEpDevice) {
  // On the x86_64 coverage host QnnCpuBackendEnabled() is always true.
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* cpu = MakeFakeHwDevice(12);
  ctx.device_type_map[cpu] = OrtHardwareDeviceType_CPU;
  ctx.device_vendor_map[cpu] = 0;  // CPU is accepted irrespective of vendor id.

  const OrtHardwareDevice* devices[] = {cpu};
  OrtEpDevice* ep_devices[1] = {nullptr};
  size_t num = 0;
  EXPECT_EQ(factory.GetSupportedDevices(&factory, devices, 1, ep_devices, 1, &num), nullptr);
  EXPECT_EQ(num, 1u);
  ASSERT_EQ(ctx.created_ep_devices.size(), 1u);
  EXPECT_EQ(ctx.created_ep_devices[0], cpu);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_VendorMismatch_NotCreated) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(13);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;

  OrtHardwareDevice* gpu_other = MakeFakeHwDevice(14);
  ctx.device_type_map[gpu_other] = OrtHardwareDeviceType_GPU;
  ctx.device_vendor_map[gpu_other] = 0x8086;  // non-Qualcomm → filtered out.

  const OrtHardwareDevice* devices[] = {npu, gpu_other};
  OrtEpDevice* ep_devices[4] = {nullptr};
  size_t num = 0;
  EXPECT_EQ(factory.GetSupportedDevices(&factory, devices, 2, ep_devices, 4, &num), nullptr);
  EXPECT_EQ(num, 1u);
  ASSERT_EQ(ctx.created_ep_devices.size(), 1u);
  EXPECT_EQ(ctx.created_ep_devices[0], npu);  // GPU with wrong vendor never created.
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_MaxEpDevicesTruncates) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(15);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;

  OrtHardwareDevice* gpu = MakeFakeHwDevice(16);
  ctx.device_type_map[gpu] = OrtHardwareDeviceType_GPU;
  ctx.device_vendor_map[gpu] = kQualcommVendorId;

  const OrtHardwareDevice* devices[] = {npu, gpu};
  OrtEpDevice* ep_devices[1] = {nullptr};
  size_t num = 0;
  // max_ep_devices == 1: the loop stops after the first device is created.
  EXPECT_EQ(factory.GetSupportedDevices(&factory, devices, 2, ep_devices, 1, &num), nullptr);
  EXPECT_EQ(num, 1u);
  ASSERT_EQ(ctx.created_ep_devices.size(), 1u);
  EXPECT_EQ(ctx.created_ep_devices[0], npu);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_CreateEpDeviceFails_PropagatesError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(17);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;
  ctx.fail_next_create_ep_device = 1;  // first CreateEpDevice call returns an error

  const OrtHardwareDevice* devices[] = {npu};
  OrtEpDevice* ep_devices[4] = {nullptr};
  size_t num = 0;
  // The supported NPU passes the filter, so CreateEpDevice is invoked and its
  // error status must propagate out of GetSupportedDevices.
  // Note: the create_ep_device lambda increments num_ep_devices before it
  // returns, so the caller sees num == 1 with ep_devices[0] == nullptr.
  OrtStatus* status = factory.GetSupportedDevices(&factory, devices, 1, ep_devices, 4, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_EQ(num, 1u);
  EXPECT_EQ(ep_devices[0], nullptr);
  EXPECT_TRUE(ctx.created_ep_devices.empty());
  ctx.stub_ort_api.ReleaseStatus(status);
}
// Group 4: GetHardwareDeviceIncompatibilityDetailsImpl /
//          ValidateCompiledModelCompatibilityInfoImpl — error paths that stop
//          before std::make_unique<QnnEp>. The fixture leaves the default
//          logger unset, so the "no default logger" branches are reachable.
// ===========================================================================

TEST_F(QnnUnit_ProviderFactoryTest, IncompatibilityDetails_WrongVendorNpu_SetsDetails) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(20);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = 0x8086;  // non-Qualcomm NPU → incompatible.

  auto* details = reinterpret_cast<OrtDeviceEpIncompatibilityDetails*>(kFakeToken);
  // Returns whatever SetDetails returns (stub → nullptr); the point is that the
  // SetDetails path fires and make_unique<QnnEp> is never reached.
  EXPECT_EQ(factory.GetHardwareDeviceIncompatibilityDetails(&factory, npu, details), nullptr);
  EXPECT_EQ(ctx.set_details_calls, 1);
}

TEST_F(QnnUnit_ProviderFactoryTest, IncompatibilityDetails_SupportedNpu_NoDefaultLogger_ReturnsError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(21);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;  // supported → past SetDetails.

  auto* details = reinterpret_cast<OrtDeviceEpIncompatibilityDetails*>(kFakeToken);
  OrtStatus* status = factory.GetHardwareDeviceIncompatibilityDetails(&factory, npu, details);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_NE(StubStatusMsg(ctx, status).find("Default logger is not available"), std::string::npos);
  EXPECT_EQ(ctx.set_details_calls, 0);  // supported device: SetDetails not called.
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, ValidateCompatibility_NoNpuDevice_ReturnsEpNotApplicable) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* cpu = MakeFakeHwDevice(22);
  ctx.device_type_map[cpu] = OrtHardwareDeviceType_CPU;

  const OrtHardwareDevice* devices[] = {cpu};
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  OrtStatus* status =
      factory.ValidateCompiledModelCompatibilityInfo(&factory, devices, 1, "info", &compat);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_EP_FAIL);
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, ValidateCompatibility_NpuDevice_NoDefaultLogger_ReturnsError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(23);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;

  const OrtHardwareDevice* devices[] = {npu};
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  OrtStatus* status =
      factory.ValidateCompiledModelCompatibilityInfo(&factory, devices, 1, "info", &compat);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_EP_FAIL);
  EXPECT_NE(StubStatusMsg(ctx, status).find("Default logger is not available"), std::string::npos);
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
  ctx.stub_ort_api.ReleaseStatus(status);
}

// ===========================================================================
// Group 5: Additional unit-coverable error / branch paths that model and
// integration tests skip (they always set backend_type/backend_path, bypassing
// the CreateEp autoep block, and never drive the argument-validation branches).
// All of these return before std::make_unique<QnnEp>.
// ===========================================================================

TEST_F(QnnUnit_ProviderFactoryTest, Ctor_CreateMemoryInfoFails_ReleasesMemoryInfo) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  // CreateMemoryInfo_V2 returns a non-null status → ctor takes the
  // ReleaseMemoryInfo cleanup branch. Out-param is left null per the ORT
  // convention that output params are null on failure. Must not crash.
  ctx.stub_ort_api.CreateMemoryInfo_V2 =
      [](const char*, OrtMemoryInfoDeviceType, uint32_t, int32_t,
         OrtDeviceMemoryType, size_t, OrtAllocatorType,
         OrtMemoryInfo** out) noexcept -> OrtStatus* {
    *out = nullptr;
    return reinterpret_cast<OrtStatus*>(new StatusRecord{ORT_FAIL, "stub CreateMemoryInfo_V2 failure"});
  };
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  EXPECT_STREQ(factory.GetName(&factory), "ep");
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEp_NullLogger_NoDefaultLogger_ReturnsError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtEp* ep = reinterpret_cast<OrtEp*>(0xDEAD);
  OrtStatus* status = factory.CreateEp(&factory, /*devices*/ nullptr, /*ep_metadata*/ nullptr,
                                       /*num_devices*/ 0,
                                       reinterpret_cast<const OrtSessionOptions*>(kFakeToken),
                                       /*logger*/ nullptr, &ep);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(ep, nullptr);  // *ep = nullptr happens before the logger check.
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_NE(StubStatusMsg(ctx, status).find("Logger is nullptr"), std::string::npos);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEp_ZeroDevices_ReturnsError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  // Non-null logger skips the default-logger lookup; HasSessionConfigEntry
  // always reports "not present" so the autoep block is entered, where
  // num_devices == 0 errors out.
  const auto* logger = reinterpret_cast<const OrtLogger*>(kFakeToken);
  OrtEp* ep = nullptr;
  OrtStatus* status = factory.CreateEp(&factory, nullptr, nullptr, 0,
                                       reinterpret_cast<const OrtSessionOptions*>(kFakeToken),
                                       logger, &ep);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_NE(StubStatusMsg(ctx, status).find("No devices were provided"), std::string::npos);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEp_SingleCpuDevice_NoDefaultBackend_ReturnsError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* cpu = MakeFakeHwDevice(30);
  ctx.device_type_map[cpu] = OrtHardwareDeviceType_CPU;

  const OrtHardwareDevice* devices[] = {cpu};
  const auto* logger = reinterpret_cast<const OrtLogger*>(kFakeToken);
  OrtEp* ep = nullptr;
  // Single CPU device → kDefaultBackends (NPU/GPU only) lookup misses → error.
  OrtStatus* status = factory.CreateEp(&factory, devices, nullptr, 1,
                                       reinterpret_cast<const OrtSessionOptions*>(kFakeToken),
                                       logger, &ep);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_NE(StubStatusMsg(ctx, status).find("Could not determine default backend"), std::string::npos);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEp_MultipleDevicesNoNpuGpu_ReturnsError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* cpu0 = MakeFakeHwDevice(31);
  OrtHardwareDevice* cpu1 = MakeFakeHwDevice(32);
  ctx.device_type_map[cpu0] = OrtHardwareDeviceType_CPU;
  ctx.device_type_map[cpu1] = OrtHardwareDeviceType_CPU;

  const OrtHardwareDevice* devices[] = {cpu0, cpu1};
  const auto* logger = reinterpret_cast<const OrtLogger*>(kFakeToken);
  OrtEp* ep = nullptr;
  // Multiple devices but neither NPU nor GPU present → error.
  OrtStatus* status = factory.CreateEp(&factory, devices, nullptr, 2,
                                       reinterpret_cast<const OrtSessionOptions*>(kFakeToken),
                                       logger, &ep);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_NE(StubStatusMsg(ctx, status).find("neither an NPU nor a GPU"), std::string::npos);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, ValidateCompatibility_CreateSessionOptionsFails_ReturnsError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  // CreateSessionOptions fails → EP_NOT_APPLICABLE + propagated status.
  ctx.stub_ort_api.CreateSessionOptions = [](OrtSessionOptions** out) noexcept -> OrtStatus* {
    *out = nullptr;
    return reinterpret_cast<OrtStatus*>(new StatusRecord{ORT_FAIL, "stub CreateSessionOptions failure"});
  };
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(33);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;

  const OrtHardwareDevice* devices[] = {npu};
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  OrtStatus* status =
      factory.ValidateCompiledModelCompatibilityInfo(&factory, devices, 1, "info", &compat);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, ValidateCompatibility_AddSessionConfigEntryFails_ReturnsError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  // CreateSessionOptions OK (default), AddSessionConfigEntry fails.
  ctx.stub_ort_api.AddSessionConfigEntry =
      [](OrtSessionOptions*, const char*, const char*) noexcept -> OrtStatus* {
    return reinterpret_cast<OrtStatus*>(new StatusRecord{ORT_FAIL, "stub AddSessionConfigEntry failure"});
  };
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(34);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;

  const OrtHardwareDevice* devices[] = {npu};
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  OrtStatus* status =
      factory.ValidateCompiledModelCompatibilityInfo(&factory, devices, 1, "info", &compat);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_VersionTooLow_FallbackApiNull_ReturnsNull) {
  FactoryStubContext ctx;
  ctx.version_string = "1.5.0";  // parses to 5 (< 24)
  ctx.fail_get_api = true;       // GetApi(5) returns nullptr → early return nullptr
  UseFactoryStubs use(ctx);
  OrtApiBase base = MakeFakeApiBase();

  OrtEpFactory* factories[1] = {nullptr};
  size_t num = 0;
  EXPECT_EQ(CreateEpFactories("ep", &base, nullptr, factories, 1, &num), nullptr);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEpFactories_ValidVersion_GetApiNull_ReturnsNull) {
  FactoryStubContext ctx;
  ctx.version_string = "1.99.0";  // parses to 99 (>= 24), passes the version gate
  ctx.fail_get_api = true;        // GetApi(requested) returns null → return before InitApi
  UseFactoryStubs use(ctx);
  OrtApiBase base = MakeFakeApiBase();

  OrtEpFactory* factories[1] = {nullptr};
  size_t num = 0;
  EXPECT_EQ(CreateEpFactories("ep", &base, nullptr, factories, 1, &num), nullptr);
}

TEST_F(QnnUnit_ProviderFactoryTest, ReleaseAllocator_UnknownType_NoCrash) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  // Default qnn_allocator_type_ is NONE → neither HTP-shared nor DX12 → the
  // "unknown type" warning branch runs. Must not crash.
  auto* allocator = reinterpret_cast<OrtAllocator*>(kFakeToken);
  factory.ReleaseAllocator(&factory, allocator);
}

TEST_F(QnnUnit_ProviderFactoryTest, ReleaseEpFactory_NullPointer_ReturnsNull) {
  EXPECT_EQ(ReleaseEpFactory(nullptr), nullptr);
}

// ===========================================================================
// Group 6: Real-HTP-backend paths (QnnUnit_ProviderFactoryHtpTest)
//
// These tests exercise factory code paths that need a real QnnEp instance:
//   - CreateEpImpl autoep branch (clone session options, add backend_path).
//   - ValidateCompiledModelCompatibilityInfoImpl temp-EP construction path.
//   - GetHardwareDeviceIncompatibilityDetailsImpl temp-EP construction path.
//
// QnnEp construction dlopens libQnnHtp.so via QnnBackendManager; the fixture
// GTEST_SKIP()s when the backend is unavailable, mirroring
// QnnUnit_BackendManagerHtpTest / QnnUnit_ExecutionProviderHtpTest.
//
// Strategy: stubbed OrtApi (Groups 1–5 infra, extended with session-config
// read-back and Logger_GetLoggingSeverityLevel=FATAL) + OrtGlobalApiOverride
// (so Ort::Logger's C++ ctor short-circuits) + real dlopen inside
// QnnBackendManager. The tests verify factory *behaviour* — the recorded
// added_config_entries / clone_session_options_calls — which is what actually
// changes coverage: the target lines are hit before make_unique<QnnEp>
// completes, so status inspection is best-effort (both OK and QnnEp
// construction failures are tolerated with a diagnostic).
// ===========================================================================

class QnnUnit_ProviderFactoryHtpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    OrtLoggingManager::SetDefaultLogger(nullptr);
    QnnRealHtpBackendContext htp_check;
    if (!htp_check.IsValid()) {
      GTEST_SKIP() << "libQnnHtp.so not available";
    }
  }

  void TearDown() override { OrtLoggingManager::SetDefaultLogger(nullptr); }
};

// Verifies that CreateEpImpl enters the autoep branch when neither
// backend_type nor backend_path is set: it must CloneSessionOptions once and
// AddSessionConfigEntry a "backend_path" entry whose value ends in
// libQnnHtp.so (derived from kDefaultBackends[NPU]). QnnEp construction may
// or may not succeed depending on whether the auto-computed path is
// resolvable at runtime; either outcome is acceptable — the autoep
// side-effects are what this test locks in.
TEST_F(QnnUnit_ProviderFactoryHtpTest,
       CreateEp_AutoepNpuQualcomm_ClonesAndAddsBackendPath) {
  FactoryStubContext ctx;  // empty session_config → autoep branch
  UseGlobalFactoryStubs use(ctx);
  QnnEpFactory factory("QNNExecutionProvider", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(20);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;

  auto* fake_session_opts = reinterpret_cast<OrtSessionOptions*>(kFakeToken);
  auto* fake_logger = reinterpret_cast<OrtLogger*>(kFakeToken);
  OrtEp* ep = nullptr;
  const OrtHardwareDevice* devices[] = {npu};
  const OrtKeyValuePairs* metadata[] = {nullptr};

  OrtStatus* status = factory.CreateEp(&factory, devices, metadata, 1,
                                       fake_session_opts, fake_logger, &ep);

  // Autoep side-effects — the coverage target — must have happened regardless
  // of whether make_unique<QnnEp> succeeded.
  EXPECT_EQ(ctx.clone_session_options_calls, 1)
      << "autoep should have cloned session options exactly once";
  bool added_backend_path = false;
  for (const auto& kv : ctx.added_config_entries) {
    if (kv.first.find("backend_path") != std::string::npos &&
        kv.second.find("libQnnHtp.so") != std::string::npos) {
      added_backend_path = true;
      break;
    }
  }
  EXPECT_TRUE(added_backend_path)
      << "autoep should have added a backend_path entry ending in libQnnHtp.so";

  if (status == nullptr) {
    EXPECT_NE(ep, nullptr);
    factory.ReleaseEp(&factory, ep);
  } else {
    ctx.stub_ort_api.ReleaseStatus(status);
  }
}

// Multi-device autoep: NPU + GPU present, no backend_type/backend_path — the
// factory should pick the NPU (and log a warning). Covers the multi-device
// std::find_if(is_npu) branch of CreateEpImpl.
TEST_F(QnnUnit_ProviderFactoryHtpTest,
       CreateEp_AutoepMultiDeviceNpuAndGpu_PicksNpu) {
  FactoryStubContext ctx;
  UseGlobalFactoryStubs use(ctx);
  QnnEpFactory factory("QNNExecutionProvider", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(21);
  OrtHardwareDevice* gpu = MakeFakeHwDevice(22);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_type_map[gpu] = OrtHardwareDeviceType_GPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;
  ctx.device_vendor_map[gpu] = kQualcommVendorId;

  auto* fake_session_opts = reinterpret_cast<OrtSessionOptions*>(kFakeToken);
  auto* fake_logger = reinterpret_cast<OrtLogger*>(kFakeToken);
  OrtEp* ep = nullptr;
  const OrtHardwareDevice* devices[] = {gpu, npu};  // GPU listed first
  const OrtKeyValuePairs* metadata[] = {nullptr, nullptr};

  OrtStatus* status = factory.CreateEp(&factory, devices, metadata, 2,
                                       fake_session_opts, fake_logger, &ep);

  // Autoep must clone + add backend_path for HTP (NPU wins over GPU).
  EXPECT_EQ(ctx.clone_session_options_calls, 1);
  bool added_htp_path = false;
  for (const auto& kv : ctx.added_config_entries) {
    if (kv.first.find("backend_path") != std::string::npos &&
        kv.second.find("libQnnHtp.so") != std::string::npos) {
      added_htp_path = true;
      break;
    }
  }
  EXPECT_TRUE(added_htp_path);

  if (status == nullptr) {
    EXPECT_NE(ep, nullptr);
    factory.ReleaseEp(&factory, ep);
  } else {
    ctx.stub_ort_api.ReleaseStatus(status);
  }
}

// ValidateCompiledModelCompatibilityInfoImpl temp-EP path:
// factory->qnn_ep_ is nullptr initially; passing an NPU device causes
// backend_type="htp" to be determined and CreateSessionOptions +
// AddSessionConfigEntry to fire before std::make_unique<QnnEp> for the temp
// EP. Covers lines 393-421 of qnn_provider_factory.cc.
TEST_F(QnnUnit_ProviderFactoryHtpTest,
       ValidateCompatibilityInfo_NpuNoQnnEp_WiresHtpBackendType) {
  FactoryStubContext ctx;
  UseGlobalFactoryStubs use(ctx);
  QnnEpFactory factory("QNNExecutionProvider", ctx.MakeApiPtrs());

  // ValidateCompiledModelCompatibilityInfoImpl requires HasDefaultLogger().
  auto* fake_logger = reinterpret_cast<OrtLogger*>(kFakeToken);
  OrtLoggingManager::SetDefaultLogger(fake_logger);

  OrtHardwareDevice* npu = MakeFakeHwDevice(23);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;

  const OrtHardwareDevice* devices[] = {npu};
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  OrtStatus* status = factory.ValidateCompiledModelCompatibilityInfo(
      &factory, devices, 1, "1:2.0.0:1.22.0:3.0.0:73:0", &compat);

  bool added_htp_backend_type = false;
  for (const auto& kv : ctx.added_config_entries) {
    if (kv.first.find("backend_type") != std::string::npos && kv.second == "htp") {
      added_htp_backend_type = true;
      break;
    }
  }
  EXPECT_TRUE(added_htp_backend_type)
      << "Validate should have added backend_type=htp before temp-EP construction";

  if (status != nullptr) ctx.stub_ort_api.ReleaseStatus(status);
}

// GetHardwareDeviceIncompatibilityDetailsImpl temp-EP path: NPU/Qualcomm
// device passes the type/vendor gate, then CreateSessionOptions +
// AddSessionConfigEntry("backend_type","htp") fire, then temp-EP
// construction is attempted. Covers lines 473-500.
TEST_F(QnnUnit_ProviderFactoryHtpTest,
       GetHardwareDeviceIncompatibilityDetails_NpuQualcomm_WiresHtpBackendType) {
  FactoryStubContext ctx;
  UseGlobalFactoryStubs use(ctx);
  QnnEpFactory factory("QNNExecutionProvider", ctx.MakeApiPtrs());

  auto* fake_logger = reinterpret_cast<OrtLogger*>(kFakeToken);
  OrtLoggingManager::SetDefaultLogger(fake_logger);

  OrtHardwareDevice* npu = MakeFakeHwDevice(24);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;

  auto* fake_details = reinterpret_cast<OrtDeviceEpIncompatibilityDetails*>(kFakeToken);
  OrtStatus* status = factory.GetHardwareDeviceIncompatibilityDetails(
      &factory, npu, fake_details);

  bool added_htp_backend_type = false;
  for (const auto& kv : ctx.added_config_entries) {
    if (kv.first.find("backend_type") != std::string::npos && kv.second == "htp") {
      added_htp_backend_type = true;
      break;
    }
  }
  EXPECT_TRUE(added_htp_backend_type)
      << "Incompatibility should have added backend_type=htp before temp-EP construction";

  if (status != nullptr) ctx.stub_ort_api.ReleaseStatus(status);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
