// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for QnnEpFactory (qnn_provider_factory.cc).
//
// All tests here run without loading any real QNN backend shared library, and
// without constructing a real QnnEp instance. They exercise:
//   - Basic getters (name/vendor/version/vendor-id/is-stream-aware) and the
//     ctor's CreateMemoryInfo_V2 wiring.
//   - Factory data-transfer creation.
//   - Trivial factory methods that never dispatch through the OrtApi tables
//     (ReleaseEpImpl(nullptr)).
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
// tests GTEST_SKIP() when the platform HTP backend is unavailable. This mirrors
// QnnUnit_BackendManagerHtpTest — component-level tests with a real backend
// load but no ORT session.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_allocator.h"

#include "test/providers/qnn/infra/qnn_test_windows_compat.h"

#include "core/providers/qnn/qnn_provider_factory.h"
#include "core/providers/qnn/genie/genie_backend_manager.h"
#include "core/providers/qnn/soc_utility/soc_utils.h"
#include "onnxruntime_config.h"
#include "onnxruntime_ep_device_ep_metadata_keys.h"

#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

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

// Convenience: Qualcomm vendor ID computed the same way as in the factory
// ('Q' | 'C'<<8 | 'O'<<16 | 'M'<<24).
static constexpr uint32_t kQualcommVendorId =
    static_cast<uint32_t>('Q') |
    (static_cast<uint32_t>('C') << 8) |
    (static_cast<uint32_t>('O') << 16) |
    (static_cast<uint32_t>('M') << 24);

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
  using KeyValues = std::unordered_map<std::string, std::string>;
  KeyValues env_config;
  std::unordered_map<const OrtHardwareDevice*, KeyValues> hardware_metadata;
  std::unordered_map<const OrtHardwareDevice*, uint32_t> device_ids;
  std::unordered_map<const OrtHardwareDevice*, KeyValues> ep_metadata;
  std::unordered_map<const OrtHardwareDevice*, KeyValues> ep_options;
  std::vector<std::unique_ptr<KeyValues>> key_values;
  std::vector<const OrtHardwareDevice*> synthesized_devices;
  std::vector<const OrtHardwareDevice*> released_hardware;
  std::vector<std::string> log_messages;
  std::unordered_map<const OrtHardwareDevice*, uint32_t> device_id_map;
  std::unordered_set<const OrtHardwareDevice*> virtual_devices;
  bool fail_get_env_config = false;
  bool fail_create_hardware = false;
  int fail_create_ep_device_call = 0;
  int create_ep_device_calls = 0;
  int create_session_options_calls = 0;
  int release_session_options_calls = 0;
  OrtLoggingLevel log_severity = ORT_LOGGING_LEVEL_FATAL;
  uint32_t synthesized_device_id = 0;

  OrtHardwareDevice* AddDevice(OrtHardwareDeviceType type, bool is_virtual = false,
                               uint32_t vendor = kQualcommVendorId, uint32_t id = 0) {
    auto* device = MakeFakeHwDevice(next_device_tag_++);
    device_type_map[device] = type;
    device_vendor_map[device] = vendor;
    device_ids[device] = id;
    if (is_virtual) hardware_metadata[device][kOrtHardwareDevice_MetadataKey_IsVirtual] = "1";
    return device;
  }

  // Ordered log of (key, value) pairs passed to AddKeyValuePair (e.g., hw device metadata).
  std::vector<std::pair<std::string, std::string>> added_kv_pairs;

  // Ordered log of every OrtHardwareDevice* passed to CreateEpDevice.
  std::vector<const OrtHardwareDevice*> created_ep_devices;

  // If non-zero, the next call to CreateEpDevice returns an error status
  // instead of a fake pointer; the counter is decremented for each call.
  int fail_next_create_ep_device = 0;
  bool fail_after_first_ep_device = false;
  size_t released_hw_device_count = 0;

  // Count of EpDevice_AddAllocatorInfo invocations.
  int add_allocator_info_calls = 0;

  // If non-zero, the next EpDevice_AddAllocatorInfo call returns an error.
  int fail_next_add_allocator_info = 0;

  // Count of ReleaseEpDevice invocations.
  int release_ep_device_calls = 0;

  // Version string reported by MakeFakeApiBase()'s GetVersionString. An empty
  // string is reported back as a nullptr (exercises the "(null)" branch of the
  // parse-error message); a non-empty string is passed through verbatim.
  std::string version_string = "1.99.0";

  // If true, MakeFakeApiBase()'s GetApi returns nullptr (fallback-api failure).
  bool fail_get_api = false;

  // Capture the compatibility result, including failures reported by the EP.
  int set_details_calls = 0;
  uint32_t last_incompatibility_reason = OrtDeviceEpIncompatibility_NONE;
  int32_t last_incompatibility_error_code = 0;
  std::string last_incompatibility_message;

  // Session-config plumbing for construction and compatibility probes:
  //   session_config — entries pre-populated by each test and read by QnnEp.
  //   added_config_entries — entries written by the factory (only CPU probes
  //                          should inject backend_type).
  //   clone_session_options_calls — verifies default selection leaves the
  //                                 caller's session options unchanged.
  std::unordered_map<std::string, std::string> session_config;
  std::vector<std::pair<std::string, std::string>> added_config_entries;
  int clone_session_options_calls = 0;

  FactoryStubContext() {
    // QnnEpFactory owns the real Ort::CustomOpDomain objects it registers in
    // its constructor. Forward these callbacks to the real ORT API so the
    // domains can be released safely while the global stub API is installed.
    const OrtApi* real_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    stub_ort_api.CreateCustomOpDomain = real_ort_api->CreateCustomOpDomain;
    stub_ort_api.ReleaseCustomOpDomain = real_ort_api->ReleaseCustomOpDomain;
    stub_ort_api.CustomOpDomain_Add = real_ort_api->CustomOpDomain_Add;

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
  uintptr_t next_device_tag_ = 0x100;

  void InstallOrtApiStubs() {
    stub_ort_api.GetEpApi = []() noexcept -> const OrtEpApi* {
      auto* self = current_;
      return self == nullptr ? nullptr : &self->stub_ep_api;
    };
    stub_ort_api.GetModelEditorApi = []() noexcept -> const OrtModelEditorApi* {
      auto* self = current_;
      return self == nullptr ? nullptr : &self->stub_editor_api;
    };

    // QnnEp probes the v28 EPContext experimental API during construction.
    // This fake runtime has no experimental APIs, so report them unavailable
    // rather than leaving the lookup entry null.
    stub_ort_api.GetExperimentalFunction = [](const char*) noexcept -> OrtExperimentalFnPtr {
      return nullptr;
    };
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
    stub_ort_api.MemoryInfoGetDeviceMemType =
        [](const OrtMemoryInfo*) noexcept -> OrtDeviceMemoryType {
      return OrtDeviceMemoryType_HOST_ACCESSIBLE;
    };

    stub_ort_api.CreateKeyValuePairs = [](OrtKeyValuePairs** out) noexcept {
      auto& pairs = current_->key_values.emplace_back(std::make_unique<KeyValues>());
      *out = reinterpret_cast<OrtKeyValuePairs*>(pairs.get());
    };
    stub_ort_api.AddKeyValuePair = [](OrtKeyValuePairs* pairs, const char* key, const char* value) noexcept {
      (*reinterpret_cast<KeyValues*>(pairs))[key] = value;
      current_->added_kv_pairs.emplace_back(key, value);
    };
    // Key/value storage belongs to the stub context; device creation copies it.
    stub_ort_api.ReleaseKeyValuePairs = [](OrtKeyValuePairs*) noexcept {};
    stub_ort_api.GetKeyValuePairs = [](const OrtKeyValuePairs* pairs, const char* const** keys,
                                       const char* const** values, size_t* num_entries) noexcept {
      if (pairs == nullptr) {
        *keys = nullptr;
        *values = nullptr;
        *num_entries = 0;
        return;
      }
      const auto& map = *reinterpret_cast<const KeyValues*>(pairs);
      static thread_local std::vector<const char*> key_storage;
      static thread_local std::vector<const char*> value_storage;
      key_storage.clear();
      value_storage.clear();
      for (const auto& [key, value] : map) {
        key_storage.push_back(key.c_str());
        value_storage.push_back(value.c_str());
      }
      *keys = key_storage.data();
      *values = value_storage.data();
      *num_entries = key_storage.size();
    };
    stub_ort_api.GetKeyValue = [](const OrtKeyValuePairs* pairs, const char* key) noexcept -> const char* {
      if (pairs == nullptr || key == nullptr) return nullptr;
      const auto& values = *reinterpret_cast<const KeyValues*>(pairs);
      const auto it = values.find(key);
      return it == values.end() ? nullptr : it->second.c_str();
    };
    stub_ort_api.HardwareDevice_Metadata = [](const OrtHardwareDevice* device) noexcept -> const OrtKeyValuePairs* {
      auto* self = current_;
      const auto it = self->hardware_metadata.find(device);
      if (it != self->hardware_metadata.end()) return reinterpret_cast<const OrtKeyValuePairs*>(&it->second);
      if (self->virtual_devices.count(device)) {
        static const KeyValues virtual_metadata{{kOrtHardwareDevice_MetadataKey_IsVirtual, "1"}};
        return reinterpret_cast<const OrtKeyValuePairs*>(&virtual_metadata);
      }
      return nullptr;
    };
    stub_ort_api.HardwareDevice_DeviceId = [](const OrtHardwareDevice* device) noexcept {
      const auto current_id = current_->device_ids.find(device);
      if (current_id != current_->device_ids.end()) return current_id->second;
      const auto legacy_id = current_->device_id_map.find(device);
      return legacy_id == current_->device_id_map.end() ? 0u : legacy_id->second;
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
      ++current_->create_session_options_calls;
      *out = reinterpret_cast<OrtSessionOptions*>(kFakeToken);
      return nullptr;
    };
    stub_ort_api.CloneSessionOptions =
        [](const OrtSessionOptions*, OrtSessionOptions** out) noexcept -> OrtStatus* {
      if (auto* self = current_) ++self->clone_session_options_calls;
      *out = reinterpret_cast<OrtSessionOptions*>(kFakeToken);
      return nullptr;
    };
    stub_ort_api.ReleaseSessionOptions = [](OrtSessionOptions*) noexcept {
      ++current_->release_session_options_calls;
    };
    stub_ort_api.AddSessionConfigEntry =
        [](OrtSessionOptions*, const char* key, const char* value) noexcept -> OrtStatus* {
      if (auto* self = current_) {
        self->added_config_entries.emplace_back(key ? key : "", value ? value : "");
      }
      return nullptr;
    };
    stub_ort_api.Logger_LogMessage =
        [](const OrtLogger*, OrtLoggingLevel, const char* message, const ORTCHAR_T*, int,
           const char*) noexcept -> OrtStatus* {
      current_->log_messages.emplace_back(message);
      return nullptr;
    };
    // Default to FATAL; tests can enable logging to inspect backend selection.
    // Both logger callbacks are stubbed so the fake token is never dereferenced.
    stub_ort_api.Logger_GetLoggingSeverityLevel =
        [](const OrtLogger*, OrtLoggingLevel* out) noexcept -> OrtStatus* {
      *out = current_->log_severity;
      return nullptr;
    };
  }

  void InstallOrtEpApiStubs() {
    stub_ep_api.GetEnvConfigEntries = [](OrtKeyValuePairs** out) noexcept -> OrtStatus* {
      if (current_->fail_get_env_config) {
        return current_->stub_ort_api.CreateStatus(ORT_FAIL, "stub environment failure");
      }
      *out = reinterpret_cast<OrtKeyValuePairs*>(&current_->env_config);
      return nullptr;
    };
    stub_ep_api.CreateEpDevice =
        [](OrtEpFactory*, const OrtHardwareDevice* device,
           const OrtKeyValuePairs* metadata, const OrtKeyValuePairs* options,
           OrtEpDevice** ep_device) noexcept -> OrtStatus* {
      auto* self = current_;
      if (self && self->fail_after_first_ep_device && self->created_ep_devices.size() == 1) {
        self->fail_after_first_ep_device = false;
        *ep_device = nullptr;
        return self->stub_ort_api.CreateStatus(ORT_FAIL, "stub second CreateEpDevice failure");
      }
      if (++self->create_ep_device_calls == self->fail_create_ep_device_call) {
        *ep_device = nullptr;
        return self->stub_ort_api.CreateStatus(ORT_FAIL, "stub CreateEpDevice failure");
      }
      if (self && self->fail_next_create_ep_device > 0) {
        --self->fail_next_create_ep_device;
        *ep_device = nullptr;
        return reinterpret_cast<OrtStatus*>(new StatusRecord{ORT_FAIL, "stub CreateEpDevice failure"});
      }
      self->created_ep_devices.push_back(device);
      self->ep_metadata[device] = *reinterpret_cast<const KeyValues*>(metadata);
      self->ep_options[device] = *reinterpret_cast<const KeyValues*>(options);
      *ep_device = reinterpret_cast<OrtEpDevice*>(0x1000 + self->created_ep_devices.size());
      return nullptr;
    };
    stub_ep_api.ReleaseEpDevice = [](OrtEpDevice*) noexcept {
      if (auto* self = current_) ++self->release_ep_device_calls;
    };
    stub_ep_api.CreateHardwareDevice =
        [](OrtHardwareDeviceType type, uint32_t vendor, uint32_t id, const char*,
           const OrtKeyValuePairs* metadata, OrtHardwareDevice** out) noexcept -> OrtStatus* {
      auto* self = current_;
      if (self->fail_create_hardware) {
        *out = nullptr;
        return self->stub_ort_api.CreateStatus(ORT_FAIL, "stub CreateHardwareDevice failure");
      }
      self->synthesized_device_id = id;
      *out = self->AddDevice(type, false, vendor, id);
      self->hardware_metadata[*out] = *reinterpret_cast<const KeyValues*>(metadata);
      self->synthesized_devices.push_back(*out);
      return nullptr;
    };
    stub_ep_api.ReleaseHardwareDevice = [](OrtHardwareDevice* device) noexcept {
      ++current_->released_hw_device_count;
      current_->released_hardware.push_back(device);
    };
    stub_ep_api.EpDevice_AddAllocatorInfo =
        [](OrtEpDevice*, const OrtMemoryInfo*) noexcept -> OrtStatus* {
      if (auto* self = current_) {
        ++self->add_allocator_info_calls;
        if (self->fail_next_add_allocator_info > 0) {
          --self->fail_next_add_allocator_info;
          return reinterpret_cast<OrtStatus*>(new StatusRecord{ORT_FAIL, "stub AddAllocatorInfo failure"});
        }
      }
      return nullptr;
    };
    stub_ep_api.DeviceEpIncompatibilityDetails_SetDetails =
        [](OrtDeviceEpIncompatibilityDetails*, uint32_t reasons, int32_t error_code,
           const char* message) noexcept -> OrtStatus* {
      if (auto* self = current_) {
        ++self->set_details_calls;
        self->last_incompatibility_reason = reasons;
        self->last_incompatibility_error_code = error_code;
        self->last_incompatibility_message = message ? message : "";
      }
      return nullptr;
    };
  }
};

thread_local FactoryStubContext* FactoryStubContext::current_ = nullptr;

// RAII installer for FactoryStubContext::current_ and the matching global ORT
// API. Stub lambdas read the active context through this TLS pointer since C
// function pointers have no user-data slot. The API override is also required
// while constructing QnnEpFactory because its custom-op setup uses Ort C++
// wrappers, and Windows keeps separate API state in the test EXE and provider
// DLL.
class UseFactoryStubs {
 public:
  explicit UseFactoryStubs(FactoryStubContext& ctx)
      : global_override_(&ctx.stub_ort_api) {
    prev_ = FactoryStubContext::current_;
    FactoryStubContext::current_ = &ctx;
  }
  ~UseFactoryStubs() { FactoryStubContext::current_ = prev_; }

  UseFactoryStubs(const UseFactoryStubs&) = delete;
  UseFactoryStubs& operator=(const UseFactoryStubs&) = delete;

 private:
  OrtGlobalApiOverride global_override_;
  FactoryStubContext* prev_;
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

// Base fixture: make HasDefaultLogger()==false in both the test executable and
// provider DLL by default, then restore their previous state after each test.
class QnnUnit_ProviderFactoryTest : public ::testing::Test {
 protected:
  ScopedDefaultLoggerOverride default_logger_scope_{nullptr};
};

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
  EXPECT_STREQ(factory.GetVersion(&factory), ORT_QNN_EP_VERSION);
}

TEST_F(QnnUnit_ProviderFactoryTest, IsStreamAware_ReturnsFalse) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  EXPECT_FALSE(factory.IsStreamAware(&factory));
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateDataTransfer_ReturnsInitializedTransfer) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  OrtDataTransferImpl* transfer = nullptr;
  EXPECT_EQ(factory.CreateDataTransfer(&factory, &transfer), nullptr);
  ASSERT_NE(transfer, nullptr);
  EXPECT_EQ(transfer->ort_version_supported, ORT_API_VERSION);
  EXPECT_NE(transfer->CanCopy, nullptr);
  EXPECT_NE(transfer->CopyTensors, nullptr);
  EXPECT_NE(transfer->Release, nullptr);
  transfer->Release(transfer);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateAllocator_HostAccessible_DoesNotLoadRpcMem) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  // ORT calls the factory allocator callback while registering every
  // advertised allocator. Creating that allocator must not load RPCMEM, which
  // is intentionally unavailable in this host-only unit test.
  OrtAllocator* allocator = nullptr;
  OrtStatus* status = factory.CreateAllocator(
      &factory, reinterpret_cast<const OrtMemoryInfo*>(kFakeToken), nullptr, &allocator);

  EXPECT_EQ(status, nullptr);
  ASSERT_NE(allocator, nullptr);
  EXPECT_NE(allocator->Alloc, nullptr);
  EXPECT_NE(allocator->Free, nullptr);
  EXPECT_NE(allocator->Info, nullptr);
  EXPECT_EQ(allocator->Reserve, allocator->Alloc);
  EXPECT_EQ(allocator->Info(allocator), reinterpret_cast<const OrtMemoryInfo*>(kFakeToken));
#ifndef _WIN32
  EXPECT_EQ(allocator->Alloc, qnn::HtpSharedMemoryAllocator::AllocImpl);
#endif
  factory.ReleaseAllocator(&factory, allocator);
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateAllocator_NullMemoryInfo_ReturnsNullAllocator) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtAllocator* allocator = reinterpret_cast<OrtAllocator*>(kFakeToken);
  EXPECT_EQ(factory.CreateAllocator(&factory, nullptr, nullptr, &allocator), nullptr);
  EXPECT_EQ(allocator, nullptr);
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
  // QNN test_main opts into CPU advertisement on ARM64 before EP registration.
  // On x86, CPU advertisement is enabled by default for the HTP emulator.
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

#if defined(_WIN32) && QNN_ARCH_ARM64 && \
    (QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 40))

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_EnvironmentError_PropagatesError) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  ctx.fail_get_env_config = true;
  OrtEpDevice* ep_devices[4] = {};
  size_t num = 0;
  OrtStatus* status = factory.GetSupportedDevices(&factory, nullptr, 0, ep_devices, 4, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_EQ(num, 0u);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_InvalidHardwareId_RejectsBeforeAdvertising) {
  for (const char* value : {"", "0x", "-1", "+1", " 41304430", "41304430junk", "100000000"}) {
    SCOPED_TRACE(value);
    FactoryStubContext ctx;
    UseFactoryStubs use(ctx);
    QnnEpFactory factory("ep", ctx.MakeApiPtrs());
    ctx.env_config["ep_factory.ep.qnn_hardware_device_id"] = value;
    OrtEpDevice* ep_devices[4] = {};
    size_t num = 0;
    OrtStatus* status = factory.GetSupportedDevices(&factory, nullptr, 0, ep_devices, 4, &num);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(StubStatusCode(ctx, status), ORT_INVALID_ARGUMENT);
    EXPECT_EQ(num, 0u);
    ctx.stub_ort_api.ReleaseStatus(status);
  }
}

#endif  // defined(_WIN32) && QNN_ARCH_ARM64 && QNN API >= 2.40

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
  // Failed creation must not publish an output slot.
  OrtStatus* status = factory.GetSupportedDevices(&factory, devices, 1, ep_devices, 4, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_EQ(num, 0u);
  EXPECT_EQ(ep_devices[0], nullptr);
  EXPECT_TRUE(ctx.created_ep_devices.empty());
  EXPECT_EQ(ctx.add_allocator_info_calls, 0);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_AddAllocatorInfoFails_DoesNotPublishDevice) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtHardwareDevice* npu = MakeFakeHwDevice(18);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;
  ctx.fail_next_add_allocator_info = 1;

  const OrtHardwareDevice* devices[] = {npu};
  OrtEpDevice* ep_devices[4] = {nullptr};
  size_t num = 0;
  OrtStatus* status = factory.GetSupportedDevices(&factory, devices, 1, ep_devices, 4, &num);

  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_FAIL);
  EXPECT_EQ(num, 0u);
  EXPECT_EQ(ep_devices[0], nullptr);
  EXPECT_EQ(ctx.add_allocator_info_calls, 1);
  EXPECT_EQ(ctx.release_ep_device_calls, 1);
  ctx.stub_ort_api.ReleaseStatus(status);
}

#if defined(_WIN32) && QNN_ARCH_ARM64 && \
    (QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 40))

TEST_F(QnnUnit_ProviderFactoryTest, CreateEp_VirtualNpuRejectsConflictingSocWithSameHtpArchitecture) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("QNNExecutionProvider", ctx.MakeApiPtrs());
  const auto npu = MakeFakeHwDevice(125);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.virtual_devices.insert(npu);
  ctx.env_config["target_architecture"] = "60";
  // Both SoCs use HTP V73, but the selected SoC must still be preserved.
  ctx.session_config["ep.qnnexecutionprovider.soc_model"] = "43";
  const OrtHardwareDevice* devices[] = {npu};
  const OrtKeyValuePairs* metadata[] = {reinterpret_cast<const OrtKeyValuePairs*>(kFakeToken)};
  OrtEp* ep = nullptr;
  OrtStatus* status = factory.CreateEp(
      &factory, devices, metadata, 1, reinterpret_cast<const OrtSessionOptions*>(kFakeToken),
      reinterpret_cast<const OrtLogger*>(kFakeToken), &ep);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_INVALID_ARGUMENT);
  EXPECT_NE(StubStatusMsg(ctx, status).find("soc_model"), std::string::npos);
  EXPECT_EQ(ep, nullptr);
  ctx.stub_ort_api.ReleaseStatus(status);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_TargetRequiresIdentity) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  ctx.env_config["ep_factory.ep.qnn_target_architecture"] = "60";
  OrtEpDevice* ep_devices[4] = {};
  size_t num = 0;
  OrtStatus* status = factory.GetSupportedDevices(&factory, nullptr, 0, ep_devices, 4, &num);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(StubStatusCode(ctx, status), ORT_INVALID_ARGUMENT);
  EXPECT_EQ(num, 0u);
  ctx.stub_ort_api.ReleaseStatus(status);
}

#endif  // defined(_WIN32) && QNN_ARCH_ARM64 && QNN API >= 2.40

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_PartialFailureReleasesOutputs) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  const auto npu = MakeFakeHwDevice(117);
  const auto gpu = MakeFakeHwDevice(118);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_type_map[gpu] = OrtHardwareDeviceType_GPU;
  ctx.device_vendor_map[npu] = ctx.device_vendor_map[gpu] = kQualcommVendorId;
  ctx.fail_after_first_ep_device = true;
  const OrtHardwareDevice* hardware[] = {npu, gpu};
  OrtEpDevice* output[4] = {};
  size_t count = 0;
  OrtStatus* status = factory.GetSupportedDevices(&factory, hardware, 2, output, 4, &count);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(count, 0u);
  EXPECT_EQ(output[0], nullptr);
  EXPECT_EQ(ctx.release_ep_device_calls, 1);
  ctx.stub_ort_api.ReleaseStatus(status);
}

#if defined(_WIN32) && QNN_ARCH_ARM64 && \
    (QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 40))

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_InvalidTargetRejectsBeforeAdvertising) {
  for (const char* target : {"", "0", "-60", "+60", "60junk", "4294967296", "htp-v73", " 60"}) {
    SCOPED_TRACE(target);
    FactoryStubContext ctx;
    UseFactoryStubs use(ctx);
    QnnEpFactory factory("ep", ctx.MakeApiPtrs());
    ctx.env_config["ep_factory.ep.qnn_hardware_device_id"] = "41304430";
    ctx.env_config["ep_factory.ep.qnn_target_architecture"] = target;
    OrtEpDevice* ep_devices[4] = {};
    size_t num = 0;
    OrtStatus* status = factory.GetSupportedDevices(&factory, nullptr, 0, ep_devices, 4, &num);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(StubStatusCode(ctx, status), ORT_INVALID_ARGUMENT);
    EXPECT_EQ(num, 0u);
    ctx.stub_ort_api.ReleaseStatus(status);
  }
}

#endif  // defined(_WIN32) && QNN_ARCH_ARM64 && QNN API >= 2.40

TEST(QnnHtpTargetTest, SocModelTargetIsDistinctFromHtpArchitecture) {
  EXPECT_EQ(qnn::soc::ParseSocModelTarget("60"), 60u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(60), 73u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(88), 81u);
  EXPECT_FALSE(qnn::soc::ParseSocModelTarget("htp-v73"));
  EXPECT_FALSE(qnn::soc::ParseSocModelTarget("0"));
  EXPECT_FALSE(qnn::soc::ParseSocModelTarget("4294967296"));
}

TEST(QnnHtpTargetTest, SessionDefaultsAreNotConcreteTargets) {
  EXPECT_EQ(qnn::soc::ParseHtpArchitecture(""), 0u);
  EXPECT_EQ(qnn::soc::ParseHtpArchitecture("0"), 0u);
  for (const char* value : {"073", "+73", "73junk", " 73", "999"}) {
    EXPECT_FALSE(qnn::soc::ParseHtpArchitecture(value));
  }
}

#if defined(_WIN32) && QNN_ARCH_ARM64 && \
    (QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 40))

TEST_F(QnnUnit_ProviderFactoryTest,
       GetSupportedDevices_Arm64CompilerTargetAdvertisesMatchingVirtualNpuAndGpu) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("QNNExecutionProvider", ctx.MakeApiPtrs());
  ctx.env_config["allow_virtual_devices"] = "1";
  ctx.env_config["ep_factory.QNNExecutionProvider.qnn_hardware_device_id"] = "41304430";
  ctx.env_config["ep_factory.QNNExecutionProvider.qnn_target_architecture"] = "60";

  OrtEpDevice* ep_devices[4] = {};
  size_t num = 0;
  EXPECT_EQ(factory.GetSupportedDevices(&factory, nullptr, 0, ep_devices, 4, &num), nullptr);
  ASSERT_EQ(num, 2u);
  ASSERT_EQ(ctx.synthesized_devices.size(), 2u);
  EXPECT_EQ(ctx.device_type_map.at(ctx.synthesized_devices[0]), OrtHardwareDeviceType_NPU);
  EXPECT_EQ(ctx.device_type_map.at(ctx.synthesized_devices[1]), OrtHardwareDeviceType_GPU);
  for (const auto* device : ctx.synthesized_devices) {
    EXPECT_EQ(ctx.device_ids.at(device), 0x41304430u);
  }
  ASSERT_EQ(ctx.created_ep_devices.size(), 2u);
  EXPECT_TRUE(std::find(ctx.added_kv_pairs.begin(), ctx.added_kv_pairs.end(),
                        std::pair<std::string, std::string>{"target_architecture", "60"}) !=
              ctx.added_kv_pairs.end());
}

TEST_F(QnnUnit_ProviderFactoryTest,
       GetSupportedDevices_Arm64OptInWithoutIdentityPreservesHostDiscovery) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("QNNExecutionProvider", ctx.MakeApiPtrs());
  ctx.env_config["allow_virtual_devices"] = "1";

  OrtEpDevice* ep_devices[4] = {};
  size_t num = 0;
  OrtStatus* status = factory.GetSupportedDevices(&factory, nullptr, 0, ep_devices, 4, &num);
  ASSERT_EQ(status, nullptr);
  EXPECT_EQ(num, 1u);  // Preserve the generic discovery path from #896.
}

#endif  // defined(_WIN32) && QNN_ARCH_ARM64 && QNN API >= 2.40

// ===========================================================================
// Group 3b: GetSupportedDevicesImpl — x86/x64 cross-compile NPU synthesis.
//
// On x86/x64 hosts (no real Qualcomm NPU), GetSupportedDevicesImpl should
// synthesize a virtual NPU OrtHardwareDevice for AOT cross-compilation.
// ===========================================================================

#if !QNN_ARCH_ARM64

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_RepeatedSynthesisRetainsHardware) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  ctx.env_config["allow_virtual_devices"] = "1";
  {
    QnnEpFactory factory("ep", ctx.MakeApiPtrs());
    for (int i = 0; i < 2; ++i) {
      OrtEpDevice* output[4] = {};
      size_t count = 0;
      EXPECT_EQ(factory.GetSupportedDevices(&factory, nullptr, 0, output, 4, &count), nullptr);
      ASSERT_EQ(count, 2u);
      EXPECT_EQ(ctx.released_hw_device_count, 0u);
    }
  }
  EXPECT_EQ(ctx.released_hw_device_count, 4u);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_CrossCompileHost_AdvertisesVirtualNpuAndGpu) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  ctx.env_config["allow_virtual_devices"] = "1";
  ctx.env_config["ep_factory.ep.qnn_target_architecture"] = "88";
  ctx.env_config["ep_factory.ep.qnn_hardware_device_id"] = "0x36334630";
  OrtEpDevice* output[2] = {};
  size_t count = 0;
  Ort::Status status(factory.GetSupportedDevices(&factory, nullptr, 0, output, 2, &count));
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_EQ(count, 2u);
  ASSERT_EQ(ctx.synthesized_devices.size(), 2u);
  const auto* npu = ctx.synthesized_devices[0];
  const auto* gpu = ctx.synthesized_devices[1];
  EXPECT_EQ(ctx.device_type_map.at(npu), OrtHardwareDeviceType_NPU);
  EXPECT_EQ(ctx.device_type_map.at(gpu), OrtHardwareDeviceType_GPU);
  EXPECT_EQ(ctx.device_ids.at(npu), 0x36334630u);
  EXPECT_EQ(ctx.device_ids.at(gpu), 0x36334630u);
  EXPECT_TRUE(ctx.ep_metadata.at(npu).empty());
  EXPECT_TRUE(ctx.ep_options.at(npu).empty());
  EXPECT_EQ(ctx.ep_metadata.at(gpu).at("target_architecture"), "88");
  EXPECT_EQ(ctx.ep_options.at(gpu).at("soc_model"), "88");
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_CrossCompileHost_CpuDeviceUnaffected) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  ctx.env_config["allow_virtual_devices"] = "1";

  // Pass only a CPU device — no NPU in the input array.
  OrtHardwareDevice* cpu = MakeFakeHwDevice(30);
  ctx.device_type_map[cpu] = OrtHardwareDeviceType_CPU;
  ctx.device_vendor_map[cpu] = 0;

  const OrtHardwareDevice* devices[] = {cpu};
  OrtEpDevice* ep_devices[4] = {nullptr};
  size_t num = 0;
  EXPECT_EQ(factory.GetSupportedDevices(&factory, devices, 1, ep_devices, 4, &num), nullptr);

  // CPU input plus the synthesized virtual NPU and GPU.
  EXPECT_EQ(num, 3u);
  ASSERT_EQ(ctx.created_ep_devices.size(), 3u);
  EXPECT_EQ(ctx.created_ep_devices[0], cpu);
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_CrossCompileHostWithoutOptIn_DoesNotAdvertiseVirtualNpu) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  OrtEpDevice* ep_devices[4] = {nullptr};
  size_t num = 0;
  EXPECT_EQ(factory.GetSupportedDevices(&factory, nullptr, 0, ep_devices, 4, &num), nullptr);

  EXPECT_EQ(num, 0u);
  EXPECT_TRUE(ctx.created_ep_devices.empty());
  EXPECT_TRUE(ctx.added_kv_pairs.empty());
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_CrossCompileHost_ExplicitNpuNoDoubleSynthesis) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  // Pass an explicit Qualcomm NPU device on x86 — synthesis should NOT fire.
  OrtHardwareDevice* npu = MakeFakeHwDevice(32);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;

  const OrtHardwareDevice* devices[] = {npu};
  OrtEpDevice* ep_devices[4] = {nullptr};
  size_t num = 0;
  EXPECT_EQ(factory.GetSupportedDevices(&factory, devices, 1, ep_devices, 4, &num), nullptr);

  // Only the explicitly provided NPU — no virtual synthesis despite being on x86.
  EXPECT_EQ(num, 1u);
  ASSERT_EQ(ctx.created_ep_devices.size(), 1u);
  EXPECT_EQ(ctx.created_ep_devices[0], npu);

  // No IsVirtual metadata should have been added for the real NPU.
  bool found_is_virtual = false;
  for (const auto& kv : ctx.added_kv_pairs) {
    if (kv.first == kOrtHardwareDevice_MetadataKey_IsVirtual) {
      found_is_virtual = true;
      break;
    }
  }
  EXPECT_FALSE(found_is_virtual) << "Explicit NPU on x86 — no synthesis, no IsVirtual metadata.";
}

#endif  // !QNN_ARCH_ARM64

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_RealNpuPresent_NoSynthesis) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());

  // Pass a real Qualcomm NPU — synthesis should NOT fire.
  OrtHardwareDevice* npu = MakeFakeHwDevice(31);
  ctx.device_type_map[npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[npu] = kQualcommVendorId;

  const OrtHardwareDevice* devices[] = {npu};
  OrtEpDevice* ep_devices[4] = {nullptr};
  size_t num = 0;
  EXPECT_EQ(factory.GetSupportedDevices(&factory, devices, 1, ep_devices, 4, &num), nullptr);

  // Only the real NPU, no synthesis.
  EXPECT_EQ(num, 1u);
  ASSERT_EQ(ctx.created_ep_devices.size(), 1u);
  EXPECT_EQ(ctx.created_ep_devices[0], npu);

  // No IsVirtual metadata should have been added.
  bool found_is_virtual = false;
  for (const auto& kv : ctx.added_kv_pairs) {
    if (kv.first == kOrtHardwareDevice_MetadataKey_IsVirtual) {
      found_is_virtual = true;
      break;
    }
  }
  EXPECT_FALSE(found_is_virtual) << "Real NPU present — no synthesis, no IsVirtual metadata.";
}

// ===========================================================================
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
  // An allocator with no recognized callback takes the warning branch and is
  // not deleted. Use a valid object because release dispatch reads Alloc.
  OrtAllocator allocator{};
  factory.ReleaseAllocator(&factory, &allocator);
}

TEST_F(QnnUnit_ProviderFactoryTest, ReleaseEpFactory_NullPointer_ReturnsNull) {
  EXPECT_EQ(ReleaseEpFactory(nullptr), nullptr);
}

// ===========================================================================
// Group 6: Temporary EP compatibility probes.
//
// Install a default logger so both factory APIs construct a real temporary
// QnnEp and delegate to it. Empty model info verifies the EP's validation
// response without loading QNN; a nonexistent backend exercises SetupBackend
// and verifies the EP's missing-dependency classification. Constructor failures
// are test failures. Neither test requires an installed QNN backend.
// ===========================================================================

TEST_F(QnnUnit_ProviderFactoryTest, ValidateCompatibility_TemporaryEpUsesDefaultHtpBackend) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ScopedDefaultLoggerOverride logger_scope(reinterpret_cast<const OrtLogger*>(kFakeToken));
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  const OrtHardwareDevice* devices[] = {ctx.AddDevice(OrtHardwareDeviceType_NPU)};
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;

  Ort::Status status(factory.ValidateCompiledModelCompatibilityInfo(&factory, devices, 1, "", &compat));

  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
  EXPECT_TRUE(std::any_of(ctx.log_messages.begin(), ctx.log_messages.end(), [](const auto& message) {
    return message.find("Using default backend path:") != std::string::npos &&
           message.find(QnnHtpBackendLibraryName()) != std::string::npos;
  }));
  EXPECT_NE(std::find(ctx.log_messages.begin(), ctx.log_messages.end(),
                      "No compatibility info to be validated."),
            ctx.log_messages.end());
  EXPECT_TRUE(ctx.added_config_entries.empty());
  EXPECT_EQ(ctx.clone_session_options_calls, 0);
  EXPECT_EQ(ctx.create_session_options_calls, 1);
  EXPECT_EQ(ctx.release_session_options_calls, 1);
}

TEST_F(QnnUnit_ProviderFactoryTest, IncompatibilityDetails_TemporaryEpReportsMissingBackend) {
  for (auto type : {OrtHardwareDeviceType_NPU, OrtHardwareDeviceType_GPU}) {
    SCOPED_TRACE(type);
    FactoryStubContext ctx;
    UseFactoryStubs use(ctx);
    ScopedDefaultLoggerOverride logger_scope(reinterpret_cast<const OrtLogger*>(kFakeToken));
    // Inject a path through the session-options stub to make backend setup fail
    // deterministically, even on hosts with QNN installed. No factory option
    // injection is needed for NPU/GPU probes.
    const auto backend_path = std::filesystem::absolute("qnn_factory_missing_backend/libQnnFake.so");
    ASSERT_FALSE(std::filesystem::exists(backend_path));
    ctx.session_config["ep.ep.backend_path"] = backend_path.string();
    QnnEpFactory factory("ep", ctx.MakeApiPtrs());
    const auto* device = ctx.AddDevice(type, false, kQualcommVendorId,
                                       type == OrtHardwareDeviceType_GPU ? 0x36334630 : 0);

    Ort::Status status(factory.GetHardwareDeviceIncompatibilityDetails(
        &factory, device, reinterpret_cast<OrtDeviceEpIncompatibilityDetails*>(kFakeToken)));

    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    ASSERT_EQ(ctx.set_details_calls, 1);
    EXPECT_EQ(ctx.last_incompatibility_reason, OrtDeviceEpIncompatibility_MISSING_DEPENDENCY);
    EXPECT_EQ(ctx.last_incompatibility_error_code, QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED);
    EXPECT_NE(ctx.last_incompatibility_message.find("Unable to load backend"), std::string::npos);
    EXPECT_TRUE(ctx.added_config_entries.empty());
    EXPECT_EQ(ctx.clone_session_options_calls, 0);
    EXPECT_EQ(ctx.create_session_options_calls, 1);
    EXPECT_EQ(ctx.release_session_options_calls, 1);
  }
}

// ===========================================================================
// Group 7: Virtual-device discovery and backend selection.
// QnnEp construction selects the backend lazily without loading its library.
// These tests verify device ordering and unchanged caller session options.
// ===========================================================================

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_VirtualGpuOptInAndCapacity) {
  for (const char* allow : {"", "0", "1", "true"}) {
    for (size_t capacity : {0u, 1u, 2u}) {
      SCOPED_TRACE(allow);
      SCOPED_TRACE(capacity);
      FactoryStubContext ctx;
      UseFactoryStubs use(ctx);
      QnnEpFactory factory("ep", ctx.MakeApiPtrs());
      // Supplying an NPU makes this independent of host NPU discovery.
      const OrtHardwareDevice* devices[] = {ctx.AddDevice(OrtHardwareDeviceType_NPU)};
      if (*allow) ctx.env_config["allow_virtual_devices"] = allow;
      OrtEpDevice* output[3] = {};
      size_t count = 99;
      Ort::Status status(factory.GetSupportedDevices(&factory, devices, 1, output, capacity, &count));
      ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
      const size_t expected = capacity == 0 ? 0 : (std::strcmp(allow, "1") == 0 && capacity == 2 ? 2 : 1);
      EXPECT_EQ(count, expected);
      EXPECT_EQ(output[capacity], nullptr);
      ASSERT_EQ(ctx.synthesized_devices.size(), expected == 2 ? 1u : 0u);
      if (expected == 2) {
        const auto* gpu = ctx.synthesized_devices.front();
        EXPECT_EQ(ctx.device_type_map.at(gpu), OrtHardwareDeviceType_GPU);
        EXPECT_EQ(ctx.device_vendor_map.at(gpu), kQualcommVendorId);
        EXPECT_EQ(ctx.hardware_metadata.at(gpu).at(kOrtHardwareDevice_MetadataKey_IsVirtual), "1");
        EXPECT_EQ(ctx.add_allocator_info_calls, 1);  // NPU only.
      }
    }
  }
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_PhysicalGpuNotDuplicated) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  ctx.env_config["allow_virtual_devices"] = "1";
  const OrtHardwareDevice* devices[] = {ctx.AddDevice(OrtHardwareDeviceType_GPU),
                                        ctx.AddDevice(OrtHardwareDeviceType_NPU)};
  OrtEpDevice* output[4] = {};
  size_t count = 0;
  Ort::Status status(factory.GetSupportedDevices(&factory, devices, 2, output, 4, &count));
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(count, 2u);
  EXPECT_TRUE(ctx.synthesized_devices.empty());
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_TargetOverrideOnlyAppliesToVirtualDevices) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  const auto host_soc = std::to_string(qnn::soc::MapSocModelFromSocId(qnn::soc::GetSocId()));
  const std::string target = host_soc == "88" ? "69" : "88";
  ctx.env_config["ep_factory.QNNExecutionProvider.qnn_target_architecture"] = target;
  const OrtHardwareDevice* devices[] = {ctx.AddDevice(OrtHardwareDeviceType_NPU),
                                        ctx.AddDevice(OrtHardwareDeviceType_GPU),
                                        ctx.AddDevice(OrtHardwareDeviceType_NPU, true),
                                        ctx.AddDevice(OrtHardwareDeviceType_GPU, true)};
  OrtEpDevice* output[4] = {};
  size_t count = 0;
  Ort::Status status(factory.GetSupportedDevices(&factory, devices, 4, output, 4, &count));
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_EQ(count, 4u);
  for (size_t i = 0; i < 4; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(ctx.ep_metadata.at(devices[i]).at("target_architecture"), i < 2 ? host_soc : target);
    if (i < 2) {
      EXPECT_TRUE(ctx.ep_options.at(devices[i]).empty());
    } else {
      EXPECT_EQ(ctx.ep_options.at(devices[i]).at("soc_model"), target);
    }
  }
}


TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_TargetParsingUsesNonzeroSocModelOrHostFallback) {
  for (const char* target : {static_cast<const char*>(nullptr), "88", "4294967295", "0", "", "-88", "+88", "88junk", "4294967296", "htp-v73", " 88"}) {
    SCOPED_TRACE(target ? target : "unset");
    FactoryStubContext ctx;
    UseFactoryStubs use(ctx);
    QnnEpFactory factory("ep", ctx.MakeApiPtrs());
    if (target) ctx.env_config["ep_factory.ep.qnn_target_architecture"] = target;
    const OrtHardwareDevice* devices[] = {ctx.AddDevice(OrtHardwareDeviceType_GPU, true)};
    OrtEpDevice* output[1] = {};
    size_t count = 0;
    Ort::Status status(factory.GetSupportedDevices(&factory, devices, 1, output, 1, &count));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    const bool valid = target && (std::strcmp(target, "88") == 0 || std::strcmp(target, "4294967295") == 0);
    const auto expected = valid ? std::string(target) : std::to_string(qnn::soc::MapSocModelFromSocId(qnn::soc::GetSocId()));
    EXPECT_EQ(ctx.ep_metadata.at(devices[0]).at("target_architecture"), expected);
    EXPECT_EQ(ctx.ep_options.at(devices[0]).at("soc_model"), expected);
  }
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_EnvironmentErrorPropagatesBeforeAdvertising) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  ctx.fail_get_env_config = true;
  OrtEpDevice* output[2] = {};
  size_t count = 99;
  Ort::Status status(factory.GetSupportedDevices(&factory, nullptr, 0, output, 2, &count));
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorMessage(), "stub environment failure");
  EXPECT_EQ(count, 0u);
  EXPECT_TRUE(ctx.created_ep_devices.empty());
}

TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_SyntheticGpuFailurePreservesPublishedNpu) {
  for (bool fail_hardware : {false, true}) {
    SCOPED_TRACE(fail_hardware);
    FactoryStubContext ctx;
    UseFactoryStubs use(ctx);
    {
      QnnEpFactory factory("ep", ctx.MakeApiPtrs());
      ctx.env_config["allow_virtual_devices"] = "1";
      ctx.fail_create_hardware = fail_hardware;
      ctx.fail_create_ep_device_call = fail_hardware ? 0 : 2;
      const OrtHardwareDevice* devices[] = {ctx.AddDevice(OrtHardwareDeviceType_NPU)};
      OrtEpDevice* output[2] = {};
      size_t count = 0;
      Ort::Status status(factory.GetSupportedDevices(&factory, devices, 1, output, 2, &count));
      ASSERT_FALSE(status.IsOK());
      // Current API behavior retains already-published outputs on a later error.
      EXPECT_EQ(count, 1u);
      EXPECT_NE(output[0], nullptr);
      EXPECT_EQ(output[1], nullptr);
      EXPECT_EQ(ctx.release_ep_device_calls, 0);
      EXPECT_TRUE(ctx.released_hardware.empty());
    }
    EXPECT_EQ(ctx.released_hardware.size(), fail_hardware ? 0u : 1u);
  }
}

// Regression fence: earlier enumeration results must not refer to released hardware.
// This currently exposes the factory's replacement of virtual_gpu_hw_device_.
TEST_F(QnnUnit_ProviderFactoryTest, GetSupportedDevices_RepeatedSynthesisRetainsHardware) {
  for (bool supply_npu : {false, true}) {
    SCOPED_TRACE(supply_npu);
    FactoryStubContext ctx;
    UseFactoryStubs use(ctx);
    ctx.env_config["allow_virtual_devices"] = "1";
    {
      QnnEpFactory factory("ep", ctx.MakeApiPtrs());
      const OrtHardwareDevice* devices[] = {ctx.AddDevice(OrtHardwareDeviceType_NPU)};
      for (int i = 0; i < 2; ++i) {
        OrtEpDevice* output[2] = {};
        size_t count = 0;
        Ort::Status status(factory.GetSupportedDevices(&factory, supply_npu ? devices : nullptr,
                                                       supply_npu ? 1 : 0, output, 2, &count));
        ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
        ASSERT_EQ(count, 2u);
        EXPECT_TRUE(ctx.released_hardware.empty());
      }
    }
    EXPECT_EQ(ctx.released_hardware.size(), ctx.synthesized_devices.size());
  }
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEp_DefaultBackendSelectionPreservesSessionOptions) {
  const std::vector<std::vector<OrtHardwareDeviceType>> combinations = {
      {OrtHardwareDeviceType_NPU}, {OrtHardwareDeviceType_GPU}, {OrtHardwareDeviceType_GPU, OrtHardwareDeviceType_NPU}, {OrtHardwareDeviceType_NPU, OrtHardwareDeviceType_GPU}, {OrtHardwareDeviceType_CPU, OrtHardwareDeviceType_GPU}};
  for (const auto& types : combinations) {
    SCOPED_TRACE(types.front());
    FactoryStubContext ctx;
    UseFactoryStubs use(ctx);
    ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
    QnnEpFactory factory("ep", ctx.MakeApiPtrs());
    std::vector<const OrtHardwareDevice*> devices;
    for (auto type : types) devices.push_back(ctx.AddDevice(type));
    OrtEp* ep = nullptr;
    Ort::Status status(factory.CreateEp(&factory, devices.data(), nullptr, devices.size(),
                                        reinterpret_cast<const OrtSessionOptions*>(kFakeToken),
                                        reinterpret_cast<const OrtLogger*>(kFakeToken), &ep));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    std::unique_ptr<QnnEp> owned_ep(static_cast<QnnEp*>(ep));
    ASSERT_NE(owned_ep, nullptr);
    const auto expected_type = std::find(types.begin(), types.end(), OrtHardwareDeviceType_NPU) != types.end()
                                   ? OrtHardwareDeviceType_NPU
                                   : OrtHardwareDeviceType_GPU;
#ifdef _WIN32
    const char* expected_backend = expected_type == OrtHardwareDeviceType_NPU ? "QnnHtp.dll" : "QnnGpu.dll";
#else
    const char* expected_backend = expected_type == OrtHardwareDeviceType_NPU ? "libQnnHtp.so" : "libQnnGpu.so";
#endif
    EXPECT_TRUE(std::any_of(ctx.log_messages.begin(), ctx.log_messages.end(), [&](const auto& message) {
      return message.find("Using backend path:") != std::string::npos &&
             message.find(expected_backend) != std::string::npos;
    }));
    EXPECT_EQ(ctx.clone_session_options_calls, 0);
    EXPECT_TRUE(ctx.added_config_entries.empty());
  }
}

TEST_F(QnnUnit_ProviderFactoryTest, CreateEp_ExplicitCpuBackendBypassesMissingDefault) {
#ifdef _WIN32
  const char* cpu_backend_path = "QnnCpu.dll";
#else
  const char* cpu_backend_path = "libQnnCpu.so";
#endif
  for (const char* option : {"backend_type", "backend_path"}) {
    FactoryStubContext ctx;
    UseFactoryStubs use(ctx);
    QnnEpFactory factory("ep", ctx.MakeApiPtrs());
    ctx.session_config[std::string("ep.ep.") + option] = std::strcmp(option, "backend_type") == 0
                                                             ? "cpu"
                                                             : cpu_backend_path;
    const OrtHardwareDevice* devices[] = {ctx.AddDevice(OrtHardwareDeviceType_CPU)};
    OrtEp* ep = nullptr;
    Ort::Status status(factory.CreateEp(&factory, devices, nullptr, 1,
                                        reinterpret_cast<const OrtSessionOptions*>(kFakeToken),
                                        reinterpret_cast<const OrtLogger*>(kFakeToken), &ep));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    ASSERT_NE(ep, nullptr);
    factory.ReleaseEp(&factory, ep);
    EXPECT_EQ(ctx.clone_session_options_calls, 0);
    EXPECT_TRUE(ctx.added_config_entries.empty());
  }
}

TEST_F(QnnUnit_ProviderFactoryTest, IncompatibilityDetails_OnlyCpuProbeInjectsBackendType) {
  for (auto type : {OrtHardwareDeviceType_CPU, OrtHardwareDeviceType_NPU, OrtHardwareDeviceType_GPU}) {
    SCOPED_TRACE(type);
    FactoryStubContext ctx;
    UseFactoryStubs use(ctx);
    QnnEpFactory factory("ep", ctx.MakeApiPtrs());
    Ort::Status status(factory.GetHardwareDeviceIncompatibilityDetails(
        &factory, ctx.AddDevice(type), reinterpret_cast<OrtDeviceEpIncompatibilityDetails*>(kFakeToken)));
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(std::string(status.GetErrorMessage()).find("Default logger"), std::string::npos);
    EXPECT_EQ(ctx.create_session_options_calls, 1);
    EXPECT_EQ(ctx.release_session_options_calls, 1);
    if (type == OrtHardwareDeviceType_CPU) {
      ASSERT_EQ(ctx.added_config_entries.size(), 1u);
      EXPECT_EQ(ctx.added_config_entries[0], (std::pair<std::string, std::string>{"ep.ep.backend_type", "cpu"}));
    } else {
      EXPECT_TRUE(ctx.added_config_entries.empty());
    }
  }
}

TEST_F(QnnUnit_ProviderFactoryTest, IncompatibilityDetails_CpuConfigFailureReleasesTemporaryOptions) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  ctx.stub_ort_api.AddSessionConfigEntry = [](OrtSessionOptions*, const char*, const char*) noexcept -> OrtStatus* {
    return FactoryStubContext::current_->stub_ort_api.CreateStatus(ORT_FAIL, "CPU config failure");
  };
  QnnEpFactory factory("ep", ctx.MakeApiPtrs());
  Ort::Status status(factory.GetHardwareDeviceIncompatibilityDetails(
      &factory, ctx.AddDevice(OrtHardwareDeviceType_CPU), reinterpret_cast<OrtDeviceEpIncompatibilityDetails*>(kFakeToken)));
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorMessage(), "CPU config failure");
  EXPECT_EQ(ctx.create_session_options_calls, 1);
  EXPECT_EQ(ctx.release_session_options_calls, 1);
}


#if defined(_WIN32) && QNN_ARCH_ARM64 && \
    (QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 40))

TEST_F(QnnUnit_ProviderFactoryHtpTest,
       CreateEp_VirtualNpu_InjectsLockdownAotDefaultsAndTarget) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  QnnEpFactory factory("QNNExecutionProvider", ctx.MakeApiPtrs());

  OrtHardwareDevice* virtual_npu = MakeFakeHwDevice(25);
  ctx.device_type_map[virtual_npu] = OrtHardwareDeviceType_NPU;
  ctx.device_vendor_map[virtual_npu] = kQualcommVendorId;
  ctx.virtual_devices.insert(virtual_npu);
  FactoryStubContext::KeyValues metadata_values{{"target_architecture", "60"}};

  auto* fake_session_opts = reinterpret_cast<OrtSessionOptions*>(kFakeToken);
  auto* fake_logger = reinterpret_cast<OrtLogger*>(kFakeToken);
  auto* fake_metadata = reinterpret_cast<const OrtKeyValuePairs*>(&metadata_values);
  OrtEp* ep = nullptr;
  const OrtHardwareDevice* devices[] = {virtual_npu};
  const OrtKeyValuePairs* metadata[] = {fake_metadata};

  OrtStatus* status = factory.CreateEp(&factory, devices, metadata, 1,
                                       fake_session_opts, fake_logger, &ep);

  const auto has_entry = [&ctx](std::string_view suffix, std::string_view value) {
    return std::any_of(ctx.added_config_entries.begin(), ctx.added_config_entries.end(),
                       [suffix, value](const auto& entry) {
                         const std::string_view key = entry.first;
                         return key.size() >= suffix.size() &&
                                key.substr(key.size() - suffix.size()) == suffix && entry.second == value;
                       });
  };
  EXPECT_TRUE(has_entry("htp_arch", "73"));
  EXPECT_TRUE(has_entry("soc_model", "60"));
  EXPECT_TRUE(has_entry("ep.context_enable", "1"));
  EXPECT_TRUE(has_entry("enable_htp_prepare_only", "1"));
  EXPECT_TRUE(has_entry("enable_htp_cross_device_prepare", "1"));

  if (status == nullptr) {
    EXPECT_NE(ep, nullptr);
    factory.ReleaseEp(&factory, ep);
  } else {
    ctx.stub_ort_api.ReleaseStatus(status);
  }
}

#endif  // defined(_WIN32) && QNN_ARCH_ARM64 && QNN API >= 2.40


TEST_F(QnnUnit_ProviderFactoryTest, BackendLibraries_ActiveManagerSurvivesFactoryRelease) {
  FactoryStubContext ctx;
  UseFactoryStubs use(ctx);
  const Ort::Logger logger = MakeNullLogger();
  qnn::QnnBackendManagerConfig config{};
  auto manager = qnn::QnnBackendManager::Create(config, ctx.MakeApiPtrs(), logger);
  const std::weak_ptr<void> lifetime = manager->GetBackendLibraryLifetime();
  {
    QnnEpFactory factory("ep", ctx.MakeApiPtrs());
    factory.RegisterQnnBackendManager(manager);
    factory.UnloadBackendLibraries();
    EXPECT_FALSE(lifetime.expired());
  }
  EXPECT_FALSE(lifetime.expired());
  manager.reset();
  EXPECT_TRUE(lifetime.expired());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
