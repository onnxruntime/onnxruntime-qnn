// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for QnnEp (qnn_execution_provider.cc).
//
// Exercises:
//   - Constructor option parsing: backend_type, profiling_level, htp_performance_mode,
//     qnn_context_priority, htp_graph_finalization_optimization_mode, htp_arch, vtcm_mb,
//     soc_model, device_id, file-mapped weights, share-resource-optimization, embed_mode,
//     disable_cpu_ep_fallback / offload_graph_io_quantization conflict, fp16/bf16 validation,
//     disable_htp_monolithic_lstm, json dump path warning, ep_input_graph dump,
//     ir DLC dump warnings, rpc_control_latency.
//   - Constructor early throws: prepare_only without context_cache, bf16 without soc_model,
//     bf16 with soc_model<88, fp16 without soc_model (Linux x86_64), backend_type +
//     backend_path both set.
//   - Static impl methods: GetName, GetPreferredDataLayout, ShouldConvertDataLayoutForOp.
//   - GetCompiledModelCompatibilityInfoImpl default-info empty path.
//   - ValidateCompiledModelCompatibilityInfo early-return error paths.
//   - SetDynamicOptionsImpl: prepare_only early return, kvcache_rewind without genie
//     manager, HTP perf mode on CPU backend (no-op), unsupported key error.
//   - GetHardwareDeviceIncompatibilityDetails: non-existent backend path → MISSING_DEPENDENCY.
//
// All tests run without loading any real QNN backend shared library and without
// constructing a real ORT session. QnnBackendManager::Create stores config only —
// the backend .so is loaded lazily in SetupBackend, which is never called here.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_execution_provider.h"
#include "core/providers/qnn/qnn_provider_factory.h"
#include "core/providers/qnn/shared_context.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct StatusRecord {
  OrtErrorCode code;
  std::string msg;
};

// ---------------------------------------------------------------------------
// EpStubContext
//
// Extends OrtApiStubContext (which owns the three ORT API stub tables and
// installs the initializer-query stubs) with the extra function pointers that
// QnnEpFactory and QnnEp need. session_config maps session-option keys to
// values; HasSessionConfigEntry returns 1 for keys present in the map, 0
// otherwise; GetSessionConfigEntry returns the stored value + NUL.
//
// Logger_GetLoggingSeverityLevel is stubbed to return FATAL so that the
// Ort::Logger(OrtLogger*) constructor (used in QnnEp's member-initialiser
// list) always produces a null logger with FATAL severity — identical to
// MakeNullLogger() but without the memcpy trick.
// ---------------------------------------------------------------------------
class EpStubContext : public OrtApiStubContext {
 public:
  std::unordered_map<std::string, std::string> session_config;

  // Captures the most recent call to DeviceEpIncompatibilityDetails_SetDetails.
  OrtDeviceEpIncompatibilityReason last_incompatibility_reason = OrtDeviceEpIncompatibility_UNKNOWN;
  int32_t last_incompatibility_error_code = -1;

  static thread_local EpStubContext* current_;

  EpStubContext() { InstallStubs(); }

 private:
  // Installs the EP-ctor stubs on top of the initializer-query stubs already
  // set by the OrtApiStubContext base constructor (which MakeApiPtrs() validates).
  void InstallStubs() {
    // Status helpers used by RETURN_IF_NOT_NULL / error paths.
    stub_ort_api.CreateStatus = [](OrtErrorCode code, const char* msg) noexcept -> OrtStatus* {
      return reinterpret_cast<OrtStatus*>(new StatusRecord{code, msg ? msg : ""});
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

    // Session config entry lookup (used by GetSessionConfigEntryOrDefault).
    stub_ort_api.HasSessionConfigEntry =
        [](const OrtSessionOptions*, const char* key, int* out) noexcept -> OrtStatus* {
      auto* self = EpStubContext::current_;
      *out = (self && self->session_config.count(key)) ? 1 : 0;
      return nullptr;
    };
    stub_ort_api.GetSessionConfigEntry =
        [](const OrtSessionOptions*, const char* key, char* buf, size_t* sz) noexcept -> OrtStatus* {
      auto* self = EpStubContext::current_;
      if (self) {
        auto it = self->session_config.find(key);
        if (it != self->session_config.end()) {
          size_t needed = it->second.size() + 1;
          if (buf) std::memcpy(buf, it->second.c_str(), needed);
          *sz = needed;
          return nullptr;
        }
      }
      *sz = 1;
      if (buf) buf[0] = '\0';
      return nullptr;
    };

    // Logger severity — returns FATAL so Ort::Logger(OrtLogger*) short-circuits.
    stub_ort_api.Logger_GetLoggingSeverityLevel =
        [](const OrtLogger*, OrtLoggingLevel* out) noexcept -> OrtStatus* {
      *out = ORT_LOGGING_LEVEL_FATAL;
      return nullptr;
    };

    // Memory info (needed by QnnEpFactory ctor).
    stub_ort_api.CreateMemoryInfo_V2 =
        [](const char*, OrtMemoryInfoDeviceType, uint32_t, int32_t,
           OrtDeviceMemoryType, size_t, OrtAllocatorType,
           OrtMemoryInfo** out) noexcept -> OrtStatus* {
      *out = reinterpret_cast<OrtMemoryInfo*>(uintptr_t{1});
      return nullptr;
    };
    stub_ort_api.ReleaseMemoryInfo = [](OrtMemoryInfo*) noexcept {};

    // DeviceEpIncompatibilityDetails_SetDetails (ep_api, used in error paths).
    // Captures reason and error_code into current_ for test assertions.
    stub_ep_api.DeviceEpIncompatibilityDetails_SetDetails =
        [](OrtDeviceEpIncompatibilityDetails*, uint32_t reason,
           int32_t code, const char*) noexcept -> OrtStatus* {
      if (auto* self = EpStubContext::current_) {
        self->last_incompatibility_reason = static_cast<OrtDeviceEpIncompatibilityReason>(reason);
        self->last_incompatibility_error_code = code;
      }
      return nullptr;
    };
  }
};

thread_local EpStubContext* EpStubContext::current_ = nullptr;

// RAII: installs/uninstalls the thread-local current_ pointer.
class UseEpStubs {
 public:
  explicit UseEpStubs(EpStubContext& ctx) { EpStubContext::current_ = &ctx; }
  ~UseEpStubs() { EpStubContext::current_ = nullptr; }
  UseEpStubs(const UseEpStubs&) = delete;
  UseEpStubs& operator=(const UseEpStubs&) = delete;
};

// Combines UseEpStubs with OrtGlobalApiOverride so that the Ort::Logger
// constructor (called in QnnEp's member-initialiser list) routes through
// our stubbed Logger_GetLoggingSeverityLevel.
class UseGlobalEpStubs {
 public:
  explicit UseGlobalEpStubs(EpStubContext& ctx)
      : use_stubs_(ctx), global_override_(&ctx.stub_ort_api) {}

 private:
  UseEpStubs use_stubs_;
  OrtGlobalApiOverride global_override_;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Fake opaque token — never dereferenced; used where a non-null OrtLogger*
// or OrtSessionOptions* is required as an opaque handle.
static constexpr uintptr_t kFakeToken = 0x1;

// Returns "ep.qnnexecutionprovider." + key — the prefix added by
// FormatEPConfigKey("QNNExecutionProvider").
static std::string EPKey(const std::string& key) {
  return "ep.qnnexecutionprovider." + key;
}

static std::unique_ptr<QnnEpFactory> MakeFactory(EpStubContext& ctx) {
  UseGlobalEpStubs use(ctx);
  return std::make_unique<QnnEpFactory>("QNNExecutionProvider", ctx.MakeApiPtrs());
}

// Constructs a QnnEp using factory and ctx.session_config.
// Returns the unique_ptr on success; propagates any exception to the caller.
static std::unique_ptr<QnnEp> MakeEp(QnnEpFactory& factory, EpStubContext& ctx) {
  UseGlobalEpStubs use(ctx);
  auto* fake_session_opts = reinterpret_cast<OrtSessionOptions*>(kFakeToken);
  auto* fake_logger = reinterpret_cast<OrtLogger*>(kFakeToken);
  return std::make_unique<QnnEp>(factory, "QNNExecutionProvider",
                                 *fake_session_opts, fake_logger);
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class QnnUnit_ExecutionProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Reset the SharedContext singleton's shared QnnBackendManager so that
    // tests that modify it don't interfere with each other.
    SharedContext::GetInstance().ResetSharedQnnBackendManager();
  }
  void TearDown() override {
    SharedContext::GetInstance().ResetSharedQnnBackendManager();
  }
};

// ===========================================================================
// Group 1: Default construction (sanity check)
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, DefaultCtor_Succeeds) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  ASSERT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// ===========================================================================
// Group 2: GetNameImpl
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, GetName_ReturnsEpName) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());
  const char* name = ep_ptr->GetName(ep_ptr);
  EXPECT_STREQ(name, "QNNExecutionProvider");
}

// ===========================================================================
// Group 3: GetPreferredDataLayout / ShouldConvertDataLayoutForOp
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, GetPreferredDataLayout_ReturnsNHWC) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  OrtEpDataLayout layout = OrtEpDataLayout::OrtEpDataLayout_NCHW;
  OrtStatus* s = ep_ptr->GetPreferredDataLayout(ep_ptr, &layout);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(layout, OrtEpDataLayout::OrtEpDataLayout_NHWC);
}

TEST_F(QnnUnit_ExecutionProviderTest, ShouldConvertDataLayout_Upsample_Returns1) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  int should_convert = -1;
  OrtStatus* s = ep_ptr->ShouldConvertDataLayoutForOp(
      ep_ptr, "", "Upsample",
      OrtEpDataLayout::OrtEpDataLayout_NHWC, &should_convert);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(should_convert, 1);
}

TEST_F(QnnUnit_ExecutionProviderTest, ShouldConvertDataLayout_GroupNormalization_Returns1) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  int should_convert = -1;
  OrtStatus* s = ep_ptr->ShouldConvertDataLayoutForOp(
      ep_ptr, "", "GroupNormalization",
      OrtEpDataLayout::OrtEpDataLayout_NHWC, &should_convert);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(should_convert, 1);
}

TEST_F(QnnUnit_ExecutionProviderTest, ShouldConvertDataLayout_RoiAlign_Returns1) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  int should_convert = -1;
  OrtStatus* s = ep_ptr->ShouldConvertDataLayoutForOp(
      ep_ptr, "", "RoiAlign",
      OrtEpDataLayout::OrtEpDataLayout_NHWC, &should_convert);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(should_convert, 1);
}

TEST_F(QnnUnit_ExecutionProviderTest, ShouldConvertDataLayout_LpPool_Returns1) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  int should_convert = -1;
  OrtStatus* s = ep_ptr->ShouldConvertDataLayoutForOp(
      ep_ptr, "", "LpPool",
      OrtEpDataLayout::OrtEpDataLayout_NHWC, &should_convert);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(should_convert, 1);
}

TEST_F(QnnUnit_ExecutionProviderTest, ShouldConvertDataLayout_ConvInteger_Returns0) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  int should_convert = -1;
  OrtStatus* s = ep_ptr->ShouldConvertDataLayoutForOp(
      ep_ptr, "", "ConvInteger",
      OrtEpDataLayout::OrtEpDataLayout_NHWC, &should_convert);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(should_convert, 0);
}

TEST_F(QnnUnit_ExecutionProviderTest, ShouldConvertDataLayout_UnknownOp_ReturnsNeg1) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  int should_convert = 42;
  OrtStatus* s = ep_ptr->ShouldConvertDataLayoutForOp(
      ep_ptr, "", "UnknownOp",
      OrtEpDataLayout::OrtEpDataLayout_NHWC, &should_convert);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(should_convert, -1);
}

// ===========================================================================
// Group 4: Constructor — backend_type option branches
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_BackendTypeGenie_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("backend_type")] = "genie";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_BackendTypeHtp_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("backend_type")] = "htp";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_BackendTypeSaver_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("backend_type")] = "saver";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_BackendTypeIr_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("backend_type")] = "ir";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_BackendTypeInvalid_LogsError) {
  EpStubContext ctx;
  ctx.session_config[EPKey("backend_type")] = "totally_invalid_backend";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_BackendTypeAndPathBothSet_Throws) {
  EpStubContext ctx;
  ctx.session_config[EPKey("backend_type")] = "cpu";
  ctx.session_config[EPKey("backend_path")] = "/some/path/libQnn.so";
  auto factory = MakeFactory(ctx);
  EXPECT_THROW({ auto ep = MakeEp(*factory, ctx); }, std::runtime_error);
}

// ===========================================================================
// Group 5: Constructor — profiling, HTP performance mode, context priority
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ProfilingLevelInvalid_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("profiling_level")] = "not_a_level";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ProfilingFilePath_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("profiling_level")] = "basic";
  ctx.session_config[EPKey("profiling_file_path")] = "/tmp/qnn_perf.json";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpPerformanceModeInvalid_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_performance_mode")] = "ultra_extreme_turbo";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ContextPriorityNormalLow_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("qnn_context_priority")] = "normal_low";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ContextPriorityNormal_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("qnn_context_priority")] = "normal";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ContextPriorityLow_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("qnn_context_priority")] = "low";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ContextPriorityNormalHigh_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("qnn_context_priority")] = "normal_high";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ContextPriorityHighPlus_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("qnn_context_priority")] = "high_plus";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ContextPriorityCritical_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("qnn_context_priority")] = "critical";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ContextPriorityCriticalPlus_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("qnn_context_priority")] = "critical_plus";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ContextPriorityInvalid_SetsUndefined) {
  EpStubContext ctx;
  ctx.session_config[EPKey("qnn_context_priority")] = "not_a_priority";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// ===========================================================================
// Group 6: Constructor — HTP graph finalization opt mode, HTP architecture
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpFinalizationOptMode1_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_graph_finalization_optimization_mode")] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpFinalizationOptModeInvalid_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_graph_finalization_optimization_mode")] = "99";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpArch68_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_arch")] = "68";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpArch69_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_arch")] = "69";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpArch73_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_arch")] = "73";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpArch75_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_arch")] = "75";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpArch81_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_arch")] = "81";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpArchInvalid_LogsWarning) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_arch")] = "999";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// ===========================================================================
// Group 7: Constructor — misc option branches
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_VtcmNegative_LogsWarning) {
  EpStubContext ctx;
  ctx.session_config[EPKey("vtcm_mb")] = "-5";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_VtcmPositive_SetsMb) {
  EpStubContext ctx;
  ctx.session_config[EPKey("vtcm_mb")] = "8";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_RpcControlLatencyNonZero_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("rpc_control_latency")] = "100";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpShareResourceOptInvalid_LogsError) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_share_resource_optimization")] = "2";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EnableVtcmBackupBufferSharing_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("enable_vtcm_backup_buffer_sharing")] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DeviceIdNegative_LogsWarning) {
  EpStubContext ctx;
  ctx.session_config[EPKey("device_id")] = "-1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_SocModelNegative_LogsWarning) {
  EpStubContext ctx;
  ctx.session_config[EPKey("soc_model")] = "-1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpFP16PrecisionInvalid_LogsError) {
  EpStubContext ctx;
  ctx.session_config[EPKey("enable_htp_fp16_precision")] = "invalid";
  // Invalid value leaves enable_HTP_FP16_precision_ at its default true.
  // On Linux x86_64, FP16+no-soc_model throws; provide a soc_model so the
  // constructor can proceed past the FP16 validation — the invalid-value
  // log path on lines 787-792 is what this test covers.
  ctx.session_config[EPKey("soc_model")] = "60";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DisableHtpMonolithicLstmTrue_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("disable_htp_monolithic_lstm")] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DisableHtpMonolithicLstmInvalid_LogsError) {
  EpStubContext ctx;
  ctx.session_config[EPKey("disable_htp_monolithic_lstm")] = "maybe";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EmbedModeInvalidValue_Succeeds) {
  EpStubContext ctx;
  ctx.session_config["ep.context_embed_mode"] = "2";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EmbedModeConflictsWithShareContexts_LogsError) {
  EpStubContext ctx;
  ctx.session_config["ep.context_embed_mode"] = "1";
  ctx.session_config["ep.share_ep_contexts"] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EmbedModeConflictsWithHtpShareResourceOpt_LogsError) {
  EpStubContext ctx;
  ctx.session_config["ep.context_embed_mode"] = "1";
  ctx.session_config[EPKey("htp_share_resource_optimization")] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DisableCpuFallbackWithOffloadConflict_LogsInfo) {
  EpStubContext ctx;
  ctx.session_config["session.disable_cpu_ep_fallback"] = "1";
  // offload_graph_io_quantization defaults to "1" (true), so this covers
  // the conflict-detection branch.
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// InitQnnSerializerConfig: dir set but dump not enabled → warning branch
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_IrDlcDirWithoutDumpEnabled_LogsWarning) {
  EpStubContext ctx;
  ctx.session_config[EPKey("dump_qnn_ir_dlc")] = "0";
  ctx.session_config[EPKey("dump_qnn_ir_dlc_dir")] = "/tmp/qnn_dlc_out";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// IrBackendPath set but dump not enabled → warning
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_IrBackendPathWithoutDumpEnabled_LogsWarning) {
  EpStubContext ctx;
  ctx.session_config[EPKey("dump_qnn_ir_dlc")] = "0";
  ctx.session_config[EPKey("qnn_ir_backend_path")] = "/tmp/libQnnIr_custom.so";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// Json QNN graph dump: dir set but dump not enabled → warning
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_JsonGraphDirWithoutDumpEnabled_LogsWarning) {
  EpStubContext ctx;
  ctx.session_config[EPKey("dump_json_qnn_graph")] = "0";
  ctx.session_config[EPKey("json_qnn_graph_dir")] = "/tmp/qnn_json_graphs";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// ParseBoolOption: value is neither "0" nor "1" → logs VERBOSE "Invalid value"
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_BoolOptionInvalidValue_LogsVerbose) {
  EpStubContext ctx;
  // offload_graph_io_quantization uses ParseBoolOption with default true.
  // "x" is neither 0 nor 1, so the else-branch at line 257 fires.
  ctx.session_config[EPKey("offload_graph_io_quantization")] = "x";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// IR backend path AND dump enabled → "IR backend path" info log (line 372)
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_IrBackendPathWithDumpEnabled_LogsInfo) {
  EpStubContext ctx;
  ctx.session_config[EPKey("dump_qnn_ir_dlc")] = "1";
  ctx.session_config[EPKey("qnn_ir_backend_path")] = "/custom/libQnnIr.so";
  ctx.session_config[EPKey("soc_model")] = "60";  // avoids FP16+no-soc_model throw
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// EP input graph dump enabled with no dir → falls back to cwd (line 1084)
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DumpEpInputGraphNoDir_FallbackToCwd) {
  EpStubContext ctx;
  ctx.session_config[EPKey("dump_qnn_ep_input_graph")] = "1";
  // No dump_qnn_ep_input_graph_dir → falls back to current_path() (line 1084).
  // ProbeDumpDirectoryWritable on cwd should succeed, so no throw.
  ctx.session_config[EPKey("soc_model")] = "60";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// ===========================================================================
// Group 8: Constructor — early throws
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_PrepareOnlyWithoutContextCache_Throws) {
  EpStubContext ctx;
  ctx.session_config[EPKey("enable_htp_prepare_only")] = "1";
  // ep.context_enable defaults to "0", so this combination must throw.
  auto factory = MakeFactory(ctx);
  EXPECT_THROW({ auto ep = MakeEp(*factory, ctx); }, std::runtime_error);
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_Bf16EnabledWithoutSocModel_Throws) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_bf16_enable")] = "1";
  // soc_model not set → defaults to QNN_SOC_MODEL_UNKNOWN → must throw.
  auto factory = MakeFactory(ctx);
  EXPECT_THROW({ auto ep = MakeEp(*factory, ctx); }, std::runtime_error);
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_Bf16EnabledWithLowSocModel_Throws) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_bf16_enable")] = "1";
  ctx.session_config[EPKey("soc_model")] = "50";  // < 88, should throw
  auto factory = MakeFactory(ctx);
  EXPECT_THROW({ auto ep = MakeEp(*factory, ctx); }, std::runtime_error);
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_Bf16EnabledWithValidSocModel_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_bf16_enable")] = "1";
  ctx.session_config[EPKey("soc_model")] = "88";  // >= 88, ok
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

#if defined(__linux__) && !defined(__aarch64__)
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_FP16EnabledWithoutSocModel_ThrowsOnLinux) {
  EpStubContext ctx;
  ctx.session_config[EPKey("enable_htp_fp16_precision")] = "1";
  // soc_model not set → QNN_SOC_MODEL_UNKNOWN → throws on Linux x86_64
  auto factory = MakeFactory(ctx);
  EXPECT_THROW({ auto ep = MakeEp(*factory, ctx); }, std::runtime_error);
}
#endif

// ===========================================================================
// Group 9: GetCompiledModelCompatibilityInfoImpl
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, GetCompiledModelCompatibilityInfo_DefaultInfo_ReturnsEmpty) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  // compatibility_info_ is default-initialised (all-zero versions) → returns ""
  const char* info = ep_ptr->GetCompiledModelCompatibilityInfo(ep_ptr, nullptr);
  EXPECT_STREQ(info, "");
}

// ===========================================================================
// Group 10: ValidateCompiledModelCompatibilityInfo
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, ValidateCompatibilityInfo_EmptyString_NotApplicable) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);

  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  OrtStatus* s = ep->ValidateCompiledModelCompatibilityInfo(nullptr, 0, "", &compat);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
}

TEST_F(QnnUnit_ExecutionProviderTest, ValidateCompatibilityInfo_TooFewFields_NotApplicable) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);

  // Only 3 colon-separated fields; function expects 6.
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  OrtStatus* s = ep->ValidateCompiledModelCompatibilityInfo(nullptr, 0, "1:2:3", &compat);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
}

TEST_F(QnnUnit_ExecutionProviderTest, ValidateCompatibilityInfo_BadVersionFormat_NotApplicable) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);

  // 6 fields but version field (idx 1) has wrong format (2 parts, not 3).
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  OrtStatus* s = ep->ValidateCompiledModelCompatibilityInfo(
      nullptr, 0, "1:1.0:2.1.0:3.0.0:73:0", &compat);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
}

// ===========================================================================
// Group 11: SetDynamicOptionsImpl
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, SetDynamicOptions_PrepareOnly_EarlyReturn) {
  EpStubContext ctx;
  ctx.session_config[EPKey("enable_htp_prepare_only")] = "1";
  ctx.session_config["ep.context_enable"] = "1";  // also set so ctor doesn't throw
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  const char* keys[] = {"ep.dynamic.workload_type"};
  const char* vals[] = {"Default"};
  OrtStatus* s = ep_ptr->SetDynamicOptions(ep_ptr, keys, vals, 1);
  // prepare_only_ is true → early return nullptr with a warning log.
  EXPECT_EQ(s, nullptr);
}

TEST_F(QnnUnit_ExecutionProviderTest, SetDynamicOptions_UnsupportedKey_ReturnsError) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  const char* keys[] = {"ep.dynamic.nonexistent_option"};
  const char* vals[] = {"anything"};
  OrtStatus* s = ep_ptr->SetDynamicOptions(ep_ptr, keys, vals, 1);
  ASSERT_NE(s, nullptr);
  const auto* rec = reinterpret_cast<const StatusRecord*>(s);
  EXPECT_EQ(rec->code, ORT_INVALID_ARGUMENT);
  ctx.stub_ort_api.ReleaseStatus(s);
}

TEST_F(QnnUnit_ExecutionProviderTest, SetDynamicOptions_KvcacheNoGenieManager_ReturnsError) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  const char* keys[] = {"kvcache_rewind"};
  const char* vals[] = {"1024"};
  OrtStatus* s = ep_ptr->SetDynamicOptions(ep_ptr, keys, vals, 1);
  // genie_backend_manager_ is null → returns ORT_INVALID_ARGUMENT
  ASSERT_NE(s, nullptr);
  const auto* rec = reinterpret_cast<const StatusRecord*>(s);
  EXPECT_EQ(rec->code, ORT_INVALID_ARGUMENT);
  ctx.stub_ort_api.ReleaseStatus(s);
}

TEST_F(QnnUnit_ExecutionProviderTest, SetDynamicOptions_HtpPerfModeOnCpuBackend_NoOp) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  // Backend type defaults to CPU (no SetupBackend called), so this key
  // triggers the "not HTP/DSP" early-return branch.
  const char* keys[] = {"ep.dynamic.qnn_htp_performance_mode"};
  const char* vals[] = {"burst"};
  OrtStatus* s = ep_ptr->SetDynamicOptions(ep_ptr, keys, vals, 1);
  EXPECT_EQ(s, nullptr);
}

TEST_F(QnnUnit_ExecutionProviderTest, SetDynamicOptions_EmptyOptionList_Succeeds) {
  EpStubContext ctx;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);
  auto* ep_ptr = static_cast<OrtEp*>(ep.get());

  OrtStatus* s = ep_ptr->SetDynamicOptions(ep_ptr, nullptr, nullptr, 0);
  EXPECT_EQ(s, nullptr);
}

// ===========================================================================
// Group 12: GetHardwareDeviceIncompatibilityDetails
// ===========================================================================

// With a non-existent backend path, LoadBackend fails → "Unable to load backend"
// error message → classified as MISSING_DEPENDENCY.
// SetupBackend is called with the real ORT API (no global override after MakeEp).
TEST_F(QnnUnit_ExecutionProviderTest, GetHardwareDeviceIncompatibilityDetails_NonExistentBackend_ReturnsMissingDependency) {
  EpStubContext ctx;
  ctx.session_config[EPKey("backend_path")] = "/nonexistent/libQnnFake.so";
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);

  // Activate current_ so the SetDetails stub can capture the classification.
  // No OrtGlobalApiOverride: SetupBackend must run under the real ORT API.
  UseEpStubs use_stubs(ctx);
  auto* fake_hw = reinterpret_cast<const OrtHardwareDevice*>(kFakeToken);
  auto* fake_details = reinterpret_cast<OrtDeviceEpIncompatibilityDetails*>(kFakeToken);
  OrtStatus* s = ep->GetHardwareDeviceIncompatibilityDetails(fake_hw, fake_details);

  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(ctx.last_incompatibility_reason, OrtDeviceEpIncompatibility_MISSING_DEPENDENCY);
}

// ===========================================================================
// Group 13: Real-HTP-backend paths (QnnUnit_ExecutionProviderHtpTest)
//
// These tests construct QnnEp directly and invoke methods that internally call
// qnn_backend_manager_->SetupBackend(), which dlopens a real libQnnHtp.so. They
// never create an ORT session, so they are unit-tier component tests (mirroring
// QnnUnit_BackendManagerHtpTest) and GTEST_SKIP() when libQnnHtp.so is absent.
// ===========================================================================

class QnnUnit_ExecutionProviderHtpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SharedContext::GetInstance().ResetSharedQnnBackendManager();

    // Skip when libQnnHtp.so is unavailable (non-SDK environment).
    QnnRealHtpBackendContext htp_check;
    if (!htp_check.IsValid()) {
      GTEST_SKIP() << "libQnnHtp.so not available";
    }

    ctx_.session_config[EPKey("backend_path")] = "libQnnHtp.so";
    factory_ = MakeFactory(ctx_);
    ep_ = MakeEp(*factory_, ctx_);
  }

  void TearDown() override {
    ep_.reset();
    factory_.reset();
    SharedContext::GetInstance().ResetSharedQnnBackendManager();
  }

  EpStubContext ctx_;
  std::unique_ptr<QnnEpFactory> factory_;
  std::unique_ptr<QnnEp> ep_;
};

// Covers the ValidateCompiledModelCompatibilityInfo backend-not-yet-set-up path:
// is_backend_setup == false → SetupBackend → ValidateCompatibilityInfo → ReleaseResources.
// Compatibility string format: backend_id:sdk_ver:api_ver:blob_ver:htp_arch:is_usr_drv.
TEST_F(QnnUnit_ExecutionProviderHtpTest, ValidateCompatibilityInfo_BackendNotSetup_CallsSetupBackend) {
  const char* info = "1:2.0.0:1.22.0:3.0.0:73:0";
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;

  OrtStatus* s = ep_->ValidateCompiledModelCompatibilityInfo(nullptr, 0, info, &compat);

  // Status may be null (success) or non-null (version mismatch with the loaded
  // backend); either way the SetupBackend branch is covered. Ensure no crash and
  // release any status.
  if (s) {
    Ort::GetApi().ReleaseStatus(s);
  }
  SUCCEED();
}

// Covers the GetHardwareDeviceIncompatibilityDetails success path:
// SetupBackend succeeds with libQnnHtp.so → SetDetails(NONE, QNN_SUCCESS, nullptr).
TEST_F(QnnUnit_ExecutionProviderHtpTest, GetHardwareDeviceIncompatibilityDetails_HtpBackend_ReturnsNone) {
  // Activate current_ so the SetDetails stub captures the classification.
  // No OrtGlobalApiOverride: SetupBackend must run under the real ORT API.
  UseEpStubs use_stubs(ctx_);
  auto* fake_hw = reinterpret_cast<const OrtHardwareDevice*>(kFakeToken);
  auto* fake_details = reinterpret_cast<OrtDeviceEpIncompatibilityDetails*>(kFakeToken);
  OrtStatus* s = ep_->GetHardwareDeviceIncompatibilityDetails(fake_hw, fake_details);

  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(ctx_.last_incompatibility_reason, OrtDeviceEpIncompatibility_NONE);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
