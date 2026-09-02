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
//     enable_htp_monolithic_lstm, json dump path warning, ep_input_graph dump,
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

struct LogRecord {
  OrtLoggingLevel severity;
  std::string message;
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
// Logger_GetLoggingSeverityLevel reports current_->log_severity, which the
// Ort::Logger(OrtLogger*) constructor (used in QnnEp's member-initialiser list)
// caches once. It defaults to FATAL so ORT_CXX_LOG calls short-circuit on the
// severity gate and never dereference the fake logger token — the safe default
// for EP methods invoked outside a stub scope. Tests that assert on ctor logging
// set ctx.log_severity = VERBOSE before MakeEp; the ctor's log statements then
// pass the gate and route through Logger_LogMessage, which records
// (severity, message) into log_records for ExpectLogged() assertions.
// ---------------------------------------------------------------------------
class EpStubContext : public OrtApiStubContext {
 public:
  std::unordered_map<std::string, std::string> session_config;

  // Captures the most recent call to DeviceEpIncompatibilityDetails_SetDetails.
  OrtDeviceEpIncompatibilityReason last_incompatibility_reason = OrtDeviceEpIncompatibility_UNKNOWN;
  int32_t last_incompatibility_error_code = -1;

  // Records every ORT_CXX_LOG emitted through this stub's Logger_LogMessage.
  std::vector<LogRecord> log_records;

  // Severity that Logger_GetLoggingSeverityLevel reports — cached once by the
  // Ort::Logger(OrtLogger*) ctor. Defaults to FATAL so that ORT_CXX_LOG calls
  // short-circuit on the severity gate and never reach Logger_LogMessage (the
  // safe default: EP methods invoked outside a stub scope, e.g. under the real
  // ORT API, then log nothing and cannot dereference the fake logger token).
  // Tests that assert on ctor logging set this to VERBOSE *before* MakeEp so the
  // ctor's log statements route through the recording sink.
  OrtLoggingLevel log_severity = ORT_LOGGING_LEVEL_FATAL;

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

    // Logger severity — returns current_->log_severity (default FATAL). Tests
    // that assert on ctor logging set ctx.log_severity = VERBOSE before MakeEp
    // so the Ort::Logger ctor caches VERBOSE and ctor logs reach Logger_LogMessage.
    stub_ort_api.Logger_GetLoggingSeverityLevel =
        [](const OrtLogger*, OrtLoggingLevel* out) noexcept -> OrtStatus* {
      auto* self = EpStubContext::current_;
      *out = self ? self->log_severity : ORT_LOGGING_LEVEL_FATAL;
      return nullptr;
    };

    // Logger_LogMessage — records (severity, message) into current_->log_records.
    stub_ort_api.Logger_LogMessage =
        [](const OrtLogger*, OrtLoggingLevel severity, const char* message,
           const ORTCHAR_T*, int, const char*) noexcept -> OrtStatus* {
      if (auto* self = EpStubContext::current_) {
        self->log_records.push_back({severity, message ? message : ""});
      }
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

// Asserts that ctx.log_records contains at least one entry at the given
// severity whose message contains substr. On failure prints all captured
// records to aid debugging.
static void ExpectLogged(const EpStubContext& ctx, OrtLoggingLevel severity,
                         const std::string& substr) {
  for (const auto& rec : ctx.log_records) {
    if (rec.severity == severity && rec.message.find(substr) != std::string::npos) {
      return;
    }
  }
  std::string dump;
  for (const auto& rec : ctx.log_records) {
    dump += "\n  [sev=" + std::to_string(rec.severity) + "] " + rec.message;
  }
  ADD_FAILURE() << "No log at severity " << severity << " containing \"" << substr
                << "\". Captured records:" << dump;
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
//
// Assertion strength (Groups 4-7): QnnEp exposes no public getter for parsed
// constructor options. Options whose only effect is a log line are asserted via
// the EpStubContext log sink — `_Logs{Error,Warning,Info,Verbose}` tests call
// ExpectLogged() to verify the exact severity + message. Early throws are
// asserted with EXPECT_THROW (Group 8); error codes in Groups 10-12. The
// remaining `_Succeeds` tests exercise a parse branch whose result is neither
// logged nor otherwise observable through the public surface (parsing into a
// private member with no log), so they can only assert that construction does
// not throw — a deliberate component-level plateau, since adding a getter purely
// for tests would break EP encapsulation.

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
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("backend_type")] = "totally_invalid_backend";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_ERROR, "Failed to parse 'backend_type' value.");
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

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_ContextPriorityInvalid_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("qnn_context_priority")] = "not_a_priority";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// ---------------------------------------------------------------------------
// Parameterized: constructor accepts each valid enum value for an option
// (htp_performance_mode, qnn_context_priority, htp_arch) without throwing.
// Invalid values are covered by the dedicated TEST_F cases above / below.
// ---------------------------------------------------------------------------
class QnnUnit_EpCtorValidOptionTest
    : public QnnUnit_ExecutionProviderTest,
      public ::testing::WithParamInterface<std::pair<std::string, std::string>> {};

TEST_P(QnnUnit_EpCtorValidOptionTest, Ctor_ValidOptionValue_Succeeds) {
  const auto& [key, value] = GetParam();
  EpStubContext ctx;
  ctx.session_config[EPKey(key)] = value;
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

// Uses the option value as the test-name suffix (values are valid name fragments).
static std::string OptionValueSuffix(
    const ::testing::TestParamInfo<std::pair<std::string, std::string>>& info) {
  return info.param.second;
}

INSTANTIATE_TEST_SUITE_P(
    HtpPerformanceMode, QnnUnit_EpCtorValidOptionTest,
    ::testing::Values(
        std::make_pair("htp_performance_mode", "balanced"),
        std::make_pair("htp_performance_mode", "default"),
        std::make_pair("htp_performance_mode", "high_performance"),
        std::make_pair("htp_performance_mode", "high_power_saver"),
        std::make_pair("htp_performance_mode", "low_balanced"),
        std::make_pair("htp_performance_mode", "low_power_saver"),
        std::make_pair("htp_performance_mode", "power_saver")),
    OptionValueSuffix);

INSTANTIATE_TEST_SUITE_P(
    QnnContextPriority, QnnUnit_EpCtorValidOptionTest,
    ::testing::Values(
        std::make_pair("qnn_context_priority", "normal_low"),
        std::make_pair("qnn_context_priority", "normal"),
        std::make_pair("qnn_context_priority", "low"),
        std::make_pair("qnn_context_priority", "normal_high"),
        std::make_pair("qnn_context_priority", "high_plus"),
        std::make_pair("qnn_context_priority", "critical"),
        std::make_pair("qnn_context_priority", "critical_plus")),
    OptionValueSuffix);

INSTANTIATE_TEST_SUITE_P(
    HtpArch, QnnUnit_EpCtorValidOptionTest,
    ::testing::Values(
        std::make_pair("htp_arch", "68"),
        std::make_pair("htp_arch", "69"),
        std::make_pair("htp_arch", "73"),
        std::make_pair("htp_arch", "75"),
        std::make_pair("htp_arch", "81")),
    OptionValueSuffix);

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

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpArchInvalid_LogsWarning) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("htp_arch")] = "999";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING, "Invalid HTP architecture: 999");
}

// ===========================================================================
// Group 7: Constructor — misc option branches
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_VtcmNegative_LogsWarning) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("vtcm_mb")] = "-5";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING, "Invalid vtcm_mb: -5 will be skipped");
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_VtcmPositive_Succeeds) {
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
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("htp_share_resource_optimization")] = "2";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_ERROR,
               "Invalid value entered for htp_share_resource_optimization: 2, only 1 is allowed.");
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EnableVtcmBackupBufferSharing_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("enable_vtcm_backup_buffer_sharing")] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DeviceIdNegative_LogsWarning) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("device_id")] = "-1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING, "Invalid device ID '-1', only >= 0 allowed.");
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_SocModelNegative_LogsWarning) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("soc_model")] = "-1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING, "Invalid soc_model: -1");
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpFP16PrecisionInvalid_LogsVerbose) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("enable_htp_fp16_precision")] = "invalid";
  // Invalid value leaves enable_HTP_FP16_precision_ at its default true.
  // On Linux x86_64, FP16+no-soc_model throws; provide a soc_model so the
  // constructor can proceed past the FP16 validation. The invalid value is
  // parsed by ParseBoolOption, which logs VERBOSE (not ERROR) in its else-branch.
  ctx.session_config[EPKey("soc_model")] = "60";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_VERBOSE,
               "Invalid value for ep.qnnexecutionprovider.enable_htp_fp16_precision");
}

// Covers the ctor WARNING that fires when enable_htp_fp16_clamp_overflow=1 but
// the QAIRT SDK in use lacks the backend enum. The test mirrors the source's
// #ifndef QNN_HTP_FP16_CLAMP_OVERFLOW_AVAILABLE guard so it self-disables when a
// future QAIRT uplevel defines the macro (see qnn_def.h for the version gate).
#ifndef QNN_HTP_FP16_CLAMP_OVERFLOW_AVAILABLE
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_HtpFp16ClampOverflowTrueOnUnsupportedSdk_LogsWarning) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("enable_htp_fp16_clamp_overflow")] = "1";
  ctx.session_config[EPKey("soc_model")] = "60";  // avoids FP16+no-soc_model throw
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING,
               "enable_htp_fp16_clamp_overflow was requested but the QNN HTP SDK in use does not support it");
}
#endif  // !QNN_HTP_FP16_CLAMP_OVERFLOW_AVAILABLE

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EnableHtpMonolithicLstmTrue_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("enable_htp_monolithic_lstm")] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EnableHtpMonolithicLstmInvalid_LogsVerbose) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("enable_htp_monolithic_lstm")] = "maybe";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_VERBOSE,
               "Invalid value for ep.qnnexecutionprovider.enable_htp_monolithic_lstm");
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_GpuPrecisionModeValidValues_Succeed) {
  // 0=FP32, 1=FP16, 2=HYBRID, 3=USER_PROVIDED (default).
  for (const char* mode : {"0", "1", "2", "3"}) {
    EpStubContext ctx;
    ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
    ctx.session_config[EPKey("gpu_precision_mode")] = mode;
    auto factory = MakeFactory(ctx);
    EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); }) << "gpu_precision_mode=" << mode;
    ExpectLogged(ctx, ORT_LOGGING_LEVEL_VERBOSE,
                 std::string("User specified gpu_precision_mode: ") + mode);
  }
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_GpuPrecisionModeInvalid_LogsError) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("gpu_precision_mode")] = "4";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_ERROR, "Invalid value for gpu_precision_mode: 4");
}

// The gpu_precision_mode string values are a public contract mapped onto
// QnnGpu_Precision_t. Pin the mapping so an SDK enum renumbering shows up as a
// local test failure rather than a silent behaviour change on device.
TEST_F(QnnUnit_ExecutionProviderTest, GpuPrecisionEnumValues_MatchDocumentedOrder) {
  EXPECT_EQ(QNN_GPU_PRECISION_FP32, static_cast<QnnGpu_Precision_t>(0));
  EXPECT_EQ(QNN_GPU_PRECISION_FP16, static_cast<QnnGpu_Precision_t>(1));
  EXPECT_EQ(QNN_GPU_PRECISION_HYBRID, static_cast<QnnGpu_Precision_t>(2));
  EXPECT_EQ(QNN_GPU_PRECISION_USER_PROVIDED, static_cast<QnnGpu_Precision_t>(3));
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DisableGpuFlagsValidValues_Succeed) {
  for (const char* key : {"disable_gpu_memory_optimizations",
                          "disable_gpu_node_optimizations",
                          "disable_gpu_queue_recording"}) {
    for (const char* value : {"0", "1"}) {
      EpStubContext ctx;
      ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
      ctx.session_config[EPKey(key)] = value;
      auto factory = MakeFactory(ctx);
      EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); }) << key << "=" << value;
      // ParseBoolOption echoes the parsed result, which also pins the documented
      // polarity: "1" disables the optimization, "0" (default) leaves it enabled.
      ExpectLogged(ctx, ORT_LOGGING_LEVEL_VERBOSE,
                   "User specified " + EPKey(key) + ": " + value);
    }
  }
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DisableGpuFlagsInvalid_LogVerbose) {
  // Parsed by ParseBoolOption, which logs VERBOSE and keeps the default
  // (false → optimizations enabled) for unrecognised values.
  for (const char* key : {"disable_gpu_memory_optimizations",
                          "disable_gpu_node_optimizations",
                          "disable_gpu_queue_recording"}) {
    EpStubContext ctx;
    ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
    ctx.session_config[EPKey(key)] = "true";
    auto factory = MakeFactory(ctx);
    EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); }) << key;
    ExpectLogged(ctx, ORT_LOGGING_LEVEL_VERBOSE, "Invalid value for " + EPKey(key));
    ExpectLogged(ctx, ORT_LOGGING_LEVEL_VERBOSE, "User specified " + EPKey(key) + ": 0");
  }
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EmbedModeInvalidValue_Succeeds) {
  EpStubContext ctx;
  ctx.session_config["ep.context_embed_mode"] = "2";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EmbedModeConflictsWithShareContexts_LogsError) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config["ep.context_embed_mode"] = "1";
  ctx.session_config["ep.share_ep_contexts"] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_ERROR,
               "Weight sharing enabled conflict with EP context embed mode.");
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_EmbedModeConflictsWithHtpShareResourceOpt_LogsError) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config["ep.context_embed_mode"] = "1";
  ctx.session_config[EPKey("htp_share_resource_optimization")] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_ERROR,
               "HTP share resource optimization enabled conflict with EP context embed mode.");
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DisableCpuFallbackWithOffloadConflict_LogsInfo) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config["session.disable_cpu_ep_fallback"] = "1";
  // offload_graph_io_quantization defaults to "1" (true), so this covers
  // the conflict-detection branch.
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_INFO, "Fallback to CPU EP is disabled");
}

// InitQnnSerializerConfig: dir set but dump not enabled → warning branch
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_IrDlcDirWithoutDumpEnabled_LogsWarning) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("dump_qnn_ir_dlc")] = "0";
  ctx.session_config[EPKey("dump_qnn_ir_dlc_dir")] = "/tmp/qnn_dlc_out";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING,
               "Provided a directory for dumping QNN graphs to DLC, but did not set dump_qnn_ir_dlc to 1.");
}

// IrBackendPath set but dump not enabled → warning
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_IrBackendPathWithoutDumpEnabled_LogsWarning) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("dump_qnn_ir_dlc")] = "0";
  ctx.session_config[EPKey("qnn_ir_backend_path")] = "/tmp/libQnnIr_custom.so";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING, "Provided a path to the Ir backend");
}

// Json QNN graph dump: dir set but dump not enabled → warning
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_JsonGraphDirWithoutDumpEnabled_LogsWarning) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("dump_json_qnn_graph")] = "0";
  ctx.session_config[EPKey("json_qnn_graph_dir")] = "/tmp/qnn_json_graphs";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING,
               "Provided a directory for dumping QNN JSON graphs, but did not enable dumping of QNN JSON graphs.");
}

// ParseBoolOption: value is neither "0" nor "1" → logs VERBOSE "Invalid value"
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_BoolOptionInvalidValue_LogsVerbose) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  // offload_graph_io_quantization uses ParseBoolOption with default true.
  // "x" is neither 0 nor 1, so the else-branch of ParseBoolOption fires.
  ctx.session_config[EPKey("offload_graph_io_quantization")] = "x";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_VERBOSE,
               "Invalid value for ep.qnnexecutionprovider.offload_graph_io_quantization");
}

// IR backend path AND dump enabled → "IR  backend path" info log in InitQnnSerializerConfig.
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_IrBackendPathWithDumpEnabled_LogsInfo) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  ctx.session_config[EPKey("dump_qnn_ir_dlc")] = "1";
  ctx.session_config[EPKey("qnn_ir_backend_path")] = "/custom/libQnnIr.so";
  ctx.session_config[EPKey("soc_model")] = "60";  // avoids FP16+no-soc_model throw
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_INFO, "IR  backend path: /custom/libQnnIr.so");
}

// EP input graph dump enabled with no dir → falls back to std::filesystem::current_path() in the QnnEp ctor.
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DumpEpInputGraphNoDir_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("dump_qnn_ep_input_graph")] = "1";
  // No dump_qnn_ep_input_graph_dir → falls back to current_path() in the QnnEp ctor.
  // ProbeDumpDirectoryWritable on cwd should succeed, so no throw.
  ctx.session_config[EPKey("soc_model")] = "60";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_VtcmMbMalformed_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("vtcm_mb")] = "abc";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_SocModelMalformed_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("soc_model")] = "xyz";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_SocModelStringName_SM8750_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("soc_model")] = "SM8750";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_SocModelStringName_Lowercase_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("soc_model")] = "sm8550";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_SocModelStringName_UnrecognizedName_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("soc_model")] = "FOOBAR";
  auto factory = MakeFactory(ctx);
  // Unrecognized name falls back to UNKNOWN (0) with a warning — must not throw.
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_SocModelBf16WithStringName_Throws) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_bf16_enable")] = "1";
  ctx.session_config[EPKey("soc_model")] = "SM8550";  // maps to 43, which is < 88
  auto factory = MakeFactory(ctx);
  EXPECT_THROW({ auto ep = MakeEp(*factory, ctx); }, std::runtime_error);
}

// Non-writable json_qnn_graph_dir exercises ProbeDumpDirectoryWritable's
// failure branch; ctor recovers (disables dump internally) rather than throwing.
TEST_F(QnnUnit_ExecutionProviderTest, Ctor_DumpJsonGraphDirNonWritable_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("dump_json_qnn_graph")] = "1";
  ctx.session_config[EPKey("json_qnn_graph_dir")] = "/proc/nonexistent/x";
  auto factory = MakeFactory(ctx);
  EXPECT_NO_THROW({ auto ep = MakeEp(*factory, ctx); });
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_GraphSplittingThreadsWithoutEnable_Succeeds) {
  EpStubContext ctx;
  ctx.session_config[EPKey("htp_graphsplitter_num_prepare_threads")] = "4";
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

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_MultiSocWithoutContextCache_Throws) {
  EpStubContext ctx;
  // Comma-separated soc_model triggers multi-SoC path.
  ctx.session_config[EPKey("soc_model")] = "60,73";
  ctx.session_config[EPKey("htp_arch")] = "73,75";
  // context_cache_enabled defaults to "0" → must throw.
  auto factory = MakeFactory(ctx);
  EXPECT_THROW({ auto ep = MakeEp(*factory, ctx); }, std::runtime_error);
}

TEST_F(QnnUnit_ExecutionProviderTest, Ctor_MultiSocWithShareEpContexts_Throws) {
  EpStubContext ctx;
  ctx.session_config[EPKey("soc_model")] = "60,73";
  ctx.session_config[EPKey("htp_arch")] = "73,75";
  ctx.session_config["ep.context_enable"] = "1";
  ctx.session_config["ep.share_ep_contexts"] = "1";
  auto factory = MakeFactory(ctx);
  EXPECT_THROW({ auto ep = MakeEp(*factory, ctx); }, std::runtime_error);
}

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
//
// On x86_64 (the only host QNN_EP_INTERNAL_SYMBOL_ACCESS covers today), the
// function returns EP_NOT_APPLICABLE at the empty-input branch or the
// #if !defined(__aarch64__)... x86-skip guard before any field parsing runs.
// The two tests below cover both reachable branches; parser branches are
// left for a future arm64 UT tier.
// ===========================================================================

TEST_F(QnnUnit_ExecutionProviderTest, ValidateCompatibilityInfo_EmptyString_LogsNoInfoAndReturnsNotApplicable) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);

  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  OrtStatus* s;
  {
    UseGlobalEpStubs use(ctx);
    s = ep->ValidateCompiledModelCompatibilityInfo(nullptr, 0, "", &compat);
  }
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING, "No compatibility info to be validated.");
}

TEST_F(QnnUnit_ExecutionProviderTest, ValidateCompatibilityInfo_NonEmptyOnX86Host_LogsSkipAndReturnsNotApplicable) {
  EpStubContext ctx;
  ctx.log_severity = ORT_LOGGING_LEVEL_VERBOSE;
  auto factory = MakeFactory(ctx);
  auto ep = MakeEp(*factory, ctx);

  // Any non-empty info string trips the x86 platform-skip guard on this test
  // host (see Group 10 header). Input shape is irrelevant — pick a 6-field
  // string that would be well-formed on aarch64 to make that clear.
  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  OrtStatus* s;
  {
    UseGlobalEpStubs use(ctx);
    s = ep->ValidateCompiledModelCompatibilityInfo(
        nullptr, 0, "1:1.0.0:2.1.0:3.0.0:73:0", &compat);
  }
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(compat, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
  ExpectLogged(ctx, ORT_LOGGING_LEVEL_WARNING, "Skip compatibility validation on x86 platforms.");
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
