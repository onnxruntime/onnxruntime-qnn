// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for QnnBackendManager (qnn_backend_manager.cc).
//
// Three groups of tests live here, all gated on QNN_EP_INTERNAL_SYMBOL_ACCESS:
//
//   1. Stub-based tests (no real QNN library) — QnnSerializerConfig, SetupBackend
//      load-failure paths, and before-setup early returns. These always run.
//   2. Real-HTP-backend tests (QnnUnit_BackendManagerHtpTest) — load the HTP backend
//      (and QnnIr / QnnSaver) and drive SetupBackend directly, with no ORT session.
//      The fixture GTEST_SKIP()s when the backend is unavailable, mirroring the
//      QnnHTPBackendTests::SetUp() convention.
//   3. File-mapped-weights DMA buffer lifecycle — also in
//      QnnUnit_BackendManagerHtpTest, additionally guarded by
//      QNN_FILE_MAPPED_WEIGHTS_AVAILABLE (Win ARM64 + QNN >= 2.32), because
//      MapDmaData / ReleaseDmaData do not exist on other platforms. These inject a
//      MockRpcMemLibrary so the FastRPC register/deregister bookkeeping is
//      observable; filter them with --gtest_filter=*MapDmaData*:*ReleaseDmaData*:*ReleaseResources*
//
// Coverage targets:
//   - QnnSerializerConfig (CreateIr / CreateSaver / GetBackendPath / SetGraphName / Configure)
//   - SetupBackend: load-failure paths (stub) + config permutations against real HTP
//     (priority / device / profiling), serializer backends (Saver / Ir)
//   - SetContextPriority / ResetContextPriority, QnnBackendProfilingManager::SetProfilingLevelETW
//   - ResetQnnLogLevel (before setup + after setup)
//   - GetContextBinaryBuffer (before setup + after setup) / LoadCachedQnnContextFromBuffer
//   - ParseLoraConfig file I/O error paths
//   - MapDmaData / ReleaseDmaData / DeallocateMappedDmaBuffers: every mapped FastRPC
//     buffer is deregistered, including on the registration- and deregistration-failure
//     paths

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
#include "test/providers/qnn/infra/mock_rpcmem_library.h"
#endif

namespace onnxruntime {
namespace test {

// ===========================================================================
// Test helpers
// ===========================================================================

// Backend library names differ by platform. Following the convention in
// bernoulli_test.cc:41-43. Used everywhere this file loads a real backend so the HTP group
// works wherever QNN_EP_INTERNAL_SYMBOL_ACCESS is enabled, not just on Linux. (Tests that
// only round-trip a path string through QnnSerializerConfig keep their literals — they load
// nothing.)
constexpr const char* kHtpBackendPath =
#if defined(_WIN32)
    "QnnHtp.dll";
#else
    "libQnnHtp.so";
#endif

constexpr const char* kSaverBackendPath =
#if defined(_WIN32)
    "QnnSaver.dll";
#else
    "libQnnSaver.so";
#endif

constexpr const char* kIrBackendPath =
#if defined(_WIN32)
    "QnnIr.dll";
#else
    "libQnnIr.so";
#endif

static std::shared_ptr<qnn::QnnBackendManager> MakeManager(
    const std::string& backend_path,
    const ApiPtrs& api_ptrs,
    const Ort::Logger& logger,
    bool skip_version_check = true,
    bool configure_host_mode = false) {
  qnn::QnnBackendManagerConfig cfg;
  cfg.backend_path = backend_path;
  // profiling_level / profiling_level_etw have no default member initializer (unlike
  // most other QnnBackendManagerConfig fields), so a bare `cfg;` leaves them
  // indeterminate. Every other call site that builds a non-brace-initialized config sets
  // both explicitly (see StubBackendManager, MakeHTPManager below, QnnModelMinimalTestContext,
  // etc.) -- match that convention here too.
  cfg.profiling_level = qnn::ProfilingLevel::OFF;
  cfg.profiling_level_etw = qnn::ProfilingLevel::OFF;
  cfg.context_priority = qnn::ContextPriority::NORMAL;
  cfg.device_id = 0;
  cfg.htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
  cfg.soc_model = 0;
  cfg.skip_qnn_version_check = skip_version_check;
  cfg.configure_host_mode = configure_host_mode;
  return qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
}

// ---------------------------------------------------------------------------
// Group 0: IsBackendHostMode — pure getter, no QNN lib needed
// ---------------------------------------------------------------------------

TEST(QnnUnit_BackendManagerTest, IsBackendHostMode_DefaultConfig_ReturnsFalse) {
  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  EXPECT_FALSE(manager->IsBackendHostMode());
}

TEST(QnnUnit_BackendManagerTest, IsBackendHostMode_ConfigureHostModeTrue_ReturnsTrue) {
  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger,
                             /*skip_version_check=*/true, /*configure_host_mode=*/true);
  ASSERT_NE(manager, nullptr);
  EXPECT_TRUE(manager->IsBackendHostMode());
}

// ---------------------------------------------------------------------------
// Group 1: QnnSerializerConfig — pure C++, no QNN lib needed
// ---------------------------------------------------------------------------

TEST(QnnUnit_BackendManagerTest, QnnSerializerConfig_CreateSaver_Properties) {
  auto cfg = qnn::QnnSerializerConfig::CreateSaver("libQnnSaver.so");
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->GetBackendPath(), "libQnnSaver.so");
  EXPECT_EQ(cfg->Configure(), nullptr);
  EXPECT_TRUE(cfg->SupportsArbitraryGraphConfigs());
}

TEST(QnnUnit_BackendManagerTest, QnnSerializerConfig_CreateIr_DefaultGraphName) {
  auto cfg = qnn::QnnSerializerConfig::CreateIr("libQnnIr.so", "/tmp/dlc_out");
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->GetBackendPath(), "libQnnIr.so");
  EXPECT_EQ(cfg->GetGraphName(), "graph");
  EXPECT_FALSE(cfg->SupportsArbitraryGraphConfigs());
}

TEST(QnnUnit_BackendManagerTest, QnnSerializerConfig_SetGraphName_ReflectsChange) {
  auto cfg = qnn::QnnSerializerConfig::CreateIr("libQnnIr.so", "/tmp/dlc_out");
  ASSERT_NE(cfg, nullptr);
  cfg->SetGraphName("my_graph");
  EXPECT_EQ(cfg->GetGraphName(), "my_graph");
}

TEST(QnnUnit_BackendManagerTest, QnnSerializerConfig_CreateIr_Configure_CreatesDir) {
  const std::filesystem::path dlc_dir =
      std::filesystem::temp_directory_path() / "qnn_ir_config_test";
  std::filesystem::remove_all(dlc_dir);

  auto cfg = qnn::QnnSerializerConfig::CreateIr("libQnnIr.so", dlc_dir.string());
  ASSERT_NE(cfg, nullptr);
  cfg->SetGraphName("test_graph");

  EXPECT_NE(cfg->Configure(), nullptr);
  EXPECT_TRUE(std::filesystem::exists(dlc_dir));

  std::filesystem::remove_all(dlc_dir);
}

TEST(QnnUnit_BackendManagerTest, QnnSerializerConfig_CreateIr_Configure_CalledTwice) {
  const std::filesystem::path dlc_dir =
      std::filesystem::temp_directory_path() / "qnn_ir_config_test2";
  std::filesystem::remove_all(dlc_dir);

  auto cfg = qnn::QnnSerializerConfig::CreateIr("libQnnIr.so", dlc_dir.string());
  ASSERT_NE(cfg, nullptr);
  cfg->SetGraphName("g1");
  EXPECT_NE(cfg->Configure(), nullptr);

  cfg->SetGraphName("g2");
  EXPECT_NE(cfg->Configure(), nullptr);

  std::filesystem::remove_all(dlc_dir);
}

// ---------------------------------------------------------------------------
// Group 2: SetupBackend — LoadBackend failures (no real .so needed)
// ---------------------------------------------------------------------------

// Non-existent library path → "Unable to load backend" error.
TEST(QnnUnit_BackendManagerTest, SetupBackend_InvalidPath_ReturnsError) {
  StubApiEnv env;
  auto manager = MakeManager("/nonexistent/path/backend.so", env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  auto status = manager->SetupBackend(false, false, false, -1, false, nullptr, dummy_map);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find("Unable to load backend"),
            std::string::npos);
}

// ---------------------------------------------------------------------------
// Group 3: ResetQnnLogLevel — before SetupBackend (early-return path)
// ---------------------------------------------------------------------------

// backend_setup_completed_ == false → early return OK without touching QNN API.
TEST(QnnUnit_BackendManagerTest, ResetQnnLogLevel_BeforeSetup_ReturnsOk) {
  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  EXPECT_TRUE(manager->ResetQnnLogLevel(std::nullopt).IsOK());
}

// ---------------------------------------------------------------------------
// Group 4: GetContextBinaryBuffer — before SetupBackend
// ---------------------------------------------------------------------------

// QNN interface is uninitialised → returns an error without calling QNN API and
// leaves the out buffer untouched.
TEST(QnnUnit_BackendManagerTest, GetContextBinaryBuffer_BeforeSetup_ReturnsError) {
  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  unsigned char* context_buffer = nullptr;
  uint64_t written_size = 0;
  auto status = manager->GetContextBinaryBuffer(/*is_multi_soc_buffer=*/false, &context_buffer, written_size);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(context_buffer, nullptr);
}

// ---------------------------------------------------------------------------
// Group 5: ParseLoraConfig — file I/O error paths (no QNN API needed)
// ---------------------------------------------------------------------------

// Config file does not exist → logs error, returns OK.
TEST(QnnUnit_BackendManagerTest, ParseLoraConfig_FileNotFound_ReturnsOk) {
  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  EXPECT_TRUE(manager->ParseLoraConfig("/nonexistent/lora_config.txt").IsOK());
}

// Config file exists but is empty → getline fails immediately, returns OK.
TEST(QnnUnit_BackendManagerTest, ParseLoraConfig_EmptyFile_ReturnsOk) {
  const std::filesystem::path cfg =
      std::filesystem::temp_directory_path() / "lora_empty.txt";
  {
    std::ofstream f(cfg);
  }

  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  EXPECT_TRUE(manager->ParseLoraConfig(cfg.string()).IsOK());
  std::filesystem::remove(cfg);
}

// Config line has no semicolon → path field is empty, falls through, returns OK.
TEST(QnnUnit_BackendManagerTest, ParseLoraConfig_NoSemicolon_ReturnsOk) {
  const std::filesystem::path cfg =
      std::filesystem::temp_directory_path() / "lora_nosemi.txt";
  {
    std::ofstream f(cfg);
    f << "graph_name_without_path\n";
  }

  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  EXPECT_TRUE(manager->ParseLoraConfig(cfg.string()).IsOK());
  std::filesystem::remove(cfg);
}

// Valid "graph;path" format but contexts_ is empty (no SetupBackend) →
// graphRetrieve loop never runs → returns error.
TEST(QnnUnit_BackendManagerTest, ParseLoraConfig_ValidFormatNoContext_ReturnsError) {
  const std::filesystem::path bin_file =
      std::filesystem::temp_directory_path() / "lora_dummy.bin";
  {
    std::ofstream f(bin_file, std::ios::binary);
    f << "dummy_lora_data";
  }

  const std::filesystem::path cfg =
      std::filesystem::temp_directory_path() / "lora_valid.txt";
  {
    std::ofstream f(cfg);
    f << "my_graph;" << bin_file.string() << "\n";
  }

  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  EXPECT_FALSE(manager->ParseLoraConfig(cfg.string()).IsOK());

  std::filesystem::remove(cfg);
  std::filesystem::remove(bin_file);
}

// ---------------------------------------------------------------------------
// Group 6: HTP arch / SoC model holders, and const-qualified getters
// ---------------------------------------------------------------------------

// Builds a manager with a user-provided HTP arch and SoC model. Nothing is dlopen'd —
// Create() only stores the config.
static std::shared_ptr<qnn::QnnBackendManager> MakeManagerWithHtpArch(
    QnnHtpDevice_Arch_t htp_arch,
    uint32_t soc_model,
    const ApiPtrs& api_ptrs,
    const Ort::Logger& logger) {
  qnn::QnnBackendManagerConfig cfg{};
  cfg.backend_path = kHtpBackendPath;
  cfg.context_priority = qnn::ContextPriority::NORMAL;
  cfg.device_id = 0;
  cfg.htp_arch = htp_arch;
  cfg.soc_model = soc_model;
  cfg.skip_qnn_version_check = true;
  return qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
}

// A user-provided htp_arch alone does NOT make GetHtpArch() report it — the internal
// holder stays NONE until SetupDeviceAndContext runs. soc_model has no such split and
// is readable straight from the config.
TEST(QnnUnit_BackendManagerTest, GetHtpArch_UserProvidedArchBeforeSetup_ReturnsNone) {
  StubApiEnv env;
  auto manager = MakeManagerWithHtpArch(QNN_HTP_DEVICE_ARCH_V73, QNN_SOC_MODEL_SM8550,
                                        env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
  EXPECT_EQ(manager->GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_SM8550));
}

// SetupDeviceAndContext is guarded by backend_partial_setup_completed_, so calling it
// without SetupBackendExceptDeviceAndContext first must fail and must not publish the
// requested arch to the internal holder.
TEST(QnnUnit_BackendManagerTest, SetupDeviceAndContext_WithoutPartialSetup_ReturnsErrorAndLeavesArchNone) {
  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  auto status = manager->SetupDeviceAndContext(QNN_HTP_DEVICE_ARCH_V79, QNN_SOC_MODEL_SM8550);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find("partially setup"), std::string::npos)
      << "Expected the partial-setup guard to reject the call; got: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
}

// ReleaseDeviceAndContext resets both arch holders and the SoC model to their defaults.
// Safe to call before any setup: ReleaseContext / ReleaseDevice report failures via the
// logger and the reset still happens.
TEST(QnnUnit_BackendManagerTest, ReleaseDeviceAndContext_BeforeSetup_ResetsArchAndSocModel) {
  StubApiEnv env;
  auto manager = MakeManagerWithHtpArch(QNN_HTP_DEVICE_ARCH_V73, QNN_SOC_MODEL_SM8550,
                                        env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  manager->ReleaseDeviceAndContext();

  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
  EXPECT_EQ(manager->GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_UNKNOWN));
}

// QnnModelWrapper stores a `const QnnBackendManager&`, so every getter it calls has to be
// const-qualified. Calling them through a const reference is the compile-time check;
// the static_asserts state the requirement explicitly so a future non-const overload
// fails here rather than in the wrapper.
TEST(QnnUnit_BackendManagerTest, ConstGetters_CallableOnConstManager) {
  StubApiEnv env;
  auto manager = MakeManager(kHtpBackendPath, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  const qnn::QnnBackendManager& const_manager = *manager;

  static_assert(std::is_invocable_v<decltype(&qnn::QnnBackendManager::GetQnnInterface),
                                    const qnn::QnnBackendManager&>,
                "GetQnnInterface must be const — QnnModelWrapper holds a const manager");
  static_assert(std::is_invocable_v<decltype(&qnn::QnnBackendManager::GetQnnValidatorInterface),
                                    const qnn::QnnBackendManager&>,
                "GetQnnValidatorInterface must be const");
  static_assert(std::is_invocable_v<decltype(&qnn::QnnBackendManager::GetQnnSystemInterface),
                                    const qnn::QnnBackendManager&>,
                "GetQnnSystemInterface must be const");
  static_assert(std::is_invocable_v<decltype(&qnn::QnnBackendManager::GetQnnBackendHandle),
                                    const qnn::QnnBackendManager&>,
                "GetQnnBackendHandle must be const");
  static_assert(std::is_invocable_v<decltype(&qnn::QnnBackendManager::GetQnnValidatorBackendHandle),
                                    const qnn::QnnBackendManager&>,
                "GetQnnValidatorBackendHandle must be const");
  static_assert(std::is_invocable_v<decltype(&qnn::QnnBackendManager::GetQnnDeviceHandle),
                                    const qnn::QnnBackendManager&>,
                "GetQnnDeviceHandle must be const");
  static_assert(std::is_invocable_v<decltype(&qnn::QnnBackendProfilingManager::HasProfileHandle),
                                    const qnn::QnnBackendProfilingManager&>,
                "HasProfileHandle must be const");
  static_assert(std::is_invocable_v<decltype(&qnn::QnnBackendManager::GetQnnBackendType),
                                    const qnn::QnnBackendManager&>,
                "GetQnnBackendType must be const");
  static_assert(std::is_invocable_v<decltype(&qnn::QnnBackendManager::GetHtpArch),
                                    const qnn::QnnBackendManager&>,
                "GetHtpArch must be const — op builders query it through a const manager");

  // Before setup these are the zero-initialised defaults; the point is that the calls
  // compile and are readable on a const manager.
  EXPECT_EQ(const_manager.GetQnnBackendHandle(), nullptr);
  EXPECT_EQ(const_manager.GetQnnValidatorBackendHandle(), nullptr);
  EXPECT_EQ(const_manager.GetQnnDeviceHandle(), nullptr);
  EXPECT_FALSE(const_manager.GetProfilingManager().HasProfileHandle());
  EXPECT_FALSE(const_manager.GetProfilingManager().OrtProfilingActive());
  EXPECT_EQ(const_manager.GetQnnBackendType(), qnn::QnnBackendType::CPU);
  EXPECT_EQ(const_manager.GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
  EXPECT_EQ(const_manager.GetQnnInterface().backendCreate,
            const_manager.GetQnnValidatorInterface().backendCreate);
  // qnn_sys_interface_ is only bound by SetupBackend, so every entry is still NULL here.
  EXPECT_EQ(const_manager.GetQnnSystemInterface().systemContextCreate, nullptr);
}

// ===========================================================================
// Real-HTP-backend tests
//
// The tests below load a real QNN backend (kHtpBackendPath, and for the serializer
// cases kIrBackendPath / kSaverBackendPath) and drive QnnBackendManager directly — no
// ORT session is created. They target qnn_backend_manager.cc code paths that are
// only reachable once a real backend interface is bound: SetupBackend config
// permutations (priority / device / profiling), context serialization, and
// serializer-backend loading.
// ===========================================================================

// Creates a manager configured to use the real HTP backend.
static std::shared_ptr<qnn::QnnBackendManager> MakeHTPManager(
    const ApiPtrs& api_ptrs,
    const Ort::Logger& logger,
    qnn::ContextPriority context_priority = qnn::ContextPriority::NORMAL,
    uint32_t soc_model = 0,
    qnn::ProfilingLevel profiling_level = qnn::ProfilingLevel::OFF,
    qnn::ProfilingLevel profiling_level_etw = qnn::ProfilingLevel::OFF,
    QnnHtpDevice_Arch_t htp_arch = QNN_HTP_DEVICE_ARCH_NONE,
    bool skip_version_check = true,
    bool configure_host_mode = false) {
  qnn::QnnBackendManagerConfig cfg;
  cfg.backend_path = kHtpBackendPath;
  cfg.profiling_level = profiling_level;
  cfg.profiling_level_etw = profiling_level_etw;
  cfg.context_priority = context_priority;
  cfg.device_id = 0;
  cfg.htp_arch = htp_arch;
  cfg.soc_model = soc_model;
  cfg.skip_qnn_version_check = skip_version_check;
  cfg.configure_host_mode = configure_host_mode;
  return qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
}

// Creates a manager configured to use a QNN serializer (Saver or Ir) backend
// with the given validator backend.
static std::shared_ptr<qnn::QnnBackendManager> MakeSerializerManager(
    const std::string& validator_backend_path,
    std::shared_ptr<qnn::QnnSerializerConfig> serializer_config,
    const ApiPtrs& api_ptrs,
    const Ort::Logger& logger,
    bool skip_version_check = true) {
  qnn::QnnBackendManagerConfig cfg{};  // value-init to zero all fields
  cfg.backend_path = validator_backend_path;
  cfg.qnn_serializer_config = std::move(serializer_config);
  cfg.context_priority = qnn::ContextPriority::NORMAL;
  cfg.skip_qnn_version_check = skip_version_check;
  return qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
}

// Calls SetupBackend with standard test parameters (no shared context, no rpcmem).
static Ort::Status SetupBackendHtp(qnn::QnnBackendManager& manager) {
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  return manager.SetupBackend(false, false, false, -1, false, nullptr, dummy_map);
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
// As SetupBackendHtp, but enables file-mapped weights and supplies the rpcmem library.
// file_mapped_weights_enabled_ only latches when the backend type is HTP
// (qnn_backend_manager.cc:2145), which is why the file-mapping tests live in the real-HTP
// fixture rather than the stub group.
static Ort::Status SetupBackendHtpWithFileMapping(qnn::QnnBackendManager& manager,
                                                  std::shared_ptr<qnn::IRpcMemLibrary> rpcmem) {
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  return manager.SetupBackend(false, false, false, -1,
                              /*enable_file_mapped_weights=*/true, std::move(rpcmem), dummy_map);
}
#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

// Fixture: probes HTP backend availability once (cached) and skips the whole
// group via GTEST_SKIP() when the HTP backend library cannot be loaded — mirroring the
// established QnnHTPBackendTests::SetUp() convention (backend unavailable → skip,
// not fail). This keeps CI signal clean on environments without the HTP library
// while leaving each test's ASSERT_TRUE(status.IsOK()) as a genuine behavioral
// check once HTP is confirmed present.
class QnnUnit_BackendManagerHtpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!HtpAvailable()) {
      GTEST_SKIP() << "QNN HTP backend (" << kHtpBackendPath << ") is not available! Skipping test.";
    }
  }

 private:
  // Probe once: create an HTP manager and run SetupBackend. Cached for the process.
  static bool HtpAvailable() {
    static const bool available = [] {
      StubApiEnv env;
      auto manager = MakeHTPManager(env.api_ptrs, env.logger);
      return manager != nullptr && SetupBackendHtp(*manager).IsOK();
    }();
    return available;
  }
};

// ---------------------------------------------------------------------------
// HTP backend — basic setup and backend type
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << kHtpBackendPath << " setup failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// Second SetupBackend call on the same manager returns OK immediately (no-op).
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_CalledTwice_SecondCallIsNoOp) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackendHtp(*manager);
    ASSERT_TRUE(s.IsOK()) << "SetupBackend failed: " << s.GetErrorMessage();
  }
  EXPECT_TRUE(SetupBackendHtp(*manager).IsOK());
}

// skip_version_check=false exercises the GetQnnInterfaceProvider version-check loop.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithVersionCheck_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::OFF, qnn::ProfilingLevel::OFF,
                                QNN_HTP_DEVICE_ARCH_NONE, /*skip_version_check=*/false);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// ---------------------------------------------------------------------------
// HTP backend — cross device preparation (SetGlobalConfig)
// ---------------------------------------------------------------------------

// configure_host_mode=true drives SetGlobalConfig() to set
// QNN_GLOBAL_CONFIG_OPTION_MACHINE_TYPE_HOST via globalConfigSet during SetupBackend.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithHostModeConfigured_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::OFF, qnn::ProfilingLevel::OFF,
                                QNN_HTP_DEVICE_ARCH_NONE, /*skip_version_check=*/true,
                                /*configure_host_mode=*/true);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(manager->IsBackendHostMode());
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend with host mode configured failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// ---------------------------------------------------------------------------
// HTP backend — context priority configs
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithLowPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::LOW);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithNormalHighPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::NORMAL_HIGH);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithHighPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::HIGH);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// SetContextPriority after HTP setup: covers SetContextPriority body and
// calls SetQnnContextConfig for LOW and NORMAL_HIGH.
TEST_F(QnnUnit_BackendManagerHtpTest, SetContextPriority_HTP_ChangesLevel) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackendHtp(*manager);
    ASSERT_TRUE(s.IsOK()) << "SetupBackend failed: " << s.GetErrorMessage();
  }

  EXPECT_TRUE(manager->SetContextPriority(qnn::ContextPriority::LOW).IsOK());
  EXPECT_TRUE(manager->SetContextPriority(qnn::ContextPriority::NORMAL_HIGH).IsOK());
  EXPECT_TRUE(manager->ResetContextPriority().IsOK());
}

// The following four tests cover the remaining SetQnnContextConfig priority
// branches (NORMAL_LOW, HIGH_PLUS, CRITICAL, CRITICAL_PLUS). All four values
// are accepted by the HTP emulator on Linux x86_64; if a future emulator
// version rejects one, the corresponding test will fail loudly so the
// regression is visible.

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithNormalLowPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::NORMAL_LOW);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithHighPlusPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::HIGH_PLUS);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithCriticalPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::CRITICAL);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithCriticalPlusPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::CRITICAL_PLUS);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// UNDEFINED priority is invalid: SetQnnContextConfig returns MAKE_EP_FAIL before
// contextCreate is called, so SetupBackend must return an error.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithUndefinedPriority_ReturnsError) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::UNDEFINED);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_FALSE(status.IsOK()) << "SetupBackend unexpectedly succeeded with UNDEFINED priority";
  ASSERT_NE(std::string(status.GetErrorMessage()).find("Invalid Qnn context priority"),
            std::string::npos)
      << "Expected 'Invalid Qnn context priority' error; got: " << status.GetErrorMessage();
}

// ---------------------------------------------------------------------------
// HTP backend — device configs (SoC model, HTP arch)
// ---------------------------------------------------------------------------

// Uses the SM8550 (Snapdragon 8 Gen 2) SoC model to exercise the HTP SoC
// model config block; the HTP emulator accepts arbitrary SoC model values.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithSocModel_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL,
                                /*soc_model=*/QNN_SOC_MODEL_SM8550);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// HTP emulator accepts arbitrary arch values.
//
// Also covers the x86 half of GetPlatformInfo(): with no real device to query, its
// #else branch copies the configured htp_arch into htp_arch_internal_, the holder
// GetHtpArch() reads. (The __aarch64__ / _M_ARM64 branch queries the hardware
// instead and is not reachable from x86 tests.)
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithHtpArch_ReportsConfiguredArch) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::OFF, qnn::ProfilingLevel::OFF,
                                QNN_HTP_DEVICE_ARCH_V73);
  ASSERT_NE(manager, nullptr);
  // A configured htp_arch reaches htp_arch_ only; the internal holder stays NONE
  // until setup runs.
  ASSERT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);

  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_V73);
}

// ---------------------------------------------------------------------------
// HTP backend — profiling and log level
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithBasicProfiling_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::BASIC);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithDetailedProfiling_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::DETAILED);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// profiling_level_etw > profiling_level → InitializeProfiling uses merged level.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_EtwLevelHigherThanMain_UsesMergedLevel) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                /*profiling_level=*/qnn::ProfilingLevel::BASIC,
                                /*profiling_level_etw=*/qnn::ProfilingLevel::DETAILED);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// SetProfilingLevelETW releases and re-creates the profile handle.
TEST_F(QnnUnit_BackendManagerHtpTest, ProfilingManager_SetProfilingLevelETW_HTP_ChangesLevel) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::BASIC);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackendHtp(*manager);
    ASSERT_TRUE(s.IsOK()) << "SetupBackend failed: " << s.GetErrorMessage();
  }

  EXPECT_TRUE(manager->GetProfilingManager().SetProfilingLevelETW(qnn::ProfilingLevel::BASIC, env.logger).IsOK());
  EXPECT_TRUE(manager->GetProfilingManager().SetProfilingLevelETW(qnn::ProfilingLevel::OFF, env.logger).IsOK());
}

// After SetupBackend each ORT log level maps to a different QNN log level.
TEST_F(QnnUnit_BackendManagerHtpTest, ResetQnnLogLevel_HTP_AfterSetup_VariousLevels_Succeed) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackendHtp(*manager);
    ASSERT_TRUE(s.IsOK()) << "SetupBackend failed: " << s.GetErrorMessage();
  }

  for (auto level : {ORT_LOGGING_LEVEL_VERBOSE, ORT_LOGGING_LEVEL_INFO,
                     ORT_LOGGING_LEVEL_WARNING, ORT_LOGGING_LEVEL_ERROR}) {
    EXPECT_TRUE(manager->ResetQnnLogLevel(level).IsOK()) << "level=" << level;
  }
  EXPECT_TRUE(manager->ResetQnnLogLevel(std::nullopt).IsOK());
}

// ---------------------------------------------------------------------------
// HTP backend — context binary buffer
// ---------------------------------------------------------------------------

// After SetupBackend the HTP context can be serialized to a non-empty buffer.
TEST_F(QnnUnit_BackendManagerHtpTest, GetContextBinaryBuffer_HTP_AfterSetup_ReturnsValidBuffer) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackendHtp(*manager);
    ASSERT_TRUE(s.IsOK()) << "SetupBackend failed: " << s.GetErrorMessage();
  }

  unsigned char* raw_buffer = nullptr;
  uint64_t written_size = 0;
  auto status = manager->GetContextBinaryBuffer(/*is_multi_soc_buffer=*/false, &raw_buffer, written_size);
  ASSERT_TRUE(status.IsOK()) << "GetContextBinaryBuffer failed: " << status.GetErrorMessage();
  std::unique_ptr<unsigned char[]> buffer(raw_buffer);  // caller owns the buffer
  EXPECT_NE(buffer, nullptr);
  EXPECT_GT(written_size, 0u);
}

// Garbage bytes are rejected without crashing.
TEST_F(QnnUnit_BackendManagerHtpTest, LoadCachedQnnContextFromBuffer_HTP_InvalidBuffer_ReturnsError) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  qnn::EpContextIoDispatch dummy_io_dispatch(nullptr);
  auto setup_status = manager->SetupBackend(true, true, false, -1, false, nullptr, dummy_map);
  ASSERT_TRUE(setup_status.IsOK()) << "SetupBackend with QnnSystem failed: " << setup_status.GetErrorMessage();

  char garbage[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                      0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> qnn_models;
  auto status = manager->LoadCachedQnnContextFromBuffer(
      garbage, sizeof(garbage), "", "test_node", qnn_models, 0, dummy_io_dispatch);
  EXPECT_FALSE(status.IsOK());
}

// ---------------------------------------------------------------------------
// IR backend loaded directly (no QnnSerializerConfig)
//
// Loading the QNN IR backend as the main backend exercises:
//   - SetQnnBackendType IR/SAVER case: backend_id → QnnBackendType::SERIALIZER
//   - CreateContext SERIALIZER branch: configs = nullptr
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_WithIrBackendDirectly_SetsSerializerBackendType) {
  StubApiEnv env;
  qnn::QnnBackendManagerConfig cfg{};  // value-init to zero all fields (profiling, device_id, etc.)
  cfg.backend_path = kIrBackendPath;
  cfg.context_priority = qnn::ContextPriority::NORMAL;
  cfg.skip_qnn_version_check = true;
  auto manager = qnn::QnnBackendManager::Create(cfg, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << kIrBackendPath << " setup failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::SERIALIZER);
}

// ---------------------------------------------------------------------------
// Serializer backends (Saver / Ir) with HTP as validator
//
// LoadQnnSerializerBackend loads both the validator backend (HTP) and the
// serializer backend (Saver or Ir). The serializer config determines which
// configs are passed to contextCreate:
//   - QnnSaverConfig::SupportsArbitraryGraphConfigs() == true  → HTP configs used
//   - QnnIrConfig::SupportsArbitraryGraphConfigs()    == false → configs nullified
// ---------------------------------------------------------------------------

// QnnSaver records all QNN API calls. With HTP as the validator backend,
// SetupBackend exercises LoadQnnSerializerBackend (loads both .so libraries and
// logs their versions).
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithQnnSaverSerializer_Succeeds) {
  StubApiEnv env;
  auto manager = MakeSerializerManager(
      kHtpBackendPath,
      qnn::QnnSerializerConfig::CreateSaver(kSaverBackendPath),
      env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "QnnSaver+HTP setup failed: " << status.GetErrorMessage();

  EXPECT_NE(manager->GetQnnSerializerConfig(), nullptr);
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// QnnIrConfig::SupportsArbitraryGraphConfigs() returns false, so CreateContext
// overrides configs to nullptr even for the HTP backend's default configs.
// Distinct from SetupBackend_WithIrBackendDirectly above which loads the IR backend
// as the main backend (SERIALIZER type) without QnnSerializerConfig.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithQnnIrSerializer_CoversNoArbitraryGraphConfigs) {
  StubApiEnv env;
  const auto tmp_dir = std::filesystem::temp_directory_path() / "qnn_ir_serializer_htp_test";
  std::filesystem::create_directories(tmp_dir);

  auto manager = MakeSerializerManager(
      kHtpBackendPath,
      qnn::QnnSerializerConfig::CreateIr(kIrBackendPath, tmp_dir.string()),
      env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "QnnIr+HTP setup failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);

  std::filesystem::remove_all(tmp_dir);
}

// ---------------------------------------------------------------------------
// HTP backend — SetupDeviceAndContext publishes the HTP arch to op builders
// ---------------------------------------------------------------------------

// The multi-SoC EP-context entry points, driven in order: partial setup, then
// SetupDeviceAndContext with an explicit arch + SoC model. SetupDeviceAndContext is
// what copies the caller-supplied arch into the internal holder GetHtpArch() reads,
// which is how an op builder sees a user-provided htp_arch on x86. Then
// ReleaseDeviceAndContext must put both holders back to their defaults.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupDeviceAndContext_HTP_PublishesHtpArchThenResetsOnRelease) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  // Nothing published before device/context setup.
  ASSERT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);

  auto partial = manager->SetupBackendExceptDeviceAndContext();
  ASSERT_TRUE(partial.IsOK()) << "SetupBackendExceptDeviceAndContext failed: "
                              << partial.GetErrorMessage();

  auto status = manager->SetupDeviceAndContext(QNN_HTP_DEVICE_ARCH_V73, QNN_SOC_MODEL_SM8550);
  ASSERT_TRUE(status.IsOK()) << "SetupDeviceAndContext failed: " << status.GetErrorMessage();

  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_V73);
  EXPECT_EQ(manager->GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_SM8550));

  manager->ReleaseDeviceAndContext();

  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
  EXPECT_EQ(manager->GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_UNKNOWN));
}

// Smoke-tests that SetupBackend accepts enable_htp_graph_splitting=true and
// the new htp_graph_splitting_num_prepare_threads parameter without crashing or
// returning an unexpected error code.
// On SDK < 2.49 (QNN_HTP_GRAPH_SPLITTING_AVAILABLE not defined), the
// enable_htp_graph_splitting parameter is consumed by ORT_UNUSED_PARAMETER in
// CreateContext and has no effect; the test still compiles and passes as a
// pure crash/link check.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithGraphSplittingEnabled_CompletesGracefully) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  auto status = manager->SetupBackend(false, false, false, -1, false, nullptr, dummy_map,
                                      qnn::EpContextIoDispatch(nullptr),
                                      false /*extended_udma*/, false /*prepare_only*/,
                                      true /*enable_htp_graph_splitting*/);
  // Success or a structured QNN error are both acceptable on headless hosts;
  // a crash or CHECK failure is not.
  (void)status;
}

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithGraphSplittingAndThreadCount_CompletesGracefully) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  auto status = manager->SetupBackend(false, false, false, -1, false, nullptr, dummy_map,
                                      qnn::EpContextIoDispatch(nullptr),
                                      false /*extended_udma*/, false /*prepare_only*/,
                                      true /*enable_htp_graph_splitting*/,
                                      4 /*htp_graph_splitting_num_prepare_threads*/);
  (void)status;
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
// ===========================================================================
// File-mapped-weights DMA buffer lifecycle
//
// MapDmaData / ReleaseDmaData / DeallocateMappedDmaBuffers make no QNN calls --
// they only drive the rpcmem function pointers and the mapped_fastrpc_buffers_
// bookkeeping. So with a MockRpcMemLibrary injected through SetupBackend, every
// branch is reachable with no model, no context binary, and no real DMA.
//
// The EP gets no return value from register_buf(), so it infers success by calling
// to_fd() afterwards. The mock mirrors that: a registration makes to_fd() return a
// valid fd, a deregistration makes it return -1, and an injected failure leaves the
// previous state in place -- exactly what the production error paths detect.
//
// Filter this group with:
//   --gtest_filter=*MapDmaData*:*ReleaseDmaData*:*ReleaseResources*
// ===========================================================================

namespace {

// Stands in for a memory-mapped context binary. Only addresses are used: MapDmaData
// registers `base + offset` with rpcmem and never reads the bytes.
constexpr uint64_t kFakeFileSize = 64 * 1024;

Qnn_ContextBinaryDataRequest_t MakeDmaRequest(uint64_t offset, uint64_t size,
                                              bool backend_mapping_needed = true) {
  Qnn_ContextBinaryDataRequest_t request{};
  request.offset = offset;
  request.size = size;
  request.isBackendMappingNeeded = backend_mapping_needed;
  return request;
}

// The release descriptor QNN hands back for a previously mapped region.
Qnn_ContextBinaryDmaDataMem_t MakeDmaDataMem(void* data, uint64_t size) {
  Qnn_ContextBinaryDmaDataMem_t data_mem{};
  data_mem.dmaBuffer.data = data;
  data_mem.memSize = size;
  return data_mem;
}

// Self-contained copy of qnn_test_utils.h:832's alias. Not included from qnn_test_utils.h
// itself: that header transitively pulls in test/util/include/... headers that are off-limits
// to component/ (see component/README.md), so the type is duplicated here instead.
using RegisteredEpDeviceUniquePtr = std::unique_ptr<const OrtEpDevice, std::function<void(const OrtEpDevice*)>>;

constexpr const char* kQnnExecutionProvider = "QNNExecutionProvider";

// Self-contained copy of qnn_test_utils.cc's RegisterQnnEpLibrary (qnn_test_utils.cc:194-272),
// registering the QNN EP as a plugin library (RegisterExecutionProviderLibrary + GetEpDevices +
// AppendExecutionProvider_V2) rather than the legacy AppendExecutionProvider(string, options)
// path, which is not how this fork makes "QNN" available to a session. Not reused from
// qnn_test_utils.h for the same reason as RegisteredEpDeviceUniquePtr above. Two deliberate
// differences from the original:
//   - Takes `ort_env` as an explicit parameter instead of calling the integration tier's
//     GetOrtEnv() singleton; component/ has no such singleton and shouldn't adopt one.
//   - Drops the `simulated` parameter: this file only ever loads the real QNN EP plugin, never
//     the simulation DLL, so that branch (qnn_test_utils.cc:202-217) is simplified away.
// ASSERT_ORTSTATUS_OK (test/util/include/api_asserts.h -- also off-limits here) is inlined as
// the plain Ort::Status/ASSERT_TRUE pair it expands to.
void RegisterQnnEpLibrary(RegisteredEpDeviceUniquePtr& registered_ep_device,
                          Ort::SessionOptions& session_options,
                          const std::string& registration_name,
                          const std::unordered_map<std::string, std::string>& ep_options,
                          Ort::Env& ort_env) {
  const OrtApi& c_api = Ort::GetApi();

  const std::filesystem::path library_path =
#if defined(_WIN32)
      "onnxruntime_providers_qnn.dll";
#else
      "libonnxruntime_providers_qnn.so";
#endif

  {
    Ort::Status status{c_api.RegisterExecutionProviderLibrary(ort_env, registration_name.c_str(),
                                                              library_path.c_str())};
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  }

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices = 0;
  {
    Ort::Status status{c_api.GetEpDevices(ort_env, &ep_devices, &num_devices)};
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  }

  // kHtpBackendPath is this file's existing platform-conditional constant (defined above,
  // mirroring kHtpBackendPath/kSaverBackendPath/kIrBackendPath) -- reused here instead of
  // re-inlining the #if _WIN32 "QnnHtp.dll" / "libQnnHtp.so" literal from the original.
  auto target_hw_device_type = OrtHardwareDeviceType_CPU;
  if ((ep_options.find("backend_type") != ep_options.end() && ep_options.at("backend_type") == "htp") ||
      (ep_options.find("backend_path") != ep_options.end() && ep_options.at("backend_path") == kHtpBackendPath)) {
#if defined(__linux__) || (defined(_WIN32) && defined(_M_X64))
    target_hw_device_type = OrtHardwareDeviceType_CPU;
#else
    target_hw_device_type = OrtHardwareDeviceType_NPU;
#endif
  } else if ((ep_options.find("backend_type") != ep_options.end() && ep_options.at("backend_type") == "gpu") ||
             (ep_options.find("backend_path") != ep_options.end() && ep_options.at("backend_path") ==
#if defined(_WIN32)
                                                                         "QnnGpu.dll"
#else
                                                                         "libQnnGpu.so"
#endif
              )) {
#if defined(__linux__)
    target_hw_device_type = OrtHardwareDeviceType_CPU;
#else
    target_hw_device_type = OrtHardwareDeviceType_GPU;
#endif
  }

  auto it = std::find_if(ep_devices, ep_devices + num_devices,
                         [&c_api, &registration_name, target_hw_device_type](const OrtEpDevice* ep_device) {
                           return c_api.EpDevice_EpName(ep_device) == registration_name &&
                                  c_api.HardwareDevice_Type(c_api.EpDevice_Device(ep_device)) ==
                                      target_hw_device_type;
                         });
  ASSERT_NE(it, ep_devices + num_devices);

  registered_ep_device = RegisteredEpDeviceUniquePtr(*it, [&ort_env, registration_name](const OrtEpDevice* /*ep*/) {
    OrtStatus* status = Ort::GetApi().UnregisterExecutionProviderLibrary(ort_env, registration_name.c_str());
    if (status != nullptr) {
      Ort::GetApi().ReleaseStatus(status);
    }
  });

  session_options.AppendExecutionProvider_V2(ort_env, {Ort::ConstEpDevice(registered_ep_device.get())}, ep_options);
}

// Compiles testdata/nhwc_conv_clip_relu.onnx into a non-embedded (ep.context_embed_mode=0)
// QNN context binary on disk and returns its path. A real Ort::Session is the only way to
// produce a context binary with real, backend-assigned weight-set offsets -- there is no
// lighter-weight production entry point for this. This is a deliberate, narrowly-scoped
// exception to this file's usual "no real session" convention (see component/README.md's
// mock-strategy ladder), made so the MapDmaData registration/deregistration lifecycle below
// can be driven by QNN's own callback-based loader (contextCreateFromBinaryWithCallback)
// against a genuine multi-weight-set binary, rather than the synthetic single-region
// MakeDmaRequest() calls used by the rest of this group.
//
// The caller owns the returned file (and must delete it); the manager/file_mapper_ that
// later memory-maps it must be destroyed before the file is removed (see
// qnn_windows_file_mapper.cc: the mapped view holds the file open for its lifetime).
static std::filesystem::path CompileMultiWeightSetContextBinary() {
  static int counter = 0;
  const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string unique_name = std::string(test_info != nullptr ? test_info->name() : "unknown") +
                                  "_" + std::to_string(counter++) + ".bin";
  const std::filesystem::path context_bin_path = std::filesystem::temp_directory_path() / unique_name;

  // Declaration order matters here: non-static locals are destroyed in reverse order, so this
  // ordering ensures `session` is destroyed before `registered_ep_device` unregisters the EP
  // library, and that `env` outlives `registered_ep_device`'s deleter (which calls
  // UnregisterExecutionProviderLibrary(env, ...) on destruction). Mirrors the same constraint
  // documented at qnn_test_utils.h:841-849 (ScopedOrtSession).
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "QnnBackendManagerTest_FileMappedWeights");
  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry("ep.context_enable", "1");
  session_options.AddConfigEntry("ep.context_embed_mode", "0");
  session_options.AddConfigEntry("ep.context_file_path", context_bin_path.string().c_str());

  std::unordered_map<std::string, std::string> provider_options;
  provider_options["backend_path"] = kHtpBackendPath;
  // Registers QNN as a plugin EP and appends it via AppendExecutionProvider_V2 (inside
  // RegisterQnnEpLibrary) instead of the legacy AppendExecutionProvider(string, options) call --
  // see the comment on RegisterQnnEpLibrary above for why.
  // Parameter order: (out-param device handle, session options, EP registration name,
  // EP options, owning Ort::Env).
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options, env);

  Ort::Session session(env, ORT_TSTR("./testdata/nhwc_conv_clip_relu.onnx"), session_options);
  return context_bin_path;
}

}  // namespace

// ---------------------------------------------------------------------------
// Headline: every mapped buffer is deregistered by teardown
// ---------------------------------------------------------------------------

// Maps three regions, releases one explicitly, then tears the manager down. The
// remaining two must be swept by DeallocateMappedDmaBuffers, leaving nothing
// registered and every registration matched by a successful deregistration.
TEST_F(QnnUnit_BackendManagerHtpTest,
       ReleaseResources_AfterMultipleMapsAndPartialReleases_AllBuffersDeregistered) {
  StubApiEnv env;  // owns the logger; must outlive the manager
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());
  ASSERT_TRUE(manager->FileMappingIsEnabled());

  std::vector<char> file(kFakeFileSize);
  constexpr uint64_t kRegionSize = 256;
  const uint64_t offsets[] = {0, 1024, 4096};

  for (uint64_t offset : offsets) {
    Qnn_ContextBinaryDmaDataResponse_t response{};
    ASSERT_EQ(manager->MapDmaData(MakeDmaRequest(offset, kRegionSize), &response, file.data(),
                                  kFakeFileSize),
              QNN_SUCCESS)
        << "offset " << offset;
  }
  ASSERT_EQ(mock->RegisteredBufferCount(), 3u);

  // Release the middle region the way QNN would, before teardown.
  ASSERT_EQ(manager->ReleaseDmaData(MakeDmaDataMem(file.data() + offsets[1], kRegionSize),
                                    file.data()),
            QNN_SUCCESS);
  EXPECT_EQ(mock->RegisteredBufferCount(), 2u);

  manager->ReleaseResources();

  EXPECT_EQ(mock->RegisteredBufferCount(), 0u);
  EXPECT_EQ(mock->RegisterCallCount(), 3u);
  EXPECT_EQ(mock->RegisterCallCount(), mock->SuccessfulDeregisterCallCount());
}

// Pins the fd/attr convention the EP and the FastRPC driver agree on: registration
// passes NULL as the fd, deregistration passes -1, and both carry
// RPCMEM_ATTR_IMPORT_BUFFER | RPCMEM_ATTR_READ_ONLY.
TEST_F(QnnUnit_BackendManagerHtpTest, MapDmaData_ThenRelease_UsesImportReadOnlyAttrAndFdConvention) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);
  constexpr uint64_t kRegionSize = 512;
  Qnn_ContextBinaryDmaDataResponse_t response{};
  ASSERT_EQ(manager->MapDmaData(MakeDmaRequest(0, kRegionSize), &response, file.data(), kFakeFileSize),
            QNN_SUCCESS);
  ASSERT_EQ(manager->ReleaseDmaData(MakeDmaDataMem(file.data(), kRegionSize), file.data()),
            QNN_SUCCESS);

  const auto calls = mock->RegisterBufCalls();
  ASSERT_EQ(calls.size(), 2u);
  const int expected_attr = qnn::rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | qnn::rpcmem::RPCMEM_ATTR_READ_ONLY;

  EXPECT_FALSE(calls[0].IsDeregister());
  EXPECT_EQ(calls[0].fd, 0);  // registration passes NULL
  EXPECT_EQ(calls[0].attr, expected_attr);
  EXPECT_EQ(calls[0].size, kRegionSize);

  EXPECT_TRUE(calls[1].IsDeregister());  // fd == -1
  EXPECT_EQ(calls[1].attr, expected_attr);
  EXPECT_EQ(calls[1].size, kRegionSize);
}

// ---------------------------------------------------------------------------
// MapDmaData
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendManagerHtpTest, MapDmaData_ValidRequest_PopulatesResponseAndRegistersBuffer) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);
  constexpr uint64_t kOffset = 2048;
  constexpr uint64_t kRegionSize = 128;
  Qnn_ContextBinaryDmaDataResponse_t response{};

  ASSERT_EQ(manager->MapDmaData(MakeDmaRequest(kOffset, kRegionSize), &response, file.data(),
                                kFakeFileSize),
            QNN_SUCCESS);

  // The EP hands QNN the fd for base+offset, and reports the region as starting at 0
  // because the registered pointer is already offset into the mapping.
  EXPECT_EQ(response.dmaBuffer.data, file.data() + kOffset);
  EXPECT_NE(response.dmaBuffer.fd, -1);
  EXPECT_EQ(response.dataStartOffset, 0u);
  EXPECT_EQ(response.alignedSize, kRegionSize);
  EXPECT_TRUE(mock->IsRegistered(file.data() + kOffset));
  EXPECT_EQ(mock->RegisteredBufferCount(), 1u);
}

// Each mapped region gets its own FastRPC registration and fd.
TEST_F(QnnUnit_BackendManagerHtpTest, MapDmaData_MultipleOffsets_RegistersDistinctFds) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);
  Qnn_ContextBinaryDmaDataResponse_t first{};
  Qnn_ContextBinaryDmaDataResponse_t second{};
  ASSERT_EQ(manager->MapDmaData(MakeDmaRequest(0, 256), &first, file.data(), kFakeFileSize),
            QNN_SUCCESS);
  ASSERT_EQ(manager->MapDmaData(MakeDmaRequest(8192, 256), &second, file.data(), kFakeFileSize),
            QNN_SUCCESS);

  EXPECT_NE(first.dmaBuffer.fd, second.dmaBuffer.fd);
  EXPECT_NE(first.dmaBuffer.data, second.dmaBuffer.data);
  EXPECT_EQ(mock->RegisteredBufferCount(), 2u);
}

// With file mapping disabled the call must bail out before touching rpcmem -- which
// also means a null rpcmem library is safe here.
TEST_F(QnnUnit_BackendManagerHtpTest, MapDmaData_FileMappingDisabled_ReturnsAborted) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtp(*manager).IsOK());
  ASSERT_FALSE(manager->FileMappingIsEnabled());

  std::vector<char> file(kFakeFileSize);
  Qnn_ContextBinaryDmaDataResponse_t response{};
  EXPECT_EQ(manager->MapDmaData(MakeDmaRequest(0, 256), &response, file.data(), kFakeFileSize),
            QNN_CONTEXT_ERROR_ABORTED);
}

TEST_F(QnnUnit_BackendManagerHtpTest, MapDmaData_InvalidRequests_ReturnInvalidArgument) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);
  Qnn_ContextBinaryDmaDataResponse_t response{};

  EXPECT_EQ(manager->MapDmaData(MakeDmaRequest(0, 256), &response, nullptr, kFakeFileSize),
            QNN_CONTEXT_ERROR_INVALID_ARGUMENT)
      << "null mapped base pointer";

  EXPECT_EQ(manager->MapDmaData(MakeDmaRequest(0, 0), &response, file.data(), kFakeFileSize),
            QNN_CONTEXT_ERROR_INVALID_ARGUMENT)
      << "zero-size region";

  EXPECT_EQ(manager->MapDmaData(MakeDmaRequest(0, 256, /*backend_mapping_needed=*/false), &response,
                                file.data(), kFakeFileSize),
            QNN_CONTEXT_ERROR_INVALID_ARGUMENT)
      << "backend mapping not requested";

  // offset + size would wrap 64 bits.
  EXPECT_EQ(manager->MapDmaData(MakeDmaRequest(UINT64_MAX, 2), &response, file.data(), kFakeFileSize),
            QNN_CONTEXT_ERROR_INVALID_ARGUMENT)
      << "offset + size overflows";

  // Region runs past the end of the mapped file.
  EXPECT_EQ(manager->MapDmaData(MakeDmaRequest(kFakeFileSize - 10, 100), &response, file.data(),
                                kFakeFileSize),
            QNN_CONTEXT_ERROR_INVALID_ARGUMENT)
      << "region extends beyond file";

  // None of the rejected requests may have reached rpcmem.
  EXPECT_TRUE(mock->RegisterBufCalls().empty());
  EXPECT_EQ(mock->RegisteredBufferCount(), 0u);
}

// When registration fails the EP must report it and track nothing -- there is no
// buffer to deregister later, so this path must not leak a bookkeeping entry.
TEST_F(QnnUnit_BackendManagerHtpTest, MapDmaData_RegistrationFails_ReturnsSystemErrorAndTracksNothing) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);
  mock->FailRegisterFor(file.data());

  Qnn_ContextBinaryDmaDataResponse_t response{};
  EXPECT_EQ(manager->MapDmaData(MakeDmaRequest(0, 256), &response, file.data(), kFakeFileSize),
            QNN_COMMON_ERROR_SYSTEM);

  EXPECT_EQ(mock->RegisterCallCount(), 1u);
  EXPECT_FALSE(mock->RegisterBufCalls().front().succeeded);
  EXPECT_EQ(mock->RegisteredBufferCount(), 0u);

  // Teardown has nothing to sweep, and must not attempt a deregistration.
  manager->ReleaseResources();
  EXPECT_EQ(mock->DeregisterCallCount(), 0u);
}

// ---------------------------------------------------------------------------
// ReleaseDmaData
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendManagerHtpTest, ReleaseDmaData_InvalidArguments_ReturnInvalidArgument) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);

  EXPECT_EQ(manager->ReleaseDmaData(MakeDmaDataMem(file.data(), 256), nullptr),
            QNN_CONTEXT_ERROR_INVALID_ARGUMENT)
      << "null mapped base pointer";

  EXPECT_EQ(manager->ReleaseDmaData(MakeDmaDataMem(nullptr, 256), file.data()),
            QNN_CONTEXT_ERROR_INVALID_ARGUMENT)
      << "null dma buffer";

  EXPECT_EQ(manager->ReleaseDmaData(MakeDmaDataMem(file.data(), 0), file.data()),
            QNN_CONTEXT_ERROR_INVALID_ARGUMENT)
      << "zero mem size";

  EXPECT_TRUE(mock->RegisterBufCalls().empty());
}

// A failed deregistration must be reported AND the entry kept, so the teardown sweep
// can retry it. Regression test for "Remove successfully deregistered fastrpc buffers
// from container".
TEST_F(QnnUnit_BackendManagerHtpTest, ReleaseDmaData_DeregistrationFails_ReturnsMemAllocAndRetainsBuffer) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);
  constexpr uint64_t kRegionSize = 256;
  Qnn_ContextBinaryDmaDataResponse_t response{};
  ASSERT_EQ(manager->MapDmaData(MakeDmaRequest(0, kRegionSize), &response, file.data(), kFakeFileSize),
            QNN_SUCCESS);

  mock->FailDeregisterFor(file.data());
  EXPECT_EQ(manager->ReleaseDmaData(MakeDmaDataMem(file.data(), kRegionSize), file.data()),
            QNN_CONTEXT_ERROR_MEM_ALLOC);

  // Still registered, so the buffer is still owned and must be retried.
  EXPECT_TRUE(mock->IsRegistered(file.data()));
  EXPECT_EQ(mock->SuccessfulDeregisterCallCount(), 0u);
}

// Deregistering something that was never registered is indistinguishable from a
// successful deregistration (to_fd() already reports -1), so the EP reports success.
TEST_F(QnnUnit_BackendManagerHtpTest, ReleaseDmaData_NeverRegisteredPointer_ReturnsSuccess) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);
  EXPECT_EQ(manager->ReleaseDmaData(MakeDmaDataMem(file.data(), 256), file.data()), QNN_SUCCESS);
  EXPECT_EQ(mock->DeregisterCallCount(), 1u);
  EXPECT_EQ(mock->RegisteredBufferCount(), 0u);
}

// ---------------------------------------------------------------------------
// Teardown sweep (DeallocateMappedDmaBuffers)
// ---------------------------------------------------------------------------

// Buffers whose deregistration fails during the sweep are retained so a later sweep
// retries them; the rest are dropped. Regression test for "Ensure all mapped fastrpc
// buffers are freed".
TEST_F(QnnUnit_BackendManagerHtpTest, ReleaseResources_DeregistrationFails_RetainsOnlyFailuresAndRetries) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);
  constexpr uint64_t kRegionSize = 256;
  void* const first = file.data();
  void* const second = file.data() + 1024;

  Qnn_ContextBinaryDmaDataResponse_t response{};
  ASSERT_EQ(manager->MapDmaData(MakeDmaRequest(0, kRegionSize), &response, file.data(), kFakeFileSize),
            QNN_SUCCESS);
  ASSERT_EQ(manager->MapDmaData(MakeDmaRequest(1024, kRegionSize), &response, file.data(), kFakeFileSize),
            QNN_SUCCESS);

  // Sweep 1: `second` fails and must survive; `first` is released and dropped.
  mock->FailDeregisterFor(second);
  manager->ReleaseResources();
  EXPECT_FALSE(mock->IsRegistered(first));
  EXPECT_TRUE(mock->IsRegistered(second));
  EXPECT_EQ(mock->SuccessfulDeregisterCallCount(), 1u);

  // Sweep 2 (ReleaseResources is not guarded against repeat calls, and the destructor
  // calls it again): with the fault cleared the retained buffer is finally released.
  mock->ClearFaults();
  manager->ReleaseResources();
  EXPECT_FALSE(mock->IsRegistered(second));
  EXPECT_EQ(mock->RegisteredBufferCount(), 0u);
  EXPECT_EQ(mock->SuccessfulDeregisterCallCount(), 2u);

  // Destruction runs a third sweep over an empty container; nothing more is attempted.
  const size_t deregister_calls = mock->DeregisterCallCount();
  manager.reset();
  EXPECT_EQ(mock->DeregisterCallCount(), deregister_calls);
}

// Combines a registration failure with a deregistration failure: of two weight-set
// buffers, only the first is ever registered; that survivor is then retained (not
// leaked, not immediately cleared) across one failed sweep before a retry sweep
// finally deallocates it.
TEST_F(QnnUnit_BackendManagerHtpTest,
       MapDmaData_SecondRegistrationFailsThenRetainedBufferDeregistrationFails_DeallocatesOnlyAfterRetry) {
  StubApiEnv env;
  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(SetupBackendHtpWithFileMapping(*manager, mock).IsOK());

  std::vector<char> file(kFakeFileSize);
  constexpr uint64_t kRegionSize = 256;
  void* const first = file.data();
  void* const second = file.data() + 1024;

  Qnn_ContextBinaryDmaDataResponse_t response{};
  ASSERT_EQ(manager->MapDmaData(MakeDmaRequest(0, kRegionSize), &response, file.data(), kFakeFileSize),
            QNN_SUCCESS);

  mock->FailRegisterFor(second);
  EXPECT_EQ(manager->MapDmaData(MakeDmaRequest(1024, kRegionSize), &response, file.data(), kFakeFileSize),
            QNN_COMMON_ERROR_SYSTEM);
  EXPECT_EQ(mock->RegisteredBufferCount(), 1u);

  // Sweep 1: the surviving buffer's deregistration fails and it must be retained --
  // neither dropped nor leaked.
  mock->FailDeregisterFor(first);
  manager->ReleaseResources();
  EXPECT_TRUE(mock->IsRegistered(first));

  // Sweep 2 (retry): with the fault cleared, the retained buffer is finally released.
  mock->ClearFaults();
  manager->ReleaseResources();
  EXPECT_FALSE(mock->IsRegistered(first));
  EXPECT_EQ(mock->RegisteredBufferCount(), 0u);
}

// End-to-end variant of the two combined tests above, driven through the real production
// entry point (LoadCachedQnnContextFromBuffer) against a genuine multi-weight-set HTP
// context binary instead of synthetic MakeDmaRequest() calls.
//
// Trace note (qnn_backend_manager.cc): a registration failure on one weight-set region does
// NOT fail the overall load.
//   - MapDmaData (:1094-1151) only appends to mapped_fastrpc_buffers_ on success (:1148); on
//     a register_buf()-then-to_fd() failure it returns QNN_COMMON_ERROR_SYSTEM (:1136-1139)
//     with no bookkeeping entry for that region.
//   - CreateContextHandleFromBinary (:1333-1403) treats that as a failure of
//     contextCreateFromBinaryWithCallback (rt != QNN_SUCCESS, :1372): it calls
//     DeallocateMappedDmaBuffers() for cleanup (:1373) and then unconditionally falls through
//     (:1382-1398) to ReadContextBinIfValid + plain contextCreateFromBinary (no file mapping).
//     Only a failure of *that* direct-read attempt propagates as an error (:1400-1402).
//   So LoadCachedQnnContextFromBuffer's overall status is expected to be IsOK() even though
//   one weight-set registration was forced to fail.
//
// This also means the EP's own fallback cleanup at :1373 runs a DeallocateMappedDmaBuffers()
// sweep before the test ever regains control. If that internal sweep were allowed to succeed,
// it would deregister (and drop the bookkeeping for) the surviving first-registered buffer
// before step 6/7 below could inspect it. So the injected hook fails every deregistration
// call too, until it is cleared right after LoadCachedQnnContextFromBuffer returns -- at which
// point the test's own ReleaseResources() sweeps take over exactly like the two tests above.
//
// Also note buffer_length semantics (:1945-1969 in LoadCachedQnnContextFromBuffer): a nonzero
// buffer_length is interpreted as an embedded context and forcibly disables file mapping
// (use_file_mapping = false), so MapDmaData would never be called. Exercising the file-mapped
// path requires buffer_length == 0 plus a real on-disk context_bin_filepath, which is why this
// test passes nullptr/0 and the path from CompileMultiWeightSetContextBinary() instead of a
// byte buffer.
TEST_F(QnnUnit_BackendManagerHtpTest,
       LoadCachedQnnContextFromBuffer_MultiWeightSetContext_SurvivesRegistrationFailureAndRetriesFailedDeregistration) {
  StubApiEnv env;  // owns the logger; must outlive the manager
  const std::filesystem::path context_bin_path = CompileMultiWeightSetContextBinary();
  // Declared before `manager` so it is destroyed after: the WindowsFileMapper owned by
  // `manager` must unmap its view of this file before the file is deleted.
  auto cleanup = gsl::finally([&]() {
    std::error_code ec;
    std::filesystem::remove(context_bin_path, ec);
  });

  auto mock = std::make_shared<MockRpcMemLibrary>();
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  auto setup_status = manager->SetupBackend(/*load_from_cached_context=*/true,
                                            /*need_load_system_lib=*/true,
                                            /*share_ep_contexts=*/false,
                                            /*htp_share_resource_optimization=*/-1,
                                            /*enable_file_mapped_weights=*/true,
                                            mock,
                                            dummy_map);
  ASSERT_TRUE(setup_status.IsOK()) << "SetupBackend failed: " << setup_status.GetErrorMessage();
  ASSERT_TRUE(manager->FileMappingIsEnabled());

  int registration_calls = 0;
  mock->SetRegisterBufHook([&](const RpcMemRegisterBufCall& call) {
    if (call.IsDeregister()) {
      // Fail every deregistration until explicitly cleared below, so the EP's own
      // fallback cleanup (DeallocateMappedDmaBuffers() at CreateContextHandleFromBinary:1373)
      // cannot wipe the surviving buffer before this test inspects it.
      return false;
    }
    ++registration_calls;
    return registration_calls != 2;  // fail only the second registration
  });

  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> qnn_models;
  qnn::EpContextIoDispatch dummy_io_dispatch(nullptr);
  auto status = manager->LoadCachedQnnContextFromBuffer(
      /*buffer=*/nullptr, /*buffer_length=*/0, context_bin_path.string(), "test_node",
      qnn_models, /*max_spill_fill_size=*/0, dummy_io_dispatch);

  // See the trace note above: a single failed weight-set registration is recovered via the
  // direct-read fallback, so the overall load still succeeds.
  EXPECT_TRUE(status.IsOK()) << "LoadCachedQnnContextFromBuffer failed: " << status.GetErrorMessage();

  ASSERT_GE(registration_calls, 2);
  ASSERT_GE(mock->RegisteredBufferCount(), 1u);

  // Deregistration can succeed again now; pick a surviving buffer and drive the same
  // fail-sweep-then-retry sequence as the synthetic tests above.
  mock->SetRegisterBufHook(nullptr);
  const auto registered = mock->RegisteredBuffers();
  ASSERT_FALSE(registered.empty());
  void* const survivor = registered.front().first;

  mock->FailDeregisterFor(survivor);
  manager->ReleaseResources();
  EXPECT_TRUE(mock->IsRegistered(survivor));

  mock->ClearFaults();
  manager->ReleaseResources();
  EXPECT_FALSE(mock->IsRegistered(survivor));
}

#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
