// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for QnnBackendManager (qnn_backend_manager.cc).
//
// Three groups of tests live here, gated on QNN_EP_INTERNAL_SYMBOL_ACCESS:
//
//   1. Stub-based tests (no real QNN library) — QnnSerializerConfig, SetupBackend
//      load-failure paths, and before-setup early returns. These always run.
//   2. Real-HTP-backend tests (QnnUnit_BackendManagerHtpTest) — load the platform
//      HTP backend (and QnnIr / QnnSaver) and drive SetupBackend directly, with no
//      ORT session. The fixture GTEST_SKIP()s when the backend is unavailable,
//      mirroring the QnnHTPBackendTests::SetUp() convention.
//   3. Binary-loading tests — use instance-local SDK/FastRPC stubs, with no driver required.
//
// Coverage targets:
//   - QnnSerializerConfig (CreateIr / CreateSaver / GetBackendPath / SetGraphName / Configure)
//   - SetupBackend: load-failure paths (stub) + config permutations against real HTP
//     (priority / device / profiling), serializer backends (Saver / Ir)
//   - SetContextPriority / ResetContextPriority, QnnBackendProfilingManager::SetProfilingLevelETW
//   - ResetQnnLogLevel (before setup + after setup)
//   - GetContextBinaryBuffer (before setup + after setup) / LoadCachedQnnContextFromBuffer
//   - ParseLoraConfig file I/O error paths

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "test/providers/qnn/infra/qnn_test_windows_compat.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/ort_api.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"
#endif

namespace onnxruntime {
namespace test {

// ===========================================================================
// Test helpers
// ===========================================================================

static std::shared_ptr<qnn::QnnBackendManager> MakeManager(
    const std::string& backend_path,
    const ApiPtrs& api_ptrs,
    const Ort::Logger& logger,
    bool skip_version_check = true,
    bool configure_host_mode = false) {
  qnn::QnnBackendManagerConfig cfg;
  cfg.backend_path = backend_path;
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
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  EXPECT_FALSE(manager->IsBackendHostMode());
}

TEST(QnnUnit_BackendManagerTest, IsBackendHostMode_ConfigureHostModeTrue_ReturnsTrue) {
  StubApiEnv env;
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger,
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
  auto manager = MakeManager(QnnHtpBackendLibraryName(), env.api_ptrs, env.logger);
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
  auto manager = MakeManager(QnnHtpBackendLibraryName(), env.api_ptrs, env.logger);
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
  auto manager = MakeManager(QnnHtpBackendLibraryName(), env.api_ptrs, env.logger);
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
  auto manager = MakeManager(QnnHtpBackendLibraryName(), env.api_ptrs, env.logger);
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
  auto manager = MakeManager(QnnHtpBackendLibraryName(), env.api_ptrs, env.logger);
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
  auto manager = MakeManager(QnnHtpBackendLibraryName(), env.api_ptrs, env.logger);
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
  cfg.backend_path = QnnHtpBackendLibraryName();
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
  auto manager = MakeManager(QnnHtpBackendLibraryName(), env.api_ptrs, env.logger);
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
  auto manager = MakeManager(QnnHtpBackendLibraryName(), env.api_ptrs, env.logger);
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
// The tests below load a real QNN backend (HTP, and for the serializer cases
// QnnIr / QnnSaver) and drive QnnBackendManager directly — no
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
  cfg.backend_path = QnnHtpBackendLibraryName();
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

// Fixture: probes HTP backend availability once (cached) and skips the whole
// group via GTEST_SKIP() when the platform HTP backend cannot be loaded — mirroring the
// established QnnHTPBackendTests::SetUp() convention (backend unavailable → skip,
// not fail). This keeps CI signal clean on environments without the HTP library
// while leaving each test's ASSERT_TRUE(status.IsOK()) as a genuine behavioral
// check once HTP is confirmed present.
class QnnUnit_BackendManagerHtpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!HtpAvailable()) {
      GTEST_SKIP() << "QNN HTP backend (" << QnnHtpBackendLibraryName() << ") is not available! Skipping test.";
    }
  }

 private:
  // Probe once with SetupBackend on platforms where repeated initialization is safe.
  static bool HtpAvailable() {
#if defined(__linux__) && defined(__aarch64__) && !defined(__ANDROID__)
    // The V68 ARM64 validation run crashes after this probe has initialized HTP
    // and the first test initializes it again. The first test below is the real
    // SetupBackend validation, so do not consume that initialization here.
    return true;
#else
    static const bool available = [] {
      StubApiEnv env;
      auto manager = MakeHTPManager(env.api_ptrs, env.logger);
      return manager != nullptr && SetupBackendHtp(*manager).IsOK();
    }();
    return available;
#endif
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
  ASSERT_TRUE(status.IsOK()) << QnnHtpBackendLibraryName() << " setup failed: " << status.GetErrorMessage();
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

// The x86 HTP emulator accepts arbitrary arch values, so GetPlatformInfo() copies
// the configured value. ARM64 instead queries the physical HTP architecture.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithHtpArch_ReportsPlatformArch) {
  StubApiEnv env;
#if QNN_ARCH_ARM64
  constexpr auto configured_htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
#else
  constexpr auto configured_htp_arch = QNN_HTP_DEVICE_ARCH_V73;
#endif
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::OFF, qnn::ProfilingLevel::OFF,
                                configured_htp_arch);
  ASSERT_NE(manager, nullptr);
  // A configured htp_arch reaches htp_arch_ only; the internal holder stays NONE
  // until setup runs.
  ASSERT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);

  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
#if QNN_ARCH_ARM64
  EXPECT_NE(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
#else
  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_V73);
#endif
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
  qnn::EpContextIoDispatch dummy_io_dispatch;
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
// Loading the platform QNN IR backend as the main backend exercises:
//   - SetQnnBackendType IR/SAVER case: backend_id → QnnBackendType::SERIALIZER
//   - CreateContext SERIALIZER branch: configs = nullptr
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_WithIrBackendDirectly_SetsSerializerBackendType) {
  StubApiEnv env;
  qnn::QnnBackendManagerConfig cfg{};  // value-init to zero all fields (profiling, device_id, etc.)
  cfg.backend_path = QnnIrBackendLibraryName();
  cfg.context_priority = qnn::ContextPriority::NORMAL;
  cfg.skip_qnn_version_check = true;
  auto manager = qnn::QnnBackendManager::Create(cfg, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << QnnIrBackendLibraryName() << " setup failed: " << status.GetErrorMessage();
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
      QnnHtpBackendLibraryName(),
      qnn::QnnSerializerConfig::CreateSaver(QnnSaverBackendLibraryName()),
      env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "QnnSaver+HTP setup failed: " << status.GetErrorMessage();

  EXPECT_NE(manager->GetQnnSerializerConfig(), nullptr);
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// QnnIrConfig::SupportsArbitraryGraphConfigs() returns false, so CreateContext
// overrides configs to nullptr even for the HTP backend's default configs.
// Distinct from SetupBackend_WithIrBackendDirectly above which loads QnnIr
// as the main backend (SERIALIZER type) without QnnSerializerConfig.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithQnnIrSerializer_CoversNoArbitraryGraphConfigs) {
  StubApiEnv env;
  const auto tmp_dir = std::filesystem::temp_directory_path() / "qnn_ir_serializer_htp_test";
  std::filesystem::create_directories(tmp_dir);

  auto manager = MakeSerializerManager(
      QnnHtpBackendLibraryName(),
      qnn::QnnSerializerConfig::CreateIr(QnnIrBackendLibraryName(), tmp_dir.string()),
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
                                      qnn::EpContextIoDispatch(),
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
                                      qnn::EpContextIoDispatch(),
                                      false /*extended_udma*/, false /*prepare_only*/,
                                      true /*enable_htp_graph_splitting*/,
                                      4 /*htp_graph_splitting_num_prepare_threads*/);
  (void)status;
}

class QnnUnit_ContextBinaryLoadingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend_ = std::make_unique<StubBackendManager>(api_ptrs_, logger_);
    manager_ = backend_->GetMutable();
    current_ = this;
    // Override only this manager's API table; no SDK or process-wide API is modified.
    auto& interface = backend_->QnnInterface();
    interface.errorGetMessage = [](Qnn_ErrorHandle_t error, const char** message) -> Qnn_ErrorHandle_t {
      if (error == QNN_COMMON_ERROR_NOT_SUPPORTED) {
        *message = "stub DMA rejection";
      } else {
        EXPECT_EQ(error, current_->sdk_result_);
        *message = "stub context loading failure";
      }
      return QNN_SUCCESS;
    };
    interface.contextCreateFromBinary = [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                           const QnnContext_Config_t** configs, const void* binary,
                                           Qnn_ContextBinarySize_t bytes, Qnn_ContextHandle_t* context,
                                           Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      ++test.ordinary_calls_;
      EXPECT_EQ(binary, test.expected_binary_);
      EXPECT_EQ(bytes, test.expected_size_);
      if (test.check_configs_) {
        EXPECT_EQ(configs, test.configs_);
      }
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
      if (test.check_configs_ && test.dma_calls_) {
        EXPECT_EQ(configs, test.dma_configs_);
      }
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        EXPECT_FALSE(test.system_freed_);
        test.events_.push_back("ordinary");
      }
#endif
      if (test.sdk_result_ == QNN_SUCCESS) {
        *context = const_cast<void*>(binary);
      }
      return test.sdk_result_;
    };
    interface.contextFree = [](Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      return QNN_SUCCESS;
    };
  }
  void TearDown() override {
    backend_.reset();
    manager_ = nullptr;
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
    EXPECT_TRUE(registrations_.empty());
    for (const auto& path : extra_paths_) {
      EXPECT_TRUE(DeleteFileW(path.c_str()));
    }
    if (!path_.empty()) {
      EXPECT_TRUE(DeleteFileW(path_.c_str()));
    }
#endif
    current_ = nullptr;
  }
  Ort::Status Load(bool request_dma, Qnn_ContextHandle_t& context) {
    qnn::EpContextIoDispatch io_dispatch;
    // A nonexistent filename makes an accidental fallback reread fail.
    return manager_->CreateContextHandleFromBinary(const_cast<void*>(expected_binary_), expected_size_, request_dma,
                                                   configs_, "not-a-context-file.bin", io_dispatch,
                                                   nullptr, context);
  }

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  void EnableDma() {
    qnn::RpcMemApi api{};
    api.register_buf = [](void* pointer, size_t size, int fd, int attributes) {
      auto& test = *current_;
      EXPECT_EQ(attributes, qnn::rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | qnn::rpcmem::RPCMEM_ATTR_READ_ONLY);
      if (fd == -1) {
        auto it = test.registrations_.find(pointer);
        ASSERT_NE(it, test.registrations_.end());
        EXPECT_EQ(size, it->second);
        ++test.deregister_calls_;
        if (!test.fail_deregister_) {
          test.registrations_.erase(it);
        }
      } else {
        EXPECT_EQ(fd, 0);
        EXPECT_EQ(test.registrations_.count(pointer), 0u);
        ++test.register_calls_;
        if (test.register_calls_ != test.fail_register_call_) {
          test.registrations_.emplace(pointer, size);
        }
      }
    };
    api.to_fd = [](void* pointer) {
      return current_->registrations_.count(pointer) ? 42 : -1;
    };
    rpcmem_ = std::make_shared<qnn::RpcMemLibrary>(api);
    backend_->FileMappingEnabled() = true;
    if (!backend_->FileMapper()) {
      backend_->FileMapper() = std::make_unique<qnn::WindowsFileMapper>(logger_);
    }
    auto& interface = backend_->QnnInterface();
    interface.contextCreateFromBinaryWithCallback = [](
                                                        Qnn_BackendHandle_t, Qnn_DeviceHandle_t, const QnnContext_Config_t** configs,
                                                        const Qnn_ContextBinaryCallback_t* callbacks, const void* binary,
                                                        Qnn_ContextBinarySize_t bytes, Qnn_ContextHandle_t* context,
                                                        Qnn_ProfileHandle_t, Qnn_SignalHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      ++test.dma_calls_;
      test.dma_configs_ = configs;
      EXPECT_EQ(binary, test.expected_binary_);
      EXPECT_EQ(bytes, test.expected_size_);
      if (test.check_configs_) {
        EXPECT_EQ(configs, test.configs_);
      }
      EXPECT_EQ(callbacks->type, QNN_CONTEXT_CALLBACK_DMA_BUFFER);
      EXPECT_EQ(test.RequestMapping(*callbacks, binary, 1, bytes - 1), QNN_SUCCESS);
      if (test.repeat_mapping_) {
        EXPECT_EQ(test.RequestMapping(*callbacks, binary, 1, bytes - 1), QNN_SUCCESS);
      }
      if (test.request_second_mapping_) {
        EXPECT_EQ(test.RequestMapping(*callbacks, binary, 0, 1), QNN_CONTEXT_ERROR_MEM_ALLOC);
      }
      if (test.dma_result_ == QNN_SUCCESS) *context = const_cast<void*>(binary);
      return test.dma_result_;
    };
  }
  Qnn_ErrorHandle_t RequestMapping(const Qnn_ContextBinaryCallback_t& callbacks, const void* binary,
                                   uint64_t offset, uint64_t size) {
    Qnn_ContextBinaryDataRequest_t request{};
    request.offset = offset;
    request.size = size;
    request.isBackendMappingNeeded = 1;
    Qnn_ContextBinaryDmaDataResponse_t response{};
    const auto& dma = callbacks.dmaBufferCallback.v1;
    auto status = dma.dataProvide(request, &response, dma.notifyParam);
    if (status == QNN_SUCCESS) {
      EXPECT_EQ(response.dmaBuffer.data, static_cast<const char*>(binary) + offset);
      EXPECT_EQ(response.dmaBuffer.fd, 42);
      EXPECT_EQ(response.alignedSize, size);
      EXPECT_EQ(response.dataStartOffset, 0u);
      mappings_.push_back({callbacks, response, size});
    }
    return status;
  }

  Qnn_ErrorHandle_t ReleaseMapping(size_t index) {
    const auto& mapping = mappings_.at(index);
    Qnn_ContextBinaryDmaDataMem_t memory{};
    memory.dmaBuffer = mapping.response.dmaBuffer;
    memory.memSize = mapping.size;
    const auto& dma = mapping.callbacks.dmaBufferCallback.v1;
    return dma.dataRelease(memory, dma.notifyParam);
  }

  void EnableListFailure(bool partial_success) {
    backend_->BackendSetupCompleted() = true;
    backend_->HtpShareResourceOptimization() = 1;
    backend_->QnnInterface().contextCreateFromBinaryListAsync = [](
                                                                    Qnn_BackendHandle_t, Qnn_DeviceHandle_t, const QnnContext_Params_t** params,
                                                                    const QnnContext_Config_t**, Qnn_SignalHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      if (params[0]->version == QNN_CONTEXT_PARAMS_VERSION_2) {
        ++test.list_dma_calls_;
        for (size_t i = 0; params[i]; ++i) {
          const auto& param = *params[i]->v2;
          EXPECT_EQ(test.RequestMapping(*param.callback, param.binaryBuffer, 1, param.binaryBufferSize - 1), QNN_SUCCESS);
          const bool succeeds = test.list_partial_success_ && i == 0;
          param.notifyFunc(succeeds ? const_cast<void*>(param.binaryBuffer) : nullptr, nullptr, nullptr,
                           QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT, param.notifyParam,
                           succeeds ? QNN_SUCCESS : QNN_COMMON_ERROR_NOT_SUPPORTED);
        }
        return QNN_COMMON_ERROR_NOT_SUPPORTED;
      }
      ++test.list_ordinary_calls_;
      for (size_t i = 0; params[i]; ++i) {
        const auto& param = params[i]->v1;
        param.notifyFunc(const_cast<void*>(param.binaryBuffer), nullptr, nullptr,
                         QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT, param.notifyParam, QNN_SUCCESS);
      }
      return QNN_SUCCESS;
    };
    list_partial_success_ = partial_success;
  }

  void PublishDma() {
    backend_->RpcMemLibrary() = rpcmem_;
  }

  class TrackedMapper : public qnn::FileMappingInterface {
   public:
    explicit TrackedMapper(QnnUnit_ContextBinaryLoadingTest& test) : test_(test), mapper_(test.logger_) {}
    Ort::Status GetContextBinMappedMemoryPtr(const std::string& path, void** data) override {
      auto status = mapper_.GetContextBinMappedMemoryPtr(path, data);
      if (status.IsOK()) test_.expected_binary_ = *data;
      return status;
    }
    Ort::Status MapContextBinary(const std::string& path, MappedMemoryPtr& data) override {
      MappedMemoryPtr view;
      auto status = mapper_.MapContextBinary(path, view);
      if (!status.IsOK()) return status;
      test_.expected_binary_ = view.get();
      test_.mapping_alive_ = true;
      if (test_.track_lifetime_) test_.events_.push_back("map");
      auto* pointer = view.get();
      auto unmap = view.get_deleter();
      view.release();
      data = MappedMemoryPtr(pointer, [this, unmap](void* ptr) {
        EXPECT_TRUE(test_.system_freed_);
        unmap(ptr);
        test_.mapping_alive_ = false;
        if (test_.track_lifetime_) test_.events_.push_back("unmap");
      });
      return status;
    }

   private:
    QnnUnit_ContextBinaryLoadingTest& test_;
    qnn::WindowsFileMapper mapper_;
  };

  void PrepareInitialLoad() {
    wchar_t folder[MAX_PATH], filename[MAX_PATH];
    ASSERT_NE(GetTempPathW(MAX_PATH, folder), 0u);
    ASSERT_NE(GetTempFileNameW(folder, L"qnn", 0, filename), 0u);
    path_ = filename;
    std::ofstream file(path_, std::ios::binary);
    file.write("native_graph", sizeof("native_graph"));
    ASSERT_TRUE(file.good());
    file.close();
    expected_size_ = sizeof("native_graph");
    check_configs_ = false;
    backend_->FileMappingEnabled() = true;
    backend_->FileMapper() = std::make_unique<TrackedMapper>(*this);

    auto& system = backend_->SystemInterface();
    system.systemContextCreate = [](QnnSystemContext_Handle_t* handle) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      test.system_freed_ = false;
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        test.events_.push_back("system_create");
      }
      *handle = &test;
      return QNN_SUCCESS;
    };
    system.systemContextGetBinaryInfo = [](
                                            QnnSystemContext_Handle_t, void* binary, uint64_t bytes,
                                            const QnnSystemContext_BinaryInfo_t** info, Qnn_ContextBinarySize_t* info_size) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      EXPECT_EQ(binary, test.expected_binary_);
      EXPECT_EQ(bytes, test.expected_size_);
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        test.events_.push_back("metadata");
      }
      if (test.metadata_result_ != QNN_SUCCESS) return test.metadata_result_;
      test.input_.version = QNN_TENSOR_VERSION_1;
      test.input_.v1.name = "input";
      test.input_.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
      test.input_.v1.dataType = QNN_DATATYPE_FLOAT_32;
      test.input_.v1.rank = 1;
      test.input_.v1.dimensions = &test.dimension_;
      test.output_ = test.input_;
      test.output_.v1.name = "output";
      test.output_.v1.type = QNN_TENSOR_TYPE_APP_READ;
      test.graph_info_.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;
      auto& graph = test.graph_info_.graphInfoV1;
      graph.graphName = static_cast<const char*>(binary);
      graph.numGraphInputs = 1;
      graph.graphInputs = &test.input_;
      graph.numGraphOutputs = 1;
      graph.graphOutputs = &test.output_;
      test.binary_info_.version = QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1;
      auto& context = test.binary_info_.contextBinaryInfoV1;
      context.contextBlobVersion = {3, 3, 3};
      context.numGraphs = 1;
      context.graphs = &test.graph_info_;
      *info = &test.binary_info_;
      *info_size = sizeof(test.binary_info_);
      return QNN_SUCCESS;
    };
    system.systemContextFree = [](QnnSystemContext_Handle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        test.events_.push_back("system_free");
      }
      test.system_freed_ = true;
      return QNN_SUCCESS;
    };
    auto& interface = backend_->QnnInterface();
    interface.graphRetrieve = [](Qnn_ContextHandle_t context, const char* name,
                                 Qnn_GraphHandle_t* graph) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      EXPECT_STREQ(name, static_cast<const char*>(test.expected_binary_));
      EXPECT_FALSE(test.system_freed_);
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        test.events_.push_back("graph");
      }
      *graph = context;
      return QNN_SUCCESS;
    };
  }

  Ort::Status LoadNative(std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& models) {
    return manager_->LoadCachedQnnContextFromBuffer(nullptr, 0, path_.string(), "native", models,
                                                    0, qnn::EpContextIoDispatch{});
  }

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION
  static OrtStatus* ORT_API_CALL ReadCallback(void* opaque, const char*, OrtAllocator* allocator,
                                              void** data, size_t* size) {
    auto& test = *static_cast<QnnUnit_ContextBinaryLoadingTest*>(opaque);
    ++test.read_calls_;
    *size = sizeof("callback_graph");
    auto status = Ort::GetApi().AllocatorAlloc(allocator, *size, data);
    if (status) return status;
    std::memcpy(*data, "callback_graph", *size);
    return nullptr;
  }
  void CheckSharingOrder(bool callback_first) {
    PrepareInitialLoad();
    ASSERT_FALSE(path_.empty());
    EnableDma();
    dma_result_ = QNN_SUCCESS;
    // Exercise SetupBackend's reused-manager branch after callback-only initialization.
    backend_->BackendSetupCompleted() = true;
    auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&Ort::GetApi());
    ASSERT_NE(setter, nullptr);
    Ort::SessionOptions options;
    Ort::ThrowOnError(setter(options, ReadCallback, this));
    qnn::EpContextIoDispatch dispatch(*options, OrtApiRuntimeCapabilities{ORT_API_VERSION});
    ASSERT_TRUE(dispatch.HasReadCallback());
    std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> models;
    auto callback_load = [&] {
      std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
      ASSERT_TRUE(manager_->SetupBackend(true, true, true, 0, true, nullptr, binaries, dispatch,
                                         false, false, false, 0)
                      .IsOK());
      qnn::EpContextBuffer output;
      ASSERT_TRUE(manager_->ReadContextBinIfValid("not-a-context-file.bin", output, dispatch).IsOK());
      expected_binary_ = output.data();
      expected_size_ = output.size();
      ASSERT_TRUE(manager_->LoadCachedQnnContextFromBuffer(output.data(), output.size(), "not-a-context-file.bin",
                                                           "callback", models, 0, dispatch)
                      .IsOK());
    };
    auto native_load = [&] {
      std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
      ASSERT_TRUE(manager_->SetupBackend(true, true, true, 0, true, rpcmem_, binaries, qnn::EpContextIoDispatch{},
                                         false, false, false, 0)
                      .IsOK());
      EXPECT_EQ(backend_->RpcMemLibrary(), rpcmem_);
      expected_size_ = sizeof("native_graph");
      ASSERT_TRUE(LoadNative(models).IsOK());
    };
    if (callback_first) {
      callback_load();
      EXPECT_EQ(backend_->RpcMemLibrary(), nullptr);
      native_load();
    } else {
      native_load();
      callback_load();
      EXPECT_EQ(backend_->RpcMemLibrary(), rpcmem_);
    }
    ASSERT_EQ(models.size(), 2u);
    EXPECT_EQ(models.at("callback")->Name(), "callback_graph");
    EXPECT_EQ(models.at("native")->Name(), "native_graph");
    EXPECT_EQ(read_calls_, 1u);
    EXPECT_EQ(ordinary_calls_, 1u);
    EXPECT_EQ(dma_calls_, 1u);
    EXPECT_TRUE(backend_->FileMappingEnabled());
  }
#endif

  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_;
  std::filesystem::path path_;
  Qnn_ErrorHandle_t dma_result_ = QNN_COMMON_ERROR_NOT_SUPPORTED;
  Qnn_ErrorHandle_t metadata_result_ = QNN_SUCCESS;
  size_t dma_calls_ = 0, register_calls_ = 0, deregister_calls_ = 0, read_calls_ = 0;
  struct DmaMapping {
    Qnn_ContextBinaryCallback_t callbacks;
    Qnn_ContextBinaryDmaDataResponse_t response;
    uint64_t size;
  };
  std::unordered_map<void*, size_t> registrations_;
  std::vector<DmaMapping> mappings_;
  std::vector<std::filesystem::path> extra_paths_;
  size_t fail_register_call_ = 0, list_dma_calls_ = 0, list_ordinary_calls_ = 0;
  bool repeat_mapping_ = false, request_second_mapping_ = false, fail_deregister_ = false, list_partial_success_ = false;
  const QnnContext_Config_t** dma_configs_ = nullptr;
  bool mapping_alive_ = false, system_freed_ = false, track_lifetime_ = false;
  std::vector<std::string> events_;
  uint32_t dimension_ = 1;
  Qnn_Tensor_t input_ = QNN_TENSOR_INIT;
  Qnn_Tensor_t output_ = QNN_TENSOR_INIT;
  QnnSystemContext_GraphInfo_t graph_info_ = QNN_SYSTEM_CONTEXT_GRAPH_INFO_INIT;
  QnnSystemContext_BinaryInfo_t binary_info_{};
#endif

  static thread_local QnnUnit_ContextBinaryLoadingTest* current_;
  OrtApi ort_api_{};
  OrtEpApi ep_api_{};
  OrtModelEditorApi editor_api_{};
  ApiPtrs api_ptrs_{ort_api_, ep_api_, editor_api_};
  Ort::Logger logger_{MakeNullLogger()};
  std::unique_ptr<StubBackendManager> backend_;
  qnn::QnnBackendManager* manager_ = nullptr;
  char binary_[4] = {1, 2, 3, 4};
  const void* expected_binary_ = binary_;
  uint64_t expected_size_ = sizeof(binary_);
  bool check_configs_ = true;
  const QnnContext_Config_t* configs_[1] = {nullptr};
  size_t ordinary_calls_ = 0;
  Qnn_ErrorHandle_t sdk_result_ = QNN_SUCCESS;
};

thread_local QnnUnit_ContextBinaryLoadingTest* QnnUnit_ContextBinaryLoadingTest::current_ = nullptr;

TEST_F(QnnUnit_ContextBinaryLoadingTest, OrdinaryLoadingUsesSuppliedBufferWithoutReadingFile) {
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(false, context).IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, binary_);
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, OrdinaryLoadingRejectsMissingSdkApi) {
  Qnn_ContextHandle_t context = nullptr;
  backend_->QnnInterface().contextCreateFromBinary = nullptr;
  auto status = Load(false, context);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(ordinary_calls_, 0u);
  EXPECT_NE(std::string(status.GetErrorMessage()).find("Invalid function pointer"), std::string::npos);
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, OrdinarySdkFailureIsPropagated) {
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  Qnn_ContextHandle_t context = nullptr;
  auto status = Load(false, context);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, nullptr);
  EXPECT_NE(std::string(status.GetErrorMessage()).find("contextCreateFromBinary failed"), std::string::npos);
  EXPECT_NE(std::string(status.GetErrorMessage()).find("stub context loading failure"), std::string::npos);
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
TEST_F(QnnUnit_ContextBinaryLoadingTest, RepeatedRequestInOneLoadSharesRegistrationUntilFinalRelease) {
  EnableDma();
  PublishDma();
  dma_result_ = QNN_SUCCESS;
  repeat_mapping_ = true;
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  ASSERT_EQ(mappings_.size(), 2u);
  EXPECT_EQ(mappings_[0].callbacks.dmaBufferCallback.v1.notifyParam,
            mappings_[1].callbacks.dmaBufferCallback.v1.notifyParam);
  EXPECT_EQ(register_calls_, 1u);
  EXPECT_EQ(ReleaseMapping(0), QNN_SUCCESS);
  EXPECT_EQ(deregister_calls_, 0u);
  EXPECT_EQ(ReleaseMapping(1), QNN_SUCCESS);
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_TRUE(registrations_.empty());
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, RepeatedRequestsAreDeregisteredOnceDuringDestruction) {
  EnableDma();
  PublishDma();
  dma_result_ = QNN_SUCCESS;
  repeat_mapping_ = true;
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  ASSERT_EQ(mappings_.size(), 2u);
  EXPECT_EQ(register_calls_, 1u);
  backend_.reset();
  manager_ = nullptr;
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_TRUE(registrations_.empty());
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, RepeatedFailedLoadRetainsFailedDeregistrationForDestruction) {
  EnableDma();
  PublishDma();
  repeat_mapping_ = true;
  fail_deregister_ = true;
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  ASSERT_EQ(mappings_.size(), 2u);
  EXPECT_EQ(register_calls_, 1u);
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_EQ(registrations_.size(), 1u);
  EXPECT_EQ(ordinary_calls_, 1u);
  fail_deregister_ = false;
  backend_.reset();
  manager_ = nullptr;
  EXPECT_EQ(deregister_calls_, 2u);
  EXPECT_TRUE(registrations_.empty());
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, MissingDmaSdkApiFallsBackToSuppliedBuffer) {
  EnableDma();
  PublishDma();
  backend_->QnnInterface().contextCreateFromBinaryWithCallback = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  EXPECT_EQ(dma_calls_, 0u);
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, binary_);
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, SdkDmaRejectionRetriesSameBufferAfterDeregistering) {
  EnableDma();
  PublishDma();
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  EXPECT_EQ(dma_calls_, 1u);
  EXPECT_EQ(register_calls_, 1u);
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, binary_);
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, OrdinaryFailureAfterSdkDmaRejectionIsPropagated) {
  EnableDma();
  PublishDma();
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  Qnn_ContextHandle_t context = nullptr;
  auto status = Load(true, context);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(dma_calls_, 1u);
  EXPECT_EQ(register_calls_, 1u);
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, nullptr);
  EXPECT_NE(std::string(status.GetErrorMessage()).find("stub context loading failure"), std::string::npos);
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, LaterMappingFailureRollsBackOnlySuccessfulRegistrations) {
  EnableDma();
  PublishDma();
  request_second_mapping_ = true;
  fail_register_call_ = 2;
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  EXPECT_EQ(register_calls_, 2u);
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_TRUE(registrations_.empty());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, binary_);
  EXPECT_EQ(binary_[0], 1);
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, FailedLoadPreservesPreviousSuccessfulContextRegistration) {
  EnableDma();
  PublishDma();
  dma_result_ = QNN_SUCCESS;
  Qnn_ContextHandle_t previous = nullptr;
  ASSERT_TRUE(Load(true, previous).IsOK());
  ASSERT_TRUE(manager_->AddQnnContextHandle(previous).IsOK());
  auto* previous_range = mappings_.at(0).response.dmaBuffer.data;

  char next_binary[4] = {5, 6, 7, 8};
  expected_binary_ = next_binary;
  dma_result_ = QNN_COMMON_ERROR_NOT_SUPPORTED;
  Qnn_ContextHandle_t next = nullptr;
  ASSERT_TRUE(Load(true, next).IsOK());
  EXPECT_EQ(manager_->GetQnnContext(0), previous);
  EXPECT_EQ(register_calls_, 2u);
  EXPECT_EQ(deregister_calls_, 1u);
  ASSERT_EQ(registrations_.size(), 1u);
  EXPECT_EQ(registrations_.count(previous_range), 1u);
  EXPECT_EQ(ReleaseMapping(0), QNN_SUCCESS);
  EXPECT_EQ(deregister_calls_, 2u);
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, FailedLoadKeepsExactRangeUsedByEarlierContext) {
  EnableDma();
  PublishDma();
  dma_result_ = QNN_SUCCESS;
  Qnn_ContextHandle_t previous = nullptr;
  ASSERT_TRUE(Load(true, previous).IsOK());
  dma_result_ = QNN_COMMON_ERROR_NOT_SUPPORTED;
  Qnn_ContextHandle_t next = nullptr;
  ASSERT_TRUE(Load(true, next).IsOK());
  EXPECT_EQ(register_calls_, 1u);
  EXPECT_EQ(deregister_calls_, 0u);
  ASSERT_EQ(registrations_.size(), 1u);
  // A late release from the failed load cannot release the surviving load's range.
  EXPECT_EQ(ReleaseMapping(1), QNN_CONTEXT_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(registrations_.size(), 1u);
  EXPECT_EQ(ReleaseMapping(0), QNN_SUCCESS);
  EXPECT_EQ(deregister_calls_, 1u);
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, SharedExactRangeIsDeregisteredAtFinalRelease) {
  EnableDma();
  PublishDma();
  dma_result_ = QNN_SUCCESS;
  Qnn_ContextHandle_t first = nullptr, second = nullptr;
  ASSERT_TRUE(Load(true, first).IsOK());
  ASSERT_TRUE(Load(true, second).IsOK());
  EXPECT_EQ(register_calls_, 1u);
  EXPECT_EQ(ReleaseMapping(0), QNN_SUCCESS);
  EXPECT_EQ(deregister_calls_, 0u);
  EXPECT_EQ(registrations_.size(), 1u);
  EXPECT_EQ(ReleaseMapping(1), QNN_SUCCESS);
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_TRUE(registrations_.empty());
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, FailedDeregistrationIsRetriedDuringDestruction) {
  EnableDma();
  PublishDma();
  fail_deregister_ = true;
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_EQ(registrations_.size(), 1u);
  fail_deregister_ = false;
  backend_.reset();
  manager_ = nullptr;
  EXPECT_EQ(deregister_calls_, 2u);
  EXPECT_TRUE(registrations_.empty());
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, FailedDmaListKeepsPreviousSuccessfulRegistration) {
  EnableDma();
  PublishDma();
  dma_result_ = QNN_SUCCESS;
  Qnn_ContextHandle_t previous = nullptr;
  ASSERT_TRUE(Load(true, previous).IsOK());
  ASSERT_TRUE(manager_->AddQnnContextHandle(previous).IsOK());
  auto* previous_range = mappings_.at(0).response.dmaBuffer.data;
  PrepareInitialLoad();
  EnableListFailure(false);
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
  binaries.emplace(path_.string(), std::make_unique<std::vector<std::string>>(1, "failed_list"));
  ASSERT_TRUE(manager_->SetupBackend(true, true, true, 1, true, rpcmem_, binaries, qnn::EpContextIoDispatch{}).IsOK());
  EXPECT_EQ(list_dma_calls_, 1u);
  EXPECT_EQ(list_ordinary_calls_, 1u);
  EXPECT_EQ(deregister_calls_, 1u);
  ASSERT_EQ(registrations_.size(), 1u);
  EXPECT_EQ(registrations_.count(previous_range), 1u);
  EXPECT_EQ(ReleaseMapping(0), QNN_SUCCESS);
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, PartiallyFailedListKeepsSuccessfulContextRegistration) {
  EnableDma();
  PublishDma();
  dma_result_ = QNN_SUCCESS;
  Qnn_ContextHandle_t previous = nullptr;
  ASSERT_TRUE(Load(true, previous).IsOK());
  ASSERT_TRUE(manager_->AddQnnContextHandle(previous).IsOK());
  auto* previous_range = mappings_.at(0).response.dmaBuffer.data;
  PrepareInitialLoad();
  auto second_path = std::filesystem::path(path_.wstring() + L"_second");
  extra_paths_.push_back(second_path);
  std::filesystem::copy_file(path_, second_path);
  EnableListFailure(true);
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
  binaries.emplace(path_.string(), std::make_unique<std::vector<std::string>>(1, "list_one"));
  binaries.emplace(second_path.string(), std::make_unique<std::vector<std::string>>(1, "list_two"));
  ASSERT_TRUE(manager_->SetupBackend(true, true, true, 1, true, rpcmem_, binaries, qnn::EpContextIoDispatch{}).IsOK());
  EXPECT_EQ(list_dma_calls_, 1u);
  EXPECT_EQ(list_ordinary_calls_, 1u);
  ASSERT_EQ(mappings_.size(), 3u);
  ASSERT_EQ(registrations_.size(), 2u);
  EXPECT_EQ(registrations_.count(previous_range), 1u);
  EXPECT_EQ(registrations_.count(mappings_[1].response.dmaBuffer.data), 1u);
  EXPECT_EQ(registrations_.count(mappings_[2].response.dmaBuffer.data), 0u);
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_EQ(ReleaseMapping(0), QNN_SUCCESS);
  EXPECT_EQ(ReleaseMapping(1), QNN_SUCCESS);
}

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION
TEST_F(QnnUnit_ContextBinaryLoadingTest, CallbackThenNativePublishesDmaOnSharedManager) {
  CheckSharingOrder(true);
}
TEST_F(QnnUnit_ContextBinaryLoadingTest, NativeThenCallbackKeepsDmaOnSharedManager) {
  CheckSharingOrder(false);
}
#endif

TEST_F(QnnUnit_ContextBinaryLoadingTest, InitialCpuMappingOutlivesSdkMetadataAndSystemContext) {
  PrepareInitialLoad();
  track_lifetime_ = true;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> models;
  ASSERT_TRUE(LoadNative(models).IsOK());
  EXPECT_FALSE(mapping_alive_);
  EXPECT_EQ(models.at("native")->Name(), "native_graph");
  EXPECT_EQ(events_, (std::vector<std::string>{"map", "system_create", "metadata", "ordinary", "graph", "system_free", "unmap"}));
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, InitialCpuMappingIsReleasedAfterSdkLoadingFailure) {
  PrepareInitialLoad();
  track_lifetime_ = true;
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> models;
  EXPECT_FALSE(LoadNative(models).IsOK());
  EXPECT_TRUE(models.empty());
  EXPECT_FALSE(mapping_alive_);
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(events_, (std::vector<std::string>{"map", "system_create", "metadata", "ordinary", "system_free", "unmap"}));
}

TEST_F(QnnUnit_ContextBinaryLoadingTest, InitialCpuMappingIsReleasedAfterMetadataFailure) {
  PrepareInitialLoad();
  track_lifetime_ = true;
  metadata_result_ = QNN_COMMON_ERROR_GENERAL;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> models;
  EXPECT_FALSE(LoadNative(models).IsOK());
  EXPECT_TRUE(models.empty());
  EXPECT_FALSE(mapping_alive_);
  EXPECT_EQ(ordinary_calls_, 0u);
  EXPECT_EQ(events_, (std::vector<std::string>{"map", "system_create", "metadata", "system_free", "unmap"}));
}
#endif

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
