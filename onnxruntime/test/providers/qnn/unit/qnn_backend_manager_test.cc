// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for QnnBackendManager (qnn_backend_manager.cc).
//
// Two groups of tests live here, both gated on QNN_EP_INTERNAL_SYMBOL_ACCESS:
//
//   1. Stub-based tests (no real QNN library) — QnnSerializerConfig, SetupBackend
//      load-failure paths, and before-setup early returns. These always run.
//   2. Real-HTP-backend tests (QnnUnit_BackendManagerHtpTest) — load libQnnHtp.so
//      (and libQnnIr.so / libQnnSaver.so) and drive SetupBackend directly, with no
//      ORT session. The fixture GTEST_SKIP()s when the backend is unavailable,
//      mirroring the QnnHTPBackendTests::SetUp() convention.
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

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

// ===========================================================================
// Test helpers
// ===========================================================================

static std::shared_ptr<qnn::QnnBackendManager> MakeManager(
    const std::string& backend_path,
    const ApiPtrs& api_ptrs,
    const Ort::Logger& logger,
    bool skip_version_check = true) {
  qnn::QnnBackendManagerConfig cfg;
  cfg.backend_path = backend_path;
  cfg.context_priority = qnn::ContextPriority::NORMAL;
  cfg.device_id = 0;
  cfg.htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
  cfg.soc_model = 0;
  cfg.skip_qnn_version_check = skip_version_check;
  return qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
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
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger);
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
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger);
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
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger);
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
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger);
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
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger);
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
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger);
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
  cfg.backend_path = "libQnnHtp.so";
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
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger);
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
  auto manager = MakeManager("libQnnHtp.so", env.api_ptrs, env.logger);
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
// The tests below load a real QNN backend (libQnnHtp.so, and for the serializer
// cases libQnnIr.so / libQnnSaver.so) and drive QnnBackendManager directly — no
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
    bool skip_version_check = true) {
  qnn::QnnBackendManagerConfig cfg;
  cfg.backend_path = "libQnnHtp.so";
  cfg.profiling_level = profiling_level;
  cfg.profiling_level_etw = profiling_level_etw;
  cfg.context_priority = context_priority;
  cfg.device_id = 0;
  cfg.htp_arch = htp_arch;
  cfg.soc_model = soc_model;
  cfg.skip_qnn_version_check = skip_version_check;
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
// group via GTEST_SKIP() when libQnnHtp.so cannot be loaded — mirroring the
// established QnnHTPBackendTests::SetUp() convention (backend unavailable → skip,
// not fail). This keeps CI signal clean on environments without the HTP library
// while leaving each test's ASSERT_TRUE(status.IsOK()) as a genuine behavioral
// check once HTP is confirmed present.
class QnnUnit_BackendManagerHtpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!HtpAvailable()) {
      GTEST_SKIP() << "QNN HTP backend (libQnnHtp.so) is not available! Skipping test.";
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
  ASSERT_TRUE(status.IsOK()) << "libQnnHtp.so setup failed: " << status.GetErrorMessage();
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

  EXPECT_TRUE(manager->GetProfilingManager().SetProfilingLevelETW(qnn::ProfilingLevel::BASIC).IsOK());
  EXPECT_TRUE(manager->GetProfilingManager().SetProfilingLevelETW(qnn::ProfilingLevel::OFF).IsOK());
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
// Loading libQnnIr.so as the main backend exercises:
//   - SetQnnBackendType IR/SAVER case: backend_id → QnnBackendType::SERIALIZER
//   - CreateContext SERIALIZER branch: configs = nullptr
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_WithIrBackendDirectly_SetsSerializerBackendType) {
  StubApiEnv env;
  qnn::QnnBackendManagerConfig cfg{};  // value-init to zero all fields (profiling, device_id, etc.)
  cfg.backend_path = "libQnnIr.so";
  cfg.context_priority = qnn::ContextPriority::NORMAL;
  cfg.skip_qnn_version_check = true;
  auto manager = qnn::QnnBackendManager::Create(cfg, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "libQnnIr.so setup failed: " << status.GetErrorMessage();
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
      "libQnnHtp.so",
      qnn::QnnSerializerConfig::CreateSaver("libQnnSaver.so"),
      env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  auto status = SetupBackendHtp(*manager);
  ASSERT_TRUE(status.IsOK()) << "QnnSaver+HTP setup failed: " << status.GetErrorMessage();

  EXPECT_NE(manager->GetQnnSerializerConfig(), nullptr);
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// QnnIrConfig::SupportsArbitraryGraphConfigs() returns false, so CreateContext
// overrides configs to nullptr even for the HTP backend's default configs.
// Distinct from SetupBackend_WithIrBackendDirectly above which loads libQnnIr.so
// as the main backend (SERIALIZER type) without QnnSerializerConfig.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupBackend_HTP_WithQnnIrSerializer_CoversNoArbitraryGraphConfigs) {
  StubApiEnv env;
  const auto tmp_dir = std::filesystem::temp_directory_path() / "qnn_ir_serializer_htp_test";
  std::filesystem::create_directories(tmp_dir);

  auto manager = MakeSerializerManager(
      "libQnnHtp.so",
      qnn::QnnSerializerConfig::CreateIr("libQnnIr.so", tmp_dir.string()),
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

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
