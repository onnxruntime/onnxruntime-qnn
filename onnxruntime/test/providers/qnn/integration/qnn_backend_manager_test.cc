// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Pipeline integration tests for QnnBackendManager (qnn_backend_manager.cc).
//
// These tests require real QNN shared libraries (libQnnHtp.so, libQnnIr.so, etc.).
// If a required library cannot be loaded the test fails with a diagnostic
// message rather than skipping silently — missing libraries are treated as
// CI environment misconfiguration, not "feature not available".
//
// Tests use the HTP emulator, which is functional on Linux x86_64.
//
// Component-level tests (no real .so required) live in unit/qnn_backend_manager_test.cc.
//
// Coverage targets:
//   - SetupBackend with HTP backend (various context priority and device configs)
//   - SetupBackend with IR backend loaded directly (SERIALIZER backend type path)
//   - SetContextPriority / ResetContextPriority
//   - ResetQnnLogLevel after setup
//   - InitializeProfiling (BASIC / DETAILED / ETW merge)
//   - SetProfilingLevelETW
//   - GetContextBinaryBuffer after setup
//   - LoadCachedQnnContextFromBuffer with invalid buffer
//   - SetupBackend called twice (idempotency)
//   - Version check path (skip_version_check=false)

// __linux__ guard: integration tests here dlopen libQnnHtp.so / libQnnIr.so
// directly, which is only exercised on Linux. Windows equivalents would use
// LoadLibrary and different .dll names; when needed they can be added in a
// separate #elif block.
// QNN_EP_INTERNAL_SYMBOL_ACCESS: required to include qnn_backend_manager.h
// (EP-internal header) and construct QnnBackendManager directly.
#if !defined(ORT_MINIMAL_BUILD) && defined(__linux__) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/ort_api.h"

// Cross-tier include: integration tests share the StubApiEnv + MakeNullLogger
// helpers with unit tests so both tiers build a QnnBackendManager the same way.
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

// ===========================================================================
// Test helpers
// ===========================================================================

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
static Ort::Status SetupBackend(qnn::QnnBackendManager& manager) {
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  return manager.SetupBackend(false, false, false, -1, false, nullptr, dummy_map);
}

// ===========================================================================
// Group 1: HTP backend — basic setup and backend type
// ===========================================================================

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "libQnnHtp.so setup failed (CI environment): " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// Second SetupBackend call on the same manager returns OK immediately (no-op).
TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_CalledTwice_SecondCallIsNoOp) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackend(*manager);
    ASSERT_TRUE(s.IsOK()) << "SetupBackend failed: " << s.GetErrorMessage();
  }
  EXPECT_TRUE(SetupBackend(*manager).IsOK());
}

// skip_version_check=false exercises the GetQnnInterfaceProvider version-check loop.
TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithVersionCheck_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::OFF, qnn::ProfilingLevel::OFF,
                                QNN_HTP_DEVICE_ARCH_NONE, /*skip_version_check=*/false);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// ===========================================================================
// Group 2: HTP backend — context priority configs
// ===========================================================================

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithLowPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::LOW);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithNormalHighPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::NORMAL_HIGH);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithHighPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::HIGH);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// SetContextPriority after HTP setup: covers SetContextPriority body and
// calls SetQnnContextConfig for LOW and NORMAL_HIGH.
TEST(QnnInteg_BackendManagerTest, SetContextPriority_HTP_ChangesLevel) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackend(*manager);
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

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithNormalLowPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::NORMAL_LOW);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithHighPlusPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::HIGH_PLUS);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithCriticalPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::CRITICAL);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithCriticalPlusPriority_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::CRITICAL_PLUS);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// UNDEFINED priority is invalid: SetQnnContextConfig returns MAKE_EP_FAIL before
// contextCreate is called, so SetupBackend must return an error.
TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithUndefinedPriority_ReturnsError) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger, qnn::ContextPriority::UNDEFINED);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_FALSE(status.IsOK()) << "SetupBackend unexpectedly succeeded with UNDEFINED priority";
  ASSERT_NE(std::string(status.GetErrorMessage()).find("Invalid Qnn context priority"),
            std::string::npos)
      << "Expected 'Invalid Qnn context priority' error; got: " << status.GetErrorMessage();
}

// ===========================================================================
// Group 3: HTP backend — device configs (SoC model, HTP arch)
// ===========================================================================

// Uses the SM8550 (Snapdragon 8 Gen 2) SoC model to exercise the HTP SoC
// model config block; the HTP emulator accepts arbitrary SoC model values.
TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithSocModel_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL,
                                /*soc_model=*/QNN_SOC_MODEL_SM8550);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// HTP emulator accepts arbitrary arch values.
TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithHtpArch_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::OFF, qnn::ProfilingLevel::OFF,
                                QNN_HTP_DEVICE_ARCH_V73);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// ===========================================================================
// Group 4: HTP backend — profiling and log level
// ===========================================================================

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithBasicProfiling_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::BASIC);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithDetailedProfiling_Succeeds) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::DETAILED);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// profiling_level_etw > profiling_level → InitializeProfiling uses merged level.
TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_EtwLevelHigherThanMain_UsesMergedLevel) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                /*profiling_level=*/qnn::ProfilingLevel::BASIC,
                                /*profiling_level_etw=*/qnn::ProfilingLevel::DETAILED);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "SetupBackend failed: " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// SetProfilingLevelETW releases and re-creates the profile handle.
TEST(QnnInteg_BackendManagerTest, SetProfilingLevelETW_HTP_ChangesLevel) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger,
                                qnn::ContextPriority::NORMAL, 0,
                                qnn::ProfilingLevel::BASIC);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackend(*manager);
    ASSERT_TRUE(s.IsOK()) << "SetupBackend failed: " << s.GetErrorMessage();
  }

  EXPECT_TRUE(manager->SetProfilingLevelETW(qnn::ProfilingLevel::BASIC).IsOK());
  EXPECT_TRUE(manager->SetProfilingLevelETW(qnn::ProfilingLevel::OFF).IsOK());
}

// After SetupBackend each ORT log level maps to a different QNN log level.
TEST(QnnInteg_BackendManagerTest, ResetQnnLogLevel_HTP_AfterSetup_VariousLevels_Succeed) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackend(*manager);
    ASSERT_TRUE(s.IsOK()) << "SetupBackend failed: " << s.GetErrorMessage();
  }

  for (auto level : {ORT_LOGGING_LEVEL_VERBOSE, ORT_LOGGING_LEVEL_INFO,
                     ORT_LOGGING_LEVEL_WARNING, ORT_LOGGING_LEVEL_ERROR}) {
    EXPECT_TRUE(manager->ResetQnnLogLevel(level).IsOK()) << "level=" << level;
  }
  EXPECT_TRUE(manager->ResetQnnLogLevel(std::nullopt).IsOK());
}

// ===========================================================================
// Group 5: HTP backend — context binary buffer
// ===========================================================================

// After SetupBackend the HTP context can be serialized to a non-empty buffer.
TEST(QnnInteg_BackendManagerTest, GetContextBinaryBuffer_HTP_AfterSetup_ReturnsValidBuffer) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  {
    auto s = SetupBackend(*manager);
    ASSERT_TRUE(s.IsOK()) << "SetupBackend failed: " << s.GetErrorMessage();
  }

  uint64_t written_size = 0;
  auto buffer = manager->GetContextBinaryBuffer(written_size);
  EXPECT_NE(buffer, nullptr);
  EXPECT_GT(written_size, 0u);
}

// Garbage bytes are rejected without crashing.
TEST(QnnInteg_BackendManagerTest, LoadCachedQnnContextFromBuffer_HTP_InvalidBuffer_ReturnsError) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  auto setup_status = manager->SetupBackend(true, true, false, -1, false, nullptr, dummy_map);
  ASSERT_TRUE(setup_status.IsOK()) << "SetupBackend with QnnSystem failed (CI environment): " << setup_status.GetErrorMessage();

  char garbage[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                      0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> qnn_models;
  auto status = manager->LoadCachedQnnContextFromBuffer(
      garbage, sizeof(garbage), "", "test_node", qnn_models, 0);
  EXPECT_FALSE(status.IsOK());
}

// ===========================================================================
// Group 6: IR backend loaded directly (no QnnSerializerConfig)
//
// Loading libQnnIr.so as the main backend exercises:
//   - SetQnnBackendType IR/SAVER case: backend_id → QnnBackendType::SERIALIZER
//   - CreateContext SERIALIZER branch: configs = nullptr
// ===========================================================================

TEST(QnnInteg_BackendManagerTest, SetupBackend_WithIrBackendDirectly_SetsSerializerBackendType) {
  StubApiEnv env;
  qnn::QnnBackendManagerConfig cfg{};  // value-init to zero all fields (profiling, device_id, etc.)
  cfg.backend_path = "libQnnIr.so";
  cfg.context_priority = qnn::ContextPriority::NORMAL;
  cfg.skip_qnn_version_check = true;
  auto manager = qnn::QnnBackendManager::Create(cfg, env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);
  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "libQnnIr.so setup failed (CI environment): " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::SERIALIZER);
}

// ===========================================================================
// Group 7: Serializer backends (Saver / Ir) with HTP as validator
//
// LoadQnnSerializerBackend loads both the validator backend (HTP) and the
// serializer backend (Saver or Ir). The serializer config determines which
// configs are passed to contextCreate:
//   - QnnSaverConfig::SupportsArbitraryGraphConfigs() == true  → HTP configs used
//   - QnnIrConfig::SupportsArbitraryGraphConfigs()    == false → configs nullified
//                                                                (line 1442 path)
// ===========================================================================

// QnnSaver records all QNN API calls. With HTP as the validator backend,
// SetupBackend exercises LoadQnnSerializerBackend (loads both .so libraries and
// logs their versions).
TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithQnnSaverSerializer_Succeeds) {
  StubApiEnv env;
  auto manager = MakeSerializerManager(
      "libQnnHtp.so",
      qnn::QnnSerializerConfig::CreateSaver("libQnnSaver.so"),
      env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "QnnSaver+HTP setup failed (CI environment): " << status.GetErrorMessage();

  EXPECT_NE(manager->GetQnnSerializerConfig(), nullptr);
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// QnnIrConfig::SupportsArbitraryGraphConfigs() returns false, so CreateContext
// overrides configs to nullptr (line 1442) even for the HTP backend's default
// configs. Distinct from SetupBackend_WithIrBackendDirectly above which loads
// libQnnIr.so as the main backend (SERIALIZER type) without QnnSerializerConfig.
TEST(QnnInteg_BackendManagerTest, SetupBackend_HTP_WithQnnIrSerializer_CoversNoArbitraryGraphConfigs) {
  StubApiEnv env;
  const auto tmp_dir = std::filesystem::temp_directory_path() / "qnn_ir_serializer_htp_test";
  std::filesystem::create_directories(tmp_dir);

  auto manager = MakeSerializerManager(
      "libQnnHtp.so",
      qnn::QnnSerializerConfig::CreateIr("libQnnIr.so", tmp_dir.string()),
      env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  auto status = SetupBackend(*manager);
  ASSERT_TRUE(status.IsOK()) << "QnnIr+HTP setup failed (CI environment): " << status.GetErrorMessage();
  EXPECT_EQ(manager->GetQnnBackendType(), qnn::QnnBackendType::HTP);

  std::filesystem::remove_all(tmp_dir);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && defined(__linux__) && QNN_EP_INTERNAL_SYMBOL_ACCESS
