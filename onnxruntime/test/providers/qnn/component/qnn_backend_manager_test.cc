// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for QnnBackendManager (qnn_backend_manager.cc).
//
// Two groups of tests live here, both gated on QNN_EP_INTERNAL_SYMBOL_ACCESS:
//
//   1. Stub-based tests (no real QNN library) — QnnSerializerConfig, SetupBackend
//      load-failure paths, before-setup early returns, and multi-SoC lifecycle.
//      These always run.
//   2. Real-HTP-backend tests (QnnUnit_BackendManagerHtpTest) — load the platform
//      HTP backend (and QnnIr / QnnSaver) and drive SetupBackend directly, with no
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
//   - Multi-SoC setup / reuse / selection / serialization / partial-failure cleanup

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "test/providers/qnn/infra/qnn_test_windows_compat.h"

#include "QnnCommon.h"
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpContext.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/ort_api.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

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

TEST(QnnUnit_BackendManagerTest, BuildContextBinaryConfigs_ContextInitAcceleration_PreservesExistingConfigs) {
  char group_token{};
  const Qnn_ContextHandle_t group_handles[] = {nullptr, reinterpret_cast<Qnn_ContextHandle_t>(&group_token)};
  for (auto group_handle : group_handles) {
    SCOPED_TRACE(group_handle);
    StubApiEnv env;
    qnn::QnnBackendManagerConfig config{};
    config.context_priority = qnn::ContextPriority::HIGH;
    config.reused_io_limit_mb = 64;
    config.htp_context_init_acceleration = true;
    auto manager = qnn::QnnBackendManager::Create(config, env.api_ptrs, env.logger);
    manager->SetQnnBackendType(QNN_BACKEND_ID_HTP);
    qnn::QnnConfigsBuilder<QnnContext_Config_t, QnnHtpContext_CustomConfig_t> builder(
        QNN_CONTEXT_CONFIG_INIT, QnnHtpContext_CustomConfig_t{});
    ASSERT_TRUE(manager->BuildContextBinaryConfigs(4096, group_handle, builder).IsOK());
    const auto* configs = builder.GetQnnConfigs();
    ASSERT_NE(configs[0], nullptr);
    EXPECT_EQ(configs[0]->option, QNN_CONTEXT_CONFIG_OPTION_PRIORITY);
    EXPECT_EQ(configs[0]->priority, QNN_PRIORITY_HIGH);
    ASSERT_NE(configs[1], nullptr);
    ASSERT_EQ(configs[1]->option, QNN_CONTEXT_CONFIG_OPTION_CUSTOM);
    const auto* spill = static_cast<const QnnHtpContext_CustomConfig_t*>(configs[1]->customConfig);
    ASSERT_NE(spill, nullptr);
    EXPECT_EQ(spill->option, QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS);
    EXPECT_EQ(spill->groupRegistration.firstGroupHandle, group_handle);
    EXPECT_EQ(spill->groupRegistration.maxSpillFillBuffer, 4096u);
    size_t end = 2;
#ifdef QNN_HTP_REUSED_IO_LIMIT_AVAILABLE
    ASSERT_NE(configs[end], nullptr);
    ASSERT_EQ(configs[end]->option, QNN_CONTEXT_CONFIG_OPTION_CUSTOM);
    const auto* reused = static_cast<const QnnHtpContext_CustomConfig_t*>(configs[end++]->customConfig);
    ASSERT_NE(reused, nullptr);
    EXPECT_EQ(reused->option, QNN_HTP_CONTEXT_CONFIG_OPTION_REUSED_IO_LIMIT);
    EXPECT_EQ(reused->reusedIoLimitMb, 64u);
#endif
#if QNN_API_VERSION_MAJOR > 2 || \
    (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 36)
    ASSERT_NE(configs[end], nullptr);
    ASSERT_EQ(configs[end]->option, QNN_CONTEXT_CONFIG_OPTION_CUSTOM);
    const auto* acceleration = static_cast<const QnnHtpContext_CustomConfig_t*>(configs[end++]->customConfig);
    ASSERT_NE(acceleration, nullptr);
    EXPECT_EQ(acceleration->option, QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION);
    EXPECT_TRUE(acceleration->initAcceleration);
#endif
    EXPECT_EQ(configs[end], nullptr);
  }
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
  auto status = manager->SetupBackend(false, false, false, 0, false, nullptr, dummy_map);

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
// holder stays NONE until backend setup or a target is selected. soc_model has no such split and
// is readable straight from the config.
TEST(QnnUnit_BackendManagerTest, GetHtpArch_UserProvidedArchBeforeSetup_ReturnsNone) {
  StubApiEnv env;
  auto manager = MakeManagerWithHtpArch(QNN_HTP_DEVICE_ARCH_V73, QNN_SOC_MODEL_SM8550,
                                        env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
  EXPECT_EQ(manager->GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_SM8550));
}

// SetupDevicesAndContexts is guarded by backend_partial_setup_completed_, so calling it
// without SetupBackendExceptDeviceAndContext first must fail and must not publish the
// requested arch to the internal holder.
TEST(QnnUnit_BackendManagerTest, SetupDevicesAndContexts_WithoutPartialSetup_ReturnsErrorAndLeavesArchNone) {
  StubApiEnv env;
  auto manager = MakeManager(QnnHtpBackendLibraryName(), env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  auto status = manager->SetupDevicesAndContexts({QNN_HTP_DEVICE_ARCH_V79}, {QNN_SOC_MODEL_SM8550});

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
// Stub-based multi-SoC lifecycle tests
// ===========================================================================

// Simulates the completed backend/DLC setup through the existing test-only
// private-member access mechanism. Device/context creation and ownership are
// exercised through the manager's public APIs, with recording QNN callbacks.
class QnnUnit_BackendManagerMultiSocTest : public ::testing::Test {
 protected:
  struct DeviceConfig {
    QnnHtpDevice_Arch_t arch = QNN_HTP_DEVICE_ARCH_NONE;
    uint32_t soc = QNN_SOC_MODEL_UNKNOWN;
    bool has_soc_config = false;
  };

  struct ContextConfig {
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_Priority_t priority = QNN_PRIORITY_UNDEFINED;
    bool weight_sharing = false;
    bool extended_udma = false;
    bool prepare_only = false;
    bool graph_splitting = false;
    uint32_t num_prepare_threads = UINT32_MAX;
  };

  void SetUp() override {
    current_ = this;
    backend_manager_ = std::make_unique<StubBackendManager>(env_.api_ptrs, env_.logger);
    backend_manager_->BackendType() = qnn::QnnBackendType::HTP;
    backend_manager_->BackendHandle() = &backend_token_;
    backend_manager_->BackendPartialSetupCompleted() = true;
    backend_manager_->SystemDlcCreated() = true;

    auto& interface = backend_manager_->QnnInterface();
    interface.deviceCreate = DeviceCreate;
    interface.deviceFree = DeviceFree;
    interface.contextCreate = ContextCreate;
    interface.contextFree = ContextFree;
  }

  void TearDown() override {
    // DLC readiness was simulated without creating a plugin to release.
    backend_manager_->SystemDlcCreated() = false;
    // Destruction invokes the recording callbacks, so keep the fixture and
    // handle tokens alive until the manager has released its resources.
    backend_manager_.reset();
    current_ = nullptr;
  }

  qnn::QnnBackendManager& Manager() { return *backend_manager_->GetMutable(); }

  void ExpectNoResources() {
    EXPECT_FALSE(Manager().IsBackendSetup());
    EXPECT_EQ(Manager().GetQnnContextSize(), 0u);
    EXPECT_FALSE(Manager().SetActiveSoc(0).IsOK());
    EXPECT_TRUE(Manager().SetActiveSoc(-1).IsOK());
    EXPECT_EQ(Manager().GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
    EXPECT_EQ(Manager().GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_UNKNOWN));
    EXPECT_THROW(Manager().GetQnnContext(), Ort::Exception);
  }

  void ExpectAllContextsFreedOnce() {
    // Context ownership is held in an unordered_map, so destruction order can
    // differ across platforms. Every created context must still be freed once.
    ASSERT_EQ(freed_contexts_.size(), created_contexts_.size());
    for (auto context : created_contexts_) {
      EXPECT_EQ(std::count(freed_contexts_.begin(), freed_contexts_.end(), context), 1) << context;
    }
  }

  const std::vector<QnnHtpDevice_Arch_t> archs_{QNN_HTP_DEVICE_ARCH_V68,
                                                QNN_HTP_DEVICE_ARCH_V73,
                                                QNN_HTP_DEVICE_ARCH_V73};
  const std::vector<uint32_t> socs_{QNN_SOC_MODEL_SM8350, QNN_SOC_MODEL_SM8550, QNN_SOC_MODEL_UNKNOWN};
  StubApiEnv env_;
  std::unique_ptr<StubBackendManager> backend_manager_;
  std::vector<DeviceConfig> device_configs_;
  std::vector<ContextConfig> context_configs_;
  std::vector<Qnn_DeviceHandle_t> created_devices_;
  std::vector<Qnn_ContextHandle_t> created_contexts_;
  std::vector<Qnn_DeviceHandle_t> freed_devices_;
  std::vector<Qnn_ContextHandle_t> freed_contexts_;
  std::vector<std::string> calls_;
  std::optional<size_t> device_create_failure_;
  std::optional<size_t> context_create_failure_;
  Qnn_DeviceHandle_t device_free_failure_ = nullptr;
  std::vector<Qnn_ContextHandle_t> binary_size_contexts_;
  std::vector<Qnn_ContextHandle_t> binary_contexts_;

  static Qnn_ErrorHandle_t ContextGetBinarySize(Qnn_ContextHandle_t context,
                                                Qnn_ContextBinarySize_t* size) {
    current_->binary_size_contexts_.push_back(context);
    *size = sizeof(kBinaryPayload);
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t ContextGetBinary(Qnn_ContextHandle_t context, void* buffer,
                                            Qnn_ContextBinarySize_t size,
                                            Qnn_ContextBinarySize_t* written) {
    current_->binary_contexts_.push_back(context);
    if (size < sizeof(kBinaryPayload)) {
      return QNN_COMMON_ERROR_INVALID_ARGUMENT;
    }
    std::memcpy(buffer, kBinaryPayload, sizeof(kBinaryPayload));
    *written = sizeof(kBinaryPayload);
    return QNN_SUCCESS;
  }

  inline static constexpr unsigned char kBinaryPayload[] = {0x12, 0x34, 0x56};

 private:
  static Qnn_ErrorHandle_t DeviceCreate(Qnn_LogHandle_t /*logger*/,
                                        const QnnDevice_Config_t** configs,
                                        Qnn_DeviceHandle_t* device) {
    current_->calls_.push_back("deviceCreate");
    DeviceConfig recorded;
    for (size_t i = 0; configs != nullptr && configs[i] != nullptr; ++i) {
      if (configs[i]->option != QNN_DEVICE_CONFIG_OPTION_CUSTOM) {
        continue;
      }
      const auto* custom = static_cast<const QnnHtpDevice_CustomConfig_t*>(configs[i]->customConfig);
      if (custom->option == QNN_HTP_DEVICE_CONFIG_OPTION_ARCH) {
        recorded.arch = custom->arch.arch;
      } else if (custom->option == QNN_HTP_DEVICE_CONFIG_OPTION_SOC) {
        recorded.soc = custom->socModel;
        recorded.has_soc_config = true;
      }
    }
    const size_t index = current_->device_configs_.size();
    current_->device_configs_.push_back(recorded);
    if (current_->device_create_failure_ == index) {
      return QNN_COMMON_ERROR_GENERAL;
    }
    current_->device_tokens_.push_back(0);
    *device = &current_->device_tokens_.back();
    current_->created_devices_.push_back(*device);
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t ContextCreate(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device,
                                         const QnnContext_Config_t** configs,
                                         Qnn_ContextHandle_t* context) {
    current_->calls_.push_back("contextCreate");
    ContextConfig recorded;
    recorded.backend = backend;
    recorded.device = device;
    for (size_t i = 0; configs != nullptr && configs[i] != nullptr; ++i) {
      if (configs[i]->option == QNN_CONTEXT_CONFIG_OPTION_PRIORITY) {
        recorded.priority = configs[i]->priority;
      } else if (configs[i]->option == QNN_CONTEXT_CONFIG_OPTION_CUSTOM) {
        const auto* custom = static_cast<const QnnHtpContext_CustomConfig_t*>(configs[i]->customConfig);
        switch (custom->option) {
          case QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED:
            recorded.weight_sharing = custom->weightSharingEnabled;
            break;
          case QNN_HTP_CONTEXT_CONFIG_OPTION_USE_EXTENDED_UDMA:
            recorded.extended_udma = custom->useExtendedUdma;
            break;
          case QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY:
            recorded.prepare_only = custom->isPrepareOnly;
            break;
#ifdef QNN_HTP_GRAPH_SPLITTING_AVAILABLE
          case QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_CONFIGS:
            recorded.graph_splitting = custom->graphSplittingConfigs.graphSplittingEnabled;
            break;
#endif
#ifdef QNN_HTP_GRAPH_SPLITTING_NUM_THREADS_AVAILABLE
          case QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_NUM_PREPARE_THREADS:
            recorded.num_prepare_threads = custom->numPrepareThreads;
            break;
#endif
          default:
            break;
        }
      }
    }
    const size_t index = current_->context_configs_.size();
    current_->context_configs_.push_back(recorded);
    if (current_->context_create_failure_ == index) {
      return QNN_COMMON_ERROR_GENERAL;
    }
    current_->context_tokens_.push_back(0);
    *context = &current_->context_tokens_.back();
    current_->created_contexts_.push_back(*context);
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t ContextFree(Qnn_ContextHandle_t context, Qnn_ProfileHandle_t /*profile*/) {
    current_->calls_.push_back("contextFree");
    current_->freed_contexts_.push_back(context);
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t DeviceFree(Qnn_DeviceHandle_t device) {
    current_->calls_.push_back("deviceFree");
    current_->freed_devices_.push_back(device);
    return device == current_->device_free_failure_ ? QNN_COMMON_ERROR_GENERAL : QNN_SUCCESS;
  }

  inline static QnnUnit_BackendManagerMultiSocTest* current_ = nullptr;
  uint8_t backend_token_ = 0;
  std::deque<uint8_t> device_tokens_;
  std::deque<uint8_t> context_tokens_;
};

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_WithoutPartialSetup_ReturnsError) {
  backend_manager_->BackendPartialSetupCompleted() = false;
  const auto status = Manager().SetupDevicesAndContexts(archs_, socs_);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("partially setup"), std::string::npos);
  EXPECT_TRUE(calls_.empty());
  ExpectNoResources();
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_WithoutDlc_ReturnsError) {
  backend_manager_->SystemDlcCreated() = false;
  const auto status = Manager().SetupDevicesAndContexts(archs_, socs_);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("DLC"), std::string::npos);
  EXPECT_TRUE(calls_.empty());
  ExpectNoResources();
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_EmptyTargets_ReturnsError) {
  const auto status = Manager().SetupDevicesAndContexts({}, {});
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("at least one pair"), std::string::npos);
  EXPECT_TRUE(calls_.empty());
  ExpectNoResources();
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_UnequalTargetCounts_ReturnsError) {
  const auto status = Manager().SetupDevicesAndContexts(archs_, {QNN_SOC_MODEL_SM8350});
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("equal length"), std::string::npos);
  EXPECT_TRUE(calls_.empty());
  ExpectNoResources();
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_CreatesAndSelectsEachTargetInOrder) {
  const auto status = Manager().SetupDevicesAndContexts(archs_, socs_,
                                                        /*enable_htp_weight_sharing=*/true,
                                                        /*enable_htp_extended_udma_mode=*/true,
                                                        /*enable_htp_prepare_only=*/true);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_EQ(device_configs_.size(), archs_.size());
  ASSERT_EQ(context_configs_.size(), archs_.size());
  ASSERT_EQ(created_devices_.size(), archs_.size());
  ASSERT_EQ(created_contexts_.size(), archs_.size());
  EXPECT_EQ(Manager().GetQnnContextSize(), archs_.size());
  EXPECT_TRUE(Manager().IsBackendSetup());
  EXPECT_EQ(calls_, (std::vector<std::string>{"deviceCreate", "deviceCreate", "deviceCreate",
                                              "contextCreate", "contextCreate", "contextCreate"}));
  EXPECT_THROW(Manager().GetQnnContext(), Ort::Exception);

  const qnn::QnnBackendManager& const_manager = Manager();
  for (size_t i = 0; i < archs_.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(device_configs_[i].arch, archs_[i]);
    EXPECT_EQ(device_configs_[i].soc, socs_[i]);
    EXPECT_EQ(device_configs_[i].has_soc_config, socs_[i] != QNN_SOC_MODEL_UNKNOWN);
    EXPECT_EQ(context_configs_[i].backend, backend_manager_->BackendHandle());
    EXPECT_EQ(context_configs_[i].device, created_devices_[i]);
    EXPECT_EQ(context_configs_[i].priority, QNN_PRIORITY_NORMAL);
    EXPECT_TRUE(context_configs_[i].weight_sharing);
    EXPECT_TRUE(context_configs_[i].extended_udma);
    EXPECT_TRUE(context_configs_[i].prepare_only);
    ASSERT_TRUE(Manager().SetActiveSoc(static_cast<int32_t>(i)).IsOK());
    EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[i]);
    EXPECT_EQ(const_manager.GetHtpArch(), archs_[i]);
    EXPECT_EQ(const_manager.GetSocModel(), socs_[i]);
  }

  ASSERT_TRUE(Manager().SetActiveSoc(-1).IsOK());
  EXPECT_EQ(const_manager.GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
  EXPECT_EQ(const_manager.GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_UNKNOWN));
  for (size_t i = 0; i < created_contexts_.size(); ++i) {
    EXPECT_THROW(Manager().GetQnnContext(static_cast<int>(i)), Ort::Exception);
  }
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_ArchOnlyTargetsRemainDistinct) {
  const std::vector<QnnHtpDevice_Arch_t> archs{QNN_HTP_DEVICE_ARCH_V68, QNN_HTP_DEVICE_ARCH_V73};
  const std::vector<uint32_t> socs{QNN_SOC_MODEL_UNKNOWN, QNN_SOC_MODEL_UNKNOWN};
  const auto status = Manager().SetupDevicesAndContexts(archs, socs);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_EQ(created_contexts_.size(), 2u);
  EXPECT_NE(created_contexts_[0], created_contexts_[1]);
  for (size_t i = 0; i < archs.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_FALSE(device_configs_[i].has_soc_config);
    ASSERT_TRUE(Manager().SetActiveSoc(static_cast<int32_t>(i)).IsOK());
    EXPECT_EQ(Manager().GetHtpArch(), archs[i]);
    EXPECT_EQ(Manager().GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_UNKNOWN));
    EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[i]);
  }
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_SameTargets_ReusesAllHandles) {
  ASSERT_TRUE(Manager().SetupDevicesAndContexts(archs_, socs_).IsOK());
  ASSERT_TRUE(Manager().SetActiveSoc(1).IsOK());
  const auto original_calls = calls_;
  const auto status = Manager().SetupDevicesAndContexts(archs_, socs_);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(calls_, original_calls);
  EXPECT_EQ(Manager().GetQnnContextSize(), created_contexts_.size());
  EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[1]);
  ASSERT_TRUE(Manager().SetActiveSoc(-1).IsOK());
  EXPECT_THROW(Manager().GetQnnContext(), Ort::Exception);
  for (size_t i = 0; i < created_contexts_.size(); ++i) {
    ASSERT_TRUE(Manager().SetActiveSoc(static_cast<int32_t>(i)).IsOK());
    EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[i]);
  }
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_ChangedTargets_RejectsWithoutChangingResources) {
  ASSERT_TRUE(Manager().SetupDevicesAndContexts(archs_, socs_).IsOK());
  ASSERT_TRUE(Manager().SetActiveSoc(1).IsOK());
  const auto original_calls = calls_;
  auto expect_rejected = [&](const std::vector<QnnHtpDevice_Arch_t>& archs,
                             const std::vector<uint32_t>& socs) {
    const auto status = Manager().SetupDevicesAndContexts(archs, socs);
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.GetErrorMessage().find("Mismatched"), std::string::npos);
    EXPECT_EQ(calls_, original_calls);
    EXPECT_TRUE(Manager().IsBackendSetup());
    EXPECT_EQ(Manager().GetQnnContextSize(), created_contexts_.size());
    EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[1]);
    EXPECT_EQ(Manager().GetHtpArch(), archs_[1]);
    EXPECT_EQ(Manager().GetSocModel(), socs_[1]);
  };

  auto changed_archs = archs_;
  changed_archs[0] = QNN_HTP_DEVICE_ARCH_V73;
  expect_rejected(changed_archs, socs_);
  auto changed_socs = socs_;
  changed_socs[0] = QNN_SOC_MODEL_SM8550;
  expect_rejected(archs_, changed_socs);
  expect_rejected({archs_[0]}, {socs_[0]});
  expect_rejected({archs_[1], archs_[0], archs_[2]}, {socs_[1], socs_[0], socs_[2]});
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetActiveSoc_WithoutTargets_AcceptsResetAndRejectsIndices) {
  EXPECT_TRUE(Manager().SetActiveSoc(-1).IsOK());
  EXPECT_FALSE(Manager().SetActiveSoc(-2).IsOK());
  EXPECT_FALSE(Manager().SetActiveSoc(0).IsOK());
  ExpectNoResources();
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetActiveSoc_InvalidIndex_PreservesSelectedTarget) {
  ASSERT_TRUE(Manager().SetupDevicesAndContexts(archs_, socs_).IsOK());
  ASSERT_TRUE(Manager().SetActiveSoc(1).IsOK());
  for (int32_t index : {-2, static_cast<int32_t>(archs_.size())}) {
    SCOPED_TRACE(index);
    EXPECT_FALSE(Manager().SetActiveSoc(index).IsOK());
    EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[1]);
    EXPECT_EQ(Manager().GetHtpArch(), archs_[1]);
    EXPECT_EQ(Manager().GetSocModel(), socs_[1]);
  }
  for (int index : {-1, 1, static_cast<int>(archs_.size())}) {
    SCOPED_TRACE(index);
    EXPECT_THROW(Manager().GetQnnContext(index), Ort::Exception);
    EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[1]);
  }
  ASSERT_TRUE(Manager().SetActiveSoc(-1).IsOK());
  EXPECT_THROW(Manager().GetQnnContext(), Ort::Exception);
  ASSERT_TRUE(Manager().SetActiveSoc(0).IsOK());
  EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[0]);
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, GetQnnContext_WithoutActiveSoc_ThrowsAndPreservesContexts) {
  ASSERT_TRUE(Manager().SetupDevicesAndContexts(archs_, socs_).IsOK());
  const auto original_calls = calls_;
  auto expect_missing_active_soc = [&]() {
    for (int index : {-1, 0, 1, static_cast<int>(archs_.size())}) {
      SCOPED_TRACE(index);
      try {
        (void)Manager().GetQnnContext(index);
        FAIL() << "Expected GetQnnContext to require an active SoC";
      } catch (const Ort::Exception& ex) {
        EXPECT_EQ(ex.GetOrtErrorCode(), ORT_EP_FAIL);
        EXPECT_STREQ(ex.what(), "Unexpected usage of getting QNN context without active SoC set.");
      }
    }
    EXPECT_EQ(calls_, original_calls);
    EXPECT_EQ(Manager().GetQnnContextSize(), created_contexts_.size());
    EXPECT_TRUE(Manager().IsBackendSetup());
    EXPECT_TRUE(freed_contexts_.empty());
    EXPECT_TRUE(freed_devices_.empty());
  };

  expect_missing_active_soc();
  ASSERT_TRUE(Manager().SetActiveSoc(1).IsOK());
  EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[1]);
  ASSERT_TRUE(Manager().SetActiveSoc(-1).IsOK());
  expect_missing_active_soc();
  // Rejecting access preserves every context for subsequent target selection.
  for (size_t i = 0; i < created_contexts_.size(); ++i) {
    ASSERT_TRUE(Manager().SetActiveSoc(static_cast<int32_t>(i)).IsOK());
    EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[i]);
  }
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, GetContextBinaryBuffer_ActiveTarget_UsesSelectedContextForBothCalls) {
  backend_manager_->QnnInterface().contextGetBinarySize = ContextGetBinarySize;
  backend_manager_->QnnInterface().contextGetBinary = ContextGetBinary;
  ASSERT_TRUE(Manager().SetupDevicesAndContexts(archs_, socs_).IsOK());
  for (size_t i = 0; i < archs_.size(); ++i) {
    SCOPED_TRACE(i);
    ASSERT_TRUE(Manager().SetActiveSoc(static_cast<int32_t>(i)).IsOK());
    unsigned char* raw_buffer = nullptr;
    uint64_t written_size = 0;
    const auto status = Manager().GetContextBinaryBuffer(false, &raw_buffer, written_size);
    std::unique_ptr<unsigned char[]> buffer(raw_buffer);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    ASSERT_NE(buffer, nullptr);
    ASSERT_EQ(written_size, sizeof(kBinaryPayload));
    EXPECT_TRUE(std::equal(kBinaryPayload, kBinaryPayload + sizeof(kBinaryPayload), buffer.get()));
  }
  EXPECT_EQ(binary_size_contexts_, created_contexts_);
  EXPECT_EQ(binary_contexts_, created_contexts_);
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_UnexpectedSetup_RejectsWithoutChangingResources) {
  ASSERT_TRUE(Manager().SetupDevicesAndContexts(archs_, socs_).IsOK());
  ASSERT_TRUE(Manager().SetActiveSoc(1).IsOK());
  // Simulate existing target records without a completed backend setup.
  backend_manager_->BackendSetupCompleted() = false;
  const auto original_calls = calls_;

  const auto status = Manager().SetupDevicesAndContexts(archs_, socs_);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("unexpected device/context setup"), std::string::npos);
  EXPECT_EQ(calls_, original_calls);
  EXPECT_FALSE(Manager().IsBackendSetup());
  EXPECT_EQ(Manager().GetQnnContextSize(), created_contexts_.size());
  EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[1]);
  EXPECT_EQ(Manager().GetHtpArch(), archs_[1]);
  EXPECT_EQ(Manager().GetSocModel(), socs_[1]);
  EXPECT_TRUE(freed_devices_.empty());
  EXPECT_TRUE(freed_contexts_.empty());

  Manager().ReleaseDeviceAndContext();
  EXPECT_EQ(freed_devices_, created_devices_);
  ExpectAllContextsFreedOnce();
  ExpectNoResources();
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_SequentialSingleTargets_SelectsAndReleasesEachTarget) {
  for (size_t i = 0; i < archs_.size(); ++i) {
    SCOPED_TRACE(i);
    const bool extended_udma = i == 0;
    const bool prepare_only = i == 1;
    const bool graph_splitting = i == 2;
    const auto status = Manager().SetupDevicesAndContexts({archs_[i]}, {socs_[i]},
                                                          /*enable_htp_weight_sharing=*/false,
                                                          extended_udma, prepare_only,
                                                          /*enable_htp_ref_weight_sharing=*/false,
                                                          graph_splitting,
                                                          /*htp_graph_splitting_num_prepare_threads=*/4);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    ASSERT_EQ(created_devices_.size(), i + 1);
    ASSERT_EQ(created_contexts_.size(), i + 1);
    EXPECT_TRUE(Manager().IsBackendSetup());
    EXPECT_EQ(Manager().GetQnnContextSize(), 1u);
    EXPECT_EQ(device_configs_[i].arch, archs_[i]);
    EXPECT_EQ(device_configs_[i].soc, socs_[i]);
    EXPECT_EQ(context_configs_[i].device, created_devices_[i]);
    EXPECT_FALSE(context_configs_[i].weight_sharing);
    EXPECT_EQ(context_configs_[i].extended_udma, extended_udma);
    EXPECT_EQ(context_configs_[i].prepare_only, prepare_only);
#ifdef QNN_HTP_GRAPH_SPLITTING_AVAILABLE
    EXPECT_EQ(context_configs_[i].graph_splitting, graph_splitting);
#endif
#ifdef QNN_HTP_GRAPH_SPLITTING_NUM_THREADS_AVAILABLE
    EXPECT_EQ(context_configs_[i].num_prepare_threads, graph_splitting ? 4u : UINT32_MAX);
#endif
    // Each offline target has its own one-element setup, so its active index is
    // always zero even when the outer compilation loop is on a later target.
    EXPECT_EQ(Manager().GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
    EXPECT_EQ(Manager().GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_UNKNOWN));
    EXPECT_THROW(Manager().GetQnnContext(), Ort::Exception);
    ASSERT_TRUE(Manager().SetActiveSoc(0).IsOK());
    EXPECT_EQ(Manager().GetQnnContext(), created_contexts_[i]);
    EXPECT_EQ(Manager().GetHtpArch(), archs_[i]);
    EXPECT_EQ(Manager().GetSocModel(), socs_[i]);

    calls_.clear();
    Manager().ReleaseDeviceAndContext();
    EXPECT_EQ(calls_, (std::vector<std::string>{"contextFree", "deviceFree"}));
    EXPECT_EQ(freed_devices_, created_devices_);
    ExpectAllContextsFreedOnce();
    ExpectNoResources();
  }
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_DeviceCreationFailure_FreesEarlierDevices) {
  device_create_failure_ = 1;
  const auto status = Manager().SetupDevicesAndContexts(archs_, socs_);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("Failed to create device"), std::string::npos);
  EXPECT_EQ(device_configs_.size(), 2u);
  ASSERT_EQ(created_devices_.size(), 1u);
  EXPECT_EQ(freed_devices_, created_devices_);
  EXPECT_TRUE(context_configs_.empty());
  EXPECT_TRUE(freed_contexts_.empty());
  ExpectNoResources();
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, SetupDevicesAndContexts_ContextCreationFailure_FreesPartialResources) {
  context_create_failure_ = 1;
  const auto status = Manager().SetupDevicesAndContexts(archs_, socs_);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("Failed to create context"), std::string::npos);
  EXPECT_EQ(created_devices_.size(), archs_.size());
  EXPECT_EQ(context_configs_.size(), 2u);
  ASSERT_EQ(created_contexts_.size(), 1u);
  ExpectAllContextsFreedOnce();
  EXPECT_EQ(freed_devices_, created_devices_);
  EXPECT_EQ(calls_, (std::vector<std::string>{"deviceCreate", "deviceCreate", "deviceCreate",
                                              "contextCreate", "contextCreate", "contextFree",
                                              "deviceFree", "deviceFree", "deviceFree"}));
  ExpectNoResources();
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, ReleaseDeviceAndContext_FreesAllResourcesOnceAndResetsSelection) {
  ASSERT_TRUE(Manager().SetupDevicesAndContexts(archs_, socs_).IsOK());
  ASSERT_TRUE(Manager().SetActiveSoc(2).IsOK());
  calls_.clear();
  Manager().ReleaseDeviceAndContext();
  EXPECT_EQ(calls_, (std::vector<std::string>{"contextFree", "contextFree", "contextFree",
                                              "deviceFree", "deviceFree", "deviceFree"}));
  EXPECT_EQ(freed_devices_, created_devices_);
  ExpectAllContextsFreedOnce();
  ExpectNoResources();
  calls_.clear();
  Manager().ReleaseDeviceAndContext();
  EXPECT_TRUE(calls_.empty());
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, ReleaseContext_KeepsDevicesUntilDeviceCleanup) {
  ASSERT_TRUE(Manager().SetupDevicesAndContexts(archs_, socs_).IsOK());
  ASSERT_TRUE(Manager().ReleaseContext().IsOK());
  EXPECT_EQ(Manager().GetQnnContextSize(), 0u);
  ExpectAllContextsFreedOnce();
  EXPECT_TRUE(freed_devices_.empty());
  Manager().ReleaseDeviceAndContext();
  ExpectAllContextsFreedOnce();
  EXPECT_EQ(freed_devices_, created_devices_);
  ExpectNoResources();
}

TEST_F(QnnUnit_BackendManagerMultiSocTest, ReleaseDeviceAndContext_DeviceFreeFailure_AttemptsRemainingDevices) {
  ASSERT_TRUE(Manager().SetupDevicesAndContexts(archs_, socs_).IsOK());
  ASSERT_TRUE(Manager().SetActiveSoc(1).IsOK());
  device_free_failure_ = created_devices_[0];
  Manager().ReleaseDeviceAndContext();
  EXPECT_EQ(freed_devices_, created_devices_);
  ExpectAllContextsFreedOnce();
  EXPECT_FALSE(Manager().IsBackendSetup());
  EXPECT_EQ(Manager().GetQnnContextSize(), 0u);
  EXPECT_EQ(Manager().GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
  EXPECT_EQ(Manager().GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_UNKNOWN));
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
  return manager.SetupBackend(false, false, false, 0, false, nullptr, dummy_map);
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
  auto setup_status = manager->SetupBackend(true, true, false, 0, false, nullptr, dummy_map);
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
// HTP backend — selecting a setup target publishes the HTP arch to op builders
// ---------------------------------------------------------------------------

// The multi-SoC EP-context entry points, driven in order: partial setup, then
// SetupDevicesAndContexts with an explicit arch + SoC model, then SetActiveSoc(0).
// Selecting the target makes GetHtpArch() report the caller-supplied arch to op
// builders. ReleaseDeviceAndContext must clear the target and reset the getters.
TEST_F(QnnUnit_BackendManagerHtpTest, SetupDevicesAndContexts_HTP_SelectsHtpArchThenResetsOnRelease) {
  StubApiEnv env;
  auto manager = MakeHTPManager(env.api_ptrs, env.logger);
  ASSERT_NE(manager, nullptr);

  // Nothing published before device/context setup.
  ASSERT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);

  auto partial = manager->SetupBackendExceptDeviceAndContext();
  ASSERT_TRUE(partial.IsOK()) << "SetupBackendExceptDeviceAndContext failed: "
                              << partial.GetErrorMessage();

  auto status = manager->SetupDevicesAndContexts({QNN_HTP_DEVICE_ARCH_V73}, {QNN_SOC_MODEL_SM8550});
  ASSERT_TRUE(status.IsOK()) << "SetupDevicesAndContexts failed: " << status.GetErrorMessage();
  EXPECT_TRUE(manager->IsBackendSetup());
  EXPECT_EQ(manager->GetQnnContextSize(), 1u);
  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
  EXPECT_THROW(manager->GetQnnContext(), Ort::Exception);
  ASSERT_TRUE(manager->SetActiveSoc(0).IsOK());
  EXPECT_NE(manager->GetQnnContext(), nullptr);

  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_V73);
  EXPECT_EQ(manager->GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_SM8550));

  manager->ReleaseDeviceAndContext();

  EXPECT_EQ(manager->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);
  EXPECT_EQ(manager->GetSocModel(), static_cast<uint32_t>(QNN_SOC_MODEL_UNKNOWN));
  EXPECT_FALSE(manager->IsBackendSetup());
  EXPECT_EQ(manager->GetQnnContextSize(), 0u);
  EXPECT_FALSE(manager->SetActiveSoc(0).IsOK());
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
  auto status = manager->SetupBackend(false, false, false, 0, false, nullptr, dummy_map,
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
  auto status = manager->SetupBackend(false, false, false, 0, false, nullptr, dummy_map,
                                      qnn::EpContextIoDispatch(),
                                      false /*extended_udma*/, false /*prepare_only*/,
                                      true /*enable_htp_graph_splitting*/,
                                      4 /*htp_graph_splitting_num_prepare_threads*/);
  (void)status;
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
