// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for QnnBackendManager (qnn_backend_manager.cc).
//
// All tests here run without loading any real QNN shared library. Tests that
// require a real QNN backend (libQnnHtp.so, libQnnIr.so, etc.) live in
// integration/qnn_backend_manager_test.cc.
//
// Coverage targets:
//   - QnnSerializerConfig (CreateIr / CreateSaver / GetBackendPath / SetGraphName / Configure)
//   - SetupBackend error paths (library load failure — invalid / empty path)
//   - ResetQnnLogLevel before backend is set up (early-return OK path)
//   - GetContextBinaryBuffer before backend is set up (returns error)
//   - ParseLoraConfig file I/O error paths

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_backend_manager.h"
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

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
