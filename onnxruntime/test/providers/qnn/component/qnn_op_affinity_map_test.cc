// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for OpAffinityMap -- JSON parse paths and the Evaluate() truth table.
// Pure logic + temp-file I/O; no QNN backend, hardware, or emulator required.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <filesystem>
#include <fstream>
#include <string>

#include "core/providers/qnn/op_affinity/qnn_op_affinity_map.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace test {

using qnn::OpAffinityMap;
using qnn::QnnBackendType;

namespace {

// Writes `contents` to a uniquely-named temp file and returns its path. Caller deletes it.
std::filesystem::path WriteTempConfig(const std::string& contents, const std::string& tag) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("op_affinity_" + tag + ".json");
  std::ofstream ofs(path);
  ofs << contents;
  ofs.close();
  return path;
}

}  // namespace

// ---------------- Parse: success ----------------

TEST(QnnUnit_OpAffinityMap, ParsesSingleString) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "HTP" } })", "single");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP).IsOK());
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ParsesLengthOneArray) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": ["GPU"] } })", "arr1");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU).IsOK());
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, BackendNameIsCaseInsensitive) {
  for (const char* spelling : {"htp", "HTP", "Htp"}) {
    const auto path = WriteTempConfig(
        std::string(R"({ "op_type": { "GroupQueryAttention": ")") + spelling + R"(" } })", "case");
    const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
    EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP).IsOK()) << spelling;
    std::filesystem::remove(path);
  }
}

// ---------------- Parse: throw paths ----------------

TEST(QnnUnit_OpAffinityMap, ThrowsWhenFileMissing) {
  const std::filesystem::path missing =
      std::filesystem::temp_directory_path() / "op_affinity_does_not_exist_12345.json";
  EXPECT_THROW(OpAffinityMap::FromConfigFile(missing), std::runtime_error);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnBadJson) {
  const auto path = WriteTempConfig("{ not valid json ", "badjson");
  EXPECT_ANY_THROW(OpAffinityMap::FromConfigFile(path));
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsWhenOpTypeMissing) {
  const auto path = WriteTempConfig(R"({ "something_else": {} })", "nooptype");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnNumericValue) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": 3 } })", "numeric");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnEmptyArray) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": [] } })", "emptyarr");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnMultiElementArray) {
  const auto path =
      WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": ["HTP", "GPU"] } })", "multiarr");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnUnknownBackend) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "NPU2" } })", "unknownbe");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsWhenOpTypeNotObject) {
  // "op_type" present but a string, not an object -> distinct branch from the missing-key case.
  const auto path = WriteTempConfig(R"({ "op_type": "GroupQueryAttention" })", "optype_notobj");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnNonStringArrayElement) {
  // Length-1 array whose element is not a string -> exercises the non-string-array-element branch.
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": [3] } })", "arr_nonstr");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

// ---------------- Evaluate: truth table ----------------

TEST(QnnUnit_OpAffinityMap, UnpinnedOpProceedsOnAnyBackend) {
  const OpAffinityMap map;  // default = unconfigured, nothing pinned
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP).IsOK());
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU).IsOK());
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::CPU).IsOK());
}

TEST(QnnUnit_OpAffinityMap, ConfiguredButOpAbsentProceeds) {
  const auto path = WriteTempConfig(R"({ "op_type": { "SomeOtherOp": "HTP" } })", "absent");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP).IsOK());
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU).IsOK());
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, PinHtpEvaluations) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "HTP" } })", "pinhtp");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP).IsOK());
  EXPECT_FALSE(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU).IsOK());
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, PinGpuEvaluations) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "GPU" } })", "pingpu");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU).IsOK());
  EXPECT_FALSE(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP).IsOK());
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, PinCpuRejectsOnAccelerators) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "CPU" } })", "pincpu");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_FALSE(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP).IsOK());
  EXPECT_FALSE(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU).IsOK());
  std::filesystem::remove(path);
}

// ---------------- SeedDefaultIfAbsent ----------------

TEST(QnnUnit_OpAffinityMap, SeedAppliesWhenOpAbsent) {
  OpAffinityMap map;  // unconfigured, no pins
  map.SeedDefaultIfAbsent("GroupQueryAttention", QnnBackendType::CPU);
  EXPECT_FALSE(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP).IsOK());
  EXPECT_FALSE(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU).IsOK());
}

TEST(QnnUnit_OpAffinityMap, SeedDoesNotOverrideExistingConfigPin) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "HTP" } })", "seed_override");
  OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  map.SeedDefaultIfAbsent("GroupQueryAttention", QnnBackendType::CPU);
  EXPECT_TRUE(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP).IsOK());
  std::filesystem::remove(path);
}

// ---------------- ValidateForSessionBackend ----------------

TEST(QnnUnit_OpAffinityMap, ValidateReportsErrorWhenPinnedToOtherAccelerator) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "GPU" } })", "validate_gpu");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  // Session runs HTP, but GQA is pinned to GPU -> must report an error.
  EXPECT_FALSE(map.ValidateForSessionBackend(QnnBackendType::HTP).IsOK());
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ValidatePassesWhenPinnedToSessionBackend) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "HTP" } })", "validate_htp");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_TRUE(map.ValidateForSessionBackend(QnnBackendType::HTP).IsOK());
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ValidatePassesWhenPinnedToCpu) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "CPU" } })", "validate_cpu");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  // CPU pin is a legitimate silent-off intent, never a validation error, regardless of session backend.
  EXPECT_TRUE(map.ValidateForSessionBackend(QnnBackendType::HTP).IsOK());
  EXPECT_TRUE(map.ValidateForSessionBackend(QnnBackendType::GPU).IsOK());
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ValidateIsNoOpWhenUnconfigured) {
  const OpAffinityMap map;  // unconfigured
  EXPECT_TRUE(map.ValidateForSessionBackend(QnnBackendType::HTP).IsOK());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
