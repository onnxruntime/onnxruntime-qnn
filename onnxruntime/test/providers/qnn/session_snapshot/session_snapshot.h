// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Session-level snapshot helpers for QNN EP unit tests.
//
// Drives a model through the full ORT InferenceSession (AOT inline,
// EnsureUniqueDQ, L1 transformers, Partition::transform_layout_fn,
// QNN EP Compile -> ComposeQnnGraph -> JSON dump) and diffs the resulting
// QNN graph against a stored golden. Captures what QNN actually receives
// after pre-partition + partition-time transforms — content the op-builder
// snapshot (AssertSnapshotJson, qnn_unit_test_utils.h) cannot see because
// it bypasses ORT optimizer / partition entirely.
//
// Per-op-type role (see evolve_ut_infra_plan.md §Op-builder vs Session
// Snapshot):
//   * Layout-agnostic ops (Clip, Add, Mul, ...): primary is the op-builder
//     snapshot; this layer is a narrow L1-regression sentinel (~1 case / op).
//   * Layout-sensitive ops (Conv, Pool, Resize, Transpose, ...): primary
//     here — only this layer captures the post-transform_layout_fn NHWC
//     graph that QNN backend op validation actually receives.
//
// L2+ regressions (ClipQuantFusion, ReluQuantFusion, ...) are intentionally
// out of scope: JSON dumps inside Partition::Compile before L2 runs. L2
// transformers live in ORT core and are not changed by QNN EP develop PRs;
// L2 regressions are caught by the ORT up-level CI lane (full accuracy).
//
// Single-partition only for now; multi-partition extension tracked under
// the same ticket as AssertSnapshotJson's analogous gap.
//
// Usage:
//
//   ProviderOptions provider_options;
//   provider_options["backend_type"] = "htp";
//   provider_options["offload_graph_io_quantization"] = "0";
//
//   auto build = [](ModelTestBuilder& builder) { /* build QDQ Clip graph */ };
//   AssertSessionSnapshotJson(build, provider_options, /*opset=*/13,
//                             "Clip_U8_DefaultMinMax_Rank4");
//
// Goldens live alongside op-builder-snapshot goldens under the golden root
// (which points directly at the goldens tree):
//   $QNN_UT_SNAPSHOT_GOLDEN_DIR/<subdir>/<basename>.json
// Distinguished from op-builder goldens by the tier directory
// (e.g. session_snapshot/ vs snapshot/) and by the `QnnUnit_*_SessionSnapshot*`
// test name. Subdir auto-derived from caller __FILE__ via
// DeriveGoldenSubdirFromFile (same as AssertSnapshotJson).
//
// Generate / update goldens with (both env vars on one line):
//   QNN_UT_SNAPSHOT_GOLDEN_DIR=<dir> QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1 ./onnxruntime_provider_test --gtest_filter='QnnUnit_*_SessionSnapshot*'

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "nlohmann/json.hpp"

#include "test/providers/qnn/qnn_test_utils.h"                    // ModelTestBuilder, RegisterQnnEpLibrary, GetTestModelFn
#include "test/providers/qnn/test_infra/snapshot_golden_utils.h"  // GetGoldenRootDir, NormalizeQnnJSONGraph, DeriveGoldenSubdirFromFile

namespace onnxruntime {
namespace test {

inline void AssertSessionSnapshotJson(
    const GetTestModelFn& build_test_case,
    ProviderOptions provider_options,
    int opset_version,
    const std::string& golden_basename,
    std::string golden_subdir = "",
    const char* caller_file = __builtin_FILE()) {
  if (golden_subdir.empty()) {
    golden_subdir = DeriveGoldenSubdirFromFile(caller_file);
    ASSERT_FALSE(golden_subdir.empty())
        << "AssertSessionSnapshotJson: could not derive golden_subdir from " << caller_file
        << "\nExpected the caller test file to live under .../providers/qnn/<tier>/<subdir>/<name>_test.cc"
        << "\nPass golden_subdir explicitly to override.";
  }

  // Stage a temp dir for QNN EP to dump JSON into. Use the test name to make
  // concurrent test runs safe and cleanup easy to spot in /tmp.
  const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string slot = test_info != nullptr
                         ? std::string(test_info->test_suite_name()) + "." + test_info->name()
                         : golden_basename;
  std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() /
                                  ("qnn_session_snapshot_" + slot);
  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  ASSERT_TRUE(std::filesystem::create_directories(tmp_dir, ec))
      << "Failed to create temp dir " << tmp_dir << ": " << ec.message();

  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = tmp_dir.string();

  // Build and serialize the model — same pattern as RunQnnModelTest.
  const std::unordered_map<std::string, int> domain_to_version = {
      {"", opset_version}, {kMSDomain, 1}};
  ModelTestBuilder helper;
  build_test_case(helper);
  for (const auto& [domain, version] : domain_to_version) {
    auto* opset_id_proto = helper.model_.add_opset_import();
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  // Register QNN EP and create a session — Compile runs during session init,
  // which triggers the JSON dump. We do not invoke Run() — session snapshot
  // only verifies graph structure, not numerics.
  Ort::SessionOptions session_options;
  session_options.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR);
  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string registration_name = "QNNExecutionProvider";
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  Ort::Session session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options);

  // Locate the dumped QNN graph JSON(s). Skip `_tensor_log.json` (a separate
  // log EP writes alongside the main graph dump). PoC scope: exactly one
  // graph partition expected.
  std::vector<std::filesystem::path> jsons;
  for (const auto& entry : std::filesystem::directory_iterator(tmp_dir)) {
    const std::string name = entry.path().filename().string();
    if (entry.path().extension() == ".json" &&
        name.find("_tensor_log.") == std::string::npos) {
      jsons.push_back(entry.path());
    }
  }
  ASSERT_EQ(jsons.size(), 1u)
      << "Expected exactly one QNN graph JSON dump in " << tmp_dir
      << ", got " << jsons.size()
      << ".\n(Multi-partition session-snapshot is out of scope for this PoC; tracked separately.)";

  std::ifstream ifs(jsons[0]);
  std::string raw_json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  ifs.close();
  std::filesystem::remove_all(tmp_dir, ec);

  nlohmann::json graph = nlohmann::json::parse(raw_json);
  NormalizeQnnJSONGraph(graph);

  const std::string current = graph.dump(2) + "\n";    // Match AssertSnapshotJson formatting.
  const std::string golden_root = GetGoldenRootDir();  // "" == golden store absent
  const bool have_root = !golden_root.empty();
  const std::string golden_dir = golden_root + "/" + golden_subdir;
  const std::string golden_path = golden_dir + "/" + golden_basename + ".json";

  const char* update_env = std::getenv("QNN_UT_SNAPSHOT_GOLDEN_UPDATE");
  bool update = (update_env != nullptr && std::string(update_env) == "1");

  if (update) {
    ASSERT_TRUE(have_root)
        << "QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1 but QNN_UT_SNAPSHOT_GOLDEN_DIR is unset — "
           "nowhere to write goldens.";
    std::filesystem::create_directories(golden_dir);
    std::ofstream out(golden_path);
    ASSERT_TRUE(out.is_open()) << "Failed to open golden file for writing: " << golden_path;
    out << current;
    out.close();
    GTEST_SKIP() << "Session-snapshot golden updated: " << golden_path;
    return;
  }

  // Absent golden store (or missing file) is not a failure: the gate treats it
  // as "run accuracy instead". [QNN_GOLDEN_ABSENT] is an inert marker here.
  std::ifstream in;
  if (have_root) in.open(golden_path);
  if (!have_root || !in.is_open()) {
    GTEST_SKIP() << "[QNN_GOLDEN_ABSENT] op=" << golden_subdir
                 << " name=" << golden_basename;
    return;
  }
  std::string expected((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_EQ(current, expected)
      << "[QNN_SNAPSHOT_DRIFT] name=" << golden_basename
      << "\nSession-snapshot diff detected. Regenerate with "
         "QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1.";
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
