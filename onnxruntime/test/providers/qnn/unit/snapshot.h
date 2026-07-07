// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Snapshot helpers (Path E1 — JSON-based) for QNN EP function-level unit tests.
//
// Usage in snapshot tests:
//
//   QnnRealCpuBackendManagerContext cpu;
//   if (!cpu.IsValid()) GTEST_SKIP() << "libQnnCpu.so not available";
//   SnapshotTestContext ctx;
//   auto wrapper = MakeSnapshotWrapperJson(ctx, cpu, {"data"}, {"output"});
//   ASSERT_TRUE(builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false).IsOK());
//   ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true));
//   AssertSnapshotJson(*wrapper, "Clip_f32_DefaultMinMax_Rank4");
//
// To generate / update goldens, run tests with env var:
//   QNN_UPDATE_GOLDENS=1 ./onnxruntime_provider_test --gtest_filter="QnnUnit_Snapshot_*"
//
// Golden files are stored at:
//   test/providers/qnn/unit/goldens/<subdir>/<test_name>.json
// (subdir auto-derived from caller __FILE__; see DeriveGoldenSubdirFromFile.)

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "nlohmann/json.hpp"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include "test/providers/qnn/unit/backend_contexts.h"
#include "test/providers/qnn/unit/golden_paths.h"

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// MakeSnapshotWrapper
//
// Constructs a stub-backed QnnModelWrapper and registers input/output tensor
// names so that the wrapper assigns correct tensor types (APP_WRITE / APP_READ)
// in the snapshot output. Use when no live backend is needed.
// ---------------------------------------------------------------------------
inline std::unique_ptr<qnn::QnnModelWrapper> MakeSnapshotWrapper(
    SnapshotTestContext& ctx,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names) {
  for (size_t i = 0; i < input_names.size(); ++i) {
    ctx.input_info.names.push_back(input_names[i]);
    ctx.input_info.indices[input_names[i]] = i;
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    ctx.output_info.names.push_back(output_names[i]);
    ctx.output_info.indices[output_names[i]] = i;
  }
  qnn::ModelSettings settings{};
  return ctx.CreateWrapper(settings);
}

// ---------------------------------------------------------------------------
// MakeSnapshotWrapperJson
//
// Like MakeSnapshotWrapper but wires up QnnRealCpuBackendManagerContext (real
// CPU backend) and calls CreateQnnGraph so ComposeQnnGraph can be called.
// Use with AssertSnapshotJson — call wrapper->ComposeQnnGraph(true) before
// asserting.
//
// The wrapper is constructed with backend_type=CPU so HTP-specific
// transforms (BF16 conversion etc.) do not rewrite the op list.
// Returns nullptr if graph initialization fails.
// ---------------------------------------------------------------------------
inline std::unique_ptr<qnn::QnnModelWrapper> MakeSnapshotWrapperJson(
    SnapshotTestContext& ctx,
    const QnnRealCpuBackendManagerContext& cpu,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names) {
  // Reset the process-singleton name counter so generated node/tensor names
  // are stable across test orderings (otherwise EP appends `_N` to dedupe
  // and snapshots accumulate counter drift).
  qnn::utils::UniqueNameGenerator().Reset();

  ctx.qnn_interface = cpu.qnn_interface;
  ctx.backend_handle = cpu.backend_handle;

  for (size_t i = 0; i < input_names.size(); ++i) {
    ctx.input_info.names.push_back(input_names[i]);
    ctx.input_info.indices[input_names[i]] = i;
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    ctx.output_info.names.push_back(output_names[i]);
    ctx.output_info.indices[output_names[i]] = i;
  }

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::CPU);

  if (!wrapper->CreateQnnGraph(cpu.context_handle, "test_graph", nullptr)) {
    return nullptr;
  }
  return wrapper;
}

// ---------------------------------------------------------------------------
// MakeSnapshotWrapperHtpJson
//
// Like MakeSnapshotWrapperJson but uses QnnRealHtpBackendManagerContext (real
// HTP backend). For test cases whose dtypes (FP16, U16 quantization, U8/U16
// mixed) are not supported by libQnnCpu — graphAddNode would otherwise reject
// with rc 3110.
//
// Note: backend_type=HTP triggers BF16 conversion ONLY for FP32 input tensors
// (qnn_model_wrapper.cc:278). FP16 / U16 graphs are unaffected — op list
// stays clean. Do NOT use this for FP32 cases (CPU variant is the right
// choice there to avoid unnecessary BF16 cast insertion).
// ---------------------------------------------------------------------------
inline std::unique_ptr<qnn::QnnModelWrapper> MakeSnapshotWrapperHtpJson(
    SnapshotTestContext& ctx,
    const QnnRealHtpBackendManagerContext& htp,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names) {
  qnn::utils::UniqueNameGenerator().Reset();

  ctx.qnn_interface = htp.qnn_interface;
  ctx.backend_handle = htp.backend_handle;

  for (size_t i = 0; i < input_names.size(); ++i) {
    ctx.input_info.names.push_back(input_names[i]);
    ctx.input_info.indices[input_names[i]] = i;
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    ctx.output_info.names.push_back(output_names[i]);
    ctx.output_info.indices[output_names[i]] = i;
  }

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  if (!wrapper->CreateQnnGraph(htp.context_handle, "test_graph", nullptr)) {
    return nullptr;
  }
  return wrapper;
}

// ---------------------------------------------------------------------------
// AssertSnapshotJson
//
// Path E1 snapshot assertion. Reads `wrapper.GetQnnJSONGraph()` (must be called
// after `wrapper.ComposeQnnGraph(true)`), normalizes it (drops unstable
// tensor `id`), pretty-prints, and either:
//   - Compares against the stored golden (default, CI mode)
//   - Writes/overwrites the golden (when QNN_UPDATE_GOLDENS=1)
//
// golden_basename: e.g. "Clip_4D_f32_DefaultMinMax" (no extension, no path)
// golden_subdir  : path relative to unit/goldens/, e.g. "builder/opbuilder/clip"
//                  (auto-derived from caller __FILE__ if empty)
//
// Golden file: <source_dir>/goldens/<golden_subdir>/<golden_basename>.json
// ---------------------------------------------------------------------------
inline void AssertSnapshotJson(qnn::QnnModelWrapper& wrapper,
                               const std::string& golden_basename,
                               std::string golden_subdir = "",
                               const char* caller_file = __builtin_FILE()) {
  if (golden_subdir.empty()) {
    golden_subdir = DeriveGoldenSubdirFromFile(caller_file);
    ASSERT_FALSE(golden_subdir.empty())
        << "AssertSnapshotJson: could not derive golden_subdir from "
        << caller_file
        << "\nExpected the caller test file to live under .../unit/<subdir>/<name>_test.cc"
        << "\nPass golden_subdir explicitly to override.";
  }

  const std::string golden_dir = GetUnitTestSourceDir() + "/goldens/" + golden_subdir;
  const std::string golden_path = golden_dir + "/" + golden_basename + ".json";

  // Copy so normalization does not mutate wrapper-owned state (Finalize returns const ref).
  nlohmann::json graph = wrapper.GetQnnJSONGraph();
  NormalizeQnnJSONGraph(graph);
  const std::string current = graph.dump(2) + "\n";  // 2-space indent + trailing newline.

  const char* update_env = std::getenv("QNN_UPDATE_GOLDENS");
  bool update = (update_env != nullptr && std::string(update_env) == "1");

  if (update) {
    std::filesystem::create_directories(golden_dir);
    std::ofstream out(golden_path);
    ASSERT_TRUE(out.is_open()) << "Failed to open golden file for writing: " << golden_path;
    out << current;
    out.close();
    GTEST_SKIP() << "Golden updated: " << golden_path;
  } else {
    std::ifstream in(golden_path);
    ASSERT_TRUE(in.is_open())
        << "Golden file not found: " << golden_path
        << "\nRun with QNN_UPDATE_GOLDENS=1 to generate.";
    std::string expected((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
    EXPECT_EQ(current, expected)
        << "JSON snapshot diff detected for " << golden_basename
        << ".\nRun with QNN_UPDATE_GOLDENS=1 to update golden.";
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
