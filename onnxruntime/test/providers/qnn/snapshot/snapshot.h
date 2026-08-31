// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Snapshot helpers (Path E1 — JSON-based) for QNN EP function-level unit tests.
//
// Usage in snapshot tests:
//
//   QnnRealHtpBackendManagerContext htp;
//   if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
//   OpBuilderTestContext ctx;
//   auto wrapper = MakeSnapshotWrapperHtpJson(ctx, htp, {"data"}, {"output"});
//   ASSERT_TRUE(builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false).IsOK());
//   ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true));
//   AssertSnapshotJson(*wrapper, "Clip_f32_DefaultMinMax_Rank4");
//
// To generate / update goldens, point the harness at a golden tree and set
// the update toggle (both env vars on one line):
//   QNN_UT_SNAPSHOT_GOLDEN_DIR=<dir> QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1 ./onnxruntime_provider_test --gtest_filter="QnnUnit_*_Snapshot*"
//
// Golden files live under $QNN_UT_SNAPSHOT_GOLDEN_DIR (which points directly at
// the goldens root):
//   $QNN_UT_SNAPSHOT_GOLDEN_DIR/<subdir>/<test_name>.json
// (subdir auto-derived from caller __FILE__; see DeriveGoldenSubdirFromFile.)
// When the env var is unset/empty the compare is skipped (see AssertSnapshotJson).

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "nlohmann/json.hpp"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include "test/providers/qnn/test_infra/backend_contexts.h"
#include "test/providers/qnn/test_infra/snapshot_golden_utils.h"

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
    OpBuilderTestContext& ctx,
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
// MakeSnapshotWrapperHtpJson
//
// Wires up QnnRealHtpBackendManagerContext (real HTP backend) and calls
// CreateQnnGraph so ComposeQnnGraph can be called. Use with AssertSnapshotJson
// — call wrapper->ComposeQnnGraph(true) before asserting.
//
// Note: backend_type=HTP triggers BF16 conversion for FP32 input tensors
// (qnn_model_wrapper.cc:278), which inserts Convert ops into the op list.
// FP16 / U16 graphs are unaffected. FP32 golden JSON must reflect the
// inserted Convert ops since there is no CPU-backend alternative anymore.
// Returns nullptr if graph initialization fails.
// ---------------------------------------------------------------------------
inline std::unique_ptr<qnn::QnnModelWrapper> MakeSnapshotWrapperHtpJson(
    OpBuilderTestContext& ctx,
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
//   - Writes/overwrites the golden (when QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1)
//   - Skips with [QNN_GOLDEN_ABSENT] when the golden store is unset/missing
//
// golden_basename: e.g. "Clip_4D_f32_DefaultMinMax" (no extension, no path)
// golden_subdir  : path relative to the golden root, e.g. "builder/opbuilder/clip"
//                  (auto-derived from caller __FILE__ if empty)
//
// Golden file: $QNN_UT_SNAPSHOT_GOLDEN_DIR/<golden_subdir>/<golden_basename>.json
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
        << "\nExpected the caller test file to live under .../providers/qnn/<tier>/<subdir>/<name>_test.cc"
        << "\nPass golden_subdir explicitly to override.";
  }

  // Copy so normalization does not mutate wrapper-owned state (Finalize returns const ref).
  nlohmann::json graph = wrapper.GetQnnJSONGraph();
  NormalizeQnnJSONGraph(graph);
  const std::string current = graph.dump(2) + "\n";  // 2-space indent + trailing newline.

  CompareOrWriteGolden(current, golden_basename, golden_subdir, "JSON snapshot");
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
