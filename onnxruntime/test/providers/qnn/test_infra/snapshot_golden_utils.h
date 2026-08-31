// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Pure golden-file path / JSON normalization helpers shared by the
// op-builder snapshot tests (snapshot.h) and the session-level snapshot
// tests (session_snapshot.h).
//
// Kept dependency-free (only nlohmann/json + stdlib) so it can be safely
// included from translation units that pull either the QNN-EP-internal world
// (ort_api.h) or the full ORT world (qnn_test_utils.h via core/graph/...).
// The two worlds both define kOnnxDomain etc. and cannot coexist in one TU.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace test {

// Golden-tree root, taken from $QNN_UT_SNAPSHOT_GOLDEN_DIR (absolute path, no
// trailing slash). Returns "" when the env var is unset or empty, which the
// snapshot harness treats as "no golden available" — there is deliberately no
// in-repo fallback: goldens live in an external store, not the source tree.
// The path reaches the test binary via the process environment rather than
// argv because gtest owns the binary's command line.
inline std::string GetGoldenRootDir() {
  const char* env = std::getenv("QNN_UT_SNAPSHOT_GOLDEN_DIR");
  return (env != nullptr && env[0] != '\0') ? std::string(env) : std::string();
}

// Strips fields from the JSON graph that are not stable across test runs.
// Currently: tensor `id` (a process-wide counter — stable within a single
// process / test, but shifts when multiple snapshot tests run in the same
// process; verified empirically 2026-05-14). Everything else (tensor name,
// node name, dims, dtype, quant_params, scalar params, params_data_hash)
// is byte-stable.
//
// Mutates `graph` in place and returns a reference to it for chaining.
inline nlohmann::json& NormalizeQnnJSONGraph(nlohmann::json& graph) {
  auto graph_it = graph.find("graph");
  if (graph_it == graph.end() || !graph_it->is_object()) return graph;
  auto tensors_it = graph_it->find("tensors");
  if (tensors_it == graph_it->end() || !tensors_it->is_object()) return graph;
  for (auto& tensor : tensors_it->items()) {
    if (tensor.value().is_object()) {
      tensor.value().erase("id");
    }
  }
  return graph;
}

// Derive `goldens/<subdir>/` from a test source file path. Strips everything
// up to and including `providers/qnn/` and trims `_test.cc` / `_test.cpp`
// suffix. The retained leading segment is the tier directory, so the golden
// tree is naturally partitioned by tier:
//   /repo/.../providers/qnn/snapshot/builder/opbuilder/clip_test.cc
//     -> "snapshot/builder/opbuilder/clip"
//   /repo/.../providers/qnn/session_snapshot/builder/opbuilder/clip_test.cc
//     -> "session_snapshot/builder/opbuilder/clip"
//
// Returns empty string if the path doesn't contain `providers/qnn/` (caller
// should fall back to an explicit subdir). Handles both forward and backward
// slashes for Windows portability.
inline std::string DeriveGoldenSubdirFromFile(std::string_view file_path) {
  static constexpr std::string_view kAnchorFwd = "/providers/qnn/";
  static constexpr std::string_view kAnchorBack = "\\providers\\qnn\\";

  auto pos = file_path.rfind(kAnchorFwd);
  size_t skip = kAnchorFwd.size();
  if (pos == std::string_view::npos) {
    pos = file_path.rfind(kAnchorBack);
    skip = kAnchorBack.size();
    if (pos == std::string_view::npos) return "";
  }

  std::string rel(file_path.substr(pos + skip));

  for (std::string_view suf : {std::string_view{"_test.cc"}, std::string_view{"_test.cpp"}}) {
    if (rel.size() >= suf.size() &&
        rel.compare(rel.size() - suf.size(), suf.size(), suf) == 0) {
      rel.erase(rel.size() - suf.size());
      break;
    }
  }

  std::replace(rel.begin(), rel.end(), '\\', '/');
  return rel;
}

// ---------------------------------------------------------------------------
// CompareOrWriteGolden
//
// Shared golden compare/write/skip protocol for both the op-builder snapshot
// tier (snapshot.h) and the session-level snapshot tier (session_snapshot.h).
// Each caller obtains `current` (the normalized, pretty-printed JSON string)
// its own way — that's the only genuinely different part between the two
// tiers — then hands it here for the identical golden-store logic:
//   - Compares against the stored golden (default, CI mode)
//   - Writes/overwrites the golden (when QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1)
//   - Skips with [QNN_GOLDEN_ABSENT] when the golden store is unset/missing
//
// `drift_label` is folded into the update/failure messages to keep them
// distinguishable per tier (e.g. "JSON snapshot" vs "Session-snapshot").
// ---------------------------------------------------------------------------
inline void CompareOrWriteGolden(const std::string& current,
                                 const std::string& golden_basename,
                                 const std::string& golden_subdir,
                                 const char* drift_label) {
  const std::string golden_root = GetGoldenRootDir();  // "" == golden store absent
  const bool have_root = !golden_root.empty();
  const std::string golden_dir = golden_root + "/" + golden_subdir;
  const std::string golden_path = golden_dir + "/" + golden_basename + ".json";

  const char* update_env = std::getenv("QNN_UT_SNAPSHOT_GOLDEN_UPDATE");
  const bool update = (update_env != nullptr && std::string(update_env) == "1");

  if (update) {
    ASSERT_TRUE(have_root)
        << "QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1 but QNN_UT_SNAPSHOT_GOLDEN_DIR is unset — "
           "nowhere to write goldens.";
    std::filesystem::create_directories(golden_dir);
    std::ofstream out(golden_path);
    ASSERT_TRUE(out.is_open()) << "Failed to open golden file for writing: " << golden_path;
    out << current;
    out.close();
    GTEST_SKIP() << drift_label << " golden updated: " << golden_path;
    return;
  }

  // Absent golden store (or missing file) is not a failure: the gate treats it
  // as "run accuracy instead". The [QNN_GOLDEN_ABSENT] tag is an inert marker
  // here; only the CI gate parses it.
  std::ifstream in;
  if (have_root) in.open(golden_path);
  if (!have_root || !in.is_open()) {
    GTEST_SKIP() << "[QNN_GOLDEN_ABSENT] op=" << golden_subdir
                 << " name=" << golden_basename;
    return;
  }
  std::string expected((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  EXPECT_EQ(current, expected)
      << "[QNN_SNAPSHOT_DRIFT] name=" << golden_basename
      << "\n"
      << drift_label << " diff detected. Regenerate with "
                        "QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1.";
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
