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
#include <string>
#include <string_view>

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

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
