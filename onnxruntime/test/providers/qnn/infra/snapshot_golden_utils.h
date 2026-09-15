// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Golden-file path and JSON normalization helpers shared by snapshot tiers.

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

// Empty means no external golden store was provided.
inline std::string GetGoldenRootDir() {
  const char* env = std::getenv("QNN_UT_SNAPSHOT_GOLDEN_DIR");
  return (env != nullptr && env[0] != '\0') ? std::string(env) : std::string();
}

// Remove fields that are not stable across test runs.
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

// Derive the golden subdirectory from a test source path, e.g.
// providers/qnn/snapshot/builder/opbuilder/clip_test.cc ->
// snapshot/builder/opbuilder/clip.
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
