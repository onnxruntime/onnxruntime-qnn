// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"

#include <fstream>

#include "nlohmann/json.hpp"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Loads the QNN JSON graph from the given dump directory.
// Returns true on success and populates `root`. Emits gtest failures on error.
static bool LoadQnnGraphJson(const std::filesystem::path& dump_dir, nlohmann::json& root) {
  std::filesystem::path json_path;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      json_path = entry.path();
      break;
    }
  }

  EXPECT_FALSE(json_path.empty()) << "No QNN JSON graph file found in " << dump_dir;
  if (json_path.empty()) return false;

  std::ifstream json_file(json_path);
  EXPECT_TRUE(json_file.is_open()) << "Failed to open QNN JSON graph: " << json_path;
  if (!json_file.is_open()) return false;

  json_file >> root;

  EXPECT_TRUE(root.contains("graph") && root["graph"].contains("nodes"))
      << "JSON missing 'graph.nodes' field in: " << json_path;
  return root.contains("graph") && root["graph"].contains("nodes");
}

void AssertOpInQnnGraph(const std::filesystem::path& dump_dir,
                        const std::string& op,
                        size_t count) {
  nlohmann::json root;
  if (!LoadQnnGraphJson(dump_dir, root)) return;

  size_t actual_count = 0;
  for (const auto& [node_name, node_json] : root["graph"]["nodes"].items()) {
    if (node_json.value("type", "") == op) {
      ++actual_count;
    }
  }

  EXPECT_EQ(actual_count, count)
      << "QNN op '" << op << "': expected " << count
      << " occurrence(s), found " << actual_count;
}

void AssertOpHasScalarParam(const std::filesystem::path& dump_dir,
                            const std::string& op_type,
                            const std::string& param_name) {
  nlohmann::json root;
  if (!LoadQnnGraphJson(dump_dir, root)) return;

  for (const auto& [node_name, node_json] : root["graph"]["nodes"].items()) {
    if (node_json.value("type", "") != op_type) continue;

    EXPECT_TRUE(node_json.contains("scalar_params") &&
                node_json["scalar_params"].contains(param_name))
        << "QNN op '" << op_type << "' (node '" << node_name
        << "') is missing scalar param '" << param_name << "'";
    return;
  }

  ADD_FAILURE() << "No QNN op of type '" << op_type << "' found in graph";
}

}  // namespace test
}  // namespace onnxruntime
