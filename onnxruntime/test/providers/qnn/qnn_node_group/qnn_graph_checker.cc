// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"

#include <fstream>

#include "nlohmann/json.hpp"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

void AssertOpInQnnGraph(const std::filesystem::path& dump_dir,
                        const std::string& op,
                        size_t count) {
  std::filesystem::path json_path;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      json_path = entry.path();
      break;
    }
  }
  ASSERT_FALSE(json_path.empty()) << "No QNN JSON graph file found in " << dump_dir;

  std::ifstream json_file(json_path);
  ASSERT_TRUE(json_file.is_open()) << "Failed to open QNN JSON graph: " << json_path;

  nlohmann::json root;
  json_file >> root;

  ASSERT_TRUE(root.contains("graph") && root["graph"].contains("nodes"))
      << "JSON missing 'graph.nodes' field in: " << json_path;

  size_t actual_count = 0;
  for (const auto& [node_name, node_json] : root["graph"]["nodes"].items()) {
    if (node_json.value("type", "") == op) {
      ++actual_count;
    }
  }

  EXPECT_EQ(actual_count, count)
      << "QNN op '" << op << "': expected " << count
      << " occurrence(s), found " << actual_count << " in " << json_path;
}

void AssertNodeNotInQnnGraph(const std::filesystem::path& dump_dir,
                             const std::string& node_name) {
  std::filesystem::path json_path;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      json_path = entry.path();
      break;
    }
  }
  ASSERT_FALSE(json_path.empty()) << "No QNN JSON graph file found in " << dump_dir;

  std::ifstream json_file(json_path);
  ASSERT_TRUE(json_file.is_open()) << "Failed to open QNN JSON graph: " << json_path;

  nlohmann::json root;
  json_file >> root;

  ASSERT_TRUE(root.contains("graph") && root["graph"].contains("nodes"))
      << "JSON missing 'graph.nodes' field in: " << json_path;

  EXPECT_FALSE(root["graph"]["nodes"].contains(node_name))
      << "Unexpected QNN node found: '" << node_name << "' in " << json_path;
}

void AssertTensorShapeInQnnGraph(const std::filesystem::path& dump_dir,
                                 const std::string& tensor_name,
                                 const std::vector<uint32_t>& expected_dims) {
  std::filesystem::path json_path;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      json_path = entry.path();
      break;
    }
  }
  ASSERT_FALSE(json_path.empty()) << "No QNN JSON graph file found in " << dump_dir;

  std::ifstream json_file(json_path);
  ASSERT_TRUE(json_file.is_open()) << "Failed to open QNN JSON graph: " << json_path;

  nlohmann::json root;
  json_file >> root;

  ASSERT_TRUE(root.contains("graph") && root["graph"].contains("tensors"))
      << "JSON missing 'graph.tensors' field in: " << json_path;
  ASSERT_TRUE(root["graph"]["tensors"].contains(tensor_name))
      << "QNN tensor '" << tensor_name << "' not found in " << json_path;

  const auto& tensor_json = root["graph"]["tensors"][tensor_name];
  ASSERT_TRUE(tensor_json.contains("dims"))
      << "QNN tensor '" << tensor_name << "' missing 'dims' in " << json_path;

  const std::vector<uint32_t> actual_dims = tensor_json["dims"].get<std::vector<uint32_t>>();
  EXPECT_EQ(actual_dims, expected_dims)
      << "QNN tensor '" << tensor_name << "': expected shape mismatch in " << json_path;
}

}  // namespace test
}  // namespace onnxruntime
