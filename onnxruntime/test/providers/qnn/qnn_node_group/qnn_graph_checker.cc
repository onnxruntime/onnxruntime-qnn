// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"

#include <fstream>

#include "nlohmann/json.hpp"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

bool FindQnnJsonGraph(const std::filesystem::path& dump_dir,
                      std::filesystem::path& json_path) {
  json_path.clear();
  try {
    std::error_code ec;
    std::filesystem::directory_iterator it(dump_dir, ec);
    if (ec) {
      return false;
    }

    const std::filesystem::directory_iterator end;
    for (; it != end; it.increment(ec)) {
      if (ec) {
        return false;
      }

      try {
        std::error_code entry_ec;
        if (it->is_regular_file(entry_ec) && !entry_ec && it->path().extension() == ".json" &&
            it->path().filename().string().find("_tensor_log") == std::string::npos) {
          json_path = it->path();
          return true;
        }
      } catch (const std::exception&) {
        continue;
      }
    }
  } catch (const std::exception&) {
    return false;
  }

  return !json_path.empty();
}

bool ParseQnnJsonGraph(const std::filesystem::path& json_path,
                       nlohmann::json& root) {
  try {
    std::ifstream json_file(json_path);
    if (!json_file.is_open()) {
      return false;
    }

    json_file >> root;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

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

void AssertConvertOutputDataType(const std::filesystem::path& dump_dir,
                                 uint32_t expected_data_type) {
  std::filesystem::path json_path;
  ASSERT_TRUE(FindQnnJsonGraph(dump_dir, json_path))
      << "No QNN JSON graph file found in " << dump_dir;

  nlohmann::json root;
  ASSERT_TRUE(ParseQnnJsonGraph(json_path, root))
      << "Failed to parse QNN JSON graph: " << json_path;
  ASSERT_TRUE(root.is_object() && root.contains("graph") && root["graph"].is_object() &&
              root["graph"].contains("nodes") && root["graph"]["nodes"].is_object() &&
              root["graph"].contains("tensors") && root["graph"]["tensors"].is_object())
      << "JSON missing 'graph.nodes' or 'graph.tensors' object in: " << json_path;

  const auto& nodes = root["graph"]["nodes"];
  const auto& tensors = root["graph"]["tensors"];
  const nlohmann::json* convert = nullptr;
  try {
    for (const auto& [node_name, node] : nodes.items()) {
      if (node.is_object() && node.contains("type") && node["type"].is_string() &&
          node["type"].get<std::string>() == "Convert") {
        ASSERT_EQ(convert, nullptr) << "Expected one Convert node, found more than one";
        convert = &node;
      }
    }
  } catch (const std::exception& ex) {
    FAIL() << "Failed to iterate QNN graph nodes in " << json_path << ": " << ex.what();
  }

  ASSERT_NE(convert, nullptr) << "No Convert node found in " << json_path;
  ASSERT_TRUE(convert->contains("output_names") && (*convert)["output_names"].is_array() &&
              (*convert)["output_names"].size() == 1 && (*convert)["output_names"][0].is_string())
      << "Convert node has invalid output_names in " << json_path;
  const std::string output_name = (*convert)["output_names"][0].get<std::string>();
  ASSERT_TRUE(tensors.contains(output_name)) << "Convert output tensor not found: " << output_name;
  ASSERT_TRUE(tensors[output_name].is_object() && tensors[output_name].contains("data_type"));
  ASSERT_TRUE(tensors[output_name]["data_type"].is_number_unsigned());
  EXPECT_EQ(tensors[output_name]["data_type"].get<uint32_t>(), expected_data_type)
      << "Unexpected Convert output datatype in " << json_path;
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

}  // namespace test
}  // namespace onnxruntime
