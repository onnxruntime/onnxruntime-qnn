// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"

#include <fstream>

#include "QnnTypes.h"
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

namespace {

constexpr size_t kUnreadableDump = static_cast<size_t>(-1);

// Summed from "dims" rather than "params_count": dims is emitted for every tensor, while
// params_count is a stringified count present only when the dump omits static data.
size_t SumFp32StaticBytes(const std::filesystem::path& dump_dir) {
  std::filesystem::path json_path;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      json_path = entry.path();
      break;
    }
  }
  if (json_path.empty()) {
    return kUnreadableDump;
  }

  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    return kUnreadableDump;
  }

  nlohmann::json root;
  json_file >> root;
  if (!root.contains("graph") || !root["graph"].contains("tensors")) {
    return kUnreadableDump;
  }

  size_t total_bytes = 0;
  for (const auto& [name, tensor_json] : root["graph"]["tensors"].items()) {
    if (tensor_json.value("type", -1) != static_cast<int>(QNN_TENSOR_TYPE_STATIC) ||
        tensor_json.value("data_type", -1) != static_cast<int>(QNN_DATATYPE_FLOAT_32)) {
      continue;
    }
    size_t num_elems = 1;
    for (const auto& dim : tensor_json.value("dims", nlohmann::json::array())) {
      num_elems *= dim.get<size_t>();
    }
    total_bytes += num_elems * sizeof(float);
  }
  return total_bytes;
}

}  // namespace

void AssertFp32StaticBytesBelow(const std::filesystem::path& dump_dir, size_t max_bytes) {
  const size_t total_bytes = SumFp32StaticBytes(dump_dir);
  ASSERT_NE(total_bytes, kUnreadableDump) << "No readable QNN JSON graph in " << dump_dir;
  EXPECT_LE(total_bytes, max_bytes)
      << "FP32 STATIC bytes in the QNN graph exceed the budget: a large weight folded to FP32 "
         "instead of staying compact.";
}

void AssertFp32StaticBytesAbove(const std::filesystem::path& dump_dir, size_t min_bytes) {
  const size_t total_bytes = SumFp32StaticBytes(dump_dir);
  ASSERT_NE(total_bytes, kUnreadableDump) << "No readable QNN JSON graph in " << dump_dir;
  EXPECT_GT(total_bytes, min_bytes)
      << "FP32 STATIC bytes in the QNN graph are below the floor: the expected fold did not "
         "materialize.";
}

}  // namespace test
}  // namespace onnxruntime
