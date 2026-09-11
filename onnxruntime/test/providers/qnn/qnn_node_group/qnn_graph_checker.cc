// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"

#include <fstream>

#include "QnnTypes.h"
#include "nlohmann/json.hpp"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

// Never throws: a missing/unreadable dump dir must surface as a gtest failure,
// not as an uncaught filesystem_error that terminates the whole test binary
// (which leaves no *.results.xml and fails the CI job with exit code 1).
bool FindQnnJsonGraph(const std::filesystem::path& dump_dir,
                      /*out*/ std::filesystem::path& json_path) {
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
        // Skip unreadable entries; a later entry may still be the graph dump.
        continue;
      }
    }
  } catch (const std::exception&) {
    return false;
  }
  return !json_path.empty();
}

// Never throws: a truncated/invalid JSON dump must surface as a gtest failure,
// not as an uncaught nlohmann::json exception that terminates the test binary.
bool ParseQnnJsonGraph(const std::filesystem::path& json_path,
                       /*out*/ nlohmann::json& root) {
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
  ASSERT_TRUE(FindQnnJsonGraph(dump_dir, json_path))
      << "No QNN JSON graph file found in " << dump_dir;

  nlohmann::json root;
  ASSERT_TRUE(ParseQnnJsonGraph(json_path, root))
      << "Failed to parse QNN JSON graph: " << json_path;

  ASSERT_TRUE(root.is_object() && root.contains("graph") && root["graph"].is_object() &&
              root["graph"].contains("nodes") && root["graph"]["nodes"].is_object())
      << "JSON missing 'graph.nodes' object in: " << json_path;

  size_t actual_count = 0;
  try {
    for (const auto& [node_name, node_json] : root["graph"]["nodes"].items()) {
      if (node_json.is_object() && node_json.contains("type") && node_json["type"].is_string() &&
          node_json["type"].get<std::string>() == op) {
        ++actual_count;
      }
    }
  } catch (const std::exception& ex) {
    FAIL() << "Failed to iterate QNN graph nodes in " << json_path << ": " << ex.what();
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
  ASSERT_TRUE(FindQnnJsonGraph(dump_dir, json_path))
      << "No QNN JSON graph file found in " << dump_dir;

  nlohmann::json root;
  ASSERT_TRUE(ParseQnnJsonGraph(json_path, root))
      << "Failed to parse QNN JSON graph: " << json_path;

  ASSERT_TRUE(root.is_object() && root.contains("graph") && root["graph"].is_object() &&
              root["graph"].contains("nodes") && root["graph"]["nodes"].is_object())
      << "JSON missing 'graph.nodes' object in: " << json_path;

  bool found = false;
  try {
    found = root["graph"]["nodes"].contains(node_name);
  } catch (const std::exception& ex) {
    FAIL() << "Failed to query QNN graph nodes in " << json_path << ": " << ex.what();
  }
  EXPECT_FALSE(found) << "Unexpected QNN node found: '" << node_name << "' in " << json_path;
}

namespace {

constexpr size_t kUnreadableDump = static_cast<size_t>(-1);

// Summed from "dims" rather than "params_count": dims is emitted for every tensor, while
// params_count is a stringified count present only when the dump omits static data.
// Never throws: any malformed field returns kUnreadableDump so the caller emits a gtest
// failure instead of terminating the test binary (which would leave no *.results.xml).
size_t SumFp32StaticBytes(const std::filesystem::path& dump_dir) {
  try {
    std::filesystem::path json_path;
    if (!FindQnnJsonGraph(dump_dir, json_path)) {
      return kUnreadableDump;
    }

    nlohmann::json root;
    if (!ParseQnnJsonGraph(json_path, root)) {
      return kUnreadableDump;
    }
    if (!root.is_object() || !root.contains("graph") || !root["graph"].is_object() ||
        !root["graph"].contains("tensors") || !root["graph"]["tensors"].is_object()) {
      return kUnreadableDump;
    }

    const int kStaticType = static_cast<int>(QNN_TENSOR_TYPE_STATIC);
    const int kFp32Type = static_cast<int>(QNN_DATATYPE_FLOAT_32);

    size_t total_bytes = 0;
    for (const auto& [name, tensor_json] : root["graph"]["tensors"].items()) {
      if (!tensor_json.is_object()) {
        continue;
      }
      // Guard with is_number(): value("type", -1) would throw if the dump ever
      // emits a string enum on some SDK/runner.
      if (!tensor_json.contains("type") || !tensor_json["type"].is_number() ||
          tensor_json["type"].get<int>() != kStaticType) {
        continue;
      }
      if (!tensor_json.contains("data_type") || !tensor_json["data_type"].is_number() ||
          tensor_json["data_type"].get<int>() != kFp32Type) {
        continue;
      }
      if (!tensor_json.contains("dims") || !tensor_json["dims"].is_array()) {
        return kUnreadableDump;
      }
      size_t num_elems = 1;
      for (const auto& dim : tensor_json["dims"]) {
        if (!dim.is_number_unsigned() && !dim.is_number_integer()) {
          return kUnreadableDump;
        }
        const long long dim_val = dim.get<long long>();
        if (dim_val < 0) {
          return kUnreadableDump;
        }
        const auto dim_u = static_cast<size_t>(dim_val);
        if (dim_u != 0 && num_elems > kUnreadableDump / dim_u) {
          // Would wrap (or collide with the kUnreadableDump sentinel): fail
          // gracefully instead of under-counting.
          return kUnreadableDump;
        }
        num_elems *= dim_u;
      }
      if (num_elems != 0 && sizeof(float) > kUnreadableDump / num_elems) {
        return kUnreadableDump;
      }
      const size_t add = num_elems * sizeof(float);
      if (total_bytes > kUnreadableDump - add) {
        return kUnreadableDump;
      }
      total_bytes += add;
      if (total_bytes == kUnreadableDump) {
        // Keep the sentinel reserved for "unreadable".
        return kUnreadableDump;
      }
    }
    return total_bytes;
  } catch (const std::exception&) {
    return kUnreadableDump;
  }
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
