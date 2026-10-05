// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"

#include <fstream>
#include <limits>
#include <map>
#include <utility>

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
  // Safety net: TestQDQModelAccuracy/RunQnnModelTest issue GTEST_SKIP() from a
  // helper on unsupported HTP arch/devices. That skip marks the test but does
  // not return from the caller's TEST_F body, so a caller that forgot its own
  // `if (IsSkipped()) return;` guard would otherwise fail spuriously on the
  // missing JSON dump. Propagate the skip instead (matches conv/matmul/
  // resize/softmax call-site guards).
  if (::testing::Test::IsSkipped()) {
    GTEST_SKIP() << "Skipped: no QNN graph dump was produced (test was already skipped).";
  }
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
  // Same skip-propagation safety net as AssertOpInQnnGraph above.
  if (::testing::Test::IsSkipped()) {
    GTEST_SKIP() << "Skipped: no QNN graph dump was produced (test was already skipped).";
  }
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

void AssertTensorShapeInQnnGraph(const std::filesystem::path& dump_dir,
                                 const std::string& tensor_name,
                                 const std::vector<uint32_t>& expected_dims) {
  std::filesystem::path json_path;
  ASSERT_TRUE(FindQnnJsonGraph(dump_dir, json_path))
      << "No QNN JSON graph file found in " << dump_dir;

  nlohmann::json root;
  ASSERT_TRUE(ParseQnnJsonGraph(json_path, root))
      << "Failed to parse QNN JSON graph: " << json_path;

  ASSERT_TRUE(root.is_object() && root.contains("graph") && root["graph"].is_object() &&
              root["graph"].contains("tensors") && root["graph"]["tensors"].is_object())
      << "JSON missing 'graph.tensors' object in: " << json_path;

  bool has_tensor = false;
  try {
    has_tensor = root["graph"]["tensors"].contains(tensor_name);
  } catch (const std::exception& ex) {
    FAIL() << "Failed to query QNN graph tensors in " << json_path << ": " << ex.what();
  }
  ASSERT_TRUE(has_tensor) << "QNN tensor '" << tensor_name << "' not found in " << json_path;

  std::vector<uint32_t> actual_dims;
  try {
    auto it = root["graph"]["tensors"].find(tensor_name);
    if (it == root["graph"]["tensors"].end()) {
      FAIL() << "QNN tensor '" << tensor_name << "' not found in " << json_path;
    }
    const auto& tensor_json = *it;
    if (!tensor_json.is_object() || !tensor_json.contains("dims") || !tensor_json["dims"].is_array()) {
      FAIL() << "QNN tensor '" << tensor_name << "' missing 'dims' array in " << json_path;
    }
    for (const auto& dim : tensor_json["dims"]) {
      if (!dim.is_number_unsigned() && !dim.is_number_integer()) {
        FAIL() << "QNN tensor '" << tensor_name << "' has non-integer dim in " << json_path;
      }
      long long dim_val = 0;
      try {
        dim_val = dim.get<long long>();
      } catch (const std::exception& ex) {
        FAIL() << "Failed to read dim of QNN tensor '" << tensor_name << "' in " << json_path << ": "
               << ex.what();
      }
      if (dim_val < 0 ||
          dim_val > static_cast<long long>(std::numeric_limits<uint32_t>::max())) {
        FAIL() << "QNN tensor '" << tensor_name << "' has out-of-range dim " << dim_val << " in " << json_path;
      }
      actual_dims.push_back(static_cast<uint32_t>(dim_val));
    }
  } catch (const std::exception& ex) {
    FAIL() << "Failed to read dims of QNN tensor '" << tensor_name << "' in " << json_path << ": " << ex.what();
  }

  EXPECT_EQ(actual_dims, expected_dims)
      << "QNN tensor '" << tensor_name << "': expected shape mismatch in " << json_path;
}

void AssertNodeInputsDistinctInQnnGraph(const std::filesystem::path& dump_dir,
                                        const std::string& op,
                                        size_t input_index) {
  // Same skip-propagation safety net as AssertOpInQnnGraph above.
  if (::testing::Test::IsSkipped()) {
    GTEST_SKIP() << "Skipped: no QNN graph dump was produced (test was already skipped).";
  }
  std::filesystem::path json_path;
  ASSERT_TRUE(FindQnnJsonGraph(dump_dir, json_path))
      << "No QNN JSON graph file found in " << dump_dir;

  nlohmann::json root;
  ASSERT_TRUE(ParseQnnJsonGraph(json_path, root))
      << "Failed to parse QNN JSON graph: " << json_path;

  ASSERT_TRUE(root.is_object() && root.contains("graph") && root["graph"].is_object() &&
              root["graph"].contains("nodes") && root["graph"]["nodes"].is_object())
      << "JSON missing 'graph.nodes' object in: " << json_path;

  std::map<std::string, std::string> input_to_node;
  try {
    for (const auto& [node_name, node_json] : root["graph"]["nodes"].items()) {
      if (!node_json.is_object() || !node_json.contains("type") || !node_json["type"].is_string() ||
          node_json["type"].get<std::string>() != op) {
        continue;
      }

      ASSERT_TRUE(node_json.contains("input_names") && node_json["input_names"].is_array() &&
                  node_json["input_names"].size() > input_index)
          << "QNN node '" << node_name << "' of type '" << op << "' has only "
          << (node_json.contains("input_names") && node_json["input_names"].is_array()
                  ? node_json["input_names"].size()
                  : 0)
          << " input(s) in " << json_path;

      const auto& input_entry = node_json["input_names"][input_index];
      ASSERT_TRUE(input_entry.is_string())
          << "QNN node '" << node_name << "' of type '" << op << "' has non-string input at index "
          << input_index << " in " << json_path;
      const std::string input_name = input_entry.get<std::string>();
      const auto [it, inserted] = input_to_node.emplace(input_name, node_name);
      EXPECT_TRUE(inserted)
          << "QNN nodes '" << it->second << "' and '" << node_name << "' (type '" << op
          << "') both read tensor '" << input_name << "' at input " << input_index << " in " << json_path;
    }
  } catch (const std::exception& ex) {
    FAIL() << "Failed to iterate QNN graph nodes in " << json_path << ": " << ex.what();
  }
}

}  // namespace test
}  // namespace onnxruntime
