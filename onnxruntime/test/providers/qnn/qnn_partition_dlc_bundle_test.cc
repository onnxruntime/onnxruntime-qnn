// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

namespace fs = std::filesystem;

class ScopedTempDir {
 public:
  ScopedTempDir() {
    auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto sanitize = [](std::string s) {
      for (char& c : s) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' ||
            c == '"' || c == '<' || c == '>' || c == '|') {
          c = '_';
        }
      }
      return s;
    };
    path_ = fs::temp_directory_path() /
            (sanitize(info->test_suite_name()) + "_" + sanitize(info->name()));
    std::error_code ec;
    fs::remove_all(path_, ec);
    fs::create_directories(path_);
  }
  ~ScopedTempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

TEST_F(QnnCPUBackendTests, PartitionDlcBundle_DisabledByDefault) {
  ScopedTempDir tmp;
  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "0";
  opts["partition_dlc_bundle_dir"] = tmp.path().string();

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQnnModelTest(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data),
                              TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      opts, 13, ExpectedEPNodeAssignment::All);

  EXPECT_FALSE(fs::exists(tmp.path() / "manifest.json"))
      << "Bundle must not be produced unless dump_partition_dlc_bundle=1";
}

TEST_F(QnnCPUBackendTests, PartitionDlcBundle_MissingDirRejected) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "0";
  opts["dump_partition_dlc_bundle"] = "1";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  EXPECT_ANY_THROW(
      RunQnnModelTest(
          BuildOpTestCase<float>("add_node", "Add",
                                 {TestInputDef<float>({1, 2, 3}, false, data),
                                  TestInputDef<float>({1, 2, 3}, false, data)},
                                 {}, {}, kOnnxDomain),
          opts, 13, ExpectedEPNodeAssignment::All));
}

// IR-backend serialization is forced when bundle is enabled; Run() therefore
// throws (same as dump_qnn_ir_dlc). Bundle is written during compile, before.
TEST_F(QnnCPUBackendTests, PartitionDlcBundle_SinglePartition) {
  ScopedTempDir tmp;
  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "0";
  opts["dump_partition_dlc_bundle"] = "1";
  opts["partition_dlc_bundle_dir"] = tmp.path().string();

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  try {
    RunQnnModelTest(
        BuildOpTestCase<float>("add_node", "Add",
                               {TestInputDef<float>({1, 2, 3}, false, data),
                                TestInputDef<float>({1, 2, 3}, false, data)},
                               {}, {}, kOnnxDomain),
        opts, 13, ExpectedEPNodeAssignment::All);
  } catch (const std::exception&) {
    // Expected: IR backend cannot execute. We only care about compile-time emission.
  }

  fs::path manifest_path = tmp.path() / "manifest.json";
  ASSERT_TRUE(fs::exists(manifest_path)) << "manifest.json must be written";

  std::ifstream ifs(manifest_path);
  auto j = nlohmann::json::parse(ifs, nullptr, false);
  ASSERT_FALSE(j.is_discarded());
  ASSERT_TRUE(j.contains("partitions"));
  ASSERT_EQ(j["partitions"].size(), 1u);
  const auto& p0 = j["partitions"][0];
  EXPECT_TRUE(p0.contains("name"));
  EXPECT_TRUE(p0.contains("dlc_path"));
  ASSERT_TRUE(p0.contains("inputs"));
  ASSERT_TRUE(p0.contains("outputs"));
  EXPECT_GT(p0["inputs"].size(), 0u);
  EXPECT_GT(p0["outputs"].size(), 0u);
  for (const auto& t : p0["inputs"]) {
    EXPECT_TRUE(t.contains("name"));
    EXPECT_TRUE(t.contains("dtype"));
    EXPECT_TRUE(t.contains("shape"));
  }

  fs::path dlc_path = tmp.path() / p0["dlc_path"].get<std::string>();
  EXPECT_TRUE(fs::exists(dlc_path)) << "DLC file must be written at " << dlc_path;

  EXPECT_EQ(j["edges"].size(), 0u) << "Single partition has no inter-partition edges";
}

TEST_F(QnnCPUBackendTests, PartitionDlcBundle_MultiPartition) {
  ScopedTempDir tmp;
  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "0";
  opts["dump_partition_dlc_bundle"] = "1";
  opts["partition_dlc_bundle_dir"] = tmp.path().string();

  auto build_model = [](ModelTestBuilder& builder) {
    std::vector<float> data = GetFloatDataInRange(-1.0f, 1.0f, 4);
    builder.MakeInput<float>("in0", {2, 2}, data);
    builder.MakeInput<float>("in1", {2, 2}, data);
    builder.AddNode("add", "Add", {"in0", "in1"}, {"add_out"});
    builder.AddNode("trilu", "Trilu", {"add_out"}, {"trilu_out"});
    builder.MakeOutput("Y");
    builder.AddNode("relu", "Relu", {"trilu_out"}, {"Y"});
  };

  try {
    RunQnnModelTest(build_model, opts, 14, ExpectedEPNodeAssignment::Some);
  } catch (const std::exception&) {
  }

  fs::path manifest_path = tmp.path() / "manifest.json";
  ASSERT_TRUE(fs::exists(manifest_path));
  std::ifstream ifs(manifest_path);
  auto j = nlohmann::json::parse(ifs, nullptr, false);
  ASSERT_FALSE(j.is_discarded());
  ASSERT_GE(j["partitions"].size(), 2u) << "Expected at least 2 QNN partitions";
  for (const auto& p : j["partitions"]) {
    fs::path dlc = tmp.path() / p["dlc_path"].get<std::string>();
    EXPECT_TRUE(fs::exists(dlc)) << "DLC missing for partition " << p["name"];
  }
}

// Fan-out: QNN-A output is consumed by a CPU op AND directly by QNN-B.
// edges only records direct QNN→QNN tensor handoffs (not QNN→CPU→QNN chains),
// so this is the topology that exercises the edge-builder with a non-empty result.
TEST_F(QnnCPUBackendTests, PartitionDlcBundle_DirectQnnToQnnEdge) {
  ScopedTempDir tmp;
  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "0";
  opts["dump_partition_dlc_bundle"] = "1";
  opts["partition_dlc_bundle_dir"] = tmp.path().string();

  auto build_model = [](ModelTestBuilder& builder) {
    std::vector<float> data = GetFloatDataInRange(-1.0f, 1.0f, 4);
    builder.MakeInput<float>("in0", {2, 2}, data);
    builder.MakeInput<float>("in1", {2, 2}, data);
    builder.AddNode("add1", "Add", {"in0", "in1"}, {"a"});
    builder.AddNode("trilu", "Trilu", {"a"}, {"t"});
    builder.MakeOutput("Y");
    builder.AddNode("add2", "Add", {"a", "t"}, {"Y"});
  };

  try {
    RunQnnModelTest(build_model, opts, 14, ExpectedEPNodeAssignment::Some);
  } catch (const std::exception&) {
  }

  fs::path manifest_path = tmp.path() / "manifest.json";
  ASSERT_TRUE(fs::exists(manifest_path));
  std::ifstream ifs(manifest_path);
  auto j = nlohmann::json::parse(ifs, nullptr, false);
  ASSERT_FALSE(j.is_discarded());
  ASSERT_GE(j["partitions"].size(), 2u) << "Expected at least 2 QNN partitions";
  ASSERT_TRUE(j.contains("edges"));
  ASSERT_GE(j["edges"].size(), 1u) << "Expected a direct QNN→QNN edge for tensor 'a'";

  std::unordered_map<std::string, std::set<std::string>> partition_outputs;
  std::unordered_map<std::string, std::set<std::string>> partition_inputs;
  for (const auto& p : j["partitions"]) {
    const std::string name = p["name"].get<std::string>();
    for (const auto& t : p["outputs"]) partition_outputs[name].insert(t["name"].get<std::string>());
    for (const auto& t : p["inputs"]) partition_inputs[name].insert(t["name"].get<std::string>());
  }
  bool found_a_edge = false;
  for (const auto& e : j["edges"]) {
    const auto producer = e["producer_partition"].get<std::string>();
    const auto consumer = e["consumer_partition"].get<std::string>();
    const auto tensor = e["tensor_name"].get<std::string>();
    EXPECT_NE(producer, consumer) << "edge endpoints must be distinct partitions";
    EXPECT_TRUE(partition_outputs[producer].count(tensor))
        << "edge tensor '" << tensor << "' must be an output of producer '" << producer << "'";
    EXPECT_TRUE(partition_inputs[consumer].count(tensor))
        << "edge tensor '" << tensor << "' must be an input of consumer '" << consumer << "'";
    if (tensor == "a") found_a_edge = true;
  }
  EXPECT_TRUE(found_a_edge) << "Expected a direct edge carrying tensor 'a' between the two QNN Add partitions";
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
