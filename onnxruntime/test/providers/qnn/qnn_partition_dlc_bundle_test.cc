// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <fstream>
#include <string>

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

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
