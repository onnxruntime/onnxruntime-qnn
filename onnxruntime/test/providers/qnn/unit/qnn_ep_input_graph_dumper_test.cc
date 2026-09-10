// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for DumpQnnEpInputGraphToJson
// (builder/qnn_ep_input_graph_dumper.cc).
//
// The dumper walks an EP-input OrtGraph through the public C++ wrappers
// (Ort::ConstGraph / ConstNode / ConstValueInfo) and serializes a
// QNN-Netron-schema JSON to disk. These tests drive it with the FakeGraph
// stub infrastructure (no real backend) and read the JSON back to assert the
// emitted structure — no coverage theater: every test validates an observable
// property of the output file or the bool return contract.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "nlohmann/json.hpp"

#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_ep_input_graph_dumper.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/unit/qnn_fake_ort_graph.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// QNN tensor-type integers the dumper emits (mirrors kQnnTensorType* in the
// implementation). Duplicated here so the test asserts against the schema
// contract rather than the implementation's private constants.
constexpr int kTypeAppWrite = QNN_TENSOR_TYPE_APP_WRITE;  // graph input  = 0
constexpr int kTypeAppRead = QNN_TENSOR_TYPE_APP_READ;    // graph output = 1
constexpr int kTypeNative = QNN_TENSOR_TYPE_NATIVE;       // intermediate = 3
constexpr int kTypeStatic = QNN_TENSOR_TYPE_STATIC;       // initializer  = 4

}  // namespace

// ---------------------------------------------------------------------------
// Fixture: owns a unique temp directory for output files and installs the
// FakeGraph OrtApi stubs + global-API override so the C++ graph wrappers route
// through the fakes instead of the real ORT runtime.
// ---------------------------------------------------------------------------
class QnnUnit_QnnEpInputGraphDumperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    InstallFakeGraphApiStubs(stub_ort_api_);
    // Unique per-test directory so parallel/serial runs never collide and each
    // test starts from a clean slate.
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    tmp_dir_ = std::filesystem::temp_directory_path() /
               (std::string("qnn_ep_input_graph_dumper_test_") + info->name() + "_" +
                std::to_string(static_cast<long long>(::getpid())));
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir_, ec);
    std::filesystem::create_directories(tmp_dir_, ec);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir_, ec);
  }

  std::filesystem::path OutPath(const std::string& file_name) const {
    return tmp_dir_ / file_name;
  }

  // Parse a written JSON file. Fails the test if the file cannot be opened.
  static nlohmann::json ReadJson(const std::filesystem::path& p) {
    std::ifstream ifs(p);
    EXPECT_TRUE(ifs.is_open()) << "could not open " << p.string();
    return nlohmann::json::parse(ifs);
  }

  OrtApi stub_ort_api_{};
  Ort::Logger logger_ = MakeNullLogger();
  std::filesystem::path tmp_dir_;
};

// ===========================================================================
// ClassifyTensorType — one branch per graph role.
// ===========================================================================

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, ClassifyTensorType_RequiredGraphInput_EmitsAppWriteType) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  x.is_required_graph_input = true;
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("required_input.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_EQ(j["graph"]["tensors"]["x"]["type"].get<int>(), kTypeAppWrite);
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, ClassifyTensorType_OptionalGraphInput_EmitsAppWriteType) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  x.is_optional_graph_input = true;
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("optional_input.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_EQ(j["graph"]["tensors"]["x"]["type"].get<int>(), kTypeAppWrite);
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, ClassifyTensorType_GraphOutput_EmitsAppReadType) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  y.is_graph_output = true;
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("graph_output.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_EQ(j["graph"]["tensors"]["y"]["type"].get<int>(), kTypeAppRead);
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, ClassifyTensorType_Intermediate_EmitsNativeType) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  // A tensor that is neither input, output, nor initializer (produced and
  // consumed inside the graph) classifies as NATIVE.
  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  x.is_required_graph_input = true;
  FakeValueInfo mid{"mid", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  y.is_graph_output = true;
  FakeNode n0{"id0", "Identity", "", 13, {&x}, {&mid}};
  FakeNode n1{"id1", "Identity", "", 13, {&mid}, {&y}};
  FakeGraph graph{{n0, n1}, {&x}, {&y}, {}};

  auto path = OutPath("intermediate.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_EQ(j["graph"]["tensors"]["mid"]["type"].get<int>(), kTypeNative);
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, ClassifyTensorType_InitializerThatIsAlsoInput_PrefersStatic) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  // An initializer that is also an optional graph input must classify as
  // STATIC (the ordering guarantee documented in ClassifyTensorType).
  FakeValueInfo w{"w", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  w.is_constant_initializer = true;
  w.is_optional_graph_input = true;
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode n{"id0", "Identity", "", 13, {&w}, {&y}};
  FakeGraph graph{{n}, {}, {&y}, {}};

  auto path = OutPath("static_over_input.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_EQ(j["graph"]["tensors"]["w"]["type"].get<int>(), kTypeStatic);
}

// ===========================================================================
// BuildTensorJson — data_type mapping, dims, dynamic shape, failures.
// ===========================================================================

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, BuildTensorJson_MappedDataType_EmitsQnnDataTypeAndDims) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  // DOUBLE maps to QNN_DATATYPE_FLOAT_64 — distinct from the FLOAT_32 fallback,
  // so a successful mapping is observable.
  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, {2, 3}};
  x.is_required_graph_input = true;
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, {2, 3}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("mapped_dtype.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  const auto& t = j["graph"]["tensors"]["x"];
  EXPECT_EQ(t["data_type"].get<int>(), static_cast<int>(QNN_DATATYPE_FLOAT_64));
  EXPECT_EQ(t["dims"], (nlohmann::json{2, 3}));
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, BuildTensorJson_UnmappedDataType_FallsBackToFloat32) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  // STRING is not in the non-quantized ONNX->QNN map, so the mapping helper
  // returns false and data_type falls back to FLOAT_32 (counted as a failure
  // that produces the aggregated WARNING).
  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, {2}};
  x.is_required_graph_input = true;
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("unmapped_dtype.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_EQ(j["graph"]["tensors"]["x"]["data_type"].get<int>(),
            static_cast<int>(QNN_DATATYPE_FLOAT_32));
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, BuildTensorJson_DynamicDim_EmitsEmptyDims) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  // A negative (dynamic) dimension makes the dumper emit an empty dims array
  // rather than a negative size.
  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 4}};
  x.is_required_graph_input = true;
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 4}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("dynamic_dim.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_TRUE(j["graph"]["tensors"]["x"]["dims"].empty());
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, BuildTensorJson_TypeInfoThrows_FallsBackToDefaults) {
  // Force value_info.TypeInfo() to throw: the catch(Ort::Exception) path must
  // leave data_type=FLOAT_32 and dims empty, and still write the file.
  stub_ort_api_.GetValueInfoTypeInfo =
      [](const OrtValueInfo*, const OrtTypeInfo** out) noexcept -> OrtStatus* {
    *out = nullptr;
    return Ort::GetApi().CreateStatus(ORT_FAIL, "no type info");
  };
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 3}};
  x.is_required_graph_input = true;
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 3}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("typeinfo_throws.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  const auto& t = j["graph"]["tensors"]["x"];
  EXPECT_EQ(t["data_type"].get<int>(), static_cast<int>(QNN_DATATYPE_FLOAT_32));
  EXPECT_TRUE(t["dims"].empty());
}

// ===========================================================================
// Node naming, disambiguation, and package/domain normalization.
// ===========================================================================

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpGraph_UnnamedNode_SynthesizesName) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode n{"", "Relu", "", 13, {&x}, {&y}};  // empty name at index 0
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("unnamed_node.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_TRUE(j["graph"]["nodes"].contains("unnamed_Relu_0"));
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpGraph_DuplicateNodeName_Disambiguates) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo m{"m", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode a{"dup", "Identity", "", 13, {&x}, {&m}};
  FakeNode b{"dup", "Identity", "", 13, {&m}, {&y}};  // same name, index 1
  FakeGraph graph{{a, b}, {&x}, {&y}, {}};

  auto path = OutPath("dup_node.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  const auto& nodes = j["graph"]["nodes"];
  EXPECT_TRUE(nodes.contains("dup"));
  EXPECT_TRUE(nodes.contains("dup__dup1"));
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpGraph_DefaultAndAiOnnxDomain_PackageIsOnnx) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo m{"m", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode empty_domain{"a", "Identity", "", 13, {&x}, {&m}};
  FakeNode ai_onnx{"b", "Identity", "ai.onnx", 13, {&m}, {&y}};
  FakeGraph graph{{empty_domain, ai_onnx}, {&x}, {&y}, {}};

  auto path = OutPath("onnx_domain.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_EQ(j["graph"]["nodes"]["a"]["package"].get<std::string>(), "onnx");
  EXPECT_EQ(j["graph"]["nodes"]["b"]["package"].get<std::string>(), "onnx");
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpGraph_CustomDomain_PackagePreserved) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode n{"a", "FusedThing", "com.microsoft", 1, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("custom_domain.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_EQ(j["graph"]["nodes"]["a"]["package"].get<std::string>(), "com.microsoft");
}

// ===========================================================================
// CollectNodeTensorNames — null optional inputs, empty names, dedup.
// ===========================================================================

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, CollectNodeTensorNames_NullOptionalInput_Skipped) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  // Second input is a null optional (not provided) — must be skipped.
  FakeNode n{"id0", "Clip", "", 13, {&x, nullptr}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("null_optional_input.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  const auto& inputs = j["graph"]["nodes"]["id0"]["input_names"];
  EXPECT_EQ(inputs.size(), 1u);
  EXPECT_EQ(inputs[0].get<std::string>(), "x");
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, CollectNodeTensorNames_EmptyName_Skipped) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo anon{"", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};  // empty name
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode n{"id0", "Clip", "", 13, {&x, &anon}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("empty_name.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  const auto& inputs = j["graph"]["nodes"]["id0"]["input_names"];
  EXPECT_EQ(inputs.size(), 1u);
  EXPECT_EQ(inputs[0].get<std::string>(), "x");
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, CollectNodeTensorNames_SharedTensor_DedupedInTensors) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  x.is_required_graph_input = true;
  FakeValueInfo mid{"mid", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  y.is_graph_output = true;
  // "mid" is the output of n0 and the input of n1 — referenced twice.
  FakeNode n0{"n0", "Identity", "", 13, {&x}, {&mid}};
  FakeNode n1{"n1", "Identity", "", 13, {&mid}, {&y}};
  FakeGraph graph{{n0, n1}, {&x}, {&y}, {}};

  auto path = OutPath("shared_tensor.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  // "mid" appears in both node name lists but exactly once in tensors.
  EXPECT_EQ(j["graph"]["nodes"]["n0"]["output_names"][0].get<std::string>(), "mid");
  EXPECT_EQ(j["graph"]["nodes"]["n1"]["input_names"][0].get<std::string>(), "mid");
  EXPECT_TRUE(j["graph"]["tensors"].contains("mid"));
}

// ===========================================================================
// Initializer gathering.
// ===========================================================================

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpGraph_Initializer_AppearsAsStaticTensor) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  x.is_required_graph_input = true;
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo w{"w", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  w.is_constant_initializer = true;
  FakeNode n{"id0", "Add", "", 14, {&x}, {&y}};
  // w is only present as a graph initializer, not referenced by any node input.
  FakeGraph graph{{n}, {&x}, {&y}, {&w}};

  auto path = OutPath("initializer.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  ASSERT_TRUE(j["graph"]["tensors"].contains("w"));
  EXPECT_EQ(j["graph"]["tensors"]["w"]["type"].get<int>(), kTypeStatic);
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpGraph_NullAndDuplicateInitializers_Skipped) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  x.is_required_graph_input = true;
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  // w is referenced as a node input AND listed as an initializer -> the
  // initializer loop must skip it (already seen). A null initializer entry
  // must also be skipped.
  FakeValueInfo w{"w", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  w.is_constant_initializer = true;
  FakeNode n{"id0", "Add", "", 14, {&x, &w}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {&w, nullptr}};

  auto path = OutPath("dup_null_initializer.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_TRUE(j["graph"]["tensors"].contains("w"));
  // No crash / no phantom keys from the null entry.
  EXPECT_FALSE(j["graph"]["tensors"].contains(""));
}

// ===========================================================================
// Top-level structure and op_types.
// ===========================================================================

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpGraph_Basic_WritesExpectedTopLevelSchema) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  x.is_required_graph_input = true;
  FakeValueInfo m{"m", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  y.is_graph_output = true;
  // Two different op types plus a duplicate — op_types must be sorted + unique.
  FakeNode n0{"n0", "Relu", "", 13, {&x}, {&m}};
  FakeNode n1{"n1", "Identity", "", 13, {&m}, {&y}};
  FakeNode n2{"n2", "Relu", "", 13, {&y}, {&m}};
  FakeGraph graph{{n0, n1, n2}, {&x}, {&y}, {}};

  auto path = OutPath("basic.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));

  nlohmann::json j = ReadJson(path);
  EXPECT_EQ(j["model.cpp"].get<std::string>(), "N/A");
  EXPECT_EQ(j["model.bin"].get<std::string>(), "N/A");
  EXPECT_TRUE(j.contains("copyright_str"));
  EXPECT_TRUE(j["graph"].contains("tensors"));
  EXPECT_TRUE(j["graph"].contains("nodes"));
  // Deduped + sorted op types.
  EXPECT_EQ(j["op_types"], (nlohmann::json{"Identity", "Relu"}));
}

// ===========================================================================
// File-write paths: success, nested-dir creation, and soft-fail returns.
// ===========================================================================

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpToJson_NestedParentDir_CreatedAndReturnsTrue) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  // Parent directory does not exist yet; the dumper must create it.
  auto path = tmp_dir_ / "nested" / "deeper" / "out.json";
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));
  EXPECT_TRUE(std::filesystem::exists(path));
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpToJson_ExistingFile_OverwrittenAndReturnsTrue) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  auto path = OutPath("overwrite.json");
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));
  // Second call finds the file present -> overwrite-warning branch, still true.
  ASSERT_TRUE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));
  EXPECT_TRUE(std::filesystem::exists(path));
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpToJson_ParentPathIsRegularFile_ReturnsFalse) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  // Create a regular file, then ask to write under it: create_directories on
  // "<file>/sub" fails, so the dumper returns false.
  auto blocker = OutPath("blocker");
  {
    std::ofstream ofs(blocker);
    ofs << "x";
  }
  auto path = blocker / "sub" / "out.json";
  EXPECT_FALSE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), path, logger_));
}

TEST_F(QnnUnit_QnnEpInputGraphDumperTest, DumpToJson_OutputPathIsDirectory_ReturnsFalse) {
  OrtGlobalApiOverride api_override(&stub_ort_api_);

  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4}};
  FakeNode n{"id0", "Identity", "", 13, {&x}, {&y}};
  FakeGraph graph{{n}, {&x}, {&y}, {}};

  // output_path is an existing directory: parent already exists, the exists()
  // overwrite-warning fires, and the ofstream open fails -> false.
  auto dir_path = OutPath("iam_a_dir");
  std::error_code ec;
  std::filesystem::create_directories(dir_path, ec);
  EXPECT_FALSE(qnn::DumpQnnEpInputGraphToJson(graph.AsGraph(), dir_path, logger_));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
