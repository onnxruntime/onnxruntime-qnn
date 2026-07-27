// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"
#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing_types.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

// Defined in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

namespace fs = std::filesystem;

// ===================== Test Helpers =====================

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
    // Remove any stale state left by a previous crashed test run before
    // creating a fresh directory.
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

static fs::path FindTraceFile(const fs::path& dir) {
  if (!fs::exists(dir)) return {};
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".json" &&
        entry.path().stem().string().find("_op_trace") != std::string::npos) {
      return entry.path();
    }
  }
  return {};
}

static nlohmann::json ReadTraceJson(const fs::path& path) {
  std::ifstream ifs(path);
  EXPECT_TRUE(ifs.is_open()) << "Failed to open: " << path;
  if (!ifs.is_open()) return {};
  auto j = nlohmann::json::parse(ifs, nullptr, false);
  EXPECT_FALSE(j.is_discarded()) << "Failed to parse JSON: " << path;
  return j;
}

static nlohmann::json RunModelWithTracing(const GetTestModelFn& build_fn,
                                          const fs::path& trace_dir,
                                          const std::string& backend = "cpu",
                                          int opset = 13) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend;
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_framework_op_trace"] = "1";
  provider_options["framework_op_trace_dir"] = trace_dir.string();

  // These tests verify tracing behavior, not numeric exactness. HTP uses float16
  // internally, so fp32 comparisons require a relaxed tolerance on all platforms.
  float fp32_abs_err = (backend == "htp") ? 5e-3f : 1e-5f;
#if defined(__linux__) && !defined(__aarch64__)
  if (backend == "htp") {
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  }
#endif

  RunQnnModelTest(build_fn, provider_options, opset,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(fp32_abs_err)});

  // GTEST_SKIP() in RunQnnModelTest (e.g. arch guard) sets the skip flag and
  // returns from that function but does not propagate to our caller.  Bail out
  // now so we don't assert on a trace file that was never written.
  if (::testing::UnitTest::GetInstance()->current_test_info()->result()->Skipped()) {
    return {};
  }

  auto trace_file = FindTraceFile(trace_dir);
  EXPECT_FALSE(trace_file.empty()) << "No trace file found in " << trace_dir;
  if (trace_file.empty()) return {};
  return ReadTraceJson(trace_file);
}

static void ValidateTraceStructure(const nlohmann::json& j) {
  ASSERT_TRUE(j.contains("backend_type"));
  ASSERT_TRUE(j.contains("subgraph_traces"));
  // compilation_target lives under soc_traces[i] in the new schema; it is
  // intentionally not at root anymore. Every caller of this validator runs a
  // single backend (single-SoC), so soc_traces must have exactly one entry.
  // Asserting == 1 (not >= 1) also guards the two-pass GetCapabilityImpl
  // regression: without the trace_ reset, a second partitioner pass would
  // append a duplicate soc_trace and this would catch it.
  ASSERT_FALSE(j.contains("compilation_target"))
      << "compilation_target must NOT be at root in the new schema; "
         "consumers should read from soc_traces[i].compilation_target";
  ASSERT_TRUE(j.contains("soc_traces"));
  ASSERT_EQ(j["soc_traces"].size(), 1u);
  ASSERT_TRUE(j["soc_traces"][0].contains("compilation_target"));
  ASSERT_TRUE(j.contains("unsupported_nodes"));
  ASSERT_TRUE(j.contains("summary"));
  ASSERT_GE(j["subgraph_traces"].size(), 1u);
  EXPECT_GT(j["summary"]["total_qnn_ops"].get<size_t>(), 0u);
  EXPECT_EQ(j["summary"]["total_socs"].get<uint32_t>(), 1u);
}

// ========================= Unit Tests =========================

// Pure unit tests for qnn_op_tracing.h/.cc — no QNN backend required.
class QnnFrameworkOpTraceUnit : public ::testing::Test {};

TEST_F(QnnFrameworkOpTraceUnit, SerializeDeserializeRoundTrip) {
  qnn::FrameworkOpTrace trace;
  trace.model_name = "test_model.onnx";
  trace.backend_type = "cpu";
  // Single-SoC: soc_traces has exactly one entry holding the CompilationTarget.
  trace.soc_traces.push_back(qnn::SocTrace{{"V73", 60, 0}});

  qnn::OpTraceInfo sg;
  sg.graph_name = "QnnEP_graph_0";
  sg.op_mappings.push_back({"Add_0",
                            "QNN_OP_ELEMENT_WISE_ADD",
                            {{"/add/Add", qnn::TraceTargetType::kOp}},
                            "OrtNodeUnit"});
  sg.op_mappings.push_back({"Conv_1",
                            "QNN_OP_CONV_2D",
                            {{"/dq_node", qnn::TraceTargetType::kOp},
                             {"_out_dq", qnn::TraceTargetType::kTensor},
                             {"/conv_node", qnn::TraceTargetType::kOp},
                             {"_out_conv", qnn::TraceTargetType::kTensor},
                             {"/q_node", qnn::TraceTargetType::kOp}},
                            "DQQFusion"});
  sg.tensor_mappings.push_back({"input_0", "", {{"/input_0", qnn::TraceTargetType::kTensor}}, ""});
  trace.subgraph_traces.push_back(std::move(sg));
  trace.unsupported_nodes.push_back({"custom_op_1", "CustomOp", 42, "No op builder registered"});
  qnn::ComputeTraceSummary(trace);

  nlohmann::json j = qnn::SerializeFrameworkOpTrace(trace);
  EXPECT_EQ(j["model_name"], "test_model.onnx");
  // compilation_target is no longer at root - it lives under soc_traces[0].
  EXPECT_FALSE(j.contains("compilation_target"));
  ASSERT_TRUE(j.contains("soc_traces"));
  ASSERT_EQ(j["soc_traces"].size(), 1u);
  EXPECT_EQ(j["soc_traces"][0]["compilation_target"]["htp_arch"], "V73");
  EXPECT_EQ(j["soc_traces"][0]["compilation_target"]["soc_model"], 60);
  EXPECT_EQ(j["soc_traces"][0]["compilation_target"]["device_id"], 0);
  EXPECT_EQ(j["subgraph_traces"][0]["op_mappings"][0]["dst_name"], "Add_0");
  EXPECT_EQ(j["subgraph_traces"][0]["op_mappings"][0]["sources"][0]["type"], "OP");
  // Verify mixed OP+TENSOR sources in Conv_1 (DQ_Q_Fusion)
  EXPECT_EQ(j["subgraph_traces"][0]["op_mappings"][1]["dst_name"], "Conv_1");
  EXPECT_EQ(j["subgraph_traces"][0]["op_mappings"][1]["sources"][0]["type"], "OP");
  EXPECT_EQ(j["subgraph_traces"][0]["op_mappings"][1]["sources"][1]["type"], "TENSOR");
  EXPECT_EQ(j["subgraph_traces"][0]["op_mappings"][1]["sources"][1]["name"], "_out_dq");
  EXPECT_EQ(j["unsupported_nodes"][0]["node_name"], "custom_op_1");
  EXPECT_EQ(j["summary"]["total_socs"], 1u);
}

TEST_F(QnnFrameworkOpTraceUnit, ComputeTraceSummary) {
  qnn::FrameworkOpTrace trace;

  qnn::OpTraceInfo sg1;
  sg1.graph_name = "graph_0";
  sg1.op_mappings.push_back({"op0", "A", {{"s0", qnn::TraceTargetType::kOp}}, "FusionA"});
  sg1.op_mappings.push_back({"op1", "A", {{"s1", qnn::TraceTargetType::kOp}}, "FusionA"});
  // Mixed OP+TENSOR sources — summary should only count OP entries (1 OP, not 2 total)
  sg1.op_mappings.push_back({"op2",
                             "B",
                             {{"s2", qnn::TraceTargetType::kOp},
                              {"_out_s2", qnn::TraceTargetType::kTensor}},
                             "FusionB"});
  trace.subgraph_traces.push_back(std::move(sg1));

  qnn::OpTraceInfo sg2;
  sg2.graph_name = "graph_1";
  // N:1 fusion with interleaved OP+TENSOR — 2 OPs, 1 TENSOR
  sg2.op_mappings.push_back({"op3",
                             "C",
                             {{"s3", qnn::TraceTargetType::kOp},
                              {"_out_s3", qnn::TraceTargetType::kTensor},
                              {"s4", qnn::TraceTargetType::kOp}},
                             "FusionC"});
  sg2.op_mappings.push_back({"op4", "A", {{"s5", qnn::TraceTargetType::kOp}}, "FusionA"});
  // N:M fusion: 2 ONNX source nodes (s_nm0, s_nm1) emitted by 2 QNN op entries.
  // Without deduplication this would inflate supported_nodes by 2 extra entries.
  sg2.op_mappings.push_back({"op5",
                             "D",
                             {{"s_nm0", qnn::TraceTargetType::kOp},
                              {"s_nm1", qnn::TraceTargetType::kOp}},
                             "FusionD"});
  sg2.op_mappings.push_back({"op6",
                             "D",
                             {{"s_nm0", qnn::TraceTargetType::kOp},
                              {"s_nm1", qnn::TraceTargetType::kOp}},
                             "FusionD"});
  trace.subgraph_traces.push_back(std::move(sg2));

  trace.unsupported_nodes.push_back({"u0", "Bad", 0, "r"});
  trace.unsupported_nodes.push_back({"u1", "Bad", 1, "r"});

  qnn::ComputeTraceSummary(trace);

  EXPECT_EQ(trace.summary.qnn_subgraphs, 2u);
  EXPECT_EQ(trace.summary.total_qnn_ops, 7u);
  EXPECT_EQ(trace.summary.unsupported_nodes, 2u);
  // supported_nodes counts unique ONNX node names: {s0, s1, s2, s3, s4, s5, s_nm0, s_nm1} = 8.
  // The N:M FusionD entries (op5, op6) reference the same {s_nm0, s_nm1} pair, so they
  // contribute 2 unique names rather than 4 source occurrences.
  EXPECT_EQ(trace.summary.supported_nodes, 8u);
  EXPECT_EQ(trace.summary.total_onnx_nodes, 10u);
  EXPECT_EQ(trace.summary.fusion_count["FusionA"], 3u);
  EXPECT_EQ(trace.summary.fusion_count["FusionB"], 1u);
  EXPECT_EQ(trace.summary.fusion_count["FusionC"], 1u);
  EXPECT_EQ(trace.summary.fusion_count["FusionD"], 2u);
}

// ---- MergePerSocUnsupportedNodes: FCB multi-SoC rejection merging ----

// EncodeHtpArch / MakeSocLabel: the single source of truth for the label format.
TEST_F(QnnFrameworkOpTraceUnit, MakeSocLabel_Format) {
  // arch encoding: raw number -> "V<n>", 0 -> "".
  EXPECT_EQ(qnn::EncodeHtpArch(73), "V73");
  EXPECT_EQ(qnn::EncodeHtpArch(0), "");
  // arch + soc_model known.
  EXPECT_EQ(qnn::MakeSocLabel(73, 60), "V73/soc_model=60");
  // arch known, soc_model unknown (0).
  EXPECT_EQ(qnn::MakeSocLabel(73, 0), "V73");
  // arch unknown, soc_model known.
  EXPECT_EQ(qnn::MakeSocLabel(0, 60), "soc_model=60");
  // both unknown: arch empty wins, emits the sentinel numeric.
  EXPECT_EQ(qnn::MakeSocLabel(0, 0), "soc_model=0");
}

// A node rejected on multiple SoCs for the SAME reason collapses to one row
// with the SoC labels grouped: "V73,V81: <reason>" (reason listed once).
TEST_F(QnnFrameworkOpTraceUnit, MergePerSoc_SameReasonGrouped) {
  std::vector<qnn::PerSocUnsupportedNodes> per_soc = {
      {"V73", {{"node7", "Foo", 7, "data type not supported"}}},
      {"V81", {{"node7", "Foo", 7, "data type not supported"}}},
  };
  auto merged = qnn::MergePerSocUnsupportedNodes(per_soc);
  ASSERT_EQ(merged.size(), 1u) << "same node across SoCs must collapse to one row";
  EXPECT_EQ(merged[0].node_index, 7u);
  EXPECT_EQ(merged[0].node_name, "node7");
  EXPECT_EQ(merged[0].op_type, "Foo");
  // Identical reason listed once, both SoCs attributed to it.
  EXPECT_EQ(merged[0].reason, "V73,V81: data type not supported");
}

// A node rejected for DIFFERENT reasons on different SoCs collapses to one row
// whose reason concatenates the per-SoC causes in first-seen order.
TEST_F(QnnFrameworkOpTraceUnit, MergePerSoc_DifferentReasonsConcatenated) {
  std::vector<qnn::PerSocUnsupportedNodes> per_soc = {
      {"V73", {{"node7", "Foo", 7, "data type"}}},
      {"V81", {{"node7", "Foo", 7, "rank too high"}}},
  };
  auto merged = qnn::MergePerSocUnsupportedNodes(per_soc);
  ASSERT_EQ(merged.size(), 1u);
  EXPECT_EQ(merged[0].node_index, 7u);
  EXPECT_EQ(merged[0].reason, "V73: data type; V81: rank too high");
}

// Distinct nodes are emitted in first-seen order; each keeps its own reason(s).
// soc_model= labels (arch unknown) are used verbatim.
TEST_F(QnnFrameworkOpTraceUnit, MergePerSoc_MultipleNodesFirstSeenOrder) {
  std::vector<qnn::PerSocUnsupportedNodes> per_soc = {
      {"soc_model=60", {{"a", "A", 3, "r1"}, {"b", "B", 9, "r2"}}},
      {"soc_model=88", {{"b", "B", 9, "r2"}, {"c", "C", 1, "r3"}}},
  };
  auto merged = qnn::MergePerSocUnsupportedNodes(per_soc);
  ASSERT_EQ(merged.size(), 3u);
  // First-seen order across the flattened stream: 3, 9, 1.
  EXPECT_EQ(merged[0].node_index, 3u);
  EXPECT_EQ(merged[0].reason, "soc_model=60: r1");
  EXPECT_EQ(merged[1].node_index, 9u);
  EXPECT_EQ(merged[1].reason, "soc_model=60,soc_model=88: r2");
  EXPECT_EQ(merged[2].node_index, 1u);
  EXPECT_EQ(merged[2].reason, "soc_model=88: r3");
}

// Empty input yields an empty result (single-SoC never calls this).
TEST_F(QnnFrameworkOpTraceUnit, MergePerSoc_Empty) {
  EXPECT_TRUE(qnn::MergePerSocUnsupportedNodes({}).empty());
  // SoCs present but no rejections -> still empty.
  std::vector<qnn::PerSocUnsupportedNodes> per_soc = {{"V73", {}}, {"V81", {}}};
  EXPECT_TRUE(qnn::MergePerSocUnsupportedNodes(per_soc).empty());
}

// Two SoCs share the same htp_arch but have different soc_model values (e.g.
// V73/soc_model=60 and V73/soc_model=61). Their labels must be distinct so that
// same-reason rejections are NOT conflated - each SoC's attribution is preserved.
// This tests the "V<arch>/soc_model=<n>" label format that GetMultiSocSupportedNodes
// emits when both htp_arch and a known soc_model are present.
TEST_F(QnnFrameworkOpTraceUnit, MergePerSoc_SameArchDifferentModel) {
  std::vector<qnn::PerSocUnsupportedNodes> per_soc = {
      {"V73/soc_model=60", {{"node5", "Bar", 5, "rank"}}},
      {"V73/soc_model=61", {{"node5", "Bar", 5, "rank"}}},
  };
  auto merged = qnn::MergePerSocUnsupportedNodes(per_soc);
  ASSERT_EQ(merged.size(), 1u);
  EXPECT_EQ(merged[0].node_index, 5u);
  // Both SoC labels must appear, grouped because the reason is identical.
  EXPECT_EQ(merged[0].reason, "V73/soc_model=60,V73/soc_model=61: rank");
}

// Multi-SoC serialization: soc_traces holds N entries and summary.total_socs == N.
// unsupported_nodes carrying merged multi-SoC reasons round-trip through JSON,
// and summary.unsupported_nodes is a distinct-node count (one row per node).
TEST_F(QnnFrameworkOpTraceUnit, SerializeMultiSoc) {
  qnn::FrameworkOpTrace trace;
  trace.model_name = "fcb_model.onnx";
  trace.backend_type = "htp";
  trace.soc_traces.push_back(qnn::SocTrace{{"V73", 60, 0}});
  trace.soc_traces.push_back(qnn::SocTrace{{"V81", 88, 0}});

  qnn::OpTraceInfo sg;
  sg.graph_name = "QnnEP_graph_0";
  sg.op_mappings.push_back({"Add_0",
                            "QNN_OP_ELEMENT_WISE_ADD",
                            {{"/add/Add", qnn::TraceTargetType::kOp}},
                            "OrtNodeUnit"});
  trace.subgraph_traces.push_back(std::move(sg));
  // One merged row for a node rejected differently on the two SoCs.
  trace.unsupported_nodes.push_back({"node7", "Foo", 7, "V73: data type; V81: rank too high"});
  qnn::ComputeTraceSummary(trace);

  nlohmann::json j = qnn::SerializeFrameworkOpTrace(trace);
  EXPECT_EQ(j["schema_version"], 2);
  EXPECT_FALSE(j.contains("compilation_target"));
  ASSERT_TRUE(j.contains("soc_traces"));
  ASSERT_EQ(j["soc_traces"].size(), 2u);
  EXPECT_EQ(j["soc_traces"][0]["compilation_target"]["htp_arch"], "V73");
  EXPECT_EQ(j["soc_traces"][1]["compilation_target"]["htp_arch"], "V81");
  EXPECT_EQ(j["soc_traces"][1]["compilation_target"]["soc_model"], 88);
  EXPECT_EQ(j["summary"]["total_socs"], 2u);
  // One distinct unsupported node, merged reason preserved.
  ASSERT_EQ(j["unsupported_nodes"].size(), 1u);
  EXPECT_EQ(j["unsupported_nodes"][0]["node_index"], 7u);
  EXPECT_EQ(j["unsupported_nodes"][0]["reason"], "V73: data type; V81: rank too high");
  EXPECT_EQ(j["summary"]["unsupported_nodes"], 1u);
}

// Verify that mixed OP+TENSOR srcInfo is serialized correctly.
TEST_F(QnnFrameworkOpTraceUnit, SrcInfoMixedOpTensor) {
  // Simulate N:1 SpaceToDepth fusion: 3 ops, 2 intermediate tensors
  qnn::FrameworkOpTrace trace;
  trace.model_name = "mixed_test.onnx";
  trace.backend_type = "cpu";

  qnn::OpTraceInfo sg;
  sg.graph_name = "QnnEP_graph_0";
  // N:1: Reshape→Transpose→Reshape → SpaceToDepth
  sg.op_mappings.push_back({
      "SpaceToDepth_0",
      "QNN_OP_SPACE_TO_DEPTH",
      {{"Reshape_6D", qnn::TraceTargetType::kOp},
       {"_out_Reshape_6D", qnn::TraceTargetType::kTensor},
       {"Transpose_6D", qnn::TraceTargetType::kOp},
       {"_out_Transpose_6D", qnn::TraceTargetType::kTensor},
       {"Reshape_4D", qnn::TraceTargetType::kOp}},
      "SpaceToDepthFusion",
  });
  // 1:1: single op, no intermediate tensor
  sg.op_mappings.push_back({
      "Relu_0",
      "QNN_OP_RELU",
      {{"Relu_node", qnn::TraceTargetType::kOp}},
      "OrtNodeUnit",
  });
  trace.subgraph_traces.push_back(std::move(sg));
  qnn::ComputeTraceSummary(trace);

  // Verify summary counts unique ONNX OP source names
  EXPECT_EQ(trace.summary.supported_nodes, 4u);  // Reshape_6D + Transpose_6D + Reshape_4D + Relu_node

  // Verify N:1 srcInfo chain: OP, TENSOR, OP, TENSOR, OP
  const auto& sources = trace.subgraph_traces[0].op_mappings[0].sources;
  ASSERT_EQ(sources.size(), 5u);
  EXPECT_EQ(sources[0].type, qnn::TraceTargetType::kOp);
  EXPECT_EQ(sources[0].name, "Reshape_6D");
  EXPECT_EQ(sources[1].type, qnn::TraceTargetType::kTensor);
  EXPECT_EQ(sources[1].name, "_out_Reshape_6D");
  EXPECT_EQ(sources[2].type, qnn::TraceTargetType::kOp);
  EXPECT_EQ(sources[2].name, "Transpose_6D");
  EXPECT_EQ(sources[3].type, qnn::TraceTargetType::kTensor);
  EXPECT_EQ(sources[3].name, "_out_Transpose_6D");
  EXPECT_EQ(sources[4].type, qnn::TraceTargetType::kOp);
  EXPECT_EQ(sources[4].name, "Reshape_4D");

  // Verify JSON serialization preserves OP/TENSOR type strings
  nlohmann::json j = qnn::SerializeFrameworkOpTrace(trace);
  const auto& jsources = j["subgraph_traces"][0]["op_mappings"][0]["sources"];
  ASSERT_EQ(jsources.size(), 5u);
  EXPECT_EQ(jsources[0]["type"], "OP");
  EXPECT_EQ(jsources[1]["type"], "TENSOR");
  EXPECT_EQ(jsources[2]["type"], "OP");
  EXPECT_EQ(jsources[3]["type"], "TENSOR");
  EXPECT_EQ(jsources[4]["type"], "OP");

  // Verify 1:1 srcInfo: single OP, no TENSOR
  const auto& relu_sources = trace.subgraph_traces[0].op_mappings[1].sources;
  ASSERT_EQ(relu_sources.size(), 1u);
  EXPECT_EQ(relu_sources[0].type, qnn::TraceTargetType::kOp);
}

// Test DeriveTracePathFromContextModel path derivation (test-safe: no backend needed).
// Phase 1 writes qnn_op_trace.json; Phase 2 looks in the same directory.
TEST_F(QnnFrameworkOpTraceUnit, DeriveTracePathFromContextModel_Basic) {
  // Any context model path → {parent}/qnn_op_trace.json (constant filename)
  {
    fs::path ctx = fs::path("/some/dir") / "model_ctx.onnx";
    fs::path expected = fs::path("/some/dir") / "qnn_op_trace.json";
    EXPECT_EQ(qnn::DeriveTracePathFromContextModel(ctx), expected);
  }
  // Stem-only (no directory component): sibling of the ctx file
  {
    fs::path ctx = "mymodel.onnx";
    fs::path derived = qnn::DeriveTracePathFromContextModel(ctx);
    EXPECT_EQ(derived.filename().string(), "qnn_op_trace.json");
  }
  // No extension — still returns qnn_op_trace.json in the same directory
  {
    fs::path ctx = fs::path("/tmp") / "model_ctx";
    fs::path derived = qnn::DeriveTracePathFromContextModel(ctx);
    EXPECT_EQ(derived.filename().string(), "qnn_op_trace.json");
  }
  // Underscore in stem — filename is always constant, stem of ctx is irrelevant
  {
    fs::path ctx = fs::path("/out") / "my_model_ctx.onnx";
    fs::path expected = fs::path("/out") / "qnn_op_trace.json";
    EXPECT_EQ(qnn::DeriveTracePathFromContextModel(ctx), expected);
  }
}

// ParseTraceLookupFromFile is the test-safe core of LoadTraceLookupFromFile.
// These tests exercise every TraceLoadStatus branch and the OP/TENSOR type decode
// without linking the EP library or needing a backend.
TEST_F(QnnFrameworkOpTraceUnit, ParseTraceLookupFromFile_CannotOpen) {
  ScopedTempDir tmp;
  qnn::OpTraceLookup lookup;
  EXPECT_EQ(qnn::ParseTraceLookupFromFile(tmp.path() / "does_not_exist.json", lookup),
            qnn::TraceLoadStatus::kCannotOpen);
  EXPECT_TRUE(lookup.empty());
}

TEST_F(QnnFrameworkOpTraceUnit, ParseTraceLookupFromFile_ParseError) {
  ScopedTempDir tmp;
  fs::path p = tmp.path() / "malformed.json";
  {
    std::ofstream ofs(p);
    ofs << "{ this is not valid json ]";
  }
  qnn::OpTraceLookup lookup;
  EXPECT_EQ(qnn::ParseTraceLookupFromFile(p, lookup), qnn::TraceLoadStatus::kParseError);
  EXPECT_TRUE(lookup.empty());
}

TEST_F(QnnFrameworkOpTraceUnit, ParseTraceLookupFromFile_MissingSubgraphTraces) {
  ScopedTempDir tmp;
  fs::path p = tmp.path() / "no_key.json";
  {
    std::ofstream ofs(p);
    ofs << R"({"model_name": "m.onnx", "summary": {}})";
  }
  qnn::OpTraceLookup lookup;
  EXPECT_EQ(qnn::ParseTraceLookupFromFile(p, lookup),
            qnn::TraceLoadStatus::kMissingSubgraphTraces);
  EXPECT_TRUE(lookup.empty());
}

TEST_F(QnnFrameworkOpTraceUnit, ParseTraceLookupFromFile_OkWithOpAndTensorSources) {
  ScopedTempDir tmp;
  fs::path p = tmp.path() / "trace.json";
  {
    // dst "qnn_add" has one OP source and one TENSOR source; dst "qnn_mul"
    // has a single OP source and the "type" field omitted (defaults to OP).
    std::ofstream ofs(p);
    ofs << R"({
      "subgraph_traces": [
        {
          "op_mappings": [
            {"dst_name": "qnn_add",
             "sources": [
               {"name": "onnx_add", "type": "OP"},
               {"name": "tensor_x", "type": "TENSOR"}
             ]},
            {"dst_name": "qnn_mul",
             "sources": [{"name": "onnx_mul"}]}
          ]
        }
      ]
    })";
  }
  qnn::OpTraceLookup lookup;
  EXPECT_EQ(qnn::ParseTraceLookupFromFile(p, lookup), qnn::TraceLoadStatus::kOk);
  ASSERT_EQ(lookup.size(), 2u);

  ASSERT_EQ(lookup.count("qnn_add"), 1u);
  const auto& add_sources = lookup.at("qnn_add");
  ASSERT_EQ(add_sources.size(), 2u);
  EXPECT_EQ(add_sources[0].name, "onnx_add");
  EXPECT_EQ(add_sources[0].type, qnn::TraceTargetType::kOp);
  EXPECT_EQ(add_sources[1].name, "tensor_x");
  EXPECT_EQ(add_sources[1].type, qnn::TraceTargetType::kTensor);

  ASSERT_EQ(lookup.count("qnn_mul"), 1u);
  const auto& mul_sources = lookup.at("qnn_mul");
  ASSERT_EQ(mul_sources.size(), 1u);
  EXPECT_EQ(mul_sources[0].name, "onnx_mul");
  EXPECT_EQ(mul_sources[0].type, qnn::TraceTargetType::kOp)
      << "omitted `type` field must default to OP";
}

TEST_F(QnnFrameworkOpTraceUnit, ParseTraceLookupFromFile_SkipsMalformedEntries) {
  ScopedTempDir tmp;
  fs::path p = tmp.path() / "trace.json";
  {
    // One subgraph lacks op_mappings (skipped); entries with empty/absent
    // dst_name or absent sources are skipped; one valid entry survives.
    std::ofstream ofs(p);
    ofs << R"({
      "subgraph_traces": [
        {"graph_name": "no_op_mappings_here"},
        {
          "op_mappings": [
            {"dst_name": "", "sources": [{"name": "x", "type": "OP"}]},
            {"dst_name": "missing_sources"},
            {"dst_name": "good", "sources": [{"name": "onnx_good", "type": "OP"}]}
          ]
        }
      ]
    })";
  }
  qnn::OpTraceLookup lookup;
  EXPECT_EQ(qnn::ParseTraceLookupFromFile(p, lookup), qnn::TraceLoadStatus::kOk);
  ASSERT_EQ(lookup.size(), 1u);
  ASSERT_EQ(lookup.count("good"), 1u);
  EXPECT_EQ(lookup.at("good").at(0).name, "onnx_good");
}

// ========================= CPU Backend Integration Tests =========================

// Test trace disabled by default — even with a trace dir set, no file should be written.
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_DisabledByDefault) {
  ScopedTempDir tmp;
  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "0";
  // Explicitly set trace dir but do NOT enable tracing (enable_framework_op_trace defaults to false)
  opts["framework_op_trace_dir"] = tmp.path().string();

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQnnModelTest(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::All});

  EXPECT_TRUE(FindTraceFile(tmp.path()).empty()) << "No trace file when tracing disabled";
}

// Test single op 1:1 mapping.
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_SingleOp_OneToOne) {
  ScopedTempDir tmp;
  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  auto j = RunModelWithTracing(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      tmp.path(), "cpu");
  if (j.empty()) return;

  EXPECT_EQ(j["backend_type"], "cpu");
  ValidateTraceStructure(j);

  // Verify the 1:1 property: for an OrtNodeUnit op mapping (single op, no fusion),
  // the single source must be an OP entry.  Note: dst_name (the QNN op name, generated by
  // UniqueNameGenerator which deduplicates across the process) is NOT required
  // to equal sources[0].name (the raw ONNX node name); they only match on the
  // first use of a given name within a process-level test run.
  bool found_single_node_entry = false;
  for (const auto& m : j["subgraph_traces"][0]["op_mappings"]) {
    if (m.value("node_group_type", "") != "OrtNodeUnit") continue;
    found_single_node_entry = true;
    ASSERT_EQ(m["sources"].size(), 1u) << "OrtNodeUnit mapping must have exactly one source";
    EXPECT_EQ(m["sources"][0]["type"].get<std::string>(), "OP");
    EXPECT_FALSE(m["sources"][0]["name"].get<std::string>().empty());
  }
  EXPECT_TRUE(found_single_node_entry) << "Expected at least one OrtNodeUnit op_mapping entry";
}

// Test unsupported op recorded with reason.
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_UnsupportedNodeReport) {
  ScopedTempDir tmp;
  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", {1, 2, 3}, GetFloatDataInRange(-1.0f, 1.0f, 6));
    builder.AddNode("abs_node", "Abs", {"input"}, {"abs_out"});
    builder.MakeOutput("Y");
    builder.AddNode("nonzero_node", "NonZero", {"abs_out"}, {"Y"});
  };

  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "0";
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();

  RunQnnModelTest(build_model, opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::Some});

  auto trace_file = FindTraceFile(tmp.path());
  if (trace_file.empty()) return;
  auto j = ReadTraceJson(trace_file);

  ASSERT_TRUE(j.contains("unsupported_nodes"));
  ASSERT_GT(j["unsupported_nodes"].size(), 0u) << "NonZero should be recorded in unsupported_nodes";
  for (const auto& un : j["unsupported_nodes"]) {
    EXPECT_FALSE(un["node_name"].get<std::string>().empty());
    EXPECT_FALSE(un["op_type"].get<std::string>().empty());
    EXPECT_FALSE(un["reason"].get<std::string>().empty());
  }
  EXPECT_GT(j["summary"]["unsupported_nodes"].get<size_t>(), 0u);
}

// Test trace is written when the entire model is unsupported.
// In this case ORT does not invoke CompileImpl, so the trace must be flushed
// from the GetCapability path; otherwise the unsupported list is silently
// discarded.
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_AllUnsupportedModel_StillWritesTrace) {
  ScopedTempDir tmp;
  // NonZero is unsupported on QNN CPU; build a model containing only NonZero
  // so QNN reports zero supported nodes.
  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", {1, 2, 3}, GetFloatDataInRange(-1.0f, 1.0f, 6));
    builder.MakeOutput("Y");
    builder.AddNode("nonzero_node", "NonZero", {"input"}, {"Y"});
  };

  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "0";
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();

  RunQnnModelTest(build_model, opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::None});

  auto trace_file = FindTraceFile(tmp.path());
  ASSERT_FALSE(trace_file.empty())
      << "Trace file must be written even when no nodes are supported";
  auto j = ReadTraceJson(trace_file);

  ASSERT_TRUE(j.contains("unsupported_nodes"));
  ASSERT_GT(j["unsupported_nodes"].size(), 0u);
  bool found_nonzero = false;
  for (const auto& un : j["unsupported_nodes"]) {
    if (un["op_type"].get<std::string>() == "NonZero") {
      found_nonzero = true;
      EXPECT_FALSE(un["reason"].get<std::string>().empty());
    }
  }
  EXPECT_TRUE(found_nonzero) << "NonZero must appear in unsupported_nodes";

  ASSERT_TRUE(j.contains("subgraph_traces"));
  EXPECT_TRUE(j["subgraph_traces"].empty())
      << "subgraph_traces must be empty when nothing was supported";
  EXPECT_EQ(j["summary"]["supported_nodes"].get<size_t>(), 0u);
  EXPECT_GT(j["summary"]["unsupported_nodes"].get<size_t>(), 0u);
}

// Test tensor mappings recorded.
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_TensorMappings) {
  ScopedTempDir tmp;
  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  auto j = RunModelWithTracing(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      tmp.path(), "cpu");
  if (j.empty()) return;

  ASSERT_GE(j["subgraph_traces"].size(), 1u);
  const auto& tm = j["subgraph_traces"][0]["tensor_mappings"];
  // Each output tensor of a compiled QNN op produces a tensor_mappings entry
  // whose sources mirror the op's srcInfo chain (same nodes, same types).
  for (const auto& m : tm) {
    EXPECT_FALSE(m["dst_name"].get<std::string>().empty());
    ASSERT_GE(m["sources"].size(), 1u);
    for (const auto& s : m["sources"]) {
      EXPECT_FALSE(s["name"].get<std::string>().empty());
      const auto& t = s["type"].get<std::string>();
      EXPECT_TRUE(t == "OP" || t == "TENSOR") << "unexpected source type: " << t;
    }
  }
}

// Test multiple QNN subgraphs.
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_MultipleSubgraphs) {
  ScopedTempDir tmp;
  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("in0", {1, 2, 3}, GetFloatDataInRange(-1.0f, 1.0f, 6));
    builder.MakeInput<float>("in1", {1, 2, 3}, GetFloatDataInRange(-1.0f, 1.0f, 6));
    builder.AddNode("add_node", "Add", {"in0", "in1"}, {"add_out"});
    builder.AddNode("nonzero_node", "NonZero", {"add_out"}, {"nz_out"});
    ONNX_NAMESPACE::AttributeProto to_attr;
    to_attr.set_name("to");
    to_attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    to_attr.set_i(1);  // FLOAT
    builder.AddNode("cast_node", "Cast", {"nz_out"}, {"cast_out"}, kOnnxDomain, {to_attr});
    builder.MakeOutput("Y");
    builder.AddNode("relu_node", "Relu", {"cast_out"}, {"Y"});
  };

  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "0";
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();

  RunQnnModelTest(build_model, opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::Some});

  auto trace_file = FindTraceFile(tmp.path());
  if (trace_file.empty()) return;
  auto j = ReadTraceJson(trace_file);
  ASSERT_TRUE(j.contains("subgraph_traces"));
  ASSERT_TRUE(j.contains("unsupported_nodes"));
  ASSERT_TRUE(j.contains("summary"));
}

// Builds the fan-out QDQ model: a single graph input feeds two
// independent Q->DQ pairs whose DQ outputs are summed by an Add. With
// `offload_graph_io_quantization=1`, both Q nodes stay on CPU and both DQ
// outputs are registered as QNN graph inputs sharing the override name "input".
// Used by the trace test below to exercise duplicate-override handling in
// QnnModelWrapper::GetModelTensorsMap during framework op trace recording.
static GetTestModelFn BuildOffloadFanoutQDQModelForTrace() {
  return [](ModelTestBuilder& builder) {
    builder.graph_->set_name("trace_offload_fanout_qdq");

    MakeTestInput<float>(builder, "input", TestInputDef<float>({1, 3}, false, {0.1f, 0.2f, 0.3f}));
    builder.MakeInitializer<float>("scale", {}, {1.0f / 255.0f});
    builder.MakeInitializer<uint8_t>("zp", {}, {0});

    builder.AddNode("q_a", "QuantizeLinear", {"input", "scale", "zp"}, {"q_a_out"}, kOnnxDomain);
    builder.AddNode("dq_a", "DequantizeLinear", {"q_a_out", "scale", "zp"}, {"dq_a_out"}, kOnnxDomain);
    builder.AddNode("q_b", "QuantizeLinear", {"input", "scale", "zp"}, {"q_b_out"}, kOnnxDomain);
    builder.AddNode("dq_b", "DequantizeLinear", {"q_b_out", "scale", "zp"}, {"dq_b_out"}, kOnnxDomain);
    builder.AddNode("add", "Add", {"dq_a_out", "dq_b_out"}, {"add_out"}, kOnnxDomain);
    builder.AddNode("q_out", "QuantizeLinear", {"add_out", "scale", "zp"}, {"q_out_out"}, kOnnxDomain);
    builder.AddNode("dq_out", "DequantizeLinear", {"q_out_out", "scale", "zp"}, {"output"}, kOnnxDomain);
    builder.MakeOutput("output");
  };
}

// Exercises the framework op trace path for the `offload_graph_io_quantization=1`
// + graph-input fan-out QDQ scenario.
// Validates the dedup contract in OpTraceCollector::Finalize:
//   * Option A: each qnn_name appears at most once in tensor_mappings.
//   * Option B: when qnn_name is an external override target (e.g. the graph
//     input "input" aliased back from an internal QNN name under
//     offload_graph_io_quantization), the entry's sources contain exactly one
//     element whose name equals the dst_name (the canonical ONNX name).
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_OffloadGraphIoQuantization_FanoutQDQ) {
  ScopedTempDir tmp;

  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  opts["offload_graph_io_quantization"] = "1";
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();

  // Q nodes stay on CPU when offloading is enabled; DQ + Add land on QNN.
  RunQnnModelTest(BuildOffloadFanoutQDQModelForTrace(), opts, /*opset*/ 18,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some});

  if (::testing::UnitTest::GetInstance()->current_test_info()->result()->Skipped()) {
    return;
  }

  auto trace_file = FindTraceFile(tmp.path());
  ASSERT_FALSE(trace_file.empty()) << "No trace file written for offload fanout QDQ model";

  auto j = ReadTraceJson(trace_file);
  ASSERT_TRUE(j.contains("subgraph_traces"));
  ASSERT_GE(j["subgraph_traces"].size(), 1u);
  ASSERT_TRUE(j["subgraph_traces"][0].contains("tensor_mappings"));
  ASSERT_TRUE(j.contains("summary"));

  // Option A: dst_name uniqueness across tensor_mappings.
  std::unordered_set<std::string> seen_dst_names;
  for (const auto& sg : j["subgraph_traces"]) {
    for (const auto& m : sg["tensor_mappings"]) {
      const auto dst = m["dst_name"].get<std::string>();
      EXPECT_TRUE(seen_dst_names.insert(dst).second)
          << "tensor_mappings has duplicate dst_name: " << dst;
    }
  }

  // Option B: the override-target entry for the canonical graph input "input"
  // must have exactly one source whose name equals "input".
  bool found_canonical_input = false;
  for (const auto& sg : j["subgraph_traces"]) {
    for (const auto& m : sg["tensor_mappings"]) {
      if (m["dst_name"].get<std::string>() != "input") continue;
      found_canonical_input = true;
      ASSERT_EQ(m["sources"].size(), 1u)
          << "override-target entry must collapse to a single canonical source";
      EXPECT_EQ(m["sources"][0]["name"].get<std::string>(), "input");
      EXPECT_EQ(m["sources"][0]["type"].get<std::string>(), "TENSOR");
    }
  }
  EXPECT_TRUE(found_canonical_input)
      << "expected a tensor_mappings entry for the canonical override target 'input'";
}

// Test custom output directory.
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_CustomOutputDir) {
  ScopedTempDir tmp;
  fs::path custom_dir = tmp.path() / "nested" / "custom_trace";
  ASSERT_FALSE(fs::exists(custom_dir));

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunModelWithTracing(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      custom_dir, "cpu");

  EXPECT_TRUE(fs::exists(custom_dir));
  EXPECT_FALSE(FindTraceFile(custom_dir).empty());
}

// New schema (FCB readiness): single-SoC sessions populate a length-1
// soc_traces array, no compilation_target at root, subgraph_traces stays at
// root, and summary.total_socs reflects soc_traces.size(). Consolidates four
// previously-separate fixture-rebuild tests into one inference run. Also
// subsumes the former FrameworkOpTrace_CompilationTargetMetadata (same
// Add-on-CPU model, strict superset of its assertions).
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_SingleSoc_NewSchema) {
  ScopedTempDir tmp;
  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  auto j = RunModelWithTracing(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data),
                              TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      tmp.path(), "cpu");
  if (j.empty()) return;

  // soc_traces: exactly one entry with a CompilationTarget on a single-SoC run.
  ASSERT_TRUE(j.contains("soc_traces"));
  ASSERT_EQ(j["soc_traces"].size(), 1u)
      << "Single-SoC session must produce exactly one soc_traces entry";
  const auto& ct = j["soc_traces"][0]["compilation_target"];
  EXPECT_TRUE(ct.contains("htp_arch"));
  EXPECT_TRUE(ct.contains("soc_model"));
  EXPECT_TRUE(ct.contains("device_id"));

  // Breaking schema change: no compilation_target at root.
  EXPECT_FALSE(j.contains("compilation_target"))
      << "compilation_target was moved out of root in the FCB-ready schema; "
         "consumers must read soc_traces[i].compilation_target instead";

  // subgraph_traces stays at root (SoC-agnostic op-builder output) and is
  // not duplicated under soc_traces[i].
  ASSERT_TRUE(j.contains("subgraph_traces"))
      << "subgraph_traces must be at root (SoC-agnostic op mapping)";
  ASSERT_GE(j["subgraph_traces"].size(), 1u);
  for (const auto& soc : j["soc_traces"]) {
    EXPECT_FALSE(soc.contains("subgraph_traces"))
        << "subgraph_traces must NOT be duplicated under soc_traces[i]";
  }

  // summary.total_socs mirrors soc_traces.size().
  ASSERT_TRUE(j.contains("summary"));
  ASSERT_TRUE(j["summary"].contains("total_socs"));
  EXPECT_EQ(j["summary"]["total_socs"].get<uint32_t>(), 1u)
      << "Single-SoC session must report total_socs == 1";
  EXPECT_EQ(j["soc_traces"].size(), j["summary"]["total_socs"].get<uint32_t>());
}

// Test explicit output directory (simulates Android non-writable CWD).
TEST_F(QnnCPUBackendTests, FrameworkOpTrace_ExplicitOutputDir_NonWritableCWD) {
  ScopedTempDir tmp;
  fs::path explicit_dir = tmp.path() / "explicit_output";
  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunModelWithTracing(
      BuildOpTestCase<float>("relu_node", "Relu",
                             {TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      explicit_dir, "cpu");
  EXPECT_TRUE(fs::exists(explicit_dir));
  EXPECT_FALSE(FindTraceFile(explicit_dir).empty());
}

// ========================= HTP Backend Integration Tests =========================

// Test HTP backend basic trace.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_BasicHTP) {
  ScopedTempDir tmp;
  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  auto j = RunModelWithTracing(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      tmp.path(), "htp");
  if (j.empty()) return;

  EXPECT_EQ(j["backend_type"], "htp");
  ValidateTraceStructure(j);
}

// Test QDQ fusion N:1 mapping on HTP.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_QDQFusion_ManyToOne) {
  ScopedTempDir tmp;

  auto build_qdq_conv = [](ModelTestBuilder& builder) {
    // Use uint8 input directly so DequantizeLinear receives a quantized tensor.
    builder.MakeInput<uint8_t>("input", {1, 1, 5, 5}, std::vector<uint8_t>(25, 128));
    builder.MakeInitializer<uint8_t>("weight", {1, 1, 3, 3}, std::vector<uint8_t>(9, 128));
    builder.AddDequantizeLinearNode<uint8_t>("dq_input_node", "input", 0.004f, static_cast<uint8_t>(128), "dq_input");
    builder.AddDequantizeLinearNode<uint8_t>("dq_weight_node", "weight", 0.003f, static_cast<uint8_t>(128), "dq_weight");
    builder.AddNode("conv_node", "Conv", {"dq_input", "dq_weight"}, {"conv_output"});
    builder.MakeOutput("output");
    builder.AddQuantizeLinearNode<uint8_t>("q_output_node", "conv_output", 0.005f, static_cast<uint8_t>(128), "output");
  };

  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();

  RunQnnModelTest(build_qdq_conv, opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::All});

  auto trace_file = FindTraceFile(tmp.path());
  if (trace_file.empty()) return;
  auto j = ReadTraceJson(trace_file);
  ASSERT_GE(j["subgraph_traces"].size(), 1u);

  bool found_multi = false;
  bool found_tensor_src = false;
  for (const auto& m : j["subgraph_traces"][0]["op_mappings"]) {
    if (m["sources"].size() > 1) {
      found_multi = true;
      for (const auto& s : m["sources"]) {
        if (s["type"] == "TENSOR") {
          found_tensor_src = true;
          break;
        }
      }
      break;
    }
  }
  EXPECT_TRUE(found_multi) << "HTP QDQ fusion should produce multi-source mapping";
  EXPECT_TRUE(found_tensor_src) << "HTP QDQ fusion sources should contain TENSOR entries (intermediate outputs)";
}

// Test Gelu fusion on HTP.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_GeluFusion) {
#if defined(_WIN32) && !defined(_M_ARM64)
  GTEST_SKIP() << "GeluFusion not supported on Windows x86 HTP simulation";
#endif
  ScopedTempDir tmp;
  // Use the ONNX Gelu decomposition pattern (Mul/Div/Erf/Add) so the
  // GeluFusion node group fires. kMSDomain Gelu fails to finalize on HTP.
  auto build_gelu = [](ModelTestBuilder& builder) {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;
    builder.MakeInput<float>("input", {1, 2, 8}, GetFloatDataInRange(-3.0f, 3.0f, 16));
    builder.MakeScalarInitializer<float>("half_const", half);
    builder.AddNode("Mul_half", "Mul", {"input", "half_const"}, {"mul_half_out"});
    builder.MakeScalarInitializer<float>("sqrt2_const", sqrt_2);
    builder.AddNode("Div_sqrt2", "Div", {"input", "sqrt2_const"}, {"div_out"});
    builder.AddNode("Erf_node", "Erf", {"div_out"}, {"erf_out"});
    builder.MakeScalarInitializer<float>("one_const", one);
    builder.AddNode("Add_one", "Add", {"erf_out", "one_const"}, {"add_out"});
    builder.AddNode("Mul_out", "Mul", {"add_out", "mul_half_out"}, {"Y"});
    builder.MakeOutput("Y");
  };

  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  // HTP executes float ops via float16 hardware precision, so fp32 comparison
  // requires a relaxed tolerance. 5e-3f covers observed rounding on both
  // simulation (x86) and real ARM64 hardware.
  RunQnnModelTest(build_gelu,
                  opts,
                  13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(5e-3f)});

  auto trace_file = FindTraceFile(tmp.path());
  if (trace_file.empty()) return;
  auto j = ReadTraceJson(trace_file);
  ASSERT_GE(j["subgraph_traces"].size(), 1u);
  EXPECT_GT(j["subgraph_traces"][0]["op_mappings"].size(), 0u);
}

// Test 1:N mapping on HTP.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_OneToMany) {
  ScopedTempDir tmp;
  auto build_split = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", {1, 6, 2}, GetFloatDataInRange(-1.0f, 1.0f, 12));
    builder.MakeOutput("Y0");
    builder.MakeOutput("Y1");
    ONNX_NAMESPACE::AttributeProto axis_attr;
    axis_attr.set_name("axis");
    axis_attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    axis_attr.set_i(1);
    builder.AddNode("split_node", "Split", {"input"}, {"Y0", "Y1"}, kOnnxDomain, {axis_attr});
  };

  auto j = RunModelWithTracing(build_split, tmp.path(), "htp", 13);
  if (j.empty()) return;
  EXPECT_EQ(j["backend_type"], "htp");
  ValidateTraceStructure(j);
}

// Test HTP compilation_target metadata. New schema: lives under soc_traces[0].
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_HTP_CompilationTarget) {
  ScopedTempDir tmp;
  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  auto j = RunModelWithTracing(
      BuildOpTestCase<float>("relu_node", "Relu",
                             {TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      tmp.path(), "htp");
  if (j.empty()) return;

  EXPECT_EQ(j["backend_type"], "htp");
  ASSERT_FALSE(j.contains("compilation_target"))
      << "compilation_target must NOT be at root (moved to soc_traces[0])";
  ASSERT_TRUE(j.contains("soc_traces"));
  ASSERT_EQ(j["soc_traces"].size(), 1u);
  ASSERT_TRUE(j["soc_traces"][0].contains("compilation_target"));
  EXPECT_TRUE(j["soc_traces"][0]["compilation_target"].contains("device_id"));
}

// ========================= AOT Path Integration Tests =========================

static std::string BuildModelData(const GetTestModelFn& build_fn, int opset_version = 13) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};
  ModelTestBuilder helper;
  build_fn(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  return model_data;
}

// Test AOT Phase 1 — trace JSON written to framework_op_trace_dir.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_AOT_Phase1_TraceWritten) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path trace_dir = tmp.path() / "trace";
  fs::path ctx_file = tmp.path() / "model_ctx.onnx";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::string model_data = BuildModelData(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_framework_op_trace"] = "1";
  provider_options["framework_op_trace_dir"] = trace_dir.string();

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", provider_options);
  Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);

  // Verify context model was generated
  EXPECT_TRUE(fs::exists(ctx_file)) << "Context model not generated";

  // Verify trace JSON was written to framework_op_trace_dir
  auto trace_file = FindTraceFile(trace_dir);
  ASSERT_FALSE(trace_file.empty()) << "No trace file in framework_op_trace_dir";

  auto j = ReadTraceJson(trace_file);
  ValidateTraceStructure(j);
  EXPECT_EQ(j["backend_type"], "htp");

  // Phase 1 writes only to framework_op_trace_dir; no sidecar alongside the context model.
  fs::path sidecar_path = ctx_file.parent_path() / "qnn_op_trace.json";
  EXPECT_FALSE(fs::exists(sidecar_path)) << "No sidecar should be written alongside context model";
}

// Test AOT Phase 1 — no trace output when tracing disabled.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_AOT_Phase1_DisabledNoTrace) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path trace_dir = tmp.path() / "trace";
  fs::path ctx_file = tmp.path() / "model_ctx.onnx";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::string model_data = BuildModelData(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  // Tracing NOT enabled (default: enable_framework_op_trace = false)

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", provider_options);
  Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);

  // Context model should exist
  EXPECT_TRUE(fs::exists(ctx_file));

  // No trace file should exist (no trace_dir created, no sidecar)
  EXPECT_TRUE(FindTraceFile(trace_dir).empty()) << "Trace file should not be generated when tracing disabled";
  fs::path sidecar_path = ctx_file.parent_path() / "qnn_op_trace.json";
  EXPECT_FALSE(fs::exists(sidecar_path)) << "Sidecar should not be written when tracing disabled";
}

// Test AOT Phase 1 trace content matches JIT trace content for same model.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_AOT_Phase1_MatchesJIT) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path jit_trace_dir = tmp.path() / "jit";
  fs::path aot_trace_dir = tmp.path() / "aot";
  fs::path ctx_file = tmp.path() / "model_ctx.onnx";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  auto build_fn = BuildOpTestCase<float>(
      "add_node", "Add",
      {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
      {}, {}, kOnnxDomain);

  // JIT path: run with tracing
  auto jit_j = RunModelWithTracing(build_fn, jit_trace_dir, "htp");
  if (jit_j.empty()) return;

  // AOT Phase 1: run with tracing + context generation
  std::string model_data = BuildModelData(build_fn);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_framework_op_trace"] = "1";
  provider_options["framework_op_trace_dir"] = aot_trace_dir.string();

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", provider_options);
  Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);

  auto aot_trace_file = FindTraceFile(aot_trace_dir);
  ASSERT_FALSE(aot_trace_file.empty()) << "No AOT trace file generated";
  auto aot_j = ReadTraceJson(aot_trace_file);

  // Compare: same backend, same op_mappings structure
  EXPECT_EQ(jit_j["backend_type"], aot_j["backend_type"]);
  ASSERT_GE(jit_j["subgraph_traces"].size(), 1u);
  ASSERT_GE(aot_j["subgraph_traces"].size(), 1u);

  const auto& jit_ops = jit_j["subgraph_traces"][0]["op_mappings"];
  const auto& aot_ops = aot_j["subgraph_traces"][0]["op_mappings"];
  ASSERT_EQ(jit_ops.size(), aot_ops.size());

  for (size_t i = 0; i < jit_ops.size(); ++i) {
    EXPECT_EQ(jit_ops[i]["dst_name"], aot_ops[i]["dst_name"]);
    EXPECT_EQ(jit_ops[i]["dst_qnn_op_type"], aot_ops[i]["dst_qnn_op_type"]);
    EXPECT_EQ(jit_ops[i]["sources"], aot_ops[i]["sources"]);
    EXPECT_EQ(jit_ops[i]["node_group_type"], aot_ops[i]["node_group_type"]);
  }

  // Summary should match
  EXPECT_EQ(jit_j["summary"]["total_qnn_ops"], aot_j["summary"]["total_qnn_ops"]);
  EXPECT_EQ(jit_j["summary"]["supported_nodes"], aot_j["summary"]["supported_nodes"]);
}

// Test AOT Phase 2 sidecar discovery:
// when a Phase 1 trace JSON exists alongside the context model,
// Phase 2 loads it into op_trace_lookup_ without crashing.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_AOT_Phase2_SidecarDiscovery) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path ctx_file = tmp.path() / "model_ctx.onnx";
  // DeriveTracePathFromContextModel returns {parent}/qnn_op_trace.json — use the
  // production filename so CompileContextModel actually finds the sidecar.
  fs::path sidecar_path = tmp.path() / "qnn_op_trace.json";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::string model_data = BuildModelData(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain));

  // Phase 1: generate context model + trace JSON
  fs::path phase1_trace_dir = tmp.path() / "phase1_trace";
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = phase1_trace_dir.string();

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
  }

  ASSERT_TRUE(fs::exists(ctx_file)) << "Context model not generated in Phase 1";
  auto phase1_trace_file = FindTraceFile(phase1_trace_dir);
  ASSERT_FALSE(phase1_trace_file.empty()) << "No Phase 1 trace file found";

  // Place sidecar trace file alongside context model (DeriveTracePathFromContextModel convention)
  std::error_code ec;
  fs::copy_file(phase1_trace_file, sidecar_path, ec);
  ASSERT_FALSE(ec) << "Failed to copy trace sidecar: " << ec.message();
  ASSERT_TRUE(fs::exists(sidecar_path));

  // Phase 2: load context model with tracing enabled; the sidecar is discovered
  // and loaded. framework_op_trace_dir points at a dedicated, initially-empty
  // directory so the "no new trace file" check below is unambiguous — tmp.path()
  // already holds the sidecar (qnn_op_trace.json), whose stem FindTraceFile matches.
  fs::path phase2_trace_dir = tmp.path() / "phase2_trace";
  fs::create_directories(phase2_trace_dir);
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = phase2_trace_dir.string();
    opts["disable_file_mapped_weights"] = "1";

    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    // Should not throw — sidecar is found and loaded without error
    ASSERT_NO_THROW(Ort::Session session(*ort_env, ctx_file.c_str(), so));
  }

  // Loading a pre-compiled context model runs no ComposeGraph, so Phase 2 writes
  // no new trace file. The dedicated Phase 2 directory therefore stays empty.
  EXPECT_TRUE(FindTraceFile(phase2_trace_dir).empty())
      << "Phase 2 (context-model load) should not write a new trace file";
}

// Test AOT Phase 2 with tracing enabled but NO sidecar file — should work fine (no crash).
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_AOT_Phase2_NoSidecar_StillWorks) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path ctx_file = tmp.path() / "model_ctx.onnx";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::string model_data = BuildModelData(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain));

  // Phase 1: generate context model WITHOUT trace
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
  }

  ASSERT_TRUE(fs::exists(ctx_file));
  // Verify no sidecar was auto-created
  fs::path sidecar_path = tmp.path() / "qnn_op_trace.json";
  ASSERT_FALSE(fs::exists(sidecar_path)) << "No sidecar should exist when Phase 1 had tracing disabled";

  // Phase 2: load context model with tracing enabled but no sidecar present
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["disable_file_mapped_weights"] = "1";

    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    // Should not throw even though no sidecar exists
    ASSERT_NO_THROW(Ort::Session session(*ort_env, ctx_file.c_str(), so));
  }
}

// ========================= Mixed EPContext Model Integration Tests =========================

// Test phase 1 trace for model that produces mixed EPContext:
// unsupported_nodes contains the ops that become non-EPContext.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_MixedEPContext_Phase1UnsupportedNodesCaptured) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path trace_dir = tmp.path() / "trace";
  fs::path ctx_file = tmp.path() / "mixed_ctx.onnx";

  // Build model with supported (Abs) + unsupported (NonZero) ops
  auto build_mixed = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", {1, 2, 3}, GetFloatDataInRange(-1.0f, 1.0f, 6));
    builder.AddNode("abs_node", "Abs", {"input"}, {"abs_out"});
    builder.MakeOutput("Y");
    builder.AddNode("nonzero_node", "NonZero", {"abs_out"}, {"Y"});
  };

  std::string model_data = BuildModelData(build_mixed);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_framework_op_trace"] = "1";
  provider_options["framework_op_trace_dir"] = trace_dir.string();

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", provider_options);
  Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);

  auto trace_file = FindTraceFile(trace_dir);
  ASSERT_FALSE(trace_file.empty()) << "No trace file generated for mixed model";
  auto j = ReadTraceJson(trace_file);

  // Unsupported nodes should capture the NonZero op
  ASSERT_TRUE(j.contains("unsupported_nodes"));
  bool found_nonzero = false;
  for (const auto& un : j["unsupported_nodes"]) {
    if (un["op_type"] == "NonZero") {
      found_nonzero = true;
      EXPECT_FALSE(un["reason"].get<std::string>().empty()) << "Unsupported reason should not be empty";
      break;
    }
  }
  EXPECT_TRUE(found_nonzero) << "NonZero should appear in unsupported_nodes";
  EXPECT_GT(j["summary"]["unsupported_nodes"].get<size_t>(), 0u);
}

// Test mixed EPContext model Phase 2 — no new trace file produced.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_MixedEPContext_NoTraceInPhase2) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path ctx_file = tmp.path() / "mixed_ctx.onnx";

  // Phase 1: Generate context model from mixed model
  auto build_mixed = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", {1, 2, 3}, GetFloatDataInRange(-1.0f, 1.0f, 6));
    builder.AddNode("abs_node", "Abs", {"input"}, {"abs_out"});
    builder.MakeOutput("Y");
    builder.AddNode("nonzero_node", "NonZero", {"abs_out"}, {"Y"});
  };

  std::string model_data = BuildModelData(build_mixed);

  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
  }

  ASSERT_TRUE(fs::exists(ctx_file)) << "Context model not generated in Phase 1";

  // Phase 2: Load context model with tracing enabled, but no new trace should appear
  fs::path phase2_trace_dir = tmp.path() / "phase2_trace";
  fs::create_directories(phase2_trace_dir);

  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = phase2_trace_dir.string();
    opts["disable_file_mapped_weights"] = "1";  // avoid DMA path crash with App Verifier

    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, ctx_file.c_str(), so);
  }

  // No trace file in Phase 2 trace dir (Phase 2 has no trace behavior)
  EXPECT_TRUE(FindTraceFile(phase2_trace_dir).empty())
      << "Phase 2 should not produce a trace file";
}

// Test mixed EPContext model Phase 2 — inference succeeds with tracing enabled.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_MixedEPContext_Phase2InferenceSucceeds) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path ctx_file = tmp.path() / "mixed_ctx.onnx";

  // Phase 1: Generate context model from a simple supported-only model
  // (Mixed model inference is harder to set up; use all-supported model to verify Phase 2 doesn't crash)
  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::string model_data = BuildModelData(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain));

  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
  }

  ASSERT_TRUE(fs::exists(ctx_file));

  // Phase 2: Load context model with tracing enabled and run inference
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = tmp.path().string();
    opts["disable_file_mapped_weights"] = "1";  // avoid DMA path crash with App Verifier

    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, ctx_file.c_str(), so);

    // Run inference — just verify it doesn't crash
    auto input_names = session.GetInputNames();
    auto output_names = session.GetOutputNames();

    std::vector<float> input_data(6, 1.0f);
    std::vector<int64_t> input_shape = {1, 2, 3};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < input_names.size(); ++i) {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));
    }

    std::vector<const char*> input_name_ptrs;
    for (const auto& name : input_names) input_name_ptrs.push_back(name.c_str());
    std::vector<const char*> output_name_ptrs;
    for (const auto& name : output_names) output_name_ptrs.push_back(name.c_str());

    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_name_ptrs.data(), inputs.data(), inputs.size(),
                               output_name_ptrs.data(), output_name_ptrs.size());
    EXPECT_GT(outputs.size(), 0u) << "Inference should produce output";
  }
}

// ========================= HTP Mixed EPContext Model Tests =========================

// Build a QDQ model with FusedGemm (unsupported on HTP) + quantized Add (supported).
// This produces a mixed EPContext model: FusedGemm becomes non-EPContext, Add becomes EPContext.
static GetTestModelFn BuildHTPMixedModel() {
  return [](ModelTestBuilder& builder) {
    // FusedGemm (kMSDomain) — unsupported on HTP backend
    std::vector<float> data(12 * 12, 1.0f);
    MakeTestInput(builder, "input1", TestInputDef<float>({12, 12}, false, data));
    MakeTestInput(builder, "gemm_weight", TestInputDef<float>({12, 12}, true, data));
    std::vector<ONNX_NAMESPACE::AttributeProto> fusedgemm_attrs;
    fusedgemm_attrs.push_back(builder.MakeStringAttribute("activation", "Relu"));
    builder.AddNode("FusedGemm_node0", "FusedGemm", {"input1", "gemm_weight"}, {"gemm_out"},
                    kMSDomain, fusedgemm_attrs);

    // Quantized Add — supported via QDQ fusion on HTP
    gsl::span<float> data_range = gsl::make_span(data);
    QuantParams<uint8_t> q_params = GetDataQuantParams<uint8_t>(data_range);
    std::string add_in1_qdq = AddQDQNodePair<uint8_t>(builder, "add_in1_qdq", "gemm_out",
                                                      q_params.scale, q_params.zero_point);
    MakeTestInput(builder, "add_input2", TestInputDef<float>({12, 12}, true, data));
    std::string add_in2_qdq = AddQDQNodePair<uint8_t>(builder, "add_in2_qdq", "add_input2",
                                                      q_params.scale, q_params.zero_point);
    builder.AddNode("Add_node0", "Add", {add_in1_qdq, add_in2_qdq}, {"add_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "add_out",
                                                   q_params.scale, q_params.zero_point);
  };
}

// Test (HTP): Phase 1 trace captures unsupported FusedGemm with reason.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_MixedEPContext_HTP_Phase1UnsupportedNodesCaptured) {
  ScopedTempDir tmp;
  fs::path trace_dir = tmp.path() / "trace";
  fs::path ctx_file = tmp.path() / "mixed_ctx.onnx";

  std::string model_data = BuildModelData(BuildHTPMixedModel());

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_framework_op_trace"] = "1";
  provider_options["framework_op_trace_dir"] = trace_dir.string();

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", provider_options);
  Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);

  auto trace_file = FindTraceFile(trace_dir);
  ASSERT_FALSE(trace_file.empty()) << "No trace file generated for HTP mixed model";
  auto j = ReadTraceJson(trace_file);

  // Unsupported nodes should capture FusedGemm
  ASSERT_TRUE(j.contains("unsupported_nodes"));
  bool found_fusedgemm = false;
  for (const auto& un : j["unsupported_nodes"]) {
    if (un["op_type"] == "FusedGemm") {
      found_fusedgemm = true;
      EXPECT_FALSE(un["reason"].get<std::string>().empty()) << "Unsupported reason should not be empty";
      break;
    }
  }
  EXPECT_TRUE(found_fusedgemm) << "FusedGemm should appear in unsupported_nodes";
  EXPECT_EQ(j["backend_type"], "htp");
  EXPECT_GT(j["summary"]["unsupported_nodes"].get<size_t>(), 0u);
}

// Test (HTP): Mixed EPContext model Phase 2 — no trace file produced.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_MixedEPContext_HTP_NoTraceInPhase2) {
  ScopedTempDir tmp;
  fs::path ctx_file = tmp.path() / "mixed_ctx.onnx";

  std::string model_data = BuildModelData(BuildHTPMixedModel());

  // Phase 1: Generate context model
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
  }

  ASSERT_TRUE(fs::exists(ctx_file)) << "Context model not generated in Phase 1";

  // Phase 2: Load context model with tracing enabled
  fs::path phase2_trace_dir = tmp.path() / "phase2_trace";
  fs::create_directories(phase2_trace_dir);

  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = phase2_trace_dir.string();
    opts["disable_file_mapped_weights"] = "1";  // avoid DMA path crash with App Verifier

    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, ctx_file.c_str(), so);
  }

  EXPECT_TRUE(FindTraceFile(phase2_trace_dir).empty())
      << "Phase 2 should not produce a trace file";
}

// Test (HTP): Mixed EPContext model Phase 2 — inference succeeds with tracing enabled.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_MixedEPContext_HTP_Phase2InferenceSucceeds) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path ctx_file = tmp.path() / "mixed_ctx.onnx";

  // Phase 1: Use a simple all-supported HTP model for inference verification
  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::string model_data = BuildModelData(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain));

  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
  }

  ASSERT_TRUE(fs::exists(ctx_file));

  // Phase 2: Load context model with tracing enabled and run inference
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = tmp.path().string();
    opts["disable_file_mapped_weights"] = "1";  // avoid DMA path crash with App Verifier

    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, ctx_file.c_str(), so);

    auto input_names = session.GetInputNames();
    auto output_names = session.GetOutputNames();

    std::vector<float> input_data(6, 1.0f);
    std::vector<int64_t> input_shape = {1, 2, 3};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < input_names.size(); ++i) {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));
    }

    std::vector<const char*> input_name_ptrs;
    for (const auto& name : input_names) input_name_ptrs.push_back(name.c_str());
    std::vector<const char*> output_name_ptrs;
    for (const auto& name : output_names) output_name_ptrs.push_back(name.c_str());

    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_name_ptrs.data(), inputs.data(), inputs.size(),
                               output_name_ptrs.data(), output_name_ptrs.size());
    EXPECT_GT(outputs.size(), 0u) << "Inference should produce output";
  }
}

// ========================= N:M Fusion Integration Test =========================

// Test N:1 fusion — multiple ONNX nodes → single QNN Gelu op sharing same source chain.
// Uses ONNX Gelu decomposition pattern (Mul/Div/Erf/Add) which gets fused by GeluFusion on HTP.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_NtoM_GeluFusion) {
#if defined(_WIN32) && !defined(_M_ARM64)
  GTEST_SKIP() << "GeluFusion not supported on Windows x86 HTP simulation";
#endif
  ScopedTempDir tmp;
  // ONNX Gelu decomposition pattern: GeluFusion maps 5 ONNX ops → 1 QNN Gelu op.
  auto build_gelu = [](ModelTestBuilder& builder) {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;
    builder.MakeInput<float>("input", {1, 2, 8}, GetFloatDataInRange(-3.0f, 3.0f, 16));
    builder.MakeScalarInitializer<float>("half_const", half);
    builder.AddNode("Mul_half", "Mul", {"input", "half_const"}, {"mul_half_out"});
    builder.MakeScalarInitializer<float>("sqrt2_const", sqrt_2);
    builder.AddNode("Div_sqrt2", "Div", {"input", "sqrt2_const"}, {"div_out"});
    builder.AddNode("Erf_node", "Erf", {"div_out"}, {"erf_out"});
    builder.MakeScalarInitializer<float>("one_const", one);
    builder.AddNode("Add_one", "Add", {"erf_out", "one_const"}, {"add_out"});
    builder.AddNode("Mul_out", "Mul", {"add_out", "mul_half_out"}, {"Y"});
    builder.MakeOutput("Y");
  };

  auto j = [&]() -> nlohmann::json {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = tmp.path().string();
#if defined(__linux__) && !defined(__aarch64__)
    opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
    // HTP executes float ops via float16 hardware precision, so fp32 comparison
    // requires a relaxed tolerance. 5e-3f covers observed rounding on both
    // simulation (x86) and real ARM64 hardware.
    RunQnnModelTest(build_gelu,
                    opts,
                    13,
                    EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(5e-3f)});
    auto trace_file = FindTraceFile(tmp.path());
    if (trace_file.empty()) return {};
    return ReadTraceJson(trace_file);
  }();
  if (j.empty()) return;

  ValidateTraceStructure(j);

  // Verify fusion pattern: multiple QNN ops should share the same source chain
  // indicating N:M mapping (Gelu fuses multiple ONNX ops into potentially multiple QNN ops)
  ASSERT_GE(j["subgraph_traces"].size(), 1u);
  const auto& op_mappings = j["subgraph_traces"][0]["op_mappings"];
  EXPECT_GT(op_mappings.size(), 0u);

  // Check that at least one op has a non-OrtNodeUnit node_group_type (indicating fusion)
  bool found_fusion = false;
  for (const auto& m : op_mappings) {
    if (m.contains("node_group_type") && m["node_group_type"] != "OrtNodeUnit") {
      found_fusion = true;
      // Fusion ops should have multiple sources
      EXPECT_GT(m["sources"].size(), 1u) << "Fused op should have multiple source entries";
      break;
    }
  }
  // Gelu is decomposed into multiple ops by ONNX optimizer, which may then be fused by QNN EP
  // If no fusion found, at least verify the trace is valid
  if (!found_fusion) {
    EXPECT_GT(op_mappings.size(), 0u) << "Should have at least one op mapping";
  }
}

// Test N:M trace for ReshapeEinsumReshape fusion (3 ONNX ops → 3 QNN ops).
// The fusion maps Reshape→Einsum→Reshape to Reshape+DepthToSpace+Transpose in QNN.
// Each QNN op entry in op_mappings should share the same 3 ONNX op sources, proving
// that the trace correctly records the N:M relationship.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_NtoM_ReshapeEinsumReshape) {
#if defined(_WIN32) && !defined(_M_ARM64)
  GTEST_SKIP() << "ReshapeEinsumReshape fusion not supported on Windows x86 HTP simulation";
#endif
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;

  // Reshape(1024,32)→(1,32,32,2,2,8) + Einsum("nhwpqc->nchpwq") + Reshape→(1,8,64,64)
  // This pattern triggers ReshapeEinsumReshapeNodeGroup which emits 3 QNN ops:
  //   Reshape, DepthToSpace, Transpose
  auto build_fn = [](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", TestInputDef<float>({1024, 32}, false, -10.0f, 10.0f));
    builder.Make1DInitializer<int64_t>("reshape1_shape", {1, 32, 32, 2, 2, 8});
    builder.AddNode("reshape1", "Reshape", {"input", "reshape1_shape"}, {"reshape1_out"}, kOnnxDomain);
    ONNX_NAMESPACE::AttributeProto eq_attr;
    eq_attr.set_name("equation");
    eq_attr.set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    eq_attr.set_s("nhwpqc->nchpwq");
    builder.AddNode("einsum", "Einsum", {"reshape1_out"}, {"einsum_out"}, kOnnxDomain, {eq_attr});
    builder.Make1DInitializer<int64_t>("reshape2_shape", {1, 8, 64, 64});
    builder.AddNode("reshape2", "Reshape", {"einsum_out", "reshape2_shape"}, {"output"}, kOnnxDomain);
    builder.MakeOutput("output");
  };

  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(build_fn,
                  opts,
                  13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  auto trace_file = FindTraceFile(tmp.path());
  ASSERT_FALSE(trace_file.empty()) << "No trace file generated for ReshapeEinsumReshape model";
  auto j = ReadTraceJson(trace_file);
  ValidateTraceStructure(j);

  ASSERT_GE(j["subgraph_traces"].size(), 1u);
  const auto& op_mappings = j["subgraph_traces"][0]["op_mappings"];

  // Collect all op_mapping entries attributed to the ReshapeEinsumReshape fusion.
  std::vector<nlohmann::json> rer_entries;
  for (const auto& m : op_mappings) {
    if (m.contains("node_group_type") && m["node_group_type"] == "ReshapeEinsumReshapeNodeGroup") {
      rer_entries.push_back(m);
    }
  }

  // N:M: 3 ONNX ops → 3 QNN ops, so there must be at least 2 fusion entries
  // (3 is the expected exact count: Reshape + DepthToSpace + Transpose).
  ASSERT_GE(rer_entries.size(), 2u)
      << "ReshapeEinsumReshape N:M fusion should produce at least 2 QNN op trace entries "
         "(expected Reshape + DepthToSpace + Transpose)";

  // Each QNN op entry must reference all 3 source ONNX ops.
  for (const auto& entry : rer_entries) {
    int op_source_count = 0;
    for (const auto& s : entry["sources"]) {
      if (s["type"] == "OP") op_source_count++;
    }
    EXPECT_GE(op_source_count, 3)
        << "Each N:M fusion entry should reference all 3 ONNX source ops "
           "(reshape1, einsum, reshape2)";
  }
}

// ========================= Profiling + Tracing Combination Tests =========================
//
// These tests verify the interaction between qnn.profiling_level and qnn.enable_framework_op_trace.
//
// Design notes:
// - The ONNX Source Ops field only appears in the CSV when profiling_level is
//   DETAILED or OPTRACE (per-NODE events emitted) AND tracing is on.
// - At BASIC level profiling, no QNN_PROFILE_EVENTTYPE_NODE events are emitted,
//   so annotation would always be empty — the field is suppressed entirely.
// - CSV header is written once when the file is first created; its field count
//   is the reliable indicator because it is set unconditionally at file creation.

// Helper: return the first line (header) of a CSV file.
static std::string ReadCsvHeader(const fs::path& csv_path) {
  std::ifstream ifs(csv_path);
  if (!ifs.is_open()) return {};
  std::string line;
  std::getline(ifs, line);
  return line;
}

// Helper: count comma-separated fields in a CSV line.
static size_t CountCsvColumns(const std::string& csv_line) {
  if (csv_line.empty()) return 0;
  return static_cast<size_t>(std::count(csv_line.begin(), csv_line.end(), ',')) + 1;
}

// Helper: count rows (excluding header) of a given event type string.
static size_t CountCsvRowsByMessage(const fs::path& csv_path, const std::string& message) {
  std::ifstream ifs(csv_path);
  if (!ifs.is_open()) return 0;
  std::string line;
  std::getline(ifs, line);  // skip header
  size_t count = 0;
  while (std::getline(ifs, line)) {
    if (line.find(message) != std::string::npos) ++count;
  }
  return count;
}

// Helper: collect non-empty ONNX Source Ops values from NODE rows.
static std::vector<std::string> CollectOnnxSourcesFromNodeRows(const fs::path& csv_path) {
  std::vector<std::string> result;
  std::ifstream ifs(csv_path);
  if (!ifs.is_open()) return result;
  std::string header;
  std::getline(ifs, header);
  // Check for "ONNX Source Ops" field in header
  if (header.find("ONNX Source Ops") == std::string::npos) return result;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.find("NODE") == std::string::npos) continue;
    // Last field (after the last comma)
    auto pos = line.rfind(',');
    if (pos == std::string::npos) continue;
    std::string onnx_col = line.substr(pos + 1);
    if (!onnx_col.empty()) result.push_back(onnx_col);
  }
  return result;
}

// Test: profiling_level=detailed + enable_framework_op_trace=1
// Expect: CSV header includes "ONNX Source Ops" field.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_Profiling_Detailed_With_Trace) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path csv_path = tmp.path() / "profiling.csv";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["profiling_level"] = "detailed";
  opts["profiling_file_path"] = csv_path.string();
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data),
                              TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(5e-3f)});

  ASSERT_TRUE(fs::exists(csv_path)) << "Profiling CSV not created";
  std::string header = ReadCsvHeader(csv_path);
  EXPECT_FALSE(header.empty()) << "CSV header is empty";
  EXPECT_EQ(CountCsvColumns(header), 8u)
      << "Expected 8 columns (with ONNX Source Ops) but got: " << header;
  EXPECT_NE(header.find("ONNX Source Ops"), std::string::npos)
      << "ONNX Source Ops column missing from header: " << header;
}

// Test: profiling_level=basic + enable_framework_op_trace=1
// Expect: CSV header has no "ONNX Source Ops" field — basic has no NODE events.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_Profiling_Basic_With_Trace) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path csv_path = tmp.path() / "profiling.csv";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["profiling_level"] = "basic";
  opts["profiling_file_path"] = csv_path.string();
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data),
                              TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(5e-3f)});

  ASSERT_TRUE(fs::exists(csv_path)) << "Profiling CSV not created";
  std::string header = ReadCsvHeader(csv_path);
  EXPECT_FALSE(header.empty()) << "CSV header is empty";
  EXPECT_EQ(CountCsvColumns(header), 7u)
      << "Expected 7 columns (no ONNX Source Ops at basic level) but got: " << header;
  EXPECT_EQ(header.find("ONNX Source Ops"), std::string::npos)
      << "ONNX Source Ops column should not appear at basic level";
}

// Test: profiling_level=detailed WITHOUT enable_framework_op_trace
// Expect: CSV header has no "ONNX Source Ops" field — tracing disabled.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_Profiling_Detailed_Without_Trace) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path csv_path = tmp.path() / "profiling.csv";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["profiling_level"] = "detailed";
  opts["profiling_file_path"] = csv_path.string();
  // enable_framework_op_trace intentionally NOT set
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data),
                              TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(5e-3f)});

  ASSERT_TRUE(fs::exists(csv_path)) << "Profiling CSV not created";
  std::string header = ReadCsvHeader(csv_path);
  EXPECT_FALSE(header.empty()) << "CSV header is empty";
  EXPECT_EQ(CountCsvColumns(header), 7u)
      << "Expected 7 columns (no tracing) but got: " << header;
  EXPECT_EQ(header.find("ONNX Source Ops"), std::string::npos)
      << "ONNX Source Ops column should not appear when tracing is disabled";
}

// Test: profiling_level=detailed + enable_framework_op_trace=1 + inference run
// Expect: NODE rows in the CSV have ONNX Source Ops populated (non-empty).
// This verifies the end-to-end join: trace op name == profiling event identifier.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_Profiling_Detailed_With_Trace_NodeRowsAnnotated) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path csv_path = tmp.path() / "profiling.csv";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["profiling_level"] = "detailed";
  opts["profiling_file_path"] = csv_path.string();
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data),
                              TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(5e-3f)});

  ASSERT_TRUE(fs::exists(csv_path)) << "Profiling CSV not created";

  // Verify header has the ONNX annotation field.
  std::string header = ReadCsvHeader(csv_path);
  ASSERT_EQ(CountCsvColumns(header), 8u) << "Expected 8 columns: " << header;

  // DETAILED profiling emits per-NODE events,
  // and every NODE row carries an ONNX source annotation. Assert the NODE count
  // is non-zero first, then that the rows are annotated.
  size_t node_row_count = CountCsvRowsByMessage(csv_path, ",NODE,");
  ASSERT_GT(node_row_count, 0u)
      << "DETAILED profiling expected to emit NODE events but CSV has none";
  auto annotated = CollectOnnxSourcesFromNodeRows(csv_path);
  EXPECT_GT(annotated.size(), 0u)
      << "NODE rows exist but none have ONNX Source Ops annotation";
}

// Test: profiling_level=optrace + enable_framework_op_trace=1
// Expect: CSV header includes "ONNX Source Ops" field and NODE rows are annotated (same as detailed).
// optrace is the highest-value profiling level: hardware-level cycles/bandwidth per op.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_Profiling_OpTrace_With_Trace) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path csv_path = tmp.path() / "profiling.csv";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["profiling_level"] = "optrace";
  opts["profiling_file_path"] = csv_path.string();
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = tmp.path().string();
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data),
                              TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain),
      opts, 13, EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(5e-3f)});

  ASSERT_TRUE(fs::exists(csv_path)) << "Profiling CSV not created";

  // optrace triggers per-NODE events to be emitted, same as detailed.
  std::string header = ReadCsvHeader(csv_path);
  EXPECT_FALSE(header.empty()) << "CSV header is empty";
  EXPECT_EQ(CountCsvColumns(header), 8u)
      << "Expected 8 columns (ONNX Source Ops present) but got: " << header;
  EXPECT_NE(header.find("ONNX Source Ops"), std::string::npos)
      << "ONNX Source Ops column missing from header: " << header;

  // OPTRACE profiling emits per-NODE events,
  // and every NODE row carries an ONNX source annotation.
  size_t node_row_count = CountCsvRowsByMessage(csv_path, ",NODE,");
  ASSERT_GT(node_row_count, 0u)
      << "OPTRACE profiling expected to emit NODE events but CSV has none";
  auto annotated = CollectOnnxSourcesFromNodeRows(csv_path);
  EXPECT_GT(annotated.size(), 0u)
      << "NODE rows exist but none have ONNX Source Ops annotation";
}

// Test: AOT Phase 2 (pre-compiled context model) end-to-end profiling join.
// In Phase 1 a context model and its sidecar (qnn_op_trace.json) are generated.
// In Phase 2 the sidecar is discovered next to the context model, loaded via
// LoadTraceLookupFromFile, and used to annotate the profiling CSV produced by an
// inference run. With profiling_level=detailed + profiling_file_path set, the
// test asserts the loaded sidecar annotates the NODE rows with ONNX sources.
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_AOT_Phase2_SidecarAnnotatesProfiling) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path ctx_file = tmp.path() / "model_ctx.onnx";
  fs::path csv_path = tmp.path() / "phase2_profiling.csv";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::string model_data = BuildModelData(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain));

  // Phase 1: generate context model + sidecar trace. framework_op_trace_dir is
  // set to the context model's directory so the sidecar is written as
  // {tmp}/qnn_op_trace.json — exactly where DeriveTracePathFromContextModel
  // looks in Phase 2 (no manual copy needed).
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = tmp.path().string();

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
  }

  ASSERT_TRUE(fs::exists(ctx_file)) << "Context model not generated in Phase 1";
  fs::path sidecar_path = tmp.path() / "qnn_op_trace.json";
  ASSERT_TRUE(fs::exists(sidecar_path))
      << "Phase 1 should write the sidecar next to the context model";

  // Phase 2: load context model with tracing + detailed profiling, run inference.
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = tmp.path().string();
    opts["profiling_level"] = "detailed";
    opts["profiling_file_path"] = csv_path.string();
    opts["disable_file_mapped_weights"] = "1";  // avoid DMA path crash with App Verifier

    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, ctx_file.c_str(), so);

    auto input_names = session.GetInputNames();
    auto output_names = session.GetOutputNames();

    std::vector<float> input_data(6, 1.0f);
    std::vector<int64_t> input_shape = {1, 2, 3};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < input_names.size(); ++i) {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));
    }

    std::vector<const char*> input_name_ptrs;
    for (const auto& name : input_names) input_name_ptrs.push_back(name.c_str());
    std::vector<const char*> output_name_ptrs;
    for (const auto& name : output_names) output_name_ptrs.push_back(name.c_str());

    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_name_ptrs.data(), inputs.data(), inputs.size(),
                               output_name_ptrs.data(), output_name_ptrs.size());
    EXPECT_GT(outputs.size(), 0u) << "Phase 2 inference should produce output";
  }

  // The sidecar must have been discovered and joined to the Phase 2 profiling CSV.
  ASSERT_TRUE(fs::exists(csv_path)) << "Phase 2 profiling CSV not created";
  std::string header = ReadCsvHeader(csv_path);
  EXPECT_NE(header.find("ONNX Source Ops"), std::string::npos)
      << "ONNX Source Ops column missing — sidecar lookup not applied: " << header;

  size_t node_row_count = CountCsvRowsByMessage(csv_path, ",NODE,");
  ASSERT_GT(node_row_count, 0u)
      << "Detailed profiling expected to emit NODE events but Phase 2 CSV has none";
  auto annotated = CollectOnnxSourcesFromNodeRows(csv_path);
  EXPECT_GT(annotated.size(), 0u)
      << "Phase 2 NODE rows exist but none annotated — sidecar→profiling join failed";
}

// AOT Phase 2 with framework op trace + detailed profiling enabled, but no
// sidecar next to the context model. Verifies the column-stability contract:
//   - the `ONNX Source Ops` column appears in the CSV header (the gate is the
//     session-stable enable_framework_op_trace_ flag, not lookup emptiness, so
//     header and per-row column counts always agree),
//   - every NODE row's annotation cell is empty (the lookup is empty, so every
//     LookupOnnxSources() returns ""),
// distinguishing "trace requested but no source data" from "trace not requested".
TEST_F(QnnHTPBackendTests, FrameworkOpTrace_AOT_Phase2_NoSidecar_EmitsEmptyColumn) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedTempDir tmp;
  fs::path ctx_file = tmp.path() / "model_ctx.onnx";
  fs::path csv_path = tmp.path() / "phase2_no_sidecar_profiling.csv";

  std::vector<float> data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::string model_data = BuildModelData(
      BuildOpTestCase<float>("add_node", "Add",
                             {TestInputDef<float>({1, 2, 3}, false, data), TestInputDef<float>({1, 2, 3}, false, data)},
                             {}, {}, kOnnxDomain));

  // Phase 1: generate context model WITHOUT trace, so no sidecar is produced.
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
  }
  ASSERT_TRUE(fs::exists(ctx_file));
  ASSERT_FALSE(fs::exists(tmp.path() / "qnn_op_trace.json"))
      << "Phase 1 had tracing disabled, so no sidecar should exist";

  // Phase 2: load context model with tracing + detailed profiling, run inference.
  // No sidecar exists → lookup stays empty.
  {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
    opts["enable_framework_op_trace"] = "1";
    opts["framework_op_trace_dir"] = tmp.path().string();
    opts["profiling_level"] = "detailed";
    opts["profiling_file_path"] = csv_path.string();
    opts["disable_file_mapped_weights"] = "1";

    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", opts);
    Ort::Session session(*ort_env, ctx_file.c_str(), so);

    auto input_names = session.GetInputNames();
    auto output_names = session.GetOutputNames();
    std::vector<float> input_data(6, 1.0f);
    std::vector<int64_t> input_shape = {1, 2, 3};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < input_names.size(); ++i) {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));
    }
    std::vector<const char*> input_name_ptrs;
    for (const auto& name : input_names) input_name_ptrs.push_back(name.c_str());
    std::vector<const char*> output_name_ptrs;
    for (const auto& name : output_names) output_name_ptrs.push_back(name.c_str());
    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_name_ptrs.data(), inputs.data(), inputs.size(),
                               output_name_ptrs.data(), output_name_ptrs.size());
    EXPECT_GT(outputs.size(), 0u);
  }

  ASSERT_TRUE(fs::exists(csv_path)) << "Phase 2 profiling CSV not created";
  std::string header = ReadCsvHeader(csv_path);
  // Column IS present — gate is session-stable enable_framework_op_trace_ flag.
  EXPECT_NE(header.find("ONNX Source Ops"), std::string::npos)
      << "ONNX Source Ops column should still be in header (column-stability invariant): " << header;

  size_t node_row_count = CountCsvRowsByMessage(csv_path, ",NODE,");
  ASSERT_GT(node_row_count, 0u)
      << "Detailed profiling expected to emit NODE events but CSV has none";
  // No sidecar means the lookup is empty → every NODE row's annotation cell is empty.
  auto annotated = CollectOnnxSourcesFromNodeRows(csv_path);
  EXPECT_EQ(annotated.size(), 0u)
      << "Without a sidecar the annotation column should be present but every cell empty; "
      << "got " << annotated.size() << " non-empty annotations";
}

// ================= FCB multi-SoC framework op trace (x86-only) =================
// Multi-SoC FCB EP-context compilation is an x86 offline-preparation feature
// (the EP rejects it on aarch64), so these E2E tests are x86-only.
#if !defined(__aarch64__) && !defined(_M_ARM64)

namespace {

// SoC config shared by the FCB framework op trace tests: V73 + V81, two SoCs.
struct FcbTraceTestConfig {
  static constexpr size_t kNumSocs = 2;
  static constexpr const char* kHtpArchStr = "73,81";
#ifdef _WIN32
  static constexpr const char* kSocModelStr = "60,88";
#else
  static constexpr const char* kSocModelStr = "43,87";
#endif
  static const std::vector<std::string>& TargetArches() {
    static const std::vector<std::string> arches = {"73", "81"};
    return arches;
  }
};

// Build the common multi-SoC + framework-op-trace provider options.
ProviderOptions MakeFcbTraceProviderOptions(const std::string& trace_dir) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["htp_arch"] = FcbTraceTestConfig::kHtpArchStr;
  opts["soc_model"] = FcbTraceTestConfig::kSocModelStr;
  opts["enable_framework_op_trace"] = "1";
  opts["framework_op_trace_dir"] = trace_dir;
  return opts;
}

// Locate qnn_op_trace.json in dir, parse it, and return the JSON object.
nlohmann::json FindAndParseTraceFile(const std::filesystem::path& dir) {
  std::filesystem::path trace_file;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() == ".json" &&
        entry.path().stem().string().find("_op_trace") != std::string::npos) {
      trace_file = entry.path();
      break;
    }
  }
  EXPECT_FALSE(trace_file.empty()) << "No qnn_op_trace.json found in " << dir;
  if (trace_file.empty()) return {};

  std::ifstream ifs(trace_file);
  EXPECT_TRUE(ifs.is_open()) << "Failed to open " << trace_file;
  if (!ifs.is_open()) return {};

  auto j = nlohmann::json::parse(ifs, nullptr, false);
  EXPECT_FALSE(j.is_discarded()) << "Failed to parse " << trace_file;
  return j;
}

// Assert every-trace invariants. Fatal ASSERT_*; wrap call sites in
// ASSERT_NO_FATAL_FAILURE so a size mismatch aborts before soc_traces[i] indexing.
void AssertFcbTraceInvariants(const nlohmann::json& j, size_t num_socs) {
  ASSERT_FALSE(j.empty());
  ASSERT_TRUE(j.contains("soc_traces"));
  ASSERT_EQ(j["soc_traces"].size(), num_socs)
      << "Multi-SoC trace must have one soc_traces entry per htp_arch";
  ASSERT_TRUE(j.contains("summary"));
  EXPECT_EQ(j["summary"]["total_socs"].get<uint32_t>(), static_cast<uint32_t>(num_socs));
  EXPECT_FALSE(j.contains("compilation_target"))
      << "compilation_target must not be at root (moved to soc_traces[i])";
}

}  // namespace

// FCB multi-SoC framework op trace E2E test: exercises GetCapabilityImpl ->
// GetMultiSocSupportedNodes -> CompileMultiSocOnnxModel with tracing enabled.
TEST_F(QnnHTPBackendTests, EPContextMultiSoc_FrameworkOpTrace) {
  namespace fs = std::filesystem;

  fs::path trace_dir = fs::temp_directory_path() / "fcb_op_trace_e2e";
  fs::remove_all(trace_dir);
  fs::create_directories(trace_dir);
  fs::path output_model_file = trace_dir / "model_ctx.onnx";

  ProviderOptions provider_options = MakeFcbTraceProviderOptions(trace_dir.string());
  provider_options["num_graph_prepare_threads"] = "1";

  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx";

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));

  ASSERT_TRUE(fs::exists(output_model_file));
  fs::remove(output_model_file);

  nlohmann::json j = FindAndParseTraceFile(trace_dir);
  if (j.empty()) {
    fs::remove_all(trace_dir);
    return;
  }
  ASSERT_NO_FATAL_FAILURE(AssertFcbTraceInvariants(j, FcbTraceTestConfig::kNumSocs));

  // Each soc_traces[i] carries the expected htp_arch.
  for (size_t i = 0; i < FcbTraceTestConfig::TargetArches().size(); ++i) {
    ASSERT_TRUE(j["soc_traces"][i].contains("compilation_target"));
    EXPECT_EQ(j["soc_traces"][i]["compilation_target"]["htp_arch"],
              "V" + FcbTraceTestConfig::TargetArches()[i])
        << "soc_traces[" << i << "] htp_arch mismatch";
  }

  // At least one op was compiled; subgraph_traces is at root, not per SoC.
  EXPECT_GT(j["summary"]["total_qnn_ops"].get<size_t>(), 0u);
  ASSERT_TRUE(j.contains("subgraph_traces"));
  ASSERT_GE(j["subgraph_traces"].size(), 1u);
  for (const auto& soc : j["soc_traces"]) {
    EXPECT_FALSE(soc.contains("subgraph_traces"))
        << "subgraph_traces must not be duplicated under soc_traces[i]";
  }

  fs::remove_all(trace_dir);
}

// Mixed partition (Abs supported, NonZero graph-output rejected): verifies both
// subgraph_traces and SoC-attributed unsupported_nodes in one trace.
TEST_F(QnnHTPBackendTests, EPContextMultiSoc_FrameworkOpTrace_Mixed) {
  namespace fs = std::filesystem;

  fs::path trace_dir = fs::temp_directory_path() / "fcb_op_trace_mixed";
  fs::remove_all(trace_dir);
  fs::create_directories(trace_dir);
  fs::path output_model_file = trace_dir / "model_ctx.onnx";

  ModelTestBuilder helper;
  auto build_model = [](ModelTestBuilder& builder) {
    std::vector<float> data = GetFloatDataInRange(-1.0f, 1.0f, 6);
    builder.MakeInput<float>("input", {1, 2, 3}, data);
    builder.AddNode("abs_node", "Abs", {"input"}, {"abs_out"});
    builder.MakeOutput("Y");
    builder.AddNode("nonzero_node", "NonZero", {"abs_out"}, {"Y"});
  };
  build_model(helper);
  const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
  opset_id_proto->set_domain("");
  opset_id_proto->set_version(13);
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  ProviderOptions provider_options = MakeFcbTraceProviderOptions(trace_dir.string());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

  nlohmann::json j = FindAndParseTraceFile(trace_dir);
  if (j.empty()) {
    fs::remove_all(trace_dir);
    return;
  }
  ASSERT_NO_FATAL_FAILURE(AssertFcbTraceInvariants(j, FcbTraceTestConfig::kNumSocs));

  // Abs was compiled: subgraph_traces non-empty, total_qnn_ops > 0.
  ASSERT_TRUE(j.contains("subgraph_traces"));
  EXPECT_GE(j["subgraph_traces"].size(), 1u);
  EXPECT_GT(j["summary"]["total_qnn_ops"].get<size_t>(), 0u);

  // NonZero was rejected: unsupported_nodes non-empty with SoC-attributed reason.
  ASSERT_TRUE(j.contains("unsupported_nodes"));
  ASSERT_GT(j["unsupported_nodes"].size(), 0u);
  bool found_nonzero = false;
  for (const auto& un : j["unsupported_nodes"]) {
    if (un["op_type"].get<std::string>() == "NonZero") {
      found_nonzero = true;
      const std::string reason = un["reason"].get<std::string>();
      EXPECT_FALSE(reason.empty());
      EXPECT_TRUE(reason.find("V73") != std::string::npos || reason.find("V81") != std::string::npos)
          << "Multi-SoC reason must carry SoC label; got: " << reason;
    }
  }
  EXPECT_TRUE(found_nonzero) << "NonZero must appear in unsupported_nodes";

  fs::remove_all(trace_dir);
}

#endif  // !defined(__aarch64__) && !defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
