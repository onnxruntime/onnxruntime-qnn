// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for onnx_ctx_model_helper.cc.
//
// Tests the EP-context-node-detection helpers using FakeGraph / FakeNode /
// FakeOpAttr from qnn_fake_ort_graph.h.
//
// OrtNodeAttrHelper (used by these helpers) reads attributes through
// Ort::ConstNode(&node).GetAttributeByName(...), which routes through the
// global Ort C++ API. Tests install OrtGlobalApiOverride so the wrapper
// calls reach our stubs instead of the real ORT runtime.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <vector>

#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "test/providers/qnn/unit/qnn_fake_ort_graph.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

using namespace onnxruntime::qnn;
using namespace onnxruntime::test;

namespace {

// Context fixture: installs the FakeGraph stubs + global API override so the
// OrtNodeAttrHelper path (via Ort::ConstNode wrappers) reaches our stubs.
struct CtxHelperTestContext {
  OrtApi api{};
  // global_guard restores the original global API on destruction.
  OrtGlobalApiOverride global_guard;

  CtxHelperTestContext()
      : api(),
        global_guard((InstallFakeGraphApiStubs(api), &api)) {}
};

}  // namespace

// =============================================================================
// GraphHasEpContextNode
//
// Logic:
//   1. Iterate every node in the graph
//   2. Match op_type == "EPContext"
//   3. Read SOURCE attr (default "") and check against accepted values
//      ("qnn", "qnnexecutionprovider", "qairtexport")
//   4. Read EP_CONTEXT_TYPE attr (default EP_CONTEXT_TYPE_BIN="bin") and
//      check it equals the requested ep_context_type
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_EmptyGraph_ReturnsFalse) {
  CtxHelperTestContext ctx;
  FakeGraph graph{};
  EXPECT_FALSE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_NoEpContextOp_ReturnsFalse) {
  CtxHelperTestContext ctx;
  FakeNode node{"some_node", "Relu", "", 13, {}, {}};
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_FALSE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_NoSourceAttr_ReturnsFalse) {
  CtxHelperTestContext ctx;
  // EPContext node with no SOURCE attribute → falls back to default "" →
  // does not match any accepted source string → returns false.
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_FALSE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_QnnSource_TypeBin_ReturnsTrue) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  // EP_CONTEXT_TYPE attr omitted → defaults to EP_CONTEXT_TYPE_BIN.
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_TRUE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_QnnExecutionProviderSource_ReturnsTrue) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "QNNExecutionProvider");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_TRUE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_QairtexportSource_ReturnsTrue) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qairtexport");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_TRUE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_UnknownSource_ReturnsFalse) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "some_other_ep");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_FALSE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_QnnSourceTypeDlc_ReturnsTrueForDlc) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeOpAttr ctx_type = FakeOpAttr::MakeString(EP_CONTEXT_TYPE, "dlc");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  node.attrs[EP_CONTEXT_TYPE] = &ctx_type;
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_TRUE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_DLC));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_QnnSourceTypeDlc_ReturnsFalseForBin) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeOpAttr ctx_type = FakeOpAttr::MakeString(EP_CONTEXT_TYPE, "dlc");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  node.attrs[EP_CONTEXT_TYPE] = &ctx_type;
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_FALSE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasEpContextNode_MixedNodes_MatchesEpContextOnly) {
  CtxHelperTestContext ctx;
  // Graph has a Relu before the EPContext; iteration must reach the second
  // node to find the match.
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeNode relu{"r", "Relu", "", 13, {}, {}};
  FakeNode ep_ctx{"ep", "EPContext", "", 1, {}, {}};
  ep_ctx.attrs[SOURCE] = &source;
  FakeGraph graph{{relu, ep_ctx}, {}, {}, {}};
  EXPECT_TRUE(GraphHasEpContextNode(graph.AsGraph(), ctx.api, EP_CONTEXT_TYPE_BIN));
}

// =============================================================================
// GraphHasDlcContextNode
//
// Thin wrapper around GraphHasEpContextNode(..., EP_CONTEXT_TYPE_DLC).
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasDlcContextNode_DlcType_ReturnsTrue) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeOpAttr ctx_type = FakeOpAttr::MakeString(EP_CONTEXT_TYPE, "dlc");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  node.attrs[EP_CONTEXT_TYPE] = &ctx_type;
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_TRUE(GraphHasDlcContextNode(graph.AsGraph(), ctx.api));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GraphHasDlcContextNode_BinType_ReturnsFalse) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  // No EP_CONTEXT_TYPE attr → defaults to "bin".
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  FakeGraph graph{{node}, {}, {}, {}};
  EXPECT_FALSE(GraphHasDlcContextNode(graph.AsGraph(), ctx.api));
}

// =============================================================================
// IsOrtGraphHasCtxNode
//
// Iterates an array of OrtGraph*, returns true if any graph has a matching
// EP context node.
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, IsOrtGraphHasCtxNode_ZeroGraphs_ReturnsFalse) {
  CtxHelperTestContext ctx;
  EXPECT_FALSE(IsOrtGraphHasCtxNode(nullptr, 0, ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, IsOrtGraphHasCtxNode_NoMatch_ReturnsFalse) {
  CtxHelperTestContext ctx;
  FakeNode relu{"r", "Relu", "", 13, {}, {}};
  FakeGraph g0{{relu}, {}, {}, {}};
  FakeGraph g1{{}, {}, {}, {}};
  const OrtGraph* graphs[] = {g0.AsGraph(), g1.AsGraph()};
  EXPECT_FALSE(IsOrtGraphHasCtxNode(graphs, 2, ctx.api, EP_CONTEXT_TYPE_BIN));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, IsOrtGraphHasCtxNode_SecondGraphMatches_ReturnsTrue) {
  CtxHelperTestContext ctx;
  FakeNode relu{"r", "Relu", "", 13, {}, {}};
  FakeGraph g0{{relu}, {}, {}, {}};
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeNode ep_ctx{"e", "EPContext", "", 1, {}, {}};
  ep_ctx.attrs[SOURCE] = &source;
  FakeGraph g1{{ep_ctx}, {}, {}, {}};
  const OrtGraph* graphs[] = {g0.AsGraph(), g1.AsGraph()};
  EXPECT_TRUE(IsOrtGraphHasCtxNode(graphs, 2, ctx.api, EP_CONTEXT_TYPE_BIN));
}

// =============================================================================
// IsOrtGraphHasDlcCtxNode
//
// Thin wrapper that delegates to IsOrtGraphHasCtxNode with EP_CONTEXT_TYPE_DLC.
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, IsOrtGraphHasDlcCtxNode_DlcMatch_ReturnsTrue) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeOpAttr ctx_type = FakeOpAttr::MakeString(EP_CONTEXT_TYPE, "dlc");
  FakeNode ep_ctx{"e", "EPContext", "", 1, {}, {}};
  ep_ctx.attrs[SOURCE] = &source;
  ep_ctx.attrs[EP_CONTEXT_TYPE] = &ctx_type;
  FakeGraph g{{ep_ctx}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  EXPECT_TRUE(IsOrtGraphHasDlcCtxNode(graphs, 1, ctx.api));
}

TEST(QnnUnit_OnnxCtxModelHelperTest, IsOrtGraphHasDlcCtxNode_BinOnly_ReturnsFalse) {
  // BIN-type EPContext node (no EP_CONTEXT_TYPE attr → defaults to "bin") →
  // the DLC delegation must return false.
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeNode ep_ctx{"e", "EPContext", "", 1, {}, {}};
  ep_ctx.attrs[SOURCE] = &source;
  FakeGraph g{{ep_ctx}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  EXPECT_FALSE(IsOrtGraphHasDlcCtxNode(graphs, 1, ctx.api));
}

// =============================================================================
// GetEpContextDlcPath
//
// Scans graphs for a DLC EPContext node and extracts the "ep_dlc_context"
// attribute as the DLC path. Returns error if no path is found.
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextDlcPath_ZeroGraphs_ReturnsError) {
  CtxHelperTestContext ctx;
  std::string dlc_path;
  auto status = GetEpContextDlcPath(nullptr, 0, ctx.api, dlc_path);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextDlcPath_NoDlcNode_ReturnsError) {
  CtxHelperTestContext ctx;
  // BIN-type node only — GetEpContextDlcPath skips it.
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  FakeGraph g{{node}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::string dlc_path;
  auto status = GetEpContextDlcPath(graphs, 1, ctx.api, dlc_path);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextDlcPath_DlcNodeNoPathAttr_ReturnsError) {
  CtxHelperTestContext ctx;
  // DLC node but no "ep_dlc_context" attribute → empty string → error.
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeOpAttr ctx_type = FakeOpAttr::MakeString(EP_CONTEXT_TYPE, "dlc");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  node.attrs[EP_CONTEXT_TYPE] = &ctx_type;
  FakeGraph g{{node}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::string dlc_path;
  auto status = GetEpContextDlcPath(graphs, 1, ctx.api, dlc_path);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextDlcPath_DlcNodeWithPath_ReturnsPath) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeOpAttr ctx_type = FakeOpAttr::MakeString(EP_CONTEXT_TYPE, "dlc");
  FakeOpAttr dlc_ctx = FakeOpAttr::MakeString("ep_dlc_context", "/path/to/model.dlc");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  node.attrs[EP_CONTEXT_TYPE] = &ctx_type;
  node.attrs["ep_dlc_context"] = &dlc_ctx;
  FakeGraph g{{node}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::string dlc_path;
  auto status = GetEpContextDlcPath(graphs, 1, ctx.api, dlc_path);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(dlc_path, "/path/to/model.dlc");
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextDlcPath_SecondGraphHasDlcNode_ReturnsPath) {
  CtxHelperTestContext ctx;
  // First graph is BIN, second is DLC with a path.
  FakeOpAttr source_bin = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeNode bin_node{"ep_bin", "EPContext", "", 1, {}, {}};
  bin_node.attrs[SOURCE] = &source_bin;
  FakeGraph g0{{bin_node}, {}, {}, {}};

  FakeOpAttr source_dlc = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeOpAttr ctx_type = FakeOpAttr::MakeString(EP_CONTEXT_TYPE, "dlc");
  FakeOpAttr dlc_ctx = FakeOpAttr::MakeString("ep_dlc_context", "model.dlc");
  FakeNode dlc_node{"ep_dlc", "EPContext", "", 1, {}, {}};
  dlc_node.attrs[SOURCE] = &source_dlc;
  dlc_node.attrs[EP_CONTEXT_TYPE] = &ctx_type;
  dlc_node.attrs["ep_dlc_context"] = &dlc_ctx;
  FakeGraph g1{{dlc_node}, {}, {}, {}};

  const OrtGraph* graphs[] = {g0.AsGraph(), g1.AsGraph()};
  std::string dlc_path;
  auto status = GetEpContextDlcPath(graphs, 2, ctx.api, dlc_path);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(dlc_path, "model.dlc");
}

// =============================================================================
// TryGetMaxSpillFillSize
//
// Iterates main_context_pos_list, reads MAX_SIZE from each EPContext node, and
// swaps the largest to position 0 in main_context_pos_list (L270: swap path).
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, TryGetMaxSpillFillSize_ZeroContexts_ReturnsOk) {
  CtxHelperTestContext ctx;
  FakeGraph g{{}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::vector<int> pos_list;
  int64_t max_size = 0;
  auto status = TryGetMaxSpillFillSize(graphs, ctx.api, 0, max_size, pos_list);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(max_size, 0);
}

TEST(QnnUnit_OnnxCtxModelHelperTest, TryGetMaxSpillFillSize_SingleContext_NoSwap) {
  CtxHelperTestContext ctx;
  FakeOpAttr max_size_attr = FakeOpAttr::MakeInt64(MAX_SIZE, 100);
  FakeNode ep_ctx{"e", "EPContext", "", 1, {}, {}};
  ep_ctx.attrs[MAX_SIZE] = &max_size_attr;
  FakeGraph g{{ep_ctx}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::vector<int> pos_list = {0};
  int64_t max_size = 0;
  auto status = TryGetMaxSpillFillSize(graphs, ctx.api, 1, max_size, pos_list);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(max_size, 100);
  EXPECT_EQ(pos_list[0], 0);  // no swap needed
}

TEST(QnnUnit_OnnxCtxModelHelperTest, TryGetMaxSpillFillSize_SecondContextLarger_SwapsToFront) {
  CtxHelperTestContext ctx;
  // g0: MAX_SIZE=50, g1: MAX_SIZE=200. pos_list=[0,1] → after swap: [1,0].
  FakeOpAttr size0 = FakeOpAttr::MakeInt64(MAX_SIZE, 50);
  FakeNode ep0{"e0", "EPContext", "", 1, {}, {}};
  ep0.attrs[MAX_SIZE] = &size0;
  FakeGraph g0{{ep0}, {}, {}, {}};

  FakeOpAttr size1 = FakeOpAttr::MakeInt64(MAX_SIZE, 200);
  FakeNode ep1{"e1", "EPContext", "", 1, {}, {}};
  ep1.attrs[MAX_SIZE] = &size1;
  FakeGraph g1{{ep1}, {}, {}, {}};

  const OrtGraph* graphs[] = {g0.AsGraph(), g1.AsGraph()};
  std::vector<int> pos_list = {0, 1};
  int64_t max_size = 0;
  auto status = TryGetMaxSpillFillSize(graphs, ctx.api, 2, max_size, pos_list);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(max_size, 200);
  EXPECT_EQ(pos_list[0], 1);  // swapped: largest is now first
  EXPECT_EQ(pos_list[1], 0);
}

// =============================================================================
// ParseIoNameOverrides
//
// Parses the IO_NAME_OVERRIDES attribute from an EPContext node into an
// internal→external name map. Edge cases:
//   - nullptr node           → L122: early return empty map
//   - no trailing semicolon  → L131: sep = encoded.size()
//   - consecutive semicolons → L136: empty pair → continue
//   - pair with no '='       → L140: malformed pair → continue
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_NullNode_ReturnsEmpty) {
  // L122: ep_context_node == nullptr → returns empty map immediately.
  auto overrides = ParseIoNameOverrides(nullptr);
  EXPECT_TRUE(overrides.empty());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_SinglePairNoTrailingSemicolon_Parsed) {
  CtxHelperTestContext ctx;
  // "a=b" — no trailing semicolon; sep falls back to encoded.size() (L131).
  FakeOpAttr attr = FakeOpAttr::MakeString(IO_NAME_OVERRIDES, "a=b");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[IO_NAME_OVERRIDES] = &attr;
  auto overrides = ParseIoNameOverrides(node.AsNode());
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("a"), "b");
}

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_EmptyPair_Skipped) {
  CtxHelperTestContext ctx;
  // ";a=b" — leading semicolon produces an empty first pair (L136: continue).
  FakeOpAttr attr = FakeOpAttr::MakeString(IO_NAME_OVERRIDES, ";a=b");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[IO_NAME_OVERRIDES] = &attr;
  auto overrides = ParseIoNameOverrides(node.AsNode());
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("a"), "b");
}

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_MalformedPairNoEquals_Skipped) {
  CtxHelperTestContext ctx;
  // "noeq;a=b" — first pair has no '=' (L140: continue). Second is valid.
  FakeOpAttr attr = FakeOpAttr::MakeString(IO_NAME_OVERRIDES, "noeq;a=b");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[IO_NAME_OVERRIDES] = &attr;
  auto overrides = ParseIoNameOverrides(node.AsNode());
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("a"), "b");
}

// =============================================================================
// GetMainContextNode
//
// Scans an array of graphs. Each graph must contain exactly one EPContext node.
// Collects the graph indices where main_context==1. Returns error if none found.
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, GetMainContextNode_ZeroGraphs_ReturnsError) {
  // count=0 → loop never executes → L115: pos empty → error.
  CtxHelperTestContext ctx;
  std::vector<int> pos;
  auto status = GetMainContextNode(nullptr, 0, ctx.api, pos);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetMainContextNode_GraphWithTwoNodes_ReturnsError) {
  // L97: num_nodes != 1 → error.
  CtxHelperTestContext ctx;
  FakeNode n0{"n0", "Relu", "", 13, {}, {}};
  FakeNode n1{"n1", "Relu", "", 13, {}, {}};
  FakeGraph g{{n0, n1}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::vector<int> pos;
  auto status = GetMainContextNode(graphs, 1, ctx.api, pos);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetMainContextNode_WrongOpType_ReturnsError) {
  // L106: op_type != EPCONTEXT_OP → error.
  CtxHelperTestContext ctx;
  FakeNode node{"relu", "Relu", "", 13, {}, {}};
  FakeGraph g{{node}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::vector<int> pos;
  auto status = GetMainContextNode(graphs, 1, ctx.api, pos);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetMainContextNode_NoMainContextAttr_ReturnsError) {
  // No MAIN_CONTEXT attr → defaults to 0 → not marked main → L115 error.
  CtxHelperTestContext ctx;
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  FakeGraph g{{node}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::vector<int> pos;
  auto status = GetMainContextNode(graphs, 1, ctx.api, pos);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetMainContextNode_MainContextOne_ReturnsPosition) {
  CtxHelperTestContext ctx;
  FakeOpAttr main_ctx_attr = FakeOpAttr::MakeInt64(MAIN_CONTEXT, 1);
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[MAIN_CONTEXT] = &main_ctx_attr;
  FakeGraph g{{node}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::vector<int> pos;
  auto status = GetMainContextNode(graphs, 1, ctx.api, pos);
  EXPECT_TRUE(status.IsOK());
  ASSERT_EQ(pos.size(), 1u);
  EXPECT_EQ(pos[0], 0);
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetMainContextNode_TwoGraphsSecondIsMain_ReturnsPosOne) {
  // First graph: main_context absent (defaults to 0). Second: main_context=1.
  CtxHelperTestContext ctx;
  FakeNode ep0{"ep0", "EPContext", "", 1, {}, {}};
  FakeGraph g0{{ep0}, {}, {}, {}};

  FakeOpAttr main_ctx_attr = FakeOpAttr::MakeInt64(MAIN_CONTEXT, 1);
  FakeNode ep1{"ep1", "EPContext", "", 1, {}, {}};
  ep1.attrs[MAIN_CONTEXT] = &main_ctx_attr;
  FakeGraph g1{{ep1}, {}, {}, {}};

  const OrtGraph* graphs[] = {g0.AsGraph(), g1.AsGraph()};
  std::vector<int> pos;
  auto status = GetMainContextNode(graphs, 2, ctx.api, pos);
  EXPECT_TRUE(status.IsOK());
  ASSERT_GE(pos.size(), 1u);
  EXPECT_EQ(pos[0], 1);
}

// =============================================================================
// GetEpContextFromMainNode — path validation (no QnnBackendManager needed)
//
// Unit-testable error paths that fire before any filesystem or backend access:
//   L159: wrong op_type
//   L178: embed_mode=false, ep_cache_context empty (default "")
//   L194: embed_mode=false, absolute path (starts with '/')
//   L198: embed_mode=false, path contains ".."
//
// Deliberately NOT unit-tested here (deferred to integration tests):
//   - embed_mode=true: calls QnnBackendManager::LoadCachedQnnContextFromBuffer,
//     which requires a real backend instance (nullptr is passed here on purpose,
//     so only the pre-backend guards above are reachable).
//   - the success path after a valid relative path: reads the cache file from
//     disk and hands the buffer to the backend — filesystem + backend territory.
// These paths depend on a live QnnBackendManager and real file I/O that the
// OrtApi-stub harness cannot fake meaningfully, so they belong to the
// end-to-end EP-context integration suite rather than this component test.
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextFromMainNode_WrongOpType_ReturnsError) {
  // L159: op_type != EPCONTEXT_OP → error before any attr or path access.
  CtxHelperTestContext ctx;
  FakeNode node{"relu", "Relu", "", 13, {}, {}};
  QnnModelLookupTable models;
  auto status = GetEpContextFromMainNode(node.AsNode(), ctx.api, "/model.onnx", nullptr, models, 0);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextFromMainNode_NonEmbedEmptyPath_ReturnsError) {
  // embed_mode=0, EP_CACHE_CONTEXT absent → defaults to "" → empty-path guard.
  CtxHelperTestContext ctx;
  FakeOpAttr embed_mode = FakeOpAttr::MakeInt64(EMBED_MODE, 0);
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[EMBED_MODE] = &embed_mode;
  QnnModelLookupTable models;
  auto status = GetEpContextFromMainNode(node.AsNode(), ctx.api, "/model.onnx", nullptr, models, 0);
  EXPECT_FALSE(status.IsOK());
  // Pin the specific guard: without this the terminal is_regular_file() check
  // catches every path case, so the test would pass even if the empty-path
  // guard were removed.
  EXPECT_NE(status.GetErrorMessage().find("should not be empty"), std::string::npos)
      << status.GetErrorMessage();
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextFromMainNode_NonEmbedAbsolutePath_ReturnsError) {
  // embed_mode=0, path starts with '/' → rejected by the absolute-path guard.
  CtxHelperTestContext ctx;
  FakeOpAttr embed_mode = FakeOpAttr::MakeInt64(EMBED_MODE, 0);
  FakeOpAttr cache_ctx = FakeOpAttr::MakeString(EP_CACHE_CONTEXT, "/absolute/path.bin");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[EMBED_MODE] = &embed_mode;
  node.attrs[EP_CACHE_CONTEXT] = &cache_ctx;
  QnnModelLookupTable models;
  auto status = GetEpContextFromMainNode(node.AsNode(), ctx.api, "/model.onnx", nullptr, models, 0);
  EXPECT_FALSE(status.IsOK());
  // Pin the absolute-path (directory-traversal) guard: removing it lets the
  // path fall through to the "does not exist" check, which this substring
  // assertion would catch.
  EXPECT_NE(status.GetErrorMessage().find("absolute path"), std::string::npos)
      << status.GetErrorMessage();
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextFromMainNode_NonEmbedDotDotPath_ReturnsError) {
  // embed_mode=0, path contains ".." → rejected by the directory-traversal guard.
  CtxHelperTestContext ctx;
  FakeOpAttr embed_mode = FakeOpAttr::MakeInt64(EMBED_MODE, 0);
  FakeOpAttr cache_ctx = FakeOpAttr::MakeString(EP_CACHE_CONTEXT, "../outside.bin");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[EMBED_MODE] = &embed_mode;
  node.attrs[EP_CACHE_CONTEXT] = &cache_ctx;
  QnnModelLookupTable models;
  auto status = GetEpContextFromMainNode(node.AsNode(), ctx.api, "/model.onnx", nullptr, models, 0);
  EXPECT_FALSE(status.IsOK());
  // Pin the ".." guard by both code and message. Removing it lets the path
  // fall through to the terminal "does not exist" check (also ORT_INVALID_GRAPH),
  // so the message substring is what actually distinguishes this guard.
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_GRAPH);
  EXPECT_NE(status.GetErrorMessage().find("'..'"), std::string::npos)
      << status.GetErrorMessage();
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextFromMainNode_NonEmbedFileNotFound_ReturnsError) {
  // embed_mode=0, valid relative path but file does not exist → L206-208 error.
  // No real file is needed; std::filesystem::is_regular_file returns false for
  // a non-existent path, triggering the "does not exist or is not accessible" guard.
  CtxHelperTestContext ctx;
  FakeOpAttr embed_mode = FakeOpAttr::MakeInt64(EMBED_MODE, 0);
  FakeOpAttr cache_ctx = FakeOpAttr::MakeString(EP_CACHE_CONTEXT, "nonexistent_ctx.bin");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[EMBED_MODE] = &embed_mode;
  node.attrs[EP_CACHE_CONTEXT] = &cache_ctx;
  QnnModelLookupTable models;
  auto status = GetEpContextFromMainNode(node.AsNode(), ctx.api, "/model.onnx", nullptr, models, 0);
  EXPECT_FALSE(status.IsOK());
}

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
