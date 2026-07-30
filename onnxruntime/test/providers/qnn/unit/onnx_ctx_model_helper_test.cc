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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "HTP/QnnHtpDevice.h"

#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "test/providers/qnn/unit/qnn_fake_ort_graph.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

// Specific using-declarations (not `using namespace`) so each helper/constant
// pulled from onnxruntime::qnn is named explicitly. The `qnn` alias only
// shortens these declarations; call sites still use the bare names. Fake* graph
// types and the stub helpers already live in onnxruntime::test, this file's own
// namespace.
namespace qnn = onnxruntime::qnn;
using qnn::CreateEPContextNodes;
using qnn::EMBED_MODE;
using qnn::EP_CACHE_CONTEXT;
using qnn::EP_CONTEXT_TYPE;
using qnn::EP_CONTEXT_TYPE_BIN;
using qnn::EP_CONTEXT_TYPE_DLC;
using qnn::EP_DLC_CONTEXT;
using qnn::GetEpContextDlcPath;
using qnn::GetEpContextFromMainNode;
using qnn::GetMainContextNode;
using qnn::GraphHasDlcContextNode;
using qnn::GraphHasEpContextNode;
using qnn::IO_NAME_OVERRIDES;
using qnn::IS_MULTI_SOC_EP_CONTEXT;
using qnn::IsOrtGraphHasCtxNode;
using qnn::IsOrtGraphHasDlcCtxNode;
using qnn::MAIN_CONTEXT;
using qnn::MAX_SIZE;
using qnn::ParseIoNameOverrides;
using qnn::QnnModelLookupTable;
using qnn::SOURCE;
using qnn::TryGetMaxSpillFillSize;

namespace {

// Context fixture: installs the FakeGraph stubs + global API override so the
// OrtNodeAttrHelper path (via Ort::ConstNode wrappers) reaches our stubs.
struct CtxHelperTestContext {
  OrtApi api{};
  // global_guard records &api and restores the original global API on
  // destruction. It only stores the pointer (the API is dereferenced at call
  // time), so filling in the stubs in the ctor body afterwards is safe and
  // avoids depending on member-declaration order for the install call.
  OrtGlobalApiOverride global_guard{&api};

  CtxHelperTestContext() { InstallFakeGraphApiStubs(api); }
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

TEST(QnnUnit_OnnxCtxModelHelperTest, GetEpContextDlcPath_DlcNodeWithPath_ReturnsLowercasedPath) {
  CtxHelperTestContext ctx;
  FakeOpAttr source = FakeOpAttr::MakeString(SOURCE, "qnn");
  FakeOpAttr ctx_type = FakeOpAttr::MakeString(EP_CONTEXT_TYPE, "dlc");
  // Mixed-case input pins the source's GetLowercaseString call; an
  // already-lowercase input could not distinguish it.
  FakeOpAttr dlc_ctx = FakeOpAttr::MakeString(EP_DLC_CONTEXT, "Path/To/Model.DLC");
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[SOURCE] = &source;
  node.attrs[EP_CONTEXT_TYPE] = &ctx_type;
  node.attrs[EP_DLC_CONTEXT] = &dlc_ctx;
  FakeGraph g{{node}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::string dlc_path;
  auto status = GetEpContextDlcPath(graphs, 1, ctx.api, dlc_path);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(dlc_path, "path/to/model.dlc");
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
  FakeOpAttr dlc_ctx = FakeOpAttr::MakeString(EP_DLC_CONTEXT, "model.dlc");
  FakeNode dlc_node{"ep_dlc", "EPContext", "", 1, {}, {}};
  dlc_node.attrs[SOURCE] = &source_dlc;
  dlc_node.attrs[EP_CONTEXT_TYPE] = &ctx_type;
  dlc_node.attrs[EP_DLC_CONTEXT] = &dlc_ctx;
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
// swaps the largest to position 0 in main_context_pos_list.
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

TEST(QnnUnit_OnnxCtxModelHelperTest, TryGetMaxSpillFillSize_NonConsecutivePosList_IndexesThroughPosList) {
  // Regression guard for the `graphs[main_context_pos_list[idx]]` indirection.
  // pos_list is non-identity ({2, 0}) and graph 1 — which the
  // pos_list never references — carries the largest MAX_SIZE as a trap.
  // Correct code reads graphs 2 then 0 (max=500, already at front → no swap).
  // A mutation to `graphs[idx]` would instead read graphs 0 then 1, pick up the
  // 999 trap, and swap the pos_list — so both the returned size and the final
  // pos_list order diverge, catching the regression.
  CtxHelperTestContext ctx;
  FakeOpAttr size0 = FakeOpAttr::MakeInt64(MAX_SIZE, 100);
  FakeNode ep0{"e0", "EPContext", "", 1, {}, {}};
  ep0.attrs[MAX_SIZE] = &size0;
  FakeGraph g0{{ep0}, {}, {}, {}};

  FakeOpAttr size1 = FakeOpAttr::MakeInt64(MAX_SIZE, 999);  // trap: only reachable via graphs[idx]
  FakeNode ep1{"e1", "EPContext", "", 1, {}, {}};
  ep1.attrs[MAX_SIZE] = &size1;
  FakeGraph g1{{ep1}, {}, {}, {}};

  FakeOpAttr size2 = FakeOpAttr::MakeInt64(MAX_SIZE, 500);
  FakeNode ep2{"e2", "EPContext", "", 1, {}, {}};
  ep2.attrs[MAX_SIZE] = &size2;
  FakeGraph g2{{ep2}, {}, {}, {}};

  const OrtGraph* graphs[] = {g0.AsGraph(), g1.AsGraph(), g2.AsGraph()};
  std::vector<int> pos_list = {2, 0};
  int64_t max_size = 0;
  auto status = TryGetMaxSpillFillSize(graphs, ctx.api, 2, max_size, pos_list);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(max_size, 500);   // graphs[2], NOT the graphs[1] trap (999)
  EXPECT_EQ(pos_list[0], 2);  // largest already at front → no swap
  EXPECT_EQ(pos_list[1], 0);
}

TEST(QnnUnit_OnnxCtxModelHelperTest, TryGetMaxSpillFillSize_GraphWithMultipleNodes_ReturnsError) {
  // Each main-context graph must contain exactly one EPContext node; a graph
  // whose node count is not 1 is rejected before any MAX_SIZE read. Two nodes
  // exercises that guard (ZeroContexts_ReturnsOk above instead skips the loop
  // via count=0, so it never reaches this check).
  CtxHelperTestContext ctx;
  FakeNode n0{"n0", "EPContext", "", 1, {}, {}};
  FakeNode n1{"n1", "EPContext", "", 1, {}, {}};
  FakeGraph g{{n0, n1}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::vector<int> pos_list = {0};
  int64_t max_size = 0;
  auto status = TryGetMaxSpillFillSize(graphs, ctx.api, 1, max_size, pos_list);
  EXPECT_FALSE(status.IsOK());
}

// =============================================================================
// ParseIoNameOverrides
//
// Parses the IO_NAME_OVERRIDES attribute from an EPContext node into an
// internal→external name map. Edge cases:
//   - nullptr node           → early return empty map
//   - no trailing semicolon  → separator falls back to encoded.size()
//   - consecutive semicolons → empty pair → continue
//   - pair with no '='       → malformed pair → continue
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_NullNode_ReturnsEmpty) {
  // ep_context_node == nullptr → returns empty map immediately.
  auto overrides = ParseIoNameOverrides(nullptr);
  EXPECT_TRUE(overrides.empty());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_SinglePairNoTrailingSemicolon_Parsed) {
  CtxHelperTestContext ctx;
  // "a=b" — no trailing semicolon; separator falls back to encoded.size().
  FakeOpAttr attr = FakeOpAttr::MakeString(IO_NAME_OVERRIDES, "a=b");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[IO_NAME_OVERRIDES] = &attr;
  auto overrides = ParseIoNameOverrides(node.AsNode());
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("a"), "b");
}

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_EmptyPair_Skipped) {
  CtxHelperTestContext ctx;
  // ";a=b" — leading semicolon produces an empty first pair (skipped via continue).
  FakeOpAttr attr = FakeOpAttr::MakeString(IO_NAME_OVERRIDES, ";a=b");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[IO_NAME_OVERRIDES] = &attr;
  auto overrides = ParseIoNameOverrides(node.AsNode());
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("a"), "b");
}

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_MalformedPairNoEquals_Skipped) {
  CtxHelperTestContext ctx;
  // "noeq;a=b" — first pair has no '=' (skipped via continue). Second is valid.
  FakeOpAttr attr = FakeOpAttr::MakeString(IO_NAME_OVERRIDES, "noeq;a=b");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[IO_NAME_OVERRIDES] = &attr;
  auto overrides = ParseIoNameOverrides(node.AsNode());
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("a"), "b");
}

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_MultiplePairs_AllParsed) {
  CtxHelperTestContext ctx;
  // "a=b;c=d" — two valid pairs; the decode loop must yield both.
  FakeOpAttr attr = FakeOpAttr::MakeString(IO_NAME_OVERRIDES, "a=b;c=d");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[IO_NAME_OVERRIDES] = &attr;
  auto overrides = ParseIoNameOverrides(node.AsNode());
  ASSERT_EQ(overrides.size(), 2u);
  EXPECT_EQ(overrides.at("a"), "b");
  EXPECT_EQ(overrides.at("c"), "d");
}

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_EqualsInsideValue_SplitsOnFirst) {
  CtxHelperTestContext ctx;
  // "a=b=c" — pair.find('=') splits on the FIRST '=', so internal="a",
  // external="b=c" (the remaining '=' stays in the value).
  FakeOpAttr attr = FakeOpAttr::MakeString(IO_NAME_OVERRIDES, "a=b=c");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[IO_NAME_OVERRIDES] = &attr;
  auto overrides = ParseIoNameOverrides(node.AsNode());
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("a"), "b=c");
}

TEST(QnnUnit_OnnxCtxModelHelperTest, ParseIoNameOverrides_EmptyInternalOrExternal_Skipped) {
  CtxHelperTestContext ctx;
  // "=b" has an empty internal, "a=" has an empty external; both fail the
  // !internal.empty() && !external.empty() guard and are skipped. Only the
  // fully-populated "c=d" survives.
  FakeOpAttr attr = FakeOpAttr::MakeString(IO_NAME_OVERRIDES, "=b;a=;c=d");
  FakeNode node{"ep", "EPContext", "", 1, {}, {}};
  node.attrs[IO_NAME_OVERRIDES] = &attr;
  auto overrides = ParseIoNameOverrides(node.AsNode());
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("c"), "d");
}

// =============================================================================
// GetMainContextNode
//
// Scans an array of graphs. Each graph must contain exactly one EPContext node.
// Collects the graph indices where main_context==1. Returns error if none found.
// =============================================================================

TEST(QnnUnit_OnnxCtxModelHelperTest, GetMainContextNode_ZeroGraphs_ReturnsError) {
  // count=0 → loop never executes → pos empty → error.
  CtxHelperTestContext ctx;
  std::vector<int> pos;
  auto status = GetMainContextNode(nullptr, 0, ctx.api, pos);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetMainContextNode_GraphWithTwoNodes_ReturnsError) {
  // num_nodes != 1 → error.
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
  // op_type != EPCONTEXT_OP → error.
  CtxHelperTestContext ctx;
  FakeNode node{"relu", "Relu", "", 13, {}, {}};
  FakeGraph g{{node}, {}, {}, {}};
  const OrtGraph* graphs[] = {g.AsGraph()};
  std::vector<int> pos;
  auto status = GetMainContextNode(graphs, 1, ctx.api, pos);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OnnxCtxModelHelperTest, GetMainContextNode_NoMainContextAttr_ReturnsError) {
  // No MAIN_CONTEXT attr → defaults to 0 → not marked main → empty pos → error.
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
//   wrong op_type
//   embed_mode=false, ep_cache_context empty (default "")
//   embed_mode=false, absolute path (starts with '/')
//   embed_mode=false, path contains ".."
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
  // op_type != EPCONTEXT_OP → error before any attr or path access.
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
  // embed_mode=0, valid relative path but file does not exist → not-found error.
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

// =============================================================================
// CreateEPContextNodes — IS_MULTI_SOC_EP_CONTEXT attribute
//
// CreateEPContextNodes serializes the enable_multi_soc_ep_context flag into the
// EPContext node as the IS_MULTI_SOC_EP_CONTEXT integer attribute (1 or 0). This
// is the only multi-SoC behaviour observable in this file: GetEpContextFromMainNode
// merely reads the same attribute back and forwards it unchanged to the backend
// manager. The actual behavioural divergence (DLC vs single-SoC context binary)
// lives in QnnBackendManager and belongs in qnn_backend_manager_test.cc.
//
// These tests capture the value the CreateOpAttr stub receives for that
// attribute, so a regression that stopped writing it (or wrote the wrong value)
// would fail here rather than silently reaching the backend.
// =============================================================================

namespace {

// Captures the IS_MULTI_SOC_EP_CONTEXT value seen by the CreateOpAttr stub.
// File-scope because the stub is a captureless lambda (OrtApi function pointer)
// and cannot close over the fixture. Reset at the start of each test run.
struct MultiSocAttrCapture {
  bool seen = false;
  int64_t value = -1;
};
MultiSocAttrCapture g_multi_soc_capture;

// Minimal QnnBackendManager + QnnModel (no real backend, empty graph I/O names)
// plus the three stubs the write path touches: Node_GetName, CreateOpAttr, and
// CreateNode. The tests drive CreateEPContextNodes in embed mode so no context
// .bin file is written to disk.
struct CreateEpCtxNodeTestContext {
  Ort::Logger logger = MakeNullLogger();
  OrtApi ort_api{};
  OrtEpApi ep_api{};
  OrtModelEditorApi editor_api{};
  std::shared_ptr<qnn::QnnBackendManager> manager;
  std::unique_ptr<qnn::QnnModel> model;

  CreateEpCtxNodeTestContext() {
    ort_api.Node_GetName = [](const OrtNode*, const char** name) noexcept -> OrtStatus* {
      *name = "graph_0";
      return nullptr;
    };
    ort_api.CreateOpAttr = [](const char* name, const void* data, int, OrtOpAttrType type,
                              OrtOpAttr** op_attr) noexcept -> OrtStatus* {
      if (type == ORT_OP_ATTR_INT && std::string(name) == IS_MULTI_SOC_EP_CONTEXT) {
        g_multi_soc_capture.seen = true;
        g_multi_soc_capture.value = *static_cast<const int64_t*>(data);
      }
      // CreateNode (stubbed below) ignores the attributes, so any non-null
      // sentinel keeps ORT_CXX_RETURN_ON_API_FAIL happy.
      static int sentinel;
      *op_attr = reinterpret_cast<OrtOpAttr*>(&sentinel);
      return nullptr;
    };
    editor_api.CreateNode = [](const char*, const char*, const char*,
                               const char* const*, size_t, const char* const*, size_t,
                               OrtOpAttr**, size_t, OrtNode** node) noexcept -> OrtStatus* {
      static int sentinel;
      *node = reinterpret_cast<OrtNode*>(&sentinel);
      return nullptr;
    };

    qnn::QnnBackendManagerConfig cfg;
    cfg.backend_path = "libQnnHtp.so";
    cfg.profiling_level_etw = qnn::ProfilingLevel::OFF;
    cfg.profiling_level = qnn::ProfilingLevel::OFF;
    cfg.context_priority = qnn::ContextPriority::NORMAL;
    cfg.device_id = 0;
    cfg.htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
    cfg.soc_model = 0;
    cfg.skip_qnn_version_check = true;

    ApiPtrs api_ptrs{ort_api, ep_api, editor_api};
    manager = qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
    if (!manager) return;
    model = std::make_unique<qnn::QnnModel>(manager.get(), api_ptrs);
  }

  bool IsValid() const { return model != nullptr; }
};

// Runs CreateEPContextNodes for a single embed-mode EPContext node with the
// given multi-SoC flag and returns the value the CreateOpAttr stub captured for
// IS_MULTI_SOC_EP_CONTEXT. Returns -1 (and fails the calling test) on any error.
int64_t RunCreateEpCtxAndCaptureMultiSoc(bool enable_multi_soc_ep_context) {
  g_multi_soc_capture = {};

  CreateEpCtxNodeTestContext ctx;
  EXPECT_TRUE(ctx.IsValid()) << "QnnBackendManager::Create failed";
  if (!ctx.IsValid()) return -1;

  QnnModelLookupTable qnn_models;
  qnn_models.emplace("graph_0", std::move(ctx.model));

  const OrtNode* fused_node = reinterpret_cast<const OrtNode*>(0x1);
  OrtNode* ep_context_node = nullptr;
  unsigned char buffer[] = {0x1, 0x2, 0x3, 0x4};
  std::basic_string<ORTCHAR_T> context_model_path;  // unused in embed mode
  std::unordered_map<std::string, std::string> no_overrides;

  auto status = CreateEPContextNodes(&fused_node, 1, &ep_context_node,
                                     ctx.ort_api, ctx.editor_api,
                                     buffer, sizeof(buffer),
                                     /*sdk_build_version=*/"1.0",
                                     qnn_models,
                                     context_model_path,
                                     /*qnn_context_embed_mode=*/true,
                                     /*max_spill_fill_buffer_size=*/0,
                                     ctx.logger,
                                     /*share_ep_contexts=*/false,
                                     /*stop_share_ep_contexts=*/false,
                                     /*ep_name=*/"QNNExecutionProvider",
                                     no_overrides,
                                     enable_multi_soc_ep_context);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_TRUE(g_multi_soc_capture.seen) << "IS_MULTI_SOC_EP_CONTEXT attribute was never written";
  return g_multi_soc_capture.value;
}

}  // namespace

TEST(QnnUnit_OnnxCtxModelHelperTest, CreateEPContextNodes_MultiSocEnabled_WritesAttrOne) {
  EXPECT_EQ(RunCreateEpCtxAndCaptureMultiSoc(/*enable_multi_soc_ep_context=*/true), 1);
}

TEST(QnnUnit_OnnxCtxModelHelperTest, CreateEPContextNodes_MultiSocDisabled_WritesAttrZero) {
  EXPECT_EQ(RunCreateEpCtxAndCaptureMultiSoc(/*enable_multi_soc_ep_context=*/false), 0);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
