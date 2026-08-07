// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for qnn_ep_utils.cc.
//
// Tests the OrtNodeGroupSelector::Check() family using FakeGraph / FakeNode /
// FakeValueInfo from qnn_fake_ort_graph.h — no real QNN backend required.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <vector>

#include "core/providers/qnn/qnn_ep_utils.h"
#include "test/providers/qnn/unit/qnn_fake_ort_graph.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

using namespace onnxruntime::QDQ;
using namespace onnxruntime::test;

// =============================================================================
// EpUtilsTestContext
//
// Installs FakeGraph stubs plus the two extras that CheckQDQNodes uses:
//   - ValueInfo_IsGraphOutput  → always false (not a graph output)
//   - ValueInfo_GetValueNumConsumers → always 1 (exactly one consumer)
//
// These two are ORT_CONTINUE_ON_ERROR — the node-group check will return
// false without them if the function pointer is null and crashes, so they
// must always be installed.
// =============================================================================
struct EpUtilsTestContext {
  OrtApi api{};

  EpUtilsTestContext() {
    InstallFakeGraphApiStubs(api);
    api.ValueInfo_IsGraphOutput = [](const OrtValueInfo*, bool* out) noexcept -> OrtStatus* {
      *out = false;
      return nullptr;
    };
    api.ValueInfo_GetValueNumConsumers = [](const OrtValueInfo*, size_t* count) noexcept -> OrtStatus* {
      *count = 1;
      return nullptr;
    };
  }
};

// =============================================================================
// OrtDropDQNodeGroupSelector::Check
//
// Signature: (graph/*unused*/, ort_api, node/*unused*/, redundant_clip_node,
//             dq_nodes, q_nodes/*unused*/)
//
// Logic:
//   1. Reject if redundant_clip_node != nullptr
//   2. Reject if dq_nodes.size() != 1
//   3. Read dq_nodes[0]->inputs[0] element type; reject if absent
// =============================================================================

TEST(QnnUnit_EpUtilsTest, DropDQ_RejectsRedundantClipNode) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode clip{"clip", "Clip", "", 13, {}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, clip.AsNode(), {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, DropDQ_RejectsEmptyDqNodes) {
  EpUtilsTestContext ctx;
  OrtDropDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {}, {}));
}

TEST(QnnUnit_EpUtilsTest, DropDQ_RejectsTwoDqNodes) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq1.AsNode(), dq2.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, DropDQ_AcceptsUint8) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, DropDQ_AcceptsInt8) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, DropDQ_AcceptsInt16) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, DropDQ_AcceptsInt4) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

// =============================================================================
// OrtUnaryNodeGroupSelector::Check
//
// Logic:
//   1. CheckQDQNodes(node, 1 dq, 1 q) — node must have 1 output, 1 consumer
//   2. Read dq_nodes[0]->inputs[0] and q_nodes[0]->outputs[0] element types
//   3. Reject if the two types differ
//
// For CheckQDQNodes to pass we need:
//   - dq_nodes.size() == 1
//   - node has 1 output (FakeNode with 1 element in outputs)
//   - q_nodes.size() == 1 == num_outputs == total_consumers
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Unary_RejectsWrongDqCount) {
  EpUtilsTestContext ctx;
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"relu", "Relu", "", 13, {}, {&main_out}};

  OrtUnaryNodeGroupSelector sel;
  // dq_nodes empty → CheckQDQNodes fails (0 != 1)
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr, {}, {}));
}

TEST(QnnUnit_EpUtilsTest, Unary_AcceptsUint8) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"relu", "Relu", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtUnaryNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Unary_RejectsTypeMismatch) {
  EpUtilsTestContext ctx;
  // DQ input = UINT8, Q output = INT8 — types differ, so the check fails
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"relu", "Relu", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtUnaryNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Unary_AcceptsInt16) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"relu", "Relu", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtUnaryNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtBinaryNodeGroupSelector::Check
//
// Needs 2 DQ inputs. Both must have the same allowed type, plus 1 Q output.
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Binary_RejectsOnlyOneDq) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"add", "Add", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtBinaryNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Binary_AcceptsTwoUint8Dqs) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"add", "Add", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtBinaryNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Binary_RejectsMixedTypes) {
  EpUtilsTestContext ctx;
  // dq1 = UINT8, dq2 = INT8 — types mismatch
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"add", "Add", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtBinaryNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtPadNodeGroupSelector::Check
//
// 1 or 2 DQ inputs allowed; input/output types must match.
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Pad_RejectsThreeDqNodes) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq3{"dq3", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"pad", "Pad", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtPadNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode(), dq3.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Pad_AcceptsOneDqMatchingType) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"pad", "Pad", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtPadNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Pad_RejectsOneDqMismatchedType) {
  EpUtilsTestContext ctx;
  // DQ input = UINT8, Q output = INT8 — mismatch
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"pad", "Pad", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtPadNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Pad_AcceptsTwoDqSameType) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"pad", "Pad", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtPadNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Pad_RejectsTwoDqDifferentTypes) {
  EpUtilsTestContext ctx;
  // dq1 = UINT8, dq2 = INT8 — mismatch
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"pad", "Pad", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtPadNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtSelectors::RegisterSelector
// =============================================================================

TEST(QnnUnit_EpUtilsTest, RegisterSelector_IncreasesSize) {
  OrtSelectors selectors;
  EXPECT_EQ(selectors.SelectorsSet().size(), 0u);
  selectors.RegisterSelector({{"Relu", {1, 6, 13}}},
                             std::make_unique<OrtUnaryNodeGroupSelector>());
  EXPECT_EQ(selectors.SelectorsSet().size(), 1u);
}

// =============================================================================
// OrtVariadicNodeGroupSelector::Check
//
// All DQ inputs must share the same type; all Q outputs must share a type.
// Node_GetNumInputs is used by CheckQDQNodes (num_dq_inputs=-1), so
// main_node.inputs.size() must equal dq_nodes.size().
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Variadic_Accepts2DqSameType) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  // inputs.size() == dq_nodes.size() == 2 for CheckQDQNodes
  FakeNode main_node{"concat", "Concat", "", 13, {&dummy, &dummy}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtVariadicNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Variadic_RejectsMixedDqTypes) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"concat", "Concat", "", 13, {&dummy, &dummy}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtVariadicNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Variadic_AcceptsInt16) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"concat", "Concat", "", 13, {&dummy, &dummy}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtVariadicNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtSplitNodeGroupSelector::Check
//
// 1 DQ input → Split → N Q outputs.  All Q output types must match the DQ.
// main_node.outputs.size() must equal q_nodes.size() for CheckQDQNodes.
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Split_RejectsRedundantClipNode) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode clip{"clip", "Clip", "", 13, {}, {}};

  OrtSplitNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, clip.AsNode(), {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, Split_Accepts1DqWith2Q) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 8}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo out1{"y1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo out2{"y2", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  // 1 input (dq_nodes.size=1), 2 outputs (q_nodes.size=2)
  FakeNode main_node{"split", "Split", "", 13, {&dummy}, {&out1, &out2}};

  FakeValueInfo q_out1{"z1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo q_out2{"z2", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q1{"q1", "QuantizeLinear", "", 13, {}, {&q_out1}};
  FakeNode q2{"q2", "QuantizeLinear", "", 13, {}, {&q_out2}};

  OrtSplitNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {q1.AsNode(), q2.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Split_RejectsQTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 8}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo out1{"y1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo out2{"y2", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"split", "Split", "", 13, {&dummy}, {&out1, &out2}};

  FakeValueInfo q_out1{"z1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo q_out2{"z2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // mismatch
  FakeNode q1{"q1", "QuantizeLinear", "", 13, {}, {&q_out1}};
  FakeNode q2{"q2", "QuantizeLinear", "", 13, {}, {&q_out2}};

  OrtSplitNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q1.AsNode(), q2.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Split_AcceptsInt4) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"split", "Split", "", 13, {&dummy}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtSplitNodeGroupSelector sel(/*req_equal_quant_params=*/false);
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {q.AsNode()}));
}

namespace {
// Conv producer map for the positional DQ check in OrtConvNodeGroupSelector::Check.
// Maps each Conv input (OrtValueInfo*) to the DQ node that produces it.
std::unordered_map<const OrtValueInfo*, const OrtNode*> g_conv_producer_map;
OrtStatus* FakeConvProducerStub(const OrtValueInfo* vi, const OrtNode** producer,
                                size_t* output_index) noexcept {
  auto it = g_conv_producer_map.find(vi);
  if (producer) *producer = (it != g_conv_producer_map.end()) ? it->second : nullptr;
  if (output_index) *output_index = 0;
  return nullptr;
}
struct ConvProducerGuard {
  explicit ConvProducerGuard(std::unordered_map<const OrtValueInfo*, const OrtNode*> map) {
    g_conv_producer_map = std::move(map);
  }
  ~ConvProducerGuard() { g_conv_producer_map.clear(); }
};
}  // namespace

// =============================================================================
// OrtConvNodeGroupSelector::Check
//
// dq[0]=input, dq[1]=weight, dq[2]=bias (optional INT32).
// An INT8 input requires an INT8 weight; other quant types are unconstrained.
// main_node.inputs.size() must equal dq_nodes.size() for CheckQDQNodes.
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Conv_Accepts2DqUint8) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_wt{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};

  // Conv inputs are the DQ outputs — wire them so the positional DQ check passes.
  FakeValueInfo conv_in0{"conv_in0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo conv_in1{"conv_in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"conv", "Conv", "", 1, {&conv_in0, &conv_in1}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  // Install producer stub: conv_in0 → dq_data, conv_in1 → dq_wt.
  ConvProducerGuard guard({{conv_in0.AsValueInfo(), dq_data.AsNode()},
                           {conv_in1.AsValueInfo(), dq_wt.AsNode()}});
  ctx.api.ValueInfo_GetValueProducer = &FakeConvProducerStub;
  OrtGlobalApiOverride global_guard(&ctx.api);

  OrtConvNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq_data.AsNode(), dq_wt.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Conv_AcceptsInt8) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_wt{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};

  // Conv inputs are the DQ outputs — wire them so the positional DQ check passes.
  FakeValueInfo conv_in0{"conv_in0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo conv_in1{"conv_in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"conv", "Conv", "", 1, {&conv_in0, &conv_in1}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  // Install producer stub: conv_in0 → dq_data, conv_in1 → dq_wt.
  ConvProducerGuard guard({{conv_in0.AsValueInfo(), dq_data.AsNode()},
                           {conv_in1.AsValueInfo(), dq_wt.AsNode()}});
  ctx.api.ValueInfo_GetValueProducer = &FakeConvProducerStub;
  OrtGlobalApiOverride global_guard(&ctx.api);

  OrtConvNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq_data.AsNode(), dq_wt.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Conv_Rejects3DqWithNonInt32Bias) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeValueInfo dq_bias_in{"b", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {4}};  // wrong — must be INT32
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_wt{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_bias{"dq2", "DequantizeLinear", "", 13, {&dq_bias_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"conv", "Conv", "", 1, {&dummy, &dummy, &dummy}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtConvNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_data.AsNode(), dq_wt.AsNode(), dq_bias.AsNode()}, {q.AsNode()}));
}

// Tests that Conv with quantized input, quantized weight, and a plain float bias
// (no DQ node for bias) is accepted.
TEST(QnnUnit_EpUtilsTest, Conv_AcceptsFloatBias) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_wt{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};

  // Conv inputs: DQ output for activation, DQ output for weight, float bias (no DQ).
  FakeValueInfo conv_in0{"conv_in0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo conv_in1{"conv_in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo float_bias{"bias", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"conv", "Conv", "", 1, {&conv_in0, &conv_in1, &float_bias}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  // Install producer stub: conv_in0 → dq_data, conv_in1 → dq_wt.
  // float_bias has no entry → producer = nullptr (float, no DQ node).
  ConvProducerGuard guard({{conv_in0.AsValueInfo(), dq_data.AsNode()},
                           {conv_in1.AsValueInfo(), dq_wt.AsNode()}});
  ctx.api.ValueInfo_GetValueProducer = &FakeConvProducerStub;
  OrtGlobalApiOverride global_guard(&ctx.api);

  OrtConvNodeGroupSelector sel;
  // 2 DQ nodes (activation + weight, no bias DQ) → float bias is allowed.
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq_data.AsNode(), dq_wt.AsNode()}, {q.AsNode()}));
}

// Tests that Conv with a float (unquantized) activation input is rejected
// when weight and bias are quantized. inputs[0] is not DQ-produced, so the
// positional DQ check fails.
TEST(QnnUnit_EpUtilsTest, Conv_RejectsFloatInputWithQuantWeight) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_wt_in{"w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo dq_bias_in{"b", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {}};
  FakeNode dq_wt{"dq1", "DequantizeLinear", "", 13, {&dq_wt_in}, {}};
  FakeNode dq_bias{"dq2", "DequantizeLinear", "", 13, {&dq_bias_in}, {}};

  // Conv inputs: float activation (no DQ), DQ output for weight, DQ output for bias.
  FakeValueInfo float_input{"input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo conv_in1{"conv_in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo conv_in2{"conv_in2", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"conv", "Conv", "", 1, {&float_input, &conv_in1, &conv_in2}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  // Install producer stub: conv_in1 → dq_wt, conv_in2 → dq_bias.
  // float_input has no entry → producer = nullptr → positional check fails at slot 0.
  ConvProducerGuard guard({{conv_in1.AsValueInfo(), dq_wt.AsNode()},
                           {conv_in2.AsValueInfo(), dq_bias.AsNode()}});
  ctx.api.ValueInfo_GetValueProducer = &FakeConvProducerStub;
  // OrtGlobalApiOverride not needed: producer for inputs[0] is nullptr so
  // GetOperatorType() is never reached.

  OrtConvNodeGroupSelector sel;
  // inputs[0] is not DQ-produced → positional check fails → false.
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_wt.AsNode(), dq_bias.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtMatMulNodeGroupSelector::Check
//
// 2 DQ inputs and a trailing Q required.
// =============================================================================

TEST(QnnUnit_EpUtilsTest, MatMul_Accepts2DqWithQ) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {4, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"mm", "MatMul", "", 13, {&dummy, &dummy}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtMatMulNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, MatMul_RejectsNoQ) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {4, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};

  // A MatMul with no trailing Q would be a MatMulIntegerToFloat, which is not selected.
  OrtMatMulNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {}));
}

// =============================================================================
// OrtLogicalComparisonNodeGroupSelector::Check
//
// 2 DQ inputs, no Q output.  Both DQ types must match.
// =============================================================================

TEST(QnnUnit_EpUtilsTest, LogicalComparison_AcceptsMatchingTypes) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};

  // 2 inputs, is_empty_q_nodes_allowed=true so CheckQDQNodes returns true without checking outputs
  FakeNode main_node{"equal", "Equal", "", 13, {&dummy, &dummy}, {}};

  OrtLogicalComparisonNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_TRUE(base.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, LogicalComparison_RejectsMismatchedTypes) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};

  FakeNode main_node{"equal", "Equal", "", 13, {&dummy, &dummy}, {}};

  OrtLogicalComparisonNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                          {dq1.AsNode(), dq2.AsNode()}, {}));
}

// =============================================================================
// OrtDropQDQNodeGroupSelector::Check — early exits only
// Happy path requires IsQDQPairSupported (reads scale OrtValue), skipped.
// =============================================================================

TEST(QnnUnit_EpUtilsTest, DropQDQ_RejectsRedundantClipNode) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode clip{"clip", "Clip", "", 13, {}, {}};

  OrtDropQDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, clip.AsNode(), {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, DropQDQ_RejectsZeroDqNodes) {
  EpUtilsTestContext ctx;
  OrtDropQDQNodeGroupSelector sel;
  // CheckQDQNodes expects 1 DQ but gets 0 → false
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {}, {}));
}

// =============================================================================
// CheckQDQNodes — produces_graph_output=true path
// =============================================================================

TEST(QnnUnit_EpUtilsTest, CheckQDQNodes_RejectsWhenOutputIsGraphOutput) {
  EpUtilsTestContext ctx;
  // Override: any output IS a graph output
  ctx.api.ValueInfo_IsGraphOutput = [](const OrtValueInfo*, bool* out) noexcept -> OrtStatus* {
    *out = true;
    return nullptr;
  };

  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"relu", "Relu", "", 13, {&dummy}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtUnaryNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

// CheckQDQNodes — total_consumers != q_nodes.size() path
// =============================================================================

TEST(QnnUnit_EpUtilsTest, CheckQDQNodes_RejectsWhenConsumerCountExceedsQNodes) {
  EpUtilsTestContext ctx;
  // Override: each output has 2 consumers, but only 1 Q node → mismatch
  ctx.api.ValueInfo_GetValueNumConsumers = [](const OrtValueInfo*, size_t* count) noexcept -> OrtStatus* {
    *count = 2;
    return nullptr;
  };

  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"relu", "Relu", "", 13, {&dq_in}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtUnaryNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtDropQDQNodeGroupSelector — type-mismatch path (reaches the type-equality check)
// =============================================================================

TEST(QnnUnit_EpUtilsTest, DropQDQ_RejectsTypeMismatchAfterCheckQDQNodes) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"id", "Identity", "", 13, {&dummy}, {&main_out}};

  // Q output INT8 ≠ DQ input UINT8 — type-equality check fails
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtDropQDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtConvNodeGroupSelector — additional rejection paths
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Conv_RejectsInputOutputTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_wt{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"conv", "Conv", "", 1, {&dummy, &dummy}, {&main_out}};
  // Q output INT8 ≠ input UINT8
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtConvNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_data.AsNode(), dq_wt.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Conv_RejectsInt8DataWithUint8Weight) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_data_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeValueInfo dq_wt_in{"w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};  // mismatch
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_data_in}, {}};
  FakeNode dq_wt{"dq1", "DequantizeLinear", "", 13, {&dq_wt_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"conv", "Conv", "", 1, {&dummy, &dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtConvNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_data.AsNode(), dq_wt.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtMatMulNodeGroupSelector — INT8 mismatch path
// =============================================================================

TEST(QnnUnit_EpUtilsTest, MatMul_RejectsInt8DataWithUint8Weight) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_data_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeValueInfo dq_wt_in{"w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};  // mismatch
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_data_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_wt_in}, {}};

  OrtMatMulNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {}));
}

// =============================================================================
// OrtGemmNodeGroupSelector — INT8/Q-type mismatch
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Gemm_RejectsInt8AWithUint8B) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_a_in{"a", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeValueInfo dq_b_in{"b", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};  // mismatch
  FakeNode dq_a{"dq_a", "DequantizeLinear", "", 13, {&dq_a_in}, {}};
  FakeNode dq_b{"dq_b", "DequantizeLinear", "", 13, {&dq_b_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"gemm", "Gemm", "", 11, {&dummy, &dummy}, {&main_out}};

  OrtGemmNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_a.AsNode(), dq_b.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, Gemm_RejectsQOutputTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq_a{"dq_a", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_b{"dq_b", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"gemm", "Gemm", "", 11, {&dummy, &dummy}, {&main_out}};

  // Q output INT8 ≠ A input UINT8
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtGemmNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_a.AsNode(), dq_b.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtEinsumNodeGroupSelector — acceptance + rejection paths
// (is_empty_q_nodes_allowed=true → CheckQDQNodes passes with empty Q)
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Einsum_AcceptsUint8NoQ) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"einsum", "Einsum", "", 12, {&dummy}, {}};

  OrtEinsumNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, Einsum_AcceptsInt8) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"einsum", "Einsum", "", 12, {&dummy}, {}};

  OrtEinsumNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, Einsum_AcceptsInt16) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"einsum", "Einsum", "", 12, {&dummy}, {}};

  OrtEinsumNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, Einsum_RejectsQOutputTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"einsum", "Einsum", "", 12, {&dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // mismatch
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtEinsumNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtReciprocalNodeGroupSelector — same pattern as Einsum
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Reciprocal_AcceptsUint8NoQ) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"recip", "Reciprocal", "", 13, {&dummy}, {}};

  OrtReciprocalNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, Reciprocal_AcceptsInt8) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"recip", "Reciprocal", "", 13, {&dummy}, {}};

  OrtReciprocalNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

// =============================================================================
// OrtWhereNodeGroupSelector — type mismatch paths
// =============================================================================

TEST(QnnUnit_EpUtilsTest, Where_RejectsDqTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // mismatch
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"where", "Where", "", 9, {&dummy, &dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtWhereNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_EpUtilsTest, Where_AcceptsInt16) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"where", "Where", "", 9, {&dummy, &dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtWhereNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtInstanceAndLayerNormalizationNodeGroupSelector — type mismatch (private)
// =============================================================================

TEST(QnnUnit_EpUtilsTest, InstanceNorm_RejectsTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_scale{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"ln", "LayerNorm", "", 17, {&dummy, &dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // mismatch
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtInstanceAndLayerNormalizationNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                          {dq_data.AsNode(), dq_scale.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtBatchNormalizationNodeGroupSelector — too few DQ + INT8 mismatch
// =============================================================================

TEST(QnnUnit_EpUtilsTest, BatchNorm_RejectsTooFewDq) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtBatchNormalizationNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, BatchNorm_RejectsInt8WithMixedScale) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_data_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeValueInfo dq_scale_in{"s", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};  // scale ≠ input type
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_data_in}, {}};
  FakeNode dq_scale{"dq1", "DequantizeLinear", "", 13, {&dq_scale_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"bn", "BatchNormalization", "", 15, {&dummy, &dummy}, {&main_out}};

  OrtBatchNormalizationNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_data.AsNode(), dq_scale.AsNode()}, {}));
}

// =============================================================================
// OrtCumSumNodeGroupSelector — type mismatch (private)
// =============================================================================

TEST(QnnUnit_EpUtilsTest, CumSum_RejectsTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"cumsum", "CumSum", "", 14, {&dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // mismatch
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtCumSumNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                          {dq.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtScatterElementsNodeGroupSelector — type mismatch (private)
// =============================================================================

TEST(QnnUnit_EpUtilsTest, ScatterElements_RejectsMixedDqTypes) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // mismatch
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"scatter", "ScatterElements", "", 11, {&dummy, &dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtScatterElementsNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                          {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtRMSNormalizationNodeGroupSelector — type mismatch (private)
// =============================================================================

TEST(QnnUnit_EpUtilsTest, RMSNorm_RejectsTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_scale{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"rms", "RMSNormalization", "", 1, {&dummy, &dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // mismatch
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtRMSNormalizationNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                          {dq_data.AsNode(), dq_scale.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtTopKNodeGroupSelector — early exits + CanCreateNodeGroup path (private)
// =============================================================================

TEST(QnnUnit_EpUtilsTest, TopK_RejectsRedundantClipNode) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode clip{"clip", "Clip", "", 13, {}, {}};

  OrtTopKNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, nullptr, clip.AsNode(),
                          {dq.AsNode()}, {}));
}

TEST(QnnUnit_EpUtilsTest, TopK_RejectsWrongDqCount) {
  EpUtilsTestContext ctx;
  OrtTopKNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, nullptr, nullptr, {}, {}));
}

TEST(QnnUnit_EpUtilsTest, TopK_CoversCanCreateNodeGroupPath) {
  // CanCreateNodeGroup is exercised here. IsQDQPairSupported will return false
  // (no scale initializer) so the final result is false, but the internal
  // CanCreateNodeGroup + type-check code paths are covered.
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"topk", "TopK", "", 11, {&dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtTopKNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  // Result is false because the Q node has no scale initializer (no inputs[1]),
  // so IsQDQPairSupported returns false. CanCreateNodeGroup and type-check paths
  // are covered.
  EXPECT_FALSE(base.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                          {dq.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// IsQOrDQScalePositiveConstantScalar + IsQDQPairSupported
// (exercised through OrtDropQDQNodeGroupSelector::Check with real scale
//  initializers backed by FakeOrtValue)
//
// Graph pattern for all tests below:
//   DQ(inputs[0]=dq_in, inputs[1]=<scale_vi>) ->
//   Identity(outputs={main_out}) ->
//   Q(inputs[0]=main_out, inputs[1]=<scale_vi>, outputs={q_out})
// =============================================================================

// Shared fixture: builds the minimum fake graph for a DropQDQ scale test.
// Scale values and which scale_vi each node uses are set per test.
struct DropQdqScaleFixture {
  FakeValueInfo q_scale_vi{"q_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_scale_vi{"dq_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"dq_in", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeValueInfo main_out{"main_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeValueInfo q_out{"q_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeOrtValue q_scale_val;
  FakeOrtValue dq_scale_val;
  // Both nodes default to the same q_scale_vi (same-name path).
  FakeNode dq_node{"dq", "DequantizeLinear", "", 13, {&dq_in, &q_scale_vi}, {}};
  FakeNode main_node{"id", "Identity", "", 13, {}, {&main_out}};
  FakeNode q_node{"q", "QuantizeLinear", "", 13, {&main_out, &q_scale_vi}, {&q_out}};
  FakeGraph graph{{}, {}, {}, {&q_scale_vi}};
};

// IsQOrDQScalePositiveConstantScalar: positive float scale → returns true.
// IsQDQPairSupported: same scale name → same_scale=true → returns true.
TEST(QnnUnit_EpUtilsTest, DropQDQ_PositiveScale_SameName_CheckReturnsTrue) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(0.5f);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_TRUE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                        {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: negative scale → returns false.
TEST(QnnUnit_EpUtilsTest, DropQDQ_NegativeScale_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(-0.5f);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: zero scale → returns false.
TEST(QnnUnit_EpUtilsTest, DropQDQ_ZeroScale_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(0.0f);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// allow_nonpositive_scale=true: IsQOrDQScalePositiveConstantScalar is skipped;
// IsQDQPairSupported (same name) returns true even for a negative scale.
TEST(QnnUnit_EpUtilsTest, DropQDQ_AllowNonpositiveScale_NegativeScaleOk) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(-1.0f);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/true);
  EXPECT_TRUE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                        {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: scale not in graph.initializers →
// GetConstantInitializer returns nullptr → returns false.
TEST(QnnUnit_EpUtilsTest, DropQDQ_ScaleNotInitializer_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.graph.initializers = {};  // scale_vi absent from graph initializers

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQDQPairSupported: different scale names, same float value → same_scale=true.
TEST(QnnUnit_EpUtilsTest, DropQDQ_DiffScaleName_SameValue_CheckReturnsTrue) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(0.5f);
  f.dq_scale_val = FakeOrtValue::MakeFloat(0.5f);
  f.q_scale_vi.initializer_value = &f.q_scale_val;
  f.dq_scale_vi.initializer_value = &f.dq_scale_val;
  // Q uses q_scale_vi ("q_scale"), DQ uses dq_scale_vi ("dq_scale").
  f.dq_node.inputs = {&f.dq_in, &f.dq_scale_vi};
  f.graph.initializers = {&f.q_scale_vi, &f.dq_scale_vi};

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_TRUE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                        {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQDQPairSupported: different scale names, different float values → false.
TEST(QnnUnit_EpUtilsTest, DropQDQ_DiffScaleName_DiffValue_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(0.5f);
  f.dq_scale_val = FakeOrtValue::MakeFloat(1.0f);  // different value
  f.q_scale_vi.initializer_value = &f.q_scale_val;
  f.dq_scale_vi.initializer_value = &f.dq_scale_val;
  f.dq_node.inputs = {&f.dq_in, &f.dq_scale_vi};
  f.graph.initializers = {&f.q_scale_vi, &f.dq_scale_vi};

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: Q node has < 2 inputs →
// num_inputs < 2 guard fires → returns false.
TEST(QnnUnit_EpUtilsTest, DropQDQ_QNodeHasNoScaleInput_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  // Q node has only 1 input (no scale at index 1).
  f.q_node.inputs = {&f.main_out};

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: double-precision scale, positive →
// element_type==DOUBLE path executed, returns true.
TEST(QnnUnit_EpUtilsTest, DropQDQ_DoubleScale_Positive_CheckReturnsTrue) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeDouble(0.5);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_TRUE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                        {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: scale is INT32 type (neither float nor
// double) → falls through to the final return false.
TEST(QnnUnit_EpUtilsTest, DropQDQ_IntTypeScale_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQDQPairSupported: different scale names, Q=FLOAT vs DQ=DOUBLE →
// q_element_type != dq_element_type → returns false.
TEST(QnnUnit_EpUtilsTest, DropQDQ_DiffScaleName_TypeMismatch_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(0.5f);   // FLOAT
  f.dq_scale_val = FakeOrtValue::MakeDouble(0.5);  // DOUBLE
  f.q_scale_vi.initializer_value = &f.q_scale_val;
  f.dq_scale_vi.initializer_value = &f.dq_scale_val;
  f.dq_node.inputs = {&f.dq_in, &f.dq_scale_vi};
  f.graph.initializers = {&f.q_scale_vi, &f.dq_scale_vi};

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQDQPairSupported: different scale names, both DOUBLE, same value →
// double-precision comparison path executed, same_scale=true → returns true.
TEST(QnnUnit_EpUtilsTest, DropQDQ_DiffScaleName_BothDouble_SameValue_CheckReturnsTrue) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeDouble(0.5);
  f.dq_scale_val = FakeOrtValue::MakeDouble(0.5);
  f.q_scale_vi.initializer_value = &f.q_scale_val;
  f.dq_scale_vi.initializer_value = &f.dq_scale_val;
  f.dq_node.inputs = {&f.dq_in, &f.dq_scale_vi};
  f.graph.initializers = {&f.q_scale_vi, &f.dq_scale_vi};

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_TRUE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                        {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// =============================================================================
// "DQ has no inputs" early-exit tests
//
// Covers the !dt_input.has_value() guards in multiple selector Check() methods.
// When the first DQ node has no inputs, GetNodeInputDataType(dq_node, 0) returns
// nullopt (index 0 >= num_inputs == 0), triggering the return-false path.
// =============================================================================

// DropQDQ: dt_input or dt_output not has_value → returns false.
TEST(QnnUnit_EpUtilsTest, DropQDQ_DQHasNoInputs_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {}, {}};  // no inputs
  FakeNode main_node{"id", "Identity", "", 13, {}, {&main_out}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  OrtDropQDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

// DropDQ: DQ node has no inputs → !dt_input.has_value() → returns false.
TEST(QnnUnit_EpUtilsTest, DropDQ_DQHasNoInputs_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {}, {}};
  OrtDropDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

// Unary: DQ node has no inputs → !dt_input.has_value() → returns false.
TEST(QnnUnit_EpUtilsTest, Unary_DQHasNoInputs_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {}, {}};  // no inputs
  FakeNode main_node{"relu", "Relu", "", 13, {}, {&main_out}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  OrtUnaryNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

// Variadic: zero DQ nodes → CheckQDQNodes fails → returns false.
TEST(QnnUnit_EpUtilsTest, Variadic_ZeroDqNodes_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  // Add node needs 2 inputs so num_dq_inputs=2; dq_nodes.size()=0 → mismatch → false.
  FakeValueInfo in1{"in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeValueInfo in2{"in2", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeNode main_node{"add", "Add", "", 13, {&in1, &in2}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  OrtVariadicNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {}, {q.AsNode()}));
}

// Variadic: DQ node has no inputs → !dt_input.has_value() → returns false.
TEST(QnnUnit_EpUtilsTest, Variadic_DQHasNoInputs_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo in1{"in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeValueInfo in2{"in2", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {}, {}};  // no inputs
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {}, {}};
  FakeNode main_node{"add", "Add", "", 13, {&in1, &in2}, {&main_out}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  OrtVariadicNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// Split: DQ node has no inputs → !dt_input.has_value() → returns false.
TEST(QnnUnit_EpUtilsTest, Split_DQHasNoInputs_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {}, {}};  // no inputs
  FakeNode main_node{"split", "Split", "", 13, {}, {&main_out}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  OrtSplitNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

// Split: req_equal_quant_params=true + Q and DQ have different scale values →
// IsQDQPairSupported returns false → returns false.
TEST(QnnUnit_EpUtilsTest, Split_ReqEqualQuant_DiffScaleValues_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(0.5f);
  f.dq_scale_val = FakeOrtValue::MakeFloat(1.0f);  // different value
  f.q_scale_vi.initializer_value = &f.q_scale_val;
  f.dq_scale_vi.initializer_value = &f.dq_scale_val;
  f.dq_node.inputs = {&f.dq_in, &f.dq_scale_vi};
  f.graph.initializers = {&f.q_scale_vi, &f.dq_scale_vi};

  OrtSplitNodeGroupSelector sel(/*req_equal_quant_params=*/true);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// Conv: first DQ node has no inputs → !dt_input.has_value() → returns false.
TEST(QnnUnit_EpUtilsTest, Conv_FirstDQHasNoInputs_CheckReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_wt_in{"w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {}, {}};  // no inputs
  FakeNode dq_wt{"dq1", "DequantizeLinear", "", 13, {&dq_wt_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"conv", "Conv", "", 1, {&dummy, &dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  OrtConvNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq_data.AsNode(), dq_wt.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtSelectorManager::GetOrtQDQSelections — version-check path
//
// Uses OrtGlobalApiOverride to make Ort::ConstNode(node).GetOperatorType() call
// the test stub (FakeNode::op_type). MaxPool has a non-empty version list {12},
// so the if(!versions.empty()) block is entered, covering L1808-1809.
// =============================================================================

TEST(QnnUnit_EpUtilsTest, GetOrtQDQSelections_MaxPoolVersionCheck_CoversVersionPath) {
  EpUtilsTestContext ctx;
  // MaxPool is registered with versions={12}. A MaxPool node with since_version=12
  // is within range → the version-check block is entered.
  FakeNode maxpool_node{"mp", "MaxPool", "", 12, {}, {}};
  FakeGraph graph{{maxpool_node}};

  OrtGlobalApiOverride global_guard(&ctx.api);
  OrtSelectorManager selector_mgr;
  auto logger = MakeNullLogger();
  auto selections = selector_mgr.GetOrtQDQSelections(graph.AsGraph(), ctx.api, logger);
  // No DQ/Q nodes → GetOrtQDQSelection finds no pattern → result is empty.
  EXPECT_TRUE(selections.empty());
}

// =============================================================================
// =============================================================================
// Selector "return false" branch coverage — additional simple tests
//
// Each test covers a specific !has_value() or type-check return path that was
// not reached by existing tests.
// =============================================================================

// OrtClipNodeGroupSelector: CheckQDQNodes fails when q_nodes is empty and
// is_empty_q_nodes_allowed=false (the Clip default) → returns false.
TEST(QnnUnit_EpUtilsTest, Clip_CheckQDQNodesFails_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"clip", "Clip", "", 11, {}, {&main_out}};
  FakeGraph graph{};
  OrtClipNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  // Empty q_nodes → CheckQDQNodes returns is_empty_q_nodes_allowed=false → L699.
  EXPECT_FALSE(base.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                          {dq.AsNode()}, {}));
}

// OrtClipNodeGroupSelector: data_producer is nullptr (ValueInfo_GetValueProducer
// default stub returns nullptr) → returns false before dt_input check.
TEST(QnnUnit_EpUtilsTest, Clip_NullProducer_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo clip_in{"ci", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {}, {}};
  FakeNode main_node{"clip", "Clip", "", 11, {&clip_in}, {&main_out}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  // OrtGlobalApiOverride needed because Clip Check calls Ort::ConstNode(producer)
  // even though producer will be nullptr (short-circuits before GetOperatorType).
  OrtGlobalApiOverride global_guard(&ctx.api);
  OrtClipNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                          {dq.AsNode()}, {q.AsNode()}));
}

// OrtBinaryNodeGroupSelector: first DQ has no inputs → !dt_input.has_value().
TEST(QnnUnit_EpUtilsTest, Binary_FirstDQHasNoInputs_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq2_in{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {}, {}};  // no inputs
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq2_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"add", "Add", "", 13, {}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  OrtBinaryNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// OrtVariadicNodeGroupSelector: Q node has no outputs →
// !dt_output.has_value() → returns false.
TEST(QnnUnit_EpUtilsTest, Variadic_QHasNoOutputs_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo in1{"in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo in2{"in2", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&in1}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&in2}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1}};
  FakeNode main_node{"add", "Add", "", 13, {&in1, &in2}, {&main_out}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {}};  // no outputs
  FakeGraph graph{};
  OrtVariadicNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// OrtEinsumNodeGroupSelector: 4-bit input accepted (no width gates).
TEST(QnnUnit_EpUtilsTest, Einsum_AcceptsInt4) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"einsum", "Einsum", "", 12, {&dummy}, {}};
  OrtEinsumNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

// OrtReciprocalNodeGroupSelector: 16-bit input accepted (no width gates).
// main_node has 1 input so CheckQDQNodes passes (num_dq_inputs=1).
TEST(QnnUnit_EpUtilsTest, Reciprocal_AcceptsInt16) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"reciprocal", "Reciprocal", "", 13, {&dummy}, {}};
  OrtReciprocalNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

// OrtReciprocalNodeGroupSelector: 4-bit input accepted (no width gates).
TEST(QnnUnit_EpUtilsTest, Reciprocal_AcceptsInt4) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"reciprocal", "Reciprocal", "", 13, {&dummy}, {}};
  OrtReciprocalNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

// OrtReciprocalNodeGroupSelector: Q node has no outputs →
// !dt_output.has_value() → returns false.
TEST(QnnUnit_EpUtilsTest, Reciprocal_QHasNoOutputs_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"reciprocal", "Reciprocal", "", 13, {&dummy}, {&main_out}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {}};  // no outputs
  OrtReciprocalNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

// OrtReciprocalNodeGroupSelector: DQ UINT8 but Q output INT8 →
// type mismatch → returns false.
TEST(QnnUnit_EpUtilsTest, Reciprocal_QOutputTypeMismatch_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"reciprocal", "Reciprocal", "", 13, {&dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // mismatch
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  OrtReciprocalNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

// OrtMatMulNodeGroupSelector: first DQ has no inputs →
// !dt_input.has_value() → returns false.
TEST(QnnUnit_EpUtilsTest, MatMul_FirstDQHasNoInputs_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq2_in{"w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {}, {}};  // no inputs
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq2_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"matmul", "MatMul", "", 13, {}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  OrtMatMulNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// OrtVariadicNodeGroupSelector: second Q node has mismatched output type →
// dt_o.value() != dt_output.value() → returns false.
TEST(QnnUnit_EpUtilsTest, Variadic_SecondQTypeMismatch_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo in1{"in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo in2{"in2", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&in1}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&in2}, {}};
  FakeValueInfo main_out1{"y1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo main_out2{"y2", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"add", "Add", "", 13, {&in1, &in2}, {&main_out1, &main_out2}};
  FakeValueInfo q1_out{"z1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo q2_out{"z2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // mismatch
  FakeNode q1{"q1", "QuantizeLinear", "", 13, {}, {&q1_out}};
  FakeNode q2{"q2", "QuantizeLinear", "", 13, {}, {&q2_out}};
  FakeGraph graph{};
  OrtVariadicNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q1.AsNode(), q2.AsNode()}));
}

// OrtMatMulNodeGroupSelector: QLinearMatMul path, CheckQDQNodes fails because
// main_node has 0 inputs → num_dq_inputs=0 ≠ 2 → returns false.
TEST(QnnUnit_EpUtilsTest, MatMul_QLinearCheckQDQNodesFails_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq1_in{"a", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo dq2_in{"b", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq1_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq2_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"matmul", "MatMul", "", 13, {}, {&main_out}};  // 0 inputs
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};
  OrtMatMulNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// Additional selector Check() branches — targeted coverage for
// qnn_ep_utils.cc L736/1037/1065/1093/1192/1206/1267
// =============================================================================

// OrtGemmNodeGroupSelector: dq_nodes.size() < 2 → returns false.
// CheckQDQNodes passes because is_empty_q_nodes_allowed=true and node has 1 input.
TEST(QnnUnit_EpUtilsTest, Gemm_RejectsFewerThan2Dq) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"a", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq_a{"dq_a", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"gemm", "Gemm", "", 11, {&dummy}, {&main_out}};

  OrtGemmNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_a.AsNode()}, {}));
}

// OrtGemmNodeGroupSelector: UINT16 A and B accepted (no width gates), and with no
// Q node and only 2 DQ nodes the selector is done → returns true.
TEST(QnnUnit_EpUtilsTest, Gemm_AcceptsUint16) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, {}};
  FakeNode dq_a{"dq_a", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_b{"dq_b", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"gemm", "Gemm", "", 11, {&dummy, &dummy}, {&main_out}};

  OrtGemmNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq_a.AsNode(), dq_b.AsNode()}, {}));
}

// OrtWhereNodeGroupSelector: CheckQDQNodes with num_dq_inputs=2 fails because
// dq_nodes.size()=3 → returns false.
TEST(QnnUnit_EpUtilsTest, Where_RejectsWrongDqCount) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq3{"dq3", "DequantizeLinear", "", 13, {&dq_in}, {}};  // extra
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"where", "Where", "", 9, {&dummy, &dummy, &dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtWhereNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode(), dq3.AsNode()}, {q.AsNode()}));
}

// OrtBatchNormalizationNodeGroupSelector: CheckQDQNodes fails because
// num_outputs (2) != q_nodes.size() (1) → returns false.
TEST(QnnUnit_EpUtilsTest, BatchNorm_CheckQDQNodesFails_ReturnsFalse) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq_scale{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out1{"y1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo main_out2{"y2", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};  // extra output
  FakeNode main_node{"bn", "BatchNormalization", "", 15, {&dummy, &dummy}, {&main_out1, &main_out2}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};  // only 1 Q for 2 outputs

  OrtBatchNormalizationNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_data.AsNode(), dq_scale.AsNode()}, {q.AsNode()}));
}

// OrtBatchNormalizationNodeGroupSelector: !has_float_output path,
// dt_output.value() != dt_input.value() → returns false.
TEST(QnnUnit_EpUtilsTest, BatchNorm_RejectsInputOutputTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_data_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeValueInfo dq_scale_in{"s", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq_data{"dq0", "DequantizeLinear", "", 13, {&dq_data_in}, {}};
  FakeNode dq_scale{"dq1", "DequantizeLinear", "", 13, {&dq_scale_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"bn", "BatchNormalization", "", 15, {&dummy, &dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // ≠ UINT8 input
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtBatchNormalizationNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq_data.AsNode(), dq_scale.AsNode()}, {q.AsNode()}));
}

// OrtTopKNodeGroupSelector: dt_input.value() != dt_output.value() → returns
// false. Requires CanCreateNodeGroup to pass first.
TEST(QnnUnit_EpUtilsTest, TopK_RejectsInputOutputTypeMismatch) {
  EpUtilsTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeNode main_node{"topk", "TopK", "", 11, {&dummy}, {&main_out}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};  // ≠ UINT8 input
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtTopKNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                          {dq.AsNode()}, {q.AsNode()}));
}

// File-scope handle for the Clip producer test stub — non-capturing lambdas
// cannot bind to a specific DQ node, so we thread it through a static.
namespace {
const OrtNode* g_clip_producer_dq = nullptr;
OrtStatus* FakeClipProducerStub(const OrtValueInfo* /*value_info*/,
                                const OrtNode** producer,
                                size_t* output_index) noexcept {
  if (producer) *producer = g_clip_producer_dq;
  if (output_index) *output_index = 0;
  return nullptr;
}
struct ClipProducerGuard {
  explicit ClipProducerGuard(const OrtNode* n) { g_clip_producer_dq = n; }
  ~ClipProducerGuard() { g_clip_producer_dq = nullptr; }
};
}  // namespace

// OrtClipNodeGroupSelector: matching UINT16 input/output types pass the type-equality check
// (no width gates) → returns true.
// Reaches the type check by installing a producer stub that returns the DQ node so
// the producer==nullptr guard passes.
TEST(QnnUnit_EpUtilsTest, Clip_AcceptsUint16) {
  EpUtilsTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, {}};
  FakeValueInfo clip_in{"ci", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"clip", "Clip", "", 11, {&clip_in}, {&main_out}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};
  FakeGraph graph{};

  ClipProducerGuard guard(dq.AsNode());
  ctx.api.ValueInfo_GetValueProducer = &FakeClipProducerStub;

  // Ort::ConstNode(producer).GetOperatorType() uses the global api.
  OrtGlobalApiOverride global_guard(&ctx.api);
  OrtClipNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_TRUE(base.Check(graph.AsGraph(), ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
