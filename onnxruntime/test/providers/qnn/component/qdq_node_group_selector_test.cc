// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for qdq_node_group_selector.cc.
//
// Tests the OrtNodeGroupSelector::Check() family using FakeGraph / FakeNode /
// FakeValueInfo from qnn_fake_ort_graph.h — no real QNN backend required.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <vector>

#include "core/providers/qnn/builder/qdq_node_group_selector.h"
#include "test/providers/qnn/infra/qnn_fake_ort_graph.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime::QDQ;
using namespace onnxruntime::test;

// =============================================================================
// QdqNodeGroupSelectorTestContext
//
// Installs FakeGraph stubs plus the two extras that CheckQDQNodes uses:
//   - ValueInfo_IsGraphOutput  → always false (not a graph output)
//   - ValueInfo_GetValueNumConsumers → always 1 (exactly one consumer)
//
// These two are ORT_CONTINUE_ON_ERROR — the node-group check will return
// false without them if the function pointer is null and crashes, so they
// must always be installed.
// =============================================================================
struct QdqNodeGroupSelectorTestContext {
  OrtApi api{};

  QdqNodeGroupSelectorTestContext() {
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropDQ_RejectsRedundantClipNode) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode clip{"clip", "Clip", "", 13, {}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, clip.AsNode(), {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropDQ_RejectsEmptyDqNodes) {
  QdqNodeGroupSelectorTestContext ctx;
  OrtDropDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropDQ_RejectsTwoDqNodes) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq1.AsNode(), dq2.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropDQ_AcceptsUint8) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropDQ_AcceptsInt8) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropDQ_AcceptsInt16) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtDropDQNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropDQ_AcceptsInt4) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Unary_RejectsWrongDqCount) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"relu", "Relu", "", 13, {}, {&main_out}};

  OrtUnaryNodeGroupSelector sel;
  // dq_nodes empty → CheckQDQNodes fails (0 != 1)
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr, {}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, Unary_AcceptsUint8) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Unary_RejectsTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
  // DQ input = UINT8, Q output = INT8 — types differ. Softmax isn't Convert-compatible
  // (see IsConvertCompatibleUnaryOp), so the mismatch is rejected rather than bridged.
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"softmax", "Softmax", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  // The dtype-mismatch branch calls Ort::ConstNode(node).GetOperatorType(), which needs the global API.
  OrtGlobalApiOverride global_guard(&ctx.api);
  OrtUnaryNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq.AsNode()}, {q.AsNode()}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, Unary_AcceptsInt16) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Binary_RejectsOnlyOneDq) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Binary_AcceptsTwoUint8Dqs) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Binary_RejectsMixedTypes) {
  QdqNodeGroupSelectorTestContext ctx;
  // dq1 = UINT8, dq2 = INT8 — types mismatch. Equal isn't Convert-compatible (see
  // IsConvertCompatibleBinaryOp), so the mismatch is rejected rather than bridged.
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"equal", "Equal", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  // The dtype-mismatch branch calls Ort::ConstNode(node).GetOperatorType(), which needs the global API.
  OrtGlobalApiOverride global_guard(&ctx.api);
  OrtBinaryNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                         {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// dq1 = UINT8, dq2 = INT8 — same bitwidth, differing signedness. Unlike Binary_RejectsMixedTypes,
// Add IS Convert-compatible (see IsConvertCompatibleBinaryOp) and both types are convertible
// fixed-point types, so the selector accepts the group; AlignBinaryInputPrecision (op-builder side)
// leaves the inputs untouched since same-width signedness mismatches aren't Converted (see
// mixed_precision_convert_utils.cc), and QNN op validation is the arbiter of whether Add accepts
// the mismatched-signedness pair natively.
TEST(QnnUnit_QdqNodeGroupSelectorTest, Binary_AcceptsSameWidthSignednessMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in1{"x1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&dq_in1}, {}};
  FakeValueInfo dq_in2{"x2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {1, 4}};
  FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&dq_in2}, {}};

  FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
  FakeNode main_node{"add", "Add", "", 13, {}, {&main_out}};

  FakeValueInfo q_out{"z", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

  OrtGlobalApiOverride global_guard(&ctx.api);
  OrtBinaryNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}));
}

// =============================================================================
// OrtPadNodeGroupSelector::Check
//
// 1 or 2 DQ inputs allowed; input/output types must match.
// =============================================================================

TEST(QnnUnit_QdqNodeGroupSelectorTest, Pad_RejectsThreeDqNodes) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Pad_AcceptsOneDqMatchingType) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Pad_RejectsOneDqMismatchedType) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Pad_AcceptsTwoDqSameType) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Pad_RejectsTwoDqDifferentTypes) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, RegisterSelector_IncreasesSize) {
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Variadic_Accepts2DqSameType) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Variadic_RejectsMixedDqTypes) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Variadic_AcceptsInt16) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Split_RejectsRedundantClipNode) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode clip{"clip", "Clip", "", 13, {}, {}};

  OrtSplitNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, clip.AsNode(), {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, Split_Accepts1DqWith2Q) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Split_RejectsQTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Split_AcceptsInt4) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Conv_Accepts2DqUint8) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Conv_AcceptsInt8) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Conv_Rejects3DqWithNonInt32Bias) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Conv_AcceptsFloatBias) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Conv_RejectsFloatInputWithQuantWeight) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, MatMul_Accepts2DqWithQ) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, MatMul_ConvertBeforeOpGroupHonorsHtpTypeRules) {
  using T = ONNXTensorElementDataType;
  constexpr T u4 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4;
  constexpr T u8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  constexpr T s8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  constexpr T u16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  constexpr T s16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;

  struct Case {
    T activation;
    T weight;
    T output;
    bool expected;
  };
  const Case cases[] = {
      {u8, u8, u16, true},    // widening
      {s8, s8, s16, true},    // widening to signed 16-bit with an int8 weight
      {u8, s16, u16, true},   // 16-bit weight with an unsigned 16-bit output
      {u8, u8, s8, true},     // same width, signedness only
      {u8, u8, s16, false},   // signed 16-bit output needs an int8 weight
      {u8, s16, s16, false},  // signed 16-bit output needs an int8 weight
      {u8, u16, s8, false},   // weight wider than the output
      {u16, u8, u8, false},   // narrowing: narrow-after-op path, which needs a Q scale the fake lacks
      {u4, u8, u8, false},    // no 4-bit activation
      {u8, u4, u16, false},   // no 4-bit weight
  };

  for (const Case& c : cases) {
    QdqNodeGroupSelectorTestContext ctx;
    FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
    FakeValueInfo act_in{"a", c.activation, {4, 4}};
    FakeValueInfo weight_in{"w", c.weight, {4, 4}};
    FakeNode dq1{"dq1", "DequantizeLinear", "", 13, {&act_in}, {}};
    FakeNode dq2{"dq2", "DequantizeLinear", "", 13, {&weight_in}, {}};

    FakeValueInfo main_out{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
    FakeNode main_node{"mm", "MatMul", "", 13, {&dummy, &dummy}, {&main_out}};

    FakeValueInfo q_out{"z", c.output, {}};
    FakeNode q{"q", "QuantizeLinear", "", 13, {}, {&q_out}};

    OrtMatMulNodeGroupSelector sel;
    EXPECT_EQ(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr, {dq1.AsNode(), dq2.AsNode()}, {q.AsNode()}),
              c.expected)
        << "activation=" << c.activation << " weight=" << c.weight << " output=" << c.output;
  }
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, MatMul_RejectsNoQ) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, LogicalComparison_AcceptsMatchingTypes) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, LogicalComparison_RejectsMismatchedTypes) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_RejectsRedundantClipNode) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode clip{"clip", "Clip", "", 13, {}, {}};

  OrtDropQDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, clip.AsNode(), {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_RejectsZeroDqNodes) {
  QdqNodeGroupSelectorTestContext ctx;
  OrtDropQDQNodeGroupSelector sel;
  // CheckQDQNodes expects 1 DQ but gets 0 → false
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {}, {}));
}

// =============================================================================
// CheckQDQNodes — produces_graph_output=true path
// =============================================================================

TEST(QnnUnit_QdqNodeGroupSelectorTest, CheckQDQNodes_RejectsWhenOutputIsGraphOutput) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, CheckQDQNodes_RejectsWhenConsumerCountExceedsQNodes) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_RejectsTypeMismatchAfterCheckQDQNodes) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Conv_RejectsInputOutputTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Conv_RejectsInt8DataWithUint8Weight) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, MatMul_RejectsInt8DataWithUint8Weight) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Gemm_RejectsInt8AWithUint8B) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Gemm_RejectsQOutputTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Einsum_AcceptsUint8NoQ) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"einsum", "Einsum", "", 12, {&dummy}, {}};

  OrtEinsumNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, Einsum_AcceptsInt8) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"einsum", "Einsum", "", 12, {&dummy}, {}};

  OrtEinsumNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, Einsum_AcceptsInt16) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"einsum", "Einsum", "", 12, {&dummy}, {}};

  OrtEinsumNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, Einsum_RejectsQOutputTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Reciprocal_AcceptsUint8NoQ) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"recip", "Reciprocal", "", 13, {&dummy}, {}};

  OrtReciprocalNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, Reciprocal_AcceptsInt8) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Where_RejectsDqTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, Where_AcceptsInt16) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, InstanceNorm_RejectsTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, BatchNorm_RejectsTooFewDq) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};

  OrtBatchNormalizationNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, BatchNorm_RejectsInt8WithMixedScale) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, CumSum_RejectsTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, ScatterElements_RejectsMixedDqTypes) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, RMSNorm_RejectsTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, TopK_RejectsRedundantClipNode) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode clip{"clip", "Clip", "", 13, {}, {}};

  OrtTopKNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, nullptr, clip.AsNode(),
                          {dq.AsNode()}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, TopK_RejectsWrongDqCount) {
  QdqNodeGroupSelectorTestContext ctx;
  OrtTopKNodeGroupSelector sel;
  OrtNodeGroupSelector& base = sel;
  EXPECT_FALSE(base.Check(nullptr, ctx.api, nullptr, nullptr, {}, {}));
}

TEST(QnnUnit_QdqNodeGroupSelectorTest, TopK_CoversCanCreateNodeGroupPath) {
  // CanCreateNodeGroup is exercised here. IsQDQPairSupported will return false
  // (no scale initializer) so the final result is false, but the internal
  // CanCreateNodeGroup + type-check code paths are covered.
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_PositiveScale_SameName_CheckReturnsTrue) {
  QdqNodeGroupSelectorTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(0.5f);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_TRUE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                        {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: negative scale → returns false.
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_NegativeScale_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(-0.5f);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: zero scale → returns false.
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_ZeroScale_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(0.0f);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// allow_nonpositive_scale=true: IsQOrDQScalePositiveConstantScalar is skipped;
// IsQDQPairSupported (same name) returns true even for a negative scale.
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_AllowNonpositiveScale_NegativeScaleOk) {
  QdqNodeGroupSelectorTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeFloat(-1.0f);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/true);
  EXPECT_TRUE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                        {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: scale not in graph.initializers →
// GetConstantInitializer returns nullptr → returns false.
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_ScaleNotInitializer_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
  DropQdqScaleFixture f;
  f.graph.initializers = {};  // scale_vi absent from graph initializers

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQDQPairSupported: different scale names, same float value → same_scale=true.
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_DiffScaleName_SameValue_CheckReturnsTrue) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_DiffScaleName_DiffValue_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_QNodeHasNoScaleInput_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
  DropQdqScaleFixture f;
  // Q node has only 1 input (no scale at index 1).
  f.q_node.inputs = {&f.main_out};

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: double-precision scale, positive →
// element_type==DOUBLE path executed, returns true.
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_DoubleScale_Positive_CheckReturnsTrue) {
  QdqNodeGroupSelectorTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val = FakeOrtValue::MakeDouble(0.5);
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_TRUE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                        {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQOrDQScalePositiveConstantScalar: scale is INT32 type (neither float nor
// double) → falls through to the final return false.
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_IntTypeScale_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
  DropQdqScaleFixture f;
  f.q_scale_val.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  f.q_scale_vi.initializer_value = &f.q_scale_val;

  OrtDropQDQNodeGroupSelector sel(/*allow_nonpositive_scale=*/false);
  EXPECT_FALSE(sel.Check(f.graph.AsGraph(), ctx.api, f.main_node.AsNode(), nullptr,
                         {f.dq_node.AsNode()}, {f.q_node.AsNode()}));
}

// IsQDQPairSupported: different scale names, Q=FLOAT vs DQ=DOUBLE →
// q_element_type != dq_element_type → returns false.
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_DiffScaleName_TypeMismatch_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_DiffScaleName_BothDouble_SameValue_CheckReturnsTrue) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropQDQ_DQHasNoInputs_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, DropDQ_DQHasNoInputs_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {}, {}};
  OrtDropDQNodeGroupSelector sel;
  EXPECT_FALSE(sel.Check(nullptr, ctx.api, nullptr, nullptr, {dq.AsNode()}, {}));
}

// Unary: DQ node has no inputs → !dt_input.has_value() → returns false.
TEST(QnnUnit_QdqNodeGroupSelectorTest, Unary_DQHasNoInputs_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Variadic_ZeroDqNodes_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Variadic_DQHasNoInputs_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Split_DQHasNoInputs_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Split_ReqEqualQuant_DiffScaleValues_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Conv_FirstDQHasNoInputs_CheckReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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

TEST(QnnUnit_QdqNodeGroupSelectorTest, GetOrtQDQSelections_MaxPoolVersionCheck_CoversVersionPath) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Clip_CheckQDQNodesFails_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Clip_NullProducer_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Binary_FirstDQHasNoInputs_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Variadic_QHasNoOutputs_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Einsum_AcceptsInt4) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Reciprocal_AcceptsInt16) {
  QdqNodeGroupSelectorTestContext ctx;
  FakeValueInfo dummy{"d", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {}};
  FakeValueInfo dq_in{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, {}};
  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&dq_in}, {}};
  FakeNode main_node{"reciprocal", "Reciprocal", "", 13, {&dummy}, {}};
  OrtReciprocalNodeGroupSelector sel;
  EXPECT_TRUE(sel.Check(nullptr, ctx.api, main_node.AsNode(), nullptr,
                        {dq.AsNode()}, {}));
}

// OrtReciprocalNodeGroupSelector: 4-bit input accepted (no width gates).
TEST(QnnUnit_QdqNodeGroupSelectorTest, Reciprocal_AcceptsInt4) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Reciprocal_QHasNoOutputs_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Reciprocal_QOutputTypeMismatch_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, MatMul_FirstDQHasNoInputs_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Variadic_SecondQTypeMismatch_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, MatMul_QLinearCheckQDQNodesFails_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
// qdq_node_group_selector.cc L736/1037/1065/1093/1192/1206/1267
// =============================================================================

// OrtGemmNodeGroupSelector: dq_nodes.size() < 2 → returns false.
// CheckQDQNodes passes because is_empty_q_nodes_allowed=true and node has 1 input.
TEST(QnnUnit_QdqNodeGroupSelectorTest, Gemm_RejectsFewerThan2Dq) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Gemm_AcceptsUint16) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Where_RejectsWrongDqCount) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, BatchNorm_CheckQDQNodesFails_ReturnsFalse) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, BatchNorm_RejectsInputOutputTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, TopK_RejectsInputOutputTypeMismatch) {
  QdqNodeGroupSelectorTestContext ctx;
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
TEST(QnnUnit_QdqNodeGroupSelectorTest, Clip_AcceptsUint16) {
  QdqNodeGroupSelectorTestContext ctx;
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
