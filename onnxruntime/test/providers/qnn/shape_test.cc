// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/shape_op_builder.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

//
// ResolveShapeBounds unit tests (pure function — no QNN runtime required).
// Scenarios mirror the ONNX Shape spec (opset >= 15) and the 20-case checklist from PR review.
//
namespace {

struct ResolvedBounds {
  int64_t start;
  int64_t end;
  int64_t output_length;
};

ResolvedBounds Resolve(int64_t rank, int64_t start_attr, int64_t end_attr) {
  ResolvedBounds r{};
  const auto [start, end] = qnn::ResolveShapeBounds(rank, start_attr, end_attr);
  r.start = start;
  r.end = end;
  r.output_length = std::max<int64_t>(0, end - start);
  return r;
}

}  // namespace

// (1) Defaults: start=0, end=rank -> full shape.
TEST(QnnResolveShapeBoundsTest, DefaultsCoverFullRank) {
  auto r = Resolve(/*rank=*/4, /*start_attr=*/0, /*end_attr=*/4);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 4);
  EXPECT_EQ(r.output_length, 4);
}

// (2) start default (0), explicit end < rank.
TEST(QnnResolveShapeBoundsTest, DefaultStartExplicitEnd) {
  auto r = Resolve(4, 0, 2);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 2);
  EXPECT_EQ(r.output_length, 2);
}

// (3) Explicit start, end default (=rank).
TEST(QnnResolveShapeBoundsTest, ExplicitStartDefaultEnd) {
  auto r = Resolve(4, 1, 4);
  EXPECT_EQ(r.start, 1);
  EXPECT_EQ(r.end, 4);
  EXPECT_EQ(r.output_length, 3);
}

// (4) Basic positive: start=1, end=3.
TEST(QnnResolveShapeBoundsTest, BasicPositiveRange) {
  auto r = Resolve(4, 1, 3);
  EXPECT_EQ(r.start, 1);
  EXPECT_EQ(r.end, 3);
  EXPECT_EQ(r.output_length, 2);
}

// (5) Full span: start=0, end=rank.
TEST(QnnResolveShapeBoundsTest, FullSpanZeroToRank) {
  auto r = Resolve(5, 0, 5);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 5);
  EXPECT_EQ(r.output_length, 5);
}

// (6) Negative start: start=-1 normalizes to rank-1.
TEST(QnnResolveShapeBoundsTest, NegativeStart) {
  auto r = Resolve(4, -1, 4);
  EXPECT_EQ(r.start, 3);
  EXPECT_EQ(r.end, 4);
  EXPECT_EQ(r.output_length, 1);
}

// (7) Negative end: end=-1 normalizes to rank-1.
TEST(QnnResolveShapeBoundsTest, NegativeEnd) {
  auto r = Resolve(4, 0, -1);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 3);
  EXPECT_EQ(r.output_length, 3);
}

// (8) Negative start exactly at -rank: start=-r normalizes to 0.
TEST(QnnResolveShapeBoundsTest, NegativeStartEqualMinusRank) {
  auto r = Resolve(4, -4, 4);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 4);
  EXPECT_EQ(r.output_length, 4);
}

// (9) Both negative: start=-2, end=-1 -> (r-2, r-1).
TEST(QnnResolveShapeBoundsTest, BothNegative) {
  auto r = Resolve(4, -2, -1);
  EXPECT_EQ(r.start, 2);
  EXPECT_EQ(r.end, 3);
  EXPECT_EQ(r.output_length, 1);
}

// (10) Clamp high: end > rank -> end clamped to rank.
TEST(QnnResolveShapeBoundsTest, ClampEndAboveRank) {
  auto r = Resolve(4, 0, 100);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 4);
  EXPECT_EQ(r.output_length, 4);
}

// (11) Clamp high: start > rank -> start clamped to rank (empty result).
TEST(QnnResolveShapeBoundsTest, ClampStartAboveRank) {
  auto r = Resolve(4, 100, 4);
  EXPECT_EQ(r.start, 4);
  EXPECT_EQ(r.end, 4);
  EXPECT_EQ(r.output_length, 0);
}

// (12) Clamp low: start < -rank -> 0.
TEST(QnnResolveShapeBoundsTest, ClampStartBelowMinusRank) {
  auto r = Resolve(4, -100, 4);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 4);
  EXPECT_EQ(r.output_length, 4);
}

// (13) Clamp low: end < -rank -> 0.
TEST(QnnResolveShapeBoundsTest, ClampEndBelowMinusRank) {
  auto r = Resolve(4, 0, -100);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 0);
  EXPECT_EQ(r.output_length, 0);
}

// (14) start > end -> empty result, output_length clamped to 0.
TEST(QnnResolveShapeBoundsTest, StartGreaterThanEndIsEmpty) {
  auto r = Resolve(4, 3, 1);
  EXPECT_EQ(r.start, 3);
  EXPECT_EQ(r.end, 1);
  EXPECT_EQ(r.output_length, 0);
}

// (15) start == end (mid-range) -> empty result.
TEST(QnnResolveShapeBoundsTest, StartEqualsEndIsEmpty) {
  auto r = Resolve(4, 2, 2);
  EXPECT_EQ(r.start, 2);
  EXPECT_EQ(r.end, 2);
  EXPECT_EQ(r.output_length, 0);
}

// (16) Scalar input (rank=0): defaults yield (0, 0).
TEST(QnnResolveShapeBoundsTest, ScalarRankYieldsEmpty) {
  auto r = Resolve(0, 0, 0);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 0);
  EXPECT_EQ(r.output_length, 0);
}

// (17) Rank 1: start=0, end=1 -> full shape (single dim).
TEST(QnnResolveShapeBoundsTest, Rank1FullSpan) {
  auto r = Resolve(1, 0, 1);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 1);
  EXPECT_EQ(r.output_length, 1);
}

// (18) Boundary: start=rank, end=rank -> empty.
TEST(QnnResolveShapeBoundsTest, BothAtRankIsEmpty) {
  auto r = Resolve(4, 4, 4);
  EXPECT_EQ(r.start, 4);
  EXPECT_EQ(r.end, 4);
  EXPECT_EQ(r.output_length, 0);
}

// (19) Boundary: start=0, end=0 -> empty.
TEST(QnnResolveShapeBoundsTest, BothAtZeroIsEmpty) {
  auto r = Resolve(4, 0, 0);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 0);
  EXPECT_EQ(r.output_length, 0);
}

// (20) Negative one past the lower edge: start=-(rank+1) -> clamped to 0.
TEST(QnnResolveShapeBoundsTest, NegativeOnePastLowerEdge) {
  auto r = Resolve(4, -5, 4);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 4);
  EXPECT_EQ(r.output_length, 4);
}

//
// Integration tests (QNN runtime required).
//

// Runs a Shape model on the specified QNN backend. Checks the graph node assignment and that inference
// outputs for QNN EP and CPU EP match.
static void RunShapeOpTest(TestInputDef<float> input_def,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           const std::string& backend_name = "cpu",
                           int opset = 15) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;

  RunQnnModelTest(BuildOpTestCase<float>("shape_node", "Shape", {input_def}, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Builds a QDQ model wrapping ONNX Shape. Shape's output is int64 (data-independent), so only
// the input is quantized -- there is no output Q/DQ node.
template <typename QType = uint8_t>
static GetTestQDQModelFn<QType> BuildQDQShapeTestCase(TestInputDef<float> input_def,
                                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                      bool use_contrib_qdq = false) {
  return [input_def, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                             std::vector<QuantParams<QType>>& output_qparams) {
    QNN_TEST_UNUSED_PARAMETER(output_qparams);
    MakeTestInput(builder, "X", input_def);
    QuantParams<QType> input_qparams = GetTestInputQuantParams<QType>(input_def);
    std::string x_dq_name = AddQDQNodePair<QType>(builder, "qdq1", "X", input_qparams.scale,
                                                  input_qparams.zero_point, use_contrib_qdq);

    // DQ -> Shape
    builder.AddNode("shape_node", "Shape", {x_dq_name.c_str()}, {"Y"}, "", attrs);
    builder.MakeOutput("Y");
  };
}

// Runs a QDQ Shape model on the QNN HTP backend and checks output accuracy vs the CPU EP baseline.
template <typename QType = uint8_t>
static void RunQDQShapeOpTest(TestInputDef<float> input_def,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              int opset = 15,
                              bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>("shape_node", "Shape", {input_def}, {}, attrs),
                       BuildQDQShapeTestCase<QType>(input_def, attrs, use_contrib_qdq),
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Shape with default attributes (no start/end) works on QNN CPU backend.
// Input: float32 [3, 4, 5]. Expected output: int64 [3, 4, 5] (QNN EP downcasts to int32 internally,
// then BaseOpBuilder::ProcessOutputs casts back to int64 for the graph output).
TEST_F(QnnCPUBackendTests, Shape_Default_Float) {
  RunShapeOpTest(TestInputDef<float>({3, 4, 5}, false, -10.0f, 10.0f),
                 {},  // Default attributes: start=0, end=rank.
                 ExpectedEPNodeAssignment::All, "cpu", 15);
}

// Test that Shape with explicit start=1 and end=3 works on QNN CPU backend.
// Input: float32 [2, 3, 4, 5]. Expected output: int64 [3, 4].
TEST_F(QnnCPUBackendTests, Shape_StartEnd_Float) {
  RunShapeOpTest(TestInputDef<float>({2, 3, 4, 5}, false, -10.0f, 10.0f),
                 {test::MakeAttribute("start", static_cast<int64_t>(1)),
                  test::MakeAttribute("end", static_cast<int64_t>(3))},
                 ExpectedEPNodeAssignment::All, "cpu", 15);
}

// Test that Shape with a negative start index is normalized correctly on QNN CPU backend.
// Input: float32 [2, 3, 4]. start=-2 normalizes to rank+(-2) = 3-2 = 1, end=3.
// Expected output: int64 [3, 4].
TEST_F(QnnCPUBackendTests, Shape_NegativeStart_Float) {
  RunShapeOpTest(TestInputDef<float>({2, 3, 4}, false, -10.0f, 10.0f),
                 {test::MakeAttribute("start", static_cast<int64_t>(-2)),
                  test::MakeAttribute("end", static_cast<int64_t>(3))},
                 ExpectedEPNodeAssignment::All, "cpu", 15);
}

// Test that Shape on a 1-D input works on QNN CPU backend.
// Input: float32 [7]. Expected output: int64 [7].
TEST_F(QnnCPUBackendTests, Shape_1D_Float) {
  RunShapeOpTest(TestInputDef<float>({7}, false, -10.0f, 10.0f),
                 {},  // Default attributes: start=0, end=rank=1.
                 ExpectedEPNodeAssignment::All, "cpu", 15);
}

// Test that an empty shape slice (start == end) is NOT assigned to QNN EP.
// ONNX defines output_length = max(0, end - start), so start == end is a valid empty slice
// (length 0). However, QNN's Shape op (QnnOpDef MasterOpDef) requires end in [start + 1, N] and
// cannot represent a zero-length output, so the op builder rejects it during IsOpSupported() and
// the node falls back to the CPU EP.
// Input: float32 [2, 3, 4, 5]. start=2, end=2 -> empty slice.
TEST_F(QnnCPUBackendTests, Shape_EmptySlice_Float) {
  RunShapeOpTest(TestInputDef<float>({2, 3, 4, 5}, false, -10.0f, 10.0f),
                 {test::MakeAttribute("start", static_cast<int64_t>(2)),
                  test::MakeAttribute("end", static_cast<int64_t>(2))},
                 ExpectedEPNodeAssignment::None, "cpu", 15);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test that Shape with default attributes works on QNN HTP backend (FP32 input).
// Shape is a data-independent op (output depends only on input shape, not values),
// so HTP support may vary by SDK version. The test verifies that when the op is
// assigned to QNN EP the outputs match the CPU EP baseline.
TEST_F(QnnHTPBackendTests, Shape_Default_Float_HTP) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("shape_node", "Shape",
                                         {TestInputDef<float>({3, 4, 5}, false, -10.0f, 10.0f)},
                                         {}, {}),
                  provider_options,
                  15,
                  ExpectedEPNodeAssignment::All);
}

// Test that Shape with start=1 and end=3 works on QNN HTP backend (FP32 input).
// Input: float32 [2, 3, 4, 5]. Expected output: int64 [3, 4].
TEST_F(QnnHTPBackendTests, Shape_StartEnd_Float_HTP) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("shape_node", "Shape",
                                         {TestInputDef<float>({2, 3, 4, 5}, false, -10.0f, 10.0f)},
                                         {},
                                         {test::MakeAttribute("start", static_cast<int64_t>(1)),
                                          test::MakeAttribute("end", static_cast<int64_t>(3))}),
                  provider_options,
                  15,
                  ExpectedEPNodeAssignment::All);
}

// QDQ (uint8) Shape with default attributes on HTP. Shape is data-independent so only the
// input is quantized; output is int64 and passes through unquantized.
TEST_F(QnnHTPBackendTests, Shape_Default_QDQ_U8_HTP) {
  RunQDQShapeOpTest<uint8_t>(TestInputDef<float>({3, 4, 5}, false, -10.0f, 10.0f),
                             {},
                             ExpectedEPNodeAssignment::All);
}

// QDQ (uint8) Shape with start=1, end=3 on HTP.
// Input: uint8-quantized [2, 3, 4, 5]. Expected output: int64 [3, 4].
TEST_F(QnnHTPBackendTests, Shape_StartEnd_QDQ_U8_HTP) {
  RunQDQShapeOpTest<uint8_t>(TestInputDef<float>({2, 3, 4, 5}, false, -10.0f, 10.0f),
                             {test::MakeAttribute("start", static_cast<int64_t>(1)),
                              test::MakeAttribute("end", static_cast<int64_t>(3))},
                             ExpectedEPNodeAssignment::All);
}

// QDQ (uint16) Shape with default attributes on HTP.
TEST_F(QnnHTPBackendTests, Shape_Default_QDQ_U16_HTP) {
  RunQDQShapeOpTest<uint16_t>(TestInputDef<float>({3, 4, 5}, false, -10.0f, 10.0f),
                              {},
                              ExpectedEPNodeAssignment::All,
                              15,     // opset
                              true);  // Use com.microsoft Q/DQ ops (uint16 zero-point not in ONNX opset 15)
}

// HtpOpDefSupplement caps Shape's input rank at 4 on HTP. A rank-5 input must fall back to CPU EP
// (ExpectedEPNodeAssignment::None), mirroring ArgMaxMinU8_RankGreaterThan4_Unsupported.
TEST_F(QnnHTPBackendTests, Shape_RankGreaterThan4_Unsupported) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("shape_node", "Shape",
                                         {TestInputDef<float>({2, 3, 4, 5, 6}, false, -10.0f, 10.0f)},
                                         {}, {}),
                  provider_options,
                  15,
                  ExpectedEPNodeAssignment::None);
}

// Builds a model from `build_model`, runs it on the CPU EP and the QNN HTP EP, and compares each
// graph output element-by-element. Used by the Shape -> Gather composition tests below.
static void RunShapeCompositionTest(const GetTestModelFn& build_model, int opset_version,
                                    ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All) {
  ModelTestBuilder helper;
  build_model(helper);
  const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
  opset_id_proto->set_domain(kOnnxDomain);
  opset_id_proto->set_version(opset_version);
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  std::vector<Ort::Value> expected;
  InferenceModelCPU(model_data, "Shape_Composition_CPU", helper.feeds_, expected);
  std::vector<Ort::Value> actual;
  InferenceModel(model_data, "Shape_Composition_QNN", provider_options, expected_ep_assignment,
                 helper.feeds_, actual);

  ASSERT_EQ(expected.size(), actual.size());
  for (size_t out_idx = 0; out_idx < expected.size(); ++out_idx) {
    auto exp_info = expected[out_idx].GetTensorTypeAndShapeInfo();
    auto act_info = actual[out_idx].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(exp_info.GetElementType(), act_info.GetElementType())
        << "Element type mismatch for output " << out_idx;
    auto exp_shape = exp_info.GetShape();
    auto act_shape = act_info.GetShape();
    ASSERT_EQ(exp_shape, act_shape) << "Shape mismatch for output " << out_idx;
    const size_t element_count = exp_info.GetElementCount();
    const int64_t* exp_data = expected[out_idx].GetTensorData<int64_t>();
    const int64_t* act_data = actual[out_idx].GetTensorData<int64_t>();
    for (size_t i = 0; i < element_count; ++i) {
      EXPECT_EQ(exp_data[i], act_data[i]) << "Mismatch at output " << out_idx << " index " << i;
    }
  }
}

// Shape output is consumed by Gather (intermediate-output path). The Shape tensor is INT_32 on the
// QNN side via ShapeOpBuilder::GetSupportedOutputDataType; Gather then consumes it without an
// INT_64 -> INT_32 cast (ProcessInt64Tensors is a no-op because the tensor is already INT_32).
TEST_F(QnnHTPBackendTests, Shape_To_Gather_HTP) {
  RunShapeCompositionTest([](ModelTestBuilder& builder) {
    TestInputDef<float> input_def({2, 3, 4}, false, -10.0f, 10.0f);
    MakeTestInput<float>(builder, "X", input_def);

    // Shape(X) -> shape_out [3] (int64, intermediate).
    builder.AddNode("shape_node", "Shape", {"X"}, {"shape_out"}, kOnnxDomain);

    // Gather(shape_out, idx=1, axis=0) -> Y = shape_out[1] = 3.
    builder.MakeInitializer<int64_t>("idx", {1}, {1});
    builder.AddNode("gather_node", "Gather", {"shape_out", "idx"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))});
    builder.MakeOutput<int64_t>("Y", std::vector<int64_t>{1});
  },
                          15);
}

// Shape output is BOTH a graph output AND consumed by a downstream node. This exercises the
// dual-use path: BaseOpBuilder::ProcessOutputs inserts a Cast (INT_32 -> INT_64) for the graph
// output, while the QNN Gather node consumes the native INT_64 graph-output tensor (which the
// downstream op's ProcessInt64Tensors then casts back to INT_32).
TEST_F(QnnHTPBackendTests, Shape_DualUse_HTP) {
  RunShapeCompositionTest([](ModelTestBuilder& builder) {
    TestInputDef<float> input_def({2, 3, 4}, false, -10.0f, 10.0f);
    MakeTestInput<float>(builder, "X", input_def);

    // Shape(X) -> shape_out is BOTH a graph output and a Gather input.
    builder.AddNode("shape_node", "Shape", {"X"}, {"shape_out"}, kOnnxDomain);
    builder.MakeOutput<int64_t>("shape_out", std::vector<int64_t>{3});

    builder.MakeInitializer<int64_t>("idx", {1}, {1});
    builder.AddNode("gather_node", "Gather", {"shape_out", "idx"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))});
    builder.MakeOutput<int64_t>("Y", std::vector<int64_t>{1});
  },
                          15);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
