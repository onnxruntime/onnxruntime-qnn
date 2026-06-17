// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

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

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
