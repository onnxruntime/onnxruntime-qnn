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

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test that Shape with default attributes works on QNN HTP backend.
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

// Test that Shape with start=1 and end=3 works on QNN HTP backend.
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

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
