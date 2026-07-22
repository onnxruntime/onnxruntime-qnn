// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

// Builds a Size graph with a constant initializer input.
//   Input: constant tensor "X" with the given shape and data
//   Output: scalar int64 tensor "Y"
template <typename DataType>
static GetTestModelFn BuildSizeTestCase(const std::vector<int64_t>& shape,
                                        const std::vector<DataType>& data) {
  return [shape, data](ModelTestBuilder& builder) {
    builder.MakeInitializer<DataType>("X", shape, data);
    builder.AddNode("size_node", "Size", {"X"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  };
}

// Builds a graph: Reshape(X_flat, new_shape) -> Size -> Y (graph output).
// Tests that Size handles the output of another QNN op as its input.
static GetTestModelFn BuildSizeAfterReshapeTestCase() {
  return [](ModelTestBuilder& builder) {
    // X_flat: float constant [12]
    builder.MakeInitializer<float>("X_flat", {12}, std::vector<float>(12, 1.0f));
    // new_shape: int64 constant [3, 4]
    builder.MakeInitializer<int64_t>("new_shape", {2}, std::vector<int64_t>{3, 4});
    builder.AddNode("reshape_node", "Reshape", {"X_flat", "new_shape"}, {"X_reshaped"}, kOnnxDomain);
    builder.AddNode("size_node", "Size", {"X_reshaped"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");  // expected: 12
  };
}

// Builds a graph: X (const) -> Size -> Cast(to float) -> Z (graph output).
// Tests that the Size output (int64 folded constant) correctly feeds a downstream op.
static GetTestModelFn BuildSizeAsIntermediateTestCase() {
  return [](ModelTestBuilder& builder) {
    builder.MakeInitializer<float>("X", {3, 4}, std::vector<float>(12, 1.0f));
    builder.AddNode("size_node", "Size", {"X"}, {"size_out"}, kOnnxDomain);
    builder.AddNode("cast_node", "Cast", {"size_out"}, {"Z"}, kOnnxDomain,
                    {builder.MakeScalarAttribute(
                        "to", static_cast<int64_t>(
                                  ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.MakeOutput("Z");  // expected: 12.0f
  };
}

// Runs a Size model test on the QNN HTP backend.
template <typename DataType>
static void RunSizeTest(const std::vector<int64_t>& shape,
                        const std::vector<DataType>& data,
                        ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildSizeTestCase<DataType>(shape, data),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{expected_ep_assignment});
}

//
// HTP backend tests — constant initializer input
//

// 2-D input: 3x4 = 12
TEST_F(QnnHTPBackendTests, Size_2D_Float) {
  RunSizeTest<float>({3, 4}, std::vector<float>(12, 1.0f), ExpectedEPNodeAssignment::All);
}

// 4-D input: 2x3x4x5 = 120
TEST_F(QnnHTPBackendTests, Size_4D_Float) {
  RunSizeTest<float>({2, 3, 4, 5}, std::vector<float>(120, 1.0f), ExpectedEPNodeAssignment::All);
}

// 0-D (scalar) input: size = 1
TEST_F(QnnHTPBackendTests, Size_Scalar_Float) {
  RunSizeTest<float>({}, std::vector<float>{1.0f}, ExpectedEPNodeAssignment::All);
}

// 1-D int32 input: size = 7
TEST_F(QnnHTPBackendTests, Size_1D_Int32) {
  RunSizeTest<int32_t>({7}, std::vector<int32_t>(7, 0), ExpectedEPNodeAssignment::All);
}

// Input with a zero-sized dimension: size = 0
TEST_F(QnnHTPBackendTests, Size_ZeroDim_Float) {
  RunSizeTest<float>({3, 0, 4}, std::vector<float>{}, ExpectedEPNodeAssignment::All);
}

//
// HTP backend tests — Size input is the output of another QNN op
//

// Reshape -> Size -> graph output  (tests NATIVE tensor input to Size)
TEST_F(QnnHTPBackendTests, Size_AfterReshape_GraphOutput) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildSizeAfterReshapeTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
}

// Size -> Cast -> graph output  (tests Size output feeding a downstream op)
TEST_F(QnnHTPBackendTests, Size_Intermediate_FeedsDownstreamOp) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildSizeAsIntermediateTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
