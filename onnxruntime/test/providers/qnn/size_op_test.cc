// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

// Builds a Size graph with a live (non-constant) input.
//   Input: live tensor "X" with the given shape, values in [min_val, max_val]
//   Output: scalar int64 "Y" (Size result) and float "X_pass" (Identity passthrough)
//
// The Identity node is required: QNN requires every APP_WRITE (live graph input) to
// have at least one consumer node. In production models X is always consumed by other
// ops alongside Size; Identity replicates that pattern in the isolated test graph.
template <typename DataType>
static GetTestModelFn BuildSizeLiveInputTestCase(const std::vector<int64_t>& shape,
                                                 DataType min_val, DataType max_val) {
  return [shape, min_val, max_val](ModelTestBuilder& builder) {
    builder.MakeInput<DataType>("X", shape, min_val, max_val);
    builder.AddNode("size_node", "Size", {"X"}, {"Y"}, kOnnxDomain);
    builder.AddNode("identity_node", "Identity", {"X"}, {"X_pass"}, kOnnxDomain);
    builder.MakeOutput("Y");
    builder.MakeOutput("X_pass");
  };
}

// Runs a Size model test on the QNN HTP backend (constant initializer input).
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

// Runs a Size model test on the QNN HTP backend (live input).
template <typename DataType>
static void RunSizeLiveInputTest(const std::vector<int64_t>& shape,
                                 DataType min_val, DataType max_val,
                                 ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildSizeLiveInputTestCase<DataType>(shape, min_val, max_val),
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
// HTP backend tests — live input (non-constant, static shape)
//

// 2-D live input: 3x4 = 12
TEST_F(QnnHTPBackendTests, Size_2D_Float_LiveInput) {
  RunSizeLiveInputTest<float>({3, 4}, 0.0f, 1.0f, ExpectedEPNodeAssignment::All);
}

// 4-D live input: 2x3x4x5 = 120
TEST_F(QnnHTPBackendTests, Size_4D_Float_LiveInput) {
  RunSizeLiveInputTest<float>({2, 3, 4, 5}, 0.0f, 1.0f, ExpectedEPNodeAssignment::All);
}

// 1-D int32 live input: size = 7
TEST_F(QnnHTPBackendTests, Size_1D_Int32_LiveInput) {
  RunSizeLiveInputTest<int32_t>({7}, 0, 10, ExpectedEPNodeAssignment::All);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
