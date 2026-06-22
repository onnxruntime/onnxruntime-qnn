// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

// Builds a graph with a single Size node.
//   Input: tensor named "X" with the given shape and data
//   Output: scalar int64 tensor named "Y"
template <typename DataType>
static GetTestModelFn BuildSizeTestCase(const std::vector<int64_t>& shape,
                                        const std::vector<DataType>& data) {
  return [shape, data](ModelTestBuilder& builder) {
    builder.MakeInput<DataType>("X", shape, data);
    builder.AddNode("size_node", "Size", {"X"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput<int64_t>("Y", std::vector<int64_t>{});  // 0-D scalar output
  };
}

// Runs a Size model test on the QNN CPU backend.
template <typename DataType>
static void RunSizeTest(const std::vector<int64_t>& shape,
                        const std::vector<DataType>& data,
                        ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildSizeTestCase<DataType>(shape, data),
                  provider_options,
                  /*opset_version=*/13,
                  expected_ep_assignment);
}

//
// CPU backend tests
//

// 2-D input: 3x4 = 12
TEST_F(QnnCPUBackendTests, Size_2D_Float) {
  RunSizeTest<float>({3, 4}, std::vector<float>(12, 1.0f), ExpectedEPNodeAssignment::All);
}

// 4-D input: 2x3x4x5 = 120
TEST_F(QnnCPUBackendTests, Size_4D_Float) {
  RunSizeTest<float>({2, 3, 4, 5}, std::vector<float>(120, 1.0f), ExpectedEPNodeAssignment::All);
}

// 0-D (scalar) input: size = 1
TEST_F(QnnCPUBackendTests, Size_Scalar_Float) {
  RunSizeTest<float>({}, std::vector<float>{1.0f}, ExpectedEPNodeAssignment::All);
}

// 1-D int32 input: size = 7
TEST_F(QnnCPUBackendTests, Size_1D_Int32) {
  RunSizeTest<int32_t>({7}, std::vector<int32_t>(7, 0), ExpectedEPNodeAssignment::All);
}

// Input with a zero-sized dimension: size = 0
TEST_F(QnnCPUBackendTests, Size_ZeroDim_Float) {
  RunSizeTest<float>({3, 0, 4}, std::vector<float>{}, ExpectedEPNodeAssignment::All);
}


}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
