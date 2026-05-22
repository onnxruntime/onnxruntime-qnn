// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qdq_test_utils.h"
#include <type_traits>
#include <utility>

namespace onnxruntime {
namespace test {

GetQDQTestCaseFn BuildQDQReshapeTestCase(const std::vector<int64_t>& input_shape,
                                         const std::vector<int64_t>& reshape_shape) {
  return [input_shape, reshape_shape](ModelTestBuilder& builder) {
    builder.MakeInput<uint8_t>("X", input_shape,
                               std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    builder.MakeOutput("Y");

    // add DQ
    builder.AddDequantizeLinearNode<uint8_t>("dq", "X", .003f, 1, "dq_out");

    // add Reshape
    builder.Make1DInitializer<int64_t>("shape", reshape_shape);
    builder.AddNode("reshape", "Reshape",
                    {"dq_out", "shape"},
                    {"reshape_output"});

    // add Q
    builder.AddQuantizeLinearNode<uint8_t>(
        "q", "reshape_output",
        .003f, 1, "q_out");
  };
}

}  // namespace test
}  // namespace onnxruntime
