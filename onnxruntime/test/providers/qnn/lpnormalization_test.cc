// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// LpNormalization with p=2 maps to QNN's L2Norm operator.
TEST_F(QnnHTPBackendTests, LpNormalization_u8_rank4) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunQdqModelOnHtp<uint8_t>("LpNormalization",
                            {TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                            {utils::MakeAttribute("axis", static_cast<int64_t>(-1)),
                             utils::MakeAttribute("p", static_cast<int64_t>(2))},
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpNormalization_u16_rank4) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunQdqModelOnHtp<uint16_t>("LpNormalization",
                             {TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                             {utils::MakeAttribute("axis", static_cast<int64_t>(-1)),
                              utils::MakeAttribute("p", static_cast<int64_t>(2))},
                             13,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
