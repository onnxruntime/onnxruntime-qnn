// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, Reciprocal_Basic_FLOAT) {
  RunF32ModelOnHtp<float>("Reciprocal",
                          {TestInputDef<float>({2, 2}, false, {1.0f, 2.0f, 0.5f, 4.0f})},
                          {},
                          13,
                          ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Reciprocal_QU8) {
  RunQdqModelOnHtp<uint8_t>("Reciprocal",
                            {TestInputDef<float>({2, 2}, false, GetFloatDataInRange(1.0f, 5.0f, 4))},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
