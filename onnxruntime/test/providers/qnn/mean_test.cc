// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, Mean_TwoInputs) {
  std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input2 = {5.0f, 6.0f, 7.0f, 8.0f};
  RunF32ModelOnHtp<float>("Mean",
                          {TestInputDef<float>({4}, false, std::move(input1)),
                           TestInputDef<float>({4}, false, std::move(input2))},
                          {},
                          13,
                          ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Mean_FourInputs) {
  std::vector<float> input1 = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> input2 = {2.0f, 2.0f, 2.0f, 2.0f};
  std::vector<float> input3 = {3.0f, 3.0f, 3.0f, 3.0f};
  std::vector<float> input4 = {4.0f, 4.0f, 4.0f, 4.0f};
  RunF32ModelOnHtp<float>("Mean",
                          {TestInputDef<float>({4}, false, std::move(input1)),
                           TestInputDef<float>({4}, false, std::move(input2)),
                           TestInputDef<float>({4}, false, std::move(input3)),
                           TestInputDef<float>({4}, false, std::move(input4))},
                          {},
                          13,
                          ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Mean_TwoInputs_QU8) {
  RunQdqModelOnHtp<uint8_t>("Mean",
                            {TestInputDef<float>({1, 2, 2}, false, GetFloatDataInRange(0.0f, 10.0f, 4)),
                             TestInputDef<float>({1, 2, 2}, false, GetFloatDataInRange(10.0f, 20.0f, 4))},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Mean_FourInputs_QU8) {
  RunQdqModelOnHtp<uint8_t>("Mean",
                            {TestInputDef<float>({1, 2, 2}, false, GetFloatDataInRange(0.0f, 10.0f, 4)),
                             TestInputDef<float>({1, 2, 2}, false, GetFloatDataInRange(10.0f, 20.0f, 4)),
                             TestInputDef<float>({1, 2, 2}, false, GetFloatDataInRange(20.0f, 30.0f, 4)),
                             TestInputDef<float>({1, 2, 2}, false, GetFloatDataInRange(30.0f, 40.0f, 4))},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
