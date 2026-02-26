// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// ============================================================
// CPU backend tests
// ============================================================

// TODO: Flaky test tails often.
// Value of: expected_tensor.DataAsSpan<float>()
// Expected: contains 16 values, where each value and its corresponding value in 16-byte object
// <10-00 00-00 00-00 00-00 40-00 23-D1 82-02 00-00> are an almost-equal pair
// Actual: 16-byte object <10-00 00-00 00-00 00-00 40-00 12-D1 82-02 00-00>, where the value pair (2, 0.1) at
// index #2 don't match, which is -1.9 from 2
// If/when fixed, enable QNN EP in cpu test TensorOpTest.SpaceToDepthTest_1. Fixed by QNN 2.32.
TEST_F(QnnCPUBackendTests, SpaceToDepth_Flaky) {
  std::vector<float> X = {0.0f, 0.1f, 0.2f, 0.3f,
                          1.0f, 1.1f, 1.2f, 1.3f,
                          2.0f, 2.1f, 2.2f, 2.3f,
                          3.0f, 3.1f, 3.2f, 3.3f};

  for (size_t i = 0; i < 4; i++) {
    RunF32ModelOnCpu("SpaceToDepth",
                     {TestInputDef<float>({1, 2, 2, 4}, false, X)},
                     {utils::MakeAttribute("blocksize", static_cast<int64_t>(2))},
                     7,
                     ExpectedEPNodeAssignment::All);
  }
}

// Value of: expected_tensor.DataAsSpan<float>()
// Expected: contains 108 values, where each value and its corresponding value in 16-byte object
// <6C-00 00-00 00-00 00-00 40-00 23-BB 0E-02 00-00> are an almost-equal pair
// Actual: 16-byte object <6C-00 00-00 00-00 00-00 40-00 12-BB 0E-02 00-00>, where the value pair (18, 1)
// at index #2 don't match, which is -17 from 18
// If/when fixed, enable QNN EP in cpu test TensorOpTest.SpaceToDepthTest_2. Fixed by QNN 2.32.
TEST_F(QnnCPUBackendTests, SpaceToDepth_Flaky2) {
  const std::vector<float> X = {
      0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10.,
      11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
      22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
      33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43.,
      44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54.,
      55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
      66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76.,
      77., 78., 79., 80., 81., 82., 83., 84., 85., 86., 87.,
      88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98.,
      99., 100., 101., 102., 103., 104., 105., 106., 107.};

  for (size_t i = 0; i < 4; i++) {
    RunF32ModelOnCpu("SpaceToDepth",
                     {TestInputDef<float>({2, 3, 3, 6}, false, X)},
                     {utils::MakeAttribute("blocksize", static_cast<int64_t>(3))},
                     7,
                     ExpectedEPNodeAssignment::All);
  }
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ============================================================
// HTP backend tests
// ============================================================

TEST_F(QnnHTPBackendTests, DepthToSpaceOp_CRD) {
  const std::vector<float> X = {0., 1., 2., 3., 4., 5., 9., 10., 11., 12., 13., 14.,
                                 18., 19., 20., 21., 22., 23., 27., 28., 29., 30., 31., 32.};
  RunQdqModelOnHtp<uint8_t>("DepthToSpace",
                            {TestInputDef<float>({1, 4, 2, 3}, false, X)},
                            {utils::MakeAttribute("blocksize", static_cast<int64_t>(2)),
                             utils::MakeAttribute("mode", "CRD")},
                            11,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, DepthToSpaceOp_U16_CRD) {
  const std::vector<float> X = {0., 1., 2., 3., 4., 5., 9., 10., 11., 12., 13., 14.,
                                 18., 19., 20., 21., 22., 23., 27., 28., 29., 30., 31., 32.};
  RunQdqModelOnHtp<uint16_t>("DepthToSpace",
                             {TestInputDef<float>({1, 4, 2, 3}, false, X)},
                             {utils::MakeAttribute("blocksize", static_cast<int64_t>(2)),
                              utils::MakeAttribute("mode", "CRD")},
                             11,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft domain for Q/DQ ops
}

TEST_F(QnnHTPBackendTests, DepthToSpaceOp_DCR) {
  const std::vector<float> X = {0., 1., 2., 3., 4., 5., 9., 10., 11., 12., 13., 14.,
                                 18., 19., 20., 21., 22., 23., 27., 28., 29., 30., 31., 32.};
  RunQdqModelOnHtp<uint8_t>("DepthToSpace",
                            {TestInputDef<float>({1, 4, 2, 3}, false, X)},
                            {utils::MakeAttribute("blocksize", static_cast<int64_t>(2)),
                             utils::MakeAttribute("mode", "DCR")},
                            11,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, SpaceToDepthOp) {
  const std::vector<float> X = {0.0f, 0.1f, 0.2f, 0.3f,
                                 1.0f, 1.1f, 1.2f, 1.3f,
                                 2.0f, 2.1f, 2.2f, 2.3f,
                                 3.0f, 3.1f, 3.2f, 3.3f};
  RunQdqModelOnHtp<uint8_t>("SpaceToDepth",
                            {TestInputDef<float>({1, 2, 2, 4}, false, X)},
                            {utils::MakeAttribute("blocksize", static_cast<int64_t>(2))},
                            11,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, SpaceToDepthOp_U16) {
  const std::vector<float> X = {0.0f, 0.1f, 0.2f, 0.3f,
                                 1.0f, 1.1f, 1.2f, 1.3f,
                                 2.0f, 2.1f, 2.2f, 2.3f,
                                 3.0f, 3.1f, 3.2f, 3.3f};
  RunQdqModelOnHtp<uint16_t>("SpaceToDepth",
                             {TestInputDef<float>({1, 2, 2, 4}, false, X)},
                             {utils::MakeAttribute("blocksize", static_cast<int64_t>(2))},
                             11,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft domain for Q/DQ ops
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
