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

TEST_F(QnnHTPBackendTests, GridSample_Bilinear) {
  RunQdqModelOnHtp<uint8_t>("GridSample",
                            {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                             TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                            {utils::MakeAttribute("align_corners", static_cast<int64_t>(0)),
                             utils::MakeAttribute("mode", "bilinear"),
                             utils::MakeAttribute("padding_mode", "zeros")},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GridSample_U16_Bilinear) {
  RunQdqModelOnHtp<uint16_t>("GridSample",
                             {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                              TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                             {utils::MakeAttribute("align_corners", static_cast<int64_t>(0)),
                              utils::MakeAttribute("mode", "bilinear"),
                              utils::MakeAttribute("padding_mode", "zeros")},
                             17,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft Q/DQ ops
}

TEST_F(QnnHTPBackendTests, GridSample_AlignCorners) {
  RunQdqModelOnHtp<uint8_t>("GridSample",
                            {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                             TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                            {utils::MakeAttribute("align_corners", static_cast<int64_t>(1)),
                             utils::MakeAttribute("mode", "bilinear"),
                             utils::MakeAttribute("padding_mode", "zeros")},
                            17,
                            ExpectedEPNodeAssignment::All,
                            kOnnxDomain,
                            false,
                            QDQTolerance(0.008f));
}

TEST_F(QnnHTPBackendTests, GridSample_U16_AlignCorners) {
  RunQdqModelOnHtp<uint16_t>("GridSample",
                             {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                              TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                             {utils::MakeAttribute("align_corners", static_cast<int64_t>(1)),
                              utils::MakeAttribute("mode", "bilinear"),
                              utils::MakeAttribute("padding_mode", "zeros")},
                             17,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft Q/DQ ops
}

// Issue fixed in 2.30.
TEST_F(QnnHTPBackendTests, GridSample_BorderPadding) {
  RunQdqModelOnHtp<uint8_t>("GridSample",
                            {TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                             TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f)},
                            {utils::MakeAttribute("mode", "bilinear"),
                             utils::MakeAttribute("padding_mode", "border")},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GridSample_Nearest) {
  RunQdqModelOnHtp<uint8_t>("GridSample",
                            {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                             TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                            {utils::MakeAttribute("mode", "nearest")},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GridSample_U16_Nearest) {
  RunQdqModelOnHtp<uint16_t>("GridSample",
                             {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                              TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                             {utils::MakeAttribute("mode", "nearest")},
                             17,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);
}

TEST_F(QnnHTPBackendTests, GridSample_Linear_ZerosPadding) {
  RunQdqModelOnHtp<uint8_t>(
      "GridSample",
      {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
       TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))},
      {utils::MakeAttribute("mode", "linear"), utils::MakeAttribute("padding_mode", "zeros")},
      20,
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GridSample_Linear_AlignCorners_BorderPadding) {
  RunQdqModelOnHtp<uint8_t>(
      "GridSample",
      {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
       TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))},
      {utils::MakeAttribute("align_corners", static_cast<int64_t>(1)),
       utils::MakeAttribute("mode", "linear"),
       utils::MakeAttribute("padding_mode", "border")},
      20,
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GridSample_Linear_ReflectionPadding_U16) {
  RunQdqModelOnHtp<uint16_t>(
      "GridSample",
      {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
       TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))},
      {utils::MakeAttribute("mode", "linear"), utils::MakeAttribute("padding_mode", "reflection")},
      21,
      ExpectedEPNodeAssignment::All,
      kOnnxDomain,
      true);
}

// Fixed by QNN 2.32.
TEST_F(QnnHTPBackendTests, GridSample_ReflectionPaddingMode) {
  RunQdqModelOnHtp<uint8_t>("GridSample",
                            {TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                             TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f)},
                            {utils::MakeAttribute("padding_mode", "reflection")},
                            17,
                            ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
