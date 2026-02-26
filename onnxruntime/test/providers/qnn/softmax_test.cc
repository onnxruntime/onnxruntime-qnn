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

// ============================================================
// Softmax tests
// ============================================================

// Test that the default axis (-1) for SoftMax opset 13 works.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_DefaultAxis) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQdqModelOnHtp<uint8_t>("Softmax",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {},  // Uses default axis of -1 for opset 13
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_U16_DefaultAxis) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQdqModelOnHtp<uint16_t>("Softmax",
                             {TestInputDef<float>({1, 2, 3}, false, input_data)},
                             {},  // Uses default axis of -1 for opset 13
                             13,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft domain for Q/DQ ops
}

// QNN EP will wrap the operator with transposes.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_NonLastAxis) {
  const std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f, 100.0f, 110.0f, 120.0f,
                                         1.0856307f, 0.99734545f, 0.2829785f, 1.5062947f, 0.5786002f, 1.6514366f,
                                         2.4266791f, 0.42891264f, 1.2659363f};
  RunQdqModelOnHtp<uint8_t>("Softmax",
                            {TestInputDef<float>({1, 2, 3, 3}, false, input_data)},
                            {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                            13,
                            ExpectedEPNodeAssignment::All);
}

// This is a configuration used in one of our partner's models.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_NonLastAxis_LargeInput) {
  const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
  RunQdqModelOnHtp<uint8_t>("Softmax",
                            {TestInputDef<float>({1, 124, 1}, false, input_data)},
                            {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_U16_NonLastAxis_LargeInput) {
  const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
  RunQdqModelOnHtp<uint16_t>("Softmax",
                             {TestInputDef<float>({1, 124, 1}, false, input_data)},
                             {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                             13,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);
}

// Test that the default axis (1) for SoftMax opset < 13 works.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax11_DefaultAxis) {
  RunQdqModelOnHtp<uint8_t>("Softmax",
                            {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
                            {},  // Uses default axis of 1 for opset < 13.
                            11,
                            ExpectedEPNodeAssignment::All);
}

// Test that setting an axis value of -1 works for Softmax opset < 13.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax11_SetAxis) {
  RunQdqModelOnHtp<uint8_t>("Softmax",
                            {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
                            {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                            11,
                            ExpectedEPNodeAssignment::All);
}

// ============================================================
// LogSoftmax tests
// ============================================================

// Test that the default axis (-1) for LogSoftmax opset 13 works.
TEST_F(QnnHTPBackendTests, UnaryOp_LogSoftmax13_DefaultAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQdqModelOnHtp<uint8_t>("LogSoftmax",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {},  // Uses default axis of -1 for opset 13
                            13,
                            ExpectedEPNodeAssignment::All);
}

// QNN EP will wrap the operator with transposes.
TEST_F(QnnHTPBackendTests, UnaryOp_LogSoftmax13_NonLastAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQdqModelOnHtp<uint8_t>("LogSoftmax",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                            13,
                            ExpectedEPNodeAssignment::All);
}

// Test that the default axis (1) for LogSoftmax opset < 13 works.
TEST_F(QnnHTPBackendTests, UnaryOp_LogSoftmax11_DefaultAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQdqModelOnHtp<uint8_t>("LogSoftmax",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {},  // Uses default axis of 1 for opset < 13.
                            11,
                            ExpectedEPNodeAssignment::All);
}

// Test that setting an axis value of -1 works for LogSoftmax opset < 13.
TEST_F(QnnHTPBackendTests, UnaryOp_LogSoftmax11_SetAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQdqModelOnHtp<uint8_t>("LogSoftmax",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                            11,
                            ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
