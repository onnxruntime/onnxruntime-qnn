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

TEST_F(QnnHTPBackendTests, BinaryOp_Add4D) {
  RunQdqModelOnHtp<uint8_t>("Add",
                            {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                             TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f)},
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Add4D_U16) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunQdqModelOnHtp<uint16_t>("Add",
                             {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                              TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                             {},
                             17,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft Q/DQ ops
}

TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D) {
  RunQdqModelOnHtp<uint8_t>("Sub",
                            {TestInputDef<float>({1, 3, 8, 8}, false, -10.0f, 10.0f),
                             TestInputDef<float>({1, 3, 8, 8}, false, -10.0f, 10.0f)},
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D_U16) {
  std::vector<float> input0_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  std::vector<float> input1_data = GetFloatDataInRange(0.0f, 20.0f, 8);
  RunQdqModelOnHtp<uint16_t>("Sub",
                             {TestInputDef<float>({1, 2, 2, 2}, false, input0_data),
                              TestInputDef<float>({1, 2, 2, 2}, false, input1_data)},
                             {},
                             17,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft Q/DQ ops
}

TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D_LargeInputs) {
  RunQdqModelOnHtp<uint8_t>("Sub",
                            {TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                             TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f)},
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D_Broadcast) {
  RunQdqModelOnHtp<uint8_t>("Sub",
                            {TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                             TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f})},
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

// TODO: This fails on Linux (HTP emulation). Works on Windows ARM64.
// Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.051073111593723297, zero_point=2.
// Expected val: 0.0099999997764825821
// QNN QDQ val: 12.921497344970703 (err 12.911497116088867)
// CPU QDQ val: -0.10214622318744659 (err 0.11214622110128403)
#if defined(__linux__)
TEST_F(QnnHTPBackendTests, DISABLED_BinaryOp_Pow) {
#else
TEST_F(QnnHTPBackendTests, BinaryOp_Pow) {
#endif
  std::vector<float> bases_input = {-10.0f, -8.0f, -6.0f, 1.0f, 2.0f, 3.0f, 5.5f, 10.0f};
  std::vector<float> exponents_input = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 1.5f, 0.2f};
  RunQdqModelOnHtp<uint8_t>("Pow",
                            {TestInputDef<float>({1, 2, 2, 2}, false, bases_input),
                             TestInputDef<float>({1, 2, 2, 2}, false, exponents_input)},
                            {},
                            15,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_PRelu_DynamicSlopes) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  std::vector<float> slopes_data = GetFloatDataInRange(-1.0f, 1.0f, 8);
  RunQdqModelOnHtp<uint8_t>("PRelu",
                            {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                             TestInputDef<float>({1, 2, 2, 2}, false, slopes_data)},
                            {},
                            16,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_PRelu_StaticSlopes) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  std::vector<float> slopes_data = GetFloatDataInRange(-1.0f, 1.0f, 8);
  RunQdqModelOnHtp<uint8_t>("PRelu",
                            {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                             TestInputDef<float>({1, 2, 2, 2}, true, slopes_data)},
                            {},
                            16,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Div4D_SmallInputs) {
  std::vector<float> input0_data = {-10.0f, -8.0f, -1.0f, 0.0f, 1.0f, 2.1f, 8.0f, 10.0f};
  std::vector<float> input1_data = {5.0f, 4.0f, 1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 5.0f};
  RunQdqModelOnHtp<uint8_t>("Div",
                            {TestInputDef<float>({1, 2, 2, 2}, false, input0_data),
                             TestInputDef<float>({1, 2, 2, 2}, false, input1_data)},
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Div4D_U16_SmallInputs) {
  std::vector<float> input0_data = {-10.0f, -8.0f, -1.0f, 0.0f, 1.0f, 2.1f, 8.0f, 10.0f};
  std::vector<float> input1_data = {5.0f, 4.0f, 1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 5.0f};
  RunQdqModelOnHtp<uint16_t>("Div",
                             {TestInputDef<float>({1, 2, 2, 2}, false, input0_data),
                              TestInputDef<float>({1, 2, 2, 2}, false, input1_data)},
                             {},
                             17,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft Q/DQ ops
}

// TODO: Enable when this is fixed.
// QNN v2.13: Inaccuracy detected for output 'output', element 2551923.
// Output quant params: scale=4100.92626953125, zero_point=126.
// Expected val: -277957.3125
// QNN QDQ val: 0 (err 277957.3125)
// CPU QDQ val: -516716.71875 (err 238759.40625)
TEST_F(QnnHTPBackendTests, DISABLED_BinaryOp_Div4D_LargeInputs) {
  RunQdqModelOnHtp<uint8_t>("Div",
                            {TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                             TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f)},
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Div4D_Broadcast) {
  RunQdqModelOnHtp<uint8_t>("Div",
                            {TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                             TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f})},
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Mul4D) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0, 10.0f, 8);
  RunQdqModelOnHtp<uint8_t>("Mul",
                            {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                             TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Mul4D_U16) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunQdqModelOnHtp<uint16_t>("Mul",
                             {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                              TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                             {},
                             17,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft Q/DQ ops
}

TEST_F(QnnHTPBackendTests, BinaryOp_And4D) {
  RunF32ModelOnHtp<bool>("And",
                         {TestInputDef<bool>({1, 4}, false, {false, false, true, true}),
                          TestInputDef<bool>({1, 4}, false, {false, true, false, true})},
                         {},
                         17,
                         ExpectedEPNodeAssignment::All);
}

// Or is not supported on the HTP backend.
TEST_F(QnnHTPBackendTests, BinaryOp_HTP_Or_Unsupported) {
  RunF32ModelOnHtp<bool>("Or",
                         {TestInputDef<bool>({1, 4}, false, {false, false, true, true}),
                          TestInputDef<bool>({1, 4}, false, {false, true, false, true})},
                         {},
                         17,
                         ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
