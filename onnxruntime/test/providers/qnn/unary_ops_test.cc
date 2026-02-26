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

// Disabled because QNN SDK 2.17 Relu treats inf as FLT_MAX.
// TODO: When this is fixed, enable ActivationOpTest.Relu test in cpu/activation/activation_op_test tests.
// Log: the value pair (inf, 3.40282347e+38) at index #12 don't match
TEST_F(QnnCPUBackendTests, DISABLED_UnaryOp_Relu) {
  std::vector<float> input_data{-1.0f, 0, 1.0f,
                                100.0f, -100.0f, 1000.0f, -1000.0f,
                                FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,
                                FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity()};
  RunF32ModelOnCpu("Relu",
                   {TestInputDef<float>({13}, false, input_data)},
                   {},
                   14,
                   ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ============================================================
// HTP backend tests
// ============================================================

TEST_F(QnnHTPBackendTests, UnaryOp_Sigmoid) {
  RunQdqModelOnHtp<uint8_t>("Sigmoid",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Sigmoid_U16) {
  RunQdqModelOnHtp<uint16_t>("Sigmoid",
                             {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                             {},
                             13,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use MS domain Q/DQ ops
}

TEST_F(QnnHTPBackendTests, UnaryOp_Tanh) {
  RunQdqModelOnHtp<uint8_t>("Tanh",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

// disabled for QNN 2.28.0.241029 backendValidateOpConfig failed
// still fails on QNN 2.28.2 and QNN 2.30.0
// QnnDsp <E> [4294967295] has incorrect Value -32768, expected equal to 0.
// QnnDsp <V> validateNativeOps node_token_6:qti.aisw:Tanh htp op validator failed 3110
// QnnDsp <V> registered validator failed => 3110
// QnnDsp <E> QnnBackend_validateOpConfig failed 3110
// QnnDsp <V> Wake up free backend (id: 1)'s thread(s)
// QnnDsp <E> Failed to validate op node_token_6 with error 0xc26
// We now skip QNN validation as a workaround for QNN SDK 2.28.0 to 2.30.0
TEST_F(QnnHTPBackendTests, UnaryOp_Tanh_U16) {
  RunQdqModelOnHtp<uint16_t>("Tanh",
                             {TestInputDef<float>({1, 2, 64}, false, GetFloatDataInRange(-10.0f, 10.0f, 128))},
                             {},
                             13,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use MS domain Q/DQ ops
}

// Check that QNN compiles DQ -> Gelu -> Q as a single unit. Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Gelu) {
  RunQdqModelOnHtp<uint8_t>("Gelu",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {},
                            11,
                            ExpectedEPNodeAssignment::All,
                            kMSDomain);  // GeLu is a contrib op.
}

// TODO(adrianlizarraga): Inaccuracy detected for output 'output', element 5.
// Output quant params: scale=0.00015259021893143654, zero_point=0.
// Expected val: 10
// QNN QDQ val: 9.997406005859375 (err 0.002593994140625)
// CPU QDQ val: 9.999847412109375 (err 0.000152587890625)
TEST_F(QnnHTPBackendTests, UnaryOp_Gelu_U16) {
  const std::vector<float> input_data = {-10.0f, -8.4f, 0.0f, 4.3f, 7.1f, 10.0f};
  RunQdqModelOnHtp<uint16_t>("Gelu",
                             {TestInputDef<float>({1, 2, 3}, false, input_data)},
                             {},
                             11,
                             ExpectedEPNodeAssignment::All,
                             kMSDomain,  // GeLu is a contrib op.
                             true);      // Use MS domain Q/DQ ops.
}

// Check that QNN compiles DQ -> Elu -> Q as a single unit. Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Elu) {
  RunQdqModelOnHtp<uint8_t>("Elu",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {},
                            11,
                            ExpectedEPNodeAssignment::All);
}

// TODO(adrianlizarraga): Re-enable. This works on QNN SDK 2.14.1! Issue fixed in 2.30.
// Inaccuracy detected for output 'output', element 1.
// Output quant params: scale=0.00011093531065853313, zero_point=8992.
// Expected val: -0.99751651287078857
// QNN QDQ val: 6.2726154327392578 (err 7.2701320648193359)
// CPU QDQ val: -0.99753034114837646 (err 1.3828277587890625e-05)
TEST_F(QnnHTPBackendTests, UnaryOp_Elu_U16) {
  RunQdqModelOnHtp<uint16_t>("Elu",
                             {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                             {},
                             11,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);
}

// Tests accuracy of QDQ Relu.
// TODO: Relu does not set negative values to zero!
// Could be due to ORT's ReluQuantFusion!
// Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.039215687662363052, zero_point=0.
// Expected val: 0
// QNN QDQ val: -10 (err 10)
// CPU QDQ val: 0 (err 0)
TEST_F(QnnHTPBackendTests, UnaryOp_Relu) {
  RunQdqModelOnHtp<uint8_t>("Relu",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {},
                            14,
                            ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> HardSwish -> Q as a single unit. Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSwish) {
  RunQdqModelOnHtp<uint8_t>("HardSwish",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {},
                            14,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_HardSwish_U16) {
  const std::vector<float> input_data = {-10.0f, -8.4f, 0.0f, 4.3f, 7.1f, 10.0f};
  RunQdqModelOnHtp<uint16_t>("HardSwish",
                             {TestInputDef<float>({1, 2, 3}, false, input_data)},
                             {},
                             14,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);
}

// Check that QNN compiles DQ -> Atan -> Q as a single unit. Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Atan) {
  RunQdqModelOnHtp<uint8_t>("Atan",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {},
                            14,
                            ExpectedEPNodeAssignment::All);
}

// TODO(adrianlizarraga): Inaccuracy detected for output 'output', element 1.
// Output quant params: scale=4.4895936298416927e-05, zero_point=32768.
// Expected val: -1.4219063520431519
// QNN QDQ val: -1.4220787286758423 (err 0.00017237663269042969)
// CPU QDQ val: -1.4218991994857788 (err 7.152557373046875e-06)
TEST_F(QnnHTPBackendTests, UnaryOp_Atan_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQdqModelOnHtp<uint16_t>("Atan",
                             {TestInputDef<float>({1, 2, 3}, false, input_data)},
                             {},
                             14,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Q/DQ op domain is com.microsoft
}

// Check that QNN compiles DQ -> Asin -> Q as a single unit. Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Asin) {
  RunQdqModelOnHtp<uint8_t>("Asin",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-0.5, 0.5, 6))},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Sign -> Q as a single unit. Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Sign) {
  RunQdqModelOnHtp<uint8_t>("Sign",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Sign_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQdqModelOnHtp<uint16_t>("Sign",
                             {TestInputDef<float>({1, 2, 3}, false, input_data)},
                             {},
                             13,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft Q/DQ op domains
}

// Check that QNN compiles DQ -> Sin -> Q as a single unit. Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Sin) {
  RunQdqModelOnHtp<uint8_t>("Sin",
                            {TestInputDef<float>({1, 2, 3}, false, -3.14159f, 3.14159f)},
                            {},
                            11,
                            ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Cos -> Q as a single unit. Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Cos) {
  RunQdqModelOnHtp<uint8_t>("Cos",
                            {TestInputDef<float>({1, 2, 3}, false, {-3.14159f, -1.5f, -0.5f, 0.0f, 1.5, 3.14159f})},
                            {},
                            11,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Cos_InaccurateFixed) {
  RunQdqModelOnHtp<uint8_t>(
      "Cos",
      {TestInputDef<float>({1, 2, 3}, false,
                           {-3.14159f, -1.88436f, -0.542863f, 0.0f, 1.05622f, 3.14159f})},
      {},
      11,
      ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Log -> Q as a single unit. Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Log) {
  RunQdqModelOnHtp<uint8_t>(
      "Log",
      {TestInputDef<float>({1, 2, 3}, false,
                           {3.14159f, 100.88436f, 10.542863f, 9.1f, 1.05622f, 3.14159f})},
      {},
      11,
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Log_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(1.0f, 128.0f, 6);
  RunQdqModelOnHtp<uint16_t>("Log",
                             {TestInputDef<float>({1, 2, 3}, false, input_data)},
                             {},
                             11,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft domain for Q/DQ ops
}

TEST_F(QnnHTPBackendTests, UnaryOp_Exp) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQdqModelOnHtp<uint8_t>("Exp",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Sqrt) {
  std::vector<float> input_data = GetFloatDataInRange(0.0f, 20.0f, 9);
  RunQdqModelOnHtp<uint8_t>("Sqrt",
                            {TestInputDef<float>({1, 3, 3}, false, input_data)},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Neg) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQdqModelOnHtp<uint8_t>("Neg",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Not) {
  RunF32ModelOnHtp<bool>("Not",
                         {TestInputDef<bool>({1, 4}, false, {false, false, true, true})},
                         {},
                         17,
                         ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Round) {
  std::vector<float> input_data = GetFloatDataInRange(-9.0f, 9.0f, 6);
  RunQdqModelOnHtp<uint8_t>("Round",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {},
                            11,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Abs) {
  RunQdqModelOnHtp<uint8_t>("Abs",
                            {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Abs_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQdqModelOnHtp<uint16_t>("Abs",
                             {TestInputDef<float>({1, 2, 3}, false, input_data)},
                             {},
                             13,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft domain for Q/DQ ops
}

// Broken on v79 and v81 devices:
// Inaccuracy detected for output 'output_0', element 0
// output_range=24, tolerance=0.40000000596046448%.
// Expected val (f32@CPU_EP): -12
// qdq@QNN_EP val: -11.011764526367188 (err: 0.9882354736328125, err/output_range: 4.1176481246948242%)
// qdq@CPU_EP val: -12.047059059143066 (err: 0.047059059143066406, err/output_range: 0.19607941806316376%)
// abs(qdq@QNN_EP - qdq@CPU_EP) / output_range = 3.9215683937072754%
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
TEST_F(QnnHTPBackendTests, DISABLED_UnaryOp_Ceil) {
#else
TEST_F(QnnHTPBackendTests, UnaryOp_Ceil) {
#endif
  const std::vector<float> input_data = GetFloatDataInRange(-12.0f, 12.0f, 6);
  RunQdqModelOnHtp<uint8_t>("Ceil",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

// TODO(adrianlizarraga): Fails in QNN SDK 2.21 (linux). On Windows ARM64, fails in QNN SDK 2.19.
// Issue fixed in 2.30.
TEST_F(QnnHTPBackendTests, UnaryOp_Ceil_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(-12.0f, 12.0f, 6);
  RunQdqModelOnHtp<uint16_t>("Ceil",
                             {TestInputDef<float>({1, 2, 3}, false, input_data)},
                             {},
                             13,
                             ExpectedEPNodeAssignment::All,
                             kOnnxDomain,
                             true);  // Use com.microsoft domain for Q/DQ ops
}

TEST_F(QnnHTPBackendTests, UnaryOp_Floor) {
  const std::vector<float> input_data = GetFloatDataInRange(-12.0f, 12.0f, 6);
  RunQdqModelOnHtp<uint8_t>("Floor",
                            {TestInputDef<float>({1, 2, 3}, false, input_data)},
                            {},
                            13,
                            ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
