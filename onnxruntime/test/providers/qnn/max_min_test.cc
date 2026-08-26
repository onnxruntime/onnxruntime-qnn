// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs an Max/Min model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunCPUMinOrMaxOpTest(const std::string& op_type,
                                 const std::vector<TestInputDef<float>>& input_defs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, {}, kOnnxDomain),
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment});
}

// Runs a QDQ Max/Min model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment, and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (when compared to the baseline float32 model).
template <typename QType = uint8_t>
static void RunQDQMinOrMaxOpTest(const std::string& op_type,
                                 const std::vector<TestInputDef<float>>& input_defs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 13,
                                 bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, {}, kOnnxDomain),  // baseline float32 model
                       BuildQDQOpTestCase<QType>(op_type + "_node", op_type, input_defs, {}, {}, kOnnxDomain,
                                                 use_contrib_qdq),  // QDQ model
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Runs an int32 Max/Min model on the given QNN backend ("cpu" or "htp"). Checks the graph node
// assignment, and that inference outputs for QNN EP and ORT CPU EP match exactly (int32 has no
// quantization tolerance to account for).
static void RunInt32MinOrMaxOpTest(const std::string& op_type,
                                   const std::vector<TestInputDef<int32_t>>& input_defs,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   const std::string& backend_name,
                                   int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<int32_t>(op_type + "_node", op_type, input_defs, {}, {}, kOnnxDomain),
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment});
}

// Runs an int64 Max/Min model on the given QNN backend ("cpu" or "htp"). An int64 graph output
// exercises the needs_int64_cast path (fold in int32, trailing Cast back to int64). Checks the
// graph node assignment, and that inference outputs for QNN EP and ORT CPU EP match exactly.
static void RunInt64MinOrMaxOpTest(const std::string& op_type,
                                   const std::vector<TestInputDef<int64_t>>& input_defs,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   const std::string& backend_name,
                                   int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<int64_t>(op_type + "_node", op_type, input_defs, {}, {}, kOnnxDomain),
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment});
}

//
// CPU tests:
//

// Test that Min with 1 input is *NOT* supported on CPU backend.
TEST_F(QnnCPUBackendTests, Min_1Input_NotSupported) {
  RunCPUMinOrMaxOpTest("Min",
                       {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
                       ExpectedEPNodeAssignment::None, 13);
}

// Test that Max with 1 input is *NOT* supported on CPU backend.
TEST_F(QnnCPUBackendTests, Max_1Input_NotSupported) {
  RunCPUMinOrMaxOpTest("Max",
                       {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
                       ExpectedEPNodeAssignment::None, 13);
}

// Test Min with 2 inputs on CPU backend.
TEST_F(QnnCPUBackendTests, Min_2Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunCPUMinOrMaxOpTest("Min",
                       {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                       ExpectedEPNodeAssignment::All, 13);
}

// Test Max with 2 inputs on CPU backend.
TEST_F(QnnCPUBackendTests, Max_2Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunCPUMinOrMaxOpTest("Max",
                       {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                       ExpectedEPNodeAssignment::All, 13);
}

// Test int32 Min with 2 inputs on CPU backend.
TEST_F(QnnCPUBackendTests, Min_2Inputs_Int32) {
  std::vector<int32_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int32_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  RunInt32MinOrMaxOpTest("Min",
                         {TestInputDef<int32_t>({3, 4}, false, input0_data),
                          TestInputDef<int32_t>({3, 4}, false, input1_data)},
                         ExpectedEPNodeAssignment::All, "cpu", 13);
}

// Test int32 Max with 2 inputs on CPU backend.
TEST_F(QnnCPUBackendTests, Max_2Inputs_Int32) {
  std::vector<int32_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int32_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  RunInt32MinOrMaxOpTest("Max",
                         {TestInputDef<int32_t>({3, 4}, false, input0_data),
                          TestInputDef<int32_t>({3, 4}, false, input1_data)},
                         ExpectedEPNodeAssignment::All, "cpu", 13);
}

// Variadic (>2 input) Min/Max are decomposed into a left-folded QNN binary-op chain.

// Test float Min with 3 inputs on CPU backend.
TEST_F(QnnCPUBackendTests, Min_3Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunCPUMinOrMaxOpTest("Min",
                       {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                       ExpectedEPNodeAssignment::All, 13);
}

// Test float Max with 3 inputs on CPU backend.
TEST_F(QnnCPUBackendTests, Max_3Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunCPUMinOrMaxOpTest("Max",
                       {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                       ExpectedEPNodeAssignment::All, 13);
}

// Test float Max with 4 inputs on CPU backend (chain depth > 1).
TEST_F(QnnCPUBackendTests, Max_4Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunCPUMinOrMaxOpTest("Max",
                       {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                       ExpectedEPNodeAssignment::All, 13);
}

// Variadic float Min with mixed-rank broadcasting on CPU backend: {2,3,4},{4},{3,1} -> {2,3,4}.
TEST_F(QnnCPUBackendTests, Min_3Inputs_BroadcastMixedRank) {
  RunCPUMinOrMaxOpTest("Min",
                       {TestInputDef<float>({2, 3, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 24)),
                        TestInputDef<float>({4}, false, GetFloatDataInRange(-10.0f, 10.0f, 4)),
                        TestInputDef<float>({3, 1}, false, GetFloatDataInRange(-10.0f, 10.0f, 3))},
                       ExpectedEPNodeAssignment::All, 13);
}

// Variadic float Max with mixed-rank broadcasting on CPU backend: {2,3,4},{4},{3,1} -> {2,3,4}.
TEST_F(QnnCPUBackendTests, Max_3Inputs_BroadcastMixedRank) {
  RunCPUMinOrMaxOpTest("Max",
                       {TestInputDef<float>({2, 3, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 24)),
                        TestInputDef<float>({4}, false, GetFloatDataInRange(-10.0f, 10.0f, 4)),
                        TestInputDef<float>({3, 1}, false, GetFloatDataInRange(-10.0f, 10.0f, 3))},
                       ExpectedEPNodeAssignment::All, 13);
}

// Test int32 Max with 3 inputs on CPU backend (non-quantized integer fold branch).
TEST_F(QnnCPUBackendTests, Max_3Inputs_Int32) {
  std::vector<int32_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int32_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  std::vector<int32_t> input2_data = {0, 5, -1, 2, -6, 3, 7, -8, -2, 4, 9, -6};
  RunInt32MinOrMaxOpTest("Max",
                         {TestInputDef<int32_t>({3, 4}, false, input0_data),
                          TestInputDef<int32_t>({3, 4}, false, input1_data),
                          TestInputDef<int32_t>({3, 4}, false, input2_data)},
                         ExpectedEPNodeAssignment::All, "cpu", 13);
}

// Test int64 Min with 3 inputs on CPU backend
TEST_F(QnnCPUBackendTests, Min_3Inputs_Int64) {
  std::vector<int64_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int64_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  std::vector<int64_t> input2_data = {0, 5, -1, 2, -6, 3, 7, -8, -2, 4, 9, -6};
  RunInt64MinOrMaxOpTest("Min",
                         {TestInputDef<int64_t>({3, 4}, false, input0_data),
                          TestInputDef<int64_t>({3, 4}, false, input1_data),
                          TestInputDef<int64_t>({3, 4}, false, input2_data)},
                         ExpectedEPNodeAssignment::All, "cpu", 13);
}

// Test int64 Max with 3 inputs on CPU backend
TEST_F(QnnCPUBackendTests, Max_3Inputs_Int64) {
  std::vector<int64_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int64_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  std::vector<int64_t> input2_data = {0, 5, -1, 2, -6, 3, 7, -8, -2, 4, 9, -6};
  RunInt64MinOrMaxOpTest("Max",
                         {TestInputDef<int64_t>({3, 4}, false, input0_data),
                          TestInputDef<int64_t>({3, 4}, false, input1_data),
                          TestInputDef<int64_t>({3, 4}, false, input2_data)},
                         ExpectedEPNodeAssignment::All, "cpu", 13);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test that Min with 1 input is *NOT* supported on HTP backend.
TEST_F(QnnHTPBackendTests, Min_1Input_NotSupported) {
  RunQDQMinOrMaxOpTest("Min",
                       {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
                       ExpectedEPNodeAssignment::None, 13);
}

// Test that Max with 1 input is *NOT* supported on HTP backend.
TEST_F(QnnHTPBackendTests, Max_1Input_NotSupported) {
  RunQDQMinOrMaxOpTest("Max",
                       {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
                       ExpectedEPNodeAssignment::None, 13);
}

// Test accuracy of 8-bit Q/DQ Min with 2 inputs on HTP backend.
TEST_F(QnnHTPBackendTests, Min_2Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQMinOrMaxOpTest<uint8_t>("Min",
                                {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                 TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                                ExpectedEPNodeAssignment::All, 13);
}

// Test accuracy of 8-bit Q/DQ Max with 2 inputs on HTP backend.
TEST_F(QnnHTPBackendTests, Max_2Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQMinOrMaxOpTest<uint8_t>("Max",
                                {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                 TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                                ExpectedEPNodeAssignment::All, 13);
}

// Test int32 Min with 2 inputs on HTP backend.
TEST_F(QnnHTPBackendTests, Min_2Inputs_Int32) {
  std::vector<int32_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int32_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  RunInt32MinOrMaxOpTest("Min",
                         {TestInputDef<int32_t>({3, 4}, false, input0_data),
                          TestInputDef<int32_t>({3, 4}, false, input1_data)},
                         ExpectedEPNodeAssignment::All, "htp", 13);
}

// Test int32 Max with 2 inputs on HTP backend.
TEST_F(QnnHTPBackendTests, Max_2Inputs_Int32) {
  std::vector<int32_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int32_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  RunInt32MinOrMaxOpTest("Max",
                         {TestInputDef<int32_t>({3, 4}, false, input0_data),
                          TestInputDef<int32_t>({3, 4}, false, input1_data)},
                         ExpectedEPNodeAssignment::All, "htp", 13);
}

// NEGATIVE: int32 Min with 1 input is still *NOT* supported on HTP backend (ExplicitOpCheck's
// 2-input restriction is dtype-agnostic; unchanged by int32 enablement).
TEST_F(QnnHTPBackendTests, Min_1Input_NotSupported_Int32) {
  RunInt32MinOrMaxOpTest("Min",
                         {TestInputDef<int32_t>({1, 3, 4, 4}, false, std::vector<int32_t>(48, 1))},
                         ExpectedEPNodeAssignment::None, "htp", 13);
}

// NEGATIVE: int32 Max with 1 input is still *NOT* supported on HTP backend (ExplicitOpCheck's
// 2-input restriction is dtype-agnostic; unchanged by int32 enablement).
TEST_F(QnnHTPBackendTests, Max_1Input_NotSupported_Int32) {
  RunInt32MinOrMaxOpTest("Max",
                         {TestInputDef<int32_t>({1, 3, 4, 4}, false, std::vector<int32_t>(48, 1))},
                         ExpectedEPNodeAssignment::None, "htp", 13);
}

// Variadic (>2 input) Min/Max on HTP: DQ -> float chain -> Q.

// Test accuracy of 8-bit Q/DQ Min with 3 inputs on HTP backend.
TEST_F(QnnHTPBackendTests, Min_3Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQMinOrMaxOpTest<uint8_t>("Min",
                                {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                 TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                 TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                                ExpectedEPNodeAssignment::All, 13);
}

// Variadic 8-bit Q/DQ Min with mixed-rank broadcasting on HTP: {2,3,4},{4},{3,1} -> {2,3,4}.
TEST_F(QnnHTPBackendTests, Min_3Inputs_BroadcastMixedRank_U8) {
  RunQDQMinOrMaxOpTest<uint8_t>("Min",
                                {TestInputDef<float>({2, 3, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 24)),
                                 TestInputDef<float>({4}, false, GetFloatDataInRange(-10.0f, 10.0f, 4)),
                                 TestInputDef<float>({3, 1}, false, GetFloatDataInRange(-10.0f, 10.0f, 3))},
                                ExpectedEPNodeAssignment::All, 13);
}

// Variadic 8-bit Q/DQ Max with mixed-rank broadcasting on HTP: {2,3,4},{4},{3,1} -> {2,3,4}.
TEST_F(QnnHTPBackendTests, Max_3Inputs_BroadcastMixedRank_U8) {
  RunQDQMinOrMaxOpTest<uint8_t>("Max",
                                {TestInputDef<float>({2, 3, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 24)),
                                 TestInputDef<float>({4}, false, GetFloatDataInRange(-10.0f, 10.0f, 4)),
                                 TestInputDef<float>({3, 1}, false, GetFloatDataInRange(-10.0f, 10.0f, 3))},
                                ExpectedEPNodeAssignment::All, 13);
}

// Test accuracy of 8-bit Q/DQ Max with 3 inputs on HTP backend.
TEST_F(QnnHTPBackendTests, Max_3Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQMinOrMaxOpTest<uint8_t>("Max",
                                {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                 TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                 TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                                ExpectedEPNodeAssignment::All, 13);
}

// Test accuracy of 16-bit Q/DQ Min with 3 inputs on HTP backend (16-bit requant path).
TEST_F(QnnHTPBackendTests, Min_3Inputs_U16) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQMinOrMaxOpTest<uint16_t>("Min",
                                 {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                  TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                  TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                                 ExpectedEPNodeAssignment::All, 13,
                                 /*use_contrib_qdq=*/true);  // 16-bit zero-point needs com.microsoft Q/DQ
}

// Test accuracy of 16-bit Q/DQ Max with 3 inputs on HTP backend (16-bit requant path).
TEST_F(QnnHTPBackendTests, Max_3Inputs_U16) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQMinOrMaxOpTest<uint16_t>("Max",
                                 {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                  TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                  TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                                 ExpectedEPNodeAssignment::All, 13,
                                 /*use_contrib_qdq=*/true);  // 16-bit zero-point needs com.microsoft Q/DQ
}

// Test int32 Min with 3 inputs on HTP backend (non-quantized integer fold branch).
TEST_F(QnnHTPBackendTests, Min_3Inputs_Int32) {
  std::vector<int32_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int32_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  std::vector<int32_t> input2_data = {0, 5, -1, 2, -6, 3, 7, -8, -2, 4, 9, -6};

  RunInt32MinOrMaxOpTest("Min",
                         {TestInputDef<int32_t>({3, 4}, false, input0_data),
                          TestInputDef<int32_t>({3, 4}, false, input1_data),
                          TestInputDef<int32_t>({3, 4}, false, input2_data)},
                         ExpectedEPNodeAssignment::All, "htp", 13);
}

// Test int32 Max with 3 inputs on HTP backend (non-quantized integer fold branch).
TEST_F(QnnHTPBackendTests, Max_3Inputs_Int32) {
  std::vector<int32_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int32_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  std::vector<int32_t> input2_data = {0, 5, -1, 2, -6, 3, 7, -8, -2, 4, 9, -6};
  RunInt32MinOrMaxOpTest("Max",
                         {TestInputDef<int32_t>({3, 4}, false, input0_data),
                          TestInputDef<int32_t>({3, 4}, false, input1_data),
                          TestInputDef<int32_t>({3, 4}, false, input2_data)},
                         ExpectedEPNodeAssignment::All, "htp", 13);
}

// Test int64 Min with 3 inputs on HTP backend (needs_int64_cast branch: int32 fold + Cast to int64).
TEST_F(QnnHTPBackendTests, Min_3Inputs_Int64) {
  std::vector<int64_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int64_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  std::vector<int64_t> input2_data = {0, 5, -1, 2, -6, 3, 7, -8, -2, 4, 9, -6};
  RunInt64MinOrMaxOpTest("Min",
                         {TestInputDef<int64_t>({3, 4}, false, input0_data),
                          TestInputDef<int64_t>({3, 4}, false, input1_data),
                          TestInputDef<int64_t>({3, 4}, false, input2_data)},
                         ExpectedEPNodeAssignment::All, "htp", 13);
}

// Test int64 Max with 3 inputs on HTP backend (needs_int64_cast branch: int32 fold + Cast to int64).
TEST_F(QnnHTPBackendTests, Max_3Inputs_Int64) {
  std::vector<int64_t> input0_data = {-10, -5, 0, 5, 10, -3, 7, -8, 2, -1, 9, -6};
  std::vector<int64_t> input1_data = {3, -5, 1, -2, 6, -3, -7, 8, 2, 4, -9, 6};
  std::vector<int64_t> input2_data = {0, 5, -1, 2, -6, 3, 7, -8, -2, 4, 9, -6};
  RunInt64MinOrMaxOpTest("Max",
                         {TestInputDef<int64_t>({3, 4}, false, input0_data),
                          TestInputDef<int64_t>({3, 4}, false, input1_data),
                          TestInputDef<int64_t>({3, 4}, false, input2_data)},
                         ExpectedEPNodeAssignment::All, "htp", 13);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
