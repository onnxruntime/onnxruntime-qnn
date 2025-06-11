// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <filesystem>
#include <variant>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a non-QDQ model on the QNN CPU backend and compares output to CPU EP.
template <typename InputType = float>
static void RunOpTestOnCPU(const std::string& op_type,
                           const std::vector<TestInputDef<InputType>>& input_defs,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           const std::string& op_packages,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           const std::string& op_domain = kOnnxDomain) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["op_packages"] = op_packages;

  RunQnnModelTest(BuildOpTestCase<InputType>(op_type, input_defs, {}, attrs, op_domain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  1e-5f,
                  logging::Severity::kVERBOSE);
}

#if defined(__linux__)

TEST_F(QnnCPUBackendTests, CustomOp_Softmax11_Last_Axis) {
  auto test_input = TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f);
  RunOpTestOnCPU("Softmax",
                 {test_input},
                 {utils::MakeAttribute("axis", static_cast<int64_t>(2))},
                 "Softmax:./qnn_custom_op-build/x86_64-linux-clang/libSoftmaxOpPackage.so:SoftmaxOpPackageInterfaceProvider",
                 11,
                 ExpectedEPNodeAssignment::All);
}
#endif  // defined(__linux__)

#if defined(_WIN32)

TEST_F(QnnCPUBackendTests, CustomOp_Softmax11_Last_Axis) {
  auto test_input = TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f);
  RunOpTestOnCPU("Softmax",
                 {test_input},
                 {utils::MakeAttribute("axis", static_cast<int64_t>(2))},
                 "Softmax:.\\qnn_custom_op-build\\SoftmaxOpPackage.dll:SoftmaxOpPackageInterfaceProvider",
                 11,
                 ExpectedEPNodeAssignment::All);
}
#endif  // defined(__linux__)

#if defined(__aarch64__) || defined(_M_ARM64)

TEST_F(QnnCPUBackendTests, CustomOp_Softmax11_Last_Axis) {
  auto test_input = TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f);
  RunOpTestOnCPU("Softmax",
                 {test_input},
                 {utils::MakeAttribute("axis", static_cast<int64_t>(2))},
                 "Softmax:./qnn_custom_op-build/aarch64-android/libSoftmaxOpPackage.so:SoftmaxOpPackageInterfaceProvider",
                 11,
                 ExpectedEPNodeAssignment::All);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64)

// // Tests the accuracy of a QDQ model on QNN EP by comparing to CPU EP, which runs both the fp32 model
// // and the QDQ model.
// template <typename InputQType = uint8_t>
// static void RunQDQOpTest(const std::string& op_type,
//                          const std::vector<TestInputDef<float>>& input_defs,
//                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
//                          int opset_version,
//                          ExpectedEPNodeAssignment expected_ep_assignment,
//                          const std::string& op_domain = kOnnxDomain,
//                          bool use_contrib_qdq = false,
//                          QDQTolerance tolerance = QDQTolerance()) {
//   ProviderOptions provider_options;
//   provider_options["backend_type"] = "htp";
//   provider_options["offload_graph_io_quantization"] = "0";

//   TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, input_defs, {}, attrs, op_domain),
//                        BuildQDQOpTestCase<InputQType>(op_type, input_defs, {}, attrs, op_domain, use_contrib_qdq),
//                        provider_options,
//                        opset_version,
//                        expected_ep_assignment,
//                        tolerance);
// }

// // Runs a non-QDQ model on HTP and compares output to CPU EP.
// template <typename InputType = float>
// static void RunOpTest(const std::string& op_type,
//                       const std::vector<TestInputDef<InputType>>& input_defs,
//                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
//                       int opset_version,
//                       ExpectedEPNodeAssignment expected_ep_assignment,
//                       const std::string& op_domain = kOnnxDomain,
//                       float fp32_abs_err = 1e-5f,
//                       bool enable_htp_fp16_precision = false) {
//   ProviderOptions provider_options;
//   provider_options["backend_type"] = "htp";

//   if (enable_htp_fp16_precision) {
//     provider_options["enable_htp_fp16_precision"] = "1";
//   }

//   // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
//   RunQnnModelTest(BuildOpTestCase<InputType>(op_type, input_defs, {}, attrs, op_domain),
//                   provider_options,
//                   opset_version,
//                   expected_ep_assignment,
//                   fp32_abs_err);
// }

// // Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// // Test that the default axis (-1) for SoftMax opset 13 works.
// TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_DefaultAxis) {
//   const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
//   RunQDQOpTest<uint8_t>("Softmax",
//                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
//                         {},  // Uses default axis of -1 for opset 13
//                         13,
//                         ExpectedEPNodeAssignment::All);
// }

// // Tests accuracy of 16-bit QDQ Softmax (opset 13) with default axis
// TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_U16_DefaultAxis) {
//   const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
//   RunQDQOpTest<uint16_t>("Softmax",
//                          {TestInputDef<float>({1, 2, 3}, false, input_data)},
//                          {},  // Uses default axis of -1 for opset 13
//                          13,
//                          ExpectedEPNodeAssignment::All,
//                          kOnnxDomain,  // Sofmax's domain
//                          true);        // Use com.microsoft domain for Q/DQ ops
// }

// // Test that 8-bit QDQ Softmax (opset 13) with axis != -1 is supported by QNN EP.
// // QNN EP will wrap the operator with transposes.
// TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_NonLastAxis) {
//   const std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f, 100.0f, 110.0f, 120.0f,
//                                          1.0856307f, 0.99734545f, 0.2829785f, 1.5062947f, 0.5786002f, 1.6514366f,
//                                          2.4266791f, 0.42891264f, 1.2659363f};
//   RunQDQOpTest<uint8_t>("Softmax",
//                         {TestInputDef<float>({1, 2, 3, 3}, false, input_data)},
//                         {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
//                         13,
//                         ExpectedEPNodeAssignment::All);
// }

// // Test that 8-bit QDQ Softmax (opset 13) with axis != -1 is supported by QNN EP.
// // QNN EP will wrap the operator with transposes.
// // This is a configuration used in one of our partner's models.
// TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_NonLastAxis_LargeInput) {
//   const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
//   RunQDQOpTest<uint8_t>("Softmax",
//                         {TestInputDef<float>({1, 124, 1}, false, input_data)},
//                         {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
//                         13,
//                         ExpectedEPNodeAssignment::All);
// }

// // Test that 16-bit QDQ Softmax (opset 13) with axis != -1 is supported by QNN EP.
// // QNN EP will wrap the operator with transposes.
// // This is a configuration used in one of our partner's models.
// TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_U16_NonLastAxis_LargeInput) {
//   const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
//   RunQDQOpTest<uint16_t>("Softmax",
//                          {TestInputDef<float>({1, 124, 1}, false, input_data)},
//                          {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
//                          13,
//                          ExpectedEPNodeAssignment::All,
//                          kOnnxDomain,
//                          true);
// }

// // Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// // Test that the default axis (1) for SoftMax opset < 13 works.
// TEST_F(QnnHTPBackendTests, UnaryOp_Softmax11_DefaultAxis) {
//   RunQDQOpTest<uint8_t>("Softmax",
//                         {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
//                         {},  // Uses default axis of 1 for opset < 13.
//                         11,
//                         ExpectedEPNodeAssignment::All);
// }

// // Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// // Test that setting an axis value of -1 works for Softmax opset < 13.
// TEST_F(QnnHTPBackendTests, UnaryOp_Softmax11_SetAxis) {
//   RunQDQOpTest<uint8_t>("Softmax",
//                         {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
//                         {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
//                         11,
//                         ExpectedEPNodeAssignment::All);
// }

// #endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
