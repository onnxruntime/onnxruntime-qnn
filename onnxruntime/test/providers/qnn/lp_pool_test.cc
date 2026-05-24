// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs an LpPool model on the QNN CPU/GPU backend. Checks graph node assignment and that inference
// outputs for QNN and CPU match.
static void RunLpPoolOpTest(const std::vector<TestInputDef<float>>& input_defs,
                            const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            const std::string& backend_name = "cpu",
                            int opset = 22) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("LpPool_node", "LpPool", input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU backend tests
//

TEST_F(QnnCPUBackendTests, LpPool_Basic) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("p", static_cast<int64_t>(2))},
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LpPool_WithStrides) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2})},
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LpPool_WithPads) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LpPool_Rank3) {
  RunLpPoolOpTest({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2})},
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LpPool_AutoPad_SameUpper) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("auto_pad", "SAME_UPPER")},
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LpPool_AutoPad_SameLower) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("auto_pad", "SAME_LOWER")},
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LpPool_AutoPad_Valid) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                   test::MakeAttribute("auto_pad", "VALID")},
                  ExpectedEPNodeAssignment::All);
}

// Rejection: p=1 is not supported by QNN L2Pool2d.
TEST_F(QnnCPUBackendTests, LpPool_Reject_P1) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("p", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::None);
}

// Rejection: ceil_mode=1 is not supported by QNN L2Pool2d.
TEST_F(QnnCPUBackendTests, LpPool_Reject_CeilMode) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("ceil_mode", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::None);
}

// Rejection: dilations > 1 are not supported by QNN L2Pool2d.
TEST_F(QnnCPUBackendTests, LpPool_Reject_Dilation) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("dilations", std::vector<int64_t>{2, 2})},
                  ExpectedEPNodeAssignment::None);
}

// Rejection: rank-5 inputs are not supported.
TEST_F(QnnCPUBackendTests, LpPool_Reject_Rank5) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 128))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2})},
                  ExpectedEPNodeAssignment::None);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// HTP backend tests
//

TEST_F(QnnHTPBackendTests, LpPool_HTP_Float32_Basic) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2})},
                  ExpectedEPNodeAssignment::All,
                  "htp");
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_Float32_WithPads) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                  ExpectedEPNodeAssignment::All,
                  "htp");
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_Float32_AutoPad_SameUpper) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("auto_pad", "SAME_UPPER")},
                  ExpectedEPNodeAssignment::All,
                  "htp");
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_Float32_Rank3) {
  RunLpPoolOpTest({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2})},
                  ExpectedEPNodeAssignment::All,
                  "htp");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(__aarch64__) || defined(_M_ARM64)

static void RunLpPoolHTPBF16Test(const std::vector<TestInputDef<float>>& input_defs,
                                 const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 22,
                                 float tolerance = 0.008f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["htp_bf16_enable"] = "1";
  provider_options["soc_model"] = "88";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("LpPool_node", "LpPool", input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  tolerance);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_BF16_Basic) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunLpPoolHTPBF16Test({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2})},
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_BF16_WithStridesAndPads) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunLpPoolHTPBF16Test({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                        test::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                        test::MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_BF16_Rank3) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunLpPoolHTPBF16Test({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                        test::MakeAttribute("strides", std::vector<int64_t>{2})},
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_BF16_AutoPad_SameUpper) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunLpPoolHTPBF16Test({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                        test::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                        test::MakeAttribute("auto_pad", "SAME_UPPER")},
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_BF16_AsymmetricKernel) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunLpPoolHTPBF16Test({TestInputDef<float>({1, 2, 6, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 96))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 2}),
                        test::MakeAttribute("strides", std::vector<int64_t>{2, 1})},
                       ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

#if defined(_M_ARM64)

//
// GPU backend tests
//

TEST_F(QnnGPUBackendTests, LpPool_GPU_Basic) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2})},
                  ExpectedEPNodeAssignment::All,
                  "gpu");
}

TEST_F(QnnGPUBackendTests, LpPool_GPU_WithPads) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                  ExpectedEPNodeAssignment::All,
                  "gpu");
}

#endif  // defined(_M_ARM64) — GPU tests

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
