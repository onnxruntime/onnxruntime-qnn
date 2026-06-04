// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs an LpPool model on the given QNN backend with FP32 inputs.
// HTP callers can enable_htp_fp16_precision=true to execute the
// FP32 model at FP16 precision.
static void RunLpPoolOpTest(const std::vector<TestInputDef<float>>& input_defs,
                            const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            const std::string& backend_name = "cpu",
                            int opset = 22,
                            float fp32_abs_err = 1e-5f,
                            bool enable_htp_fp16_precision = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  if (enable_htp_fp16_precision) {
#if defined(_WIN32)
    SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
#if defined(__linux__) && !defined(__aarch64__)
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  RunQnnModelTest(BuildOpTestCase<float>("LpPool_node", "LpPool", input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Runs a native FP16 LpPool model on the given QNN backend (HTP or GPU).
static void RunLpPoolFP16Test(const std::vector<TestInputDef<float>>& input_defs,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              const std::string& backend_name = "htp",
                              int opset = 22,
                              float tolerance = 0.008f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  std::vector<TestInputDef<Ort::Float16_t>> input_fp16_defs;
  input_fp16_defs.reserve(input_defs.size());
  for (const auto& def : input_defs) {
    input_fp16_defs.push_back(ConvertToFP16InputDef(def));
  }

  RunQnnModelTest(BuildOpTestCase<Ort::Float16_t>("LpPool_node", "LpPool", input_fp16_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  tolerance);
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

TEST_F(QnnCPUBackendTests, LpPool_Rank5) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 128))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})},
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

// Rejection: ceil_mode=1 is not supported on the CPU backend.
// QNN CPU's PoolAvg2d silently ignores rounding_mode and produces a floor-shape output, which
// would leave the extra ceil-mode positions filled with garbage.
TEST_F(QnnCPUBackendTests, LpPool_Reject_CeilMode) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 5, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 50))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("ceil_mode", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::None);
}

TEST_F(QnnCPUBackendTests, LpPool_P1_Basic) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("p", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::All);
}

// Verifies Reshape bracketing on the Abs (p=1) path.
TEST_F(QnnCPUBackendTests, LpPool_P1_Rank3) {
  RunLpPoolOpTest({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2}),
                   test::MakeAttribute("p", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LpPool_P1_Rank5) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 128))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2, 2}),
                   test::MakeAttribute("p", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::All);
}

// Rejection: p >= 3 is not supported (no L_p generalization in this builder).
TEST_F(QnnCPUBackendTests, LpPool_Reject_P3) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("p", static_cast<int64_t>(3))},
                  ExpectedEPNodeAssignment::None);
}

// Rejection: dilations > 1 are not supported by QNN AvgPool.
TEST_F(QnnCPUBackendTests, LpPool_Reject_Dilation) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("dilations", std::vector<int64_t>{2, 2})},
                  ExpectedEPNodeAssignment::None);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// HTP backend tests
//

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_Basic) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2})},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.02f);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_WithPads) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.02f);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_AutoPad_SameUpper) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("auto_pad", "SAME_UPPER")},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.02f);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_Rank3) {
  RunLpPoolOpTest({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2})},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.02f);
}

// Rejection: rank-5 (3D pooling) is not supported on the HTP backend for FP32/FP16 inputs.
// PoolAvg3d native-float fails dry-run validation. BF16 rank-5 works on V81+ (see
// LpPool_HTP_BF16_Rank5); QDQ rank-5 support is deferred to the QDQ follow-up PR.
TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_Reject_Rank5) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 128))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})},
                  ExpectedEPNodeAssignment::None,
                  "htp", 22, 0.008f);
}

// Rejection: ceil_mode=1 is not supported on the HTP backend.
// QNN HTP's PoolAvg2d reads out-of-bounds memory at ceil-mode boundary windows (positions whose
// window extends past the input), producing NaN/garbage values regardless of how the chain is
// configured. CPU rejects for a different reason (rounding_mode silently ignored).
TEST_F(QnnHTPBackendTests, LpPool_HTP_Reject_CeilMode) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 5, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 50))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("ceil_mode", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::None,
                  "htp", 22, 0.008f);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_P1) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("p", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.05f);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_P1_Rank3) {
  RunLpPoolOpTest({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2}),
                   test::MakeAttribute("p", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.05f);
}

// FP32 model executed at FP16 precision on HTP (uses enable_htp_fp16_precision=true).
TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_AS_FP16_Basic) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2})},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.02f, /*enable_htp_fp16_precision=*/true);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_AS_FP16_WithPads) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.02f, /*enable_htp_fp16_precision=*/true);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_AS_FP16_Rank3) {
  RunLpPoolOpTest({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2})},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.02f, /*enable_htp_fp16_precision=*/true);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP32_AS_FP16_P1) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("p", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::All,
                  "htp", 22, 0.05f, /*enable_htp_fp16_precision=*/true);
}

// Native FP16 tests.
TEST_F(QnnHTPBackendTests, LpPool_HTP_FP16_Basic) {
  RunLpPoolFP16Test({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                    {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                     test::MakeAttribute("strides", std::vector<int64_t>{2, 2})},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP16_WithPads) {
  RunLpPoolFP16Test({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                    {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                     test::MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP16_Rank3) {
  RunLpPoolFP16Test({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                    {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                     test::MakeAttribute("strides", std::vector<int64_t>{2})},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP16_P1) {
  RunLpPoolFP16Test({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                    {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                     test::MakeAttribute("p", static_cast<int64_t>(1))},
                    ExpectedEPNodeAssignment::All,
                    "htp", 22, 0.05f);
}

TEST_F(QnnHTPBackendTests, LpPool_HTP_FP16_P1_Rank3) {
  RunLpPoolFP16Test({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                    {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                     test::MakeAttribute("strides", std::vector<int64_t>{2}),
                     test::MakeAttribute("p", static_cast<int64_t>(1))},
                    ExpectedEPNodeAssignment::All);
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

// BF16 rank-5: works on V81+ HTP because BF16 PoolAvg3d has a kernel there. Native FP32/FP16
// rank-5 is rejected in IsOpSupported (see LpPool_HTP_FP32_Reject_Rank5); QDQ rank-5 will be
// added in the QDQ follow-up PR.
TEST_F(QnnHTPBackendTests, LpPool_HTP_BF16_Rank5) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunLpPoolHTPBF16Test({TestInputDef<float>({1, 2, 4, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 128))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                        test::MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})},
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

TEST_F(QnnHTPBackendTests, LpPool_HTP_BF16_P1) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunLpPoolHTPBF16Test({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                        test::MakeAttribute("p", static_cast<int64_t>(1))},
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

// Rejection: rank-5 (3D pooling) is not supported on the QNN GPU backend (no PoolAvg3d kernel).
TEST_F(QnnGPUBackendTests, LpPool_GPU_Reject_Rank5) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 128))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})},
                  ExpectedEPNodeAssignment::None,
                  "gpu");
}

TEST_F(QnnGPUBackendTests, LpPool_GPU_P1) {
  RunLpPoolOpTest({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                   test::MakeAttribute("p", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::All,
                  "gpu");
}

TEST_F(QnnGPUBackendTests, LpPool_GPU_P1_Rank3) {
  RunLpPoolOpTest({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                  {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                   test::MakeAttribute("strides", std::vector<int64_t>{2}),
                   test::MakeAttribute("p", static_cast<int64_t>(1))},
                  ExpectedEPNodeAssignment::All,
                  "gpu");
}

TEST_F(QnnGPUBackendTests, LpPool_GPU_FP16_Basic) {
  RunLpPoolFP16Test({TestInputDef<float>({1, 2, 6, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72))},
                    {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                     test::MakeAttribute("strides", std::vector<int64_t>{2, 2})},
                    ExpectedEPNodeAssignment::All,
                    "gpu");
}

TEST_F(QnnGPUBackendTests, LpPool_GPU_FP16_WithPads) {
  RunLpPoolFP16Test({TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                    {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                     test::MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                    ExpectedEPNodeAssignment::All,
                    "gpu");
}

TEST_F(QnnGPUBackendTests, LpPool_GPU_FP16_Rank3) {
  RunLpPoolFP16Test({TestInputDef<float>({1, 4, 8}, false, GetFloatDataInRange(-10.0f, 10.0f, 32))},
                    {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                     test::MakeAttribute("strides", std::vector<int64_t>{2})},
                    ExpectedEPNodeAssignment::All,
                    "gpu");
}

#endif  // defined(_M_ARM64) — GPU tests

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
