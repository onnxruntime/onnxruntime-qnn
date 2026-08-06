// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <limits>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

template <typename DataType>
static void RunIsInfTest(const std::vector<TestInputDef<DataType>>& input_defs,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                         ExpectedEPNodeAssignment expected_ep_assignment,
                         float fp32_abs_err = 1e-5,
                         const std::string& backend_name = "cpu",
                         int opset = 20,
                         bool enable_htp_fp16_precision = false) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";
  if (enable_htp_fp16_precision) {
    provider_options["enable_htp_fp16_precision"] = "1";
#if defined(__linux__) && !defined(__aarch64__)
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  }

  RunQnnModelTest(BuildOpTestCase<DataType>("isinf_node", "IsInf", input_defs, {}, attrs, kOnnxDomain),
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(fp32_abs_err)});
}

TEST_F(QnnCPUBackendTests, IsInfScalarPosInf) {
  const std::vector<int64_t> input_shape{};
  const std::vector<float> input_data{std::numeric_limits<float>::infinity()};

  RunIsInfTest<float>({TestInputDef<float>(input_shape, false, input_data)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      0.0f);
}

TEST_F(QnnCPUBackendTests, IsInfMix2D) {
  const std::vector<int64_t> input_shape{2, 4};
  const std::vector<float> input_data{
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 1.0f, 2.0f,
      3.0f, 4.0f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};

  RunIsInfTest<float>({TestInputDef<float>(input_shape, false, input_data)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      0.0f);
}

TEST_F(QnnCPUBackendTests, IsInfDetectPositiveOnly) {
  const std::vector<int64_t> input_shape{2, 2};
  const std::vector<float> input_data{
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0.0f, 1.0f};

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(test::MakeAttribute("detect_positive", static_cast<int64_t>(1)));
  attrs.push_back(test::MakeAttribute("detect_negative", static_cast<int64_t>(0)));

  RunIsInfTest<float>({TestInputDef<float>(input_shape, false, input_data)},
                      attrs,
                      ExpectedEPNodeAssignment::All,
                      0.0f);
}

TEST_F(QnnCPUBackendTests, IsInfDetectNegativeOnly) {
  const std::vector<int64_t> input_shape{2, 2};
  const std::vector<float> input_data{
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0.0f, 1.0f};

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(test::MakeAttribute("detect_positive", static_cast<int64_t>(0)));
  attrs.push_back(test::MakeAttribute("detect_negative", static_cast<int64_t>(1)));

  RunIsInfTest<float>({TestInputDef<float>(input_shape, false, input_data)},
                      attrs,
                      ExpectedEPNodeAssignment::All,
                      0.0f);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// IsInf is intentionally declined on the HTP backend: the only registered HTP IsInf kernel
// is fp16 crouton-tiled with no TCM/flat fallback, so QnnHtp graphFinalize fails with error
// 1002 (QNN_COMMON_ERROR_MEM_ALLOC) even though op-config validation accepts the node. The
// IsInfOpBuilder::IsOpSupported override rejects the NPU backend so ORT falls IsInf back to the
// CPU EP. This test asserts that fallback: no node is assigned to QNN.
TEST_F(QnnHTPBackendTests, IsInfMix2D) {
  const std::vector<int64_t> input_shape{2, 4};
  const std::vector<float> input_data{
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 1.0f, 2.0f,
      3.0f, 4.0f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};

  RunIsInfTest<float>({TestInputDef<float>(input_shape, false, input_data)},
                      {},
                      ExpectedEPNodeAssignment::None,
                      0.0f,
                      "htp",
                      20,
                      true);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
