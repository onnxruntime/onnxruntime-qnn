// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/leakyrelu_specs.h"

namespace onnxruntime {
namespace test {

// Reuse legacy HTP setup, including platform discovery before architecture skips.
class QnnAcc_LeakyRelu_AccuracyTest : public QnnHTPBackendTests,
                                      public ::testing::WithParamInterface<LeakyReluSpec> {};

TEST_P(QnnAcc_LeakyRelu_AccuracyTest, Case) {
  const auto& spec = GetParam();
  ProviderOptions options;
  options["backend_type"] = "htp";
  options["offload_graph_io_quantization"] = "0";
  const TestInputDef<float> input(spec.shape, false, spec.input_data);
  const std::vector<ONNX_NAMESPACE::AttributeProto> attrs{MakeAttribute("alpha", spec.alpha)};
  auto reference = BuildOpTestCase<float>("LeakyRelu_node", "LeakyRelu", {input}, {}, attrs);

  // Helper defaults preserve legacy tolerance/verifier and Linux ARM64 V68 skips.
  if (spec.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    TestQDQModelAccuracy(reference,
                         BuildQDQOpTestCase<uint8_t>("LeakyRelu_node", "LeakyRelu", {input}, {}, attrs),
                         options, spec.opset, ExpectedEPNodeAssignment::All);
  } else {
#if defined(_WIN32)
    SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
    auto half_input = ConvertToFP16InputDef(input);
    TestFp16ModelAccuracy(reference,
                          BuildOpTestCase<Ort::Float16_t>("LeakyRelu_node", "LeakyRelu", {half_input}, {}, attrs),
                          options, spec.opset, ExpectedEPNodeAssignment::All);
  }
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_LeakyRelu_AccuracyTest, ::testing::ValuesIn(kLeakyReluSpecs),
    [](const ::testing::TestParamInfo<LeakyReluSpec>& info) { return std::string(info.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif
