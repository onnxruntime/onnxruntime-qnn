// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

template <typename DataType>
static GetTestModelFn BuildIdentityTestCase(const TestInputDef<DataType>& input_def) {
  return [input_def](ModelTestBuilder& builder) {
    MakeTestInput<DataType>(builder, "input", input_def);
    builder.MakeOutput("Y");
    builder.AddNode("identity", "Identity", {"input"}, {"Y"}, kOnnxDomain);
  };
}

template <typename QuantType>
static GetTestQDQModelFn<QuantType> BuildQDQIdentityTestCase(const TestInputDef<float>& input_def,
                                                             float output_scale_perturbation = 1.0f) {
  return [input_def, output_scale_perturbation](ModelTestBuilder& builder,
                                                std::vector<QuantParams<QuantType>>& output_qparams) {
    QNN_TEST_UNUSED_PARAMETER(output_qparams);
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                  input_qparams.scale, input_qparams.zero_point);
    builder.AddNode("identity", "Identity", {input_qdq}, {"Y"}, kOnnxDomain);
    const float output_scale = input_qparams.scale * output_scale_perturbation;
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "Y",
                                                     output_scale,
                                                     input_qparams.zero_point);
  };
}

template <typename QuantType = uint8_t>
static void RunIdentityQDQTest(const TestInputDef<float>& input_def,
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               float output_scale_perturbation = 1.0f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildIdentityTestCase<float>(input_def),
                       BuildQDQIdentityTestCase<QuantType>(input_def, output_scale_perturbation),
                       provider_options,
                       18,
                       expected_ep_assignment);
}

template <typename DataType>
static void RunIdentityNonQDQOnHTP(const TestInputDef<DataType>& input_def,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnModelTest(BuildIdentityTestCase<DataType>(input_def),
                  provider_options,
                  13,
                  expected_ep_assignment,
                  fp32_abs_err);
}

TEST_F(QnnHTPBackendTests, IdentityU8) {
  // Use slightly different output scale (1.0001x) to exercise
  // SetOutputQParamEqualToInputIfNearlyEqual snapping output qparams to input.
  RunIdentityQDQTest(TestInputDef<float>({1, 3, 4, 4}, false, 0.0f, 1.0f),
                     ExpectedEPNodeAssignment::All, /*output_scale_perturbation=*/1.0001f);
}

TEST_F(QnnHTPBackendTests, IdentityF32) {
  RunIdentityNonQDQOnHTP<float>(TestInputDef<float>({1, 3, 4, 4}, false, 0.0f, 10.0f),
                                ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, IdentityI32) {
  RunIdentityNonQDQOnHTP<int32_t>(TestInputDef<int32_t>({1, 3, 4, 4}, false, -100, 100),
                                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, IdentityRank1F32) {
  RunIdentityNonQDQOnHTP<float>(TestInputDef<float>({16}, false, 0.0f, 10.0f),
                                ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
