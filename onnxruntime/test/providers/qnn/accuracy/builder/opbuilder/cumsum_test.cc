// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Accuracy and full-assignment tests for CumSum focused graph contracts.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/cumsum_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

std::vector<ONNX_NAMESPACE::AttributeProto> CumSumAttrs() {
  return {test::MakeAttribute("exclusive", static_cast<int64_t>(0)),
          test::MakeAttribute("reverse", static_cast<int64_t>(0))};
}

ProviderOptions CumSumProviderOptions([[maybe_unused]] const CumSumSpec& spec) {
  ProviderOptions options;
  options["backend_type"] = "htp";
  options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  // Only the legacy F32 cases select this simulator SoC explicitly.
  if (spec.data_type == CumSumDataType::Float) {
    options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  }
#endif
  return options;
}

GetTestModelFn BuildCumSumF32(const CumSumSpec& spec) {
  const TestInputDef<float> input{{3, 2}, false, kCumSumInputData};
  if (spec.axis_is_int64) {
    return BuildOpTestCase<float, int64_t>("cumsum", "CumSum", {input},
                                           {TestInputDef<int64_t>({}, true, {spec.axis})}, CumSumAttrs());
  }
  return BuildOpTestCase<float, int32_t>("cumsum", "CumSum", {input},
                                         {TestInputDef<int32_t>({}, true, {static_cast<int32_t>(spec.axis)})},
                                         CumSumAttrs());
}

GetTestQDQModelFn<uint8_t> BuildCumSumU8QDQ(const CumSumSpec& spec) {
  return [axis = static_cast<int32_t>(spec.axis)](ModelTestBuilder& builder,
                                                  std::vector<QuantParams<uint8_t>>& output_qparams) {
    const TestInputDef<float> input{{3, 2}, false, kCumSumInputData};
    MakeTestInput(builder, "input", input);
    const QuantParams<uint8_t> qparams = GetTestInputQuantParams<uint8_t>(input);
    const std::string input_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_in", "input", qparams.scale,
                                                          qparams.zero_point);
    builder.MakeScalarInitializer("axis", axis);
    builder.AddNode("cumsum", "CumSum", {input_qdq, "axis"}, {"cumsum_out"}, "", CumSumAttrs());
    // Preserve the CPU F32 output-derived parameters supplied by the accuracy helper.
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "cumsum_out", output_qparams[0].scale,
                                                   output_qparams[0].zero_point);
  };
}

void RunCumSumAccuracy(const CumSumSpec& spec) {
  const ProviderOptions options = CumSumProviderOptions(spec);
  if (spec.data_type == CumSumDataType::Float) {
    RunQnnModelTest(BuildCumSumF32(spec), options, /*opset_version=*/17,
                    EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
    return;
  }
  TestQDQModelAccuracy(BuildCumSumF32(spec), BuildCumSumU8QDQ(spec), options, /*opset_version=*/17,
                       ExpectedEPNodeAssignment::All);
}

}  // namespace

class QnnAcc_CumSum_AccuracyTest : public ::testing::TestWithParam<CumSumSpec> {};

TEST_P(QnnAcc_CumSum_AccuracyTest, Case) {
  RunCumSumAccuracy(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_CumSum_AccuracyTest, ::testing::ValuesIn(kCumSumSpecs),
    [](const ::testing::TestParamInfo<CumSumSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS
