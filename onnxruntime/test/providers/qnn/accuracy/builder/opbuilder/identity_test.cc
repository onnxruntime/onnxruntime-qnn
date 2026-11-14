// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Op-builder-paired accuracy tests for IdentityOpBuilder.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/identity_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

ProviderOptions MakeHtpProviderOptions(bool is_qdq) {
  ProviderOptions options;
  options["backend_type"] = "htp";
  if (is_qdq) {
    options["offload_graph_io_quantization"] = "0";
  }
  return options;
}

GetTestModelFn BuildIdentityFloatModel(const IdentitySpec& spec) {
  auto input = TestInputDef<float>(spec.shape, false, spec.float_data);
  return BuildOpTestCase<float>("Identity_node", "Identity", {input}, {}, {});
}

GetTestModelFn BuildIdentityInt32Model(const IdentitySpec& spec) {
  auto input = TestInputDef<int32_t>(spec.shape, false, spec.int32_data);
  return BuildOpTestCase<int32_t>("Identity_node", "Identity", {input}, {}, {});
}

GetTestQDQModelFn<uint8_t> BuildQDQIdentityModel(const IdentitySpec& spec) {
  return [spec](ModelTestBuilder& builder, std::vector<QuantParams<uint8_t>>& output_qparams) {
    QNN_TEST_UNUSED_PARAMETER(output_qparams);
    auto input = TestInputDef<float>(spec.shape, false, spec.float_data);
    MakeTestInput(builder, "input", input);

    const QuantParams<uint8_t> input_qparams{spec.input_scale, spec.zero_point};
    const std::string input_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_in", "input",
                                                          input_qparams.scale, input_qparams.zero_point);
    builder.AddNode("Identity_node", "Identity", {input_qdq}, {"identity_out"});

    const QuantParams<uint8_t> output_qparams_for_model{spec.output_scale, spec.zero_point};
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "identity_out",
                                                   output_qparams_for_model.scale,
                                                   output_qparams_for_model.zero_point);
  };
}

void RunIdentityAccuracy(const IdentitySpec& spec) {
  switch (spec.kind) {
    case IdentityDataKind::Float:
      RunQnnModelTest(BuildIdentityFloatModel(spec), MakeHtpProviderOptions(false), spec.opset,
                      EPVerificationParams{ExpectedEPNodeAssignment::All,
                                           ElementwiseAbsoluteVerifier(1e-5f)});
      break;
    case IdentityDataKind::Int32:
      RunQnnModelTest(BuildIdentityInt32Model(spec), MakeHtpProviderOptions(false), spec.opset,
                      EPVerificationParams{ExpectedEPNodeAssignment::All});
      break;
    case IdentityDataKind::QDQUint8:
      TestQDQModelAccuracy(BuildIdentityFloatModel(spec), BuildQDQIdentityModel(spec),
                           MakeHtpProviderOptions(true), spec.opset,
                           ExpectedEPNodeAssignment::All);
      break;
    default:
      ADD_FAILURE() << "Unsupported Identity data kind";
  }
}

}  // namespace

class QnnAcc_Identity_AccuracyTest : public ::testing::TestWithParam<IdentitySpec> {};

TEST_P(QnnAcc_Identity_AccuracyTest, Case) {
  RunIdentityAccuracy(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_Identity_AccuracyTest,
    ::testing::ValuesIn(kIdentitySpecs),
    [](const ::testing::TestParamInfo<IdentitySpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT
