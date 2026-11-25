// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Accuracy tests paired with GlobalMaxPool session snapshots.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/globalmaxpool_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

TestInputDef<float> MakeGlobalMaxPoolInput(const GlobalMaxPoolSpec& spec) {
  if (spec.use_random_input) {
    return TestInputDef<float>(spec.input_shape, false, -10.0f, 10.0f);
  }
  const int64_t element_count = std::accumulate(spec.input_shape.begin(), spec.input_shape.end(),
                                                int64_t{1}, std::multiplies<>{});
  return TestInputDef<float>(spec.input_shape, false,
                             GetFloatDataInRange(-10.0f, 10.0f, element_count));
}

GetTestModelFn BuildGlobalMaxPoolF32Model(const GlobalMaxPoolSpec& spec) {
  return BuildOpTestCase<float>("globalmaxpool", "GlobalMaxPool", {MakeGlobalMaxPoolInput(spec)}, {}, {});
}

template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildGlobalMaxPoolQDQModel(const GlobalMaxPoolSpec& spec) {
  const TestInputDef<float> input_def = MakeGlobalMaxPoolInput(spec);
  return [input_def, use_contrib_qdq = spec.use_contrib_qdq](
             ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) {
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq = AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                                            input_qparams.scale, input_qparams.zero_point,
                                                            use_contrib_qdq);
    builder.AddNode("globalmaxpool", "GlobalMaxPool", {input_qdq}, {"pool_out"});

    // GlobalMaxPool shares MaxPool's equal input/output QDQ contract.
    output_qparams[0] = input_qparams;
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "pool_out",
                                                     input_qparams.scale, input_qparams.zero_point,
                                                     use_contrib_qdq);
  };
}

void RunGlobalMaxPoolAccuracy(const GlobalMaxPoolSpec& spec) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  if (spec.data_type == GlobalMaxPoolDataType::UInt8) {
    TestQDQModelAccuracy(BuildGlobalMaxPoolF32Model(spec), BuildGlobalMaxPoolQDQModel<uint8_t>(spec),
                         provider_options, /*opset_version=*/18, ExpectedEPNodeAssignment::All);
  } else {
    TestQDQModelAccuracy(BuildGlobalMaxPoolF32Model(spec), BuildGlobalMaxPoolQDQModel<uint16_t>(spec),
                         provider_options, /*opset_version=*/18, ExpectedEPNodeAssignment::All);
  }
}

}  // namespace

class QnnAcc_GlobalMaxPool_AccuracyTest : public ::testing::TestWithParam<GlobalMaxPoolSpec> {};

TEST_P(QnnAcc_GlobalMaxPool_AccuracyTest, Case) {
  RunGlobalMaxPoolAccuracy(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_GlobalMaxPool_AccuracyTest,
    ::testing::ValuesIn(kGlobalMaxPoolSpecs),
    [](const ::testing::TestParamInfo<GlobalMaxPoolSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS
