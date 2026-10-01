// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Accuracy tests paired with AveragePool-family session snapshots.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT

#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/averagepool_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

std::vector<ONNX_NAMESPACE::AttributeProto> MakeAveragePoolAttributes(const AveragePoolSpec& spec) {
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  if (!spec.kernel_shape.empty()) attrs.push_back(test::MakeAttribute("kernel_shape", spec.kernel_shape));
  if (!spec.strides.empty()) attrs.push_back(test::MakeAttribute("strides", spec.strides));
  if (!spec.pads.empty()) attrs.push_back(test::MakeAttribute("pads", spec.pads));
  if (spec.count_include_pad != 0) attrs.push_back(test::MakeAttribute("count_include_pad", spec.count_include_pad));
  if (std::string(spec.auto_pad) != "NOTSET") attrs.push_back(test::MakeAttribute("auto_pad", spec.auto_pad));
  return attrs;
}

TestInputDef<float> MakeAveragePoolInput(const AveragePoolSpec& spec) {
  if (!spec.input_data.empty()) return TestInputDef<float>(spec.input_shape, false, spec.input_data);
  const int64_t count = std::accumulate(spec.input_shape.begin(), spec.input_shape.end(), int64_t{1}, std::multiplies<>{});
  return TestInputDef<float>(spec.input_shape, false, GetFloatDataInRange(spec.input_min, spec.input_max, count));
}

GetTestModelFn BuildAveragePoolF32Model(const AveragePoolSpec& spec) {
  return BuildOpTestCase<float>("averagepool", spec.op_type, {MakeAveragePoolInput(spec)}, {}, MakeAveragePoolAttributes(spec));
}

GetTestQDQModelFn<uint8_t> BuildAveragePoolQDQModel(const AveragePoolSpec& spec) {
  return BuildQDQOpTestCase<uint8_t>("averagepool", spec.op_type, {MakeAveragePoolInput(spec)}, {}, MakeAveragePoolAttributes(spec));
}

void CheckSingleQnnSubgraph(const Ort::Session& session) {
  size_t count = 0;
  for (const auto& subgraph : session.GetEpGraphAssignmentInfo()) {
    if (subgraph.GetEpName() == kQnnExecutionProvider) ++count;
  }
  EXPECT_EQ(count, 1u) << "Expected one fused QNN subgraph for AveragePool rank-3 input.";
}

void RunAveragePoolAccuracy(const AveragePoolSpec& spec) {
  ProviderOptions options;
  options["backend_type"] = "htp";
  options["offload_graph_io_quantization"] = "0";
  if (spec.data_type == AveragePoolDataType::Float) {
    std::function<void(const Ort::Session&)> check;
    EPVerificationParams verification{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-5f)};
    if (spec.expect_single_qnn_subgraph) {
      check = CheckSingleQnnSubgraph;
      verification.graph_verifier = &check;
    }
    RunQnnModelTest(BuildAveragePoolF32Model(spec), options, 18, verification);
    return;
  }
  TestQDQModelAccuracy(BuildAveragePoolF32Model(spec), BuildAveragePoolQDQModel(spec), options, 18,
                       ExpectedEPNodeAssignment::All);
}

}  // namespace

class QnnAcc_AveragePool_AccuracyTest : public ::testing::TestWithParam<AveragePoolSpec> {};

TEST_P(QnnAcc_AveragePool_AccuracyTest, Case) {
  RunAveragePoolAccuracy(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_AveragePool_AccuracyTest,
    ::testing::ValuesIn(kAveragePoolSpecs),
    [](const ::testing::TestParamInfo<AveragePoolSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT
