// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Accuracy tests paired with MaxPool session snapshots.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT

#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/maxpool_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

std::vector<ONNX_NAMESPACE::AttributeProto> MakeMaxPoolAttributes(const MaxPoolSpec& spec) {
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(test::MakeAttribute("kernel_shape", spec.kernel_shape));
  attrs.push_back(test::MakeAttribute("strides", spec.strides));
  if (!spec.pads.empty()) {
    attrs.push_back(test::MakeAttribute("pads", spec.pads));
  }
  attrs.push_back(test::MakeAttribute("ceil_mode", spec.ceil_mode));
  attrs.push_back(test::MakeAttribute("auto_pad", spec.auto_pad));
  return attrs;
}

TestInputDef<float> MakeMaxPoolInput(const MaxPoolSpec& spec) {
  const int64_t element_count = std::accumulate(spec.input_shape.begin(), spec.input_shape.end(),
                                                int64_t{1}, std::multiplies<>{});
  return TestInputDef<float>(spec.input_shape, false,
                             GetFloatDataInRange(-10.0f, 10.0f, element_count));
}

GetTestModelFn BuildMaxPoolF32Model(const MaxPoolSpec& spec) {
  return BuildOpTestCase<float>("maxpool", "MaxPool", {MakeMaxPoolInput(spec)}, {},
                                MakeMaxPoolAttributes(spec));
}

template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildMaxPoolQDQModel(const MaxPoolSpec& spec) {
  const TestInputDef<float> input_def = MakeMaxPoolInput(spec);
  const std::vector<ONNX_NAMESPACE::AttributeProto> attrs = MakeMaxPoolAttributes(spec);
  return [input_def, attrs, use_contrib_qdq = spec.use_contrib_qdq](
             ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) {
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq = AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                                            input_qparams.scale, input_qparams.zero_point,
                                                            use_contrib_qdq);
    builder.AddNode("maxpool", "MaxPool", {input_qdq}, {"pool_out"}, "", attrs);

    // Mirror the legacy MaxPool tests and QNN equal-qparam requirement.
    output_qparams[0] = input_qparams;
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "pool_out",
                                                     input_qparams.scale, input_qparams.zero_point,
                                                     use_contrib_qdq);
  };
}

void CheckSingleQnnSubgraph(const Ort::Session& session) {
  size_t num_qnn_subgraphs = 0;
  for (const auto& subgraph : session.GetEpGraphAssignmentInfo()) {
    if (subgraph.GetEpName() == kQnnExecutionProvider) {
      ++num_qnn_subgraphs;
    }
  }
  EXPECT_EQ(num_qnn_subgraphs, 1u) << "Expected one QNN fused node for MaxPool rank-3 input.";
}

void RunMaxPoolAccuracy(const MaxPoolSpec& spec) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  if (spec.data_type == MaxPoolDataType::Float) {
    std::function<void(const Ort::Session&)> check;
    EPVerificationParams verification{ExpectedEPNodeAssignment::All,
                                      ElementwiseAbsoluteVerifier(1e-5f)};
    if (spec.expect_single_qnn_subgraph) {
      check = CheckSingleQnnSubgraph;
      verification.graph_verifier = &check;
    }
    RunQnnModelTest(BuildMaxPoolF32Model(spec), provider_options, /*opset_version=*/18, verification);
    return;
  }

  const QDQTolerance tolerance = spec.qdq_tolerance == 0.0f
                                     ? QDQTolerance()
                                     : QDQTolerance(spec.qdq_tolerance);
  if (spec.data_type == MaxPoolDataType::UInt8) {
    TestQDQModelAccuracy(BuildMaxPoolF32Model(spec), BuildMaxPoolQDQModel<uint8_t>(spec),
                         provider_options, /*opset_version=*/18, ExpectedEPNodeAssignment::All, tolerance);
  } else if (spec.data_type == MaxPoolDataType::UInt16) {
    TestQDQModelAccuracy(BuildMaxPoolF32Model(spec), BuildMaxPoolQDQModel<uint16_t>(spec),
                         provider_options, /*opset_version=*/18, ExpectedEPNodeAssignment::All, tolerance);
  } else {
    ADD_FAILURE() << "RunMaxPoolAccuracy: unsupported data type";
  }
}

}  // namespace

class QnnAcc_MaxPool_AccuracyTest : public ::testing::TestWithParam<MaxPoolSpec> {};

TEST_P(QnnAcc_MaxPool_AccuracyTest, Case) {
  RunMaxPoolAccuracy(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_MaxPool_AccuracyTest,
    ::testing::ValuesIn(kMaxPoolSpecs),
    [](const ::testing::TestParamInfo<MaxPoolSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT
