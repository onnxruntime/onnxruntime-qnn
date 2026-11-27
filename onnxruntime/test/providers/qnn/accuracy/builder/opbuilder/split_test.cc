// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/split_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime::test {
namespace {

size_t OutputCount(const SplitSpec& spec) {
  return spec.num_outputs > 0 ? static_cast<size_t>(spec.num_outputs) : spec.split.size();
}

std::vector<ONNX_NAMESPACE::AttributeProto> SplitAttrs(const SplitSpec& spec) {
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  if (spec.opset < 13) attrs.push_back(MakeAttribute("split", spec.split));
  if (spec.num_outputs > 0) attrs.push_back(MakeAttribute("num_outputs", spec.num_outputs));
  attrs.push_back(MakeAttribute("axis", spec.axis));
  return attrs;
}

template <typename T>
TestInputDef<T> MakeInput(const SplitSpec& spec) {
  return TestInputDef<T>(spec.shape, false, SplitInputValues<T>(spec));
}

template <typename T>
GetTestModelFn BuildSplitModel(const SplitSpec& spec) {
  return [spec, input = MakeInput<T>(spec)](ModelTestBuilder& builder) {
    MakeTestInput(builder, "input", input);
    if (spec.opset >= 13 && !spec.split.empty()) builder.Make1DInitializer<int64_t>("split", spec.split);
    std::vector<std::string> outputs;
    for (size_t i = 0; i < OutputCount(spec); ++i) {
      outputs.push_back("Y" + std::to_string(i));
      builder.MakeOutput(outputs.back());
    }
    std::vector<std::string> inputs{"input"};
    if (spec.opset >= 13 && !spec.split.empty()) inputs.push_back("split");
    builder.AddNode("Split", "Split", inputs, outputs, kOnnxDomain, SplitAttrs(spec));
  };
}

template <typename QType>
GetTestQDQModelFn<QType> BuildQDQSplitModel(const SplitSpec& spec) {
  return [spec, input = MakeInput<float>(spec)](ModelTestBuilder& builder,
                                                std::vector<QuantParams<QType>>& output_qparams) {
    MakeTestInput(builder, "input", input);
    const auto qparams = GetTestInputQuantParams<QType>(input);
    const std::string qdq_input = AddQDQNodePair<QType>(builder, "qdq_in", "input", qparams.scale,
                                                        qparams.zero_point, spec.contrib_qdq);
    if (spec.opset >= 13 && !spec.split.empty()) builder.Make1DInitializer<int64_t>("split", spec.split);
    std::vector<std::string> split_outputs;
    for (size_t i = 0; i < OutputCount(spec); ++i) split_outputs.push_back("split_out" + std::to_string(i));
    std::vector<std::string> inputs{qdq_input};
    if (spec.opset >= 13 && !spec.split.empty()) inputs.push_back("split");
    builder.AddNode("Split", "Split", inputs, split_outputs, kOnnxDomain, SplitAttrs(spec));
    QNN_ASSERT(output_qparams.size() == split_outputs.size());
    for (size_t i = 0; i < split_outputs.size(); ++i) {
      output_qparams[i] = qparams;
      AddQDQNodePairWithOutputAsGraphOutput<QType>(builder, "qdq_out" + std::to_string(i), split_outputs[i],
                                                   qparams.scale, qparams.zero_point, spec.contrib_qdq);
    }
  };
}

ProviderOptions SplitOptions(const SplitSpec& spec) {
  ProviderOptions options;
  options["backend_type"] = "htp";
  if (spec.quant_type != SplitQuantType::None) options["offload_graph_io_quantization"] = "0";
  return options;
}

void RunSplitAccuracy(const SplitSpec& spec) {
  const auto options = SplitOptions(spec);
  if (spec.quant_type == SplitQuantType::None) {
    RunQnnModelTest(BuildSplitModel<int32_t>(spec), options, spec.opset,
                    EPVerificationParams{ExpectedEPNodeAssignment::All});
  } else if (spec.quant_type == SplitQuantType::UInt8) {
    TestQDQModelAccuracy<uint8_t>(BuildSplitModel<float>(spec), BuildQDQSplitModel<uint8_t>(spec), options,
                                  spec.opset, ExpectedEPNodeAssignment::All);
  } else {
    TestQDQModelAccuracy<uint16_t>(BuildSplitModel<float>(spec), BuildQDQSplitModel<uint16_t>(spec), options,
                                   spec.opset, ExpectedEPNodeAssignment::All);
  }
}

}  // namespace

class QnnAcc_Split_AccuracyTest : public QnnHTPBackendTests, public ::testing::WithParamInterface<SplitSpec> {};
TEST_P(QnnAcc_Split_AccuracyTest, Case) { RunSplitAccuracy(GetParam()); }
INSTANTIATE_TEST_SUITE_P(, QnnAcc_Split_AccuracyTest, ::testing::ValuesIn(kSplitSpecs),
                         [](const ::testing::TestParamInfo<SplitSpec>& info) { return std::string(info.param.name); });

}  // namespace onnxruntime::test

#endif
