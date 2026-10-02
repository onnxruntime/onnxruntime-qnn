// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Session-level snapshot tests for MaxPool.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/maxpool_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/session_snapshot/session_snapshot.h"

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
GetTestModelFn BuildMaxPoolQDQModel(const MaxPoolSpec& spec) {
  return [spec](ModelTestBuilder& builder) {
    const TestInputDef<float> input_def = MakeMaxPoolInput(spec);
    MakeTestInput(builder, "input", input_def);

    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq = AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                                            input_qparams.scale, input_qparams.zero_point,
                                                            spec.use_contrib_qdq);
    builder.AddNode("maxpool", "MaxPool", {input_qdq}, {"pool_out"}, "", MakeMaxPoolAttributes(spec));

    // QNN MaxPool requires identical input and output qparams.
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "pool_out",
                                                     input_qparams.scale, input_qparams.zero_point,
                                                     spec.use_contrib_qdq);
  };
}

GetTestModelFn BuildMaxPoolSessionModel(const MaxPoolSpec& spec) {
  switch (spec.data_type) {
    case MaxPoolDataType::Float:
      return BuildMaxPoolF32Model(spec);
    case MaxPoolDataType::UInt8:
      return BuildMaxPoolQDQModel<uint8_t>(spec);
    case MaxPoolDataType::UInt16:
      return BuildMaxPoolQDQModel<uint16_t>(spec);
  }

  ADD_FAILURE() << "BuildMaxPoolSessionModel: unsupported data type";
  return {};
}

void RunMaxPoolSessionSnapshot(const MaxPoolSpec& spec) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  AssertSessionSnapshotJson(BuildMaxPoolSessionModel(spec), provider_options, /*opset_version=*/18, spec.name);
}

}  // namespace

class QnnSnapshot_MaxPool_SessionTest : public ::testing::TestWithParam<MaxPoolSpec> {};

TEST_P(QnnSnapshot_MaxPool_SessionTest, Case) {
  RunMaxPoolSessionSnapshot(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_MaxPool_SessionTest,
    ::testing::ValuesIn(kMaxPoolSpecs),
    [](const ::testing::TestParamInfo<MaxPoolSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
