// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Session-level snapshot tests for the AveragePool family.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/averagepool_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/session_snapshot/session_snapshot.h"
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

GetTestModelFn BuildAveragePoolSessionModel(const AveragePoolSpec& spec) {
  const auto input = MakeAveragePoolInput(spec);
  const auto attrs = MakeAveragePoolAttributes(spec);
  if (spec.data_type == AveragePoolDataType::Float) {
    return BuildOpTestCase<float>("averagepool", spec.op_type, {input}, {}, attrs);
  }
  auto build_qdq = BuildQDQOpTestCase<uint8_t>("averagepool", spec.op_type, {input}, {}, attrs);
  return [build_qdq](ModelTestBuilder& builder) {
    std::vector<QuantParams<uint8_t>> output_qparams(1);
    build_qdq(builder, output_qparams);
  };
}

void RunAveragePoolSessionSnapshot(const AveragePoolSpec& spec) {
  ProviderOptions options;
  options["backend_type"] = "htp";
  options["offload_graph_io_quantization"] = "0";
  AssertSessionSnapshotJson(BuildAveragePoolSessionModel(spec), options, 18, spec.name);
}

}  // namespace

class QnnSnapshot_AveragePool_SessionTest : public ::testing::TestWithParam<AveragePoolSpec> {};

TEST_P(QnnSnapshot_AveragePool_SessionTest, Case) {
  RunAveragePoolSessionSnapshot(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_AveragePool_SessionTest,
    ::testing::ValuesIn(kAveragePoolSpecs),
    [](const ::testing::TestParamInfo<AveragePoolSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
