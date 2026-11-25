// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Session-level snapshot tests for GlobalMaxPool.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/globalmaxpool_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/session_snapshot/session_snapshot.h"

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

template <typename QuantType>
GetTestModelFn BuildGlobalMaxPoolQDQSessionModel(const GlobalMaxPoolSpec& spec) {
  return [spec](ModelTestBuilder& builder) {
    const TestInputDef<float> input_def = MakeGlobalMaxPoolInput(spec);
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq = AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                                            input_qparams.scale, input_qparams.zero_point,
                                                            spec.use_contrib_qdq);
    builder.AddNode("globalmaxpool", "GlobalMaxPool", {input_qdq}, {"pool_out"});
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "pool_out",
                                                     input_qparams.scale, input_qparams.zero_point,
                                                     spec.use_contrib_qdq);
  };
}

GetTestModelFn BuildGlobalMaxPoolSessionModel(const GlobalMaxPoolSpec& spec) {
  if (spec.data_type == GlobalMaxPoolDataType::UInt8) {
    return BuildGlobalMaxPoolQDQSessionModel<uint8_t>(spec);
  }
  return BuildGlobalMaxPoolQDQSessionModel<uint16_t>(spec);
}

void RunGlobalMaxPoolSessionSnapshot(const GlobalMaxPoolSpec& spec) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  AssertSessionSnapshotJson(BuildGlobalMaxPoolSessionModel(spec), provider_options, /*opset_version=*/18, spec.name);
}

}  // namespace

class QnnSnapshot_GlobalMaxPool_SessionTest : public ::testing::TestWithParam<GlobalMaxPoolSpec> {};

TEST_P(QnnSnapshot_GlobalMaxPool_SessionTest, Case) {
  RunGlobalMaxPoolSessionSnapshot(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_GlobalMaxPool_SessionTest,
    ::testing::ValuesIn(kGlobalMaxPoolSpecs),
    [](const ::testing::TestParamInfo<GlobalMaxPoolSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
