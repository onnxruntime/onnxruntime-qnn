// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Direct QNN graph snapshots for CumSumOpBuilder.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/cumsum_specs.h"
#include "test/providers/qnn/snapshot/snapshot.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {
namespace {

void RegisterAxis(const CumSumSpec& spec) {
  g_mock_init_reg.clear();
  if (spec.axis_is_int64) {
    g_mock_init_reg.AddScalarInt64("axis", spec.axis);
  } else {
    g_mock_init_reg.AddScalarInt32("axis", static_cast<int32_t>(spec.axis));
  }
}

OrtNodeUnitIODef MakeAxis(const CumSumSpec& spec) {
  return MakeMockIODef("axis", spec.axis_is_int64 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                       std::vector<int64_t>{});
}

void RunCumSumSnapshot(const CumSumSpec& spec) {
  const IOpBuilder* builder = GetOpBuilder("CumSum");
  ASSERT_NE(builder, nullptr);

  RegisterAxis(spec);
  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
  auto wrapper = MakeSnapshotWrapperHtpJson(ctx, htp, {"input"}, {"output"});
  ASSERT_NE(wrapper, nullptr) << "Failed to initialize QNN graph for CumSum snapshot";

  const auto axis = MakeAxis(spec);
  const auto shape = std::vector<int64_t>{3, 2};
  Ort::Status status;
  if (spec.data_type == CumSumDataType::Float) {
    auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
    auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
    auto node_unit = MakeMockNodeUnit("CumSum", {input, axis}, {output}, "cumsum", {}, 14);
    status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  } else {
    auto scale = g_mock_init_reg.AddScalarFloat("scale", 0.1f);
    auto zero_point = g_mock_init_reg.AddScalarUint8("zero_point", 0);
    auto input = MakeMockQDQIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, shape, scale, zero_point);
    auto output = MakeMockQDQIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, shape, scale, zero_point);
    auto node_unit = MakeMockQDQNodeUnit("CumSum", {input, axis}, {output}, "cumsum", {}, 14);
    status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  }
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true));
  AssertSnapshotJson(*wrapper, spec.name);
}

}  // namespace

class QnnSnapshot_CumSum_OpBuilderTest : public ::testing::TestWithParam<CumSumSpec> {};

TEST_P(QnnSnapshot_CumSum_OpBuilderTest, Case) {
  RunCumSumSnapshot(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_CumSum_OpBuilderTest, ::testing::ValuesIn(kCumSumSpecs),
    [](const ::testing::TestParamInfo<CumSumSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
