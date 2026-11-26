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
#include "test/providers/qnn/qnn_test_utils.h"
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
    const TestInputDef<float> input_def{shape, false, kCumSumInputData};
    const auto input_qparams = GetTestInputQuantParams<uint8_t>(input_def);
    // Match legacy Accuracy: output QDQ is derived from CPU F32 CumSum output.
    auto build_f32 = BuildOpTestCase<float, int32_t>(
        "cumsum", "CumSum", {input_def},
        {TestInputDef<int32_t>({}, true, {static_cast<int32_t>(spec.axis)})},
        {test::MakeAttribute("exclusive", static_cast<int64_t>(0)),
         test::MakeAttribute("reverse", static_cast<int64_t>(0))});
    std::unique_ptr<ModelAndBuilder> model;
    CreateModelInMemory(model, build_f32, /*opset_version=*/17);
    std::vector<Ort::Value> outputs;
    InferenceModelCPU(model->model_data, "cumsum_snapshot_f32", model->builder.feeds_, outputs);
    ASSERT_EQ(outputs.size(), 1u);
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const auto output_qparams = GetDataQuantParams<uint8_t>(
        gsl::span<const float>(outputs[0].GetTensorData<float>(), info.GetElementCount()));
    auto input_scale = g_mock_init_reg.AddScalarFloat("input_scale", input_qparams.scale);
    auto input_zp = g_mock_init_reg.AddScalarUint8("input_zero_point", input_qparams.zero_point);
    auto output_scale = g_mock_init_reg.AddScalarFloat("output_scale", output_qparams.scale);
    auto output_zp = g_mock_init_reg.AddScalarUint8("output_zero_point", output_qparams.zero_point);
    auto input = MakeMockQDQIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, shape, input_scale, input_zp);
    auto output = MakeMockQDQIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, shape, output_scale, output_zp);
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
