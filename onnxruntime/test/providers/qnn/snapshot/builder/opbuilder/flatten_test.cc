// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/flatten_graph_assertions.h"
#include "test/providers/qnn/infra/flatten_test_data.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/flatten_specs.h"
#include "test/providers/qnn/snapshot/snapshot.h"

namespace onnxruntime {
namespace test {

class QnnSnapshot_Flatten_OpBuilderTest : public ::testing::TestWithParam<FlattenSpec> {};

TEST_P(QnnSnapshot_Flatten_OpBuilderTest, Case) {
  const auto& s = GetParam();
  const bool u16 = s.profile == FlattenProfile::QdqU16;
  const bool qdq = u16 || s.profile == FlattenProfile::QdqU8 || s.profile == FlattenProfile::QdqU8GraphIO;
  const auto dtype = qdq ? (u16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
                         : (s.profile == FlattenProfile::Int32 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                                                               : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  // Compute data with the real ORT API before OpBuilderTestContext installs
  // its fake API. Both input and output qparams match legacy helper results.
  const bool graph_io = s.profile == FlattenProfile::QdqU8GraphIO;
  const auto data = qdq && !graph_io ? GetFlattenQdqData(s.input_shape, u16)
                                     : FlattenQdqData{{}, {1.0f, 0}, {1.0f, 0}};
  g_mock_init_reg.clear();
  const OrtValueInfo* input_scale = nullptr;
  const OrtValueInfo* input_zero_point = nullptr;
  const OrtValueInfo* output_scale = nullptr;
  const OrtValueInfo* output_zero_point = nullptr;
  if (qdq) {
    input_scale = g_mock_init_reg.AddScalarFloat("input_scale", data.input.scale);
    output_scale = g_mock_init_reg.AddScalarFloat("output_scale", data.output.scale);
    input_zero_point = u16 ? g_mock_init_reg.AddScalarUint16("input_zero_point", static_cast<uint16_t>(data.input.zero_point))
                           : g_mock_init_reg.AddScalarUint8("input_zero_point", static_cast<uint8_t>(data.input.zero_point));
    output_zero_point = u16 ? g_mock_init_reg.AddScalarUint16("output_zero_point", static_cast<uint16_t>(data.output.zero_point))
                            : g_mock_init_reg.AddScalarUint8("output_zero_point", static_cast<uint8_t>(data.output.zero_point));
  }

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
  auto wrapper = MakeSnapshotWrapperHtpJson(ctx, htp, {"input"}, {"output"});
  ASSERT_NE(wrapper, nullptr);
  auto input = qdq ? MakeMockQDQIODef("input", dtype, s.input_shape, input_scale, input_zero_point)
                   : MakeMockIODef("input", dtype, s.input_shape);
  auto output = qdq ? MakeMockQDQIODef("output", dtype, s.output_shape, output_scale, output_zero_point)
                    : MakeMockIODef("output", dtype, s.output_shape);
  std::vector<FakeOpAttr> attrs;
  if (s.axis) attrs.push_back(FakeOpAttr::MakeInt64("axis", *s.axis));
  auto node = qdq ? MakeMockQDQNodeUnit("Flatten", {input}, {output}, "flatten", "", 13, 0, attrs)
                  : MakeMockNodeUnit("Flatten", {input}, {output}, "flatten", "", 13, 0, attrs);
  const auto* builder = qnn::GetOpBuilder("Flatten");
  ASSERT_NE(builder, nullptr);
  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(true));
  const auto qnn_dtype = qdq ? (u16 ? QNN_DATATYPE_UFIXED_POINT_16 : QNN_DATATYPE_UFIXED_POINT_8)
                             : (s.profile == FlattenProfile::Int32 ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_FLOAT_32);
  ASSERT_NO_FATAL_FAILURE(AssertFlattenReshapeGraph(wrapper->GetQnnJSONGraph(), "input", "output",
                                                    s.input_shape, s.output_shape, qnn_dtype, !qdq || graph_io));
  if (qdq) {
    for (const auto& entry : std::vector<std::pair<std::string, FlattenQparams>>{{"input", data.input}, {"output", data.output}}) {
      const auto& quant = wrapper->GetQnnTensorWrapper(entry.first).GetQnnQuantParams().Get();
      const auto scale_offset = quant.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET
                                    ? Qnn_ScaleOffset_t{quant.bwScaleOffsetEncoding.scale, quant.bwScaleOffsetEncoding.offset}
                                    : quant.scaleOffsetEncoding;
      EXPECT_FLOAT_EQ(scale_offset.scale, entry.second.scale);
      EXPECT_EQ(scale_offset.offset, -static_cast<int32_t>(entry.second.zero_point));
    }
  }
  AssertSnapshotJson(*wrapper, s.name);
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_Flatten_OpBuilderTest, ::testing::ValuesIn(kFlattenSpecs),
    [](const ::testing::TestParamInfo<FlattenSpec>& info) { return std::string(info.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
