// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/mock_init_registry.h"
#include "test/providers/qnn/infra/mock_node_unit.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

// Minimal repro of tetracode#21892: Add with UINT8 + UINT16 inputs and UINT8
// output at an EP partition boundary (output is a subgraph/graph output).
TEST(QnnUnit_MixedPrecisionConvertTest, Add_U8_U16_To_U8_GraphOutput) {
  g_mock_init_reg.clear();
  auto scale_in0 = g_mock_init_reg.AddScalarFloat("scale_in0", 0.0335f);
  auto zp_in0 = g_mock_init_reg.AddScalarUint8("zp_in0", 114);
  auto scale_in1 = g_mock_init_reg.AddScalarFloat("scale_in1", 0.00017949f);
  auto zp_in1 = g_mock_init_reg.AddScalarUint16("zp_in1", 46009);
  auto scale_out = g_mock_init_reg.AddScalarFloat("scale_out", 0.04008f);
  auto zp_out = g_mock_init_reg.AddScalarUint8("zp_out", 165);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ctx.output_info.names.push_back("add_out");
  ctx.output_info.indices["add_out"] = 0;
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto in0 = MakeMockQDQIODef("in0", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                              std::vector<int64_t>{1, 4}, scale_in0, zp_in0);
  auto in1 = MakeMockQDQIODef("in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                              std::vector<int64_t>{1, 4}, scale_in1, zp_in1);
  auto out = MakeMockQDQIODef("add_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                              std::vector<int64_t>{1, 4}, scale_out, zp_out);
  auto node_unit = MakeMockQDQNodeUnit("Add", {in0, in1}, {out}, "Add_6");

  const IOpBuilder* builder = GetOpBuilder("Add");
  ASSERT_NE(builder, nullptr);
  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();

  const auto& tensors = wrapper->GetModelTensorsMap();
  auto it = tensors.find("add_out");
  ASSERT_TRUE(it != tensors.end());
  EXPECT_EQ(it->second.GetTensorType(), QNN_TENSOR_TYPE_APP_READ);
  EXPECT_EQ(it->second.GetTensorDataType(), QNN_DATATYPE_UFIXED_POINT_8);
}

TEST(QnnUnit_MixedPrecisionConvertTest, Add_U8_U16_To_U8_Internal) {
  g_mock_init_reg.clear();
  auto scale_in0 = g_mock_init_reg.AddScalarFloat("scale_in0", 0.0335f);
  auto zp_in0 = g_mock_init_reg.AddScalarUint8("zp_in0", 114);
  auto scale_in1 = g_mock_init_reg.AddScalarFloat("scale_in1", 0.00017949f);
  auto zp_in1 = g_mock_init_reg.AddScalarUint16("zp_in1", 46009);
  auto scale_out = g_mock_init_reg.AddScalarFloat("scale_out", 0.04008f);
  auto zp_out = g_mock_init_reg.AddScalarUint8("zp_out", 165);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto in0 = MakeMockQDQIODef("in0", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                              std::vector<int64_t>{1, 4}, scale_in0, zp_in0);
  auto in1 = MakeMockQDQIODef("in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                              std::vector<int64_t>{1, 4}, scale_in1, zp_in1);
  auto out = MakeMockQDQIODef("add_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                              std::vector<int64_t>{1, 4}, scale_out, zp_out);
  auto node_unit = MakeMockQDQNodeUnit("Add", {in0, in1}, {out}, "Add_6");

  const IOpBuilder* builder = GetOpBuilder("Add");
  ASSERT_NE(builder, nullptr);
  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();

  const auto& tensors = wrapper->GetModelTensorsMap();
  auto it = tensors.find("add_out");
  ASSERT_TRUE(it != tensors.end());
  EXPECT_EQ(it->second.GetTensorType(), QNN_TENSOR_TYPE_NATIVE);
  EXPECT_EQ(it->second.GetTensorDataType(), QNN_DATATYPE_UFIXED_POINT_8);
}

}  // namespace test
}  // namespace onnxruntime

#endif
