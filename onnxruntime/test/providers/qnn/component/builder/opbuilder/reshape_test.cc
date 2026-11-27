// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

void RegisterShape(const std::vector<int64_t>& values) {
  MockInitSpec spec;
  spec.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  spec.dims = {static_cast<int64_t>(values.size())};
  spec.raw_bytes.resize(values.size() * sizeof(int64_t));
  std::memcpy(spec.raw_bytes.data(), values.data(), spec.raw_bytes.size());
  g_mock_init_reg.Add("shape", std::move(spec));
}

void CheckAllowZero(const std::vector<int64_t>& input_shape,
                    const std::vector<int64_t>& output_shape, bool constant_shape,
                    bool expected_supported, const char* error) {
  g_mock_init_reg.clear();
  if (constant_shape) RegisterShape(output_shape);
  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ctx.qnn_interface.backendValidateOpConfig = [](Qnn_BackendHandle_t, Qnn_OpConfig_t) -> Qnn_ErrorHandle_t { return QNN_SUCCESS; };
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_shape);
  auto shape = MakeMockIODef("shape", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{2});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, output_shape);
  auto node = MakeMockNodeUnit("Reshape", {input, shape}, {output}, "reshape", "", 19, 0,
                               {FakeOpAttr::MakeInt64("allowzero", 1)});
  const auto* builder = qnn::GetOpBuilder("Reshape");
  ASSERT_NE(builder, nullptr);
  const auto status = builder->IsOpSupported(*wrapper, node, ctx.ort_logger);
  EXPECT_EQ(status.IsOK(), expected_supported) << status.GetErrorMessage();
  if (expected_supported) {
    EXPECT_EQ(wrapper->GetQnnTensorWrapper("output").GetTensorDims(),
              std::vector<uint32_t>(output_shape.begin(), output_shape.end()));
    EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist("shape"));
  } else {
    EXPECT_NE(std::string(status.GetErrorMessage()).find(error), std::string::npos) << status.GetErrorMessage();
    EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist("output"));
  }
}

void CheckOutputQuantization(float output_scale, uint8_t output_zp, bool per_channel,
                             bool expect_override) {
  constexpr float input_scale = 1.0f / 1024.0f;
  g_mock_init_reg.clear();
  RegisterShape({1, 3});
  const auto* in_scale = per_channel ? g_mock_init_reg.AddTensorFloat("in_scale", {3}, {input_scale, input_scale, input_scale})
                                     : g_mock_init_reg.AddScalarFloat("in_scale", input_scale);
  const auto* out_scale = per_channel ? g_mock_init_reg.AddTensorFloat("out_scale", {3}, {output_scale, output_scale, output_scale})
                                      : g_mock_init_reg.AddScalarFloat("out_scale", output_scale);
  const auto* in_zp = per_channel ? g_mock_init_reg.AddTensorUint8("in_zp", {3}, {0, 0, 0})
                                  : g_mock_init_reg.AddScalarUint8("in_zp", 0);
  const auto* out_zp = per_channel ? g_mock_init_reg.AddTensorUint8("out_zp", {3}, {output_zp, output_zp, output_zp})
                                   : g_mock_init_reg.AddScalarUint8("out_zp", output_zp);
  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  // Removing unit dimensions preserves the channel axis and all three channels.
  const std::optional<int64_t> axis = per_channel ? std::optional<int64_t>{1} : std::nullopt;
  auto input = MakeMockQDQIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                                std::vector<int64_t>{1, 3, 1, 1}, in_scale, in_zp, axis);
  auto output = MakeMockQDQIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                                 std::vector<int64_t>{1, 3}, out_scale, out_zp, axis);
  auto shape = MakeMockIODef("shape", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{2});
  auto node = MakeMockQDQNodeUnit("Reshape", {input, shape}, {output}, "reshape", "", 19);
  const auto* builder = qnn::GetOpBuilder("Reshape");
  ASSERT_NE(builder, nullptr);
  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  const auto& quant = wrapper->GetQnnTensorWrapper("output").GetQnnQuantParams();
  std::vector<float> scales;
  ASSERT_TRUE(quant.GetScales(scales).IsOK());
  ASSERT_EQ(scales.size(), per_channel ? 3u : 1u);
  for (float scale : scales) EXPECT_EQ(scale, expect_override ? input_scale : output_scale);
  if (per_channel) {
    EXPECT_TRUE(quant.IsPerChannel());
    EXPECT_EQ(quant.Get().axisScaleOffsetEncoding.axis, 1);
    for (uint32_t i = 0; i < 3; ++i) {
      EXPECT_EQ(quant.Get().axisScaleOffsetEncoding.scaleOffset[i].offset, -static_cast<int32_t>(output_zp));
    }
  } else {
    EXPECT_EQ(quant.Get().scaleOffsetEncoding.offset, expect_override ? 0 : -static_cast<int32_t>(output_zp));
  }
}

}  // namespace

TEST(QnnUnit_Reshape_OpBuilderTest, AllowZero_ConstantNonzeroShape_Accepted) {
  CheckAllowZero({1, 3, 4, 4}, {1, 48}, true, true, "");
}

TEST(QnnUnit_Reshape_OpBuilderTest, AllowZero_ConstantZeroShape_Rejected) {
  CheckAllowZero({2, 0, 3}, {6, 0}, true, false, "zero dimensions");
}

TEST(QnnUnit_Reshape_OpBuilderTest, AllowZero_DynamicShape_Rejected) {
  CheckAllowZero({1, 3, 4, 4}, {1, 48}, false, false, "Initializer not found");
}

TEST(QnnUnit_Reshape_OpBuilderTest, NearlyEqualScale_SynchronizesOutputParameters) {
  CheckOutputQuantization(std::nextafter(1.0f / 1024.0f, std::numeric_limits<float>::infinity()), 0, false, true);
}

TEST(QnnUnit_Reshape_OpBuilderTest, DifferentScale_PreservesOutputParameters) {
  CheckOutputQuantization(0.1f, 0, false, false);
}

TEST(QnnUnit_Reshape_OpBuilderTest, DifferentZeroPoint_PreservesOutputParameters) {
  CheckOutputQuantization(std::nextafter(1.0f / 1024.0f, std::numeric_limits<float>::infinity()), 1, false, false);
}

TEST(QnnUnit_Reshape_OpBuilderTest, PerChannelUnitDimensionReshape_PreservesOutputEncoding) {
  CheckOutputQuantization(0.1f, 1, true, false);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
