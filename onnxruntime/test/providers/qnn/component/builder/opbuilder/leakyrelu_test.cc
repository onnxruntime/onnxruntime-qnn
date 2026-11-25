// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/leakyrelu_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

// Common alpha construction: stub backend, no HTP validation or session.
void CheckAlpha(const LeakyReluSpec& spec, bool explicit_alpha = true) {
  OpBuilderTestContext ctx;
  auto node = MakeLeakyReluNode(spec, explicit_alpha);
  SetupMockInitRegistryStubs(ctx);
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  const auto* builder = qnn::GetOpBuilder("LeakyRelu");
  ASSERT_NE(builder, nullptr);
  auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  const qnn::QnnTensorWrapper* alpha = nullptr;
  for (const auto& entry : wrapper->GetModelTensorsMap()) {
    if (entry.second.GetTensorType() == QNN_TENSOR_TYPE_STATIC) {
      ASSERT_EQ(alpha, nullptr) << "Expected exactly one static alpha tensor";
      alpha = &entry.second;
    }
  }
  ASSERT_NE(alpha, nullptr);
  EXPECT_EQ(alpha->GetTensorDims(), (std::vector<uint32_t>{1}));
  const auto& buffer = qnn::GetQnnTensorClientBuf(alpha->GetQnnTensor());
  ASSERT_NE(buffer.data, nullptr);
  const auto& quant = qnn::GetQnnTensorQParams(alpha->GetQnnTensor());
  const float expected_alpha = explicit_alpha ? spec.alpha : 0.01f;
  if (spec.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    EXPECT_EQ(alpha->GetTensorDataType(), QNN_DATATYPE_UFIXED_POINT_8);
    EXPECT_EQ(quant.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
    EXPECT_FLOAT_EQ(quant.scaleOffsetEncoding.scale, expected_alpha / 255.0f);
    EXPECT_EQ(quant.scaleOffsetEncoding.offset, 0);
    ASSERT_EQ(buffer.dataSize, 1u);
    EXPECT_EQ(*static_cast<const uint8_t*>(buffer.data), 255u);
  } else if (spec.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    EXPECT_EQ(alpha->GetTensorDataType(), QNN_DATATYPE_FLOAT_16);
    EXPECT_EQ(quant.encodingDefinition, QNN_DEFINITION_UNDEFINED);
    ASSERT_EQ(buffer.dataSize, sizeof(uint16_t));
    uint16_t bits;
    std::memcpy(&bits, buffer.data, sizeof(bits));
    EXPECT_EQ(bits, Ort::Float16_t(expected_alpha).val);
  } else {
    EXPECT_EQ(alpha->GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
    EXPECT_EQ(quant.encodingDefinition, QNN_DEFINITION_UNDEFINED);
    ASSERT_EQ(buffer.dataSize, sizeof(float));
    float value;
    std::memcpy(&value, buffer.data, sizeof(value));
    EXPECT_FLOAT_EQ(value, expected_alpha);
  }
}

}  // namespace

TEST(QnnUnit_LeakyRelu_ComponentTest, QuantizedAlpha) {
  CheckAlpha(kLeakyReluSpecs[0]);
}

TEST(QnnUnit_LeakyRelu_ComponentTest, FP16Alpha) {
  CheckAlpha(kLeakyReluSpecs[2]);
}

TEST(QnnUnit_LeakyRelu_ComponentTest, DefaultAlpha) {
  auto spec = kLeakyReluSpecs[2];
  spec.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  CheckAlpha(spec, false);
}

}  // namespace test
}  // namespace onnxruntime

#endif
