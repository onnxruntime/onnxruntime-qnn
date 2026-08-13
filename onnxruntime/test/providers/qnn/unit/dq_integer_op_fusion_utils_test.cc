// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for PreDequantizePerChannelWeight() in dq_integer_op_fusion_utils.cc.
//
// It reads the bytes QnnModelWrapper::UnpackInitializerData() produced as int8_t / uint8_t. A
// sub-byte initializer arrives one byte per element with the unused high bits masked off, so an
// INT4 -1 would dequantize as 15 unless it is sign-extended first.
//
// Initializer access is mocked through mock_init_registry, so no real ORT graph is needed.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/dq_integer_op_fusion_utils.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/unit/mock_init_registry.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// An IODef naming a plain (non-quantized) initializer, e.g. a fusion's B_scale input.
OrtNodeUnitIODef MakeInitIODef(const std::string& name,
                               ONNXTensorElementDataType type,
                               std::vector<int64_t> shape) {
  OrtNodeUnitIODef io_def;
  io_def.name = name;
  io_def.type = type;
  io_def.shape = std::optional<std::vector<int64_t>>(std::move(shape));
  return io_def;
}

std::vector<float> AsFloat(const std::vector<uint8_t>& bytes) {
  std::vector<float> out(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
  return out;
}

// Masks each value to `bits`, i.e. what UnpackInitializerData() hands over for a sub-byte type.
std::vector<uint8_t> MaskedSubByteBytes(const std::vector<int8_t>& values, int bits) {
  const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size());
  for (int8_t v : values) {
    bytes.push_back(static_cast<uint8_t>(v) & mask);
  }
  return bytes;
}

}  // namespace

TEST(QnnUnit_DqIntegerOpFusionUtilsTest, PreDequantizeInt4_MatchesReferenceDequant) {
  MockInitWrapperFixture fx;
  // [K=4, N=4] weight holding every representable INT4 value; per-channel scales apply along the
  // last axis. Symmetric (no zero point), as w4a16 checkpoints emit.
  constexpr uint32_t kOutChannels = 4;
  const std::vector<int8_t> values{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<float> scales{0.5f, 0.25f, 0.125f, 1.0f};
  const OrtValueInfo* w_vi = g_mock_init_reg.AddTensorInt4As8bit("w", {4, 4}, values);
  g_mock_init_reg.AddTensorFloat("w_scale", {kOutChannels}, scales);
  auto scale_iodef = MakeInitIODef("w_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {kOutChannels});

  // Feed the helper exactly what the fusion feeds it: the masked bytes from the wrapper.
  std::vector<uint8_t> quant_bytes;
  ASSERT_TRUE(fx.wrapper->UnpackInitializerData(w_vi, quant_bytes).IsOK());
  ASSERT_EQ(quant_bytes.size(), values.size());

  std::vector<uint8_t> float_bytes;
  ASSERT_TRUE(qnn::PreDequantizePerChannelWeight(*fx.wrapper, scale_iodef, /*b_zp_iodef=*/nullptr,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, kOutChannels,
                                                 quant_bytes, float_bytes)
                  .IsOK());

  const std::vector<float> got = AsFloat(float_bytes);
  ASSERT_EQ(got.size(), values.size());
  for (size_t i = 0; i < got.size(); ++i) {
    // Without the sign extension the negative half would come out as (q + 16) * scale.
    EXPECT_FLOAT_EQ(got[i], scales[i % kOutChannels] * static_cast<float>(values[i]))
        << "element " << i;
  }
}

TEST(QnnUnit_DqIntegerOpFusionUtilsTest, PreDequantizeInt2_MatchesReferenceDequant) {
  MockInitWrapperFixture fx;
  constexpr uint32_t kOutChannels = 2;
  const std::vector<float> scales{0.5f, 0.25f};
  const std::vector<int8_t> values{-2, -1, 0, 1};
  g_mock_init_reg.AddTensorFloat("w_scale", {kOutChannels}, scales);
  auto scale_iodef = MakeInitIODef("w_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {kOutChannels});

  const std::vector<uint8_t> quant_bytes = MaskedSubByteBytes(values, /*bits=*/2);
  std::vector<uint8_t> float_bytes;
  ASSERT_TRUE(qnn::PreDequantizePerChannelWeight(*fx.wrapper, scale_iodef, /*b_zp_iodef=*/nullptr,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2, kOutChannels,
                                                 quant_bytes, float_bytes)
                  .IsOK());

  const std::vector<float> got = AsFloat(float_bytes);
  ASSERT_EQ(got.size(), values.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i], scales[i % kOutChannels] * static_cast<float>(values[i]))
        << "element " << i;
  }
}

TEST(QnnUnit_DqIntegerOpFusionUtilsTest, PreDequantizeUint4_IsNotSignExtended) {
  // An unsigned sub-byte masked byte already holds the value, so 15 must stay 15, not become -1.
  MockInitWrapperFixture fx;
  constexpr uint32_t kOutChannels = 2;
  const std::vector<float> scales{0.5f, 0.25f};
  const std::vector<uint8_t> values{0, 1, 8, 15};
  g_mock_init_reg.AddTensorFloat("w_scale", {kOutChannels}, scales);
  auto scale_iodef = MakeInitIODef("w_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {kOutChannels});

  std::vector<uint8_t> float_bytes;
  ASSERT_TRUE(qnn::PreDequantizePerChannelWeight(*fx.wrapper, scale_iodef, /*b_zp_iodef=*/nullptr,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4, kOutChannels,
                                                 values, float_bytes)
                  .IsOK());

  const std::vector<float> got = AsFloat(float_bytes);
  ASSERT_EQ(got.size(), values.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i], scales[i % kOutChannels] * static_cast<float>(values[i]))
        << "element " << i;
  }
}

TEST(QnnUnit_DqIntegerOpFusionUtilsTest, PreDequantizeInt8_MatchesReferenceDequant) {
  // The 8-bit path must stay exactly as it was; this is the control for the sub-byte cases above.
  MockInitWrapperFixture fx;
  constexpr uint32_t kOutChannels = 2;
  const std::vector<float> scales{0.5f, 0.25f};
  const std::vector<int8_t> values{-128, -1, 0, 127};
  g_mock_init_reg.AddTensorFloat("w_scale", {kOutChannels}, scales);
  auto scale_iodef = MakeInitIODef("w_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {kOutChannels});

  std::vector<uint8_t> quant_bytes(values.size());
  std::memcpy(quant_bytes.data(), values.data(), values.size());

  std::vector<uint8_t> float_bytes;
  ASSERT_TRUE(qnn::PreDequantizePerChannelWeight(*fx.wrapper, scale_iodef, /*b_zp_iodef=*/nullptr,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, kOutChannels,
                                                 quant_bytes, float_bytes)
                  .IsOK());

  const std::vector<float> got = AsFloat(float_bytes);
  ASSERT_EQ(got.size(), values.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i], scales[i % kOutChannels] * static_cast<float>(values[i]))
        << "element " << i;
  }
}

TEST(QnnUnit_DqIntegerOpFusionUtilsTest, PreDequantizeUint8WithZeroPoint_SubtractsZeroPoint) {
  // Pins the zero-point arithmetic the decoded value feeds into.
  MockInitWrapperFixture fx;
  constexpr uint32_t kOutChannels = 2;
  const std::vector<float> scales{0.5f, 0.25f};
  const std::vector<uint8_t> values{0, 100, 200, 255};
  const std::vector<uint8_t> zps{128, 10};
  g_mock_init_reg.AddTensorFloat("w_scale", {kOutChannels}, scales);
  g_mock_init_reg.AddTensorUint8("w_zp", {kOutChannels}, zps);
  auto scale_iodef = MakeInitIODef("w_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {kOutChannels});
  auto zp_iodef = MakeInitIODef("w_zp", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {kOutChannels});

  std::vector<uint8_t> float_bytes;
  ASSERT_TRUE(qnn::PreDequantizePerChannelWeight(*fx.wrapper, scale_iodef, &zp_iodef,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, kOutChannels,
                                                 values, float_bytes)
                  .IsOK());

  const std::vector<float> got = AsFloat(float_bytes);
  ASSERT_EQ(got.size(), values.size());
  for (size_t i = 0; i < got.size(); ++i) {
    const size_t c = i % kOutChannels;
    EXPECT_FLOAT_EQ(got[i],
                    scales[c] * (static_cast<float>(values[i]) - static_cast<float>(zps[c])))
        << "element " << i;
  }
}

TEST(QnnUnit_DqIntegerOpFusionUtilsTest, PreDequantizeSubByteZeroPoint_FailsClosed) {
  // ONNX requires the zero point to share the weight's type, but ReadZeroPointAsInt32() reads only
  // INT8 / UINT8. An asymmetric sub-byte weight is therefore declined, not silently mis-decoded.
  MockInitWrapperFixture fx;
  constexpr uint32_t kOutChannels = 2;
  g_mock_init_reg.AddTensorFloat("w_scale", {kOutChannels}, {0.5f, 0.25f});
  g_mock_init_reg.AddTensorInt4As8bit("w_zp", {kOutChannels}, {1, -1});
  auto scale_iodef = MakeInitIODef("w_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {kOutChannels});
  auto zp_iodef = MakeInitIODef("w_zp", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {kOutChannels});

  const std::vector<uint8_t> quant_bytes = MaskedSubByteBytes({-1, 2, -3, 4}, /*bits=*/4);
  std::vector<uint8_t> float_bytes;
  EXPECT_FALSE(qnn::PreDequantizePerChannelWeight(*fx.wrapper, scale_iodef, &zp_iodef,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, kOutChannels,
                                                  quant_bytes, float_bytes)
                   .IsOK());
}

TEST(QnnUnit_DqIntegerOpFusionUtilsTest, PreDequantizeWiderThanEightBitWeight_FailsClosed) {
  // quant_bytes.size() is taken as the element count, which only holds for types delivered one
  // byte per element. Anything wider must be rejected rather than dequantized as garbage.
  MockInitWrapperFixture fx;
  constexpr uint32_t kOutChannels = 2;
  g_mock_init_reg.AddTensorFloat("w_scale", {kOutChannels}, {0.5f, 0.25f});
  auto scale_iodef = MakeInitIODef("w_scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {kOutChannels});

  const std::vector<uint8_t> quant_bytes(8, 0);
  for (ONNXTensorElementDataType type : {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}) {
    std::vector<uint8_t> float_bytes;
    EXPECT_FALSE(qnn::PreDequantizePerChannelWeight(*fx.wrapper, scale_iodef,
                                                    /*b_zp_iodef=*/nullptr, type, kOutChannels,
                                                    quant_bytes, float_bytes)
                     .IsOK())
        << "type " << static_cast<int>(type) << " must be rejected";
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
