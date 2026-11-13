// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

namespace {

void AddShapeConst(const std::vector<int64_t>& dims) {
  MockInitSpec spec;
  spec.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  spec.dims = {static_cast<int64_t>(dims.size())};
  spec.raw_bytes.resize(dims.size() * sizeof(int64_t));
  std::memcpy(spec.raw_bytes.data(), dims.data(), spec.raw_bytes.size());
  g_mock_init_reg.Add("shape", std::move(spec));
}

// Returns (dims, QNN dtype) of every STATIC tensor the builder added.
std::vector<std::pair<std::vector<uint32_t>, Qnn_DataType_t>> RunExpandStatics(
    MockNodeUnit node_unit) {
  const IOpBuilder* builder = GetOpBuilder("Expand");
  EXPECT_NE(builder, nullptr);
  if (builder == nullptr) {
    return {};
  }

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger,
                                           /*do_op_validation=*/false);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();

  std::vector<std::pair<std::vector<uint32_t>, Qnn_DataType_t>> statics;
  for (const auto& [name, tensor] : wrapper->GetModelTensorsMap()) {
    if (tensor.GetTensorType() == QNN_TENSOR_TYPE_STATIC) {
      statics.emplace_back(tensor.GetTensorDims(), tensor.GetTensorDataType());
    }
  }
  return statics;
}

}  // namespace

// Partial broadcast: [1,8,1] -> [1,8,16] needs 16 ones, not 128.
TEST(QnnUnitExpandComponentTest, MinimalOnesPartialBroadcastFloat) {
  g_mock_init_reg.clear();
  AddShapeConst({1, 8, 16});

  auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                            std::vector<int64_t>{1, 8, 1});
  auto shape = MakeMockIODef("shape", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                             std::vector<int64_t>{3});
  auto output = MakeMockIODef("expand_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 8, 16});
  auto node_unit = MakeMockNodeUnit("Expand", {data, shape}, {output}, "expand_node");

  auto statics = RunExpandStatics(std::move(node_unit));
  ASSERT_EQ(statics.size(), 1u);
  const auto& dims = statics[0].first;
  EXPECT_EQ(dims, (std::vector<uint32_t>{1, 1, 16}));
}

// Quantized partial broadcast: ones are minimal and UFIXED_POINT_8.
TEST(QnnUnitExpandComponentTest, MinimalOnesPartialBroadcastQuantizedUint8) {
  g_mock_init_reg.clear();
  AddShapeConst({1, 8, 16});
  auto scale_vi = g_mock_init_reg.AddScalarFloat("scale", 0.00392157f);
  auto zp_vi = g_mock_init_reg.AddScalarUint8("zp", 0);

  auto data = MakeMockQDQIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                               std::vector<int64_t>{1, 8, 1}, scale_vi, zp_vi);
  auto shape = MakeMockIODef("shape", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                             std::vector<int64_t>{3});
  auto output = MakeMockIODef("expand_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 8, 16});
  auto node_unit = MakeMockQDQNodeUnit("Expand", {data, shape}, {output}, "expand_node");

  auto statics = RunExpandStatics(std::move(node_unit));
  ASSERT_EQ(statics.size(), 1u);
  const auto& [dims, dtype] = statics[0];
  EXPECT_EQ(dims, (std::vector<uint32_t>{1, 1, 16}));
  EXPECT_EQ(dtype, QNN_DATATYPE_UFIXED_POINT_8);
}

// Equal shapes: ones collapse to a single element.
TEST(QnnUnitExpandComponentTest, MinimalOnesNoBroadcastSingleton) {
  g_mock_init_reg.clear();
  AddShapeConst({1, 8, 16});

  auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                            std::vector<int64_t>{1, 8, 16});
  auto shape = MakeMockIODef("shape", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                             std::vector<int64_t>{3});
  auto output = MakeMockIODef("expand_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 8, 16});
  auto node_unit = MakeMockNodeUnit("Expand", {data, shape}, {output}, "expand_node");

  auto statics = RunExpandStatics(std::move(node_unit));
  ASSERT_EQ(statics.size(), 1u);
  const auto& dims = statics[0].first;
  EXPECT_EQ(dims, (std::vector<uint32_t>{1, 1, 1}));
}

// Rank mismatch: fall back to full-size ones.
TEST(QnnUnitExpandComponentTest, FullOnesRankMismatchFallback) {
  g_mock_init_reg.clear();
  AddShapeConst({1, 8, 16});

  auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                            std::vector<int64_t>{8});
  auto shape = MakeMockIODef("shape", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                             std::vector<int64_t>{3});
  auto output = MakeMockIODef("expand_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 8, 16});
  auto node_unit = MakeMockNodeUnit("Expand", {data, shape}, {output}, "expand_node");

  auto statics = RunExpandStatics(std::move(node_unit));
  ASSERT_EQ(statics.size(), 1u);
  const auto& dims = statics[0].first;
  EXPECT_EQ(dims, (std::vector<uint32_t>{1, 8, 16}));
}

// Non-broadcastable input: fall back to full-size ones.
TEST(QnnUnitExpandComponentTest, FullOnesInvalidBroadcastFallback) {
  g_mock_init_reg.clear();
  AddShapeConst({1, 8, 16});

  auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                            std::vector<int64_t>{2, 8, 1});
  auto shape = MakeMockIODef("shape", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                             std::vector<int64_t>{3});
  auto output = MakeMockIODef("expand_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 8, 16});
  auto node_unit = MakeMockNodeUnit("Expand", {data, shape}, {output}, "expand_node");

  auto statics = RunExpandStatics(std::move(node_unit));
  ASSERT_EQ(statics.size(), 1u);
  const auto& dims = statics[0].first;
  EXPECT_EQ(dims, (std::vector<uint32_t>{1, 8, 16}));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
