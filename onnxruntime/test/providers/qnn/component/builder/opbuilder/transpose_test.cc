// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Component coverage for Transpose OpBuilder build-time-only behavior.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

std::vector<FakeOpAttr> MakePermAttr(const std::vector<int64_t>& perm) {
  return {FakeOpAttr::MakeInt64s("perm", perm)};
}

void AddStaticFloatTensor(qnn::QnnModelWrapper& wrapper, const std::string& name,
                          const std::vector<uint32_t>& shape, const std::vector<float>& values) {
  auto tensor_shape = shape;
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  ASSERT_TRUE(wrapper.AddTensorWrapper(qnn::QnnTensorWrapper(name, QNN_TENSOR_TYPE_STATIC,
                                                             QNN_DATATYPE_FLOAT_32,
                                                             qnn::QnnQuantParamsWrapper(), std::move(tensor_shape),
                                                             std::move(bytes))));
}

}  // namespace

TEST(QnnUnit_Transpose_ComponentTest, StaticInput_FoldsToPermutedStaticOutput) {
  OpBuilderTestContext ctx;
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  ASSERT_NE(wrapper, nullptr);
  AddStaticFloatTensor(*wrapper, "input", {2, 3}, {0, 1, 2, 3, 4, 5});

  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{2, 3});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 2});
  auto node = MakeMockNodeUnit("Transpose", {input}, {output}, "transpose", "", 13, 0,
                               MakePermAttr({1, 0}));
  const auto* builder = qnn::GetOpBuilder("Transpose");
  ASSERT_NE(builder, nullptr);

  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, /*do_op_validation=*/false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  const auto& folded = wrapper->GetQnnTensorWrapper("output");
  EXPECT_EQ(folded.GetTensorType(), QNN_TENSOR_TYPE_STATIC);
  EXPECT_EQ(folded.GetTensorDims(), (std::vector<uint32_t>{3, 2}));
  const auto& buffer = qnn::GetQnnTensorClientBuf(folded.GetQnnTensor());
  ASSERT_EQ(buffer.dataSize, 6 * sizeof(float));
  const float expected[] = {0, 3, 1, 4, 2, 5};
  EXPECT_EQ(std::memcmp(buffer.data, expected, sizeof(expected)), 0);
}

TEST(QnnUnit_Transpose_ComponentTest, EmptyInputs_IsNoOp) {
  OpBuilderTestContext ctx;
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  ASSERT_NE(wrapper, nullptr);
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 2});
  auto node = MakeMockNodeUnit("Transpose", {}, {output}, "transpose");
  const auto* builder = qnn::GetOpBuilder("Transpose");
  ASSERT_NE(builder, nullptr);

  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, /*do_op_validation=*/false);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist("output"));
}

TEST(QnnUnit_Transpose_ComponentTest, Int64GraphOutput_EmitsCastInsteadOfReshape) {
  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ctx.output_info.names.push_back("output");
  ctx.output_info.indices["output"] = 0;
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  ASSERT_NE(wrapper, nullptr);
  qnn::utils::UniqueNameGenerator().Reset();

  auto input = MakeMockIODef("input_cast_int32", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                             std::vector<int64_t>{128, 1, 128});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                              std::vector<int64_t>{1, 128, 128});
  auto node = MakeMockNodeUnit("Transpose", {input}, {output}, "transpose", "", 13, 0,
                               MakePermAttr({1, 0, 2}));
  const auto* builder = qnn::GetOpBuilder("Transpose");
  ASSERT_NE(builder, nullptr);

  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, /*do_op_validation=*/false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  const auto& tensors = wrapper->GetModelTensorsMap();
  const auto output_it = tensors.find("output");
  ASSERT_NE(output_it, tensors.end());
  EXPECT_EQ(output_it->second.GetTensorType(), QNN_TENSOR_TYPE_APP_READ);
  EXPECT_EQ(output_it->second.GetTensorDataType(), QNN_DATATYPE_INT_32);
  EXPECT_NE(tensors.find("output_cast_int64"), tensors.end());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
