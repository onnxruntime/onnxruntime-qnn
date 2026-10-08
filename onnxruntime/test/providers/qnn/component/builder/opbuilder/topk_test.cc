// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <string>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

TEST(QnnUnit_TopK_ComponentTest, RejectsSecondOutputAxisOutsideRank) {
  const IOpBuilder* builder = GetOpBuilder("TopK");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarInt64("k", 1);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                             std::vector<int64_t>{1, 2, 3});
  auto k = MakeMockIODef("k", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                         std::vector<int64_t>{1});
  auto values = MakeMockIODef("values", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 2, 1});
  auto indices = MakeMockIODef("indices", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                               std::vector<int64_t>{1});
  auto node_unit = MakeMockNodeUnit("TopK", {input, k}, {values, indices}, "topk_node", "", 1, 0,
                                    {FakeOpAttr::MakeInt64("axis", 1)});

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger,
                                           /*do_op_validation=*/false);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("axis must be smaller"), std::string::npos);
}

TEST(QnnUnit_TopK_ComponentTest, NormalizesNegativeAxisBeforeValidatingOutputRank) {
  const IOpBuilder* builder = GetOpBuilder("TopK");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarInt64("k", 1);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                             std::vector<int64_t>{1, 2, 3});
  auto k = MakeMockIODef("k", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                         std::vector<int64_t>{1});
  auto values = MakeMockIODef("values", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 2, 1});
  auto indices = MakeMockIODef("indices", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                               std::vector<int64_t>{1});
  auto node_unit = MakeMockNodeUnit("TopK", {input, k}, {values, indices}, "topk_node", "", 1, 0,
                                    {FakeOpAttr::MakeInt64("axis", -2)});

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger,
                                           /*do_op_validation=*/false);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("axis must be smaller"), std::string::npos);
}

}  // namespace test
}  // namespace onnxruntime
