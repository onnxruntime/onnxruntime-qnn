// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for PoolOpBuilder shared support boundaries.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {
namespace {

std::unique_ptr<QnnModelWrapper> MakeHtpStubWrapper(OpBuilderTestContext& ctx) {
  ModelSettings settings{};
  return ctx.CreateWrapper(settings, QnnBackendType::HTP);
}

void ExpectPoolOpBuilderRejection(std::vector<FakeOpAttr> attrs, const std::string& expected_error) {
  const IOpBuilder* builder = GetOpBuilder("AveragePool");
  ASSERT_NE(builder, nullptr);

  OpBuilderTestContext ctx;
  auto wrapper = MakeHtpStubWrapper(ctx);
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                             std::vector<int64_t>{1, 2, 3, 3});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 2, 1, 1});
  auto node_unit = MakeMockNodeUnit("AveragePool", {input}, {output}, "averagepool", "", 1, 0,
                                    std::move(attrs));
  auto status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find(expected_error), std::string::npos);
}

}  // namespace

TEST(QnnUnit_PoolOpBuilder_ComponentTest, InvalidAutoPad_Unsupported) {
  ExpectPoolOpBuilderRejection(
      {FakeOpAttr::MakeString("auto_pad", "INVALID")},
      "do not support 'auto_pad' value: INVALID");
}

TEST(QnnUnit_PoolOpBuilder_ComponentTest, DilationsNotOne_Unsupported) {
  ExpectPoolOpBuilderRejection(
      {FakeOpAttr::MakeInts("dilations", {2, 1})},
      "only supports dilations 1");
}

TEST(QnnUnit_PoolOpBuilder_ComponentTest, Rank2_Unsupported) {
  const IOpBuilder* builder = GetOpBuilder("AveragePool");
  ASSERT_NE(builder, nullptr);

  OpBuilderTestContext ctx;
  auto wrapper = MakeHtpStubWrapper(ctx);
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                             std::vector<int64_t>{1, 3});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 1});
  auto node_unit = MakeMockNodeUnit("AveragePool", {input}, {output}, "averagepool_rank2");
  auto status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find("QNN Pool only supports rank 3, 4, or 5"),
            std::string::npos);
}

TEST(QnnUnit_PoolOpBuilder_ComponentTest, MultipleOutputs_Unsupported) {
  const IOpBuilder* builder = GetOpBuilder("MaxPool");
  ASSERT_NE(builder, nullptr);

  OpBuilderTestContext ctx;
  auto wrapper = MakeHtpStubWrapper(ctx);
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                             std::vector<int64_t>{1, 2, 3, 3});
  auto values = MakeMockIODef("values", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 2, 1, 1});
  auto indices = MakeMockIODef("indices", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                               std::vector<int64_t>{1, 2, 1, 1});
  auto node_unit = MakeMockNodeUnit("MaxPool", {input}, {values, indices}, "maxpool_indices");
  auto status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find("QNN Pool only supports 1 output"),
            std::string::npos);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
