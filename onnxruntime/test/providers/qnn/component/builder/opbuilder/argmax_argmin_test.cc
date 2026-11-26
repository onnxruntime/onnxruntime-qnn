// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

Ort::Status AddArgMaxMin(const std::string& op_type, std::vector<int64_t> input_shape,
                         std::vector<int64_t> output_shape, std::vector<FakeOpAttr> attrs,
                         bool graph_output = false) {
  const qnn::IOpBuilder* builder = qnn::GetOpBuilder(op_type);
  EXPECT_NE(builder, nullptr);
  if (builder == nullptr) {
    return Ort::Status{};
  }

  g_mock_init_reg.clear();
  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  if (graph_output) {
    ctx.output_info.names.push_back("output");
    ctx.output_info.indices["output"] = 0;
  }
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{}, qnn::QnnBackendType::HTP);

  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::move(input_shape));
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::move(output_shape));
  auto node_unit = MakeMockNodeUnit(op_type, {input}, {output}, "argm_node", "", 13, 0, std::move(attrs));
  return builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, /*do_op_validation=*/false);
}

}  // namespace

TEST(QnnUnit_ArgMaxMin_ComponentTest, DefaultsAndGraphOutputInt64ConversionSucceed) {
  for (const char* op_type : {"ArgMax", "ArgMin"}) {
    SCOPED_TRACE(op_type);
    const auto status = AddArgMaxMin(op_type, {1, 3, 4, 4}, {1, 3, 4, 4}, {}, true);
    EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
  }
}

TEST(QnnUnit_ArgMaxMin_ComponentTest, NegativeAxisAndKeepDimsFalseSucceed) {
  for (const char* op_type : {"ArgMax", "ArgMin"}) {
    SCOPED_TRACE(op_type);
    const auto status = AddArgMaxMin(op_type, {1, 3, 4, 4}, {1, 3, 4},
                                     {FakeOpAttr::MakeInt64("axis", -1),
                                      FakeOpAttr::MakeInt64("keepdims", 0)});
    EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
  }
}

TEST(QnnUnit_ArgMaxMin_ComponentTest, SelectLastIndexNonZeroIsRejected) {
  for (const char* op_type : {"ArgMax", "ArgMin"}) {
    SCOPED_TRACE(op_type);
    const auto status = AddArgMaxMin(op_type, {1, 3, 4, 4}, {1, 3, 4, 1},
                                     {FakeOpAttr::MakeInt64("select_last_index", 1)});
    EXPECT_FALSE(status.IsOK());
  }
}

TEST(QnnUnit_ArgMaxMin_ComponentTest, AxisOutsideInputRankIsRejected) {
  for (const char* op_type : {"ArgMax", "ArgMin"}) {
    SCOPED_TRACE(op_type);
    const auto status = AddArgMaxMin(op_type, {1, 3, 4}, {1, 3, 1},
                                     {FakeOpAttr::MakeInt64("axis", 3)});
    EXPECT_FALSE(status.IsOK());
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
