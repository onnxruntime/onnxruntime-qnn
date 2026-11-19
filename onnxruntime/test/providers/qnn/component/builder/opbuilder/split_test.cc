// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
using namespace onnxruntime;
using namespace onnxruntime::qnn;
namespace onnxruntime::test {
TEST(QnnUnit_Split_ComponentTest, RejectsDynamicSplitInput) {
  const IOpBuilder* builder = GetOpBuilder("Split");
  ASSERT_NE(builder, nullptr);
  OpBuilderTestContext ctx;
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, QnnBackendType::HTP);
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{4, 2});
  auto split = MakeMockIODef("split", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{2});
  auto y0 = MakeMockIODef("y0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{2, 2});
  auto y1 = MakeMockIODef("y1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{2, 2});
  auto node = MakeMockNodeUnit("Split", {input, split}, {y0, y1}, "split_dynamic", "", 13);
  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find("dynamic split"), std::string::npos);
}
}  // namespace onnxruntime::test
#endif
