// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for TileOpBuilder.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>
#include <string>
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

std::unique_ptr<QnnModelWrapper> MakeStubWrapper(OpBuilderTestContext& ctx) {
  ModelSettings settings{};
  return ctx.CreateWrapper(settings);
}

}  // namespace

TEST(QnnUnit_Tile_ComponentTest, Tile_DynamicRepeats_Unsupported) {
  const IOpBuilder* builder = GetOpBuilder("Tile");
  ASSERT_NE(builder, nullptr);

  OpBuilderTestContext ctx;
  auto wrapper = MakeStubWrapper(ctx);

  auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                            std::vector<int64_t>{2, 2});
  auto repeats = MakeMockIODef("repeats", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                               std::vector<int64_t>{2});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{2, 4});
  auto node_unit = MakeMockNodeUnit("Tile", {data, repeats}, {output}, "tile_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger,
                                           /*do_op_validation=*/true);
  EXPECT_FALSE(status.IsOK()) << "Expected rejection for dynamic repeats input";
  EXPECT_NE(std::string(status.GetErrorMessage()).find("dynamic repeats"), std::string::npos);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
