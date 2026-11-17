// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <optional>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

std::string CheckGru(const std::vector<int64_t>& input_shape, const std::string& direction,
                     const std::vector<int64_t>& output_shape, bool translate,
                     const std::optional<std::vector<int64_t>>& final_state_shape = std::nullopt) {
  g_mock_init_reg.clear();
  OpBuilderTestContext ctx;
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{}, qnn::QnnBackendType::CPU);
  const int64_t directions = direction == "bidirectional" ? 2 : 1;
  auto x = MakeMockIODef("X", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_shape);
  auto w = MakeMockIODef("W", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{directions, 3, 1});
  auto r = MakeMockIODef("R", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{directions, 3, 1});
  auto y = MakeMockIODef("Y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, output_shape);
  auto yh = MakeMockIODef("Y_h", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                          final_state_shape.value_or(std::vector<int64_t>{directions, 1, 1}));
  auto node = MakeMockNodeUnit("GRU", {x, w, r}, {y, yh}, "gru", "", 14, 0,
                               {FakeOpAttr::MakeInt64("hidden_size", 1), FakeOpAttr::MakeString("direction", direction)});
  const auto* builder = qnn::GetOpBuilder("GRU");
  auto status = translate ? builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false)
                          : builder->IsOpSupported(*wrapper, node, ctx.ort_logger);
  return status.IsOK() ? std::string{} : status.GetErrorMessage();
}

}  // namespace

TEST(QnnUnit_GRU_ComponentTest, InvalidInputRank_SupportAndTranslationRejectBeforeIndexing) {
  ASSERT_NE(qnn::GetOpBuilder("GRU"), nullptr);
  for (const std::string direction : {"forward", "reverse", "bidirectional"}) {
    for (size_t rank : {size_t{0}, size_t{1}, size_t{2}, size_t{4}}) {
      for (bool translate : {false, true}) {
        SCOPED_TRACE(direction + " rank=" + std::to_string(rank) + " translate=" + std::to_string(translate));
        auto error = CheckGru(std::vector<int64_t>(rank, 1), direction, {1, 1, 1, 1}, translate);
        ASSERT_FALSE(error.empty());
        EXPECT_NE(error.find("GRU input X must be rank 3"), std::string::npos);
      }
    }
  }
}

TEST(QnnUnit_GRU_ComponentTest, BidirectionalShortOutputRank_RejectsBeforeConcatAxisSubtraction) {
  for (size_t rank : {size_t{0}, size_t{1}, size_t{2}}) {
    SCOPED_TRACE(rank);
    auto error = CheckGru({1, 1, 1}, "bidirectional", std::vector<int64_t>(rank, 1), true);
    ASSERT_FALSE(error.empty());
    EXPECT_NE(error.find("Concat output must have rank >= 3"), std::string::npos);
  }
}

TEST(QnnUnit_GRU_ComponentTest, BidirectionalShortFinalStateRank_RejectsBeforeConcatAxisSubtraction) {
  for (size_t rank : {size_t{0}, size_t{1}, size_t{2}}) {
    SCOPED_TRACE(rank);
    auto error = CheckGru({1, 1, 1}, "bidirectional", {1, 2, 1, 1}, true, std::vector<int64_t>(rank, 1));
    ASSERT_FALSE(error.empty());
    EXPECT_NE(error.find("Concat output must have rank >= 3"), std::string::npos);
  }
}

TEST(QnnUnit_GRU_ComponentTest, ValidInputRank_AllDirectionsSupportAndTranslationSucceed) {
  for (const std::string direction : {"forward", "reverse", "bidirectional"}) {
    SCOPED_TRACE(direction);
    const int64_t directions = direction == "bidirectional" ? 2 : 1;
    EXPECT_TRUE(CheckGru({1, 1, 1}, direction, {1, directions, 1, 1}, false).empty());
    EXPECT_TRUE(CheckGru({1, 1, 1}, direction, {1, directions, 1, 1}, true).empty());
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
