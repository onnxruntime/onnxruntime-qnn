// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

std::string CheckGroupCount(int64_t groups, bool translate, bool channel_last = false) {
  g_mock_init_reg.clear();
  OpBuilderTestContext ctx;
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{}, qnn::QnnBackendType::CPU);
  const bool nhwc = translate || channel_last;
  const std::vector<int64_t> shape = nhwc ? std::vector<int64_t>{1, 2, 4} : std::vector<int64_t>{1, 4, 2};
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
  auto scale = MakeMockIODef("scale", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{4});
  auto bias = MakeMockIODef("bias", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{4});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
  auto node = MakeMockNodeUnit("GroupNormalization", {input, scale, bias}, {output}, "groupnorm",
                               nhwc ? kMSInternalNHWCDomain : "", 21, 0,
                               {FakeOpAttr::MakeInt64("num_groups", groups)});
  const auto* builder = qnn::GetOpBuilder("GroupNormalization");
  auto status = translate ? builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false)
                          : builder->IsOpSupported(*wrapper, node, ctx.ort_logger);
  return status.IsOK() ? std::string{} : status.GetErrorMessage();
}

}  // namespace

TEST(QnnUnit_GroupNormalization_ComponentTest, IsOpSupported_RejectsInvalidGroupCountsBeforeModulo) {
  ASSERT_NE(qnn::GetOpBuilder("GroupNormalization"), nullptr);
  for (bool channel_last : {false, true}) {
    SCOPED_TRACE(channel_last);
    for (int64_t groups : {int64_t{-1}, int64_t{0}, int64_t{5}, int64_t{1} << 32,
                           (int64_t{1} << 32) + 2, std::numeric_limits<int64_t>::max()}) {
      SCOPED_TRACE(groups);
      auto error = CheckGroupCount(groups, false, channel_last);
      ASSERT_FALSE(error.empty());
      EXPECT_NE(error.find("range [1, num_channels]"), std::string::npos);
    }
  }
}

TEST(QnnUnit_GroupNormalization_ComponentTest, AddToModelBuilder_RejectsInvalidGroupCountsBeforeNarrowing) {
  ASSERT_NE(qnn::GetOpBuilder("GroupNormalization"), nullptr);
  for (int64_t groups : {int64_t{-1}, int64_t{0}, int64_t{1} << 32,
                         (int64_t{1} << 32) + 2, std::numeric_limits<int64_t>::max()}) {
    SCOPED_TRACE(groups);
    auto error = CheckGroupCount(groups, true);
    ASSERT_FALSE(error.empty());
    EXPECT_NE(error.find("outside the supported uint32 range"), std::string::npos);
  }
}

TEST(QnnUnit_GroupNormalization_ComponentTest, IsOpSupported_RejectsNonDivisor) {
  auto error = CheckGroupCount(3, false);
  ASSERT_FALSE(error.empty());
  EXPECT_NE(error.find("divisible by num_groups"), std::string::npos);
}

TEST(QnnUnit_GroupNormalization_ComponentTest, ValidGroupCounts_SupportAndTranslationSucceed) {
  for (int64_t groups : {1, 2, 4}) {
    SCOPED_TRACE(groups);
    EXPECT_TRUE(CheckGroupCount(groups, false).empty());
    EXPECT_TRUE(CheckGroupCount(groups, true).empty());
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
