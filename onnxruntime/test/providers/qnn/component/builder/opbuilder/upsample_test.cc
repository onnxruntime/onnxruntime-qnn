// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

std::string CheckUpsample(qnn::QnnBackendType backend, const std::vector<int64_t>& input_shape,
                          const std::vector<int64_t>& output_shape) {
  g_mock_init_reg.clear();
  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  std::vector<float> scale_values(input_shape.size(), 1.0f);
  if (!input_shape.empty() && output_shape.size() == input_shape.size() && input_shape.back() != 0) {
    scale_values.back() = static_cast<float>(output_shape.back()) / static_cast<float>(input_shape.back());
  }
  const std::vector<int64_t> scales_shape{static_cast<int64_t>(scale_values.size())};
  g_mock_init_reg.AddTensorFloat("scales", scales_shape, scale_values);
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{}, backend);
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_shape);
  auto scales = MakeMockIODef("scales", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, scales_shape);
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, output_shape);
  auto node = MakeMockNodeUnit("Upsample", {input, scales}, {output}, "upsample", "", 9);
  auto status = qnn::GetOpBuilder("Upsample")->IsOpSupported(*wrapper, node, ctx.ort_logger);
  return status.IsOK() ? std::string{} : status.GetErrorMessage();
}

}  // namespace

TEST(QnnUnit_Upsample_ComponentTest, ShortInputRank_AllBackendsRejectBeforeIndexing) {
  ASSERT_NE(qnn::GetOpBuilder("Upsample"), nullptr);
  for (auto backend : {qnn::QnnBackendType::CPU, qnn::QnnBackendType::GPU, qnn::QnnBackendType::HTP}) {
    for (size_t rank : {size_t{0}, size_t{1}, size_t{2}}) {
      SCOPED_TRACE("backend=" + std::to_string(static_cast<int>(backend)) + " rank=" + std::to_string(rank));
      auto error = CheckUpsample(backend, std::vector<int64_t>(rank, 1), std::vector<int64_t>(rank, 1));
      ASSERT_FALSE(error.empty());
      EXPECT_NE(error.find("input must have a rank >= 3"), std::string::npos);
    }
  }
}

TEST(QnnUnit_Upsample_ComponentTest, MismatchedOutputRank_AllBackendsRejectBeforeIndexing) {
  for (auto backend : {qnn::QnnBackendType::CPU, qnn::QnnBackendType::GPU, qnn::QnnBackendType::HTP}) {
    for (size_t rank : {size_t{0}, size_t{1}, size_t{2}, size_t{4}}) {
      SCOPED_TRACE("backend=" + std::to_string(static_cast<int>(backend)) + " output rank=" + std::to_string(rank));
      auto error = CheckUpsample(backend, {1, 1, 2}, std::vector<int64_t>(rank, 1));
      ASSERT_FALSE(error.empty());
      EXPECT_NE(error.find("input and output ranks must match"), std::string::npos);
    }
  }
}

TEST(QnnUnit_Upsample_ComponentTest, ValidRanks_AllBackendsSupportSpatialResize) {
  for (auto backend : {qnn::QnnBackendType::CPU, qnn::QnnBackendType::GPU, qnn::QnnBackendType::HTP}) {
    for (size_t rank : {size_t{3}, size_t{4}, size_t{5}}) {
      auto input_shape = std::vector<int64_t>(rank, 2);
      auto output_shape = input_shape;
      output_shape.back() = 4;
      EXPECT_TRUE(CheckUpsample(backend, input_shape, output_shape).empty());
    }
  }
}

TEST(QnnUnit_Upsample_ComponentTest, RankSix_NpuRejectsAndNonNpuSupportCheckSucceeds) {
  const std::vector<int64_t> shape(6, 1);
  auto error = CheckUpsample(qnn::QnnBackendType::HTP, shape, shape);
  ASSERT_FALSE(error.empty());
  EXPECT_NE(error.find("maximum rank is 5 on the NPU"), std::string::npos);
  EXPECT_TRUE(CheckUpsample(qnn::QnnBackendType::CPU, shape, shape).empty());
  EXPECT_TRUE(CheckUpsample(qnn::QnnBackendType::GPU, shape, shape).empty());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
