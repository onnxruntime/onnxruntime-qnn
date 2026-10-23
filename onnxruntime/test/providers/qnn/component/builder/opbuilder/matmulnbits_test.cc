// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <limits>
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

MockNodeUnit MakeMatMulNBitsNode(int64_t K, int64_t N, int64_t bits, int64_t block_size) {
  auto input = MakeMockIODef("A", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                             std::vector<int64_t>{1, 1, 64});
  auto weight = MakeMockIODef("B", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                              std::vector<int64_t>{32, 8, 4});
  auto scales = MakeMockIODef("scales", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{32, 8});
  auto output = MakeMockIODef("Y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 1, 32});

  std::vector<FakeOpAttr> attributes;
  attributes.push_back(FakeOpAttr::MakeInt64("K", K));
  attributes.push_back(FakeOpAttr::MakeInt64("N", N));
  attributes.push_back(FakeOpAttr::MakeInt64("bits", bits));
  attributes.push_back(FakeOpAttr::MakeInt64("block_size", block_size));
  return MakeMockNodeUnit("MatMulNBits", {input, weight, scales}, {output},
                          "matmul_nbits", kMSDomain, 1, 0, std::move(attributes));
}

Ort::Status CheckMatMulNBitsSupport(const IOpBuilder& builder,
                                    int64_t K, int64_t N, int64_t bits, int64_t block_size) {
  OpBuilderTestContext ctx;
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, QnnBackendType::HTP);
  auto node_unit = MakeMatMulNBitsNode(K, N, bits, block_size);
  return builder.IsOpSupported(*wrapper, node_unit, ctx.ort_logger);
}

}  // namespace

TEST(QnnUnit_MatMulNBits_ComponentTest, RejectsInvalidDimensionsBeforeNarrowingOrTranspose) {
  const IOpBuilder* builder = GetOpBuilder("MatMulNBits");
  ASSERT_NE(builder, nullptr);

  const Ort::Status negative_dimensions = CheckMatMulNBitsSupport(*builder, -256, -256, 4, 8);
  EXPECT_FALSE(negative_dimensions.IsOK());
  EXPECT_NE(negative_dimensions.GetErrorMessage().find("K and N must both be positive"), std::string::npos);

  EXPECT_FALSE(CheckMatMulNBitsSupport(
                   *builder,
                   static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1, 32, 4, 8)
                   .IsOK());
  EXPECT_FALSE(CheckMatMulNBitsSupport(
                   *builder,
                   64, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1, 4, 8)
                   .IsOK());
}

TEST(QnnUnit_MatMulNBits_ComponentTest, RejectsZeroBlockSizeBeforeDivision) {
  const IOpBuilder* builder = GetOpBuilder("MatMulNBits");
  ASSERT_NE(builder, nullptr);

  const Ort::Status status = CheckMatMulNBitsSupport(*builder, 64, 32, 4, 0);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("bits and block_size must both be positive"), std::string::npos);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
