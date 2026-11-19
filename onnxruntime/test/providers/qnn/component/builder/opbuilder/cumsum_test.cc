// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level partitioning guards for CumSumOpBuilder.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {
namespace {

std::unique_ptr<QnnModelWrapper> MakeStubWrapper(OpBuilderTestContext& ctx) {
  ModelSettings settings{};
  return ctx.CreateWrapper(settings, QnnBackendType::HTP);
}

std::vector<FakeOpAttr> MakeCumSumAttrs(int64_t exclusive, int64_t reverse) {
  return {FakeOpAttr::MakeInt64("exclusive", exclusive), FakeOpAttr::MakeInt64("reverse", reverse)};
}

}  // namespace

TEST(QnnUnit_CumSum_ComponentTest, RejectsDynamicAxis) {
  const IOpBuilder* builder = GetOpBuilder("CumSum");
  ASSERT_NE(builder, nullptr);

  OpBuilderTestContext ctx;
  auto wrapper = MakeStubWrapper(ctx);
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 2});
  auto axis = MakeMockIODef("axis", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, std::vector<int64_t>{});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 2});
  auto node_unit = MakeMockNodeUnit("CumSum", {input, axis}, {output}, "cumsum_dynamic_axis", {}, 14);

  auto status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find("axis as a param"), std::string::npos);
}

TEST(QnnUnit_CumSum_ComponentTest, RejectsUnsupportedExclusiveAndReverse) {
  const IOpBuilder* builder = GetOpBuilder("CumSum");
  ASSERT_NE(builder, nullptr);

  for (const auto& [exclusive, reverse, expected] : {
           std::tuple<int64_t, int64_t, const char*>{1, 0, "exclusive"},
           std::tuple<int64_t, int64_t, const char*>{0, 1, "reverse"},
       }) {
    g_mock_init_reg.clear();
    g_mock_init_reg.AddScalarInt32("axis", 0);
    OpBuilderTestContext ctx;
    SetupMockInitRegistryStubs(ctx);
    auto wrapper = MakeStubWrapper(ctx);
    auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 2});
    auto axis = MakeMockIODef("axis", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, std::vector<int64_t>{});
    auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 2});
    auto node_unit = MakeMockNodeUnit("CumSum", {input, axis}, {output}, "cumsum_unsupported_attr", "", 14, 0,
                                      MakeCumSumAttrs(exclusive, reverse));

    auto status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);
    EXPECT_FALSE(status.IsOK());
    EXPECT_NE(std::string(status.GetErrorMessage()).find(expected), std::string::npos);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
