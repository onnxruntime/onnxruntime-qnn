// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <string>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

// GetCapability must reject the complete decomposition if any added QNN op
// is unsupported, so ORT can fall back before graph compilation.
TEST(QnnUnit_TopK_ComponentTest, ValidatesReverseAndIndexMapping) {
  const IOpBuilder* builder = GetOpBuilder("TopK");
  ASSERT_NE(builder, nullptr);
  for (const char* rejected_op : {QNN_OP_STRIDED_SLICE, QNN_OP_ELEMENT_WISE_SUBTRACT}) {
    SCOPED_TRACE(rejected_op);
    g_mock_init_reg.clear();
    g_mock_init_reg.AddScalarInt64("k", 2);
    OpBuilderTestContext ctx;
    SetupMockInitRegistryStubs(ctx);
    struct ValidationState {
      const char* rejected_op;
      bool rejected = false;
    } state{rejected_op};
    ctx.backend_handle = &state;
    ctx.qnn_interface.backendValidateOpConfig = [](Qnn_BackendHandle_t handle,
                                                   Qnn_OpConfig_t config) -> Qnn_ErrorHandle_t {
      auto& validation = *static_cast<ValidationState*>(handle);
      if (std::strcmp(config.v1.typeName, validation.rejected_op) == 0) {
        validation.rejected = true;
        return QNN_COMMON_ERROR_NOT_SUPPORTED;
      }
      return QNN_SUCCESS;
    };
    auto wrapper = ctx.CreateWrapper(ModelSettings{});
    auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{1, 6});
    auto k = MakeMockIODef("k", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{1});
    auto values = MakeMockIODef("values", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{1, 2});
    auto indices = MakeMockIODef("indices", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{1, 2});
    auto node_unit = MakeMockNodeUnit("TopK", {input, k}, {values, indices}, "topk_node");
    EXPECT_FALSE(builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger).IsOK());
    EXPECT_TRUE(state.rejected);
  }
}

// Validate the early input cast and both INT64 graph-output casts. Rejecting
// any of them must reject the entire TopK decomposition during GetCapability.
TEST(QnnUnit_TopK_ComponentTest, ValidatesInt64Casts) {
  const IOpBuilder* builder = GetOpBuilder("TopK");
  ASSERT_NE(builder, nullptr);
  for (const char* rejected_tensor : {"input", "values", "indices", ""}) {
    SCOPED_TRACE(rejected_tensor);
    g_mock_init_reg.clear();
    g_mock_init_reg.AddScalarInt64("k", 3);
    OpBuilderTestContext ctx;
    SetupMockInitRegistryStubs(ctx);
    ctx.input_info.indices["input"] = 0;
    ctx.output_info.indices["values"] = 0;
    ctx.output_info.indices["indices"] = 1;
    struct ValidationState {
      const char* rejected_tensor;
      bool rejected = false;
      size_t casts = 0;
    } state{rejected_tensor};
    ctx.backend_handle = &state;
    ctx.qnn_interface.backendValidateOpConfig = [](Qnn_BackendHandle_t handle,
                                                   Qnn_OpConfig_t config) -> Qnn_ErrorHandle_t {
      auto& validation = *static_cast<ValidationState*>(handle);
      if (std::strcmp(config.v1.typeName, QNN_OP_CAST) == 0) {
        ++validation.casts;
        if (std::strcmp(config.v1.inputTensors[0].v1.name, validation.rejected_tensor) == 0 ||
            std::strcmp(config.v1.outputTensors[0].v1.name, validation.rejected_tensor) == 0) {
          validation.rejected = true;
          return QNN_COMMON_ERROR_NOT_SUPPORTED;
        }
      }
      return QNN_SUCCESS;
    };
    auto wrapper = ctx.CreateWrapper(ModelSettings{});
    auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{1, 6});
    auto k = MakeMockIODef("k", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{1});
    auto values = MakeMockIODef("values", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{1, 3});
    auto indices = MakeMockIODef("indices", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{1, 3});
    auto node_unit = MakeMockNodeUnit("TopK", {input, k}, {values, indices}, "topk_node");
    const bool expect_rejection = rejected_tensor[0] != '\0';
    EXPECT_EQ(builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger).IsOK(), !expect_rejection);
    EXPECT_EQ(state.rejected, expect_rejection);
    if (!expect_rejection) {
      EXPECT_EQ(state.casts, 3u);
    }
  }
}

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

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
