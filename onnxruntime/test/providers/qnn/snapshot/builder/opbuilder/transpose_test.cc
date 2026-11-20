// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Graph-translation snapshots for the standalone Transpose OpBuilder.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/snapshot/snapshot.h"

namespace onnxruntime {
namespace test {
namespace {

std::vector<FakeOpAttr> MakePermAttr(const std::vector<int64_t>& perm) {
  return {FakeOpAttr::MakeInt64s("perm", perm)};
}

void RunTransposeSnapshot(const std::vector<int64_t>& input_shape,
                          const std::vector<int64_t>& output_shape,
                          const std::vector<int64_t>& perm,
                          ONNXTensorElementDataType data_type,
                          bool qdq,
                          const std::string& golden_basename) {
  const qnn::IOpBuilder* builder = qnn::GetOpBuilder("Transpose");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();
  const OrtValueInfo* scale = nullptr;
  const OrtValueInfo* zero_point = nullptr;
  if (qdq) {
    scale = g_mock_init_reg.AddScalarFloat("scale", 1.0f / 255.0f);
    zero_point = g_mock_init_reg.AddScalarUint8("zero_point", 0);
  }

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";

  auto wrapper = MakeSnapshotWrapperHtpJson(ctx, htp, {"input"}, {"output"});
  ASSERT_NE(wrapper, nullptr) << "Failed to initialize QNN graph for snapshot test";

  auto input = qdq ? MakeMockQDQIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, input_shape, scale, zero_point)
                   : MakeMockIODef("input", data_type, input_shape);
  auto output = qdq ? MakeMockQDQIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, output_shape, scale, zero_point)
                    : MakeMockIODef("output", data_type, output_shape);
  auto node_unit = qdq ? MakeMockQDQNodeUnit("Transpose", {input}, {output}, "transpose", "", 13, 0, MakePermAttr(perm))
                       : MakeMockNodeUnit("Transpose", {input}, {output}, "transpose", "", 13, 0, MakePermAttr(perm));

  const auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true));
  AssertSnapshotJson(*wrapper, golden_basename);
}

}  // namespace

TEST(QnnSnapshot_Transpose_OpBuilderTest, NormalPerm_Fp32) {
  RunTransposeSnapshot({1, 3, 224, 128}, {1, 224, 128, 3}, {0, 2, 3, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, false,
                       "NormalPerm_Fp32");
}

TEST(QnnSnapshot_Transpose_OpBuilderTest, NormalPerm_Int32) {
  RunTransposeSnapshot({1, 3, 224, 128}, {1, 224, 128, 3}, {0, 2, 3, 1},
                       ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, false, "NormalPerm_Int32");
}

TEST(QnnSnapshot_Transpose_OpBuilderTest, NormalPerm_QdqU8) {
  RunTransposeSnapshot({1, 3, 224, 128}, {1, 224, 128, 3}, {0, 2, 3, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, true,
                       "NormalPerm_QdqU8");
}

TEST(QnnSnapshot_Transpose_OpBuilderTest, UnitDimensionPerm_LowersToReshape_Fp32) {
  RunTransposeSnapshot({128, 1, 128}, {1, 128, 128}, {1, 0, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, false,
                       "UnitDimensionPerm_LowersToReshape_Fp32");
}

TEST(QnnSnapshot_Transpose_OpBuilderTest, UnitDimensionPerm_LowersToReshape_QdqU8) {
  RunTransposeSnapshot({128, 1, 128}, {1, 128, 128}, {1, 0, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, true,
                       "UnitDimensionPerm_LowersToReshape_QdqU8");
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
