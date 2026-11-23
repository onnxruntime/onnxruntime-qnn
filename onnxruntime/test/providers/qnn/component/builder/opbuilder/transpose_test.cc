// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Component coverage for Transpose OpBuilder build-time-only behavior.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

struct CapturedTransposeNode {
  std::string type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

std::vector<CapturedTransposeNode> captured_transpose_nodes;

Qnn_ErrorHandle_t CaptureTransposeNode(Qnn_GraphHandle_t, Qnn_OpConfig_t op_config) {
  CapturedTransposeNode node;
  node.type = op_config.v1.typeName;
  for (uint32_t i = 0; i < op_config.v1.numOfInputs; ++i) {
    node.inputs.emplace_back(qnn::GetQnnTensorName(op_config.v1.inputTensors[i]));
  }
  for (uint32_t i = 0; i < op_config.v1.numOfOutputs; ++i) {
    node.outputs.emplace_back(qnn::GetQnnTensorName(op_config.v1.outputTensors[i]));
  }
  captured_transpose_nodes.push_back(std::move(node));
  return QNN_GRAPH_NO_ERROR;
}

std::vector<FakeOpAttr> MakePermAttr(const std::vector<int64_t>& perm) {
  return {FakeOpAttr::MakeInt64s("perm", perm)};
}

void AddStaticFloatTensor(qnn::QnnModelWrapper& wrapper, const std::string& name,
                          const std::vector<uint32_t>& shape, const std::vector<float>& values) {
  auto tensor_shape = shape;
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  ASSERT_TRUE(wrapper.AddTensorWrapper(qnn::QnnTensorWrapper(name, QNN_TENSOR_TYPE_STATIC,
                                                             QNN_DATATYPE_FLOAT_32,
                                                             qnn::QnnQuantParamsWrapper(), std::move(tensor_shape),
                                                             std::move(bytes))));
}

}  // namespace

TEST(QnnUnit_Transpose_ComponentTest, StaticInput_FoldsToPermutedStaticOutput) {
  OpBuilderTestContext ctx;
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  ASSERT_NE(wrapper, nullptr);
  AddStaticFloatTensor(*wrapper, "input", {2, 3}, {0, 1, 2, 3, 4, 5});

  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{2, 3});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 2});
  auto node = MakeMockNodeUnit("Transpose", {input}, {output}, "transpose", "", 13, 0,
                               MakePermAttr({1, 0}));
  const auto* builder = qnn::GetOpBuilder("Transpose");
  ASSERT_NE(builder, nullptr);

  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, /*do_op_validation=*/false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  const auto& folded = wrapper->GetQnnTensorWrapper("output");
  EXPECT_EQ(folded.GetTensorType(), QNN_TENSOR_TYPE_STATIC);
  EXPECT_EQ(folded.GetTensorDims(), (std::vector<uint32_t>{3, 2}));
  const auto& buffer = qnn::GetQnnTensorClientBuf(folded.GetQnnTensor());
  ASSERT_EQ(buffer.dataSize, 6 * sizeof(float));
  const float expected[] = {0, 3, 1, 4, 2, 5};
  EXPECT_EQ(std::memcmp(buffer.data, expected, sizeof(expected)), 0);
}

TEST(QnnUnit_Transpose_ComponentTest, EmptyInputs_IsNoOp) {
  OpBuilderTestContext ctx;
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  ASSERT_NE(wrapper, nullptr);
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 2});
  auto node = MakeMockNodeUnit("Transpose", {}, {output}, "transpose");
  const auto* builder = qnn::GetOpBuilder("Transpose");
  ASSERT_NE(builder, nullptr);

  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, /*do_op_validation=*/false);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist("output"));
}

TEST(QnnUnit_Transpose_ComponentTest, OversizedStaticInput_DeclinesFoldAndEmitsTranspose) {
  OpBuilderTestContext ctx;
  ctx.qnn_interface.tensorCreateGraphTensor = [](Qnn_GraphHandle_t, Qnn_Tensor_t*) {
    return static_cast<Qnn_ErrorHandle_t>(QNN_TENSOR_NO_ERROR);
  };
  ctx.qnn_interface.graphAddNode = CaptureTransposeNode;
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  ASSERT_NE(wrapper, nullptr);
  qnn::utils::UniqueNameGenerator().Reset();

  // The fold budget is 1 MiB. A valid FP32 tensor above it must fall back
  // to a runtime Transpose rather than producing a folded STATIC output.
  AddStaticFloatTensor(*wrapper, "input", {513, 513}, std::vector<float>(513 * 513, 1.0f));
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                             std::vector<int64_t>{513, 513});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{513, 513});
  auto node = MakeMockNodeUnit("Transpose", {input}, {output}, "transpose", "", 13, 0,
                               MakePermAttr({1, 0}));
  const auto* builder = qnn::GetOpBuilder("Transpose");
  ASSERT_NE(builder, nullptr);
  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  const auto& output_tensor = wrapper->GetQnnTensorWrapper("output");
  EXPECT_EQ(output_tensor.GetTensorType(), QNN_TENSOR_TYPE_NATIVE);
  EXPECT_EQ(output_tensor.GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(output_tensor.GetTensorDims(), (std::vector<uint32_t>{513, 513}));

  captured_transpose_nodes.clear();
  ASSERT_TRUE(wrapper->ComposeQnnGraph());
  ASSERT_EQ(captured_transpose_nodes.size(), 1u);
  EXPECT_EQ(captured_transpose_nodes[0].type, QNN_OP_TRANSPOSE);
  EXPECT_EQ(captured_transpose_nodes[0].inputs, (std::vector<std::string>{"input"}));
  EXPECT_EQ(captured_transpose_nodes[0].outputs, (std::vector<std::string>{"output"}));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
