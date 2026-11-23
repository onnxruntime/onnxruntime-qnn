// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Regression coverage for int64 graph-output translation by Transpose OpBuilder.

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

void RunInt64GraphOutputTest(const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& output_shape,
                             const std::vector<int64_t>& perm) {
  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ctx.qnn_interface.tensorCreateGraphTensor = [](Qnn_GraphHandle_t, Qnn_Tensor_t*) {
    return static_cast<Qnn_ErrorHandle_t>(QNN_TENSOR_NO_ERROR);
  };
  ctx.qnn_interface.graphAddNode = CaptureTransposeNode;
  ctx.input_info.names.push_back("input");
  ctx.input_info.indices["input"] = 0;
  ctx.output_info.names.push_back("output");
  ctx.output_info.indices["output"] = 0;
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});
  ASSERT_NE(wrapper, nullptr);
  qnn::utils::UniqueNameGenerator().Reset();

  // A legal ONNX Transpose preserves dtype. Let BaseOpBuilder create the
  // int64-to-int32 input Cast rather than manufacturing its internal name.
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                             input_shape);
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                              output_shape);
  auto node = MakeMockNodeUnit("Transpose", {input}, {output}, "transpose", "", 13, 0,
                               MakePermAttr(perm));
  const auto* builder = qnn::GetOpBuilder("Transpose");
  ASSERT_NE(builder, nullptr);

  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, /*do_op_validation=*/false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  const auto& tensors = wrapper->GetModelTensorsMap();
  const auto input_it = tensors.find("input");
  ASSERT_NE(input_it, tensors.end());
  EXPECT_EQ(input_it->second.GetTensorType(), QNN_TENSOR_TYPE_APP_WRITE);
  EXPECT_EQ(input_it->second.GetTensorDataType(), QNN_DATATYPE_INT_64);
  EXPECT_EQ(input_it->second.GetTensorDims(), (std::vector<uint32_t>(input_shape.begin(), input_shape.end())));
  const auto cast_input_it = tensors.find("input_cast_int32");
  ASSERT_NE(cast_input_it, tensors.end());
  EXPECT_EQ(cast_input_it->second.GetTensorType(), QNN_TENSOR_TYPE_NATIVE);
  EXPECT_EQ(cast_input_it->second.GetTensorDataType(), QNN_DATATYPE_INT_32);
  EXPECT_EQ(cast_input_it->second.GetTensorDims(), (std::vector<uint32_t>(input_shape.begin(), input_shape.end())));
  const auto output_it = tensors.find("output");
  ASSERT_NE(output_it, tensors.end());
  EXPECT_EQ(output_it->second.GetTensorType(), QNN_TENSOR_TYPE_APP_READ);
  EXPECT_EQ(output_it->second.GetTensorDataType(), QNN_DATATYPE_INT_64);
  EXPECT_EQ(output_it->second.GetTensorDims(), (std::vector<uint32_t>(output_shape.begin(), output_shape.end())));
  const auto intermediate_it = tensors.find("output_cast_int64");
  ASSERT_NE(intermediate_it, tensors.end());
  EXPECT_EQ(intermediate_it->second.GetTensorType(), QNN_TENSOR_TYPE_NATIVE);
  EXPECT_EQ(intermediate_it->second.GetTensorDataType(), QNN_DATATYPE_INT_32);
  EXPECT_EQ(intermediate_it->second.GetTensorDims(), (std::vector<uint32_t>(output_shape.begin(), output_shape.end())));

  captured_transpose_nodes.clear();
  ASSERT_TRUE(wrapper->ComposeQnnGraph());
  // Exactly Cast -> Transpose -> Cast excludes Reshape and proves both input
  // narrowing and output restoration are connected to the Transpose.
  ASSERT_EQ(captured_transpose_nodes.size(), 3u);
  EXPECT_EQ(captured_transpose_nodes[0].type, QNN_OP_CAST);
  EXPECT_EQ(captured_transpose_nodes[0].inputs, (std::vector<std::string>{"input"}));
  EXPECT_EQ(captured_transpose_nodes[0].outputs, (std::vector<std::string>{"input_cast_int32"}));
  EXPECT_EQ(captured_transpose_nodes[1].type, QNN_OP_TRANSPOSE);
  EXPECT_EQ(captured_transpose_nodes[1].inputs, captured_transpose_nodes[0].outputs);
  EXPECT_EQ(captured_transpose_nodes[1].outputs, (std::vector<std::string>{"output_cast_int64"}));
  EXPECT_EQ(captured_transpose_nodes[2].type, QNN_OP_CAST);
  EXPECT_EQ(captured_transpose_nodes[2].inputs, captured_transpose_nodes[1].outputs);
  EXPECT_EQ(captured_transpose_nodes[2].outputs, (std::vector<std::string>{"output"}));
}

}  // namespace

TEST(QnnUnit_TransposeInt64_ComponentTest, NormalPerm_PreservesOutputTypeShapeAndConnections) {
  RunInt64GraphOutputTest({2, 3}, {3, 2}, {1, 0});
}

TEST(QnnUnit_TransposeInt64_ComponentTest, UnitDimensionPerm_EmitsConnectedCastsInsteadOfReshape) {
  RunInt64GraphOutputTest({128, 1, 128}, {1, 128, 128}, {1, 0, 2});
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
