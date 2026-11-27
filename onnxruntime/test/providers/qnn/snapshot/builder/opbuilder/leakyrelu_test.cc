// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/leakyrelu_test_utils.h"
#include "test/providers/qnn/snapshot/snapshot.h"

namespace onnxruntime {
namespace test {

class QnnSnapshot_LeakyRelu_OpBuilderTest : public ::testing::TestWithParam<LeakyReluSpec> {};

TEST_P(QnnSnapshot_LeakyRelu_OpBuilderTest, Case) {
  const auto& spec = GetParam();
  OpBuilderTestContext ctx;
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "QNN HTP backend unavailable";
  auto node = MakeLeakyReluNode(spec);
  SetupMockInitRegistryStubs(ctx);
  auto wrapper = MakeSnapshotWrapperHtpJson(ctx, htp, {"data"}, {"output"});
  ASSERT_NE(wrapper, nullptr);
  const auto* builder = qnn::GetOpBuilder("LeakyRelu");
  ASSERT_NE(builder, nullptr);
  auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(true));

  const auto& graph = wrapper->GetQnnJSONGraph().at("graph");
  const auto& nodes = graph.at("nodes");
  ASSERT_EQ(nodes.size(), 1u);
  const auto node_it = nodes.begin();
  const auto& prelu = node_it.value();
  EXPECT_EQ(prelu.at("type"), "Prelu");
  ASSERT_EQ(prelu.at("input_names").size(), 2u);
  EXPECT_EQ(prelu.at("input_names")[0], "data");
  EXPECT_EQ(prelu.at("output_names"), nlohmann::json::array({"output"}));
  EXPECT_TRUE(prelu.at("scalar_params").empty());
  EXPECT_TRUE(prelu.at("tensor_params").empty());
  const auto alpha_name = prelu.at("input_names")[1].get<std::string>();
  const auto& alpha = wrapper->GetQnnTensorWrapper(alpha_name);
  EXPECT_EQ(alpha.GetTensorType(), QNN_TENSOR_TYPE_STATIC);
  EXPECT_EQ(alpha.GetTensorDims(), (std::vector<uint32_t>{1}));
  const bool quantized = spec.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  const auto dtype = quantized ? QNN_DATATYPE_UFIXED_POINT_8 : QNN_DATATYPE_FLOAT_16;
  EXPECT_EQ(alpha.GetTensorDataType(), dtype);
  for (const auto* name : {"data", "output"}) {
    const auto& tensor = wrapper->GetQnnTensorWrapper(name);
    EXPECT_EQ(tensor.GetTensorDataType(), dtype);
    EXPECT_EQ(tensor.GetTensorDims(), (std::vector<uint32_t>{1, 2, 3}));
    if (quantized) {
      const bool input = std::string(name) == "data";
      const auto& params = qnn::GetQnnTensorQParams(tensor.GetQnnTensor());
      EXPECT_EQ(params.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
      EXPECT_FLOAT_EQ(params.scaleOffsetEncoding.scale, input ? kLeakyReluInputScale : kLeakyReluOutputScale);
      EXPECT_EQ(params.scaleOffsetEncoding.offset, input ? -kLeakyReluInputZeroPoint : -kLeakyReluOutputZeroPoint);
    }
  }
  const auto& params = qnn::GetQnnTensorQParams(alpha.GetQnnTensor());
  if (quantized) {
    EXPECT_EQ(params.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
    EXPECT_FLOAT_EQ(params.scaleOffsetEncoding.scale, spec.alpha / 255.0f);
    EXPECT_EQ(params.scaleOffsetEncoding.offset, 0);
  } else {
    EXPECT_EQ(params.encodingDefinition, QNN_DEFINITION_UNDEFINED);
  }
  AssertSnapshotJson(*wrapper, spec.name);
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_LeakyRelu_OpBuilderTest, ::testing::ValuesIn(kLeakyReluSpecs),
    [](const ::testing::TestParamInfo<LeakyReluSpec>& info) { return std::string(info.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif
