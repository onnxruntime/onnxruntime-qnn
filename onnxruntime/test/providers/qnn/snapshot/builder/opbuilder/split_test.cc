// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/split_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/snapshot/snapshot.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime::test {
namespace {

std::vector<std::vector<int64_t>> OutputShapes(const SplitSpec& spec) {
  const size_t count = spec.num_outputs > 0 ? static_cast<size_t>(spec.num_outputs) : spec.split.size();
  std::vector<std::vector<int64_t>> shapes(count, spec.shape);
  if (!spec.split.empty()) {
    for (size_t i = 0; i < count; ++i) shapes[i][spec.axis] = spec.split[i];
  } else {
    const int64_t step = (spec.shape[spec.axis] + static_cast<int64_t>(count) - 1) / static_cast<int64_t>(count);
    int64_t remaining = spec.shape[spec.axis];
    for (auto& shape : shapes) {
      shape[spec.axis] = std::min(step, remaining);
      remaining -= shape[spec.axis];
    }
  }
  return shapes;
}

std::vector<FakeOpAttr> SplitAttrs(const SplitSpec& spec) {
  std::vector<FakeOpAttr> attrs{FakeOpAttr::MakeInt64("axis", spec.axis)};
  if (spec.opset < 13) attrs.push_back(FakeOpAttr::MakeInt64s("split", spec.split));
  if (spec.num_outputs > 0) attrs.push_back(FakeOpAttr::MakeInt64("num_outputs", spec.num_outputs));
  return attrs;
}

void RunSplitSnapshot(const SplitSpec& spec) {
  const IOpBuilder* builder = GetOpBuilder("Split");
  ASSERT_NE(builder, nullptr);
  g_mock_init_reg.clear();

  const bool split_is_input = spec.opset >= 13 && !spec.split.empty();
  if (split_is_input) g_mock_init_reg.AddTensorInt64("split", {static_cast<int64_t>(spec.split.size())}, spec.split);

  const OrtValueInfo* scale = nullptr;
  const OrtValueInfo* zero_point = nullptr;
  ONNXTensorElementDataType data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  if (spec.quant_type != SplitQuantType::None) {
    const TestInputDef<float> input(spec.shape, false, SplitInputValues<float>(spec));
    if (spec.quant_type == SplitQuantType::UInt8) {
      const auto qparams = GetTestInputQuantParams<uint8_t>(input);
      scale = g_mock_init_reg.AddScalarFloat("scale", qparams.scale);
      data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      zero_point = g_mock_init_reg.AddScalarUint8("zero_point", qparams.zero_point);
    } else {
      const auto qparams = GetTestInputQuantParams<uint16_t>(input);
      scale = g_mock_init_reg.AddScalarFloat("scale", qparams.scale);
      data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
      zero_point = g_mock_init_reg.AddScalarUint16("zero_point", qparams.zero_point);
    }
  }

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";

  const auto output_shapes = OutputShapes(spec);
  std::vector<std::string> output_names;
  for (size_t i = 0; i < output_shapes.size(); ++i) output_names.push_back("y" + std::to_string(i));
  auto wrapper = MakeSnapshotWrapperHtpJson(ctx, htp, {"input"}, output_names);
  ASSERT_NE(wrapper, nullptr);

  std::vector<OrtNodeUnitIODef> inputs;
  std::vector<OrtNodeUnitIODef> outputs;
  if (spec.quant_type == SplitQuantType::None) {
    inputs.push_back(MakeMockIODef("input", data_type, spec.shape));
    for (size_t i = 0; i < output_shapes.size(); ++i) outputs.push_back(MakeMockIODef(output_names[i], data_type, output_shapes[i]));
  } else {
    inputs.push_back(MakeMockQDQIODef("input", data_type, spec.shape, scale, zero_point));
    for (size_t i = 0; i < output_shapes.size(); ++i) outputs.push_back(MakeMockQDQIODef(output_names[i], data_type, output_shapes[i], scale, zero_point));
  }
  if (split_is_input) inputs.push_back(MakeMockIODef("split", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                                     std::vector<int64_t>{static_cast<int64_t>(spec.split.size())}));

  auto node = spec.quant_type == SplitQuantType::None
                  ? MakeMockNodeUnit("Split", inputs, outputs, "split", "", spec.opset, 0, SplitAttrs(spec))
                  : MakeMockQDQNodeUnit("Split", inputs, outputs, "split", "", spec.opset, 0, SplitAttrs(spec));
  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(true));
  const auto& graph = wrapper->GetQnnJSONGraph().at("graph");
  const auto& nodes = graph.at("nodes");
  ASSERT_EQ(nodes.size(), 1u);
  const auto qnn_node = nodes.begin().value();
  EXPECT_EQ(qnn_node.at("type"), output_shapes.size() == 1 ? "Reshape" : "Split");
  EXPECT_EQ(qnn_node.at("input_names"), nlohmann::json(std::vector<std::string>{"input"}));
  EXPECT_EQ(qnn_node.at("output_names"), nlohmann::json(output_names));
  const auto expected_dtype = spec.quant_type == SplitQuantType::None    ? QNN_DATATYPE_INT_32
                              : spec.quant_type == SplitQuantType::UInt8 ? QNN_DATATYPE_UFIXED_POINT_8
                                                                         : QNN_DATATYPE_UFIXED_POINT_16;
  EXPECT_EQ(graph.at("tensors").at("input").at("dims"), nlohmann::json(spec.shape));
  EXPECT_EQ(graph.at("tensors").at("input").at("data_type"), static_cast<uint32_t>(expected_dtype));
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    const auto& tensor = graph.at("tensors").at(output_names[i]);
    EXPECT_EQ(tensor.at("dims"), nlohmann::json(output_shapes[i]));
    EXPECT_EQ(tensor.at("data_type"), static_cast<uint32_t>(expected_dtype));
  }
  AssertSnapshotJson(*wrapper, spec.name);
}

}  // namespace

class QnnSnapshot_Split_OpBuilderTest : public ::testing::TestWithParam<SplitSpec> {};
TEST_P(QnnSnapshot_Split_OpBuilderTest, Case) { RunSplitSnapshot(GetParam()); }
INSTANTIATE_TEST_SUITE_P(, QnnSnapshot_Split_OpBuilderTest, ::testing::ValuesIn(kSplitSpecs),
                         [](const ::testing::TestParamInfo<SplitSpec>& info) { return std::string(info.param.name); });

}  // namespace onnxruntime::test

#endif
