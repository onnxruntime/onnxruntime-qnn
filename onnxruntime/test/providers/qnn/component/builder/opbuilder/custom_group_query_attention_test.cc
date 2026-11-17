// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "gtest/gtest.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

struct CapturedGqa {
  std::string package;
  std::vector<std::string> inputs;
  std::vector<Qnn_DataType_t> types;
  std::vector<std::vector<uint32_t>> shapes;
  std::vector<std::string> params;
  int32_t window = 0;
  uint32_t smooth = 1;
  float scale = -1.0f;
};

CapturedGqa captured;

Qnn_ErrorHandle_t CaptureGqa(Qnn_BackendHandle_t, Qnn_OpConfig_t config) {
  if (std::string(config.v1.typeName) != "GroupQueryAttention") return QNN_SUCCESS;
  captured.package = config.v1.packageName;
  for (uint32_t i = 0; i < config.v1.numOfInputs; ++i) {
    const auto& tensor = config.v1.inputTensors[i];
    captured.inputs.emplace_back(qnn::GetQnnTensorName(tensor));
    captured.types.push_back(qnn::GetQnnTensorDataType(tensor));
    const auto* dims = qnn::GetQnnTensorDims(tensor);
    const auto rank = qnn::GetQnnTensorRank(tensor);
    captured.shapes.push_back(rank == 0 ? std::vector<uint32_t>{} : std::vector<uint32_t>(dims, dims + rank));
  }
  for (uint32_t i = 0; i < config.v1.numOfParams; ++i) {
    const auto& param = config.v1.params[i];
    captured.params.emplace_back(param.name);
    EXPECT_EQ(param.paramType, QNN_PARAMTYPE_SCALAR);
    if (captured.params.back() == "local_window_size") captured.window = param.scalarParam.int32Value;
    if (captured.params.back() == "smooth_softmax") captured.smooth = param.scalarParam.uint32Value;
    if (captured.params.back() == "scale") captured.scale = param.scalarParam.floatValue;
  }
  return QNN_SUCCESS;
}

MockNodeUnit MakeGqa(bool packed = true, bool rotary = false, bool adapters = false,
                     ONNXTensorElementDataType query_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                     std::vector<FakeOpAttr> extra_attrs = {}) {
  const auto half = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  const auto i32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  auto io = [](const char* name, ONNXTensorElementDataType type, std::vector<int64_t> shape) {
    return MakeMockIODef(name, type, std::move(shape));
  };
  auto absent = MakeMockIODef("", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, std::nullopt);
  std::vector<OrtNodeUnitIODef> inputs{
      io("query", query_type, {1, 2, packed ? 64 : 32}),
      packed ? absent : io("key", half, {1, 2, 16}),
      packed ? absent : io("value", half, {1, 2, 16}),
      io("past_key", half, {1, 2, 32, 8}), io("past_value", half, {1, 2, 32, 8}),
      io("seqlens_k", i32, adapters ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{1}),
      io("total_sequence_length", i32, {1})};
  if (rotary) {
    inputs.push_back(io("cos_cache", half, {64, 4}));
    inputs.push_back(io("sin_cache", half, {64, 4}));
    inputs.push_back(io("position_ids", adapters ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : i32, {1, 2}));
    inputs.push_back(io("attention_bias", half, {1, 1, 2, 64}));
  }
  std::vector<OrtNodeUnitIODef> outputs{
      io("output", half, {1, 2, 32}), io("present_key", half, {1, 2, 32, 8}), io("present_value", half, {1, 2, 32, 8})};
  std::vector<FakeOpAttr> attrs{FakeOpAttr::MakeInt64("num_heads", 4), FakeOpAttr::MakeInt64("kv_num_heads", 2),
                                FakeOpAttr::MakeInt64("do_rotary", rotary ? 1 : 0)};
  for (auto& attr : extra_attrs) attrs.push_back(std::move(attr));
  return MakeMockNodeUnit("GroupQueryAttention", std::move(inputs), std::move(outputs), "gqa", "com.microsoft", 1, 0, std::move(attrs));
}

class QnnUnit_CustomGqa_ComponentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_mock_init_reg.clear();
    SetupMockInitRegistryStubs(ctx);
    ctx.qnn_interface.backendValidateOpConfig = CaptureGqa;
    ctx.backend_manager.BackendType() = qnn::QnnBackendType::HTP;
    settings.use_custom_gqa = true;
    captured = {};
    builder = qnn::GetOpBuilder("CustomGroupQueryAttention");
    ASSERT_NE(builder, nullptr);
  }
  OpBuilderTestContext ctx;
  qnn::ModelSettings settings;
  const qnn::IOpBuilder* builder = nullptr;
};

TEST_F(QnnUnit_CustomGqa_ComponentTest, PackedContractAndDefaults) {
  auto node = MakeGqa();
  auto wrapper = ctx.CreateWrapper(settings);
  auto status = builder->IsOpSupported(*wrapper, node, ctx.ort_logger);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(captured.package, "gqa_op_package");
  EXPECT_EQ(captured.inputs, (std::vector<std::string>{"query", "seqlens_k", "total_sequence_length", "past_key", "past_value"}));
  EXPECT_EQ(captured.params, (std::vector<std::string>{"num_heads", "kv_num_heads", "do_rotary", "scale", "causal", "local_window_size",
                                                       "sliding_window_cache", "smooth_softmax", "softcap", "qk_output", "qk_norm_epsilon"}));
  EXPECT_EQ(captured.window, -1);
  EXPECT_EQ(captured.smooth, 0u);
  EXPECT_FLOAT_EQ(captured.scale, 0.0f);
}

TEST_F(QnnUnit_CustomGqa_ComponentTest, SeparateQkvRotaryBiasAndAdapters) {
  auto node = MakeGqa(false, true, true, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                      {FakeOpAttr::MakeInt64("sliding_window_cache", 1), FakeOpAttr::MakeInt64("local_window_size", 31)});
  auto wrapper = ctx.CreateWrapper(settings);
  auto status = builder->IsOpSupported(*wrapper, node, ctx.ort_logger);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_EQ(captured.inputs.size(), 11u);
  EXPECT_EQ(captured.inputs[3], "key");
  EXPECT_EQ(captured.inputs[4], "value");
  EXPECT_EQ(captured.inputs[10], "attention_bias");
  EXPECT_NE(captured.inputs[1], "seqlens_k");
  EXPECT_EQ(captured.shapes[1], (std::vector<uint32_t>{1}));
  EXPECT_NE(captured.inputs[9], "position_ids");
  EXPECT_EQ(captured.types[9], QNN_DATATYPE_INT_32);
  EXPECT_EQ(captured.shapes[9], (std::vector<uint32_t>{1, 2}));
  EXPECT_EQ(captured.window, 31);
}

TEST_F(QnnUnit_CustomGqa_ComponentTest, RejectsQuantizedInputAndUnsupportedCachePolicy) {
  {
    auto node = MakeGqa(true, false, false, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    auto wrapper = ctx.CreateWrapper(settings);
    EXPECT_FALSE(builder->IsOpSupported(*wrapper, node, ctx.ort_logger).IsOK());
  }
  {
    auto node = MakeGqa(true, false, false, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                        {FakeOpAttr::MakeInt64("local_window_size", 31)});
    auto wrapper = ctx.CreateWrapper(settings);
    EXPECT_FALSE(builder->IsOpSupported(*wrapper, node, ctx.ort_logger).IsOK());
  }
  {
    auto node = MakeGqa();
    auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::CPU);
    EXPECT_FALSE(builder->IsOpSupported(*wrapper, node, ctx.ort_logger).IsOK());
  }
}

TEST_F(QnnUnit_CustomGqa_ComponentTest, PackedPositionIdsKeepTheirShapeAfterCompaction) {
  auto node = MakeGqa(true, true, true);
  auto wrapper = ctx.CreateWrapper(settings);
  auto status = builder->IsOpSupported(*wrapper, node, ctx.ort_logger);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_EQ(captured.inputs.size(), 9u);
  EXPECT_EQ(captured.types[7], QNN_DATATYPE_INT_32);
  EXPECT_EQ(captured.shapes[7], (std::vector<uint32_t>{1, 2}));
  EXPECT_EQ(captured.inputs[8], "attention_bias");
}

TEST_F(QnnUnit_CustomGqa_ComponentTest, SessionDispatchAndNativeIsolationAfterUdoRegistration) {
  // Registering a custom package must not redirect a later native session via the global fusion map.
  qnn::registerUDO("GroupQueryAttention", "gqa_op_package");
  for (bool custom : {true, false}) {
    SCOPED_TRACE(custom);
    captured = {};
    auto mock = MakeGqa();
    FakeGraph graph{{*reinterpret_cast<const FakeNode*>(&mock.Get().GetNode())}, {}, {}, {}};
    InstallFakeGraphApiStubs(ctx.stub_ort_api);
    OrtNodeUnit unit(graph.nodes[0].AsNode(), ctx.stub_ort_api);
    ASSERT_EQ(unit.Inputs()[0].name, "query");
    SetupMockInitRegistryStubs(ctx);
    settings.use_custom_gqa = custom;
    qnn::QnnModelWrapper wrapper(*graph.AsGraph(), ctx.api_ptrs, ctx.ort_logger, *ctx.backend_manager.Get(),
                                 ctx.input_info, ctx.output_info, settings);
    std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_map{{graph.nodes[0].AsNode(), &unit}};
    std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> groups;
    ASSERT_TRUE(qnn::GetQnnNodeGroups(groups, wrapper, node_map, 1, ctx.ort_logger).IsOK());
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0]->Type(), "OrtNodeUnit");
    auto status = groups[0]->IsSupported(wrapper, ctx.ort_logger);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    EXPECT_EQ(captured.package, custom ? "gqa_op_package" : QNN_OP_PACKAGE_NAME_QTI_AISW);
    ASSERT_TRUE(groups[0]->AddToModelBuilder(wrapper, ctx.ort_logger).IsOK());
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
