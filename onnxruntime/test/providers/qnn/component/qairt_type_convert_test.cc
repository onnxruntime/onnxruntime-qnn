// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if defined(USE_QAIRT_API) && QNN_EP_INTERNAL_SYMBOL_ACCESS
#include "core/providers/qnn/runtime/qairt/qairt_type_convert.h"
#include "core/providers/qnn/runtime/qairt/qairt_profiling_manager.h"
#include "core/providers/qnn/runtime/qairt/qairt_backend_manager.h"
#include "core/providers/qnn/runtime/qairt/qairt_backend_configuration.h"
#include "core/providers/qnn/runtime/qairt/qairt_graph_emitter.h"

namespace onnxruntime::test {
namespace {

class QairtConversionTests : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef _WIN32
    api_ = std::make_unique<qairt::Api>("QairtHtp.dll");
#else
    api_ = std::make_unique<qairt::Api>("libQairtHtp.so");
#endif
    tensor_ = api_->make<qairt::Tensor>();
    tensor_.setDimensions({2, 3});
    params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  }
  Ort::Status Convert() { return qnn::qairt_convert::ApplyQuantizeParams(params_, *api_, tensor_); }
  std::unique_ptr<qairt::Api> api_;
  qairt::Tensor tensor_;
  Qnn_QuantizeParams_t params_ = QNN_QUANTIZE_PARAMS_INIT;
};

TEST_F(QairtConversionTests, AxisEncodingRejectsMissingScalesAndInvalidAxis) {
  params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  auto& axis = params_.axisScaleOffsetEncoding;
  axis.axis = 1;
  axis.numScaleOffsets = 3;
  axis.scaleOffset = nullptr;
  EXPECT_FALSE(Convert().IsOK());
  Qnn_ScaleOffset_t scales[] = {{0.5f, -1}, {1.0f, -2}, {2.0f, -3}};
  axis.scaleOffset = scales;
  ASSERT_TRUE(Convert().IsOK());
  const auto& converted = tensor_.getQuantizeParams().getAxisScaleOffsetEncoding();
  EXPECT_EQ(converted.getAxis(), 1);
  ASSERT_EQ(converted.getScaleOffsets().size(), 3u);
  EXPECT_FLOAT_EQ(converted.getScaleOffsets()[2].getScale(), 2.0f);
  EXPECT_EQ(converted.getScaleOffsets()[2].getOffset(), -3);
  axis.axis = -1;
  EXPECT_FALSE(Convert().IsOK());
  axis.axis = 2;
  EXPECT_FALSE(Convert().IsOK());
  axis.axis = 1;
  axis.numScaleOffsets = 2;
  EXPECT_FALSE(Convert().IsOK());
}

TEST_F(QairtConversionTests, BwAxisEncodingAllowsAbsentOffsetsButRequiresScales) {
  params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
  auto& axis = params_.bwAxisScaleOffsetEncoding;
  axis.axis = 1;
  axis.bitwidth = 8;
  axis.numElements = 3;
  axis.scales = nullptr;
  EXPECT_FALSE(Convert().IsOK());
  float scales[] = {0.5f, 1.0f, 2.0f};
  axis.scales = scales;
  axis.offsets = nullptr;
  ASSERT_TRUE(Convert().IsOK());
  const auto& converted = tensor_.getQuantizeParams().getBwAxisScaleOffsetEncoding();
  EXPECT_EQ(converted.getScales(), (std::vector<float>{0.5f, 1.0f, 2.0f}));
  EXPECT_EQ(converted.getOffsets(), (std::vector<int32_t>{0, 0, 0}));
  axis.numElements = 0;
  EXPECT_FALSE(Convert().IsOK());
}

TEST_F(QairtConversionTests, ScaleOffsetAndUnsupportedEncoding) {
  params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  params_.scaleOffsetEncoding = {0.25f, -7};
  ASSERT_TRUE(Convert().IsOK());
  EXPECT_FLOAT_EQ(tensor_.getQuantizeParams().getScaleOffsetEncoding().getScale(), 0.25f);
  EXPECT_EQ(tensor_.getQuantizeParams().getScaleOffsetEncoding().getOffset(), -7);
  params_.quantizationEncoding = static_cast<Qnn_QuantizationEncoding_t>(0x7ffffffe);
  EXPECT_FALSE(Convert().IsOK());
}

TEST_F(QairtConversionTests, UnsupportedEncodingsFailWithoutAccessingPayload) {
  for (auto encoding : {QNN_QUANTIZATION_ENCODING_VECTOR, QNN_QUANTIZATION_ENCODING_FLOAT_BLOCK,
                        QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET_MAPPED,
                        QNN_QUANTIZATION_ENCODING_BW_BLOCK_MAPPED,
                        QNN_QUANTIZATION_ENCODING_BW_BLOCKWISE_EXPANSION_MAPPED,
                        QNN_QUANTIZATION_ENCODING_MICROSCALING, QNN_QUANTIZATION_ENCODING_ARRAY_OF}) {
    SCOPED_TRACE(static_cast<int>(encoding));
    params_.quantizationEncoding = encoding;
    EXPECT_FALSE(Convert().IsOK());
  }
}

TEST_F(QairtConversionTests, BitwidthScaleOffsetPreservesValues) {
  params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  params_.bwScaleOffsetEncoding.bitwidth = 4;
  params_.bwScaleOffsetEncoding.scale = 0.125f;
  params_.bwScaleOffsetEncoding.offset = -3;
  ASSERT_TRUE(Convert().IsOK());
  const auto& converted = tensor_.getQuantizeParams().getBwScaleOffsetEncoding();
  EXPECT_EQ(converted.getBitwidth(), 4u);
  EXPECT_FLOAT_EQ(converted.getScale(), 0.125f);
  EXPECT_EQ(converted.getOffset(), -3);
}

TEST_F(QairtConversionTests, BlockEncodingUsesCeilingBlockCountAndRejectsMissingStorage) {
  params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCK;
  uint32_t blocks[] = {2, 2};
  Qnn_ScaleOffset_t scales[] = {{0.25f, -1}, {0.5f, -2}};
  params_.blockEncoding = {blocks, scales};
  ASSERT_TRUE(Convert().IsOK());
  const auto& converted = tensor_.getQuantizeParams().getBlockEncoding();
  EXPECT_EQ(converted.getBlockSizes(), (std::vector<uint32_t>{2, 2}));
  ASSERT_EQ(converted.getScaleOffsets().size(), 2u);
  EXPECT_FLOAT_EQ(converted.getScaleOffsets()[1].getScale(), 0.5f);
  EXPECT_EQ(converted.getScaleOffsets()[1].getOffset(), -2);
  params_.blockEncoding.scaleOffset = nullptr;
  EXPECT_FALSE(Convert().IsOK());
  params_.blockEncoding.scaleOffset = scales;
  blocks[0] = 0;
  EXPECT_FALSE(Convert().IsOK());
  params_.blockEncoding.blockSize = nullptr;
  EXPECT_FALSE(Convert().IsOK());
}

TEST_F(QairtConversionTests, FloatBlockEncodingPreservesFloatingOffsets) {
  params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK;
  uint32_t blocks[] = {2, 2};
  Qnn_FloatScaleOffset_t scales[] = {{0.25f, -0.5f}, {0.5f, -1.5f}};
  params_.bwFloatBlockEncoding = {8, blocks, scales};
  ASSERT_TRUE(Convert().IsOK());
  const auto& converted = tensor_.getQuantizeParams().getBwFloatBlockEncoding();
  EXPECT_EQ(converted.getBitwidth(), 8u);
  ASSERT_EQ(converted.getFloatScaleOffsets().size(), 2u);
  EXPECT_FLOAT_EQ(converted.getFloatScaleOffsets()[1].getOffset(), -1.5f);
  params_.bwFloatBlockEncoding.floatScaleOffset = nullptr;
  EXPECT_FALSE(Convert().IsOK());
}

TEST_F(QairtConversionTests, BlockwiseExpansionPreservesBothStorageWidths) {
  params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION;
  Qnn_BlockwiseExpansion_t expansion = QNN_BLOCKWISE_EXPANSION_INIT;
  Qnn_ScaleOffset_t scales[] = {{0.25f, 0}, {0.5f, 0}};
  uint8_t storage8[] = {1, 2, 3, 4};
  uint16_t storage16[] = {256, 257, 258, 259};
  expansion.axis = 0;
  expansion.scaleOffsets = scales;
  expansion.numBlocksPerAxis = 2;
  expansion.blockScaleBitwidth = 8;
  expansion.blockScaleStorageType = QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8;
  expansion.blocksScale8 = storage8;
  params_.blockwiseExpansion = &expansion;
  ASSERT_TRUE(Convert().IsOK());
  EXPECT_EQ(tensor_.getQuantizeParams().getBlockwiseExpansion().getBlocksScale8(),
            (std::vector<uint8_t>{1, 2, 3, 4}));
  expansion.blockScaleBitwidth = 12;
  expansion.blockScaleStorageType = QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16;
  expansion.blocksScale16 = storage16;
  ASSERT_TRUE(Convert().IsOK());
  EXPECT_EQ(tensor_.getQuantizeParams().getBlockwiseExpansion().getBlocksScale16(),
            (std::vector<uint16_t>{256, 257, 258, 259}));
  expansion.blocksScale16 = nullptr;
  EXPECT_FALSE(Convert().IsOK());
  params_.blockwiseExpansion = nullptr;
  EXPECT_FALSE(Convert().IsOK());
}

TEST_F(QairtConversionTests, UndefinedEncodingAndMissingTensorDimensions) {
  params_.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  ASSERT_TRUE(Convert().IsOK());
  params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  ASSERT_TRUE(Convert().IsOK());
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.v1.rank = 2;
  tensor.v1.dimensions = nullptr;
  qairt::Tensor converted;
  EXPECT_FALSE(qnn::qairt_convert::FromQnnTensor(tensor, *api_, converted).IsOK());
}

TEST(QairtProfilingTests, DisabledScopeDoesNotOwnSessionLock) {
  qnn::QairtProfilingManager manager(nullptr, qnn::ProfilingLevel::OFF, "", false);
  qnn::QairtProfilingManager::Scope scope;
  ASSERT_TRUE(manager.Begin(scope, "graph", qnn::OrtProfilingOperation::EXECUTE, nullptr).IsOK());
  EXPECT_FALSE(scope.lock.owns_lock());
  EXPECT_EQ(scope.owner, nullptr);
  EXPECT_EQ(scope.profile, nullptr);
  EXPECT_TRUE(manager.Complete(scope).IsOK());
}

TEST(QairtRuntimeTests, ConfigurationHooksAndRetrievedGraphLifetime) {
  int backend_configs = 0, device_configs = 0, context_configs = 0, graph_configs = 0, retrieves = 0;
  auto configuration = std::make_shared<qnn::QairtBackendConfiguration>();
  configuration->backend = [&](const qairt::Api& api) {
    ++backend_configs;
    return api.make<qairt::BackendConfiguration>();
  };
  configuration->device = [&](const qairt::Api& api) {
    ++device_configs;
    return api.make<qairt::DeviceConfiguration>();
  };
  configuration->context = [&](const qairt::Api& api) {
    ++context_configs;
    auto config = api.make<qairt::ContextConfiguration>();
    // HTP rejects the persistent-binary configuration option even when false.
    // Leave it unspecified; future accelerators may opt in through this hook.
    return config;
  };
  configuration->graph = [&](const qairt::Api& api) {
    ++graph_configs;
    return api.make<qairt::GraphConfiguration>();
  };
  configuration->after_retrieve = [&](qairt::Graph&) {
    ++retrieves;
    return Ort::Status();
  };
  qnn::QairtBackendManager::Config config;
#ifdef _WIN32
  config.backend_lib_path = "QairtHtp.dll";
#else
  config.backend_lib_path = "libQairtHtp.so";
#endif
  config.configuration = configuration;
  auto manager = std::make_shared<qnn::QairtBackendManager>(std::move(config));
  ASSERT_TRUE(manager->Setup().IsOK());
  auto emitter = manager->CreateGraphEmitter();
  ASSERT_TRUE(emitter->InitializeGraph(nullptr, nullptr).IsOK());
  ASSERT_TRUE(emitter->CreateGraph("hook_graph").IsOK());
  uint32_t shape[] = {1, 4};
  Qnn_Tensor_t inputs[] = {QNN_TENSOR_INIT}, outputs[] = {QNN_TENSOR_INIT};
  inputs[0].v1.name = "input";
  inputs[0].v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  inputs[0].v1.dataType = QNN_DATATYPE_FLOAT_32;
  inputs[0].v1.rank = 2;
  inputs[0].v1.dimensions = shape;
  inputs[0].v1.memType = QNN_TENSORMEMTYPE_RAW;
  outputs[0].v1 = inputs[0].v1;
  outputs[0].v1.name = "output";
  outputs[0].v1.type = QNN_TENSOR_TYPE_APP_READ;
  ASSERT_TRUE(emitter->CreateTensor(inputs[0]).IsOK());
  ASSERT_TRUE(emitter->CreateTensor(outputs[0]).IsOK());
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = "relu";
  op.v1.packageName = "qti.aisw";
  op.v1.typeName = "Relu";
  op.v1.numOfInputs = op.v1.numOfOutputs = 1;
  op.v1.inputTensors = inputs;
  op.v1.outputTensors = outputs;
  ASSERT_TRUE(emitter->AddNode(op).IsOK());
  ASSERT_TRUE(emitter->Finalize(nullptr).IsOK());
  std::vector<unsigned char> exported;
  ASSERT_TRUE(manager->ExportContext(exported).IsOK());
  std::vector<char> binary(exported.begin(), exported.end());
  std::vector<qnn::QairtBackendManager::CachedGraph> cached;
  ASSERT_TRUE(manager->LoadContext(binary, cached).IsOK());
  ASSERT_EQ(cached.size(), 1u);
  EXPECT_FALSE(binary.empty());
  auto loaded = manager->CreateGraphEmitter();
  ASSERT_TRUE(static_cast<qnn::QairtGraphEmitter&>(*loaded).RetrieveGraph(cached[0].context, "hook_graph").IsOK());
  EXPECT_EQ(backend_configs, 1);
  EXPECT_EQ(device_configs, 1);
  EXPECT_EQ(context_configs, 2);
  EXPECT_EQ(graph_configs, 1);
  EXPECT_EQ(retrieves, 1);
  cached.clear();
  emitter.reset();
  manager.reset();
  exported.clear();
  binary.assign(1024, '\0');
  float input[] = {-1.f, 2.f, -3.f, 4.f}, output[4] = {};
  inputs[0].v1.clientBuf = {input, sizeof(input)};
  outputs[0].v1.clientBuf = {output, sizeof(output)};
  qnn::GraphExecutionResult result;
  ASSERT_TRUE(loaded->Execute(inputs, 1, outputs, 1, nullptr, result).IsOK());
  EXPECT_FLOAT_EQ(output[0], 0.f);
  EXPECT_FLOAT_EQ(output[1], 2.f);
  EXPECT_FLOAT_EQ(output[2], 0.f);
  EXPECT_FLOAT_EQ(output[3], 4.f);
}

}  // namespace
}  // namespace onnxruntime::test
#endif
