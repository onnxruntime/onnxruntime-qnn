// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

void CheckMixedPrecisionNativeBiasRejected(ONNXTensorElementDataType activation_type,
                                           ONNXTensorElementDataType output_type) {
  QnnRealHtpBackendContext backend;
  if (!backend.IsValid()) GTEST_SKIP() << "QNN HTP backend not available";

  const qnn::IOpBuilder* builder = qnn::GetOpBuilder("Gemm");
  ASSERT_NE(builder, nullptr);

  for (const auto& bias_shape : {std::vector<int64_t>{8}, std::vector<int64_t>{1, 8}}) {
    SCOPED_TRACE(bias_shape.size());
    g_mock_init_reg.clear();
    const auto* scale = g_mock_init_reg.AddScalarFloat("scale", 0.01f);

    OpBuilderTestContext ctx;
    SetupMockInitRegistryStubs(ctx);
    ctx.qnn_validator_interface = backend.qnn_interface;
    ctx.validator_backend_handle = backend.backend_handle;
    auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{});

    auto activation = MakeMockQDQIODef("A", activation_type, std::vector<int64_t>{2, 16}, scale);
    auto weight = MakeMockQDQIODef("B", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                                   std::vector<int64_t>{16, 8}, scale);
    auto bias = MakeMockIODef("C", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, bias_shape);
    auto output = MakeMockQDQIODef("Y", output_type, std::vector<int64_t>{2, 8}, scale);
    auto node_unit = MakeMockQDQNodeUnit("Gemm", {activation, weight, bias}, {output}, "gemm");

    ASSERT_EQ(wrapper->GetTensorType("C"), QNN_TENSOR_TYPE_NATIVE);
    const auto status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.GetErrorMessage().find("FullyConnected + Add decomposition"), std::string::npos)
        << status.GetErrorMessage();
    EXPECT_TRUE(wrapper->GetModelTensorsMap().empty());
  }
}

}  // namespace

TEST(QnnUnitGemmComponentTest, MixedPrecisionU16ActivationU8OutputNativeBiasRejected) {
  CheckMixedPrecisionNativeBiasRejected(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
}

TEST(QnnUnitGemmComponentTest, MixedPrecisionU8ActivationU16OutputNativeBiasRejected) {
  CheckMixedPrecisionNativeBiasRejected(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
