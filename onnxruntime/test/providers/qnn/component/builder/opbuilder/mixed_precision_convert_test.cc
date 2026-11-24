// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/mock_init_registry.h"
#include "test/providers/qnn/infra/mock_node_unit.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

// Minimal repro of tetracode#21892: Add with UINT8 + UINT16 inputs and UINT8
// output at an EP partition boundary (output is a subgraph/graph output).
TEST(QnnUnit_MixedPrecisionConvertTest, Add_U8_U16_To_U8_GraphOutput) {
  g_mock_init_reg.clear();
  auto scale_in0 = g_mock_init_reg.AddScalarFloat("scale_in0", 0.0335f);
  auto zp_in0 = g_mock_init_reg.AddScalarUint8("zp_in0", 114);
  auto scale_in1 = g_mock_init_reg.AddScalarFloat("scale_in1", 0.00017949f);
  auto zp_in1 = g_mock_init_reg.AddScalarUint16("zp_in1", 46009);
  auto scale_out = g_mock_init_reg.AddScalarFloat("scale_out", 0.04008f);
  auto zp_out = g_mock_init_reg.AddScalarUint8("zp_out", 165);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ctx.output_info.names.push_back("add_out");
  ctx.output_info.indices["add_out"] = 0;
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto in0 = MakeMockQDQIODef("in0", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                              std::vector<int64_t>{1, 4}, scale_in0, zp_in0);
  auto in1 = MakeMockQDQIODef("in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                              std::vector<int64_t>{1, 4}, scale_in1, zp_in1);
  auto out = MakeMockQDQIODef("add_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                              std::vector<int64_t>{1, 4}, scale_out, zp_out);
  auto node_unit = MakeMockQDQNodeUnit("Add", {in0, in1}, {out}, "Add_6");

  const IOpBuilder* builder = GetOpBuilder("Add");
  ASSERT_NE(builder, nullptr);
  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();

  const auto& tensors = wrapper->GetModelTensorsMap();
  auto it = tensors.find("add_out");
  ASSERT_TRUE(it != tensors.end());
  EXPECT_EQ(it->second.GetTensorType(), QNN_TENSOR_TYPE_APP_READ);
  EXPECT_EQ(it->second.GetTensorDataType(), QNN_DATATYPE_UFIXED_POINT_8);
}

TEST(QnnUnit_MixedPrecisionConvertTest, Add_U8_U16_To_U8_Internal) {
  g_mock_init_reg.clear();
  auto scale_in0 = g_mock_init_reg.AddScalarFloat("scale_in0", 0.0335f);
  auto zp_in0 = g_mock_init_reg.AddScalarUint8("zp_in0", 114);
  auto scale_in1 = g_mock_init_reg.AddScalarFloat("scale_in1", 0.00017949f);
  auto zp_in1 = g_mock_init_reg.AddScalarUint16("zp_in1", 46009);
  auto scale_out = g_mock_init_reg.AddScalarFloat("scale_out", 0.04008f);
  auto zp_out = g_mock_init_reg.AddScalarUint8("zp_out", 165);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto in0 = MakeMockQDQIODef("in0", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                              std::vector<int64_t>{1, 4}, scale_in0, zp_in0);
  auto in1 = MakeMockQDQIODef("in1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                              std::vector<int64_t>{1, 4}, scale_in1, zp_in1);
  auto out = MakeMockQDQIODef("add_out", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                              std::vector<int64_t>{1, 4}, scale_out, zp_out);
  auto node_unit = MakeMockQDQNodeUnit("Add", {in0, in1}, {out}, "Add_6");

  const IOpBuilder* builder = GetOpBuilder("Add");
  ASSERT_NE(builder, nullptr);
  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();

  const auto& tensors = wrapper->GetModelTensorsMap();
  auto it = tensors.find("add_out");
  ASSERT_TRUE(it != tensors.end());
  EXPECT_EQ(it->second.GetTensorType(), QNN_TENSOR_TYPE_NATIVE);
  EXPECT_EQ(it->second.GetTensorDataType(), QNN_DATATYPE_UFIXED_POINT_8);
}

namespace {

struct MatMulConversionCase {
  const char* name;
  bool enabled;
  QnnHtpDevice_Arch_t arch;
  float rhs_scale;
  uint16_t rhs_zero_point;
  Qnn_DataType_t expected_type;
  bool expect_convert = true;
  bool static_rhs = false;
  qnn::QnnBackendType backend_type = qnn::QnnBackendType::HTP;
};

class QnnUnit_MatMulComponentTest : public ::testing::TestWithParam<MatMulConversionCase> {};

TEST_P(QnnUnit_MatMulComponentTest, DynamicU16Conversion) {
  const auto& test_case = GetParam();
  const qnn::IOpBuilder* builder = qnn::GetOpBuilder("MatMul");
  ASSERT_NE(builder, nullptr);

  for (bool do_op_validation : {false, true}) {
    SCOPED_TRACE(do_op_validation);
    g_mock_init_reg.clear();
    const auto* lhs_scale = g_mock_init_reg.AddScalarFloat("lhs_scale", 0.001f);
    const auto* lhs_zp = g_mock_init_reg.AddScalarUint16("lhs_zp", 32768);
    const auto* rhs_scale = g_mock_init_reg.AddScalarFloat("rhs_scale", test_case.rhs_scale);
    const auto* rhs_zp = g_mock_init_reg.AddScalarUint16("rhs_zp", test_case.rhs_zero_point);
    if (test_case.static_rhs) {
      MockInitSpec rhs;
      rhs.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
      rhs.dims = {3, 2};
      rhs.raw_bytes.resize(6 * sizeof(uint16_t));
      g_mock_init_reg.Add("input1", std::move(rhs));
    }

    OpBuilderTestContext ctx;
    SetupMockInitRegistryStubs(ctx);
    ctx.qnn_interface.backendValidateOpConfig = [](Qnn_BackendHandle_t, Qnn_OpConfig_t) -> Qnn_ErrorHandle_t {
      return QNN_BACKEND_NO_ERROR;
    };
    ctx.backend_manager.HtpArch() = test_case.arch;
    qnn::ModelSettings settings;
    settings.use_native_matmul = true;
    settings.htp_matmul_use_symmetric_uint16 = test_case.enabled;
    auto wrapper = ctx.CreateWrapper(settings, test_case.backend_type);

    auto lhs = MakeMockQDQIODef("input0", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                                std::vector<int64_t>{2, 3}, lhs_scale, lhs_zp);
    auto rhs = MakeMockQDQIODef("input1", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                                std::vector<int64_t>{3, 2}, rhs_scale, rhs_zp);
    auto output = MakeMockQDQIODef("Y", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                                   std::vector<int64_t>{2, 2}, lhs_scale, lhs_zp);
    auto node_unit = MakeMockQDQNodeUnit("MatMul", {lhs, rhs}, {output}, "matmul");

    const auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, do_op_validation);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    size_t convert_count = 0;
    for (const auto& [name, tensor] : wrapper->GetModelTensorsMap()) {
      if (name.find("input1_convert") == std::string::npos) {
        continue;
      }
      ++convert_count;
      EXPECT_EQ(tensor.GetTensorDataType(), test_case.expected_type);
      const auto& quant_params = tensor.GetQnnQuantParams().Get();
      ASSERT_EQ(quant_params.encodingDefinition, QNN_DEFINITION_DEFINED);
      ASSERT_EQ(quant_params.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
      if (test_case.expected_type == QNN_DATATYPE_UFIXED_POINT_16) {
        EXPECT_EQ(quant_params.scaleOffsetEncoding.offset, -32768);
      }
    }
    EXPECT_EQ(convert_count, test_case.expect_convert ? 1u : 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ConversionPolicy, QnnUnit_MatMulComponentTest,
    ::testing::Values(
        MatMulConversionCase{"DefaultLowRange", false, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_8},
        MatMulConversionCase{"DefaultHighRange", false, QNN_HTP_DEVICE_ARCH_V73, 0.002f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16},
        MatMulConversionCase{"EnabledLowRange", true, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16},
        MatMulConversionCase{"EnabledHighRange", true, QNN_HTP_DEVICE_ARCH_V73, 0.002f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16},
        MatMulConversionCase{"EnabledV68", true, QNN_HTP_DEVICE_ARCH_V68, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_8},
        MatMulConversionCase{"EnabledV69", true, QNN_HTP_DEVICE_ARCH_V69, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_8},
        MatMulConversionCase{"EnabledV81", true, QNN_HTP_DEVICE_ARCH_V81, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16},
        MatMulConversionCase{"UnspecifiedArch", true, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_8},
        MatMulConversionCase{"DefaultUnspecifiedArch", false, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_8},
        MatMulConversionCase{"UnspecifiedArchHighRange", true, QNN_HTP_DEVICE_ARCH_NONE, 0.002f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16},
        MatMulConversionCase{"EnabledV69HighRange", true, QNN_HTP_DEVICE_ARCH_V69, 0.002f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16},
        MatMulConversionCase{"SymmetricDefault", false, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32768,
                             QNN_DATATYPE_UFIXED_POINT_16, false},
        MatMulConversionCase{"SymmetricEnabled", true, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32768,
                             QNN_DATATYPE_UFIXED_POINT_16, false},
        MatMulConversionCase{"SymmetricUnspecifiedArch", true, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32768,
                             QNN_DATATYPE_UFIXED_POINT_16, false},
        MatMulConversionCase{"StaticRhs", true, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32787,
                             QNN_DATATYPE_SFIXED_POINT_16, true, true},
        MatMulConversionCase{"StaticUnspecifiedArch", true, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32787,
                             QNN_DATATYPE_SFIXED_POINT_16, true, true},
        MatMulConversionCase{"CpuBackend", true, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16, false, false, qnn::QnnBackendType::CPU},
        MatMulConversionCase{"CpuUnspecifiedArch", true, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16, false, false, qnn::QnnBackendType::CPU}),
    [](const ::testing::TestParamInfo<MatMulConversionCase>& info) { return info.param.name; });

}  // namespace

}  // namespace test
}  // namespace onnxruntime

#endif
