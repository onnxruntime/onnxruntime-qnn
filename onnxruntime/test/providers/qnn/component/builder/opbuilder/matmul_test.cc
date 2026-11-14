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
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

struct MatMulConversionCase {
  const char* name;
  bool enabled;
  QnnHtpDevice_Arch_t arch;
  float rhs_scale;
  uint16_t rhs_zero_point;
  Qnn_DataType_t expected_type;
  const char* expected_arch_warning = nullptr;
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

    std::vector<std::string> warnings;
    OrtApi logging_api = Ort::GetApi();
    logging_api.Logger_GetLoggingSeverityLevel =
        [](const OrtLogger*, OrtLoggingLevel* severity) noexcept -> OrtStatus* {
      *severity = ORT_LOGGING_LEVEL_WARNING;
      return nullptr;
    };
    logging_api.Logger_LogMessage =
        [](const OrtLogger* logger, OrtLoggingLevel severity, const char* message,
           const ORTCHAR_T*, int, const char*) noexcept -> OrtStatus* {
      if (severity == ORT_LOGGING_LEVEL_WARNING) {
        auto* captured = reinterpret_cast<std::vector<std::string>*>(const_cast<OrtLogger*>(logger));
        captured->emplace_back(message ? message : "");
      }
      return nullptr;
    };
    OrtGlobalApiOverride logging_override(&logging_api);
    Ort::Logger logger(reinterpret_cast<const OrtLogger*>(&warnings));

    const auto status = builder->AddToModelBuilder(*wrapper, node_unit, logger, do_op_validation);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    size_t warning_count = 0;
    for (const auto& warning : warnings) {
      if (warning.find("htp_matmul_use_symmetric_uint16") == std::string::npos) {
        continue;
      }
      ++warning_count;
      if (test_case.expected_arch_warning) {
        EXPECT_NE(warning.find(std::string("HTP architecture is ") + test_case.expected_arch_warning),
                  std::string::npos);
      }
      EXPECT_NE(warning.find("HTP v73 or newer is required"), std::string::npos);
      EXPECT_NE(warning.find("using the default range heuristic"), std::string::npos);
    }
    EXPECT_EQ(warning_count, test_case.expected_arch_warning ? 1u : 0u);

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
                             QNN_DATATYPE_UFIXED_POINT_8, "v68"},
        MatMulConversionCase{"EnabledV69", true, QNN_HTP_DEVICE_ARCH_V69, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_8, "v69"},
        MatMulConversionCase{"EnabledV81", true, QNN_HTP_DEVICE_ARCH_V81, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16},
        MatMulConversionCase{"UnspecifiedArch", true, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_8, "unknown"},
        MatMulConversionCase{"DefaultUnspecifiedArch", false, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_8},
        MatMulConversionCase{"UnspecifiedArchHighRange", true, QNN_HTP_DEVICE_ARCH_NONE, 0.002f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16, "unknown"},
        MatMulConversionCase{"EnabledV69HighRange", true, QNN_HTP_DEVICE_ARCH_V69, 0.002f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16, "v69"},
        MatMulConversionCase{"SymmetricDefault", false, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32768,
                             QNN_DATATYPE_UFIXED_POINT_16, nullptr, false},
        MatMulConversionCase{"SymmetricEnabled", true, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32768,
                             QNN_DATATYPE_UFIXED_POINT_16, nullptr, false},
        MatMulConversionCase{"SymmetricUnspecifiedArch", true, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32768,
                             QNN_DATATYPE_UFIXED_POINT_16, nullptr, false},
        MatMulConversionCase{"StaticRhs", true, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32787,
                             QNN_DATATYPE_SFIXED_POINT_16, nullptr, true, true},
        MatMulConversionCase{"StaticUnspecifiedArch", true, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32787,
                             QNN_DATATYPE_SFIXED_POINT_16, nullptr, true, true},
        MatMulConversionCase{"CpuBackend", true, QNN_HTP_DEVICE_ARCH_V73, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16, nullptr, false, false, qnn::QnnBackendType::CPU},
        MatMulConversionCase{"CpuUnspecifiedArch", true, QNN_HTP_DEVICE_ARCH_NONE, 0.0001525f, 32787,
                             QNN_DATATYPE_UFIXED_POINT_16, nullptr, false, false, qnn::QnnBackendType::CPU}),
    [](const ::testing::TestParamInfo<MatMulConversionCase>& info) { return info.param.name; });

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
