// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for qnn_utils.cc — pure utility functions that
// have no dependency on a real QNN backend or OrtApi at runtime.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl>

#include "QnnInterface.h"
#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

// Forward declarations for operator<< overloads defined in qnn_utils.cc
// but not declared in qnn_utils.h (Qnn_DataType_t IS declared there).
// These exist in the EP shared library.
namespace onnxruntime {
namespace qnn {
namespace utils {
std::ostream& operator<<(std::ostream& out, const Qnn_Scalar_t& scalar);
std::ostream& operator<<(std::ostream& out, const Qnn_Definition_t& definition);
std::ostream& operator<<(std::ostream& out, const Qnn_QuantizationEncoding_t& encoding);
std::ostream& operator<<(std::ostream& out, const Qnn_QuantizeParams_t& quantize_params);
std::ostream& operator<<(std::ostream& out, const Qnn_TensorType_t& tensor_type);
std::ostream& operator<<(std::ostream& out, const Qnn_TensorMemType_t& mem_type);
std::ostream& operator<<(std::ostream& out, const Qnn_ParamType_t& param_type);
}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime

// Bring everything into scope at file level so TEST() bodies can use them.
namespace onnxruntime {
namespace test {

// operator<< overloads for Qnn_* C types live in qnn::utils. The streamed
// types are in the global namespace, so ADL cannot find them — a single
// using-declaration (not a wildcard using-directive) is required so that
// `oss << qnn_value` resolves to the qnn::utils overload.
using qnn::utils::operator<<;

// =============================================================================
// qnn::utils::GetElementSizeByType(Qnn_DataType_t)
// =============================================================================

TEST(QnnUnit_UtilsTest, GetElementSizeByType_Qnn_KnownTypes) {
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_INT_8), 1u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_INT_16), 2u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_INT_32), 4u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_INT_64), 8u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_UINT_8), 1u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_UINT_16), 2u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_UINT_32), 4u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_UINT_64), 8u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_FLOAT_16), 2u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_FLOAT_32), 4u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_FLOAT_64), 8u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_BFLOAT_16), 2u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_BOOL_8), 1u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_SFIXED_POINT_8), 1u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_SFIXED_POINT_16), 2u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_SFIXED_POINT_32), 4u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_UFIXED_POINT_8), 1u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_UFIXED_POINT_16), 2u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_UFIXED_POINT_32), 4u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_SFIXED_POINT_4), 1u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_UFIXED_POINT_4), 1u);
  EXPECT_EQ(qnn::utils::GetElementSizeByType(QNN_DATATYPE_UNDEFINED), 1u);
}

TEST(QnnUnit_UtilsTest, GetElementSizeByType_Qnn_UnknownThrows) {
  EXPECT_THROW(qnn::utils::GetElementSizeByType(static_cast<Qnn_DataType_t>(9999)), Ort::Exception);
}

// =============================================================================
// qnn::utils::GetElementSizeByType(ONNXTensorElementDataType)
// =============================================================================

TEST(QnnUnit_UtilsTest, GetElementSizeByType_OnnxElem_UnknownThrows) {
  EXPECT_THROW(
      qnn::utils::GetElementSizeByType(static_cast<ONNXTensorElementDataType>(9999)),
      Ort::Exception);
}

// =============================================================================
// GetElementNameByType
// =============================================================================

TEST(QnnUnit_UtilsTest, GetElementNameByType_AllKnownTypes) {
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2), "int2_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2), "uint2_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4), "int4_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4), "uint4_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8), "int8_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16), "int16_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32), "int32_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), "int64_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8), "uint8_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16), "uint16_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32), "uint32_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64), "uint64_t");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16), "float16");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), "float32");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE), "double");
  EXPECT_EQ(qnn::utils::GetElementNameByType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL), "bool");
}

TEST(QnnUnit_UtilsTest, GetElementNameByType_UnknownThrows) {
  EXPECT_THROW(
      qnn::utils::GetElementNameByType(static_cast<ONNXTensorElementDataType>(9999)),
      Ort::Exception);
}

// =============================================================================
// GetQnnTensorDataSizeInBytes
// =============================================================================

TEST(QnnUnit_UtilsTest, GetQnnTensorDataSizeInBytes_4BitRoundsUp) {
  EXPECT_EQ(qnn::utils::GetQnnTensorDataSizeInBytes(4u, QNN_DATATYPE_SFIXED_POINT_4), 2u);
  EXPECT_EQ(qnn::utils::GetQnnTensorDataSizeInBytes(5u, QNN_DATATYPE_SFIXED_POINT_4), 3u);
  EXPECT_EQ(qnn::utils::GetQnnTensorDataSizeInBytes(1u, QNN_DATATYPE_UFIXED_POINT_4), 1u);
  EXPECT_EQ(qnn::utils::GetQnnTensorDataSizeInBytes(6u, QNN_DATATYPE_UFIXED_POINT_4), 3u);
}

TEST(QnnUnit_UtilsTest, GetQnnTensorDataSizeInBytes_RegularTypes) {
  EXPECT_EQ(qnn::utils::GetQnnTensorDataSizeInBytes(4u, QNN_DATATYPE_SFIXED_POINT_8), 4u);
  EXPECT_EQ(qnn::utils::GetQnnTensorDataSizeInBytes(3u, QNN_DATATYPE_SFIXED_POINT_16), 6u);
  EXPECT_EQ(qnn::utils::GetQnnTensorDataSizeInBytes(2u, QNN_DATATYPE_FLOAT_32), 8u);
}

TEST(QnnUnit_UtilsTest, GetQnnTensorDataSizeInBytes_SpanEmptyThrows) {
  // Empty shape represents a 0D scalar: size should equal one element.
  EXPECT_EQ(
      qnn::utils::GetQnnTensorDataSizeInBytes(gsl::span<const uint32_t>{}, QNN_DATATYPE_FLOAT_32),
      4u);
}

TEST(QnnUnit_UtilsTest, GetQnnTensorDataSizeInBytes_SpanMultiDim) {
  std::vector<uint32_t> shape = {2, 3, 4};
  gsl::span<const uint32_t> shape_span(shape.data(), shape.size());
  EXPECT_EQ(
      qnn::utils::GetQnnTensorDataSizeInBytes(shape_span, QNN_DATATYPE_FLOAT_32),
      2u * 3u * 4u * sizeof(float));
}

// =============================================================================
// QnnTensorHasDynamicShape
// =============================================================================

TEST(QnnUnit_UtilsTest, QnnTensorHasDynamicShape_V1TensorAlwaysFalse) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  EXPECT_FALSE(qnn::utils::QnnTensorHasDynamicShape(tensor));
}

#ifdef QNN_TENSOR_V2_INIT
TEST(QnnUnit_UtilsTest, QnnTensorHasDynamicShape_V2AllStaticDims) {
  Qnn_Tensor_t tensor;
  std::memset(&tensor, 0, sizeof(tensor));
  tensor.version = QNN_TENSOR_VERSION_2;
  std::vector<uint8_t> is_dynamic = {0, 0, 0};
  std::vector<uint32_t> dims = {1, 2, 3};
  tensor.v2.rank = static_cast<uint32_t>(dims.size());
  tensor.v2.dimensions = dims.data();
  tensor.v2.isDynamicDimensions = is_dynamic.data();
  EXPECT_FALSE(qnn::utils::QnnTensorHasDynamicShape(tensor));
}

TEST(QnnUnit_UtilsTest, QnnTensorHasDynamicShape_V2OneDynamicDim) {
  Qnn_Tensor_t tensor;
  std::memset(&tensor, 0, sizeof(tensor));
  tensor.version = QNN_TENSOR_VERSION_2;
  std::vector<uint8_t> is_dynamic = {0, 1, 0};
  std::vector<uint32_t> dims = {1, 2, 3};
  tensor.v2.rank = static_cast<uint32_t>(dims.size());
  tensor.v2.dimensions = dims.data();
  tensor.v2.isDynamicDimensions = is_dynamic.data();
  EXPECT_TRUE(qnn::utils::QnnTensorHasDynamicShape(tensor));
}
#endif  // QNN_TENSOR_V2_INIT

// =============================================================================
// operator<< for Qnn types
// =============================================================================

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_INT64PrintsNotSupported) {
  Qnn_Scalar_t s{};
  s.dataType = QNN_DATATYPE_INT_64;
  std::ostringstream oss;
  oss << s;
  EXPECT_FALSE(oss.str().empty());
}

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_UINT64PrintsNotSupported) {
  Qnn_Scalar_t s{};
  s.dataType = QNN_DATATYPE_UINT_64;
  std::ostringstream oss;
  oss << s;
  EXPECT_FALSE(oss.str().empty());
}

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_FLOAT16DoesNotCrash) {
  Qnn_Scalar_t s{};
  s.dataType = QNN_DATATYPE_FLOAT_16;
  std::ostringstream oss;
  oss << s;                        // no output in the switch case, just verifying no crash
  EXPECT_TRUE(oss.str().empty());  // FLOAT_16 case has no output (empty break)
}

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_SFixedPointVariants) {
  for (auto dt : {QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_16,
                  QNN_DATATYPE_SFIXED_POINT_32, QNN_DATATYPE_UFIXED_POINT_8,
                  QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_32}) {
    Qnn_Scalar_t s{};
    s.dataType = dt;
    std::ostringstream oss;
    oss << s;
    EXPECT_FALSE(oss.str().empty()) << "for dataType=" << static_cast<int>(dt);
  }
}

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_INT8PrintsValue) {
  Qnn_Scalar_t s{};
  s.dataType = QNN_DATATYPE_INT_8;
  s.int8Value = -42;
  std::ostringstream oss;
  oss << s;
  EXPECT_EQ(oss.str(), "-42");
}

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_INT16PrintsValue) {
  Qnn_Scalar_t s{};
  s.dataType = QNN_DATATYPE_INT_16;
  s.int16Value = 1234;
  std::ostringstream oss;
  oss << s;
  EXPECT_EQ(oss.str(), "1234");
}

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_UINT8PrintsValue) {
  Qnn_Scalar_t s{};
  s.dataType = QNN_DATATYPE_UINT_8;
  s.uint8Value = 200;
  std::ostringstream oss;
  oss << s;
  EXPECT_EQ(oss.str(), "200");
}

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_UINT16PrintsValue) {
  Qnn_Scalar_t s{};
  s.dataType = QNN_DATATYPE_UINT_16;
  s.uint16Value = 50000;
  std::ostringstream oss;
  oss << s;
  EXPECT_EQ(oss.str(), "50000");
}

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_FLOAT64PrintsValue) {
  Qnn_Scalar_t s{};
  s.dataType = QNN_DATATYPE_FLOAT_64;
  s.doubleValue = 3.14;
  std::ostringstream oss;
  oss << s;
  EXPECT_FALSE(oss.str().empty());
}

TEST(QnnUnit_UtilsTest, OstreamQnnScalar_UnknownThrows) {
  Qnn_Scalar_t s{};
  s.dataType = static_cast<Qnn_DataType_t>(9999);
  std::ostringstream oss;
  EXPECT_THROW(oss << s, Ort::Exception);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_UnknownThrows) {
  Qnn_DataType_t dt = static_cast<Qnn_DataType_t>(9999);
  std::ostringstream oss;
  EXPECT_THROW(oss << dt, Ort::Exception);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_Int8) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_INT_8;
  EXPECT_NE(oss.str().find("INT_8"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_Int16) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_INT_16;
  EXPECT_NE(oss.str().find("INT_16"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_Uint16) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_UINT_16;
  EXPECT_NE(oss.str().find("UINT_16"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_Int64) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_INT_64;
  EXPECT_NE(oss.str().find("INT_64"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_Uint64) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_UINT_64;
  EXPECT_NE(oss.str().find("UINT_64"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_Float64) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_FLOAT_64;
  EXPECT_NE(oss.str().find("FLOAT_64"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_SFixedPoint8) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_SFIXED_POINT_8;
  EXPECT_NE(oss.str().find("SFIXED_POINT_8"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_SFixedPoint16) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_SFIXED_POINT_16;
  EXPECT_NE(oss.str().find("SFIXED_POINT_16"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_SFixedPoint32) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_SFIXED_POINT_32;
  EXPECT_NE(oss.str().find("SFIXED_POINT_32"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_UFixedPoint8) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_UFIXED_POINT_8;
  EXPECT_NE(oss.str().find("UFIXED_POINT_8"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_UFixedPoint16) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_UFIXED_POINT_16;
  EXPECT_NE(oss.str().find("UFIXED_POINT_16"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_UFixedPoint32) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_UFIXED_POINT_32;
  EXPECT_NE(oss.str().find("UFIXED_POINT_32"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_BFloat16) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_BFLOAT_16;
  EXPECT_NE(oss.str().find("BFLOAT_16"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_Bool8) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_BOOL_8;
  EXPECT_NE(oss.str().find("BOOL_8"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_SFixedPoint4) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_SFIXED_POINT_4;
  EXPECT_NE(oss.str().find("SFIXED_POINT_4"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_UFixedPoint4) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_UFIXED_POINT_4;
  EXPECT_NE(oss.str().find("UFIXED_POINT_4"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDataType_Undefined) {
  std::ostringstream oss;
  oss << QNN_DATATYPE_UNDEFINED;
  EXPECT_NE(oss.str().find("UNDEFINED"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnDefinition_AllBranches) {
  auto check = [](Qnn_Definition_t def, std::string_view expected_substr) {
    std::ostringstream oss;
    oss << def;
    EXPECT_NE(oss.str().find(expected_substr), std::string::npos);
  };
  check(QNN_DEFINITION_IMPL_GENERATED, "IMPL_GENERATED");
  check(QNN_DEFINITION_DEFINED, "DEFINED");
  check(QNN_DEFINITION_UNDEFINED, "UNDEFINED");
  std::ostringstream oss;
  oss << static_cast<Qnn_Definition_t>(999);
  EXPECT_FALSE(oss.str().empty());
}

TEST(QnnUnit_UtilsTest, OstreamQnnQuantEncoding_AllBranches) {
  auto check = [](Qnn_QuantizationEncoding_t enc, std::string_view expected_substr) {
    std::ostringstream oss;
    oss << enc;
    EXPECT_NE(oss.str().find(expected_substr), std::string::npos);
  };
  check(QNN_QUANTIZATION_ENCODING_SCALE_OFFSET, "SCALE_OFFSET");
  check(QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET, "AXIS_SCALE_OFFSET");
  check(QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET, "BW_SCALE_OFFSET");
  check(QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET, "BW_AXIS_SCALE_OFFSET");
  check(QNN_QUANTIZATION_ENCODING_UNDEFINED, "UNDEFINED");
  std::ostringstream oss;
  oss << static_cast<Qnn_QuantizationEncoding_t>(999);
  EXPECT_FALSE(oss.str().empty());
}

TEST(QnnUnit_UtilsTest, OstreamQnnQuantizeParams_BwScaleOffset) {
  Qnn_QuantizeParams_t qp{};
  qp.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  qp.bwScaleOffsetEncoding.bitwidth = 8;
  qp.bwScaleOffsetEncoding.scale = 0.01f;
  qp.bwScaleOffsetEncoding.offset = 0;
  std::ostringstream oss;
  oss << qp;
  EXPECT_NE(oss.str().find("bitwidth"), std::string::npos);
}

TEST(QnnUnit_UtilsTest, OstreamQnnQuantizeParams_AllEncodingBranches) {
  auto check = [](Qnn_QuantizeParams_t qp, std::string_view expected_substr) {
    std::ostringstream oss;
    oss << qp;
    EXPECT_NE(oss.str().find(expected_substr), std::string::npos);
  };
  // SCALE_OFFSET
  {
    Qnn_QuantizeParams_t qp{};
    qp.encodingDefinition = QNN_DEFINITION_DEFINED;
    qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    qp.scaleOffsetEncoding.scale = 0.5f;
    qp.scaleOffsetEncoding.offset = 0;
    check(qp, "scale");
  }
  // AXIS_SCALE_OFFSET
  {
    Qnn_QuantizeParams_t qp{};
    qp.encodingDefinition = QNN_DEFINITION_DEFINED;
    qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
    check(qp, "AXIS_SCALE_OFFSET");
  }
  // BW_AXIS_SCALE_OFFSET
  {
    Qnn_QuantizeParams_t qp{};
    qp.encodingDefinition = QNN_DEFINITION_DEFINED;
    qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
    check(qp, "BW_AXIS_SCALE_OFFSET");
  }
  // UNDEFINED encoding
  {
    Qnn_QuantizeParams_t qp{};
    qp.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    check(qp, "UNDEFINED");
  }
}

TEST(QnnUnit_UtilsTest, OstreamQnnTensorType_AllBranches) {
  auto check = [](Qnn_TensorType_t tt, std::string_view expected_substr) {
    std::ostringstream oss;
    oss << tt;
    EXPECT_NE(oss.str().find(expected_substr), std::string::npos);
  };
  check(QNN_TENSOR_TYPE_APP_WRITE, "APP_WRITE");
  check(QNN_TENSOR_TYPE_APP_READ, "APP_READ");
  check(QNN_TENSOR_TYPE_APP_READWRITE, "APP_READWRITE");
  check(QNN_TENSOR_TYPE_NATIVE, "NATIVE");
  check(QNN_TENSOR_TYPE_STATIC, "STATIC");
  check(QNN_TENSOR_TYPE_NULL, "NULL");
  std::ostringstream oss;
  oss << static_cast<Qnn_TensorType_t>(999);
  EXPECT_FALSE(oss.str().empty());
}

TEST(QnnUnit_UtilsTest, OstreamQnnTensorMemType_AllBranches) {
  {
    std::ostringstream oss;
    oss << QNN_TENSORMEMTYPE_RAW;
    EXPECT_NE(oss.str().find("RAW"), std::string::npos);
  }
  {
    std::ostringstream oss;
    oss << QNN_TENSORMEMTYPE_MEMHANDLE;
    EXPECT_NE(oss.str().find("MEMHANDLE"), std::string::npos);
  }
  {
    std::ostringstream oss;
    oss << static_cast<Qnn_TensorMemType_t>(999);
    EXPECT_FALSE(oss.str().empty());
  }
}

TEST(QnnUnit_UtilsTest, OstreamQnnParamType_DefaultBranch) {
  std::ostringstream oss;
  oss << static_cast<Qnn_ParamType_t>(999);
  EXPECT_NE(oss.str().find("Unknown"), std::string::npos);
}

// =============================================================================
// QnnJSONGraph — exercises AppendQnnElemsToJSONArray data-type branches.
// TensorFixture owns the shape vector and raw data so Qnn_Tensor_t pointers
// remain valid throughout the test.
// =============================================================================

struct TensorFixture {
  std::string name;
  std::vector<uint32_t> shape;
  std::vector<uint8_t> raw_bytes;
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;

  template <typename T>
  void Init(std::string tensor_name, Qnn_DataType_t dt, std::vector<T> typed_data) {
    name = std::move(tensor_name);
    shape = {static_cast<uint32_t>(typed_data.size())};
    raw_bytes.resize(typed_data.size() * sizeof(T));
    std::memcpy(raw_bytes.data(), typed_data.data(), raw_bytes.size());
    tensor = QNN_TENSOR_INIT;
    qnn::SetQnnTensorName(tensor, name.c_str());
    qnn::SetQnnTensorType(tensor, QNN_TENSOR_TYPE_STATIC);
    qnn::SetQnnTensorDataType(tensor, dt);
    qnn::SetQnnTensorDim(tensor, shape);
    qnn::SetQnnTensorMemType(tensor, QNN_TENSORMEMTYPE_RAW);
    qnn::SetQnnTensorClientBuf(tensor, raw_bytes.data(),
                               static_cast<uint32_t>(raw_bytes.size()));
  }

  void InitIO(std::string tensor_name, Qnn_DataType_t dt, std::vector<uint32_t> dims) {
    name = std::move(tensor_name);
    shape = std::move(dims);
    tensor = QNN_TENSOR_INIT;
    qnn::SetQnnTensorName(tensor, name.c_str());
    qnn::SetQnnTensorDataType(tensor, dt);
    qnn::SetQnnTensorDim(tensor, shape);
  }
};

template <typename DataT>
static void RunJsonGraphParamTest(const char* op_name, Qnn_DataType_t param_dt,
                                  Qnn_DataType_t io_dt, std::vector<DataT> data) {
  TensorFixture param_fix, in_fix, out_fix;
  param_fix.Init<DataT>(std::string(op_name) + "_p", param_dt, std::move(data));
  in_fix.InitIO(std::string(op_name) + "_in", io_dt,
                {static_cast<uint32_t>(param_fix.shape[0])});
  out_fix.InitIO(std::string(op_name) + "_out", io_dt, in_fix.shape);

  Qnn_Param_t param{};
  param.paramType = QNN_PARAMTYPE_TENSOR;
  param.name = "w";
  param.tensorParam = param_fix.tensor;

  qnn::QnnOpConfigWrapper op(op_name, "qti.aisw", "Relu",
                             {in_fix.tensor}, {out_fix.tensor}, {param});
  qnn::utils::QnnJSONGraph json_graph;
  json_graph.AddOp(op);
  auto j = json_graph.Finalize();
  EXPECT_FALSE(j.dump().empty()) << "for op=" << op_name;
  ASSERT_TRUE(j.contains("graph")) << "missing 'graph' key for op=" << op_name;
  EXPECT_TRUE(!j["graph"]["nodes"].empty()) << "for op=" << op_name;
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_SFixedPoint8) {
  RunJsonGraphParamTest<int8_t>("op_i8", QNN_DATATYPE_SFIXED_POINT_8,
                                QNN_DATATYPE_SFIXED_POINT_8, {-1, 0, 1, 2});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_SFixedPoint16) {
  RunJsonGraphParamTest<int16_t>("op_i16", QNN_DATATYPE_SFIXED_POINT_16,
                                 QNN_DATATYPE_SFIXED_POINT_16, {-100, 0, 100});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Int32) {
  RunJsonGraphParamTest<int32_t>("op_i32", QNN_DATATYPE_INT_32,
                                 QNN_DATATYPE_INT_32, {1, 2});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Int64) {
  RunJsonGraphParamTest<int64_t>("op_i64", QNN_DATATYPE_INT_64,
                                 QNN_DATATYPE_INT_64, {100000LL});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_UFixedPoint8) {
  RunJsonGraphParamTest<uint8_t>("op_u8", QNN_DATATYPE_UFIXED_POINT_8,
                                 QNN_DATATYPE_UFIXED_POINT_8, {10u, 20u, 30u});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_UFixedPoint16) {
  RunJsonGraphParamTest<uint16_t>("op_u16", QNN_DATATYPE_UFIXED_POINT_16,
                                  QNN_DATATYPE_UFIXED_POINT_16, {1u, 2u});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_UFixedPoint32) {
  RunJsonGraphParamTest<uint32_t>("op_u32", QNN_DATATYPE_UFIXED_POINT_32,
                                  QNN_DATATYPE_UFIXED_POINT_32, {5u});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Uint64) {
  RunJsonGraphParamTest<uint64_t>("op_u64", QNN_DATATYPE_UINT_64,
                                  QNN_DATATYPE_UINT_64, {999999ULL});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Float32) {
  RunJsonGraphParamTest<float>("op_f32", QNN_DATATYPE_FLOAT_32,
                               QNN_DATATYPE_FLOAT_32, {0.5f, -0.5f});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Float16) {
  RunJsonGraphParamTest<Ort::Float16_t>("op_f16", QNN_DATATYPE_FLOAT_16,
                                        QNN_DATATYPE_FLOAT_16,
                                        {Ort::Float16_t(1.0f)});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Int8Raw) {
  RunJsonGraphParamTest<int8_t>("op_i8_raw", QNN_DATATYPE_INT_8,
                                QNN_DATATYPE_INT_8, {-1, 0, 1, 2});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Int16Raw) {
  RunJsonGraphParamTest<int16_t>("op_i16_raw", QNN_DATATYPE_INT_16,
                                 QNN_DATATYPE_INT_16, {-100, 0, 100});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Uint8Raw) {
  RunJsonGraphParamTest<uint8_t>("op_u8_raw", QNN_DATATYPE_UINT_8,
                                 QNN_DATATYPE_UINT_8, {10u, 20u, 30u});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Uint16Raw) {
  RunJsonGraphParamTest<uint16_t>("op_u16_raw", QNN_DATATYPE_UINT_16,
                                  QNN_DATATYPE_UINT_16, {1u, 2u});
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_AddOp_Uint32Raw) {
  RunJsonGraphParamTest<uint32_t>("op_u32_raw", QNN_DATATYPE_UINT_32,
                                  QNN_DATATYPE_UINT_32, {5u, 10u});
}

// =============================================================================
// QnnJSONGraph — exercises GetQnnScalarParamJSON via SCALAR params.
// =============================================================================

static void RunJsonGraphScalarParamTest(const char* op_name, Qnn_DataType_t scalar_dt) {
  TensorFixture in_fix, out_fix;
  in_fix.InitIO(std::string(op_name) + "_in", QNN_DATATYPE_FLOAT_32, {4});
  out_fix.InitIO(std::string(op_name) + "_out", QNN_DATATYPE_FLOAT_32, {4});

  Qnn_Scalar_t scalar_val{};
  scalar_val.dataType = scalar_dt;

  Qnn_Param_t param{};
  param.paramType = QNN_PARAMTYPE_SCALAR;
  param.name = "axis";
  param.scalarParam = scalar_val;

  qnn::QnnOpConfigWrapper op(op_name, "qti.aisw", "Relu",
                             {in_fix.tensor}, {out_fix.tensor}, {param});
  qnn::utils::QnnJSONGraph json_graph;
  json_graph.AddOp(op);
  EXPECT_FALSE(json_graph.Finalize().dump().empty()) << "for op=" << op_name;
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_ScalarParam_Int16CoversBranch) {
  RunJsonGraphScalarParamTest("op_sp_i16", QNN_DATATYPE_INT_16);
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_ScalarParam_Uint8CoversBranch) {
  RunJsonGraphScalarParamTest("op_sp_u8", QNN_DATATYPE_UINT_8);
}

TEST(QnnUnit_UtilsTest, QnnJSONGraph_ScalarParam_Uint16CoversBranch) {
  RunJsonGraphScalarParamTest("op_sp_u16", QNN_DATATYPE_UINT_16);
}

// =============================================================================
// OnnxDataTypeToQnnDataType — quantized path + GPU backend
// =============================================================================

TEST(QnnUnit_UtilsTest, OnnxDataTypeToQnnDataType_QuantizedPath) {
  Qnn_DataType_t qnn_type = QNN_DATATYPE_UNDEFINED;
  bool ok = qnn::utils::OnnxDataTypeToQnnDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
                                                  qnn_type, /*is_quantized=*/true);
  EXPECT_TRUE(ok);
  EXPECT_EQ(qnn_type, QNN_DATATYPE_SFIXED_POINT_8);
}

TEST(QnnUnit_UtilsTest, OnnxDataTypeToQnnDataType_QuantizedPath_MoreTypes) {
  struct Mapping {
    ONNXTensorElementDataType onnx;
    Qnn_DataType_t expected;
  };
  const Mapping table[] = {
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, QNN_DATATYPE_UFIXED_POINT_8},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, QNN_DATATYPE_SFIXED_POINT_16},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, QNN_DATATYPE_UFIXED_POINT_16},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, QNN_DATATYPE_FLOAT_16},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, QNN_DATATYPE_FLOAT_32},
  };
  for (const auto& m : table) {
    Qnn_DataType_t qnn_type = QNN_DATATYPE_UNDEFINED;
    bool ok = qnn::utils::OnnxDataTypeToQnnDataType(m.onnx, qnn_type, /*is_quantized=*/true);
    EXPECT_TRUE(ok) << "for onnx type " << m.onnx;
    EXPECT_EQ(qnn_type, m.expected) << "for onnx type " << m.onnx;
  }
}

TEST(QnnUnit_UtilsTest, OnnxDataTypeToQnnDataType_UnquantizedPath) {
  Qnn_DataType_t qnn_type = QNN_DATATYPE_UNDEFINED;
  bool ok = qnn::utils::OnnxDataTypeToQnnDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                  qnn_type, /*is_quantized=*/false);
  EXPECT_TRUE(ok);
  EXPECT_EQ(qnn_type, QNN_DATATYPE_FLOAT_32);
}

TEST(QnnUnit_UtilsTest, OnnxDataTypeToQnnDataType_UnknownReturnsFalse) {
  Qnn_DataType_t qnn_type = QNN_DATATYPE_UNDEFINED;
  bool ok = qnn::utils::OnnxDataTypeToQnnDataType(
      static_cast<ONNXTensorElementDataType>(9999), qnn_type);
  EXPECT_FALSE(ok);
}

TEST(QnnUnit_UtilsTest, OnnxDataTypeToQnnDataType_GpuBackendInt4) {
  Qnn_DataType_t qnn_type = QNN_DATATYPE_UNDEFINED;
  bool ok = qnn::utils::OnnxDataTypeToQnnDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
                                                  qnn_type, /*is_quantized=*/false,
                                                  qnn::QnnBackendType::GPU);
  EXPECT_TRUE(ok);
  EXPECT_EQ(qnn_type, QNN_DATATYPE_SFIXED_POINT_4);
}

// =============================================================================
// Quantize
// =============================================================================

TEST(QnnUnit_UtilsTest, Quantize_Int8_BasicValue) {
  int qval = 0;
  Ort::Status st = qnn::utils::Quantize(0.5, 0.01f, 0, QNN_DATATYPE_SFIXED_POINT_8, qval);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(qval, 50);
}

TEST(QnnUnit_UtilsTest, Quantize_Int8_ClampedToMax) {
  int qval = 0;
  Ort::Status st = qnn::utils::Quantize(1000.0, 0.01f, 0, QNN_DATATYPE_SFIXED_POINT_8, qval);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(qval, 127);
}

TEST(QnnUnit_UtilsTest, Quantize_UnsupportedTypeFails) {
  int qval = 0;
  Ort::Status st = qnn::utils::Quantize(1.0, 1.0f, 0, QNN_DATATYPE_FLOAT_32, qval);
  EXPECT_FALSE(st.IsOK());
}

// =============================================================================
// CheckMinMax
// =============================================================================

TEST(QnnUnit_UtilsTest, CheckMinMax_NormalRange) {
  auto [rmin, rmax] = qnn::utils::CheckMinMax(-1.0f, 2.0f);
  EXPECT_FLOAT_EQ(rmin, -1.0f);
  EXPECT_FLOAT_EQ(rmax, 2.0f);
}

TEST(QnnUnit_UtilsTest, CheckMinMax_ForceIncludeZero) {
  auto [rmin, rmax] = qnn::utils::CheckMinMax(1.0f, 2.0f);
  EXPECT_FLOAT_EQ(rmin, 0.0f);
}

TEST(QnnUnit_UtilsTest, CheckMinMax_EnsureMinRange) {
  auto [rmin, rmax] = qnn::utils::CheckMinMax(0.0f, 0.0f);
  EXPECT_FLOAT_EQ(rmin, 0.0f);
  EXPECT_GE(rmax, 0.0001f);
}

// =============================================================================
// ShapeSizeCalc
// =============================================================================

TEST(QnnUnit_UtilsTest, ShapeSizeCalc_BasicProduct) {
  std::vector<uint32_t> shape = {2, 3, 4};
  EXPECT_EQ(qnn::utils::ShapeSizeCalc(shape, 0, 3), 24u);
  EXPECT_EQ(qnn::utils::ShapeSizeCalc(shape, 0, 2), 6u);
  EXPECT_EQ(qnn::utils::ShapeSizeCalc(shape, 1, 3), 12u);
  EXPECT_EQ(qnn::utils::ShapeSizeCalc(shape, 2, 3), 4u);
  EXPECT_EQ(qnn::utils::ShapeSizeCalc(shape, 0, 0), 1u);
}

// =============================================================================
// DequantizePerChannel — INT16 / UINT16 / INT32 switch branches + error
// =============================================================================

TEST(QnnUnit_UtilsTest, DequantizePerChannel_Int8PerTensor) {
  std::vector<int8_t> quant = {-10, 0, 10};
  std::vector<uint32_t> shape = {3};
  std::vector<float> scales = {0.1f};
  std::vector<int32_t> offsets = {0};
  std::vector<float> out(3);
  auto quant_bytes = gsl::make_span(
      reinterpret_cast<const uint8_t*>(quant.data()), quant.size());
  Ort::Status st = qnn::utils::DequantizePerChannel(quant_bytes, shape, scales, offsets, out,
                                                    QNN_DATATYPE_SFIXED_POINT_8);
  EXPECT_TRUE(st.IsOK());
  EXPECT_NEAR(out[0], -1.0f, 1e-4f);
  EXPECT_NEAR(out[2], 1.0f, 1e-4f);
}

TEST(QnnUnit_UtilsTest, DequantizePerChannel_Uint8PerTensor) {
  std::vector<uint8_t> quant = {100, 128, 200};
  std::vector<uint32_t> shape = {3};
  std::vector<float> scales = {0.01f};
  std::vector<int32_t> offsets = {0};
  std::vector<float> out(3);
  auto quant_bytes = gsl::make_span(quant.data(), quant.size());
  Ort::Status st = qnn::utils::DequantizePerChannel(quant_bytes, shape, scales, offsets, out,
                                                    QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_TRUE(st.IsOK());
  EXPECT_NEAR(out[0], 1.0f, 1e-4f);
}

TEST(QnnUnit_UtilsTest, DequantizePerChannel_Int16CoversBranch) {
  std::vector<int16_t> quant = {-1000, 0, 1000};
  std::vector<uint32_t> shape = {3};
  std::vector<float> scales = {0.001f};
  std::vector<int32_t> offsets = {0};
  std::vector<float> out(3);
  auto quant_bytes = gsl::make_span(
      reinterpret_cast<const uint8_t*>(quant.data()), quant.size() * sizeof(int16_t));
  Ort::Status st = qnn::utils::DequantizePerChannel(quant_bytes, shape, scales, offsets, out,
                                                    QNN_DATATYPE_SFIXED_POINT_16);
  EXPECT_TRUE(st.IsOK());
  EXPECT_NEAR(out[0], -1.0f, 1e-3f);
  EXPECT_NEAR(out[2], 1.0f, 1e-3f);
}

TEST(QnnUnit_UtilsTest, DequantizePerChannel_Uint16CoversBranch) {
  std::vector<uint16_t> quant = {0, 32768, 65535};
  std::vector<uint32_t> shape = {3};
  std::vector<float> scales = {1.0f / 65535.0f};
  std::vector<int32_t> offsets = {0};
  std::vector<float> out(3);
  auto quant_bytes = gsl::make_span(
      reinterpret_cast<const uint8_t*>(quant.data()), quant.size() * sizeof(uint16_t));
  Ort::Status st = qnn::utils::DequantizePerChannel(quant_bytes, shape, scales, offsets, out,
                                                    QNN_DATATYPE_UFIXED_POINT_16);
  EXPECT_TRUE(st.IsOK());
  EXPECT_NEAR(out[0], 0.0f, 1e-4f);
  EXPECT_NEAR(out[2], 1.0f, 1e-4f);
}

TEST(QnnUnit_UtilsTest, DequantizePerChannel_Int32CoversBranch) {
  std::vector<int32_t> quant = {-100000, 0, 100000};
  std::vector<uint32_t> shape = {3};
  std::vector<float> scales = {1e-5f};
  std::vector<int32_t> offsets = {0};
  std::vector<float> out(3);
  auto quant_bytes = gsl::make_span(
      reinterpret_cast<const uint8_t*>(quant.data()), quant.size() * sizeof(int32_t));
  Ort::Status st = qnn::utils::DequantizePerChannel(quant_bytes, shape, scales, offsets, out,
                                                    QNN_DATATYPE_SFIXED_POINT_32);
  EXPECT_TRUE(st.IsOK());
  EXPECT_NEAR(out[0], -1.0f, 1e-3f);
  EXPECT_NEAR(out[2], 1.0f, 1e-3f);
}

TEST(QnnUnit_UtilsTest, DequantizePerChannel_UnsupportedTypeReturnsError) {
  std::vector<float> dummy(3, 0.5f);
  std::vector<uint32_t> shape = {3};
  std::vector<float> scales = {1.0f};
  std::vector<int32_t> offsets = {0};
  std::vector<float> out(3);
  auto quant_bytes = gsl::make_span(
      reinterpret_cast<const uint8_t*>(dummy.data()), dummy.size() * sizeof(float));
  Ort::Status st = qnn::utils::DequantizePerChannel(quant_bytes, shape, scales, offsets, out,
                                                    QNN_DATATYPE_FLOAT_32);
  EXPECT_FALSE(st.IsOK());
}

TEST(QnnUnit_UtilsTest, DequantizePerChannel_PerChannelAxis) {
  // shape = {2, 3} with axis=0: 2 channels of 3 elements each
  std::vector<int8_t> quant = {-1, 0, 1, 2, 3, 4};
  std::vector<uint32_t> shape = {2, 3};
  std::vector<float> scales = {0.1f, 0.1f};
  std::vector<int32_t> offsets = {0, 0};
  std::vector<float> out(6);
  auto quant_bytes = gsl::make_span(
      reinterpret_cast<const uint8_t*>(quant.data()), quant.size());
  Ort::Status st = qnn::utils::DequantizePerChannel(quant_bytes, shape, scales, offsets, out,
                                                    QNN_DATATYPE_SFIXED_POINT_8, 0LL);
  EXPECT_TRUE(st.IsOK());
  EXPECT_NEAR(out[0], -0.1f, 1e-4f);
  EXPECT_NEAR(out[3], 0.2f, 1e-4f);
}

// =============================================================================
// QuantizeData — per-channel axis path
// =============================================================================

TEST(QnnUnit_UtilsTest, QuantizeData_PerChannelAxis_Int8) {
  std::vector<float> data = {-0.1f, 0.1f, 0.2f, 0.4f};
  std::vector<uint32_t> shape = {2, 2};
  std::vector<float> scales = {0.01f, 0.01f};
  std::vector<int32_t> offsets = {0, 0};
  std::vector<uint8_t> quant_bytes(4);
  Ort::Status st = qnn::utils::QuantizeData(data, shape, scales, offsets, quant_bytes,
                                            QNN_DATATYPE_SFIXED_POINT_8, 0LL);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(static_cast<int8_t>(quant_bytes[0]), -10);
  EXPECT_EQ(static_cast<int8_t>(quant_bytes[1]), 10);
}

TEST(QnnUnit_UtilsTest, QuantizeData_PerChannelAxis_Int16) {
  std::vector<float> data = {-0.1f, 0.1f, 0.2f, 0.4f};
  std::vector<uint32_t> shape = {2, 2};
  std::vector<float> scales = {0.001f, 0.001f};
  std::vector<int32_t> offsets = {0, 0};
  std::vector<uint8_t> quant_bytes(4 * sizeof(int16_t));
  Ort::Status st = qnn::utils::QuantizeData(data, shape, scales, offsets, quant_bytes,
                                            QNN_DATATYPE_SFIXED_POINT_16, 0LL);
  EXPECT_TRUE(st.IsOK());
  int16_t val0 = 0;
  std::memcpy(&val0, quant_bytes.data(), sizeof(int16_t));
  EXPECT_EQ(val0, static_cast<int16_t>(-100));
}

TEST(QnnUnit_UtilsTest, QuantizeData_PerChannelAxis_Uint16) {
  std::vector<float> data = {0.0f, 0.1f, 0.5f, 1.0f};
  std::vector<uint32_t> shape = {2, 2};
  std::vector<float> scales = {1.0f / 65535.0f, 1.0f / 65535.0f};
  std::vector<int32_t> offsets = {0, 0};
  std::vector<uint8_t> quant_bytes(4 * sizeof(uint16_t));
  Ort::Status st = qnn::utils::QuantizeData(data, shape, scales, offsets, quant_bytes,
                                            QNN_DATATYPE_UFIXED_POINT_16, 0LL);
  EXPECT_TRUE(st.IsOK());
  uint16_t val0 = 1;
  std::memcpy(&val0, quant_bytes.data(), sizeof(uint16_t));
  EXPECT_EQ(val0, static_cast<uint16_t>(0));
}

// =============================================================================
// GetDataQuantParams — per-channel axis path
// =============================================================================

TEST(QnnUnit_UtilsTest, GetDataQuantParams_PerChannelAxis) {
  std::vector<float> data = {
      -1.0f,
      0.0f,
      0.5f,
      1.0f,
      -0.5f,
      0.0f,
      0.0f,
      0.5f,
      0.0f,
      0.0f,
      0.0f,
      1.0f,
  };
  std::vector<uint32_t> shape = {3, 4};
  std::vector<float> scales(3);
  std::vector<int32_t> offsets(3);
  Ort::Status st = qnn::utils::GetDataQuantParams(data, shape, scales, offsets,
                                                  QNN_DATATYPE_SFIXED_POINT_8, false, 0LL);
  EXPECT_TRUE(st.IsOK());
  EXPECT_GT(scales[0], 0.0f);
  EXPECT_GT(scales[1], 0.0f);
  EXPECT_GT(scales[2], 0.0f);
}

// =============================================================================
// UniqueNameGenerator — uniqueness contract and Reset()
// =============================================================================

class UniqueNameGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override { qnn::utils::UniqueNameGenerator().Reset(); }
};

TEST_F(UniqueNameGeneratorTest, UniqueNameGenerator_FirstCallReturnsBaseName) {
  EXPECT_EQ(qnn::utils::UniqueNameGenerator().New("tensor"), "tensor");
}

TEST_F(UniqueNameGeneratorTest, UniqueNameGenerator_DuplicateGetsCounter) {
  qnn::utils::UniqueNameGenerator().New("tensor");
  EXPECT_EQ(qnn::utils::UniqueNameGenerator().New("tensor"), "tensor_2");
}

TEST_F(UniqueNameGeneratorTest, UniqueNameGenerator_ResetClearsCounter) {
  qnn::utils::UniqueNameGenerator().New("tensor");
  qnn::utils::UniqueNameGenerator().Reset();
  EXPECT_EQ(qnn::utils::UniqueNameGenerator().New("tensor"), "tensor");
}

TEST_F(UniqueNameGeneratorTest, UniqueNameGenerator_WithSuffix) {
  EXPECT_EQ(qnn::utils::UniqueNameGenerator().New("conv", "_weight"), "conv_weight");
}

// =============================================================================
// GetPermToLastAxis — permutation correctness and error path
// =============================================================================

TEST(QnnUnit_UtilsTest, GetPermToLastAxis_MoveFirstToLast) {
  std::vector<uint32_t> perm;
  Ort::Status st = qnn::utils::GetPermToLastAxis(0u, 4u, perm);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(perm, (std::vector<uint32_t>{3u, 1u, 2u, 0u}));
}

TEST(QnnUnit_UtilsTest, GetPermToLastAxis_MoveMiddleToLast) {
  std::vector<uint32_t> perm;
  Ort::Status st = qnn::utils::GetPermToLastAxis(2u, 4u, perm);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(perm, (std::vector<uint32_t>{0u, 1u, 3u, 2u}));
}

TEST(QnnUnit_UtilsTest, GetPermToLastAxis_LastAxisIsIdentity) {
  std::vector<uint32_t> perm;
  Ort::Status st = qnn::utils::GetPermToLastAxis(3u, 4u, perm);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(perm, (std::vector<uint32_t>{0u, 1u, 2u, 3u}));
}

TEST(QnnUnit_UtilsTest, GetPermToLastAxis_AxisOutOfRangeFails) {
  std::vector<uint32_t> perm;
  Ort::Status st = qnn::utils::GetPermToLastAxis(4u, 4u, perm);
  EXPECT_FALSE(st.IsOK());
}

// =============================================================================
// CheckBiasScaleMatch — tolerance logic
// =============================================================================

TEST(QnnUnit_UtilsTest, CheckBiasScaleMatch_ExactMatchReturnsTrue) {
  EXPECT_TRUE(qnn::utils::CheckBiasScaleMatch(0.006f, 0.02f, 0.3f));
}

TEST(QnnUnit_UtilsTest, CheckBiasScaleMatch_WithinToleranceReturnsTrue) {
  // expected = 0.02f * 0.3f = 0.006f; diff = 1e-6f < default 1e-5f
  EXPECT_TRUE(qnn::utils::CheckBiasScaleMatch(0.006001f, 0.02f, 0.3f));
}

TEST(QnnUnit_UtilsTest, CheckBiasScaleMatch_OutsideToleranceReturnsFalse) {
  // expected = 0.006f; diff = 0.001f >> 1e-5f
  EXPECT_FALSE(qnn::utils::CheckBiasScaleMatch(0.007f, 0.02f, 0.3f));
}

// =============================================================================
// qnn::utils::TransposeFromNchwToHwcn(raw) / qnn::utils::TransposeFromCnhwToHwcn(raw)
// =============================================================================

// NCHW {N=1,C=2,H=2,W=2} → HWCN: each (h,w) slot gets [C0,C1] interleaved
TEST(QnnUnit_UtilsTest, TransposeFromNchwToHwcn_Raw_HappyPath) {
  std::vector<int64_t> shape = {1, 2, 2, 2};
  std::vector<uint8_t> input = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint8_t> output(input.size());
  Ort::Status st = qnn::utils::TransposeFromNchwToHwcn(std::move(shape), /*elem_byte_size=*/1,
                                                       input, output);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(output, (std::vector<uint8_t>{0, 4, 1, 5, 2, 6, 3, 7}));
}

TEST(QnnUnit_UtilsTest, TransposeFromNchwToHwcn_Raw_WrongRankFails) {
  std::vector<int64_t> shape = {2, 3, 4};
  std::vector<uint8_t> input(24, 0);
  std::vector<uint8_t> output(24);
  EXPECT_FALSE(qnn::utils::TransposeFromNchwToHwcn(std::move(shape), 1, input, output).IsOK());
}

TEST(QnnUnit_UtilsTest, TransposeFromNchwToHwcn_Raw_OutputSizeMismatchFails) {
  std::vector<int64_t> shape = {1, 2, 2, 2};
  std::vector<uint8_t> input = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint8_t> output(4);  // too small
  EXPECT_FALSE(qnn::utils::TransposeFromNchwToHwcn(std::move(shape), 1, input, output).IsOK());
}

// CNHW {C=2,N=1,H=2,W=2} → HWCN
TEST(QnnUnit_UtilsTest, TransposeFromCnhwToHwcn_Raw_HappyPath) {
  std::vector<int64_t> shape = {2, 1, 2, 2};
  std::vector<uint8_t> input = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint8_t> output(input.size());
  Ort::Status st = qnn::utils::TransposeFromCnhwToHwcn(std::move(shape), 1, input, output);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(output, (std::vector<uint8_t>{0, 4, 1, 5, 2, 6, 3, 7}));
}

TEST(QnnUnit_UtilsTest, TransposeFromCnhwToHwcn_Raw_WrongRankFails) {
  std::vector<int64_t> shape = {2, 3, 4};
  std::vector<uint8_t> input(24, 0);
  std::vector<uint8_t> output(24);
  EXPECT_FALSE(qnn::utils::TransposeFromCnhwToHwcn(std::move(shape), 1, input, output).IsOK());
}

// =============================================================================
// qnn::utils::TwoDimensionTranspose(raw)
//
// Element sizes 1/2/4/8 run a 32x32 tiled loop and everything else an untiled one, so the shape
// sweeps below cover partial tiles in either dimension, single rows/columns, and multi-tile extents.
// =============================================================================

// Distinct byte pattern per element, so a misplaced element is visible in the comparison.
static std::vector<uint8_t> MakeTransposeInput(size_t num_elems, size_t elem_byte_size) {
  std::vector<uint8_t> input(num_elems * elem_byte_size);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<uint8_t>(i % 251);  // Prime modulus: no aliasing with any tile or row stride.
  }
  return input;
}

static std::vector<uint8_t> ReferenceTranspose2D(size_t rows, size_t cols, size_t elem_byte_size,
                                                 const std::vector<uint8_t>& input) {
  std::vector<uint8_t> expected(input.size());
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      std::memcpy(&expected[(col * rows + row) * elem_byte_size],
                  &input[(row * cols + col) * elem_byte_size],
                  elem_byte_size);
    }
  }
  return expected;
}

TEST(QnnUnit_UtilsTest, TwoDimensionTranspose_Raw_HappyPath) {
  // [2, 3] -> [3, 2], two bytes per element.
  std::vector<uint8_t> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<uint8_t> output(input.size());
  Ort::Status st = qnn::utils::TwoDimensionTranspose(2, 3, /*elem_byte_size=*/2, input, output);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(output, (std::vector<uint8_t>{0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11}));
}

TEST(QnnUnit_UtilsTest, TwoDimensionTranspose_Raw_TiledElementSizesAcrossTileBoundaries) {
  // Empty, degenerate rank-2, just under / exactly one tile, a partial tile in one dimension only,
  // and partial tiles in both dimensions across several tiles.
  const std::vector<std::pair<size_t, size_t>> shapes = {
      {0, 0}, {0, 5}, {5, 0}, {1, 1}, {1, 70}, {70, 1}, {31, 31}, {32, 32}, {33, 31}, {31, 33}, {65, 97}};

  for (size_t elem_byte_size : {1u, 2u, 4u, 8u}) {
    for (const auto& [rows, cols] : shapes) {
      const std::vector<uint8_t> input = MakeTransposeInput(rows * cols, elem_byte_size);
      std::vector<uint8_t> output(input.size());
      Ort::Status st = qnn::utils::TwoDimensionTranspose(rows, cols, elem_byte_size, input, output);
      EXPECT_TRUE(st.IsOK()) << "[" << rows << ", " << cols << "] x " << elem_byte_size << " bytes";
      EXPECT_EQ(output, ReferenceTranspose2D(rows, cols, elem_byte_size, input))
          << "[" << rows << ", " << cols << "] x " << elem_byte_size << " bytes";
    }
  }
}

TEST(QnnUnit_UtilsTest, TwoDimensionTranspose_Raw_UncommonElementSizesUseUntiledPath) {
  for (size_t elem_byte_size : {3u, 5u, 6u, 7u, 16u}) {
    const std::vector<uint8_t> input = MakeTransposeInput(33 * 35, elem_byte_size);
    std::vector<uint8_t> output(input.size());
    Ort::Status st = qnn::utils::TwoDimensionTranspose(33, 35, elem_byte_size, input, output);
    EXPECT_TRUE(st.IsOK()) << elem_byte_size << " bytes per element";
    EXPECT_EQ(output, ReferenceTranspose2D(33, 35, elem_byte_size, input))
        << elem_byte_size << " bytes per element";
  }
}

TEST(QnnUnit_UtilsTest, TwoDimensionTranspose_Raw_BufferSizeMismatchFails) {
  const std::vector<uint8_t> input(2 * 3 * 4, 0);
  std::vector<uint8_t> output(input.size());
  EXPECT_TRUE(qnn::utils::TwoDimensionTranspose(2, 3, 4, input, output).IsOK());

  std::vector<uint8_t> short_output(input.size() - 4);
  EXPECT_FALSE(qnn::utils::TwoDimensionTranspose(2, 3, 4, input, short_output).IsOK());
  EXPECT_FALSE(qnn::utils::TwoDimensionTranspose(3, 3, 4, input, output).IsOK());
  EXPECT_FALSE(qnn::utils::TwoDimensionTranspose(2, 3, 0, input, output).IsOK());
}

// =============================================================================
// RequantizeBiasTensor — per-tensor and per-channel round-trip
// =============================================================================

TEST(QnnUnit_UtilsTest, RequantizeBiasTensor_PerTensor_RoundTrip) {
  std::vector<int32_t> int32_vals = {10, -20};
  std::vector<uint8_t> bias_data(int32_vals.size() * sizeof(int32_t));
  std::memcpy(bias_data.data(), int32_vals.data(), bias_data.size());

  std::vector<uint32_t> bias_shape = {2};
  std::vector<float> current_scales = {0.0001f};
  std::vector<int32_t> current_offsets = {0};
  std::vector<float> weights_scales = {0.01f};
  float activation_scale = 0.01f;  // new_scale = 0.01 * 0.01 = 0.0001 == current

  std::vector<uint8_t> requantized;
  std::vector<float> new_scales;
  std::vector<int32_t> new_offsets;
  Ort::Status st = qnn::utils::RequantizeBiasTensor(bias_data, bias_shape,
                                                    current_scales, current_offsets,
                                                    weights_scales, activation_scale,
                                                    QNN_DATATYPE_SFIXED_POINT_32,
                                                    requantized, new_scales, new_offsets);
  EXPECT_TRUE(st.IsOK());
  EXPECT_FLOAT_EQ(new_scales[0], 0.0001f);
  EXPECT_EQ(new_offsets[0], 0);
  std::vector<int32_t> out_vals(2);
  std::memcpy(out_vals.data(), requantized.data(), requantized.size());
  EXPECT_EQ(out_vals[0], 10);
  EXPECT_EQ(out_vals[1], -20);
}

TEST(QnnUnit_UtilsTest, RequantizeBiasTensor_PerChannel_ScalesUpdated) {
  std::vector<int32_t> int32_vals = {5, -10};
  std::vector<uint8_t> bias_data(int32_vals.size() * sizeof(int32_t));
  std::memcpy(bias_data.data(), int32_vals.data(), bias_data.size());

  std::vector<uint32_t> bias_shape = {2};
  std::vector<float> current_scales = {0.0002f, 0.0002f};
  std::vector<int32_t> current_offsets = {0, 0};
  std::vector<float> weights_scales = {0.02f, 0.02f};
  float activation_scale = 0.01f;  // new_scale[i] = 0.02 * 0.01 = 0.0002 == current

  std::vector<uint8_t> requantized;
  std::vector<float> new_scales;
  std::vector<int32_t> new_offsets;
  Ort::Status st = qnn::utils::RequantizeBiasTensor(bias_data, bias_shape,
                                                    current_scales, current_offsets,
                                                    weights_scales, activation_scale,
                                                    QNN_DATATYPE_SFIXED_POINT_32,
                                                    requantized, new_scales, new_offsets,
                                                    /*axis=*/0LL);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(new_scales.size(), 2u);
  EXPECT_FLOAT_EQ(new_scales[0], 0.0002f);
  EXPECT_FLOAT_EQ(new_scales[1], 0.0002f);
  std::vector<int32_t> out_vals(2);
  std::memcpy(out_vals.data(), requantized.data(), requantized.size());
  EXPECT_EQ(out_vals[0], 5);
  EXPECT_EQ(out_vals[1], -10);
}

// =============================================================================
// GetQnnErrorMessage / GetVerboseQnnErrorMessage
// =============================================================================

TEST(QnnUnit_UtilsTest, GetQnnErrorMessage_ReturnsNonEmptyString) {
  QnnRealHtpBackendContext backend;
  ASSERT_TRUE(backend.IsValid()) << "libQnnHtp.so not available";
  std::string msg = qnn::utils::GetQnnErrorMessage(backend.qnn_interface,
                                                   static_cast<Qnn_ErrorHandle_t>(1));
  EXPECT_FALSE(msg.empty());
}

TEST(QnnUnit_UtilsTest, GetVerboseQnnErrorMessage_ReturnsNonEmptyString) {
  QnnRealHtpBackendContext backend;
  ASSERT_TRUE(backend.IsValid()) << "libQnnHtp.so not available";
  std::string msg = qnn::utils::GetVerboseQnnErrorMessage(backend.qnn_interface,
                                                          static_cast<Qnn_ErrorHandle_t>(1));
  EXPECT_FALSE(msg.empty());
}

// =============================================================================
// GetTimeStampInUs
// =============================================================================

TEST(QnnUnit_UtilsTest, GetTimeStampInUs_ReturnsNonZero) {
  uint64_t t = qnn::utils::GetTimeStampInUs();
  EXPECT_GT(t, 0u);
}

TEST(QnnUnit_UtilsTest, GetTimeStampInUs_Monotonic) {
  uint64_t t1 = qnn::utils::GetTimeStampInUs();
  uint64_t t2 = qnn::utils::GetTimeStampInUs();
  EXPECT_GE(t2, t1);
}

// =============================================================================
// PtrToString
// =============================================================================

TEST(QnnUnit_UtilsTest, PtrToString_NullPointer) {
  std::string s = qnn::utils::PtrToString(nullptr);
  EXPECT_FALSE(s.empty());
}

TEST(QnnUnit_UtilsTest, PtrToString_NonNullPointer) {
  int x = 42;
  std::string s = qnn::utils::PtrToString(&x);
  EXPECT_FALSE(s.empty());
  // Two calls with the same pointer must return the same string.
  EXPECT_EQ(qnn::utils::PtrToString(&x), qnn::utils::PtrToString(&x));
}

// =============================================================================
// GetQuantParams
// =============================================================================

TEST(QnnUnit_UtilsTest, GetQuantParams_Uint8_Asymmetric_BasicRange) {
  float scale = 0.0f;
  int32_t zero_point = 0;
  // rmin=-1, rmax=1, UFIXED_POINT_8, asymmetric
  // scale = (1 - (-1)) / 255 ≈ 0.00784, initial_zp = 0 - (-1/scale) ≈ 127.5 → 128
  // zero_point is negated: -128
  auto st = qnn::utils::GetQuantParams(-1.0f, 1.0f, QNN_DATATYPE_UFIXED_POINT_8, scale, zero_point, false);
  EXPECT_TRUE(st.IsOK());
  EXPECT_NEAR(scale, 2.0f / 255.0f, 1e-5f);
  EXPECT_EQ(zero_point, -128);
}

TEST(QnnUnit_UtilsTest, GetQuantParams_Int8_Symmetric_BasicRange) {
  float scale = 0.0f;
  int32_t zero_point = 0;
  // rmin=-1, rmax=0.8 → abs_max=1, scale=2/254≈0.00787, zero_point=0
  auto st = qnn::utils::GetQuantParams(-1.0f, 0.8f, QNN_DATATYPE_SFIXED_POINT_8, scale, zero_point, true);
  EXPECT_TRUE(st.IsOK());
  EXPECT_GT(scale, 0.0f);
  EXPECT_EQ(zero_point, 0);  // symmetric → zero_point always 0
}

TEST(QnnUnit_UtilsTest, GetQuantParams_Int16_Asymmetric) {
  float scale = 0.0f;
  int32_t zero_point = 0;
  auto st = qnn::utils::GetQuantParams(0.0f, 1.0f, QNN_DATATYPE_UFIXED_POINT_16, scale, zero_point, false);
  EXPECT_TRUE(st.IsOK());
  EXPECT_NEAR(scale, 1.0f / 65535.0f, 1e-8f);
}

// =============================================================================
// NchwShapeToHwcn / CnhwShapeToHwcn — rank error paths
// =============================================================================

TEST(QnnUnit_UtilsTest, NchwShapeToHwcn_Rank4_HappyPath) {
  std::array<int64_t, 4> nchw = {2, 3, 4, 5};
  std::array<int64_t, 4> hwcn = {};
  auto st = qnn::utils::NchwShapeToHwcn<int64_t>(
      gsl::make_span(nchw.data(), nchw.size()),
      gsl::make_span(hwcn.data(), hwcn.size()));
  EXPECT_TRUE(st.IsOK());
  // H=4, W=5, C=3, N=2
  EXPECT_EQ(hwcn[0], 4);
  EXPECT_EQ(hwcn[1], 5);
  EXPECT_EQ(hwcn[2], 3);
  EXPECT_EQ(hwcn[3], 2);
}

TEST(QnnUnit_UtilsTest, NchwShapeToHwcn_Rank3_ReturnsError) {
  std::array<int64_t, 3> nchw = {2, 3, 4};
  std::array<int64_t, 3> hwcn = {};
  auto st = qnn::utils::NchwShapeToHwcn<int64_t>(
      gsl::make_span(nchw.data(), nchw.size()),
      gsl::make_span(hwcn.data(), hwcn.size()));
  EXPECT_FALSE(st.IsOK());
}

TEST(QnnUnit_UtilsTest, CnhwShapeToHwcn_Rank4_HappyPath) {
  std::array<int64_t, 4> cnhw = {2, 3, 4, 5};
  std::array<int64_t, 4> hwcn = {};
  auto st = qnn::utils::CnhwShapeToHwcn<int64_t>(
      gsl::make_span(cnhw.data(), cnhw.size()),
      gsl::make_span(hwcn.data(), hwcn.size()));
  EXPECT_TRUE(st.IsOK());
  // H=4, W=5, C=2, N=3
  EXPECT_EQ(hwcn[0], 4);
  EXPECT_EQ(hwcn[1], 5);
  EXPECT_EQ(hwcn[2], 2);
  EXPECT_EQ(hwcn[3], 3);
}

TEST(QnnUnit_UtilsTest, CnhwShapeToHwcn_Rank6_ReturnsError) {
  std::array<int64_t, 6> cnhw = {1, 2, 3, 4, 5, 6};
  std::array<int64_t, 6> hwcn = {};
  auto st = qnn::utils::CnhwShapeToHwcn<int64_t>(
      gsl::make_span(cnhw.data(), cnhw.size()),
      gsl::make_span(hwcn.data(), hwcn.size()));
  EXPECT_FALSE(st.IsOK());
}

// =============================================================================
// TransformUnsignedToSignedFixedPoint
// =============================================================================

TEST(QnnUnit_UtilsTest, TransformUnsignedToSigned_Bits8) {
  // mask = 0b10000000; each byte XOR'd with 128
  std::vector<uint8_t> data = {0x00, 0x7F, 0x80, 0xFF};
  auto st = qnn::utils::TransformUnsignedToSignedFixedPoint(data, 8);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(data[0], 0x80u);
  EXPECT_EQ(data[1], 0xFFu);
  EXPECT_EQ(data[2], 0x00u);
  EXPECT_EQ(data[3], 0x7Fu);
}

TEST(QnnUnit_UtilsTest, TransformUnsignedToSigned_Bits4) {
  // mask = 0b10001000
  std::vector<uint8_t> data = {0x00, 0xFF};
  auto st = qnn::utils::TransformUnsignedToSignedFixedPoint(data, 4);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(data[0], 0x88u);
  EXPECT_EQ(data[1], 0x77u);
}

TEST(QnnUnit_UtilsTest, TransformUnsignedToSigned_Bits2) {
  // mask = 0b10101010
  std::vector<uint8_t> data = {0x00};
  auto st = qnn::utils::TransformUnsignedToSignedFixedPoint(data, 2);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(data[0], 0xAAu);
}

TEST(QnnUnit_UtilsTest, TransformUnsignedToSigned_UnsupportedBitsReturnsError) {
  std::vector<uint8_t> data = {0x00};
  auto st = qnn::utils::TransformUnsignedToSignedFixedPoint(data, 3);
  EXPECT_FALSE(st.IsOK());
}

// =============================================================================
// UnpackInt4ToInt8 (Signed and Unsigned)
// =============================================================================

TEST(QnnUnit_UtilsTest, UnpackInt4ToInt8_Signed_BasicUnpack) {
  // INT4 packs two 4-bit values per byte.
  // 0x31 packs two INT4 values: lower nibble=1, upper nibble=3
  // After unpack + mask: values are 1 and 3 (lower 4 bits only)
  std::vector<uint8_t> data = {0x31};  // 1 packed byte → 2 INT4 values
  auto st = qnn::utils::UnpackInt4ToInt8<true>(2, data);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(data.size(), 2u);
  // Both values masked to lower 4 bits
  EXPECT_EQ(data[0] & 0x0F, data[0]);
  EXPECT_EQ(data[1] & 0x0F, data[1]);
}

TEST(QnnUnit_UtilsTest, UnpackInt4ToInt8_Unsigned_BasicUnpack) {
  std::vector<uint8_t> data = {0x52};  // two UINT4: lower=2, upper=5
  auto st = qnn::utils::UnpackInt4ToInt8<false>(2, data);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(data.size(), 2u);
}

// =============================================================================
// ConvertBlockQuantScalesToLpbq (basic smoke test)
// =============================================================================

TEST(QnnUnit_UtilsTest, ConvertBlockQuantScalesToLpbq_BasicSmoke) {
  // Only bitwidth==4 is supported; bq_scales layout is block-major: [b * num_channels + c]
  std::vector<float> bq_scales = {1.0f, 0.5f};  // 2 blocks, 1 channel
  std::vector<int32_t> bq_offsets = {0, 0};
  std::vector<float> per_channel_scales;
  std::vector<uint8_t> per_block_int_scales;
  std::vector<int32_t> offsets;
  auto st = qnn::utils::ConvertBlockQuantScalesToLpbq(
      gsl::make_span(bq_scales.data(), bq_scales.size()),
      gsl::make_span(bq_offsets.data(), bq_offsets.size()),
      /*num_blocks_per_channel=*/2u,
      /*num_channels=*/1u,
      /*bitwidth=*/4u,
      per_channel_scales, per_block_int_scales, offsets);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(per_channel_scales.size(), 1u);    // one per channel
  EXPECT_EQ(per_block_int_scales.size(), 2u);  // num_channels * num_blocks_per_channel
  EXPECT_EQ(offsets.size(), 1u);
}

TEST(QnnUnit_UtilsTest, ConvertBlockQuantScalesToLpbq_UnsupportedBitwdithReturnsError) {
  std::vector<float> bq_scales = {1.0f, 0.5f};
  std::vector<int32_t> bq_offsets;
  std::vector<float> per_channel_scales;
  std::vector<uint8_t> per_block_int_scales;
  std::vector<int32_t> offsets;
  auto st = qnn::utils::ConvertBlockQuantScalesToLpbq(
      gsl::make_span(bq_scales.data(), bq_scales.size()),
      gsl::make_span(bq_offsets.data(), bq_offsets.size()),
      /*num_blocks_per_channel=*/2u,
      /*num_channels=*/1u,
      /*bitwidth=*/8u,  // unsupported
      per_channel_scales, per_block_int_scales, offsets);
  EXPECT_FALSE(st.IsOK());
}

// =============================================================================
// BroadcastShape — right-aligned broadcasting rules
// =============================================================================

TEST(QnnUnit_UtilsTest, BroadcastShape_SameShape) {
  std::vector<uint32_t> a = {2, 3, 4};
  std::vector<uint32_t> b = {2, 3, 4};
  std::vector<uint32_t> out;
  auto st = qnn::utils::BroadcastShape<uint32_t>(a, b, out);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(out, (std::vector<uint32_t>{2, 3, 4}));
}

TEST(QnnUnit_UtilsTest, BroadcastShape_DimOneBroadcasts) {
  // {1, 3} vs {2, 1} → {2, 3}
  std::vector<uint32_t> a = {1, 3};
  std::vector<uint32_t> b = {2, 1};
  std::vector<uint32_t> out;
  auto st = qnn::utils::BroadcastShape<uint32_t>(a, b, out);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(out, (std::vector<uint32_t>{2, 3}));
}

TEST(QnnUnit_UtilsTest, BroadcastShape_DifferentRanksRightAligned) {
  // {3} vs {2, 3} → {2, 3}
  std::vector<uint32_t> a = {3};
  std::vector<uint32_t> b = {2, 3};
  std::vector<uint32_t> out;
  auto st = qnn::utils::BroadcastShape<uint32_t>(a, b, out);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(out, (std::vector<uint32_t>{2, 3}));
}

TEST(QnnUnit_UtilsTest, BroadcastShape_HigherRankMixed) {
  // {5, 1, 4} vs {3, 1} → {5, 3, 4}
  std::vector<uint32_t> a = {5, 1, 4};
  std::vector<uint32_t> b = {3, 1};
  std::vector<uint32_t> out;
  auto st = qnn::utils::BroadcastShape<uint32_t>(a, b, out);
  EXPECT_TRUE(st.IsOK());
  EXPECT_EQ(out, (std::vector<uint32_t>{5, 3, 4}));
}

// =============================================================================
// SignExtendUnpackedSubByteData
//
// UnpackInitializerData() expands a sub-byte initializer to one byte per element with the unused
// high bits masked off, which QNN needs but which reads as a positive number in any int8_t-based
// decode. These tests pin the recovery of the original two's-complement value.
// =============================================================================

TEST(QnnUnit_UtilsTest, SignExtendUnpackedSubByteData_Int4_RecoversNegativeValues) {
  // Every representable INT4 value, so both nibble positions and the whole negative half are hit.
  const std::vector<int8_t> values{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint8_t> bytes, expected;
  for (int8_t v : values) {
    bytes.push_back(static_cast<uint8_t>(v) & 0x0F);  // as UnpackInitializerData() leaves it
    expected.push_back(static_cast<uint8_t>(v));
  }
  qnn::utils::SignExtendUnpackedSubByteData(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
                                            gsl::make_span(bytes));
  EXPECT_EQ(bytes, expected);
}

TEST(QnnUnit_UtilsTest, SignExtendUnpackedSubByteData_Int2_RecoversNegativeValues) {
  const std::vector<int8_t> values{-2, -1, 0, 1};
  std::vector<uint8_t> bytes, expected;
  for (int8_t v : values) {
    bytes.push_back(static_cast<uint8_t>(v) & 0x03);
    expected.push_back(static_cast<uint8_t>(v));
  }
  qnn::utils::SignExtendUnpackedSubByteData(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2,
                                            gsl::make_span(bytes));
  EXPECT_EQ(bytes, expected);
}

TEST(QnnUnit_UtilsTest, SignExtendUnpackedSubByteData_UnsignedAndByteWideTypesAreLeftAlone) {
  // UINT4 / UINT2 masked bytes already hold the value, and types stored one element per their own
  // width were never masked, so all of these must pass through untouched.
  const std::vector<uint8_t> original{0x00, 0x03, 0x0F, 0x80, 0xFF};
  for (ONNXTensorElementDataType type : {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}) {
    std::vector<uint8_t> bytes = original;
    qnn::utils::SignExtendUnpackedSubByteData(type, gsl::make_span(bytes));
    EXPECT_EQ(bytes, original) << "type " << static_cast<int>(type) << " must not be modified";
  }
}

TEST(QnnUnit_UtilsTest, SignExtendUnpackedSubByteData_EmptySpanIsSafe) {
  std::vector<uint8_t> bytes;
  qnn::utils::SignExtendUnpackedSubByteData(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
                                            gsl::make_span(bytes));
  EXPECT_TRUE(bytes.empty());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
