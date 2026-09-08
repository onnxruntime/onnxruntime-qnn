// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for qnn_def.cc — QNN tensor utility functions and
// backend-type predicates.
//
// These tests exercise pure logic that does not invoke any QNN SDK runtime
// functions, so no QNN backend, hardware, or emulator is required.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <unordered_map>
#include <vector>

#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace test {

// =============================================================================
// memscpy
// =============================================================================

TEST(QnnUnit_DefTest, Memscpy_FullCopy) {
  char dst[8] = {};
  const char src[] = "hello";
  size_t copied = qnn::memscpy(dst, sizeof(dst), src, sizeof(src));
  EXPECT_EQ(copied, sizeof(src));
  EXPECT_STREQ(dst, "hello");
}

TEST(QnnUnit_DefTest, Memscpy_TruncatesWhenDstSmaller) {
  char dst[3] = {};
  const char src[] = "hello";
  size_t copied = qnn::memscpy(dst, 3, src, sizeof(src));
  EXPECT_EQ(copied, 3u);
  EXPECT_EQ(dst[0], 'h');
  EXPECT_EQ(dst[1], 'e');
  EXPECT_EQ(dst[2], 'l');
}

TEST(QnnUnit_DefTest, Memscpy_NullDstReturnsZero) {
  const char src[] = "hello";
  size_t copied = qnn::memscpy(nullptr, 8, src, sizeof(src));
  EXPECT_EQ(copied, 0u);
}

TEST(QnnUnit_DefTest, Memscpy_NullSrcReturnsZero) {
  char dst[8] = {};
  size_t copied = qnn::memscpy(dst, sizeof(dst), nullptr, 5);
  EXPECT_EQ(copied, 0u);
}

TEST(QnnUnit_DefTest, Memscpy_ZeroDstSizeReturnsZero) {
  char dst[8] = {};
  const char src[] = "hello";
  size_t copied = qnn::memscpy(dst, 0, src, sizeof(src));
  EXPECT_EQ(copied, 0u);
}

TEST(QnnUnit_DefTest, Memscpy_ZeroCopySizeReturnsZero) {
  char dst[8] = {};
  const char src[] = "hello";
  size_t copied = qnn::memscpy(dst, sizeof(dst), src, 0);
  EXPECT_EQ(copied, 0u);
}

// =============================================================================
// SetQnnTensorType / GetQnnTensorType
// =============================================================================

TEST(QnnUnit_DefTest, SetGetQnnTensorType_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  qnn::SetQnnTensorType(t, QNN_TENSOR_TYPE_APP_WRITE);
  EXPECT_EQ(qnn::GetQnnTensorType(t), QNN_TENSOR_TYPE_APP_WRITE);

  qnn::SetQnnTensorType(t, QNN_TENSOR_TYPE_APP_READ);
  EXPECT_EQ(qnn::GetQnnTensorType(t), QNN_TENSOR_TYPE_APP_READ);
}

TEST(QnnUnit_DefTest, SetQnnTensorType_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);  // unknown version
  EXPECT_THROW(qnn::SetQnnTensorType(t, QNN_TENSOR_TYPE_APP_WRITE), Ort::Exception);
}

TEST(QnnUnit_DefTest, GetQnnTensorType_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorType(t), Ort::Exception);
}

// =============================================================================
// SetQnnTensorDataType / GetQnnTensorDataType
// =============================================================================

TEST(QnnUnit_DefTest, SetGetQnnTensorDataType_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  qnn::SetQnnTensorDataType(t, QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(qnn::GetQnnTensorDataType(t), QNN_DATATYPE_FLOAT_32);

  qnn::SetQnnTensorDataType(t, QNN_DATATYPE_INT_8);
  EXPECT_EQ(qnn::GetQnnTensorDataType(t), QNN_DATATYPE_INT_8);
}

TEST(QnnUnit_DefTest, SetQnnTensorDataType_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::SetQnnTensorDataType(t, QNN_DATATYPE_FLOAT_32), Ort::Exception);
}

// =============================================================================
// SetQnnTensorMemType / GetQnnTensorMemType
// =============================================================================

TEST(QnnUnit_DefTest, SetGetQnnTensorMemType_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  qnn::SetQnnTensorMemType(t, QNN_TENSORMEMTYPE_RAW);
  EXPECT_EQ(qnn::GetQnnTensorMemType(t), QNN_TENSORMEMTYPE_RAW);

  qnn::SetQnnTensorMemType(t, QNN_TENSORMEMTYPE_MEMHANDLE);
  EXPECT_EQ(qnn::GetQnnTensorMemType(t), QNN_TENSORMEMTYPE_MEMHANDLE);
}

TEST(QnnUnit_DefTest, SetQnnTensorMemType_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::SetQnnTensorMemType(t, QNN_TENSORMEMTYPE_RAW), Ort::Exception);
}

// =============================================================================
// SetQnnTensorName
// =============================================================================

TEST(QnnUnit_DefTest, SetQnnTensorName_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  const char* name = "my_tensor";
  qnn::SetQnnTensorName(t, name);
  EXPECT_STREQ(t.v1.name, "my_tensor");
}

TEST(QnnUnit_DefTest, SetQnnTensorName_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::SetQnnTensorName(t, "x"), Ort::Exception);
}

// =============================================================================
// SetQnnTensorDim
// =============================================================================

TEST(QnnUnit_DefTest, SetQnnTensorDim_V1_SetsRankAndPointer) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  std::vector<uint32_t> dims = {1, 3, 224, 224};
  qnn::SetQnnTensorDim(t, dims);
  EXPECT_EQ(t.v1.rank, 4u);
  EXPECT_EQ(t.v1.dimensions, dims.data());
}

TEST(QnnUnit_DefTest, SetQnnTensorDim_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  std::vector<uint32_t> dims = {1};
  EXPECT_THROW(qnn::SetQnnTensorDim(t, dims), Ort::Exception);
}

// =============================================================================
// SetQnnTensorQParams
// =============================================================================

TEST(QnnUnit_DefTest, SetQnnTensorQParams_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  Qnn_QuantizeParams_t qp = QNN_QUANTIZE_PARAMS_INIT;
  qp.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp.scaleOffsetEncoding.scale = 0.5f;
  qp.scaleOffsetEncoding.offset = -128;

  qnn::SetQnnTensorQParams(t, qp);

  EXPECT_EQ(t.v1.quantizeParams.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(t.v1.quantizeParams.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_FLOAT_EQ(t.v1.quantizeParams.scaleOffsetEncoding.scale, 0.5f);
  EXPECT_EQ(t.v1.quantizeParams.scaleOffsetEncoding.offset, -128);
}

// =============================================================================
// SetQnnTensorClientBuf (vector overload)
// =============================================================================

TEST(QnnUnit_DefTest, SetQnnTensorClientBuf_Vector_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  std::vector<uint8_t> buf = {1, 2, 3, 4};
  qnn::SetQnnTensorClientBuf(t, buf);
  EXPECT_EQ(t.v1.clientBuf.data, static_cast<void*>(buf.data()));
  EXPECT_EQ(t.v1.clientBuf.dataSize, static_cast<uint32_t>(buf.size()));
}

// =============================================================================
// SetQnnTensorClientBuf (raw pointer overload)
// =============================================================================

TEST(QnnUnit_DefTest, SetQnnTensorClientBuf_RawPtr_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  uint8_t data[4] = {10, 20, 30, 40};
  qnn::SetQnnTensorClientBuf(t, data, sizeof(data));
  EXPECT_EQ(t.v1.clientBuf.data, static_cast<void*>(data));
  EXPECT_EQ(t.v1.clientBuf.dataSize, static_cast<uint32_t>(sizeof(data)));
}

// =============================================================================
// SetQnnTensorClientBufSize
// =============================================================================

TEST(QnnUnit_DefTest, SetQnnTensorClientBufSize_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  qnn::SetQnnTensorClientBufSize(t, 1024u);
  EXPECT_EQ(t.v1.clientBuf.dataSize, 1024u);
}

// =============================================================================
// SetQnnTensorClientBufData
// =============================================================================

TEST(QnnUnit_DefTest, SetQnnTensorClientBufData_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  uint8_t sentinel = 42;
  qnn::SetQnnTensorClientBufData(t, &sentinel);
  EXPECT_EQ(t.v1.clientBuf.data, static_cast<void*>(&sentinel));
}

// =============================================================================
// Backend type predicates
// =============================================================================

TEST(QnnUnit_DefTest, IsCpuBackend) {
  EXPECT_TRUE(qnn::IsCpuBackend(qnn::QnnBackendType::CPU));
  EXPECT_FALSE(qnn::IsCpuBackend(qnn::QnnBackendType::HTP));
  EXPECT_FALSE(qnn::IsCpuBackend(qnn::QnnBackendType::GPU));
  EXPECT_FALSE(qnn::IsCpuBackend(qnn::QnnBackendType::DSP));
  EXPECT_FALSE(qnn::IsCpuBackend(qnn::QnnBackendType::SERIALIZER));
}

TEST(QnnUnit_DefTest, IsNpuBackend) {
  EXPECT_TRUE(qnn::IsNpuBackend(qnn::QnnBackendType::HTP));
  EXPECT_TRUE(qnn::IsNpuBackend(qnn::QnnBackendType::DSP));
  EXPECT_FALSE(qnn::IsNpuBackend(qnn::QnnBackendType::CPU));
  EXPECT_FALSE(qnn::IsNpuBackend(qnn::QnnBackendType::GPU));
  EXPECT_FALSE(qnn::IsNpuBackend(qnn::QnnBackendType::SERIALIZER));
}

TEST(QnnUnit_DefTest, IsGpuBackend) {
  EXPECT_TRUE(qnn::IsGpuBackend(qnn::QnnBackendType::GPU));
  EXPECT_FALSE(qnn::IsGpuBackend(qnn::QnnBackendType::CPU));
  EXPECT_FALSE(qnn::IsGpuBackend(qnn::QnnBackendType::HTP));
}

TEST(QnnUnit_DefTest, IsIrBackend) {
  EXPECT_TRUE(qnn::IsIrBackend(qnn::QnnBackendType::SERIALIZER));
  EXPECT_FALSE(qnn::IsIrBackend(qnn::QnnBackendType::CPU));
  EXPECT_FALSE(qnn::IsIrBackend(qnn::QnnBackendType::HTP));
}

TEST(QnnUnit_DefTest, IsQpuBackend_NpuOrGpu) {
  EXPECT_TRUE(qnn::IsQpuBackend(qnn::QnnBackendType::HTP));
  EXPECT_TRUE(qnn::IsQpuBackend(qnn::QnnBackendType::DSP));
  EXPECT_TRUE(qnn::IsQpuBackend(qnn::QnnBackendType::GPU));
  EXPECT_FALSE(qnn::IsQpuBackend(qnn::QnnBackendType::CPU));
  EXPECT_FALSE(qnn::IsQpuBackend(qnn::QnnBackendType::SERIALIZER));
}

// =============================================================================
// GetQnnTensorID
// =============================================================================

TEST(QnnUnit_DefTest, GetQnnTensorID_V1_ReturnsId) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.v1.id = 42u;
  EXPECT_EQ(qnn::GetQnnTensorID(t), 42u);
}

TEST(QnnUnit_DefTest, GetQnnTensorID_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorID(t), Ort::Exception);
}

// =============================================================================
// GetQnnTensorName
// =============================================================================

TEST(QnnUnit_DefTest, GetQnnTensorName_V1_ReturnsName) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  const char* name = "test_name";
  qnn::SetQnnTensorName(t, name);
  EXPECT_STREQ(qnn::GetQnnTensorName(t), "test_name");
}

TEST(QnnUnit_DefTest, GetQnnTensorName_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorName(t), Ort::Exception);
}

// =============================================================================
// SetQnnTensorDataFormat / GetQnnTensorDataFormat
// =============================================================================

TEST(QnnUnit_DefTest, SetGetQnnTensorDataFormat_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  qnn::SetQnnTensorDataFormat(t, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
  EXPECT_EQ(qnn::GetQnnTensorDataFormat(t), QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
}

TEST(QnnUnit_DefTest, SetQnnTensorDataFormat_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::SetQnnTensorDataFormat(t, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER), Ort::Exception);
}

TEST(QnnUnit_DefTest, GetQnnTensorDataFormat_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorDataFormat(t), Ort::Exception);
}

// =============================================================================
// GetQnnTensorRank
// =============================================================================

TEST(QnnUnit_DefTest, GetQnnTensorRank_V1_ReturnsRank) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  std::vector<uint32_t> dims = {1, 3, 224, 224};
  qnn::SetQnnTensorDim(t, dims);
  EXPECT_EQ(qnn::GetQnnTensorRank(t), 4u);
}

TEST(QnnUnit_DefTest, GetQnnTensorRank_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorRank(t), Ort::Exception);
}

// =============================================================================
// GetQnnTensorDims
// =============================================================================

TEST(QnnUnit_DefTest, GetQnnTensorDims_V1_ReturnsDimsPointer) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  std::vector<uint32_t> dims = {2, 3};
  qnn::SetQnnTensorDim(t, dims);
  uint32_t* got = qnn::GetQnnTensorDims(t);
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(got[0], 2u);
  EXPECT_EQ(got[1], 3u);
}

TEST(QnnUnit_DefTest, GetQnnTensorDims_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorDims(t), Ort::Exception);
}

// =============================================================================
// GetQnnTensorClientBuf
// =============================================================================

TEST(QnnUnit_DefTest, GetQnnTensorClientBuf_V1_ReturnsClientBuf) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  uint8_t data[4] = {1, 2, 3, 4};
  qnn::SetQnnTensorClientBuf(t, static_cast<void*>(data), static_cast<uint32_t>(sizeof(data)));
  const auto& buf = qnn::GetQnnTensorClientBuf(t);
  EXPECT_EQ(buf.data, static_cast<void*>(data));
  EXPECT_EQ(buf.dataSize, 4u);
}

TEST(QnnUnit_DefTest, GetQnnTensorClientBuf_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorClientBuf(t), Ort::Exception);
}

// =============================================================================
// SetQnnTensorMemHandle / GetQnnTensorMemHandle
// =============================================================================

TEST(QnnUnit_DefTest, SetGetQnnTensorMemHandle_V1) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  int dummy = 0;
  Qnn_MemHandle_t handle = static_cast<Qnn_MemHandle_t>(&dummy);
  qnn::SetQnnTensorMemHandle(t, handle);
  EXPECT_EQ(qnn::GetQnnTensorMemHandle(t), handle);
}

TEST(QnnUnit_DefTest, SetQnnTensorMemHandle_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::SetQnnTensorMemHandle(t, nullptr), Ort::Exception);
}

TEST(QnnUnit_DefTest, GetQnnTensorMemHandle_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorMemHandle(t), Ort::Exception);
}

// =============================================================================
// GetQnnTensorQParams
// =============================================================================

TEST(QnnUnit_DefTest, GetQnnTensorQParams_V1_ReturnsQParams) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  Qnn_QuantizeParams_t qp = QNN_QUANTIZE_PARAMS_INIT;
  qp.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp.scaleOffsetEncoding.scale = 1.5f;
  qp.scaleOffsetEncoding.offset = -64;
  qnn::SetQnnTensorQParams(t, qp);
  const auto& got = qnn::GetQnnTensorQParams(t);
  EXPECT_EQ(got.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_FLOAT_EQ(got.scaleOffsetEncoding.scale, 1.5f);
  EXPECT_EQ(got.scaleOffsetEncoding.offset, -64);
}

TEST(QnnUnit_DefTest, GetQnnTensorQParams_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorQParams(t), Ort::Exception);
}

// =============================================================================
// GetQnnTensorIsDynamicDimensions
// =============================================================================

TEST(QnnUnit_DefTest, GetQnnTensorIsDynamicDimensions_V1_ReturnsNull) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  EXPECT_EQ(qnn::GetQnnTensorIsDynamicDimensions(t), nullptr);
}

TEST(QnnUnit_DefTest, GetQnnTensorIsDynamicDimensions_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorIsDynamicDimensions(t), Ort::Exception);
}

// =============================================================================
// Missing InvalidVersionThrows for already-covered setters
// =============================================================================

TEST(QnnUnit_DefTest, SetQnnTensorClientBuf_Vector_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  std::vector<uint8_t> buf = {1, 2};
  EXPECT_THROW(qnn::SetQnnTensorClientBuf(t, buf), Ort::Exception);
}

TEST(QnnUnit_DefTest, SetQnnTensorClientBuf_RawPtr_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  uint8_t data[2] = {};
  EXPECT_THROW(qnn::SetQnnTensorClientBuf(t, static_cast<void*>(data), 2u), Ort::Exception);
}

TEST(QnnUnit_DefTest, SetQnnTensorClientBufSize_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::SetQnnTensorClientBufSize(t, 64u), Ort::Exception);
}

TEST(QnnUnit_DefTest, SetQnnTensorClientBufData_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::SetQnnTensorClientBufData(t, nullptr), Ort::Exception);
}

TEST(QnnUnit_DefTest, SetQnnTensorQParams_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  Qnn_QuantizeParams_t qp = QNN_QUANTIZE_PARAMS_INIT;
  EXPECT_THROW(qnn::SetQnnTensorQParams(t, qp), Ort::Exception);
}

TEST(QnnUnit_DefTest, GetQnnTensorDataType_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorDataType(t), Ort::Exception);
}

TEST(QnnUnit_DefTest, GetQnnTensorMemType_InvalidVersionThrows) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = static_cast<Qnn_TensorVersion_t>(0);
  EXPECT_THROW(qnn::GetQnnTensorMemType(t), Ort::Exception);
}

// =============================================================================
// CalcQnnTensorNumElems
// =============================================================================

TEST(QnnUnit_DefTest, CalcQnnTensorNumElems_1D) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  std::vector<uint32_t> dims = {7};
  qnn::SetQnnTensorDim(t, dims);
  EXPECT_EQ(qnn::CalcQnnTensorNumElems(t), 7u);
}

TEST(QnnUnit_DefTest, CalcQnnTensorNumElems_MultiDim) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  std::vector<uint32_t> dims = {2, 3, 4};
  qnn::SetQnnTensorDim(t, dims);
  EXPECT_EQ(qnn::CalcQnnTensorNumElems(t), 24u);
}

// =============================================================================
// CompareQnnQuantParams
// =============================================================================

TEST(QnnUnit_DefTest, CompareQnnQuantParams_BothUndefined_ZeroDiffs) {
  Qnn_QuantizeParams_t qp0 = QNN_QUANTIZE_PARAMS_INIT;  // QNN_DEFINITION_UNDEFINED
  Qnn_QuantizeParams_t qp1 = QNN_QUANTIZE_PARAMS_INIT;
  float sd = 99.f;
  int32_t od = 99;
  auto s = qnn::CompareQnnQuantParams(qp0, qp1, sd, od);
  EXPECT_FALSE(bool(s));  // success
  EXPECT_FLOAT_EQ(sd, 0.f);
  EXPECT_EQ(od, 0);
}

TEST(QnnUnit_DefTest, CompareQnnQuantParams_ScaleOffset_NoDiff) {
  Qnn_QuantizeParams_t qp = QNN_QUANTIZE_PARAMS_INIT;
  qp.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp.scaleOffsetEncoding.scale = 0.5f;
  qp.scaleOffsetEncoding.offset = -128;
  float sd = 0.f;
  int32_t od = 0;
  auto s = qnn::CompareQnnQuantParams(qp, qp, sd, od);
  EXPECT_FALSE(bool(s));  // success
  EXPECT_FLOAT_EQ(sd, 0.f);
  EXPECT_EQ(od, 0);
}

TEST(QnnUnit_DefTest, CompareQnnQuantParams_ScaleOffset_WithDiff) {
  Qnn_QuantizeParams_t qp0 = QNN_QUANTIZE_PARAMS_INIT;
  qp0.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp0.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp0.scaleOffsetEncoding.scale = 1.0f;
  qp0.scaleOffsetEncoding.offset = 0;

  Qnn_QuantizeParams_t qp1 = QNN_QUANTIZE_PARAMS_INIT;
  qp1.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp1.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp1.scaleOffsetEncoding.scale = 0.5f;
  qp1.scaleOffsetEncoding.offset = -10;

  float sd = 0.f;
  int32_t od = 0;
  auto s = qnn::CompareQnnQuantParams(qp0, qp1, sd, od);
  EXPECT_FALSE(bool(s));  // success — diff is valid output, not an error
  EXPECT_FLOAT_EQ(sd, 0.5f);
  EXPECT_EQ(od, 10);
}

TEST(QnnUnit_DefTest, CompareQnnQuantParams_MismatchedTypes_ReturnsError) {
  Qnn_QuantizeParams_t qp0 = QNN_QUANTIZE_PARAMS_INIT;
  qp0.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp0.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;

  Qnn_QuantizeParams_t qp1 = QNN_QUANTIZE_PARAMS_INIT;
  qp1.encodingDefinition = QNN_DEFINITION_UNDEFINED;  // mismatch

  float sd = 0.f;
  int32_t od = 0;
  auto s = qnn::CompareQnnQuantParams(qp0, qp1, sd, od);
  EXPECT_TRUE(bool(s));  // failure
}

TEST(QnnUnit_DefTest, CompareQnnQuantParams_UnsupportedEncoding_ReturnsError) {
  Qnn_QuantizeParams_t qp = QNN_QUANTIZE_PARAMS_INIT;
  qp.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;  // not SCALE_OFFSET

  float sd = 0.f;
  int32_t od = 0;
  auto s = qnn::CompareQnnQuantParams(qp, qp, sd, od);
  EXPECT_TRUE(bool(s));  // failure — unsupported encoding
}

// =============================================================================
// CreateTensorInQnnGraph — paths that do not invoke the QNN SDK
// =============================================================================

TEST(QnnUnit_DefTest, CreateTensorInQnnGraph_AlreadyExists_ReturnsTrueWithMsg) {
  QNN_INTERFACE_VER_TYPE qnn_iface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_GraphHandle_t graph = nullptr;
  std::unordered_map<std::string, uint32_t> table = {{"my_tensor", 42u}};
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  std::string err;

  bool result = qnn::CreateTensorInQnnGraph(qnn_iface, graph,
                                            "node0", "my_tensor", t, table, err);
  // When the tensor already exists the function returns true and copies the
  // previously assigned ID into the tensor.  No error message is produced.
  EXPECT_TRUE(result);
  EXPECT_TRUE(err.empty());
  EXPECT_EQ(qnn::GetQnnTensorID(t), 42u);
}

TEST(QnnUnit_DefTest, CreateTensorInQnnGraph_StaticTensor_WrongMemType_ReturnsFalse) {
  QNN_INTERFACE_VER_TYPE qnn_iface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_GraphHandle_t graph = nullptr;
  std::unordered_map<std::string, uint32_t> table;
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  qnn::SetQnnTensorType(t, QNN_TENSOR_TYPE_STATIC);
  qnn::SetQnnTensorDataType(t, QNN_DATATYPE_FLOAT_32);
  qnn::SetQnnTensorMemType(t, QNN_TENSORMEMTYPE_MEMHANDLE);  // wrong for static
  std::string err;

  bool result = qnn::CreateTensorInQnnGraph(qnn_iface, graph,
                                            "node0", "static_tensor", t, table, err);
  EXPECT_FALSE(result);
  EXPECT_FALSE(err.empty());
}

TEST(QnnUnit_DefTest, CreateTensorInQnnGraph_StaticTensor_SizeMismatch_ReturnsFalse) {
  QNN_INTERFACE_VER_TYPE qnn_iface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_GraphHandle_t graph = nullptr;
  std::unordered_map<std::string, uint32_t> table;
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  // dims {2}, float32 → expected data size = 2 * 4 = 8 bytes
  std::vector<uint32_t> dims = {2};
  qnn::SetQnnTensorType(t, QNN_TENSOR_TYPE_STATIC);
  qnn::SetQnnTensorDataType(t, QNN_DATATYPE_FLOAT_32);
  qnn::SetQnnTensorMemType(t, QNN_TENSORMEMTYPE_RAW);
  qnn::SetQnnTensorDim(t, dims);
  qnn::SetQnnTensorClientBufSize(t, 4u);  // mismatch: expected 8, got 4
  std::string err;

  bool result = qnn::CreateTensorInQnnGraph(qnn_iface, graph,
                                            "node0", "static_tensor", t, table, err);
  EXPECT_FALSE(result);
  EXPECT_FALSE(err.empty());
}

// =============================================================================
// QnnParamWrapper::CreateQnnGraphParam — scalar path (no SDK call)
// =============================================================================

TEST(QnnUnit_DefTest, QnnParamWrapper_CreateQnnGraphParam_Scalar_ReturnsTrueWithMsg) {
  Qnn_Scalar_t scalar{};
  scalar.dataType = QNN_DATATYPE_UINT_32;
  scalar.uint32Value = 7u;
  qnn::QnnParamWrapper param(0, "node0", "stride", scalar);

  QNN_INTERFACE_VER_TYPE qnn_iface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_GraphHandle_t graph = nullptr;
  std::unordered_map<std::string, uint32_t> table;
  std::string err;

  bool result = param.CreateQnnGraphParam(qnn_iface, graph, "node0", table, err);
  EXPECT_TRUE(result);
  EXPECT_FALSE(err.empty());  // "Add scalar parameter: stride"
}

// =============================================================================
// QnnOpConfigWrapper constructor — covers SetNames, SetNums, SetData
// =============================================================================

TEST(QnnUnit_DefTest, QnnOpConfigWrapper_Constructor_SetsV1Fields) {
  std::vector<Qnn_Tensor_t> inputs = {QNN_TENSOR_INIT};
  std::vector<Qnn_Tensor_t> outputs = {QNN_TENSOR_INIT, QNN_TENSOR_INIT};
  std::vector<Qnn_Param_t> params;

  qnn::QnnOpConfigWrapper op("relu0", "qti.aisw", "Relu",
                             std::move(inputs), std::move(outputs),
                             std::move(params));

  const auto& cfg = op.GetQnnOpConfig();
  ASSERT_EQ(cfg.version, QNN_OPCONFIG_VERSION_1);
  EXPECT_STREQ(cfg.v1.name, "relu0");
  EXPECT_STREQ(cfg.v1.packageName, "qti.aisw");
  EXPECT_STREQ(cfg.v1.typeName, "Relu");
  EXPECT_EQ(cfg.v1.numOfInputs, 1u);
  EXPECT_EQ(cfg.v1.numOfOutputs, 2u);
  EXPECT_EQ(cfg.v1.numOfParams, 0u);
  EXPECT_NE(cfg.v1.inputTensors, nullptr);
  EXPECT_NE(cfg.v1.outputTensors, nullptr);
}

// =============================================================================
// V2 tensor paths (QNN_TENSOR_V2_INIT)
// =============================================================================

// Helper: create a zero-initialized V2 tensor.
// QNN_TENSOR_V2_INIT is a C-style aggregate initializer that does not compile
// directly as a variable initializer in C++. Use value-init + version override.
static Qnn_Tensor_t MakeV2Tensor() {
  Qnn_Tensor_t t{};
  t.version = QNN_TENSOR_VERSION_2;
  return t;
}

TEST(QnnUnit_DefTest, SetGetQnnTensorType_V2) {
  auto t = MakeV2Tensor();
  qnn::SetQnnTensorType(t, QNN_TENSOR_TYPE_APP_WRITE);
  EXPECT_EQ(qnn::GetQnnTensorType(t), QNN_TENSOR_TYPE_APP_WRITE);
}

TEST(QnnUnit_DefTest, SetGetQnnTensorName_V2) {
  auto t = MakeV2Tensor();
  qnn::SetQnnTensorName(t, "v2_tensor");
  EXPECT_STREQ(qnn::GetQnnTensorName(t), "v2_tensor");
}

TEST(QnnUnit_DefTest, SetGetQnnTensorDataFormat_V2) {
  auto t = MakeV2Tensor();
  qnn::SetQnnTensorDataFormat(t, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
  EXPECT_EQ(qnn::GetQnnTensorDataFormat(t), QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
}

TEST(QnnUnit_DefTest, SetGetQnnTensorDataType_V2) {
  auto t = MakeV2Tensor();
  qnn::SetQnnTensorDataType(t, QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(qnn::GetQnnTensorDataType(t), QNN_DATATYPE_FLOAT_32);
}

TEST(QnnUnit_DefTest, SetQnnTensorDim_V2_SetsRankAndPointer) {
  auto t = MakeV2Tensor();
  std::vector<uint32_t> dims = {2, 4};
  qnn::SetQnnTensorDim(t, dims);
  EXPECT_EQ(qnn::GetQnnTensorRank(t), 2u);
  uint32_t* got = qnn::GetQnnTensorDims(t);
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(got[0], 2u);
  EXPECT_EQ(got[1], 4u);
}

TEST(QnnUnit_DefTest, SetGetQnnTensorMemType_V2) {
  auto t = MakeV2Tensor();
  qnn::SetQnnTensorMemType(t, QNN_TENSORMEMTYPE_MEMHANDLE);
  EXPECT_EQ(qnn::GetQnnTensorMemType(t), QNN_TENSORMEMTYPE_MEMHANDLE);
}

TEST(QnnUnit_DefTest, SetQnnTensorClientBuf_Vector_V2) {
  auto t = MakeV2Tensor();
  std::vector<uint8_t> buf = {1, 2, 3};
  qnn::SetQnnTensorClientBuf(t, buf);
  const auto& got = qnn::GetQnnTensorClientBuf(t);
  EXPECT_EQ(got.data, static_cast<void*>(buf.data()));
  EXPECT_EQ(got.dataSize, 3u);
}

TEST(QnnUnit_DefTest, SetQnnTensorClientBuf_RawPtr_V2) {
  auto t = MakeV2Tensor();
  uint8_t data[2] = {10, 20};
  qnn::SetQnnTensorClientBuf(t, static_cast<void*>(data), 2u);
  const auto& got = qnn::GetQnnTensorClientBuf(t);
  EXPECT_EQ(got.data, static_cast<void*>(data));
  EXPECT_EQ(got.dataSize, 2u);
}

TEST(QnnUnit_DefTest, SetQnnTensorClientBufSize_V2) {
  auto t = MakeV2Tensor();
  qnn::SetQnnTensorClientBufSize(t, 512u);
  EXPECT_EQ(qnn::GetQnnTensorClientBuf(t).dataSize, 512u);
}

TEST(QnnUnit_DefTest, SetQnnTensorClientBufData_V2) {
  auto t = MakeV2Tensor();
  uint8_t sentinel = 7;
  qnn::SetQnnTensorClientBufData(t, &sentinel);
  EXPECT_EQ(qnn::GetQnnTensorClientBuf(t).data, static_cast<void*>(&sentinel));
}

TEST(QnnUnit_DefTest, SetGetQnnTensorMemHandle_V2) {
  auto t = MakeV2Tensor();
  int dummy = 0;
  Qnn_MemHandle_t handle = static_cast<Qnn_MemHandle_t>(&dummy);
  qnn::SetQnnTensorMemHandle(t, handle);
  EXPECT_EQ(qnn::GetQnnTensorMemHandle(t), handle);
}

TEST(QnnUnit_DefTest, SetGetQnnTensorQParams_V2) {
  auto t = MakeV2Tensor();
  Qnn_QuantizeParams_t qp = QNN_QUANTIZE_PARAMS_INIT;
  qp.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp.scaleOffsetEncoding.scale = 2.0f;
  qp.scaleOffsetEncoding.offset = -32;
  qnn::SetQnnTensorQParams(t, qp);
  const auto& got = qnn::GetQnnTensorQParams(t);
  EXPECT_FLOAT_EQ(got.scaleOffsetEncoding.scale, 2.0f);
  EXPECT_EQ(got.scaleOffsetEncoding.offset, -32);
}

TEST(QnnUnit_DefTest, GetQnnTensorID_V2) {
  auto t = MakeV2Tensor();
  t.v2.id = 99u;
  EXPECT_EQ(qnn::GetQnnTensorID(t), 99u);
}

TEST(QnnUnit_DefTest, GetQnnTensorIsDynamicDimensions_V2_ReturnsField) {
  auto t = MakeV2Tensor();
  // isDynamicDimensions is nullptr after zero-init
  EXPECT_EQ(qnn::GetQnnTensorIsDynamicDimensions(t), t.v2.isDynamicDimensions);
}

TEST(QnnUnit_DefTest, CalcQnnTensorNumElems_V2) {
  auto t = MakeV2Tensor();
  std::vector<uint32_t> dims = {3, 5};
  qnn::SetQnnTensorDim(t, dims);
  EXPECT_EQ(qnn::CalcQnnTensorNumElems(t), 15u);
}

// =============================================================================
// CreateTensorInQnnGraph — SDK call paths via mock function pointer
// =============================================================================

namespace {
Qnn_ErrorHandle_t StubTensorCreateSuccess(Qnn_GraphHandle_t, Qnn_Tensor_t*) {
  return QNN_TENSOR_NO_ERROR;
}
Qnn_ErrorHandle_t StubTensorCreateFailure(Qnn_GraphHandle_t, Qnn_Tensor_t*) {
  return QNN_TENSOR_ERROR_INVALID_HANDLE;
}
}  // namespace

TEST(QnnUnit_DefTest, CreateTensorInQnnGraph_SdkCallSucceeds) {
  QNN_INTERFACE_VER_TYPE qnn_iface = QNN_INTERFACE_VER_TYPE_INIT;
  qnn_iface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  qnn::SetQnnTensorDataType(t, QNN_DATATYPE_FLOAT_32);
  std::unordered_map<std::string, uint32_t> table;
  std::string err;

  bool result = qnn::CreateTensorInQnnGraph(qnn_iface, nullptr,
                                            "node0", "my_tensor", t, table, err);
  EXPECT_TRUE(result);
  EXPECT_TRUE(table.count("my_tensor"));
}

TEST(QnnUnit_DefTest, CreateTensorInQnnGraph_SdkCallFails) {
  QNN_INTERFACE_VER_TYPE qnn_iface = QNN_INTERFACE_VER_TYPE_INIT;
  qnn_iface.tensorCreateGraphTensor = StubTensorCreateFailure;
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  qnn::SetQnnTensorDataType(t, QNN_DATATYPE_FLOAT_32);
  std::unordered_map<std::string, uint32_t> table;
  std::string err;

  bool result = qnn::CreateTensorInQnnGraph(qnn_iface, nullptr,
                                            "node0", "my_tensor", t, table, err);
  EXPECT_FALSE(result);
  EXPECT_FALSE(err.empty());
}

// =============================================================================
// QnnParamWrapper::CreateQnnGraphParam — TENSOR path (table pre-populated)
// =============================================================================

TEST(QnnUnit_DefTest, QnnParamWrapper_CreateQnnGraphParam_Tensor_AlreadyExists) {
  std::vector<uint32_t> shape = {1};
  std::vector<uint8_t> data(sizeof(float), 0);
  qnn::QnnParamWrapper param(0, "node0", "w", QNN_DATATYPE_FLOAT_32,
                             std::move(shape), std::move(data));

  // Pre-populate the table with the param's tensor name → early-return, no SDK call.
  std::unordered_map<std::string, uint32_t> table = {{param.GetParamTensorName(), 1u}};
  QNN_INTERFACE_VER_TYPE qnn_iface = QNN_INTERFACE_VER_TYPE_INIT;
  std::string err;

  bool result = param.CreateQnnGraphParam(qnn_iface, nullptr, "node0", table, err);
  EXPECT_TRUE(result);
}

// =============================================================================
// QnnParamWrapper::CreateQnnGraphParam — default case (unknown paramType)
// =============================================================================

TEST(QnnUnit_DefTest, QnnParamWrapper_CreateQnnGraphParam_DefaultCase) {
  Qnn_Scalar_t scalar{};
  scalar.dataType = QNN_DATATYPE_UINT_32;
  scalar.uint32Value = 1u;
  qnn::QnnParamWrapper param(0, "node0", "k", scalar);

  // Force an unknown paramType via the non-const GetQnnParam() accessor.
  param.GetQnnParam().paramType = static_cast<Qnn_ParamType_t>(99);

  QNN_INTERFACE_VER_TYPE qnn_iface = QNN_INTERFACE_VER_TYPE_INIT;
  std::unordered_map<std::string, uint32_t> table;
  std::string err;

  bool result = param.CreateQnnGraphParam(qnn_iface, nullptr, "node0", table, err);
  EXPECT_TRUE(result);
  EXPECT_FALSE(err.empty());
}

}  // namespace test
}  // namespace onnxruntime

// =============================================================================
// QnnBackendTypeToString
// =============================================================================

namespace onnxruntime {
namespace test {

TEST(QnnUnit_DefTest, QnnBackendTypeToString_AllKnownValues) {
  using qnn::QnnBackendType;
  using qnn::QnnBackendTypeToString;

  EXPECT_EQ(QnnBackendTypeToString(QnnBackendType::CPU), "cpu");
  EXPECT_EQ(QnnBackendTypeToString(QnnBackendType::GPU), "gpu");
  EXPECT_EQ(QnnBackendTypeToString(QnnBackendType::DSP), "dsp");
  EXPECT_EQ(QnnBackendTypeToString(QnnBackendType::HTP), "htp");
  EXPECT_EQ(QnnBackendTypeToString(QnnBackendType::HTP_FP16), "htp_fp16");
  EXPECT_EQ(QnnBackendTypeToString(QnnBackendType::SERIALIZER), "ir");
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
