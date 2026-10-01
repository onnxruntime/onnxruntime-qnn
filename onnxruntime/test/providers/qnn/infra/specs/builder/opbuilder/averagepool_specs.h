// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared spec literals for AveragePool-family tier-based tests.

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

enum class AveragePoolDataType {
  Float,
  UInt8,
};

struct AveragePoolSpec {
  const char* name;
  const char* op_type;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  const char* auto_pad;
  int64_t count_include_pad;
  AveragePoolDataType data_type;
  std::vector<float> input_data;
  float input_min;
  float input_max;
  bool expect_single_qnn_subgraph;
};

// Maps 1:1 to the enabled AveragePool and GlobalAveragePool HTP cases in
// averagepool_test.cc. QnnCpu and QNN GPU coverage remain legacy-only.
inline const std::vector<AveragePoolSpec> kAveragePoolSpecs = {
    {"AveragePool_U8_AsGlobal_Rank4", "AveragePool", {1, 2, 3, 3}, {3, 3}, {3, 3}, {}, "NOTSET", 0, AveragePoolDataType::UInt8,
     {32.1289f, -59.981f, -17.2799f, 62.7263f, 33.6205f, -19.3515f, -54.0113f, 37.5648f, 61.5357f, -52.5769f, 27.3637f, -9.01382f, -65.5612f, 19.9497f, -47.9228f, 26.9813f, 83.064f, 0.362503f}, -10.0f, 10.0f, false},
    {"AveragePool_U8_CountIncludePad_Rank4", "AveragePool", {1, 2, 3, 3}, {1, 1}, {}, {}, "NOTSET", 1, AveragePoolDataType::UInt8,
     {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, -10.0f, 10.0f, false},
    {"AveragePool_U8_AutoPadSameUpper_Rank4", "AveragePool", {1, 2, 3, 3}, {1, 1}, {}, {}, "SAME_UPPER", 0, AveragePoolDataType::UInt8,
     {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, -10.0f, 10.0f, false},
    {"AveragePool_U8_AutoPadSameLower_Rank4", "AveragePool", {1, 2, 3, 3}, {1, 1}, {}, {}, "SAME_LOWER", 0, AveragePoolDataType::UInt8,
     {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, -10.0f, 10.0f, false},
    {"AveragePool_U8_Rank5", "AveragePool", {1, 2, 8, 8, 8}, {3, 3, 3}, {2, 2, 2}, {}, "NOTSET", 0, AveragePoolDataType::UInt8, {}, -10.0f, 10.0f, false},
    {"AveragePool_U8_Rank5_AutoPadSameUpper", "AveragePool", {1, 2, 8, 8, 8}, {2, 2, 2}, {}, {}, "SAME_UPPER", 0, AveragePoolDataType::UInt8, {}, -10.0f, 10.0f, false},
    {"AveragePool_U8_Rank5_AutoPadSameLower", "AveragePool", {1, 2, 8, 8, 8}, {2, 2, 2}, {}, {}, "SAME_LOWER", 0, AveragePoolDataType::UInt8, {}, -10.0f, 10.0f, false},
    {"AveragePool_F32_Rank3_ReshapeFusion", "AveragePool", {1, 3, 3}, {3}, {3}, {0, 0}, "NOTSET", 0, AveragePoolDataType::Float, {}, -10.0f, 10.0f, true},
    {"AveragePool_U8_Rank3", "AveragePool", {1, 3, 5}, {3}, {1}, {1, 1}, "NOTSET", 0, AveragePoolDataType::UInt8, {}, -10.0f, 10.0f, false},
    {"AveragePool_U8_Rank3_CountIncludePad", "AveragePool", {1, 3, 5}, {3}, {1}, {1, 1}, "NOTSET", 1, AveragePoolDataType::UInt8, {}, -10.0f, 10.0f, false},
    {"AveragePool_U8_Rank3_AutoPadSameUpper", "AveragePool", {1, 3, 4}, {3}, {2}, {}, "SAME_UPPER", 0, AveragePoolDataType::UInt8, {}, -10.0f, 10.0f, false},
    {"AveragePool_U8_Rank3_AutoPadSameLower", "AveragePool", {1, 3, 4}, {3}, {2}, {}, "SAME_LOWER", 0, AveragePoolDataType::UInt8, {}, -10.0f, 10.0f, false},
    {"GlobalAveragePool_U8_Rank4", "GlobalAveragePool", {1, 2, 3, 3}, {}, {}, {}, "NOTSET", 0, AveragePoolDataType::UInt8, {}, -32.0f, 32.0f, false},
    {"GlobalAveragePool_U8_Rank3", "GlobalAveragePool", {1, 8, 5}, {}, {}, {}, "NOTSET", 0, AveragePoolDataType::UInt8, {}, -10.0f, 10.0f, false},
};

}  // namespace test
}  // namespace onnxruntime
