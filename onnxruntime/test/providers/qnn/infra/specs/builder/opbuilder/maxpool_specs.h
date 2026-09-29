// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared spec literals for MaxPool tier-based tests.

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

enum class MaxPoolDataType {
  Float,
  UInt8,
  UInt16,
};

struct MaxPoolSpec {
  const char* name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  const char* auto_pad;
  int64_t ceil_mode;
  MaxPoolDataType data_type;
  bool use_contrib_qdq;
  float qdq_tolerance;
  bool expect_single_qnn_subgraph;
};

// The first group maps 1:1 to enabled QnnHTPBackendTests MaxPool cases in
// maxpool_test.cc. Shapes, QDQ types, attributes, and the large-input
// tolerance are preserved. CPU MaxPool tests target the retired QnnCpu
// pipeline and are intentionally out of scope. GlobalMaxPool is a separate
// ONNX op and has its own migration inventory.
inline const std::vector<MaxPoolSpec> kMaxPoolSpecs = {
    {"MaxPool_U8_Global", {1, 2, 3, 3}, {3, 3}, {3, 3}, {0, 0, 0, 0}, "NOTSET", 0, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_LargeInput", {1, 125, 8, 56}, {2, 2}, {2, 2}, {0, 0, 0, 0}, "NOTSET", 0, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_F32_Rank3_ReshapeFusion", {1, 3, 3}, {3}, {3}, {0, 0}, "NOTSET", 0, MaxPoolDataType::Float, false, 0.0f, true},
    {"MaxPool_U8_Rank3_Stride1", {1, 3, 3}, {3}, {1}, {1, 1}, "NOTSET", 0, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_Rank3", {1, 3, 3}, {3}, {3}, {0, 0}, "NOTSET", 0, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_Rank3_Ceil", {1, 3, 3}, {3}, {3}, {0, 0}, "NOTSET", 1, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_Rank3_Ceil_Valid", {1, 3, 3}, {3}, {3}, {0, 0}, "VALID", 1, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_Rank3_Ceil_SameUpper", {1, 3, 3}, {3}, {3}, {0, 0}, "SAME_UPPER", 1, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_Rank3_Ceil_SameLower", {1, 3, 3}, {3}, {3}, {0, 0}, "SAME_LOWER", 1, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_Rank4_Ceil", {1, 2, 3, 3}, {3, 3}, {3, 3}, {0, 0, 0, 0}, "NOTSET", 1, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_LargeInput_Ceil", {1, 128, 16, 113}, {2, 2}, {2, 2}, {0, 0, 0, 0}, "NOTSET", 1, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_LargeInput_AutoPadValid", {1, 160, 14, 20}, {2, 2}, {2, 2}, {0, 0, 0, 0}, "VALID", 0, MaxPoolDataType::UInt8, false, 0.0f, false},
    {"MaxPool_U8_LargeInput_OnePads", {1, 64, 384, 576}, {3, 3}, {2, 2}, {1, 1, 1, 1}, "NOTSET", 0, MaxPoolDataType::UInt8, false, 0.00417f, false},
    {"MaxPool_U16_LargeInput_OnePads", {1, 64, 384, 576}, {3, 3}, {2, 2}, {1, 1, 1, 1}, "NOTSET", 0, MaxPoolDataType::UInt16, true, 0.0f, false},
    {"MaxPool_U8_AutoPad_SameLower_LegacyShape", {1, 3, 16, 24}, {2, 2}, {2, 2}, {}, "SAME_LOWER", 0, MaxPoolDataType::UInt8, true, 0.0f, false},
    {"MaxPool_U8_AutoPad_SameUpper_LegacyShape", {1, 3, 16, 24}, {2, 2}, {2, 2}, {}, "SAME_UPPER", 0, MaxPoolDataType::UInt8, true, 0.0f, false},

    // Additional coverage: odd spatial dimensions make SAME_UPPER and
    // SAME_LOWER resolve to distinct leading and trailing pads.
    {"MaxPool_U8_AutoPad_SameUpper_OddSpatial", {1, 3, 15, 23}, {2, 2}, {2, 2}, {}, "SAME_UPPER", 0, MaxPoolDataType::UInt8, true, 0.0f, false},
    {"MaxPool_U8_AutoPad_SameLower_OddSpatial", {1, 3, 15, 23}, {2, 2}, {2, 2}, {}, "SAME_LOWER", 0, MaxPoolDataType::UInt8, true, 0.0f, false},
    {"MaxPool_U16_Rank4_ExplicitPadded", {1, 3, 8, 8}, {3, 3}, {2, 2}, {1, 1, 1, 1}, "NOTSET", 0, MaxPoolDataType::UInt16, true, 0.0f, false},
};

}  // namespace test
}  // namespace onnxruntime
