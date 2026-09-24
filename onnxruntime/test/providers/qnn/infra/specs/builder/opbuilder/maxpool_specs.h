// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared spec literals for MaxPool tier-based tests.

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

enum class MaxPoolQuantType {
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
  MaxPoolQuantType quant_type;
  bool use_contrib_qdq;
};

// These cases map to QnnHTPBackendTests MaxPool cases. They deliberately use
// small tensors except where input size is itself the regression under test.
inline const std::vector<MaxPoolSpec> kMaxPoolSpecs = {
    {"MaxPool_U8_Rank4_Explicit", {1, 2, 3, 3}, {3, 3}, {3, 3}, {0, 0, 0, 0},
     "NOTSET", 0, MaxPoolQuantType::UInt8, false},
    {"MaxPool_U8_Rank3_Padded", {1, 3, 3}, {3}, {1}, {1, 1},
     "NOTSET", 0, MaxPoolQuantType::UInt8, false},
    {"MaxPool_U8_Rank3_Ceil_SameUpper", {1, 3, 3}, {3}, {3}, {0, 0},
     "SAME_UPPER", 1, MaxPoolQuantType::UInt8, false},
    {"MaxPool_U8_Rank3_Ceil_SameLower", {1, 3, 3}, {3}, {3}, {0, 0},
     "SAME_LOWER", 1, MaxPoolQuantType::UInt8, false},
    // Odd spatial dimensions make SAME_UPPER and SAME_LOWER produce different
    // leading/trailing pad amounts. The legacy cases use 16x24, where both
    // modes resolve to zero padding and cannot distinguish the branches.
    {"MaxPool_U8_Rank4_SameUpper", {1, 3, 15, 23}, {2, 2}, {2, 2}, {},
     "SAME_UPPER", 0, MaxPoolQuantType::UInt8, true},
    {"MaxPool_U8_Rank4_SameLower", {1, 3, 15, 23}, {2, 2}, {2, 2}, {},
     "SAME_LOWER", 0, MaxPoolQuantType::UInt8, true},
    {"MaxPool_U16_Rank4_ExplicitPadded", {1, 3, 8, 8}, {3, 3}, {2, 2}, {1, 1, 1, 1},
     "NOTSET", 0, MaxPoolQuantType::UInt16, true},
};

}  // namespace test
}  // namespace onnxruntime
