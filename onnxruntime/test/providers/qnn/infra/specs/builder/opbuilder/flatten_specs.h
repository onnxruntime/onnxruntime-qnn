// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "onnxruntime_c_api.h"

namespace onnxruntime {
namespace test {

enum class FlattenProfile { Float,
                            Int32,
                            QdqU8,
                            QdqU16,
                            QdqU8GraphIO };

// Stable paired identities. Output shapes are ONNX Flatten's two products;
// Accuracy drives real shape inference rather than supplying these shapes.
struct FlattenSpec {
  const char* name;
  FlattenProfile profile;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  std::optional<int64_t> axis;
};

inline const std::vector<FlattenSpec> kFlattenSpecs = {
    {"Rank4_Axis0_Fp32", FlattenProfile::Float, {1, 3, 4, 4}, {1, 48}, 0},
    {"Rank4_AxisNeg1_Fp32", FlattenProfile::Float, {1, 3, 4, 4}, {12, 4}, -1},
    {"Rank5_Axis2_Fp32", FlattenProfile::Float, {1, 2, 3, 4, 4}, {2, 48}, 2},
    {"Rank4_Axis0_QdqU8", FlattenProfile::QdqU8, {1, 3, 4, 4}, {1, 48}, 0},
    {"Rank4_Axis0_QdqU16_Contrib", FlattenProfile::QdqU16, {1, 3, 4, 4}, {1, 48}, 0},
    {"Rank4_AxisNeg1_QdqU8", FlattenProfile::QdqU8, {1, 3, 4, 4}, {12, 4}, -1},
    {"Rank4_AxisNeg1_QdqU16_Contrib", FlattenProfile::QdqU16, {1, 3, 4, 4}, {12, 4}, -1},
    {"Rank5_Axis2_QdqU8_GraphIO", FlattenProfile::QdqU8GraphIO, {1, 2, 3, 4, 5}, {2, 60}, 2},
    {"Rank4_Axis2_Int32", FlattenProfile::Int32, {1, 3, 2, 2}, {3, 4}, 2},
    {"Rank5_Axis2_Int32", FlattenProfile::Int32, {1, 3, 2, 2, 2}, {3, 8}, 2},
    // Enhancements: omitted default axis and the legal inclusive upper bound.
    {"Rank4_DefaultAxis_Fp32", FlattenProfile::Float, {2, 3, 4, 5}, {2, 60}, std::nullopt},
    {"Rank4_AxisRank_Fp32", FlattenProfile::Float, {2, 3, 4, 5}, {120, 1}, 4},
};

}  // namespace test
}  // namespace onnxruntime
