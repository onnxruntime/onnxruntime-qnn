// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <vector>
namespace onnxruntime::test {
enum class SplitQuantType { None,
                            UInt8,
                            UInt16 };
struct SplitSpec {
  const char* name;
  std::vector<int64_t> shape;
  std::vector<int64_t> split;
  int64_t axis;
  int64_t num_outputs;
  int opset;
  SplitQuantType quant_type;
  bool contrib_qdq;
};
inline const std::vector<SplitSpec> kSplitSpecs = {
    {"Int32_Opset13", {4, 2}, {2, 2}, 0, -1, 13, SplitQuantType::None, false},
    {"U8_Opset18_Initializer", {6, 2}, {2, 2, 2}, 0, -1, 18, SplitQuantType::UInt8, false},
    {"U8_Opset18_NumOutputsUnequal", {7, 2}, {}, 0, 3, 18, SplitQuantType::UInt8, false},
    {"U16_Opset18", {4, 2}, {2, 2}, 0, -1, 18, SplitQuantType::UInt16, true},
    {"U8_Opset11_Axis1Unequal", {2, 4}, {1, 3}, 1, -1, 11, SplitQuantType::UInt8, false},
    {"U8_Opset13_Noop", {1, 2}, {1}, 0, -1, 13, SplitQuantType::UInt8, false},
};
}  // namespace onnxruntime::test
