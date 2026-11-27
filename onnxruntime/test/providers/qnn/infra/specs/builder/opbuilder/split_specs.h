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
  int input_start;
};
inline const std::vector<SplitSpec> kSplitSpecs = {
    {"Int32_Opset13", {4, 2}, {2, 2}, 0, -1, 13, SplitQuantType::None, false, 1},
    {"Int32_Opset13_Noop", {1, 2}, {1}, 0, -1, 13, SplitQuantType::None, false, 1},
    {"U8_Opset18_Initializer", {6, 2}, {2, 2, 2}, 0, -1, 18, SplitQuantType::UInt8, false, 0},
    {"U8_Opset18_NumOutputsEqual", {6, 2}, {}, 0, 3, 18, SplitQuantType::UInt8, false, 0},
    {"U8_Opset18_InitializerUnequal", {7, 2}, {3, 3, 1}, 0, -1, 18, SplitQuantType::UInt8, false, 0},
    {"U8_Opset18_NumOutputsUnequal", {7, 2}, {}, 0, 3, 18, SplitQuantType::UInt8, false, 0},
    {"U16_Opset18", {4, 2}, {2, 2}, 0, -1, 18, SplitQuantType::UInt16, true, 1},
    {"U16_Opset18_NumOutputs", {4, 2}, {}, 0, 2, 18, SplitQuantType::UInt16, true, 1},
    {"U8_Opset13_Equal", {4, 2}, {2, 2}, 0, -1, 13, SplitQuantType::UInt8, false, 1},
    {"U8_Opset11_Equal", {4, 2}, {2, 2}, 0, -1, 11, SplitQuantType::UInt8, false, 1},
    {"U8_Opset13_Axis1Unequal", {2, 4}, {1, 3}, 1, -1, 13, SplitQuantType::UInt8, false, 1},
    {"U8_Opset11_Axis1Unequal", {2, 4}, {1, 3}, 1, -1, 11, SplitQuantType::UInt8, false, 1},
    {"U8_Opset13_Noop", {1, 2}, {1}, 0, -1, 13, SplitQuantType::UInt8, false, 1},
};

// Preserve the literal legacy input sequence in both paired tiers.
template <typename T>
std::vector<T> SplitInputValues(const SplitSpec& spec) {
  size_t count = 1;
  for (int64_t dim : spec.shape) count *= static_cast<size_t>(dim);
  std::vector<T> values(count);
  for (size_t i = 0; i < count; ++i) values[i] = static_cast<T>(i + spec.input_start);
  return values;
}
}  // namespace onnxruntime::test
