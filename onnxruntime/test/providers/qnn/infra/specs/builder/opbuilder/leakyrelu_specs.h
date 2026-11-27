// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include "onnxruntime_c_api.h"

namespace onnxruntime {
namespace test {

// Shared identities and setup for paired graph and numerical scenarios.
struct LeakyReluSpec {
  const char* name;
  int opset;
  ONNXTensorElementDataType dtype;
  std::vector<int64_t> shape;
  std::vector<float> input_data;
  float alpha;
};

inline const std::vector<LeakyReluSpec> kLeakyReluSpecs = {
    {"U8_OpSet15", 15, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 2, 3}, {-40.0f, -20.0f, 0.0f, 10.0f, 30.0f, 40.0f}, 0.2f},
    {"U8_OpSet16", 16, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 2, 3}, {-40.0f, -20.0f, 0.0f, 10.0f, 30.0f, 40.0f}, 0.2f},
    {"FP16_OpSet16", 16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {1, 2, 3}, {-40.0f, -20.0f, 1.0f, 10.0f, 30.0f, 40.0f}, 0.2f},
};

// Effective U8 boundary params from the legacy helpers' range derivation.
// Output quantization uses the LeakyRelu output range, independently of input.
inline constexpr float kLeakyReluInputScale = 0.3137255012989044f;
inline constexpr uint8_t kLeakyReluInputZeroPoint = 127;
inline constexpr float kLeakyReluOutputScale = 0.1882352977991104f;
inline constexpr uint8_t kLeakyReluOutputZeroPoint = 42;

}  // namespace test
}  // namespace onnxruntime
