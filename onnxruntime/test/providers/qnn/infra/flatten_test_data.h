// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

// Plain data across the full-ORT/model-builder and fake-ORT/snapshot boundary.
// Implementation uses the same generator and quantization helpers as legacy.
struct FlattenQparams {
  float scale;
  uint32_t zero_point;
};

struct FlattenQdqData {
  std::vector<float> input_values;
  FlattenQparams input;
  FlattenQparams output;
};

FlattenQdqData GetFlattenQdqData(const std::vector<int64_t>& shape, bool use_uint16);

}  // namespace test
}  // namespace onnxruntime
