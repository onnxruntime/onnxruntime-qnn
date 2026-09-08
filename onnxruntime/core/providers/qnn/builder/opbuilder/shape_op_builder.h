// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

namespace onnxruntime {
namespace qnn {

// Resolves the ONNX Shape `start`/`end` attributes against the input rank per the ONNX spec
// (opset >= 15). Mirrors `data.shape[start:end]`: negative values count from the end (add rank),
// then both are clamped to [0, rank]. Returns the resolved (start, end) pair.
inline std::pair<int64_t, int64_t> ResolveShapeBounds(int64_t rank, int64_t start_attr, int64_t end_attr) {
  int64_t start = start_attr;
  int64_t end = end_attr;
  if (start < 0) {
    start += rank;
  }
  if (end < 0) {
    end += rank;
  }
  start = std::min<int64_t>(std::max<int64_t>(start, 0), rank);
  end = std::min<int64_t>(std::max<int64_t>(end, 0), rank);
  return {start, end};
}

}  // namespace qnn
}  // namespace onnxruntime
