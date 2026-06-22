// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>

namespace onnxruntime {
namespace qnn {

// Resolves the ONNX Shape `start`/`end` attributes against the input rank per the ONNX spec
// (opset >= 15). Mirrors `data.shape[start:end]`: negative values count from the end (add rank),
// then both are clamped to [0, rank]. Returns the resolved (start, end, output_length) where
// output_length = max(0, end - start).
//
// Defined inline in this header so it can be unit-tested directly from the test binary; the
// op-builder also includes this header to ensure a single source of truth.
inline void ResolveShapeBounds(int64_t rank, int64_t start_attr, int64_t end_attr,
                               int64_t& start, int64_t& end, int64_t& output_length) {
  start = start_attr;
  end = end_attr;
  if (start < 0) {
    start += rank;
  }
  if (end < 0) {
    end += rank;
  }
  start = std::min<int64_t>(std::max<int64_t>(start, 0), rank);
  end = std::min<int64_t>(std::max<int64_t>(end, 0), rank);
  output_length = std::max<int64_t>(0, end - start);
}

}  // namespace qnn
}  // namespace onnxruntime
