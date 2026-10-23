// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <gsl/span>

namespace onnxruntime {
namespace qnn {

inline constexpr size_t kReshapeEinsumReshapeInputRank = 6;

struct ReshapeEinsumReshapeShapes {
  std::array<uint32_t, 2> block_size;
  std::array<uint32_t, 4> reshape_input;
  std::array<uint32_t, 4> depth_to_space_output;
  std::array<uint32_t, 4> post_reshape_output;
};

inline std::optional<ReshapeEinsumReshapeShapes> DeriveReshapeEinsumReshapeShapes(
    gsl::span<const uint32_t> input_shape) {
  if (input_shape.size() != kReshapeEinsumReshapeInputRank) {
    return std::nullopt;
  }

  return ReshapeEinsumReshapeShapes{
      {input_shape[3], input_shape[4]},
      {input_shape[0], input_shape[1], input_shape[2], input_shape[3] * input_shape[4] * input_shape[5]},
      {input_shape[0], input_shape[1] * input_shape[3], input_shape[2] * input_shape[4], input_shape[5]},
      {input_shape[0], input_shape[5], input_shape[1] * input_shape[3], input_shape[2] * input_shape[4]}};
}

}  // namespace qnn
}  // namespace onnxruntime
