// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {

// Parse positive dimensions and restore Genie's omitted singleton sequence dimension.
// On failure, clear output_shape without allowing exceptions to leave the callback.
bool ParseGenieOutputShape(const char* output_config,
                           std::vector<int64_t>& output_shape) noexcept;

}  // namespace onnxruntime
