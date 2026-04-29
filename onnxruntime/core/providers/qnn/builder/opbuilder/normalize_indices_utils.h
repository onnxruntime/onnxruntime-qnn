// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Adapts ONNX Gather/Scatter indices to QNN's accepted form. ONNX allows
// negative (count-from-end) INT_64 indices; QNN rejects both -- a single
// negative static index otherwise silently drops the node to CPU.
// Dynamic int64 indices get a runtime Cast(INT_32), but negative values at
// runtime pass through uncorrected.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gsl/gsl>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

namespace utils {

// Rewrites raw index bytes as INT_32 so every value lands in [0, axis_dim).
// Returns false if any index is out of range, or if `onnx_bytes.size()` is not
// a multiple of `sizeof(SrcType)`. `axis_dim_for_element(i)` supplies the open
// upper bound for element i so callers can encode per-op layout. `qnn_bytes`
// is resized to hold the rewritten INT_32 output.
template <typename SrcType>
bool NormalizeIndicesBytes(gsl::span<const uint8_t> onnx_bytes,
                           const std::function<int64_t(size_t)>& axis_dim_for_element,
                           std::vector<uint8_t>& qnn_bytes,
                           bool& has_negative_indices);

// ScatterND-style: indices' last dim `k` is tuple depth; column c bounds
// against `data_shape[c]`.
Ort::Status NormalizeIndicesForScatterND(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnitIODef& indices_input,
    const std::vector<uint32_t>& data_shape,
    const Ort::Logger& logger,
    std::vector<std::string>& input_names,
    bool do_op_validation);

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
