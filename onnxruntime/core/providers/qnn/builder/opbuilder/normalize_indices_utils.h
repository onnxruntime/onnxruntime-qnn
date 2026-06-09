// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// ONNX Gather/Scatter allow negative and INT_64 indices; QNN rejects both.
// A single negative static index otherwise silently drops the node to CPU.
// Dynamic INT_64 indices get a runtime Cast; negative runtime values are
// NOT corrected.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gsl/gsl>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace utils {

// Returns false on out-of-range index. `axis_dim_for_element(i)` is the
// per-element open upper bound -- lets callers encode op-specific layout
// (e.g. ScatterND's per-column bound).
template <typename SrcType>
bool NormalizeIndicesBytes(gsl::span<const uint8_t> onnx_bytes,
                           const std::function<int64_t(size_t)>& axis_dim_for_element,
                           std::vector<uint8_t>& qnn_bytes,
                           bool& has_negative_indices);

// Registers the indices tensor; for dynamic INT_64 inputs, inserts a
// runtime Cast(INT_32) so downstream QNN ops see INT_32.
Ort::Status AddNormalizedIndicesTensor(QnnModelWrapper& qnn_model_wrapper,
                                       TensorInfo indices_info,
                                       const std::string& indices_tensor_name,
                                       std::vector<uint8_t>&& qnn_indices_bytes,
                                       const Ort::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation);

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
