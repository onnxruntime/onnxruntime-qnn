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

// Makes negative indices positive and converts indices to another integer type
// (typically int32 or uint32) over a single axis dimension. Returns false on
// out-of-range index. Inputs and outputs are byte arrays.
template <typename SrcType, typename DstType>
bool MakeStaticIndicesPositiveAndValidate(const std::vector<uint8_t>& onnx_bytes,
                                          int64_t input0_axis_dim,
                                          /*out*/ std::vector<uint8_t>& qnn_bytes,
                                          /*out*/ bool* has_negative_indices) {
  const size_t num_elems = onnx_bytes.size() / sizeof(SrcType);
  gsl::span<const SrcType> onnx_indices{reinterpret_cast<const SrcType*>(onnx_bytes.data()), num_elems};

  qnn_bytes.resize(num_elems * sizeof(DstType));
  gsl::span<DstType> qnn_indices{reinterpret_cast<DstType*>(qnn_bytes.data()), num_elems};

  for (size_t i = 0; i < num_elems; i++) {
    SrcType onnx_index = onnx_indices[i];

    if (onnx_index < 0) {
      if (has_negative_indices != nullptr) {
        *has_negative_indices = true;
      }
      onnx_index += static_cast<SrcType>(input0_axis_dim);
    }

    if (onnx_index < 0 || static_cast<int64_t>(onnx_index) >= input0_axis_dim) {
      return false;  // QNN does not support out-of-bounds indices.
    }

    qnn_indices[i] = static_cast<DstType>(onnx_index);
  }

  return true;
}

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
