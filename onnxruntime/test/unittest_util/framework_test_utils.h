// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <map>
#include <string>

#include "onnxruntime_cxx_api.h"

#include <gsl/gsl>
#include <gtest/gtest.h>

namespace onnxruntime {
namespace test {

template <class T>
void CreateMLValue(const OrtMemoryInfo* memory_info,
                   gsl::span<const int64_t> dims,
                   const std::vector<T>& value,
                   Ort::Value& p_mlvalue) {
  // Allocate CPU tensor memory owned by Ort::Value.
  Ort::MemoryInfo mem_info_to_use = memory_info ? Ort::MemoryInfo(const_cast<OrtMemoryInfo*>(memory_info)) : Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // Allocate tensor with ORT-owned buffer (Arena allocator).
  Ort::AllocatorWithDefaultOptions allocator;
  p_mlvalue = Ort::Value::CreateTensor<T>(
      allocator,
      dims.data(),
      dims.size());

  // Copy data (or zero-fill if empty vector provided).
  T* dst = p_mlvalue.GetTensorMutableData<T>();
  if (!value.empty()) {
    memcpy(dst, value.data(), value.size() * sizeof(T));
  } else {
    // total element count from dims
    size_t tensor_size = 1;
    for (auto dim : dims) {
      tensor_size *= static_cast<size_t>(dim);
    }
    // Use loop for proper value initialization instead of memset
    // This works correctly for both trivial and non-trivial types
    for (size_t i = 0; i < tensor_size; ++i) {
      dst[i] = T{};
    }
  }
}

// Specialization declaration for std::vector<bool> which doesn't have .data() method
template <>
inline void CreateMLValue<bool>(const OrtMemoryInfo* memory_info,
                                gsl::span<const int64_t> dims,
                                const std::vector<bool>& value,
                                Ort::Value& p_mlvalue) {
  // Create memory info if not provided
  Ort::MemoryInfo mem_info_to_use = memory_info ? Ort::MemoryInfo(const_cast<OrtMemoryInfo*>(memory_info)) : Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // Allocate tensor with ORT-owned buffer (Arena allocator).
  Ort::AllocatorWithDefaultOptions allocator;
  p_mlvalue = Ort::Value::CreateTensor<bool>(
      allocator,
      dims.data(),
      dims.size());

  bool* dst = p_mlvalue.GetTensorMutableData<bool>();

  size_t tensor_size = 1;
  for (auto dim : dims) {
    tensor_size *= static_cast<size_t>(dim);
  }

  if (!value.empty()) {
    const size_t n = std::min(value.size(), tensor_size);
    for (size_t i = 0; i < n; ++i) {
      dst[i] = value[i];
    }
    for (size_t i = n; i < tensor_size; ++i) {
      dst[i] = false;
    }
  } else {
    for (size_t i = 0; i < tensor_size; ++i) {
      dst[i] = false;
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
