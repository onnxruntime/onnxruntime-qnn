// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <map>
#include <string>

#include "core/framework/execution_provider.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/ort_value.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <gsl/gsl>
#include <gtest/gtest.h>

#ifdef USE_CUDA
#include "core/providers/providers.h"
#endif
#ifdef USE_NNAPI
#include "core/providers/nnapi/nnapi_builtin/nnapi_execution_provider.h"
#endif
#ifdef USE_RKNPU
#include "core/providers/rknpu/rknpu_execution_provider.h"
#endif
#ifdef USE_COREML
#include "core/providers/coreml/coreml_execution_provider.h"
#endif

namespace onnxruntime {
class Graph;

namespace test {
template <typename T>
inline void CopyVectorToTensor(gsl::span<const T> value, Tensor& tensor) {
  gsl::copy(value, tensor.MutableDataAsSpan<T>());
}

template <class T>
inline void CopyVectorToTensor(const std::vector<T>& value, Tensor& tensor) {
  gsl::copy(AsSpan(value), tensor.MutableDataAsSpan<T>());
}

// vector<bool> is specialized so we need to handle it separately
template <>
inline void CopyVectorToTensor<bool>(const std::vector<bool>& value, Tensor& tensor) {
  auto output_span = tensor.MutableDataAsSpan<bool>();
  for (size_t i = 0, end = value.size(); i < end; ++i) {
    output_span[i] = value[i];
  }
}

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

template <typename T>
void AllocateMLValue(AllocatorPtr alloc, gsl::span<const int64_t> dims, OrtValue* p_mlvalue) {
  TensorShape shape(dims);
  auto element_type = DataTypeImpl::GetType<T>();
  Tensor::InitOrtValue(element_type, shape, std::move(alloc), *p_mlvalue);
}

using OpCountMap = std::map<std::string, int>;

// Returns a map with the number of occurrences of each operator in the graph.
// Helper function to check that the graph transformations have been successfully applied.
OpCountMap CountOpsInGraph(const Graph& graph, bool recurse_into_subgraphs = true);

// Gets the op count from the OpCountMap.
// Can be called with a const OpCountMap, unlike OpCountMap::operator[].
inline int OpCount(const OpCountMap& op_count_map, const std::string& op_type) {
  if (auto it = op_count_map.find(op_type); it != op_count_map.end()) {
    return it->second;
  }
  return 0;
}

#if !defined(DISABLE_SPARSE_TENSORS)
void SparseIndicesChecker(const ONNX_NAMESPACE::TensorProto& indices_proto, gsl::span<const int64_t> expected_indicies);
#endif  // DISABLE_SPARSE_TENSORS

template <typename T = float>
void VerifyOutputs(const Tensor& tensor, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, tensor.Shape());
  const std::vector<T> found(tensor.Data<T>(),
                             tensor.Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

inline void VerifySingleOutput(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                               const std::vector<float>& expected_values) {
  ASSERT_EQ(1u, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  VerifyOutputs(rtensor, expected_dims, expected_values);
}

}  // namespace test
}  // namespace onnxruntime
