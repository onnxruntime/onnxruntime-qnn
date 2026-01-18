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
// Doesn't work with ExecutionProviders class and KernelRegistryManager
IExecutionProvider* TestCPUExecutionProvider();

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
  Ort::MemoryInfo mem_info_to_use = memory_info ?
    Ort::MemoryInfo(const_cast<OrtMemoryInfo*>(memory_info)) :
    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

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
    memset(dst, 0, tensor_size * sizeof(T));
  }
}

// Specialization declaration for std::vector<bool> which doesn't have .data() method
template<>
void CreateMLValue<bool>(const OrtMemoryInfo* memory_info,
  gsl::span<const int64_t> dims,
  const std::vector<bool>& value,
  Ort::Value& p_mlvalue);

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

}  // namespace test
}  // namespace onnxruntime
