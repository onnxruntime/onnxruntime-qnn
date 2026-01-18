// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "test/unittest_util/framework_test_utils.h"
#include "core/graph/graph.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <memory>

namespace onnxruntime {
namespace test {
IExecutionProvider* TestCPUExecutionProvider() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return &cpu_provider;
}

static void CountOpsInGraphImpl(const Graph& graph, bool recurse_into_subgraphs, OpCountMap& ops) {
  for (auto& node : graph.Nodes()) {
    std::string key = node.Domain() + (node.Domain().empty() ? "" : ".") + node.OpType();

    ++ops[key];

    if (recurse_into_subgraphs && node.ContainsSubgraph()) {
      for (auto& subgraph : node.GetSubgraphs()) {
        CountOpsInGraphImpl(*subgraph, recurse_into_subgraphs, ops);
      }
    }
  }
}

// Returns a map with the number of occurrences of each operator in the graph.
// Helper function to check that the graph transformations have been successfully applied.
OpCountMap CountOpsInGraph(const Graph& graph, bool recurse_into_subgraphs) {
  OpCountMap ops;
  CountOpsInGraphImpl(graph, recurse_into_subgraphs, ops);

  return ops;
}

// Specialization for std::vector<bool> which doesn't have .data() method
template<>
void CreateMLValue<bool>(const OrtMemoryInfo* memory_info,
  gsl::span<const int64_t> dims,
  const std::vector<bool>& value,
  Ort::Value& p_mlvalue) {
  // Create memory info if not provided
  Ort::MemoryInfo mem_info_to_use = memory_info ?
    Ort::MemoryInfo(const_cast<OrtMemoryInfo*>(memory_info)) :
    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

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
