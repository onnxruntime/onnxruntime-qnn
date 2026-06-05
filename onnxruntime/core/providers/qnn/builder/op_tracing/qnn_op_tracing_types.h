// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Data types and free-function declarations for the framework op tracing feature.
//
// This header is intentionally test-safe: it does not include qnn_def.h,
// ort_api.h, or any other EP-internal header, so it can be included from test
// translation units without pulling in large template libraries or causing
// conflicts with ORT core public headers. nlohmann/json_fwd.hpp is included
// only for the SerializeFrameworkOpTrace return-type declaration; the full
// nlohmann/json.hpp is needed at call sites that actually use the return value.
//
// Implementations of ComputeTraceSummary and SerializeFrameworkOpTrace live
// in qnn_op_tracing_serialization.cc.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "nlohmann/json_fwd.hpp"

namespace onnxruntime {
namespace qnn {

// Source target type: matches the integer encoding used by the QNN SDK
// (TENSOR = 0, OP = 1, defined in QnnInterface.h).
enum class TraceTargetType : uint8_t {
  kTensor = 0,
  kOp = 1,
};

struct TraceSourcePair {
  std::string name;
  TraceTargetType type;
};

struct TraceMapping {
  std::string dst_name;         // QNN op or tensor name
  std::string dst_qnn_op_type;  // QNN op type string, empty for tensor mappings
  std::vector<TraceSourcePair> sources;
  // Populated from IQnnNodeGroup::Type(). These strings are part of the JSON
  // schema and must remain stable across SDK versions; changing them is a
  // breaking change for any consumer that keys on fusion_count entries.
  std::string node_group_type;  // e.g., "DQQFusion", "OrtNodeUnit"
};

struct OpTraceInfo {
  std::string graph_name;
  std::vector<TraceMapping> op_mappings;
  std::vector<TraceMapping> tensor_mappings;
};

struct UnsupportedNodeInfo {
  std::string node_name;
  std::string op_type;
  size_t node_index;
  std::string reason;
};

struct TraceSummary {
  size_t total_onnx_nodes = 0;
  // Count of unique ONNX source nodes that appear as OP-typed sources across
  // all op_mappings. Deduplicated by source name so an N:M fusion (which
  // emits the same ONNX sources from multiple QNN op entries) is counted
  // once per ONNX node rather than once per QNN op entry.
  size_t supported_nodes = 0;
  size_t unsupported_nodes = 0;
  size_t qnn_subgraphs = 0;
  size_t total_qnn_ops = 0;
  // std::map keeps keys in sorted order, which produces deterministic JSON
  // output and makes trace files diff-friendly across runs.
  std::map<std::string, size_t> fusion_count;
};

struct CompilationTarget {
  std::string htp_arch;
  uint32_t soc_model = 0;
  uint32_t device_id = 0;
};

struct FrameworkOpTrace {
  std::string model_name;
  std::string backend_type;
  CompilationTarget compilation_target;
  std::vector<OpTraceInfo> subgraph_traces;
  std::vector<UnsupportedNodeInfo> unsupported_nodes;
  TraceSummary summary;
};

// Compute summary statistics for a FrameworkOpTrace.
// Defined in qnn_op_tracing_serialization.cc.
void ComputeTraceSummary(FrameworkOpTrace& trace);

// Serialize a FrameworkOpTrace to JSON.
// Defined in qnn_op_tracing_serialization.cc.  Call sites that use the return value must
// include nlohmann/json.hpp separately.
nlohmann::json SerializeFrameworkOpTrace(const FrameworkOpTrace& trace);

}  // namespace qnn
}  // namespace onnxruntime
