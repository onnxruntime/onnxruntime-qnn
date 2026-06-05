// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Implementations of ComputeTraceSummary and SerializeFrameworkOpTrace.
//
// This translation unit (.cc file) is compiled directly into the test binary
// (via cmake) so that unit tests can call these functions without linking
// against the EP library. It intentionally does NOT include ort_api.h (or any
// header that pulls it in) to avoid the ORT_API_MANUAL_INIT mismatch linker
// error on Windows.

#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing_types.h"

#include <unordered_set>

#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace qnn {

void ComputeTraceSummary(FrameworkOpTrace& trace) {
  TraceSummary& summary = trace.summary;
  summary.unsupported_nodes = trace.unsupported_nodes.size();
  summary.qnn_subgraphs = trace.subgraph_traces.size();
  summary.total_qnn_ops = 0;
  summary.fusion_count.clear();

  // An ONNX node is "supported" if it appears as an OP-typed source on any QNN
  // op_mapping. N:M fusions emit the same ONNX source set from multiple QNN op
  // entries (e.g. ReshapeEinsumReshape: 3 ONNX ops referenced by 3 QNN op
  // entries = 9 OP source occurrences but only 3 unique ONNX nodes), so we
  // dedup by source name before counting.
  std::unordered_set<std::string> supported_onnx_node_names;
  for (const auto& subgraph : trace.subgraph_traces) {
    summary.total_qnn_ops += subgraph.op_mappings.size();
    for (const auto& mapping : subgraph.op_mappings) {
      summary.fusion_count[mapping.node_group_type]++;
      for (const auto& src : mapping.sources) {
        if (src.type == TraceTargetType::kOp) {
          supported_onnx_node_names.insert(src.name);
        }
      }
    }
  }

  summary.supported_nodes = supported_onnx_node_names.size();
  summary.total_onnx_nodes = summary.supported_nodes + summary.unsupported_nodes;
}

namespace detail {

static nlohmann::json SerializeTraceSourcePair(const TraceSourcePair& pair) {
  return {
      {"name", pair.name},
      {"type", pair.type == TraceTargetType::kOp ? "OP" : "TENSOR"},
  };
}

static nlohmann::json SerializeTraceMapping(const TraceMapping& mapping) {
  nlohmann::json j;
  j["dst_name"] = mapping.dst_name;

  if (!mapping.dst_qnn_op_type.empty()) {
    j["dst_qnn_op_type"] = mapping.dst_qnn_op_type;
  }

  nlohmann::json sources = nlohmann::json::array();
  for (const auto& src : mapping.sources) {
    sources.push_back(SerializeTraceSourcePair(src));
  }
  j["sources"] = std::move(sources);

  if (!mapping.node_group_type.empty()) {
    j["node_group_type"] = mapping.node_group_type;
  }

  return j;
}

}  // namespace detail

nlohmann::json SerializeFrameworkOpTrace(const FrameworkOpTrace& trace) {
  nlohmann::json j;
  j["model_name"] = trace.model_name;
  j["backend_type"] = trace.backend_type;

  nlohmann::json ct;
  ct["htp_arch"] = trace.compilation_target.htp_arch;
  ct["soc_model"] = trace.compilation_target.soc_model;
  ct["device_id"] = trace.compilation_target.device_id;
  j["compilation_target"] = std::move(ct);

  nlohmann::json subgraphs = nlohmann::json::array();
  for (const auto& sg : trace.subgraph_traces) {
    nlohmann::json sg_json;
    sg_json["graph_name"] = sg.graph_name;

    nlohmann::json op_mappings = nlohmann::json::array();
    for (const auto& m : sg.op_mappings) {
      op_mappings.push_back(detail::SerializeTraceMapping(m));
    }
    sg_json["op_mappings"] = std::move(op_mappings);

    nlohmann::json tensor_mappings = nlohmann::json::array();
    for (const auto& m : sg.tensor_mappings) {
      tensor_mappings.push_back(detail::SerializeTraceMapping(m));
    }
    sg_json["tensor_mappings"] = std::move(tensor_mappings);

    subgraphs.push_back(std::move(sg_json));
  }
  j["subgraph_traces"] = std::move(subgraphs);

  nlohmann::json unsupported = nlohmann::json::array();
  for (const auto& un : trace.unsupported_nodes) {
    unsupported.push_back({
        {"node_name", un.node_name},
        {"op_type", un.op_type},
        {"node_index", un.node_index},
        {"reason", un.reason},
    });
  }
  j["unsupported_nodes"] = std::move(unsupported);

  const auto& s = trace.summary;
  nlohmann::json summary;
  summary["total_onnx_nodes"] = s.total_onnx_nodes;
  summary["supported_nodes"] = s.supported_nodes;
  summary["unsupported_nodes"] = s.unsupported_nodes;
  summary["qnn_subgraphs"] = s.qnn_subgraphs;
  summary["total_qnn_ops"] = s.total_qnn_ops;

  nlohmann::json fc;
  for (const auto& [type, count] : s.fusion_count) {
    fc[type] = count;
  }
  summary["fusion_count"] = std::move(fc);
  j["summary"] = std::move(summary);

  return j;
}

}  // namespace qnn
}  // namespace onnxruntime
