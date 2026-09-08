// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Implementations of ComputeTraceSummary and SerializeFrameworkOpTrace,
// and DeriveTracePathFromContextModel.
//
// This translation unit (.cc file) is compiled directly into the test binary
// (via cmake) so that unit tests can call these functions without linking
// against the EP library. It intentionally does NOT include ort_api.h (or any
// header that pulls it in) to avoid the ORT_API_MANUAL_INIT mismatch linker
// error on Windows.

#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing_types.h"

#include <algorithm>
#include <fstream>
#include <unordered_set>

#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace qnn {

void ComputeTraceSummary(FrameworkOpTrace& trace) {
  TraceSummary& summary = trace.summary;
  summary.unsupported_nodes = trace.unsupported_nodes.size();
  summary.qnn_subgraphs = trace.subgraph_traces.size();
  summary.total_qnn_ops = 0;
  summary.total_socs = static_cast<uint32_t>(trace.soc_traces.size());
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

std::string EncodeHtpArch(uint32_t htp_arch) {
  if (htp_arch == kSocUnknown) {
    return {};
  }
  return "V" + std::to_string(htp_arch);
}

std::string MakeSocLabel(uint32_t htp_arch, uint32_t soc_model) {
  std::string arch = EncodeHtpArch(htp_arch);
  if (arch.empty()) {
    return "soc_model=" + std::to_string(soc_model);
  }
  if (soc_model != kSocUnknown) {
    return arch + "/soc_model=" + std::to_string(soc_model);
  }
  return arch;
}

std::vector<UnsupportedNodeInfo> MergePerSocUnsupportedNodes(
    const std::vector<PerSocUnsupportedNodes>& per_soc) {
  // One merged node: identity fields plus reasons (first-seen order), each with
  // its contributing SoC labels. Label lists are tiny (<= SoC count), so dedup
  // uses a linear find rather than a parallel set.
  struct Merged {
    std::string node_name;
    std::string op_type;
    size_t node_index = 0;
    std::vector<std::string> reason_order;  // distinct reasons, first-seen
    std::unordered_map<std::string, std::vector<std::string>> soc_labels_by_reason;
  };

  std::vector<size_t> node_order;  // first-seen node_index order
  std::unordered_map<size_t, Merged> merged;

  for (const auto& soc : per_soc) {
    for (const auto& un : soc.nodes) {
      auto it = merged.find(un.node_index);
      if (it == merged.end()) {
        node_order.push_back(un.node_index);
        Merged m;
        m.node_name = un.node_name;
        m.op_type = un.op_type;
        m.node_index = un.node_index;
        it = merged.emplace(un.node_index, std::move(m)).first;
      }
      Merged& m = it->second;

      // New reason for this node? Record its first-seen position.
      auto& labels = m.soc_labels_by_reason[un.reason];
      if (labels.empty()) {
        m.reason_order.push_back(un.reason);
      }
      // Attribute this SoC to the reason, deduping identical (reason, label).
      if (std::find(labels.begin(), labels.end(), soc.soc_label) == labels.end()) {
        labels.push_back(soc.soc_label);
      }
    }
  }

  std::vector<UnsupportedNodeInfo> out;
  out.reserve(node_order.size());
  for (size_t node_index : node_order) {
    const Merged& m = merged[node_index];
    // Format: "<labels joined by ','>: <reason>" fragments joined by "; ".
    std::string reason;
    for (size_t r = 0; r < m.reason_order.size(); ++r) {
      const std::string& reason_text = m.reason_order[r];
      const std::vector<std::string>& labels = m.soc_labels_by_reason.at(reason_text);
      if (r != 0) {
        reason += "; ";
      }
      for (size_t l = 0; l < labels.size(); ++l) {
        if (l != 0) {
          reason += ",";
        }
        reason += labels[l];
      }
      reason += ": ";
      reason += reason_text;
    }
    out.push_back({m.node_name, m.op_type, m.node_index, std::move(reason)});
  }
  return out;
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
  j["schema_version"] = kFrameworkOpTraceSchemaVersion;
  j["model_name"] = trace.model_name;
  j["backend_type"] = trace.backend_type;

  // One entry per SoC iteration; compilation_target lives here, not at root.
  nlohmann::json soc_traces = nlohmann::json::array();
  for (const auto& soc : trace.soc_traces) {
    nlohmann::json soc_json;
    nlohmann::json ct;
    ct["htp_arch"] = soc.compilation_target.htp_arch;
    ct["soc_model"] = soc.compilation_target.soc_model;
    ct["device_id"] = soc.compilation_target.device_id;
    soc_json["compilation_target"] = std::move(ct);
    soc_traces.push_back(std::move(soc_json));
  }
  j["soc_traces"] = std::move(soc_traces);

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
  summary["total_socs"] = s.total_socs;

  nlohmann::json fc;
  for (const auto& [type, count] : s.fusion_count) {
    fc[type] = count;
  }
  summary["fusion_count"] = std::move(fc);
  j["summary"] = std::move(summary);

  return j;
}

std::filesystem::path DeriveTracePathFromContextModel(const std::filesystem::path& ctx_model_path) {
  return ctx_model_path.parent_path() / "qnn_op_trace.json";
}

TraceLoadStatus ParseTraceLookupFromFile(const std::filesystem::path& trace_path,
                                         OpTraceLookup& out_lookup,
                                         size_t* skipped_entries) {
  std::ifstream ifs(trace_path);
  if (!ifs.is_open()) {
    return TraceLoadStatus::kCannotOpen;
  }
  auto j = nlohmann::json::parse(ifs, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) {
    return TraceLoadStatus::kParseError;
  }
  if (!j.contains("subgraph_traces")) {
    return TraceLoadStatus::kMissingSubgraphTraces;
  }
  size_t skipped = 0;
  for (const auto& sg : j.at("subgraph_traces")) {
    if (!sg.contains("op_mappings")) {
      continue;
    }
    for (const auto& m : sg.at("op_mappings")) {
      std::string dst_name = m.value("dst_name", "");
      if (dst_name.empty() || !m.contains("sources")) {
        ++skipped;
        continue;
      }
      std::vector<TraceSourcePair> sources;
      for (const auto& src : m.at("sources")) {
        std::string name = src.value("name", "");
        std::string type_str = src.value("type", "OP");
        sources.push_back({std::move(name),
                           type_str == "OP" ? TraceTargetType::kOp : TraceTargetType::kTensor});
      }
      out_lookup[std::move(dst_name)] = std::move(sources);
    }
  }
  if (skipped_entries != nullptr) {
    *skipped_entries = skipped;
  }
  return TraceLoadStatus::kOk;
}

}  // namespace qnn
}  // namespace onnxruntime
