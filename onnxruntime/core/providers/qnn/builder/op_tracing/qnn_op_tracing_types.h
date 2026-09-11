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
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json_fwd.hpp"

namespace onnxruntime {
namespace qnn {

// Unknown-value sentinel for htp_arch/soc_model (mirrors the SDK's 0-valued
// QNN_HTP_DEVICE_ARCH_NONE / QNN_SOC_MODEL_UNKNOWN; kept SDK-header-free here).
inline constexpr uint32_t kSocUnknown = 0;

// qnn_op_trace.json schema version; bumped on each breaking change (v2 = FCB soc_traces[]).
inline constexpr int kFrameworkOpTraceSchemaVersion = 2;

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
  // Single-SoC: raw QNN validator message. Multi-SoC: SoC-attributed by
  // MergePerSocUnsupportedNodes, e.g. "V73: A; V81: B".
  std::string reason;
};

struct TraceSummary {
  size_t total_onnx_nodes = 0;
  // Unique ONNX nodes appearing as OP-typed sources, deduped by source name.
  size_t supported_nodes = 0;
  size_t unsupported_nodes = 0;
  size_t qnn_subgraphs = 0;
  size_t total_qnn_ops = 0;
  // SoC iteration count (single-SoC = 1; multi-SoC FCB = N).
  uint32_t total_socs = 0;
  // std::map for deterministic, diff-friendly JSON key order.
  std::map<std::string, size_t> fusion_count;
};

struct CompilationTarget {
  std::string htp_arch;
  uint32_t soc_model = 0;
  uint32_t device_id = 0;
};

// One SoC iteration's target. Single-SoC = one SocTrace; multi-SoC FCB = one per SoC.
struct SocTrace {
  CompilationTarget compilation_target;
};

struct FrameworkOpTrace {
  std::string model_name;
  std::string backend_type;
  // One entry per SoC iteration.
  std::vector<SocTrace> soc_traces;
  // SoC-agnostic op mappings; stays at root and is collected once (not per SoC).
  std::vector<OpTraceInfo> subgraph_traces;
  std::vector<UnsupportedNodeInfo> unsupported_nodes;
  TraceSummary summary;
};

// Compute summary statistics for a FrameworkOpTrace.
// Defined in qnn_op_tracing_serialization.cc.
void ComputeTraceSummary(FrameworkOpTrace& trace);

// Encode a raw HTP arch number as "V<n>", or "" for 0 (QNN_HTP_DEVICE_ARCH_NONE).
// Single source of truth for the arch-string form; SDK-header-free (uint32_t).
std::string EncodeHtpArch(uint32_t htp_arch);

// Build the SoC label attributing a per-SoC rejection. `htp_arch` / `soc_model`
// are raw ids (0 = unknown). Format:
//   - arch + soc_model: "V73/soc_model=60"
//   - arch only:        "V73"
//   - soc_model only:   "soc_model=60"
std::string MakeSocLabel(uint32_t htp_arch, uint32_t soc_model);

// One SoC's rejections, tagged with its SoC label (see MakeSocLabel).
struct PerSocUnsupportedNodes {
  std::string soc_label;
  std::vector<UnsupportedNodeInfo> nodes;

  PerSocUnsupportedNodes() = default;
  // From a pre-built label.
  PerSocUnsupportedNodes(std::string soc_label, std::vector<UnsupportedNodeInfo> nodes)
      : soc_label(std::move(soc_label)), nodes(std::move(nodes)) {}
  // From raw ids, so call sites pass (htp_arch, soc_model) directly.
  PerSocUnsupportedNodes(uint32_t htp_arch, uint32_t soc_model, std::vector<UnsupportedNodeInfo> nodes)
      : soc_label(MakeSocLabel(htp_arch, soc_model)), nodes(std::move(nodes)) {}
};

// Merge per-SoC rejections into one UnsupportedNodeInfo per ONNX node, so the row
// count equals the distinct-node count. Each distinct reason is attributed to its
// SoC(s), in first-seen order:
//   - same reason on V73 and V81 -> "V73,V81: R"
//   - different reasons          -> "V73: A; V81: B"
// node_name/op_type come from the first SoC to report the node.
std::vector<UnsupportedNodeInfo> MergePerSocUnsupportedNodes(
    const std::vector<PerSocUnsupportedNodes>& per_soc);

// Serialize a FrameworkOpTrace to JSON.
// Defined in qnn_op_tracing_serialization.cc.  Call sites that use the return value must
// include nlohmann/json.hpp separately.
nlohmann::json SerializeFrameworkOpTrace(const FrameworkOpTrace& trace);

// Derives the AOT Phase 2 sidecar trace JSON path from a context model path.
// Returns {parent_dir}/qnn_op_trace.json - the same filename Phase 1 writes when
// framework_op_trace_dir is set to the context model's directory, so Phase 2
// discovery is automatic without a manual rename step.
// E.g. "/path/model_ctx.onnx" -> "/path/qnn_op_trace.json"
// Defined in qnn_op_tracing_serialization.cc (test-safe translation unit).
std::filesystem::path DeriveTracePathFromContextModel(const std::filesystem::path& ctx_model_path);

// Lookup map: QNN op name -> ONNX source pairs.
// Populated by OpTraceCollector at compile time;
// persists in QnnModel for profiling enrichment at execute time.
using OpTraceLookup = std::unordered_map<std::string, std::vector<TraceSourcePair>>;

// Outcome of ParseTraceLookupFromFile, so callers can map each failure mode to
// an appropriate log message without the parser needing a logger dependency.
enum class TraceLoadStatus {
  kOk,                     // parsed successfully (out_lookup populated)
  kCannotOpen,             // file could not be opened
  kParseError,             // file is not valid JSON
  kMissingSubgraphTraces,  // valid JSON but missing the required top-level key
};

// Parses a trace JSON sidecar into a flat dst_name->sources lookup, iterating
// all subgraph_traces[*].op_mappings. Defined in qnn_op_tracing_serialization.cc
// with no Ort/logger dependency (the LoadTraceLookupFromFile wrapper supplies
// logging). On any non-kOk status `out_lookup` is left unmodified.
//
// `skipped_entries` (optional) accumulates a count of malformed `op_mappings`
// entries (empty `dst_name`, missing `sources`) that were silently skipped to
// keep the loader forward-compatible with future schema additions. The wrapper
// can use this to surface a warning when nonzero, distinguishing "empty
// lookup because schema mismatch" from "empty lookup because no trace data".
TraceLoadStatus ParseTraceLookupFromFile(const std::filesystem::path& trace_path,
                                         OpTraceLookup& out_lookup,
                                         size_t* skipped_entries = nullptr);

}  // namespace qnn
}  // namespace onnxruntime
