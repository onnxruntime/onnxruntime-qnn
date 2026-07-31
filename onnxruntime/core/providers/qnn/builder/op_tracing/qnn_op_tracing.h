// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// EP-internal header: OpTraceCollector / NodeGroupGuard, and the
// EP-side wrappers WriteTraceToFile and LoadTraceLookupFromFile (the latter
// adds Ort logger plumbing on top of the test-safe ParseTraceLookupFromFile
// declared in qnn_op_tracing_types.h).
// Test TUs should include qnn_op_tracing_types.h (which also declares
// SerializeFrameworkOpTrace via nlohmann/json_fwd.hpp) and add
// nlohmann/json.hpp directly if they use the return value.

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "core/providers/qnn/common/inlined_containers.h"
#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing_types.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class IQnnNodeGroup;
class QnnModelWrapper;

// Collects ONNX-to-QNN op/tensor mappings during a single ComposeGraph() call.
// Lifetime: stack-allocated in QnnModel::ComposeGraph(), pointer passed to QnnModelWrapper.
class OpTraceCollector {
 public:
  OpTraceCollector() = default;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OpTraceCollector);

  // Called from QnnModelWrapper::CreateQnnNode() when a QNN op is created.
  // Also records a tensor mapping for each output tensor using the same sources.
  void RecordOpMapping(const std::string& qnn_op_name, const std::string& qnn_op_type,
                       gsl::span<const std::string> output_tensor_names);

  // Finalize the subgraph trace: record tensor mappings from the wrapper, populate
  // `out_info` with the collected op_mappings, tensor_mappings, and graph name,
  // and also populate `out_lookup` with the dst_name->sources projection of
  // op_mappings (used for profiling enrichment at execute time).
  // Call once after ComposeQnnGraph().
  void Finalize(const std::string& graph_name,
                const QnnModelWrapper& wrapper,
                OpTraceInfo& out_info,
                OpTraceLookup& out_lookup);

 private:
  // Set/clear the current node group context. Only NodeGroupGuard may call these
  // to preserve the set/clear pairing and exception-safety invariant.
  friend class NodeGroupGuard;
  void SetCurrentNodeGroup(const IQnnNodeGroup* node_group);
  void ClearCurrentNodeGroup();

  const IQnnNodeGroup* current_node_group_ = nullptr;
  // Real subgraphs frequently exceed any small inline capacity (LLM subgraphs
  // can have hundreds of ops/tensors), so use std::vector and let it grow on
  // the heap. The trace path is debug-only, so the extra allocation is
  // immaterial.
  std::vector<TraceMapping> op_mappings_;
  std::vector<TraceMapping> tensor_mappings_;
  // Tracks QNN tensor names already recorded by RecordOpMapping so that
  // Finalize can skip them and avoid duplicate entries.
  InlinedHashSet<std::string> recorded_tensor_names_;
};

// RAII guard that brackets SetCurrentNodeGroup / ClearCurrentNodeGroup around one
// AddToModelBuilder call so that ClearCurrentNodeGroup is always called even if
// AddToModelBuilder returns early via an exception.
class NodeGroupGuard {
 public:
  NodeGroupGuard(OpTraceCollector* c, const IQnnNodeGroup* g) : c_(c) {
    if (c_) c_->SetCurrentNodeGroup(g);
  }
  ~NodeGroupGuard() {
    if (c_) c_->ClearCurrentNodeGroup();
  }
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(NodeGroupGuard);

 private:
  OpTraceCollector* c_;
};

// Write trace JSON to file. Returns true on success.
bool WriteTraceToFile(const FrameworkOpTrace& trace,
                      const std::filesystem::path& output_path,
                      const Ort::Logger& logger);

// Owns the session's FrameworkOpTrace. NewSubgraphSlot / UnsupportedNodes hand
// out interior references valid only until the next mutating call.
class FrameworkOpTraceBuilder {
 public:
  // GetCapability runs twice per graph; reset before each pass so pass 2 does
  // not double-count.
  void Reset() { trace_ = FrameworkOpTrace{}; }

  // Record one SoC iteration. Raw ids (0 = unknown, omitted); uint32_t keeps
  // this header SDK-header-free.
  void AppendSoc(uint32_t htp_arch, uint32_t soc_model, uint32_t device_id);

  // Reserve a subgraph slot for ComposeGraph to fill (valid until the next call).
  OpTraceInfo* NewSubgraphSlot() {
    trace_.subgraph_traces.push_back(OpTraceInfo{});
    return &trace_.subgraph_traces.back();
  }

  // Out-param the support scan appends rejected nodes into.
  std::vector<UnsupportedNodeInfo>& UnsupportedNodes() { return trace_.unsupported_nodes; }

  // Compute the summary and write the file, skipping a wholly-empty trace.
  bool FinalizeAndWrite(const std::string& model_name,
                        const std::string& backend_type,
                        const std::filesystem::path& output_path,
                        const Ort::Logger& logger);

 private:
  FrameworkOpTrace trace_;
};

// Loads an OpTraceLookup from a trace JSON sidecar file.
// Iterates all subgraph_traces[*].op_mappings and builds a flat dst_name->sources map.
// Returns true on success; logs a warning and returns false on any parse error.
bool LoadTraceLookupFromFile(const std::filesystem::path& trace_path,
                             OpTraceLookup& out_lookup,
                             const Ort::Logger& logger);

}  // namespace qnn
}  // namespace onnxruntime
