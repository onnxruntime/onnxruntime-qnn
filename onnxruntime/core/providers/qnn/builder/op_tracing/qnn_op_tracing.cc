// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing.h"

#include <algorithm>
#include <fstream>

#include "nlohmann/json.hpp"

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

// ---------------------------------------------------------------------------
// OpTraceCollector
// ---------------------------------------------------------------------------

void OpTraceCollector::SetCurrentNodeGroup(const IQnnNodeGroup* node_group) {
  current_node_group_ = node_group;
}

void OpTraceCollector::ClearCurrentNodeGroup() {
  current_node_group_ = nullptr;
}

void OpTraceCollector::RecordOpMapping(const std::string& qnn_op_name, const std::string& qnn_op_type,
                                       gsl::span<const std::string> output_tensor_names) {
  // Guard: node group context is set by NodeGroupGuard around each AddToModelBuilder call.
  // If nullptr, this QNN op was created outside that bracket (e.g. BF16
  // conversion ops Cast added during ComposeQnnGraph postprocessing),
  // which has no corresponding ONNX node group - skip it.
  if (current_node_group_ == nullptr) {
    return;
  }

  TraceMapping mapping;
  mapping.dst_name = qnn_op_name;
  mapping.dst_qnn_op_type = qnn_op_type;
  mapping.node_group_type = std::string(current_node_group_->Type());

  // Collect all raw nodes into a flat list. Inline capacity 8 covers the
  // typical case (most node groups have <= 8 raw nodes; see
  // qnn_execution_provider.cc).
  InlinedVector<const OrtNode*, 8> all_nodes;
  for (const OrtNodeUnit* node_unit : current_node_group_->GetNodeUnits()) {
    for (const OrtNode* node : node_unit->GetAllNodesInGroup()) {
      all_nodes.push_back(node);
    }
  }

  // Build srcInfo chain:
  //   op1, tensor_out1, op2, tensor_out2, ..., opN
  // Each source op is followed by the output tensor that actually flows into
  // the next op in the chain (found by intersecting node[i]'s outputs with
  // node[i+1]'s inputs). Falls back to outputs[0] if no match is found.
  for (size_t i = 0; i < all_nodes.size(); ++i) {
    Ort::ConstNode const_node(all_nodes[i]);
    // Node names are optional per ONNX spec. Fall back to OpType+position,
    // mirroring UniqueNameGenerator::New(node_unit)'s OpType+Index fallback
    // so that source names are structurally consistent with QNN op names
    // in the 1:1 case. The position index `i` differs from the graph-global
    // NodeUnit index used by UniqueNameGenerator, so the names may not match
    // exactly for unnamed nodes — but they remain unique within this group and
    // are typed as OP (not a tensor name masquerading as an OP source).
    std::string node_name = std::string(const_node.GetName());
    if (node_name.empty()) {
      node_name = std::string(const_node.GetOperatorType()) + std::to_string(i);
    }
    mapping.sources.push_back({std::move(node_name), TraceTargetType::kOp});

    // Add linking tensor for all nodes except the last.
    if (i + 1 < all_nodes.size()) {
      std::vector<Ort::ConstValueInfo> outputs = const_node.GetOutputs();
      if (!outputs.empty()) {
        // Collect this node's output names for O(1) lookup.
        InlinedHashSet<std::string> output_names;
        for (const Ort::ConstValueInfo& out : outputs) {
          output_names.insert(std::string(out.GetName()));
        }

        // Find all outputs consumed by the next node in the chain.
        // Note: Ort::ConstValueInfo wraps a nullable OrtValueInfo pointer;
        // there is no operator bool(), so we extract the raw pointer to check.
        InlinedVector<std::string, 8> linking_tensors;
        Ort::ConstNode next_node(all_nodes[i + 1]);
        for (const Ort::ConstValueInfo& in : next_node.GetInputs()) {
          const OrtValueInfo* raw_in = in;
          if (!raw_in) continue;  // optional input not provided
          std::string in_name = std::string(in.GetName());
          if (output_names.count(in_name)) {
            linking_tensors.push_back(std::move(in_name));
          }
        }

        // Fall back to outputs[0] if the intersection is empty (should not
        // happen for a well-formed fusion, but avoids a missing entry).
        if (linking_tensors.empty()) {
          linking_tensors.push_back(std::string(outputs[0].GetName()));
        }

        for (std::string& t : linking_tensors) {
          mapping.sources.push_back({std::move(t), TraceTargetType::kTensor});
        }
      }
    }
  }

  if (mapping.sources.empty()) {
    return;
  }

  // Record tensor mappings for each output tensor of this QNN op.
  // Each output tensor carries the same srcInfo chain as the op
  // (N:1 or N:M — same sources, different dst_name).
  for (const auto& out_name : output_tensor_names) {
    TraceMapping t;
    t.dst_name = out_name;
    t.sources = mapping.sources;  // shared sources, copy intentional
    tensor_mappings_.push_back(std::move(t));
    recorded_tensor_names_.insert(out_name);
  }

  op_mappings_.push_back(std::move(mapping));
}

void OpTraceCollector::Finalize(const std::string& graph_name,
                                const QnnModelWrapper& wrapper,
                                OpTraceInfo& out_info,
                                OpTraceLookup& out_lookup) {
  // Records the direct ONNX<->QNN tensor correspondence when a tensor was
  // renamed by an op builder (onnx_name != qnn_name).
  //
  // Identity mappings (onnx_name == qnn_name, e.g. graph inputs and
  // initializers that flow through unchanged) are NOT recorded — they carry
  // no information for downstream tools and would inflate the trace with
  // zero-information entries for every graph input and weight tensor.
  //
  // Output tensors of QNN ops are recorded by RecordOpMapping (with the op's
  // full srcInfo chain) and are therefore also excluded here to avoid duplicates.
  //
  // Two emission rules ensure each qnn_name appears at most once in
  // tensor_mappings_:
  //   * Option A (merge): when several wrappers resolve to the same qnn_name
  //     for non-canonical reasons, all distinct internal source names are
  //     merged into one entry's sources list.
  //   * Option B (canonical): when qnn_name is an external override target
  //     (i.e. produced by `offload_graph_io_quantization` aliasing a QNN
  //     internal tensor back to its original ONNX graph-input/output name),
  //     the trace records the canonical mapping {dst: qnn_name, sources:
  //     [qnn_name]}. The intermediate internal names (q_*_out / dq_*_out)
  //     are an artefact of the alias and carry no information for downstream
  //     tools, so they are dropped.
  const auto& tensors_map = wrapper.GetModelTensorsMap();

  // First pass: group unique source names by qnn_name to deduplicate.
  InlinedHashMap<std::string, InlinedVector<std::string, 8>> grouped;
  for (const auto& [key, tensor_wrapper] : tensors_map) {
    const std::string& onnx_name = tensor_wrapper.GetName();
    const std::string& qnn_name = tensor_wrapper.GetResolvedTensorName();
    if (onnx_name == qnn_name || recorded_tensor_names_.count(qnn_name)) {
      continue;  // identity or already recorded by RecordOpMapping
    }
    auto& sources = grouped[qnn_name];
    if (std::find(sources.begin(), sources.end(), onnx_name) == sources.end()) {
      sources.push_back(onnx_name);
    }
  }

  // Second pass: emit one entry per unique qnn_name, applying B or A.
  for (auto& [qnn_name, sources] : grouped) {
    TraceMapping mapping;
    mapping.dst_name = qnn_name;
    if (wrapper.IsExternalOverrideTarget(qnn_name)) {
      // Option B: canonical mapping.
      mapping.sources.push_back({qnn_name, TraceTargetType::kTensor});
    } else {
      // Option A: merged sources.
      for (auto& src : sources) {
        mapping.sources.push_back({std::move(src), TraceTargetType::kTensor});
      }
    }
    tensor_mappings_.push_back(std::move(mapping));
    recorded_tensor_names_.insert(qnn_name);
  }

  out_info.graph_name = graph_name;
  out_info.op_mappings = std::move(op_mappings_);
  out_info.tensor_mappings = std::move(tensor_mappings_);

  // Build the dst_name->sources lookup used for profiling enrichment at execute time.
  for (const auto& mapping : out_info.op_mappings) {
    out_lookup[mapping.dst_name] = mapping.sources;
  }
}

bool WriteTraceToFile(const FrameworkOpTrace& trace,
                      const std::filesystem::path& output_path,
                      const Ort::Logger& logger) {
  std::error_code ec;
  auto parent = output_path.parent_path();
  // parent is empty when output_path is a filename-only path (write to CWD).
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                  ("Failed to create trace output directory: " + parent.string() +
                   " error: " + ec.message())
                      .c_str());
      return false;
    }
  }

  nlohmann::json j = SerializeFrameworkOpTrace(trace);

  // Intentional overwrite: the trace file is a diagnostic artefact and the
  // most recent run's data is always the most useful.  Log a warning so the
  // developer is aware, but do not treat it as an error.
  if (std::filesystem::exists(output_path)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("Overwriting existing framework op trace file: " + output_path.string()).c_str());
  }

  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("Could not open framework op trace file: " + output_path.string()).c_str());
    return false;
  }

  ofs << j.dump(2);  // pretty-print with 2-space indent
  ofs.close();
  if (!ofs) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("Framework op trace write failed (disk full / I/O error): " + output_path.string()).c_str());
    std::error_code rm_ec;
    std::filesystem::remove(output_path, rm_ec);
    return false;
  }

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
              ("Framework op trace written to: " + output_path.string()).c_str());
  return true;
}

bool LoadTraceLookupFromFile(const std::filesystem::path& trace_path,
                             OpTraceLookup& out_lookup,
                             const Ort::Logger& logger) {
  // Parsing lives in the logger-free ParseTraceLookupFromFile (test-safe
  // translation unit); this wrapper adds the EP logger, turning each load
  // status into a log line.
  size_t skipped_entries = 0;
  switch (ParseTraceLookupFromFile(trace_path, out_lookup, &skipped_entries)) {
    case TraceLoadStatus::kCannotOpen:
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                  ("Cannot open op trace sidecar: " + trace_path.string()).c_str());
      return false;
    case TraceLoadStatus::kParseError:
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                  ("Failed to parse op trace sidecar JSON: " + trace_path.string()).c_str());
      return false;
    case TraceLoadStatus::kMissingSubgraphTraces:
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                  ("Op trace sidecar missing required `subgraph_traces` key: " + trace_path.string()).c_str());
      return false;
    case TraceLoadStatus::kOk:
      if (skipped_entries > 0) {
        ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                    ("Op trace sidecar had " + std::to_string(skipped_entries) +
                     " malformed op_mappings entries (missing dst_name or sources) "
                     "that were skipped: " +
                     trace_path.string())
                        .c_str());
      }
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
                  ("Loaded op trace sidecar for profiling enrichment: " + trace_path.string()).c_str());
      return true;
  }
  return false;  // unreachable; silences -Wreturn-type on some compilers
}

}  // namespace qnn
}  // namespace onnxruntime
