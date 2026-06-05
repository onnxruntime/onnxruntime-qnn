// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// EP-internal header: OpTraceCollector class definition and WriteTraceToFile.
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

  // Finalize the subgraph trace: record tensor mappings from the wrapper and
  // populate `out` with the collected op_mappings, tensor_mappings, and the
  // graph name. Call once after ComposeQnnGraph().
  void Finalize(const std::string& graph_name, const QnnModelWrapper& wrapper, OpTraceInfo& out);

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

}  // namespace qnn
}  // namespace onnxruntime
