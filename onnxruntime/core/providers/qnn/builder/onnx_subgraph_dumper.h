// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <string>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Serialize the QNN-claimed partition's per-partition OrtGraph (the one ORT synthesizes for
// CompileImpl) as a self-contained, runnable ONNX model written to `output_path`. Called from
// `QnnEp::CompileImpl` immediately before `QnnModel::ComposeGraph` — i.e. after ORT has computed
// the partition I/O cuts but before any QNN op-builder rewrites the op_types into QNN_OP_*.
//
// `graph_name` is embedded as the GraphProto name; the caller passes the fused_node_name so
// the dumped file's name matches the QNN graph name that subsequently appears in profiler /
// JSON dump output.
//
// Hard-fails (returns non-OK Status, no file written) when:
//   - any node carries a GRAPH attribute (If/Loop subgraph) — round-trip would be incomplete
//   - any required initializer cannot be located
//   - the file cannot be opened for writing
//
// All initializers consumed by the partition are inlined as TensorProto.raw_data so the dumped
// file is self-contained even when the source model used external data.
Ort::Status DumpPartitionAsOnnxModel(const OrtApi& ort_api,
                                     const OrtGraph& subgraph,
                                     const std::string& graph_name,
                                     const std::filesystem::path& output_path,
                                     const Ort::Logger& logger);

}  // namespace qnn
}  // namespace onnxruntime
