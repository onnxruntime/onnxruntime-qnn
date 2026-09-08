// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// EP-internal helper that serializes the ONNX graph the QNN EP receives in
// GetCapabilityImpl (compile-time, before partitioning) into a JSON file using
// the same schema as the post-compile `dump_json_qnn_graph` output, so the
// result can be opened in QNN Netron and consumed by the offline
// source->optimized op-trace matcher.
//
// This captures the graph AFTER ORT Level 1 optimizations (QDQ preserved,
// `/duplicated` DQ copies, Transpose rearrangements) but BEFORE the QNN EP
// compiles its partition, which is otherwise not observable: the ORT optimized-
// model serializer refuses graphs containing compiled nodes.

#pragma once

#include <filesystem>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Walks the EP-input OrtGraph and writes a QNN-Netron-schema JSON description
// to `output_path`. Returns true on success; logs a WARNING and returns false
// on any failure (the caller treats the dump as a best-effort diagnostic and
// continues compilation regardless).
bool DumpQnnEpInputGraphToJson(const OrtGraph* graph,
                               const std::filesystem::path& output_path,
                               const Ort::Logger& logger);

}  // namespace qnn
}  // namespace onnxruntime
