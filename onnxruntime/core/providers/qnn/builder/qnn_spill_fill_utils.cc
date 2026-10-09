// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_spill_fill_utils.h"

#include <algorithm>
#include <string>

#include "HTP/QnnHtpSystemContext.h"

#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

uint64_t GetMaxSpillFillBufferSizeFromGraphInfo(const QnnSystemContext_GraphInfo_t* graphs_info,
                                                uint32_t graph_count) {
  uint64_t max_spill_fill_buffer_size = 0;

#ifdef QNN_HTP_SPILL_FILL_BUFFER_AVAILABLE
  if (graphs_info == nullptr) {
    return max_spill_fill_buffer_size;
  }

  for (uint32_t graph_idx = 0; graph_idx < graph_count; ++graph_idx) {
    if (graphs_info[graph_idx].version != QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      continue;
    }

    const auto* htp_graph_info = reinterpret_cast<const QnnHtpSystemContext_GraphBlobInfo_t*>(
        graphs_info[graph_idx].graphInfoV3.graphBlobInfo);
    if (htp_graph_info == nullptr ||
        htp_graph_info->version != QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
      continue;
    }

    max_spill_fill_buffer_size =
        std::max(max_spill_fill_buffer_size,
                 static_cast<uint64_t>(htp_graph_info->contextBinaryGraphBlobInfoV1.spillFillBufferSize));
  }
#else
  ORT_UNUSED_PARAMETER(graphs_info);
  ORT_UNUSED_PARAMETER(graph_count);
#endif

  return max_spill_fill_buffer_size;
}

Ort::Status ValidateSpillFillBufferSize(int64_t declared_max_spill_fill_size,
                                        uint64_t derived_max_spill_fill_size) {
  RETURN_IF(declared_max_spill_fill_size < 0, "EPContext max_size must not be negative.");

#ifdef QNN_HTP_SPILL_FILL_BUFFER_AVAILABLE
  RETURN_IF(static_cast<uint64_t>(declared_max_spill_fill_size) > derived_max_spill_fill_size,
            ("EPContext max_size " + std::to_string(declared_max_spill_fill_size) +
             " exceeds the spill-fill size " + std::to_string(derived_max_spill_fill_size) +
             " declared by the context binary.")
                .c_str());
#else
  ORT_UNUSED_PARAMETER(derived_max_spill_fill_size);
  RETURN_IF(declared_max_spill_fill_size != 0,
            "EPContext max_size is unsupported by this QNN SDK version.");
#endif

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
