// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "System/QnnSystemContext.h"

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Derives the largest spill-fill requirement carried by parsed QNN graph metadata.
// Unsupported graph/blob versions do not carry a usable spill-fill value and are ignored.
uint64_t GetMaxSpillFillBufferSizeFromGraphInfo(const QnnSystemContext_GraphInfo_t* graphs_info,
                                                uint32_t graph_count);

// Validates an EPContext node's declared max_size against the binary that node carries and
// ensures that the effective spill-fill group is large enough for the binary's requirement.
// A zero declaration is the legacy missing-attribute case; every nonzero declaration must match.
// A zero group size is used during preflight or when spill-fill group registration is not enabled.
Ort::Status ValidateSpillFillBufferSize(int64_t declared_max_spill_fill_size,
                                        uint64_t derived_max_spill_fill_size,
                                        int64_t group_max_spill_fill_size);

}  // namespace qnn
}  // namespace onnxruntime
