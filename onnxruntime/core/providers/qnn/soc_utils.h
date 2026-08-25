// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string_view>

namespace onnxruntime {
namespace qnn {
namespace soc {

int GetSocId();

// Returns true on Android arm64 if ro.soc.manufacturer is "QTI" (case-insensitive).
// Returns true on Linux arm64 if /dev/fastrpc-cdsp* is present. Always false elsewhere.
bool HasFastRpcCdspDevice();

// Resolves a SoC model name (e.g. "SM8750", case-insensitive) to the corresponding
// Qnn_SocModel_t integer value. Returns 0 (QNN_SOC_MODEL_UNKNOWN) if not found.
uint32_t SocModelFromName(std::string_view name);

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
