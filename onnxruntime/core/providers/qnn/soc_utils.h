// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace qnn {
namespace soc {

int GetSocId();

// PATCH A: Returns QNN_SOC_MODEL_* for the current device, or 0 (UNKNOWN)
// if not detected. Windows only; returns 0 on other platforms.
uint32_t DetectQnnSocModel();

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
