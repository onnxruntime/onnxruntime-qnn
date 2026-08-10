// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

namespace onnxruntime {
namespace qnn {
namespace soc {

int GetSocId();

// Returns true on Android arm64 if ro.soc.manufacturer is "QTI" (case-insensitive).
// Returns true on Linux arm64 if /dev/fastrpc-cdsp* is present. Always false elsewhere.
bool HasFastRpcCdspDevice();

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
