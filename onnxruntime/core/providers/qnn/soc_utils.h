// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

namespace onnxruntime {
namespace qnn {
namespace soc {

int GetSocId();

// Returns true on Linux/Android arm64 if a Hexagon fastRPC compute-DSP char device
// (/dev/fastrpc-cdsp*) is present. Always false elsewhere.
bool HasFastRpcCdspDevice();

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
