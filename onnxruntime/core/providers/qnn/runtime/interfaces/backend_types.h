// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace onnxruntime::qnn {

// EP-owned backend identity. This is shared by the QNN C and QAIRT C++
// implementations and is not an SDK enum.
enum class QnnBackendType : uint8_t {
  CPU = 0,
  GPU,
  DSP,
  HTP,
  HTP_FP16,
  SERIALIZER,
};

bool IsIrBackend(QnnBackendType backend_type);
bool IsCpuBackend(QnnBackendType backend_type);
bool IsNpuBackend(QnnBackendType backend_type);
bool IsGpuBackend(QnnBackendType backend_type);
bool IsQpuBackend(QnnBackendType backend_type);
std::string QnnBackendTypeToString(QnnBackendType backend_type);

}  // namespace onnxruntime::qnn
