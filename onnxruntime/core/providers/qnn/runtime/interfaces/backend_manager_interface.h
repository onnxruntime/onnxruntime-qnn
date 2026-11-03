// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "HTP/QnnHtpDevice.h"
#include "core/providers/qnn/runtime/interfaces/backend_types.h"

namespace onnxruntime::qnn {

class IGraphEmitter;

enum class BackendRuntimeKind {
  Qnn,
  Qairt,
};

// Common runtime boundary used by graph-building code. Keep this interface
// deliberately small: EP orchestration and runtime-specific services such as
// DLC authoring, memory registration, power management, and context recovery
// belong on focused components rather than growing this manager.
class IBackendManager {
 public:
  virtual ~IBackendManager() = default;

  virtual BackendRuntimeKind GetRuntimeKind() const = 0;
  virtual QnnBackendType GetBackendType() const = 0;
  virtual QnnHtpDevice_Arch_t GetHtpArch() const = 0;
  virtual uint32_t GetSocModel() const = 0;
  virtual std::unique_ptr<IGraphEmitter> CreateGraphEmitter() = 0;
};

}  // namespace onnxruntime::qnn
