// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <QnnProfile.h>
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime::qnn {

// Supplies profiling data to the common serializer without sharing backend handles.
class IBackendProfileApi {
 public:
  virtual ~IBackendProfileApi() = default;
  virtual Ort::Status Create(ProfilingLevel level) = 0;
  virtual void Release() = 0;
  virtual bool Active() const = 0;
  virtual Qnn_ErrorHandle_t GetEvents(const QnnProfile_EventId_t**, uint32_t*) = 0;
  virtual Qnn_ErrorHandle_t GetSubEvents(QnnProfile_EventId_t, const QnnProfile_EventId_t**, uint32_t*) = 0;
  virtual Qnn_ErrorHandle_t GetEventData(QnnProfile_EventId_t, QnnProfile_EventData_t*) = 0;
  virtual Qnn_ErrorHandle_t GetExtendedEventData(QnnProfile_EventId_t, QnnProfile_ExtendedEventData_t*) = 0;
};

}  // namespace onnxruntime::qnn
