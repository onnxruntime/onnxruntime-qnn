// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once
#ifdef USE_QAIRT_API
#include "QairtCpp/QairtProfile.hpp"
#include "QairtCpp/QairtBackend.hpp"

#include <map>
#include "core/providers/qnn/qnn_ep_profiler.h"
#include "core/providers/qnn/builder/backend_profile_api.h"
#include "core/providers/qnn/builder/qairt_backend_manager.h"

namespace onnxruntime::qnn {

class QairtProfileAdapter final : public IBackendProfileApi {
 public:
  explicit QairtProfileAdapter(std::shared_ptr<QairtBackendManager> manager) : manager_(std::move(manager)) {}
  Ort::Status Create(ProfilingLevel level) override;
  Ort::Status WriteLog(const std::string& path, const std::string& graph, uint64_t start, uint64_t stop,
                       OrtProfilingOperation operation);
  void Release() override;
  bool Active() const override { return profile_ != nullptr; }
  std::shared_ptr<qairt::Profile> GetProfile() const { return profile_; }
  Qnn_ErrorHandle_t GetEvents(const QnnProfile_EventId_t**, uint32_t*) override;
  Qnn_ErrorHandle_t GetSubEvents(QnnProfile_EventId_t, const QnnProfile_EventId_t**, uint32_t*) override;
  Qnn_ErrorHandle_t GetEventData(QnnProfile_EventId_t, QnnProfile_EventData_t*) override;
  Qnn_ErrorHandle_t GetExtendedEventData(QnnProfile_EventId_t, QnnProfile_ExtendedEventData_t*) override;

 private:
  struct Event {
    qairt::ProfileEventData data;
    std::vector<QnnProfile_EventId_t> children;
    std::string identifier;
    std::string string_value;
    std::string file_name;
    std::vector<unsigned char> opaque_data;
  };
  QnnProfile_EventId_t AddEvent(qairt::ProfileEventData&& event);
  std::shared_ptr<QairtBackendManager> manager_;
  std::shared_ptr<qairt::Profile> profile_;
  std::map<QnnProfile_EventId_t, Event> events_;
  std::vector<QnnProfile_EventId_t> roots_;
};

}  // namespace onnxruntime::qnn
#endif
