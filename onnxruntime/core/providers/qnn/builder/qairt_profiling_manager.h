// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#ifdef USE_QAIRT_API
#include <mutex>
#include "core/providers/qnn/builder/qairt_profile_adapter.h"
#include "core/providers/qnn/builder/qnn_profile_serializer.h"
namespace onnxruntime::qnn {
class QairtProfilingManager final {
 public:
  QairtProfilingManager(std::shared_ptr<QairtBackendManager> backend, ProfilingLevel level,
                        std::string path, bool framework_trace);
  struct Scope {
    ~Scope();
    QairtProfilingManager* owner = nullptr;
    std::unique_lock<std::recursive_mutex> lock;
    profile::ProfilingInfo info;
    std::shared_ptr<qairt::Profile> profile;
  };
  Ort::Status Begin(Scope& scope, const std::string& graph, OrtProfilingOperation operation,
                    QnnEpProfiler* profiler);
  Ort::Status Complete(Scope& scope);
  Ort::Status Acquire(const Ort::Logger&);
  Ort::Status Release();
  std::unique_lock<std::recursive_mutex> Lock() { return std::unique_lock<std::recursive_mutex>(mutex_); }
  void MergeOpTraceLookup(OpTraceLookup&& lookup);

 private:
  Ort::Status AppendEvent(QnnProfile_EventId_t id, const char* level, uint64_t root_timestamp,
                          profile::Serializer* writer, const profile::ProfilingInfo& info);
  std::recursive_mutex mutex_;
  QairtProfileAdapter adapter_;
  ProfilingLevel level_;
  std::string path_;
  bool framework_trace_;
  size_t consumers_ = 0;
  OpTraceLookup trace_lookup_;
};
}  // namespace onnxruntime::qnn
#endif
