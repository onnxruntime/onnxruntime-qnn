// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/qnn_ep_profiler.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "core/providers/qnn/builder/qnn_backend_profiling_manager.h"
#include "core/providers/qnn/builder/qnn_profile_serializer.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#if QNN_ORT_EP_PROFILING_API_ENABLED

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

// Per-thread stack of (profiler, event-record index) pairs pushed by StartEvent. Tagging with the
// profiler lets StopEvent drop entries left over from a prior session on the same compute thread
// (an unmatched StartEvent), which the session-thread StartProfiling cannot reach. The profile
// handle lock is acquired only once execution has started, so setup worker threads are never
// blocked by an enclosing session-initialization event.
struct OrtEventStart {
  const void* profiler;
  size_t index;
  int32_t thread_id;
  std::vector<onnxruntime::qnn::profile::ProfilingInfo> pending_execute_extractions;
  std::unique_lock<std::recursive_mutex> profile_handle_lock;
};
thread_local std::vector<OrtEventStart> tls_ort_event_start_indices;

struct LiveProfilerRegistry {
  std::mutex mutex;
  std::unordered_set<const void*> profilers;
};

LiveProfilerRegistry& GetLiveProfilerRegistry() {
  static LiveProfilerRegistry registry;
  return registry;
}

void RegisterLiveProfiler(const void* profiler) {
  auto& registry = GetLiveProfilerRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.profilers.insert(profiler);
}

void UnregisterLiveProfiler(const void* profiler) {
  auto& registry = GetLiveProfilerRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.profilers.erase(profiler);
}

bool IsLiveProfiler(const void* profiler) {
  auto& registry = GetLiveProfilerRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  return registry.profilers.find(profiler) != registry.profilers.end();
}

int32_t CurrentProcessId() noexcept {
#ifdef _WIN32
  return static_cast<int32_t>(::_getpid());
#else
  return static_cast<int32_t>(::getpid());
#endif
}

bool IsQnnProfileSuccess(Qnn_ErrorHandle_t status) noexcept {
  return static_cast<QnnProfile_Error_t>(status & 0xFFFF) == QNN_PROFILE_NO_ERROR;
}

}  // namespace

namespace onnxruntime {
namespace qnn {

QnnEpProfiler::QnnEpProfiler(const OrtEpApi& ep_api,
                             const OrtApi& ort_api,
                             QnnBackendProfilingManager& profiling_manager)
    : OrtEpProfilerImpl{},
      ep_api_(ep_api),
      ort_api_(ort_api),
      profiling_manager_(profiling_manager),
      ort_profiling_consumer_(profiling_manager) {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  StartProfiling = StartProfilingImpl;
  StartEvent = StartEventImpl;
  StopEvent = StopEventImpl;
  EndProfiling = EndProfilingImpl;
  RegisterLiveProfiler(this);
}

QnnEpProfiler::~QnnEpProfiler() = default;

void QnnEpProfiler::OrtProfilingConsumer::Activate() noexcept {
  if (!active_) {
    profiling_manager_.AcquireOrtProfilingConsumer();
    active_ = true;
  }
}

Ort::Status QnnEpProfiler::OrtProfilingConsumer::Reset() {
  if (!active_) {
    return Ort::Status();
  }

  profiling_manager_.ReleaseOrtProfilingConsumer();
  active_ = false;
  return profiling_manager_.ReleaseOrtProfilingHandleIfUnused();
}

void QnnEpProfiler::AppendEvent(EventRecord record) {
  std::lock_guard<std::mutex> lock(records_mutex_);
  event_records_.push_back(std::move(record));
}

void QnnEpProfiler::QueueExecuteProfilingExtraction(profile::ProfilingInfo profiling_info) {
  if (!tls_ort_event_start_indices.empty() && tls_ort_event_start_indices.back().profiler == this) {
    auto& event_start = tls_ort_event_start_indices.back();
    if (!event_start.profile_handle_lock.owns_lock()) {
      // The caller already holds the same recursive lock through graphExecute. Retain an
      // additional ownership level until StopEvent drains this run's QAIRT events, preventing a
      // concurrent profiled run from appending events to this profiler's extraction.
      event_start.profile_handle_lock = profiling_manager_.AcquireProfileHandleLock();
    }
    event_start.pending_execute_extractions.push_back(std::move(profiling_info));
  }
}

Ort::Status QnnEpProfiler::DrainExecuteProfilingExtractions(std::vector<profile::ProfilingInfo> profiling_infos) {
  for (auto& profiling_info : profiling_infos) {
    RETURN_IF_ERROR(profiling_manager_.ExtractBackendProfilingInfo(profiling_info));
  }
  return Ort::Status();
}

QnnEpProfiler* QnnEpProfiler::Current() noexcept {
  while (!tls_ort_event_start_indices.empty()) {
    const void* current = tls_ort_event_start_indices.back().profiler;
    auto* profiler = const_cast<QnnEpProfiler*>(
        static_cast<const QnnEpProfiler*>(current));
    if (profiler != nullptr && IsLiveProfiler(current)) {
      return profiler;
    }
    tls_ort_event_start_indices.pop_back();
  }
  return nullptr;
}

Ort::Status QnnEpProfiler::AppendNewQnnEventRecords(QNN_INTERFACE_VER_TYPE qnn_interface,
                                                    const QnnProfile_EventId_t* event_ids,
                                                    uint32_t num_events,
                                                    bool use_extended_event_data,
                                                    const profile::ProfilingInfo& profiling_info) {
  bool appended_event = false;
  for (uint32_t event_idx = 0; event_idx < num_events; ++event_idx) {
    const QnnProfile_EventId_t root_event_id = event_ids[event_idx];
    uint64_t root_qairt_timestamp_us = 0;
    if (use_extended_event_data) {
      QnnProfile_ExtendedEventData_t root_event = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
      if (IsQnnProfileSuccess(qnn_interface.profileGetExtendedEventData(root_event_id, &root_event)) &&
          root_event.version == QNN_PROFILE_DATA_VERSION_1) {
        root_qairt_timestamp_us = root_event.v1.timestamp;
      }
    }
    appended_event |= AppendQnnEventRecords(qnn_interface, root_event_id, "ROOT", use_extended_event_data,
                                            root_qairt_timestamp_us, profiling_info.operation_start_time_us,
                                            profiling_info.ort_profiling_operation, profiling_info.graph_name);
  }

  if (!appended_event && !profiling_info.ort_profiling_operation.empty()) {
    AppendHostOperationRecord(profiling_info.operation_start_time_us, profiling_info.operation_end_time_us,
                              profiling_info.ort_profiling_operation, profiling_info.graph_name);
  }

  return Ort::Status();
}

bool QnnEpProfiler::AppendQnnEventRecords(QNN_INTERFACE_VER_TYPE qnn_interface,
                                          QnnProfile_EventId_t event_id,
                                          const std::string& event_level,
                                          bool use_extended_event_data,
                                          uint64_t root_qairt_timestamp_us,
                                          uint64_t operation_start_time_us,
                                          const std::string& operation,
                                          const std::string& graph_name) {
  EventRecord record;
  record.category = OrtProfilingEventCategory_KERNEL;
  record.process_id = CurrentProcessId();
  record.thread_id = qnn::CurrentThreadId();

  bool event_read = false;
  if (use_extended_event_data) {
    QnnProfile_ExtendedEventData_t event_data = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
    if (IsQnnProfileSuccess(qnn_interface.profileGetExtendedEventData(event_id, &event_data)) &&
        event_data.version == QNN_PROFILE_DATA_VERSION_1) {
      event_read = true;
      record.name = event_data.v1.identifier ? event_data.v1.identifier : "(unknown)";
      record.args.push_back({"qnn_event_type", profile::GetEventTypeString(event_data.v1.type)});
      record.args.push_back({"qnn_event_identifier", record.name});
      record.args.push_back({"qnn_timing_source", "BACKEND"});
      record.args.push_back({"qnn_graph_name", graph_name});
      record.qairt_ts_us = event_data.v1.timestamp;
      if (event_data.v1.unit == QNN_PROFILE_EVENTUNIT_MICROSEC) {
        record.qairt_duration_us = static_cast<int64_t>(event_data.v1.value.uint64Value);
      } else {
        record.args.push_back({"value", profile::ExtractQnnScalarValue(event_data.v1.value)});
      }
      record.args.push_back({"unit", profile::GetUnitString(event_data.v1.unit)});
    }
  } else {
    QnnProfile_EventData_t event_data = QNN_PROFILE_EVENT_DATA_INIT;
    if (IsQnnProfileSuccess(qnn_interface.profileGetEventData(event_id, &event_data))) {
      event_read = true;
      record.name = event_data.identifier ? event_data.identifier : "(unknown)";
      record.args.push_back({"qnn_event_type", profile::GetEventTypeString(event_data.type)});
      record.args.push_back({"qnn_event_identifier", record.name});
      record.args.push_back({"qnn_timing_source", "BACKEND"});
      record.args.push_back({"qnn_graph_name", graph_name});
      if (event_data.unit == QNN_PROFILE_EVENTUNIT_MICROSEC) {
        record.qairt_duration_us = static_cast<int64_t>(event_data.value);
      } else {
        record.args.push_back({"value", std::to_string(event_data.value)});
      }
      record.args.push_back({"unit", profile::GetUnitString(event_data.unit)});
    }
  }

  bool appended_event = false;
  if (event_read) {
    uint64_t qairt_offset_us = 0;
    if (record.qairt_ts_us >= root_qairt_timestamp_us && root_qairt_timestamp_us != 0) {
      qairt_offset_us = record.qairt_ts_us - root_qairt_timestamp_us;
    }
    const uint64_t event_time_us = operation_start_time_us + qairt_offset_us;
    const int64_t host_delta_us = event_time_us >= ep_profiling_start_time_us_
                                      ? static_cast<int64_t>(event_time_us - ep_profiling_start_time_us_)
                                      : 0;
    const int64_t timestamp_ns = ep_profiling_start_offset_ns_ + host_delta_us * 1000;
    record.final_ts_us = std::max<int64_t>(timestamp_ns / 1000, 0);
    record.final_duration_us = std::max<int64_t>(record.qairt_duration_us, 0);
    record.args.push_back({"level", event_level});
    if (!operation.empty()) {
      record.args.push_back({"qnn_operation", operation});
    }
    AppendEvent(std::move(record));
    appended_event = true;
  }

  const QnnProfile_EventId_t* sub_events = nullptr;
  uint32_t num_sub_events = 0;
  if (IsQnnProfileSuccess(qnn_interface.profileGetSubEvents(event_id, &sub_events, &num_sub_events))) {
    for (uint32_t i = 0; i < num_sub_events; ++i) {
      appended_event |= AppendQnnEventRecords(qnn_interface, sub_events[i], "SUB-EVENT", use_extended_event_data,
                                              root_qairt_timestamp_us, operation_start_time_us, operation, graph_name);
    }
  }
  return appended_event;
}

void QnnEpProfiler::AppendHostOperationRecord(uint64_t operation_start_time_us,
                                              uint64_t operation_end_time_us,
                                              const std::string& operation,
                                              const std::string& graph_name) {
  EventRecord record;
  record.category = OrtProfilingEventCategory_KERNEL;
  record.process_id = CurrentProcessId();
  record.thread_id = qnn::CurrentThreadId();
  record.name = "QNN " + operation;

  const int64_t host_delta_us = operation_start_time_us >= ep_profiling_start_time_us_
                                    ? static_cast<int64_t>(operation_start_time_us - ep_profiling_start_time_us_)
                                    : 0;
  const int64_t timestamp_ns = ep_profiling_start_offset_ns_ + host_delta_us * 1000;
  record.final_ts_us = std::max<int64_t>(timestamp_ns / 1000, 0);
  record.final_duration_us = operation_end_time_us >= operation_start_time_us
                                 ? static_cast<int64_t>(operation_end_time_us - operation_start_time_us)
                                 : 0;
  record.args = {{"qnn_event_type", "HOST_OPERATION"},
                 {"qnn_event_identifier", record.name},
                 {"qnn_timing_source", "HOST"},
                 {"qnn_graph_name", graph_name},
                 {"unit", "us"},
                 {"level", "ROOT"},
                 {"qnn_operation", operation}};
  AppendEvent(std::move(record));
}

size_t QnnEpProfiler::MarkPendingExecuteProfilingExtractions() {
  if (tls_ort_event_start_indices.empty() || tls_ort_event_start_indices.back().profiler != this) {
    return 0;
  }
  return tls_ort_event_start_indices.back().pending_execute_extractions.size();
}

void QnnEpProfiler::DiscardPendingExecuteProfilingExtractionsSince(size_t mark) {
  if (tls_ort_event_start_indices.empty() || tls_ort_event_start_indices.back().profiler != this) {
    return;
  }

  auto& pending_extractions = tls_ort_event_start_indices.back().pending_execute_extractions;
  if (mark < pending_extractions.size()) {
    pending_extractions.erase(pending_extractions.begin() + static_cast<std::ptrdiff_t>(mark),
                              pending_extractions.end());
  }
}

/*static*/
void ORT_API_CALL QnnEpProfiler::ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept {
  auto* self = static_cast<QnnEpProfiler*>(this_ptr);
  UnregisterLiveProfiler(self);
  delete self;
}

/*static*/
OrtStatus* ORT_API_CALL QnnEpProfiler::StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                          int64_t ep_profiling_start_offset_ns) noexcept {
  auto* self = static_cast<QnnEpProfiler*>(this_ptr);
  self->ort_profiling_consumer_.Activate();
  try {
    self->ep_profiling_start_offset_ns_ = ep_profiling_start_offset_ns;
    self->ep_profiling_start_time_us_ = qnn::utils::GetTimeStampInUs();
    // ORT calls StartProfiling before QNN EP initialization has necessarily reached GetCapability(),
    // where SetupBackend() creates the backend and QAIRT interface. SetupBackend() creates an
    // ORT-only BASIC handle for session-initialization events; if the backend was already set up,
    // create it here instead. Initializing before either point would call profileCreate on a null backend.
    if (self->profiling_manager_.IsBackendSetup() && !self->profiling_manager_.ProfilingEnabled()) {
      Ort::Status status = self->profiling_manager_.InitializeProfilingForCurrentConsumers();
      if (!status.IsOK()) {
        ORT_IGNORE_RETURN_VALUE(self->ort_profiling_consumer_.Reset());
        return status.release();
      }
    }
    return nullptr;
  } catch (const std::exception& e) {
    ORT_IGNORE_RETURN_VALUE(self->ort_profiling_consumer_.Reset());
    return self->ort_api_.CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    ORT_IGNORE_RETURN_VALUE(self->ort_profiling_consumer_.Reset());
    return self->ort_api_.CreateStatus(ORT_FAIL, "QnnEpProfiler::StartProfiling: unknown exception");
  }
}

/*static*/
OrtStatus* ORT_API_CALL QnnEpProfiler::StartEventImpl(OrtEpProfilerImpl* this_ptr,
                                                      uint64_t /*ort_event_correlation_id*/) noexcept {
  auto* self = static_cast<QnnEpProfiler*>(this_ptr);
  try {
    size_t start_index = 0;
    {
      std::lock_guard<std::mutex> lock(self->records_mutex_);
      start_index = self->event_records_.size();
    }

    const int32_t thread_id = CurrentThreadId();
    // Drain any stale entries left by a prior session on this compute thread before pushing,
    // so the start_index reflects only this session's records.
    while (!tls_ort_event_start_indices.empty() &&
           tls_ort_event_start_indices.back().profiler != self) {
      tls_ort_event_start_indices.pop_back();
    }
    tls_ort_event_start_indices.push_back({self, start_index, thread_id, {}, {}});
    return nullptr;
  } catch (const std::exception& e) {
    return self->ort_api_.CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return self->ort_api_.CreateStatus(ORT_FAIL, "QnnEpProfiler::StartEvent: unknown exception");
  }
}

/*static*/
OrtStatus* ORT_API_CALL QnnEpProfiler::StopEventImpl(OrtEpProfilerImpl* this_ptr,
                                                     uint64_t /*ort_event_correlation_id*/,
                                                     const OrtProfilingEvent* ort_event) noexcept {
  auto* self = static_cast<QnnEpProfiler*>(this_ptr);
  try {
    // Drop entries left over from a prior session's profiler on this thread.
    while (!tls_ort_event_start_indices.empty() &&
           tls_ort_event_start_indices.back().profiler != self) {
      tls_ort_event_start_indices.pop_back();
    }
    if (tls_ort_event_start_indices.empty()) {
      return nullptr;
    }

    OrtEventStart event_start = std::move(tls_ort_event_start_indices.back());
    tls_ort_event_start_indices.pop_back();

    RETURN_IF_NOT_OK(self->DrainExecuteProfilingExtractions(std::move(event_start.pending_execute_extractions)));

    const char* ort_name = nullptr;
    int64_t ort_start_us = 0;
    if (OrtStatus* status = self->ep_api_.ProfilingEvent_GetName(ort_event, &ort_name)) {
      return status;
    }
    if (OrtStatus* status = self->ep_api_.ProfilingEvent_GetTimestampUs(ort_event, &ort_start_us)) {
      return status;
    }

    const std::string ort_event_name = ort_name ? ort_name : "";
    const int32_t thread_id = event_start.thread_id;

    std::lock_guard<std::mutex> lock(self->records_mutex_);
    for (size_t i = event_start.index; i < self->event_records_.size(); ++i) {
      EventRecord& record = self->event_records_[i];
      if (record.thread_id != thread_id || record.ort_event_start_us >= 0) {
        continue;
      }

      record.ort_event_start_us = ort_start_us;

      record.args.push_back({"parent_ort_node", ort_event_name});
    }
    return nullptr;
  } catch (const std::exception& e) {
    return self->ort_api_.CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return self->ort_api_.CreateStatus(ORT_FAIL, "QnnEpProfiler::StopEvent: unknown exception");
  }
}

/*static*/
OrtStatus* ORT_API_CALL QnnEpProfiler::EndProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                        OrtProfilingEventsContainer* events_container) noexcept {
  auto* self = static_cast<QnnEpProfiler*>(this_ptr);
  try {
    auto release_ort_only_handle = [self]() -> OrtStatus* {
      RETURN_IF_NOT_OK(self->ort_profiling_consumer_.Reset());
      return nullptr;
    };

    std::vector<EventRecord> records;
    {
      std::lock_guard<std::mutex> lock(self->records_mutex_);
      std::swap(records, self->event_records_);
    }

    std::vector<OrtProfilingEvent*> created_events;
    created_events.reserve(records.size());
    std::vector<const char*> keys;
    std::vector<const char*> values;
    for (const EventRecord& record : records) {
      if (record.ort_event_start_us < 0) {
        continue;
      }

      keys.clear();
      values.clear();
      for (const auto& arg : record.args) {
        keys.push_back(arg.first.c_str());
        values.push_back(arg.second.c_str());
      }

      OrtProfilingEvent* event = nullptr;
      OrtStatus* status = self->ep_api_.CreateProfilingEvent(
          record.category,
          record.process_id,
          record.thread_id,
          record.name.c_str(),
          record.final_ts_us,
          record.final_duration_us,
          keys.empty() ? nullptr : keys.data(),
          values.empty() ? nullptr : values.data(),
          keys.size(),
          &event);
      if (status != nullptr) {
        for (OrtProfilingEvent* created_event : created_events) {
          self->ep_api_.ReleaseProfilingEvent(created_event);
        }
        OrtStatus* release_status = release_ort_only_handle();
        if (release_status != nullptr) {
          self->ort_api_.ReleaseStatus(release_status);
        }
        return status;
      }
      created_events.push_back(event);
    }

    if (created_events.empty()) {
      return release_ort_only_handle();
    }

    OrtStatus* status = self->ep_api_.ProfilingEventsContainer_AddEvents(
        events_container,
        reinterpret_cast<const OrtProfilingEvent* const*>(created_events.data()),
        created_events.size());
    for (OrtProfilingEvent* event : created_events) {
      self->ep_api_.ReleaseProfilingEvent(event);
    }
    if (status != nullptr) {
      OrtStatus* release_status = release_ort_only_handle();
      if (release_status != nullptr) {
        self->ort_api_.ReleaseStatus(release_status);
      }
      return status;
    }
    return release_ort_only_handle();
  } catch (const std::exception& e) {
    return self->ort_api_.CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return self->ort_api_.CreateStatus(ORT_FAIL, "QnnEpProfiler::EndProfiling: unknown exception");
  }
}

}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_ORT_EP_PROFILING_API_ENABLED
