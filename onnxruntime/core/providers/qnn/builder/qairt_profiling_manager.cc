// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#ifdef USE_QAIRT_API
#include "core/providers/qnn/builder/qairt_profiling_manager.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/qnn_ep_profiler.h"

namespace onnxruntime::qnn {
QairtProfilingManager::Scope::~Scope() {
  if (owner) owner->adapter_.Release();
}
QairtProfilingManager::QairtProfilingManager(std::shared_ptr<QairtBackendManager> backend,
                                             ProfilingLevel level, std::string path, bool framework_trace)
    : adapter_(std::move(backend)), level_(level), path_(std::move(path)), framework_trace_(framework_trace) {}
Ort::Status QairtProfilingManager::Acquire(const Ort::Logger&) {
  auto lock = Lock();
  ++consumers_;
  return Ort::Status();
}
Ort::Status QairtProfilingManager::Release() {
  auto lock = Lock();
  if (consumers_) --consumers_;
  return Ort::Status();
}
void QairtProfilingManager::MergeOpTraceLookup(OpTraceLookup&& lookup) {
  auto lock = Lock();
  for (auto& entry : lookup) trace_lookup_[entry.first] = std::move(entry.second);
}
Ort::Status QairtProfilingManager::Begin(Scope& scope, const std::string& graph,
                                         OrtProfilingOperation operation, QnnEpProfiler* profiler) {
  scope.lock = Lock();
  const bool provider = level_ != ProfilingLevel::OFF && level_ != ProfilingLevel::INVALID;
  if (!provider && !profiler) return Ort::Status();
  scope.info.graph_name = graph;
  scope.info.ort_profiler = profiler;
  scope.info.ort_profiling_operation = operation;
  scope.info.operation_start_time_us = utils::GetTimeStampInUs();
  RETURN_IF_ERROR(adapter_.Create(provider ? level_ : ProfilingLevel::DETAILED));
  scope.profile = adapter_.GetProfile();
  scope.owner = this;
  return Ort::Status();
}
Ort::Status QairtProfilingManager::AppendEvent(QnnProfile_EventId_t id, const char* level,
                                               uint64_t root_timestamp, profile::Serializer* writer, const profile::ProfilingInfo& info) {
  QnnProfile_ExtendedEventData_t event = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
  RETURN_IF(adapter_.GetExtendedEventData(id, &event) != QNN_PROFILE_NO_ERROR,
            "Unable to read QAIRT profiling event.");
  if (writer) RETURN_IF_ERROR(writer->ProcessExtendedEvent(id, level, event));
#if QNN_ORT_EP_PROFILING_API_ENABLED
  if (info.ort_profiler && event.v1.unit != QNN_PROFILE_EVENTUNIT_OBJECT) {
    EventRecord record;
    record.name = event.v1.identifier ? event.v1.identifier : "(unknown)";
    record.args.push_back({"qnn_event_type", profile::GetEventTypeString(event.v1.type)});
    record.args.push_back({"qnn_event_identifier", record.name});
    record.args.push_back({"qnn_timing_source", "BACKEND"});
    record.args.push_back({"qnn_graph_name", info.graph_name});
    record.qairt_ts_us = event.v1.timestamp;
    if (event.v1.unit == QNN_PROFILE_EVENTUNIT_MICROSEC &&
        event.v1.value.dataType == QNN_DATATYPE_UINT_64) {
      record.qairt_duration_us = static_cast<int64_t>(event.v1.value.uint64Value);
    } else {
      record.args.push_back({"value", profile::ExtractQnnScalarValue(event.v1.value)});
    }
    record.args.push_back({"unit", profile::GetUnitString(event.v1.unit)});
    record.args.push_back({"level", level});
    info.ort_profiler->AppendQnnEventRecord(std::move(record), root_timestamp, info);
  }
#endif
  const QnnProfile_EventId_t* children = nullptr;
  uint32_t count = 0;
  RETURN_IF(adapter_.GetSubEvents(id, &children, &count) != QNN_PROFILE_NO_ERROR,
            "Unable to read QAIRT profiling subevents.");
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  QnnSystemProfile_ProfileEventV1_t* parent = nullptr;
  if (writer && count) {
    parent = writer->GetParentSystemEvent(id);
    if (!parent) {
      parent = writer->GetSystemEventPointer(id);
      writer->AddSubEventList(count, parent);
    }
  }
#endif
  for (uint32_t i = 0; i < count; ++i) {
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (writer) RETURN_IF_ERROR(writer->SetParentSystemEvent(children[i], parent));
#endif
    RETURN_IF_ERROR(AppendEvent(children[i], "SUB-EVENT", root_timestamp, writer, info));
  }
  return Ort::Status();
}
Ort::Status QairtProfilingManager::Complete(Scope& scope) {
  if (scope.info.graph_name.empty()) return Ort::Status();
  scope.info.operation_end_time_us = utils::GetTimeStampInUs();
  auto release = gsl::finally([&] { scope.profile.reset(); adapter_.Release(); scope.owner = nullptr; });
  const QnnProfile_EventId_t* roots = nullptr;
  uint32_t count = 0;
  RETURN_IF(adapter_.GetEvents(&roots, &count) != QNN_PROFILE_NO_ERROR, "Unable to collect QAIRT profiling events.");
  std::unique_ptr<profile::Serializer> writer;
  if (!path_.empty() && count) {
    scope.info.csv_output_filepath = path_;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    scope.info.num_events = count;
#endif
    if (framework_trace_ && level_ != ProfilingLevel::BASIC) scope.info.op_trace_lookup = &trace_lookup_;
    // The common CSV formatter consumes event values only. QNN System APIs are never called here.
    writer = std::make_unique<profile::Serializer>(scope.info, QNN_SYSTEM_INTERFACE_VER_TYPE{}, false);
    RETURN_IF_ERROR(writer->InitCsvFile());
  }
  for (uint32_t i = 0; i < count; ++i) {
    QnnProfile_ExtendedEventData_t event = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
    RETURN_IF(adapter_.GetExtendedEventData(roots[i], &event) != QNN_PROFILE_NO_ERROR,
              "Unable to read QAIRT root event.");
    RETURN_IF_ERROR(AppendEvent(roots[i], "ROOT", event.v1.timestamp, writer.get(), scope.info));
  }
#if QNN_ORT_EP_PROFILING_API_ENABLED
  if (!count && scope.info.ort_profiler) {
    scope.info.ort_profiler->AppendHostOperationRecord(scope.info.operation_start_time_us,
                                                       scope.info.operation_end_time_us, scope.info.ort_profiling_operation, scope.info.graph_name);
  }
#endif
  if (!path_.empty() && count) {
    RETURN_IF_ERROR(adapter_.WriteLog(path_, scope.info.graph_name, scope.info.operation_start_time_us,
                                      scope.info.operation_end_time_us, scope.info.ort_profiling_operation));
  }
  return Ort::Status();
}
}  // namespace onnxruntime::qnn
#endif
