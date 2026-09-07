// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_backend_profiling_manager.h"

#include <string>

#include "Saver/QnnSaver.h"

#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/qnn_ep_profiler.h"
#include "core/providers/qnn/qnn_telemetry.h"

namespace onnxruntime {
namespace qnn {

QnnBackendProfilingManager::QnnBackendProfilingManager(QnnBackendProfilingManagerDependencies dependencies,
                                                       ProfilingLevel provider_profiling_level,
                                                       ProfilingLevel etw_profiling_level,
                                                       std::string profiling_file_path,
                                                       bool enable_framework_op_trace)
    : qnn_interface_(dependencies.qnn_interface),
      backend_handle_(dependencies.backend_handle),
      qnn_system_interface_(dependencies.qnn_system_interface),
      qnn_serializer_config_(dependencies.qnn_serializer_config),
      logger_(dependencies.logger),
      backend_setup_completed_(dependencies.backend_setup_completed),
      load_qnn_system_lib_(std::move(dependencies.load_qnn_system_lib)),
      etw_profiling_level_(etw_profiling_level),
      provider_profiling_level_(provider_profiling_level),
      profiling_file_path_(std::move(profiling_file_path)),
      enable_framework_op_trace_(enable_framework_op_trace) {}

Ort::Status QnnBackendProfilingManager::InitializeProfiling(ProfilingLevel override_level) {
  std::lock_guard<std::recursive_mutex> lock(profile_handle_mutex_);
  // Idempotent: a handle already exists, so don't leak it with a second profileCreate.
  if (profile_handle_ != nullptr) {
    return Ort::Status();
  }

  // Keep the provider-selected level immutable so ETW escalation remains well-defined.
  const ProfilingLevel effective_level =
      override_level != ProfilingLevel::INVALID ? override_level : provider_profiling_level_;
  merged_profiling_level_ = effective_level;
  // Only honor ETW-driven profile escalation when the app has explicitly opted into profiling.
  // If the app did not set profiling_level (default OFF), do not let ETW silently turn it on.
  if (effective_level != ProfilingLevel::OFF &&
      etw_profiling_level_ != ProfilingLevel::INVALID &&
      etw_profiling_level_ > effective_level) {
    merged_profiling_level_ = etw_profiling_level_;
  }

  if (merged_profiling_level_ == ProfilingLevel::OFF ||
      merged_profiling_level_ == ProfilingLevel::INVALID) {
    ORT_CXX_LOG_PTR(logger_, ORT_LOGGING_LEVEL_INFO, "Profiling turned off.");
    return Ort::Status();
  }

  QnnProfile_Level_t qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  bool enable_optrace = false;
  if (merged_profiling_level_ == ProfilingLevel::BASIC) {
    ORT_CXX_LOG_PTR(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Profiling level set to basic.");
  } else if (merged_profiling_level_ == ProfilingLevel::DETAILED) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
    ORT_CXX_LOG_PTR(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Profiling level set to detailed.");
  } else if (merged_profiling_level_ == ProfilingLevel::OPTRACE) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
    enable_optrace = true;
    ORT_CXX_LOG_PTR(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Profiling level set to optrace.");
  }

  Qnn_ErrorHandle_t result = qnn_interface_.profileCreate(backend_handle_, qnn_profile_level, &profile_handle_);
  RETURN_IF(QNN_PROFILE_NO_ERROR != result,
            ("Failed to create QNN profile! Error: " + utils::GetQnnErrorMessage(qnn_interface_, result)).c_str());
  profiling_enabled_.store(true, std::memory_order_release);

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  RETURN_IF_ERROR(load_qnn_system_lib_());
  if (enable_optrace) {
    QnnProfile_Config_t optrace_config = QNN_PROFILE_CONFIG_INIT;
    optrace_config.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
    optrace_config.enableOptrace = true;
    const QnnProfile_Config_t* profile_configs[] = {&optrace_config, nullptr};
    result = qnn_interface_.profileSetConfig(profile_handle_, profile_configs);
    RETURN_IF(QNN_PROFILE_NO_ERROR != result,
              ("Failed to enable op trace! Error: " + utils::GetQnnErrorMessage(qnn_interface_, result)).c_str());
  }
#else
  if (enable_optrace) {
    ORT_CXX_LOG_PTR(logger_,
                    ORT_LOGGING_LEVEL_WARNING,
                    "Profiling level set to optrace, but QNN SDK Version is older than 2.29.0. "
                    "Profiling level will be set to detailed instead.");
  }
#endif

  return Ort::Status();
}

Ort::Status QnnBackendProfilingManager::InitializeProfilingForCurrentConsumers() {
  const ProfilingLevel override_level =
      HasActiveOrtProfilingConsumer() && !ProviderProfilingActive() ? ProfilingLevel::BASIC : ProfilingLevel::INVALID;
  return InitializeProfiling(override_level);
}

bool QnnBackendProfilingManager::ProviderProfilingOutputActive() const {
  if (!profiling_file_path_.empty()) {
    return true;
  }

#ifdef _WIN32
  const auto& provider = QnnTelemetry::Instance();
  return provider.IsEnabled() &&
         (provider.Keyword() & static_cast<uint64_t>(ORTTraceLoggingKeyword::Profiling)) != 0 &&
         provider.Level() >= 5;
#else
  return false;
#endif
}

Ort::Status QnnBackendProfilingManager::GetProfileEventsLocked(
    const QnnProfile_EventId_t*& profile_events, uint32_t& num_events) {
  profile_events = nullptr;
  num_events = 0;
  const Qnn_ErrorHandle_t result = qnn_interface_.profileGetEvents(profile_handle_, &profile_events, &num_events);
  if (qnn_serializer_config_) {  // Using QNN Saver or IR backend
    // QNN SDK 2.28.2 returns QNN_SAVER_ERROR_DUMMY_RETVALUE, but previous QNN versions return QNN_PROFILE_NO_ERROR.
    // We accept both values.
    RETURN_IF(QNN_PROFILE_NO_ERROR != result && QNN_SAVER_ERROR_DUMMY_RETVALUE != result,
              ("Failed to get profile events. Error: " + utils::GetQnnErrorMessage(qnn_interface_, result)).c_str());
  } else {
    RETURN_IF(QNN_PROFILE_NO_ERROR != result,
              ("Failed to get profile events. Error: " + utils::GetQnnErrorMessage(qnn_interface_, result)).c_str());
  }
  return Ort::Status();
}

Ort::Status QnnBackendProfilingManager::ReleaseProfileHandle() {
  std::lock_guard<std::recursive_mutex> lock(profile_handle_mutex_);
  // Free the profiling object if it was created.
  if (profile_handle_ != nullptr) {
    RETURN_IF(QNN_PROFILE_NO_ERROR != qnn_interface_.profileFree(profile_handle_),
              "Could not free backend profile handle!");
  }
  profile_handle_ = nullptr;
  merged_profiling_level_ = ProfilingLevel::OFF;
  profiling_enabled_.store(false, std::memory_order_release);
  return Ort::Status();
}

void QnnBackendProfilingManager::AcquireOrtProfilingConsumer() noexcept {
  active_ort_profiler_count_.fetch_add(1, std::memory_order_relaxed);
}

void QnnBackendProfilingManager::ReleaseOrtProfilingConsumer() noexcept {
  uint32_t current = active_ort_profiler_count_.load(std::memory_order_relaxed);
  while (current != 0 &&
         !active_ort_profiler_count_.compare_exchange_weak(current, current - 1,
                                                           std::memory_order_relaxed,
                                                           std::memory_order_relaxed)) {
  }
}

Ort::Status QnnBackendProfilingManager::ReleaseOrtProfilingHandleIfUnused() {
  if (ProviderProfilingActive() || HasActiveOrtProfilingConsumer()) {
    return Ort::Status();
  }
  return ReleaseProfileHandle();
}

QnnProfilingScope QnnBackendProfilingManager::AcquireProfilingScope(
    bool current_operation_has_ort_profiler) {
  if (!ProviderProfilingActive() && !current_operation_has_ort_profiler) {
    return QnnProfilingScope{};
  }
  if (!current_operation_has_ort_profiler && HasActiveOrtProfilingConsumer() &&
      !ProviderProfilingOutputActive()) {
    // A setup worker without the thread-local ORT scope must not add events to the shared
    // handle unless CSV or ETW consumes them; otherwise they could be attributed to a later
    // ORT operation.
    return QnnProfilingScope{};
  }

  std::unique_lock<std::recursive_mutex> lock(profile_handle_mutex_);
  return QnnProfilingScope(std::move(lock), profile_handle_);
}

Ort::Status QnnBackendProfilingManager::CreateGraphProfilingScope(
    bool current_run_has_ort_profiler, QnnProfilingScope& profiling_scope) {
  profiling_scope = QnnProfilingScope{};
  if (!ProviderProfilingActive() && !current_run_has_ort_profiler) {
    return Ort::Status();
  }
  if (!current_run_has_ort_profiler && HasActiveOrtProfilingConsumer() &&
      !ProviderProfilingOutputActive()) {
    // Parallel/background work remains provider-output only because it has no ORT scope.
    return Ort::Status();
  }

  std::unique_lock<std::recursive_mutex> lock(profile_handle_mutex_);
  if (current_run_has_ort_profiler && profile_handle_ == nullptr) {
    RETURN_IF_ERROR(InitializeProfilingForCurrentConsumers());
  }
  if (profile_handle_ == nullptr) {
    return Ort::Status();
  }
  profiling_scope = QnnProfilingScope(std::move(lock), profile_handle_);
  return Ort::Status();
}

Ort::Status QnnBackendProfilingManager::SetProfilingLevelETW(ProfilingLevel profiling_level_etw) {
  std::lock_guard<std::recursive_mutex> lock(profile_handle_mutex_);
  if (etw_profiling_level_ == profiling_level_etw) {
    return Ort::Status();
  }

  etw_profiling_level_ = profiling_level_etw;
  Ort::Status status = ReleaseProfileHandle();
  if (!status.IsOK()) {
    ORT_CXX_API_THROW("Failed to ReleaseProfilehandle for previous QNN profiling", ORT_EP_FAIL);
  }

  status = InitializeProfiling();
  if (!status.IsOK()) {
    ORT_CXX_API_THROW("Failed to Re-InitializeProfiling for QNN ETW profiling", ORT_EP_FAIL);
  }
  return Ort::Status();
}

Ort::Status QnnBackendProfilingManager::ExtractBackendProfilingInfo(profile::ProfilingInfo& profiling_info) {
  std::lock_guard<std::recursive_mutex> lock(profile_handle_mutex_);
  if (merged_profiling_level_ == ProfilingLevel::OFF ||
      merged_profiling_level_ == ProfilingLevel::INVALID) {
    return Ort::Status();
  }

  bool tracelogging_provider_ep_enabled = false;
#ifdef _WIN32
  auto& provider = QnnTelemetry::Instance();
  if (provider.IsEnabled()) {
    tracelogging_provider_ep_enabled =
        (provider.Keyword() & static_cast<uint64_t>(ORTTraceLoggingKeyword::Profiling)) != 0 &&
        provider.Level() >= 5;
  }
#endif

  // ETW disabled previously, but enabled now.
  if (etw_profiling_level_ == ProfilingLevel::INVALID && tracelogging_provider_ep_enabled) {
    ORT_CXX_LOG_PTR(logger_,
                    ORT_LOGGING_LEVEL_ERROR,
                    "ETW disabled previously, but enabled now. Can't do the switch! Won't output any profiling.");
    return Ort::Status();
  }
  // ETW enabled previously, but disabled now.
  if (etw_profiling_level_ != ProfilingLevel::INVALID && !tracelogging_provider_ep_enabled) {
    ORT_CXX_LOG_PTR(logger_,
                    ORT_LOGGING_LEVEL_ERROR,
                    "ETW enabled previously, but disabled now. Can't do the switch! Won't output any profiling.");
    return Ort::Status();
  }

  const bool has_current_ort_profiler = profiling_info.ort_profiler != nullptr;
  const bool emit_provider_profile = !profiling_file_path_.empty() || tracelogging_provider_ep_enabled;
  RETURN_IF(!emit_provider_profile && !has_current_ort_profiler,
            "Need to specify a CSV file via provider option profiling_file_path if ETW not enabled.");
  ORT_CXX_LOG_PTR(logger_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("Extracting profiling events for graph " + profiling_info.graph_name).c_str());
  RETURN_IF(profile_handle_ == nullptr, "Backend profile handle not valid.");

  const QnnProfile_EventId_t* profile_events = nullptr;
  uint32_t num_events = 0;
  RETURN_IF_ERROR(GetProfileEventsLocked(profile_events, num_events));

  const Qnn_ErrorHandle_t capability = qnn_interface_.propertyHasCapability(
      QNN_PROPERTY_PROFILE_SUPPORTS_EXTENDED_EVENT);
  const bool backend_supports_extended_event_data =
      static_cast<uint16_t>(capability & 0xFFFF) == QNN_PROPERTY_SUPPORTED;
  ORT_CXX_LOG_PTR(logger_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  backend_supports_extended_event_data ? "The QNN backend supports extended event data."
                                                       : "The QNN backend does not support extended event data.");

  if (num_events > 0) {
    ORT_CXX_LOG_PTR(logger_,
                    ORT_LOGGING_LEVEL_VERBOSE,
                    ("profile_events: " + std::to_string(*profile_events) +
                     " num_events: " + std::to_string(num_events))
                        .c_str());
  }

  if (emit_provider_profile && num_events > 0) {
    profiling_info.csv_output_filepath = profiling_file_path_;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    profiling_info.num_events = num_events;
#endif
    // When framework op tracing is enabled, attach the lookup so InitCsvFile()
    // emits the `ONNX Source Ops` column header and ProcessEvent() annotates each
    // NODE row with the originating ONNX op names. The lookup is read by pointer
    // and continues to fill as later graphs compose; only DETAILED/OPTRACE
    // profiling produces the per-NODE events this column annotates.
    const bool has_node_level_profiling = merged_profiling_level_ == ProfilingLevel::DETAILED ||
                                          merged_profiling_level_ == ProfilingLevel::OPTRACE;
    if (enable_framework_op_trace_ && has_node_level_profiling) {
      profiling_info.op_trace_lookup = &op_trace_lookup_;
    }

    profile::Serializer profile_writer(profiling_info,
                                       qnn_system_interface_,
                                       tracelogging_provider_ep_enabled);
    if (!profiling_file_path_.empty()) {
      RETURN_IF_ERROR(profile_writer.InitCsvFile());
    }
    for (size_t event_idx = 0; event_idx < num_events; ++event_idx) {
      RETURN_IF_ERROR(ExtractProfilingEvent(profile_events[event_idx],
                                            "ROOT",
                                            profile_writer,
                                            backend_supports_extended_event_data));
      RETURN_IF_ERROR(ExtractProfilingSubEvents(profile_events[event_idx],
                                                profile_writer,
                                                backend_supports_extended_event_data));
    }
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (!profiling_file_path_.empty()) {
      RETURN_IF_ERROR(profile_writer.SerializeEventsToQnnLog());
    }
#endif
    if (!profiling_file_path_.empty()) {
      ORT_CXX_LOG_PTR(logger_,
                      ORT_LOGGING_LEVEL_VERBOSE,
                      ("Wrote QNN profiling events (" + std::to_string(num_events) +
                       ") to file (" + profiling_file_path_ + ")")
                          .c_str());
    }
    if (tracelogging_provider_ep_enabled) {
      ORT_CXX_LOG_PTR(logger_,
                      ORT_LOGGING_LEVEL_VERBOSE,
                      ("Wrote QNN profiling events (" + std::to_string(num_events) + ") to ETW").c_str());
    }
  }

#if QNN_ORT_EP_PROFILING_API_ENABLED
  if (has_current_ort_profiler) {
    RETURN_IF_ERROR(profiling_info.ort_profiler->AppendNewQnnEventRecords(
        qnn_interface_, profile_events, num_events,
        backend_supports_extended_event_data, profiling_info));
  }
#endif

  // Provider-only profiling keeps its original handle lifetime. ORT profiling needs a fresh
  // handle after extraction so a later ORT scope cannot replay already-emitted events.
  if (HasActiveOrtProfilingConsumer()) {
    RETURN_IF_ERROR(ReleaseProfileHandle());
    RETURN_IF_ERROR(InitializeProfilingForCurrentConsumers());
  }
  return Ort::Status();
}

Ort::Status QnnBackendProfilingManager::ExtractProfilingSubEvents(
    QnnProfile_EventId_t profile_event_id,
    profile::Serializer& profile_writer,
    bool use_extended_event_data) {
  const QnnProfile_EventId_t* profile_sub_events = nullptr;
  uint32_t num_sub_events = 0;
  const Qnn_ErrorHandle_t result = qnn_interface_.profileGetSubEvents(profile_event_id, &profile_sub_events, &num_sub_events);
  const auto error_code = static_cast<QnnProfile_Error_t>(result & 0xFFFF);
  RETURN_IF(QNN_PROFILE_NO_ERROR != error_code,
            ("Failed to get profile sub events. Error: " + std::string(QnnProfileErrorToString(error_code))).c_str());

  if (num_sub_events == 0) {
    return Ort::Status();
  }

  ORT_CXX_LOG_PTR(logger_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("profile_sub_events: " + std::to_string(*profile_sub_events) +
                   " num_sub_events: " + std::to_string(num_sub_events))
                      .c_str());

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  QnnSystemProfile_ProfileEventV1_t* parent_system_event = profile_writer.GetParentSystemEvent(profile_event_id);
  if (parent_system_event == nullptr) {
    parent_system_event = profile_writer.GetSystemEventPointer(profile_event_id);
    profile_writer.AddSubEventList(num_sub_events, parent_system_event);
  }
#endif

  for (size_t sub_event_idx = 0; sub_event_idx < num_sub_events; ++sub_event_idx) {
    const QnnProfile_EventId_t subevent_id = profile_sub_events[sub_event_idx];
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    RETURN_IF_ERROR(profile_writer.SetParentSystemEvent(subevent_id, parent_system_event));
#endif
    RETURN_IF_ERROR(ExtractProfilingEvent(subevent_id, "SUB-EVENT", profile_writer, use_extended_event_data));
    RETURN_IF_ERROR(ExtractProfilingSubEvents(subevent_id, profile_writer, use_extended_event_data));
  }
  ORT_CXX_LOG_PTR(logger_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("Wrote QNN profiling sub events (" + std::to_string(num_sub_events) + ")").c_str());
  return Ort::Status();
}

Ort::Status QnnBackendProfilingManager::ExtractProfilingEvent(
    QnnProfile_EventId_t profile_event_id,
    const std::string& event_level,
    profile::Serializer& profile_writer,
    bool use_extended_event_data) {
  return use_extended_event_data
             ? ExtractProfilingEventExtended(profile_event_id, event_level, profile_writer)
             : ExtractProfilingEventBasic(profile_event_id, event_level, profile_writer);
}

Ort::Status QnnBackendProfilingManager::ExtractProfilingEventBasic(
    QnnProfile_EventId_t profile_event_id,
    const std::string& event_level,
    profile::Serializer& profile_writer) {
  QnnProfile_EventData_t event_data = QNN_PROFILE_EVENT_DATA_INIT;
  const auto result = qnn_interface_.profileGetEventData(profile_event_id, &event_data);
  const auto error_code = static_cast<QnnProfile_Error_t>(result & 0xFFFF);
  RETURN_IF(QNN_PROFILE_NO_ERROR != error_code,
            ("Failed to get profile event data: " + std::string(QnnProfileErrorToString(error_code))).c_str());
  return profile_writer.ProcessEvent(profile_event_id, event_level, event_data);
}

Ort::Status QnnBackendProfilingManager::ExtractProfilingEventExtended(
    QnnProfile_EventId_t profile_event_id,
    const std::string& event_level,
    profile::Serializer& profile_writer) {
  QnnProfile_ExtendedEventData_t event_data = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
  const auto result = qnn_interface_.profileGetExtendedEventData(profile_event_id, &event_data);
  const auto error_code = static_cast<QnnProfile_Error_t>(result & 0xFFFF);
  RETURN_IF(QNN_PROFILE_NO_ERROR != error_code,
            ("Failed to get profile event data: " + std::string(QnnProfileErrorToString(error_code))).c_str());
  return profile_writer.ProcessExtendedEvent(profile_event_id, event_level, event_data);
}

const char* QnnBackendProfilingManager::QnnProfileErrorToString(QnnProfile_Error_t error) {
  switch (error) {
    case QNN_PROFILE_NO_ERROR:
      return "QNN_PROFILE_NO_ERROR";
    case QNN_PROFILE_ERROR_UNSUPPORTED:
      return "QNN_PROFILE_ERROR_UNSUPPORTED";
    case QNN_PROFILE_ERROR_INVALID_ARGUMENT:
      return "QNN_PROFILE_ERROR_INVALID_ARGUMENT";
    case QNN_PROFILE_ERROR_MEM_ALLOC:
      return "QNN_PROFILE_ERROR_MEM_ALLOC";
    case QNN_PROFILE_ERROR_INVALID_HANDLE:
      return "QNN_PROFILE_ERROR_INVALID_HANDLE";
    case QNN_PROFILE_ERROR_HANDLE_IN_USE:
      return "QNN_PROFILE_ERROR_HANDLE_IN_USE";
    case QNN_PROFILE_ERROR_INCOMPATIBLE_EVENT:
      return "QNN_PROFILE_ERROR_INCOMPATIBLE_EVENT";
    default:
      return "UNKNOWN_ERROR";
  }
}

}  // namespace qnn
}  // namespace onnxruntime
