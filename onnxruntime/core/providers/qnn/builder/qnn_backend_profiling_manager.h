// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <utility>

#include <QnnInterface.h>
#include <QnnProfile.h>
#include <System/QnnSystemInterface.h>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_profile_serializer.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnSerializerConfig;

// Holds the manager-global QAIRT profiling lock across one profiled operation.
class QnnProfilingScope {
 public:
  QnnProfilingScope() = default;
  QnnProfilingScope(std::unique_lock<std::recursive_mutex>&& lock,
                    Qnn_ProfileHandle_t profile_handle)
      : lock_(std::move(lock)),
        profile_handle_(profile_handle) {}

  QnnProfilingScope(const QnnProfilingScope&) = delete;
  QnnProfilingScope& operator=(const QnnProfilingScope&) = delete;
  QnnProfilingScope(QnnProfilingScope&&) noexcept = default;
  QnnProfilingScope& operator=(QnnProfilingScope&&) noexcept = default;

  bool Active() const { return profile_handle_ != nullptr; }
  Qnn_ProfileHandle_t Handle() const { return profile_handle_; }

 private:
  std::unique_lock<std::recursive_mutex> lock_;
  Qnn_ProfileHandle_t profile_handle_ = nullptr;
};

struct QnnBackendProfilingManagerDependencies {
  QNN_INTERFACE_VER_TYPE& qnn_interface;
  Qnn_BackendHandle_t& backend_handle;
  QNN_SYSTEM_INTERFACE_VER_TYPE& qnn_system_interface;
  const QnnSerializerConfig* qnn_serializer_config;
  const Ort::Logger* const& logger;
  const bool& backend_setup_completed;
  std::function<Ort::Status()> load_qnn_system_lib;
};

// Owns the QAIRT profiling handle and routes one extraction pass to provider and ORT consumers.
class QnnBackendProfilingManager {
 public:
  QnnBackendProfilingManager(QnnBackendProfilingManagerDependencies dependencies,
                             ProfilingLevel provider_profiling_level,
                             ProfilingLevel etw_profiling_level,
                             std::string profiling_file_path,
                             bool enable_framework_op_trace);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnBackendProfilingManager);

  bool HasProfileHandle() const { return profile_handle_ != nullptr; }
  bool IsBackendSetup() const { return backend_setup_completed_; }
  // True when a QAIRT profile handle is currently allocated.
  bool ProfilingEnabled() const { return profiling_enabled_.load(std::memory_order_acquire); }
  // True when provider options explicitly request QNN profiling collection.
  bool ProviderProfilingActive() const {
    return provider_profiling_level_ != ProfilingLevel::OFF &&
           provider_profiling_level_ != ProfilingLevel::INVALID;
  }
  // True when provider-collected profiling has an output sink (CSV file or ETW).
  bool ProviderProfilingOutputActive() const;

  // Creates the shared QAIRT profile handle. override_level is used only for ORT-only profiling,
  // where no provider profiling level was requested.
  Ort::Status InitializeProfiling(ProfilingLevel override_level = ProfilingLevel::INVALID);
  Ort::Status InitializeProfilingForCurrentConsumers();
  Ort::Status ReleaseProfileHandle();

  void AcquireOrtProfilingConsumer() noexcept;
  void ReleaseOrtProfilingConsumer() noexcept;
  bool HasActiveOrtProfilingConsumer() const {
    return active_ort_profiler_count_.load(std::memory_order_relaxed) != 0;
  }
  bool OrtProfilingActive() const { return HasActiveOrtProfilingConsumer(); }
  Ort::Status ReleaseOrtProfilingHandleIfUnused();

  // ORT profiling is correlated through a thread-local profiler scope. Work without that scope
  // must not be appended to ORT; parallel finalization is intentionally provider-output only.
  QnnProfilingScope AcquireProfilingScope(bool current_operation_has_ort_profiler = false);
  std::unique_lock<std::recursive_mutex> AcquireProfileHandleLock() {
    return std::unique_lock<std::recursive_mutex>(profile_handle_mutex_);
  }
  Ort::Status CreateGraphProfilingScope(bool current_run_has_ort_profiler,
                                        QnnProfilingScope& profiling_scope);

  Ort::Status SetProfilingLevelETW(ProfilingLevel profiling_level_etw);
  Ort::Status ExtractBackendProfilingInfo(profile::ProfilingInfo& profiling_info);

  // Framework op tracing: profiling enrichment lookup, shared across all
  // QnnModels in this session. Populated by:
  //   - JIT path:  ComposeGraph() merges each per-graph lookup via
  //                MergeOpTraceLookup() after the trace collector finalizes.
  //   - AOT path:  CompileContextModel() loads the sidecar JSON via
  //                SetOpTraceLookup() before any context binary is loaded.
  // Self-attached to the local ProfilingInfo inside ExtractBackendProfilingInfo
  // when profiling is at DETAILED/OPTRACE level (per-NODE events) and op
  // tracing is enabled.
  void SetOpTraceLookup(OpTraceLookup&& lookup) { op_trace_lookup_ = std::move(lookup); }
  // Merges `other` into the session-wide lookup. On key collision the entry
  // from `other` wins (last-write-wins), matching the operator[] semantics
  // already used by OpTraceCollector::Finalize and LoadTraceLookupFromFile
  // when they populate a lookup. `other` is consumed.
  void MergeOpTraceLookup(OpTraceLookup&& other) {
    for (auto& kv : other) {
      op_trace_lookup_[kv.first] = std::move(kv.second);
    }
  }

 private:
  Ort::Status GetProfileEventsLocked(const QnnProfile_EventId_t*& profile_events, uint32_t& num_events);
  Ort::Status ExtractProfilingSubEvents(QnnProfile_EventId_t profile_event_id,
                                        profile::Serializer& profile_writer,
                                        bool use_extended_event_data);
  Ort::Status ExtractProfilingEvent(QnnProfile_EventId_t profile_event_id,
                                    const std::string& event_level,
                                    profile::Serializer& profile_writer,
                                    bool use_extended_event_data);
  Ort::Status ExtractProfilingEventBasic(QnnProfile_EventId_t profile_event_id,
                                         const std::string& event_level,
                                         profile::Serializer& profile_writer);
  Ort::Status ExtractProfilingEventExtended(QnnProfile_EventId_t profile_event_id,
                                            const std::string& event_level,
                                            profile::Serializer& profile_writer);
  static const char* QnnProfileErrorToString(QnnProfile_Error_t error);

  QNN_INTERFACE_VER_TYPE& qnn_interface_;
  Qnn_BackendHandle_t& backend_handle_;
  QNN_SYSTEM_INTERFACE_VER_TYPE& qnn_system_interface_;
  const QnnSerializerConfig* qnn_serializer_config_;
  const Ort::Logger* const& logger_;
  const bool& backend_setup_completed_;
  std::function<Ort::Status()> load_qnn_system_lib_;
  ProfilingLevel etw_profiling_level_;
  const ProfilingLevel provider_profiling_level_;
  ProfilingLevel merged_profiling_level_ = ProfilingLevel::OFF;
  const std::string profiling_file_path_;
  // ----------------------------------------------------------------------
  // Framework op tracing (profiling CSV enrichment).
  //
  // Session-scoped state used to annotate the profiling CSV's `ONNX Source Ops`
  // column. Read by ExtractBackendProfilingInfo() via &op_trace_lookup_.
  //   - enable_framework_op_trace_: fixed at construction so the CSV header
  //     and per-NODE rows agree across every graph's events.
  //   - op_trace_lookup_: populated by SetOpTraceLookup (AOT sidecar) /
  //     MergeOpTraceLookup (JIT, per-graph). See those accessor comments.
  // ----------------------------------------------------------------------
  const bool enable_framework_op_trace_;
  OpTraceLookup op_trace_lookup_;

  std::atomic<bool> profiling_enabled_{false};
  // Protects the manager-global QAIRT profile handle across use/extract/reset.
  std::recursive_mutex profile_handle_mutex_;
  // Counts ORT profiling consumers so setup and cleanup can preserve the shared QAIRT handle.
  std::atomic<uint32_t> active_ort_profiler_count_{0};
  Qnn_ProfileHandle_t profile_handle_ = nullptr;
};

}  // namespace qnn
}  // namespace onnxruntime
