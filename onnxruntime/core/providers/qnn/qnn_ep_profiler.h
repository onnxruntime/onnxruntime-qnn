// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/qnn/ort_api.h"

#if QNN_ORT_EP_PROFILING_API_ENABLED

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <QnnInterface.h>
#include <QnnProfile.h>

namespace onnxruntime {
namespace qnn {

class QnnBackendProfilingManager;
namespace profile {
struct ProfilingInfo;
}  // namespace profile

inline int32_t CurrentThreadId() noexcept {
  return static_cast<int32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

// Raw QAIRT profiling event. QAIRT timestamps are converted to the ORT timeline when extracted.
struct EventRecord {
  OrtProfilingEventCategory category = OrtProfilingEventCategory_KERNEL;
  int32_t process_id = 0;
  int32_t thread_id = 0;
  std::string name;
  std::vector<std::pair<std::string, std::string>> args;

  uint64_t qairt_ts_us = 0;       // device-clock timestamp (0 for basic events)
  int64_t qairt_duration_us = 0;  // duration for MICROSEC events; 0 otherwise

  int64_t ort_event_start_us = -1;  // set by StopEvent; -1 means unanchored

  int64_t final_ts_us = 0;  // ORT-timeline placement, computed during extraction
  int64_t final_duration_us = 0;
};

// OrtEpProfilerImpl for the QNN EP; delivers QAIRT events into ORT's unified profiling timeline.
//
// High-level flow:
//  1. ORT creates one QnnEpProfiler per profiling session and calls StartProfiling on the session thread.
//  2. ORT calls StartEvent/StopEvent around each profiled ORT node on the run thread. StartEvent pushes
//     a thread-local scope so QNN graph execution can find the current profiler without cross-thread state.
//  3. QNN operations append their ProfilingInfo to that scope. StopEvent drains those pending extractions,
//     converts QAIRT timestamps to ORT timeline timestamps, and attaches the ORT parent node name.
//  4. EndProfiling transfers completed records to ORT. AppendEvent is mutex-protected because setup/finalize
//     and execute extraction paths can run on different threads.
class QnnEpProfiler : public OrtEpProfilerImpl {
 public:
  QnnEpProfiler(const OrtEpApi& ep_api,
                const OrtApi& ort_api,
                QnnBackendProfilingManager& profiling_manager);
  ~QnnEpProfiler();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnEpProfiler);

  static QnnEpProfiler* Current() noexcept;
  void AppendEvent(EventRecord record);  // thread-safe; called while extracting QAIRT profile events
  bool AppendQnnEventRecords(QNN_INTERFACE_VER_TYPE qnn_interface,
                             QnnProfile_EventId_t event_id,
                             const std::string& event_level,
                             bool use_extended_event_data,
                             uint64_t root_qairt_timestamp_us,
                             uint64_t operation_start_time_us,
                             const std::string& operation,
                             const std::string& graph_name);
  void AppendHostOperationRecord(uint64_t operation_start_time_us,
                                 uint64_t operation_end_time_us,
                                 const std::string& operation,
                                 const std::string& graph_name);
  // Appends QAIRT events from the current profile handle extraction.
  // Called by QnnBackendProfilingManager while its QAIRT profile-handle lock is held.
  Ort::Status AppendNewQnnEventRecords(QNN_INTERFACE_VER_TYPE qnn_interface,
                                       const QnnProfile_EventId_t* event_ids,
                                       uint32_t num_events,
                                       bool use_extended_event_data,
                                       const profile::ProfilingInfo& profiling_info);
  void QueueExecuteProfilingExtraction(profile::ProfilingInfo profiling_info);

  size_t MarkPendingExecuteProfilingExtractions();
  void DiscardPendingExecuteProfilingExtractionsSince(size_t mark);

 private:
  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                    int64_t ep_profiling_start_offset_ns) noexcept;
  static OrtStatus* ORT_API_CALL StartEventImpl(OrtEpProfilerImpl* this_ptr,
                                                uint64_t ort_event_correlation_id) noexcept;
  static OrtStatus* ORT_API_CALL StopEventImpl(OrtEpProfilerImpl* this_ptr,
                                               uint64_t ort_event_correlation_id,
                                               const OrtProfilingEvent* ort_event) noexcept;
  static OrtStatus* ORT_API_CALL EndProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                  OrtProfilingEventsContainer* events_container) noexcept;

  Ort::Status DrainExecuteProfilingExtractions(std::vector<profile::ProfilingInfo> profiling_infos);
  class OrtProfilingConsumer {
   public:
    explicit OrtProfilingConsumer(QnnBackendProfilingManager& profiling_manager)
        : profiling_manager_(profiling_manager) {}
    ~OrtProfilingConsumer() { ORT_IGNORE_RETURN_VALUE(Reset()); }

    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtProfilingConsumer);

    void Activate() noexcept;
    Ort::Status Reset();

   private:
    QnnBackendProfilingManager& profiling_manager_;
    bool active_ = false;
  };

  const OrtEpApi& ep_api_;
  const OrtApi& ort_api_;
  QnnBackendProfilingManager& profiling_manager_;
  OrtProfilingConsumer ort_profiling_consumer_;

  int64_t ep_profiling_start_offset_ns_ = 0;
  uint64_t ep_profiling_start_time_us_ = 0;

  std::mutex records_mutex_;
  std::vector<EventRecord> event_records_;  // guarded by records_mutex_
};

}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_ORT_EP_PROFILING_API_ENABLED
