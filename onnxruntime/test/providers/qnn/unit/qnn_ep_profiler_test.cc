// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Host-side component tests for the ORT EP profiling API bridge.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/ort_api.h"

#if QNN_ORT_EP_PROFILING_API_ENABLED

#include "core/providers/qnn/builder/qnn_backend_profiling_manager.h"
#include "core/providers/qnn/qnn_ep_profiler.h"

#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

class ProfilingManagerContext {
 public:
  ProfilingManagerContext() {
    current_ = this;
    qnn_interface_.profileCreate = ProfileCreate;
    qnn_interface_.errorGetMessage = ErrorGetMessage;
  }

  ~ProfilingManagerContext() { current_ = nullptr; }

  std::unique_ptr<qnn::QnnBackendProfilingManager> CreateManager(bool backend_setup = false) {
    backend_setup_ = backend_setup;
    qnn::QnnBackendProfilingManagerDependencies dependencies{
        qnn_interface_, backend_handle_, qnn_system_interface_, nullptr, backend_setup_, [] { return Ort::Status{}; }};
    return std::make_unique<qnn::QnnBackendProfilingManager>(std::move(dependencies),
                                                             qnn::ProfilingLevel::OFF, qnn::ProfilingLevel::OFF,
                                                             std::string{}, false);
  }

  Qnn_ErrorHandle_t profile_create_result = QNN_PROFILE_NO_ERROR;
  int profile_create_calls = 0;

 private:
  static Qnn_ErrorHandle_t ProfileCreate(Qnn_BackendHandle_t /*backend*/, QnnProfile_Level_t /*level*/,
                                         Qnn_ProfileHandle_t* profile) {
    ++current_->profile_create_calls;
    if (current_->profile_create_result == QNN_PROFILE_NO_ERROR) {
      *profile = reinterpret_cast<Qnn_ProfileHandle_t>(current_);
    }
    return current_->profile_create_result;
  }

  static Qnn_ErrorHandle_t ErrorGetMessage(Qnn_ErrorHandle_t /*error*/, const char** /*message*/) {
    return QNN_COMMON_ERROR_NOT_SUPPORTED;
  }

  static ProfilingManagerContext* current_;
  QNN_INTERFACE_VER_TYPE qnn_interface_ = QNN_INTERFACE_VER_TYPE_INIT;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_ = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle_ = nullptr;
  bool backend_setup_ = false;
};

ProfilingManagerContext* ProfilingManagerContext::current_ = nullptr;

class StatusApi {
 public:
  StatusApi() {
    api.CreateStatus = CreateStatus;
    api.ReleaseStatus = ReleaseStatus;
  }

  OrtApi api{};

 private:
  struct Status {
    OrtErrorCode code;
    std::string message;
  };

  static OrtStatus* ORT_API_CALL CreateStatus(OrtErrorCode code, const char* message) noexcept {
    return reinterpret_cast<OrtStatus*>(new Status{code, message ? message : ""});
  }

  static void ORT_API_CALL ReleaseStatus(OrtStatus* status) noexcept {
    delete reinterpret_cast<Status*>(status);
  }
};

class ProfilingEventApi {
 public:
  ProfilingEventApi() {
    current_ = this;
    ep_api.ProfilingEvent_GetName = GetName;
    ep_api.ProfilingEvent_GetTimestampUs = GetTimestampUs;
    ep_api.CreateProfilingEvent = CreateProfilingEvent;
    ep_api.ReleaseProfilingEvent = ReleaseProfilingEvent;
    ep_api.ProfilingEventsContainer_AddEvents = AddEvents;
  }

  ~ProfilingEventApi() { current_ = nullptr; }

  OrtEpApi ep_api{};
  int create_event_calls = 0;
  int release_event_calls = 0;
  int add_events_calls = 0;
  std::string event_name;
  std::vector<std::pair<std::string, std::string>> event_args;

 private:
  static OrtStatus* ORT_API_CALL GetName(const OrtProfilingEvent* /*event*/, const char** out) noexcept {
    *out = "QNN node";
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL GetTimestampUs(const OrtProfilingEvent* /*event*/, int64_t* out) noexcept {
    *out = 42;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateProfilingEvent(
      OrtProfilingEventCategory /*category*/, int32_t /*process_id*/, int32_t /*thread_id*/, const char* event_name,
      int64_t /*timestamp_us*/, int64_t /*duration_us*/, const char* const* arg_keys, const char* const* arg_values,
      size_t num_args, OrtProfilingEvent** out) noexcept {
    ++current_->create_event_calls;
    current_->event_name = event_name ? event_name : "";
    current_->event_args.clear();
    for (size_t i = 0; i < num_args; ++i) {
      current_->event_args.emplace_back(arg_keys[i] ? arg_keys[i] : "", arg_values[i] ? arg_values[i] : "");
    }
    *out = reinterpret_cast<OrtProfilingEvent*>(current_);
    return nullptr;
  }

  static void ORT_API_CALL ReleaseProfilingEvent(OrtProfilingEvent* /*event*/) noexcept {
    ++current_->release_event_calls;
  }

  static OrtStatus* ORT_API_CALL AddEvents(OrtProfilingEventsContainer* /*container*/,
                                           const OrtProfilingEvent* const* /*events*/, size_t /*num_events*/) noexcept {
    ++current_->add_events_calls;
    return nullptr;
  }

  static ProfilingEventApi* current_;
};

ProfilingEventApi* ProfilingEventApi::current_ = nullptr;

TEST(QnnUnit_EpProfilerTest, StartAndEndProfiling_ActivatesAndReleasesOrtConsumer) {
  StubApiEnv env;
  ProfilingManagerContext manager_context;
  auto manager = manager_context.CreateManager();
  ProfilingEventApi ep_api;
  qnn::QnnEpProfiler profiler(ep_api.ep_api, env.stub_ort_api, env.logger, *manager);

  ASSERT_EQ(profiler.StartProfiling(&profiler, 0), nullptr);
  EXPECT_TRUE(manager->HasActiveOrtProfilingConsumer());

  ASSERT_EQ(profiler.EndProfiling(&profiler, reinterpret_cast<OrtProfilingEventsContainer*>(uintptr_t{1})), nullptr);
  EXPECT_FALSE(manager->HasActiveOrtProfilingConsumer());
}

TEST(QnnUnit_EpProfilerTest, StartProfiling_ProfileCreateFails_ReleasesOrtConsumer) {
  StubApiEnv env;
  ProfilingManagerContext manager_context;
  manager_context.profile_create_result = QNN_PROFILE_ERROR_MEM_ALLOC;
  auto manager = manager_context.CreateManager(true);
  ProfilingEventApi ep_api;
  qnn::QnnEpProfiler profiler(ep_api.ep_api, env.stub_ort_api, env.logger, *manager);
  StatusApi status_api;
  OrtGlobalApiOverride api_override(&status_api.api);

  OrtStatus* status = profiler.StartProfiling(&profiler, 0);

  ASSERT_NE(status, nullptr);
  status_api.api.ReleaseStatus(status);
  EXPECT_EQ(manager_context.profile_create_calls, 1);
  EXPECT_FALSE(manager->HasActiveOrtProfilingConsumer());
}

TEST(QnnUnit_EpProfilerTest, StartAndStopEvent_SetsAndClearsCurrentProfiler) {
  StubApiEnv env;
  ProfilingManagerContext manager_context;
  auto manager = manager_context.CreateManager();
  ProfilingEventApi ep_api;
  qnn::QnnEpProfiler profiler(ep_api.ep_api, env.stub_ort_api, env.logger, *manager);

  ASSERT_EQ(profiler.StartProfiling(&profiler, 0), nullptr);
  ASSERT_EQ(profiler.StartEvent(&profiler, 0), nullptr);
  EXPECT_EQ(qnn::QnnEpProfiler::Current(), &profiler);

  ASSERT_EQ(profiler.StopEvent(&profiler, 0, reinterpret_cast<const OrtProfilingEvent*>(uintptr_t{1})), nullptr);
  EXPECT_EQ(qnn::QnnEpProfiler::Current(), nullptr);
  ASSERT_EQ(profiler.EndProfiling(&profiler, reinterpret_cast<OrtProfilingEventsContainer*>(uintptr_t{1})), nullptr);
}

TEST(QnnUnit_EpProfilerTest, EndProfiling_HostRecord_CreatesAndTransfersOrtEvent) {
  StubApiEnv env;
  ProfilingManagerContext manager_context;
  auto manager = manager_context.CreateManager();
  ProfilingEventApi ep_api;
  qnn::QnnEpProfiler profiler(ep_api.ep_api, env.stub_ort_api, env.logger, *manager);

  ASSERT_EQ(profiler.StartProfiling(&profiler, 0), nullptr);
  ASSERT_EQ(profiler.StartEvent(&profiler, 0), nullptr);
  profiler.AppendHostOperationRecord(10, 20, qnn::OrtProfilingOperation::EXECUTE, "graph");
  ASSERT_EQ(profiler.StopEvent(&profiler, 0, reinterpret_cast<const OrtProfilingEvent*>(uintptr_t{1})), nullptr);
  ASSERT_EQ(profiler.EndProfiling(&profiler, reinterpret_cast<OrtProfilingEventsContainer*>(uintptr_t{1})), nullptr);

  EXPECT_EQ(ep_api.create_event_calls, 1);
  EXPECT_EQ(ep_api.add_events_calls, 1);
  EXPECT_EQ(ep_api.release_event_calls, 1);
  EXPECT_EQ(ep_api.event_name, "QNN execute");
  EXPECT_NE(std::find(ep_api.event_args.begin(), ep_api.event_args.end(),
                      std::make_pair(std::string{"parent_ort_node"}, std::string{"QNN node"})),
            ep_api.event_args.end());
}

TEST(QnnUnit_EpProfilerTest, PendingExecuteExtractions_DiscardSinceMark_RemovesRetryRecords) {
  StubApiEnv env;
  ProfilingManagerContext manager_context;
  auto manager = manager_context.CreateManager();
  ProfilingEventApi ep_api;
  qnn::QnnEpProfiler profiler(ep_api.ep_api, env.stub_ort_api, env.logger, *manager);

  ASSERT_EQ(profiler.StartProfiling(&profiler, 0), nullptr);
  ASSERT_EQ(profiler.StartEvent(&profiler, 0), nullptr);
  const size_t mark = profiler.MarkPendingExecuteProfilingExtractions();
  profiler.QueueExecuteProfilingExtraction(qnn::profile::ProfilingInfo{});
  EXPECT_EQ(profiler.MarkPendingExecuteProfilingExtractions(), mark + 1);

  profiler.DiscardPendingExecuteProfilingExtractionsSince(mark);
  EXPECT_EQ(profiler.MarkPendingExecuteProfilingExtractions(), mark);
  ASSERT_EQ(profiler.StopEvent(&profiler, 0, reinterpret_cast<const OrtProfilingEvent*>(uintptr_t{1})), nullptr);
  ASSERT_EQ(profiler.EndProfiling(&profiler, reinterpret_cast<OrtProfilingEventsContainer*>(uintptr_t{1})), nullptr);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // QNN_ORT_EP_PROFILING_API_ENABLED
#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
