// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Host-side component tests for QnnBackendProfilingManager. The QNN profile
// API is stubbed so these tests do not need a backend library or a session.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "core/providers/qnn/builder/qnn_backend_profiling_manager.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

class ProfileApiRecorder {
 public:
  ProfileApiRecorder() {
    current_ = this;
    qnn_interface_ = QNN_INTERFACE_VER_TYPE_INIT;
    qnn_interface_.profileCreate = ProfileCreate;
    qnn_interface_.profileFree = ProfileFree;
    qnn_interface_.errorGetMessage = ErrorGetMessage;
  }

  ~ProfileApiRecorder() { current_ = nullptr; }

  std::unique_ptr<qnn::QnnBackendProfilingManager> CreateManager(qnn::ProfilingLevel provider_level,
                                                                 bool backend_setup = true) {
    backend_setup_ = backend_setup;
    qnn::QnnBackendProfilingManagerDependencies dependencies{
        qnn_interface_, backend_handle_, qnn_system_interface_, nullptr, backend_setup_, [] { return Ort::Status{}; }};
    return std::make_unique<qnn::QnnBackendProfilingManager>(
        std::move(dependencies), provider_level, qnn::ProfilingLevel::OFF, std::string{}, false);
  }

  int create_calls = 0;
  int free_calls = 0;
  QnnProfile_Level_t created_level = static_cast<QnnProfile_Level_t>(0);
  Qnn_ErrorHandle_t create_result = QNN_PROFILE_NO_ERROR;
  bool block_profile_free = false;
  bool profile_free_entered = false;
  bool allow_profile_free = false;
  std::mutex callback_mutex;
  std::condition_variable callback_cv;

 private:
  static Qnn_ErrorHandle_t ProfileCreate(Qnn_BackendHandle_t /*backend*/, QnnProfile_Level_t level,
                                         Qnn_ProfileHandle_t* profile) {
    ++current_->create_calls;
    current_->created_level = level;
    if (current_->create_result == QNN_PROFILE_NO_ERROR) {
      *profile = reinterpret_cast<Qnn_ProfileHandle_t>(current_);
    }
    return current_->create_result;
  }

  static Qnn_ErrorHandle_t ProfileFree(Qnn_ProfileHandle_t /*profile*/) {
    ProfileApiRecorder* recorder = current_;
    if (recorder->block_profile_free) {
      std::unique_lock<std::mutex> lock(recorder->callback_mutex);
      recorder->profile_free_entered = true;
      recorder->callback_cv.notify_all();
      recorder->callback_cv.wait(lock, [recorder] { return recorder->allow_profile_free; });
    }
    ++recorder->free_calls;
    return QNN_PROFILE_NO_ERROR;
  }

  static Qnn_ErrorHandle_t ErrorGetMessage(Qnn_ErrorHandle_t /*error*/, const char** /*message*/) {
    return QNN_COMMON_ERROR_NOT_SUPPORTED;
  }

  static ProfileApiRecorder* current_;
  QNN_INTERFACE_VER_TYPE qnn_interface_ = QNN_INTERFACE_VER_TYPE_INIT;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_ = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle_ = nullptr;
  bool backend_setup_ = false;
};

ProfileApiRecorder* ProfileApiRecorder::current_ = nullptr;

TEST(QnnUnit_BackendProfilingManagerTest,
     InitializeForCurrentConsumers_ProfilingOff_DoesNotCreateProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);

  ASSERT_TRUE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());
  EXPECT_EQ(recorder.create_calls, 0);
  EXPECT_FALSE(manager->HasProfileHandle());
  EXPECT_FALSE(manager->ProfilingEnabled());
  EXPECT_FALSE(manager->AcquireProfilingScope().Active());
}

TEST(QnnUnit_BackendProfilingManagerTest,
     AcquireOrtProfilingConsumer_OrtConsumerOnly_CreatesOneBasicHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);

  ASSERT_TRUE(manager->AcquireOrtProfilingConsumer(env.logger).IsOK());
  ASSERT_TRUE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());

  EXPECT_EQ(recorder.create_calls, 1);
  EXPECT_EQ(recorder.created_level, QNN_PROFILE_LEVEL_BASIC);
  EXPECT_TRUE(manager->HasProfileHandle());
  EXPECT_TRUE(manager->ProfilingEnabled());
}

TEST(QnnUnit_BackendProfilingManagerTest,
     InitializeForCurrentConsumers_ProfileCreateFails_DoesNotPublishProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  recorder.create_result = QNN_PROFILE_ERROR_MEM_ALLOC;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::BASIC);

  EXPECT_FALSE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());
  EXPECT_EQ(recorder.create_calls, 1);
  EXPECT_FALSE(manager->HasProfileHandle());
  EXPECT_FALSE(manager->ProfilingEnabled());
}

TEST(QnnUnit_BackendProfilingManagerTest,
     ReleaseOrtProfilingConsumer_OnlyOrtConsumer_ReleasesProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);

  ASSERT_TRUE(manager->AcquireOrtProfilingConsumer(env.logger).IsOK());
  ASSERT_TRUE(manager->ReleaseOrtProfilingConsumer().IsOK());

  EXPECT_EQ(recorder.free_calls, 1);
  EXPECT_FALSE(manager->HasProfileHandle());
}

TEST(QnnUnit_BackendProfilingManagerTest,
     ReleaseOrtProfilingConsumer_ProviderProfiling_KeepsProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::BASIC);

  ASSERT_TRUE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());
  ASSERT_TRUE(manager->AcquireOrtProfilingConsumer(env.logger).IsOK());
  ASSERT_TRUE(manager->ReleaseOrtProfilingConsumer().IsOK());

  EXPECT_EQ(recorder.create_calls, 1);
  EXPECT_EQ(recorder.free_calls, 0);
  EXPECT_TRUE(manager->HasProfileHandle());
}

TEST(QnnUnit_BackendProfilingManagerTest,
     ReleaseOrtProfilingConsumer_ConcurrentAcquireRecreatesHandleAfterRelease) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);

  ASSERT_TRUE(manager->AcquireOrtProfilingConsumer(env.logger).IsOK());
  recorder.block_profile_free = true;

  Ort::Status release_status;
  std::thread release_thread([&] {
    release_status = manager->ReleaseOrtProfilingConsumer();
  });

  {
    std::unique_lock<std::mutex> lock(recorder.callback_mutex);
    recorder.callback_cv.wait(lock, [&] { return recorder.profile_free_entered; });
  }

  bool acquire_started = false;
  Ort::Status acquire_status;
  std::thread acquire_thread([&] {
    {
      std::lock_guard<std::mutex> lock(recorder.callback_mutex);
      acquire_started = true;
      recorder.callback_cv.notify_all();
    }
    acquire_status = manager->AcquireOrtProfilingConsumer(env.logger);
  });

  {
    std::unique_lock<std::mutex> lock(recorder.callback_mutex);
    recorder.callback_cv.wait(lock, [&] { return acquire_started; });
    recorder.allow_profile_free = true;
    recorder.callback_cv.notify_all();
  }

  release_thread.join();
  acquire_thread.join();

  ASSERT_TRUE(release_status.IsOK());
  ASSERT_TRUE(acquire_status.IsOK());
  EXPECT_TRUE(manager->HasActiveOrtProfilingConsumer());
  EXPECT_TRUE(manager->HasProfileHandle());
  EXPECT_EQ(recorder.create_calls, 2);
  EXPECT_EQ(recorder.free_calls, 1);

  recorder.block_profile_free = false;
  ASSERT_TRUE(manager->ReleaseOrtProfilingConsumer().IsOK());
}

TEST(QnnUnit_BackendProfilingManagerTest,
     CreateGraphProfilingScope_OrtOnly_UsesActiveProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);
  qnn::QnnProfilingScope profiling_scope;
  qnn::profile::ProfilingInfo profiling_info;

  ASSERT_TRUE(manager->AcquireOrtProfilingConsumer(env.logger).IsOK());
  auto* ort_profiler = reinterpret_cast<qnn::QnnEpProfiler*>(static_cast<uintptr_t>(1));
  ASSERT_TRUE(manager->CreateGraphProfilingScope(
                         profiling_info, "graph", qnn::OrtProfilingOperation::EXECUTE,
                         qnn::ProfilingMethodType::EXECUTE, ort_profiler, env.logger, profiling_scope)
                  .IsOK());

  EXPECT_TRUE(profiling_scope.Active());
  EXPECT_EQ(recorder.create_calls, 1);
  EXPECT_EQ(recorder.created_level, QNN_PROFILE_LEVEL_BASIC);
}

TEST(QnnUnit_BackendProfilingManagerTest,
     CreateGraphProfilingScope_OrtOnly_LazilyRecreatesMissingProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);
  qnn::QnnProfilingScope profiling_scope;
  qnn::profile::ProfilingInfo profiling_info;

  ASSERT_TRUE(manager->AcquireOrtProfilingConsumer(env.logger).IsOK());
  ASSERT_TRUE(manager->ReleaseProfileHandle().IsOK());
  auto* ort_profiler = reinterpret_cast<qnn::QnnEpProfiler*>(static_cast<uintptr_t>(1));
  ASSERT_TRUE(manager->CreateGraphProfilingScope(
                         profiling_info, "graph", qnn::OrtProfilingOperation::EXECUTE,
                         qnn::ProfilingMethodType::EXECUTE, ort_profiler, env.logger, profiling_scope)
                  .IsOK());

  EXPECT_TRUE(profiling_scope.Active());
  EXPECT_EQ(recorder.create_calls, 2);
  EXPECT_EQ(recorder.created_level, QNN_PROFILE_LEVEL_BASIC);
}

TEST(QnnUnit_BackendProfilingManagerTest,
     CreateGraphProfilingScope_ProfilingOff_DoesNotCreateProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);
  qnn::QnnProfilingScope profiling_scope;
  qnn::profile::ProfilingInfo profiling_info;

  ASSERT_TRUE(manager->CreateGraphProfilingScope(
                         profiling_info, "graph", qnn::OrtProfilingOperation::EXECUTE,
                         qnn::ProfilingMethodType::EXECUTE, nullptr, env.logger, profiling_scope)
                  .IsOK());
  EXPECT_FALSE(profiling_scope.Active());
  EXPECT_EQ(recorder.create_calls, 0);
}

TEST(QnnUnit_BackendProfilingManagerTest,
     CreateSetupProfilingScope_ProviderOnly_SetsGraphNameAndUsesProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::BASIC);
  ASSERT_TRUE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());
  qnn::profile::ProfilingInfo profiling_info;

  qnn::QnnProfilingScope profiling_scope;
  ASSERT_TRUE(manager->CreateSetupProfilingScope(
                         profiling_info, "graph", qnn::OrtProfilingOperation::COMPOSE,
                         qnn::ProfilingMethodType::COMPOSE_GRAPHS, profiling_scope)
                  .IsOK());

  EXPECT_TRUE(profiling_scope.Active());
  EXPECT_EQ(profiling_info.graph_name, "graph");
  EXPECT_EQ(profiling_info.ort_profiler, nullptr);
}

TEST(QnnUnit_BackendProfilingManagerTest,
     SetProfilingLevelETW_HigherLevel_RecreatesProfileHandleAtMergedLevel) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::BASIC);
  ASSERT_TRUE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());

  ASSERT_TRUE(manager->SetProfilingLevelETW(qnn::ProfilingLevel::DETAILED, env.logger).IsOK());

  EXPECT_EQ(recorder.create_calls, 2);
  EXPECT_EQ(recorder.free_calls, 1);
  EXPECT_EQ(recorder.created_level, QNN_PROFILE_LEVEL_DETAILED);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
