// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Host-side component tests for QnnBackendProfilingManager. The QNN profile
// API is stubbed so these tests do not need a backend library or a session.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>
#include <string>
#include <utility>

#include "core/providers/qnn/builder/qnn_backend_profiling_manager.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

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
    ++current_->free_calls;
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
     InitializeForCurrentConsumers_OrtConsumerOnly_CreatesOneBasicHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);

  manager->AcquireOrtProfilingConsumer();
  ASSERT_TRUE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());
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
     ReleaseOrtProfilingHandleIfUnused_OnlyOrtConsumer_ReleasesProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);

  manager->AcquireOrtProfilingConsumer();
  ASSERT_TRUE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());
  manager->ReleaseOrtProfilingConsumer();
  ASSERT_TRUE(manager->ReleaseOrtProfilingHandleIfUnused().IsOK());

  EXPECT_EQ(recorder.free_calls, 1);
  EXPECT_FALSE(manager->HasProfileHandle());
}

TEST(QnnUnit_BackendProfilingManagerTest,
     ReleaseOrtProfilingHandleIfUnused_ProviderProfiling_KeepsProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::BASIC);

  ASSERT_TRUE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());
  ASSERT_TRUE(manager->ReleaseOrtProfilingHandleIfUnused().IsOK());

  EXPECT_EQ(recorder.create_calls, 1);
  EXPECT_EQ(recorder.free_calls, 0);
  EXPECT_TRUE(manager->HasProfileHandle());
}

TEST(QnnUnit_BackendProfilingManagerTest,
     CreateGraphProfilingScope_OrtOnly_LazilyCreatesProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);
  qnn::QnnProfilingScope profiling_scope;

  manager->AcquireOrtProfilingConsumer();
  ASSERT_TRUE(manager->CreateGraphProfilingScope(true, env.logger, profiling_scope).IsOK());

  EXPECT_TRUE(profiling_scope.Active());
  EXPECT_EQ(recorder.create_calls, 1);
  EXPECT_EQ(recorder.created_level, QNN_PROFILE_LEVEL_BASIC);
}

TEST(QnnUnit_BackendProfilingManagerTest,
     CreateGraphProfilingScope_ProfilingOff_DoesNotCreateProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::OFF);
  qnn::QnnProfilingScope profiling_scope;

  ASSERT_TRUE(manager->CreateGraphProfilingScope(false, env.logger, profiling_scope).IsOK());
  EXPECT_FALSE(profiling_scope.Active());
  EXPECT_EQ(recorder.create_calls, 0);
}

TEST(QnnUnit_BackendProfilingManagerTest,
     CreateScopedProfilingInfo_ProviderOnly_SetsGraphNameAndUsesProfileHandle) {
  StubApiEnv env;
  ProfileApiRecorder recorder;
  auto manager = recorder.CreateManager(qnn::ProfilingLevel::BASIC);
  ASSERT_TRUE(manager->InitializeProfilingForCurrentConsumers(env.logger).IsOK());
  qnn::profile::ProfilingInfo profiling_info;

  auto profiling_scope = manager->CreateScopedProfilingInfo(
      profiling_info, "graph", qnn::OrtProfilingOperation::COMPOSE,
      qnn::ProfilingMethodType::COMPOSE_GRAPHS);

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
