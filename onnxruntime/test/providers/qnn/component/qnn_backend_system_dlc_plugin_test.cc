// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for QnnBackendSystemDlcPlugin (qnn_backend_system_dlc_plugin.cc).
//
// Two groups of tests live here, both gated on QNN_EP_INTERNAL_SYMBOL_ACCESS:
//
//   1. Stub-based tests (QnnUnit_BackendSystemDlcPluginTest) — the plugin is bound to a
//      StubBackendManager whose QNN / QNN System interfaces are populated with recording
//      stubs, so every success / failure branch can be driven deterministically without
//      any QNN library.
//   2. Real-HTP-backend tests (QnnUnit_BackendSystemDlcPluginHtpTest) — load libQnnHtp.so
//      and libQnnSystem.so through QnnRealHtpBackendManagerContext and exercise the plugin
//      against the real QNN System API. The fixture GTEST_SKIP()s when unavailable.
//
// Coverage targets:
//   - SetupDlc / SetupDlcFromBinary (CreateSystemLog, CreateEmptyDlc, CreateDlcFromBinary,
//     rollback on failure)
//   - Release / destructor (ReleaseDlc, ReleaseSystemLog)
//   - AddContextToDlc
//   - GetDlcBinaryBuffer
//   - GetDlcBinaryInfo / GetDlcMaxSpillFillBufferSize (GetDlcRecordBuffers)

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

// QnnCommon.h must precede the HTP/System headers: it defines the UNNAMED macro
// that turns their `union UNNAMED { ... }` into anonymous unions.
#include "QnnCommon.h"
#include "HTP/QnnHtpSystemContext.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_backend_system_dlc_plugin.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr Qnn_ErrorHandle_t kStubError = QNN_COMMON_ERROR_GENERAL;

void ExpectErrorContains(const Ort::Status& status, const std::string& expected) {
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find(expected), std::string::npos)
      << "Actual error message: " << status.GetErrorMessage();
}

}  // namespace

// ===========================================================================
// Stub-based tests
// ===========================================================================

// Fixture: binds a QnnBackendSystemDlcPlugin to a StubBackendManager and installs
// recording stubs for the QNN System (SystemLog / SystemDlc / SystemContext) API and
// the QNN contextAddToDlc / errorGetMessage API. Stubs are plain functions (QNN takes
// C function pointers), so they reach the active fixture through `current_`.
//
// Individual tests break a code path by nulling a function pointer
// (e.g. SysInterface().systemLogCreate = nullptr) or by setting a *_result field.
class QnnUnit_BackendSystemDlcPluginTest : public ::testing::Test {
 protected:
  // One graph inside a fake HTP cache record (context binary).
  struct GraphSpec {
    QnnSystemContext_GraphInfoVersion_t version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3;
    QnnHtpSystemContext_GraphInfoBlobVersion_t blob_version = QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1;
    uint64_t spill_fill_buffer_size = 0;
  };

  void SetUp() override {
    current_ = this;

    QNN_SYSTEM_INTERFACE_VER_TYPE& sys = SysInterface();
    sys.systemLogCreate = SystemLogCreate;
    sys.systemLogFree = SystemLogFree;
    sys.systemContextCreate = SystemContextCreate;
    sys.systemContextFree = SystemContextFree;
    sys.systemContextGetBinaryInfo = SystemContextGetBinaryInfo;
#ifdef QNN_SYSTEM_DLC_API_ENABLED
    sys.systemDlcCreateWithDestinationDir = SystemDlcCreateWithDestinationDir;
    sys.systemDlcCreateFromBinary = SystemDlcCreateFromBinary;
    sys.systemDlcFree = SystemDlcFree;
    sys.systemDlcGetBinarySize = SystemDlcGetBinarySize;
    sys.systemDlcGetBinary = SystemDlcGetBinary;
    sys.systemDlcGetRecordsByType = SystemDlcGetRecordsByType;
    sys.systemDlcReadRecordDataMemoryMapped = SystemDlcReadRecordDataMemoryMapped;
    QnnInterface().contextAddToDlc = ContextAddToDlc;
#endif  // QNN_SYSTEM_DLC_API_ENABLED
    // Reached by QnnBackendManager::QnnErrorHandleToString on every QNN failure path.
    QnnInterface().errorGetMessage = ErrorGetMessage;

    plugin_ = std::make_unique<qnn::QnnBackendSystemDlcPlugin>(backend_manager_.GetMutable());
  }

  void TearDown() override {
    // The plugin's destructor calls back into the manager and the stubs, so destroy it
    // while the fixture is still current.
    plugin_.reset();
    current_ = nullptr;
  }

  QNN_SYSTEM_INTERFACE_VER_TYPE& SysInterface() { return backend_manager_.SystemInterface(); }
  QNN_INTERFACE_VER_TYPE& QnnInterface() { return backend_manager_.QnnInterface(); }
  qnn::QnnBackendSystemDlcPlugin& Plugin() { return *plugin_; }

  Qnn_LogHandle_t FakeLogHandle() { return &log_sentinel_; }
  QnnSystemDlc_Handle_t FakeDlcHandle() { return &dlc_sentinel_; }
  QnnSystemContext_Handle_t FakeSysCtxHandle() { return &sys_ctx_sentinel_; }
  Qnn_ContextHandle_t FakeContextHandle() { return &context_sentinel_; }

  // Appends one HTP cache record whose QnnSystemContext binary info describes `graphs`.
  void AddRecord(const std::vector<GraphSpec>& graphs, Qnn_Version_t blob_version = {1, 2, 3}) {
    // deque keeps element addresses stable across push_back, so pointers handed out to
    // the code under test below stay valid while more records are added.
    records_.emplace_back(16, static_cast<uint8_t>(records_.size() + 1));
    std::vector<QnnSystemContext_GraphInfo_t>& graph_infos = graph_infos_.emplace_back();
    for (const GraphSpec& spec : graphs) {
      QnnSystemContext_GraphInfo_t graph_info{};
      graph_info.version = spec.version;
      if (spec.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        QnnHtpSystemContext_GraphBlobInfo_t& blob = blobs_.emplace_back();
        std::memset(&blob, 0, sizeof(blob));
        blob.version = spec.blob_version;
        blob.contextBinaryGraphBlobInfoV1.spillFillBufferSize = spec.spill_fill_buffer_size;
        graph_info.graphInfoV3.graphBlobInfo = &blob;
        graph_info.graphInfoV3.graphBlobInfoSize = static_cast<uint32_t>(sizeof(blob));
      }
      graph_infos.push_back(graph_info);
    }

    QnnSystemContext_BinaryInfo_t& binary_info = binary_infos_.emplace_back();
    std::memset(&binary_info, 0, sizeof(binary_info));
    binary_info.version = QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1;
    binary_info.contextBinaryInfoV1.numGraphs = static_cast<uint32_t>(graph_infos.size());
    binary_info.contextBinaryInfoV1.graphs = graph_infos.data();
    binary_info.contextBinaryInfoV1.contextBlobVersion = blob_version;
  }

  // ---- Recorded stub arguments and per-test result knobs ----

  std::vector<std::string> calls;

  Qnn_ErrorHandle_t log_create_result = QNN_SUCCESS;
  Qnn_ErrorHandle_t log_free_result = QNN_SUCCESS;
  QnnLog_Callback_t log_create_callback = nullptr;
  QnnLog_Level_t log_create_level = QNN_LOG_LEVEL_MAX;

  Qnn_ErrorHandle_t dlc_create_result = QNN_SUCCESS;
  Qnn_ErrorHandle_t dlc_free_result = QNN_SUCCESS;
  Qnn_LogHandle_t dlc_create_logger = nullptr;
  // Non-null sentinel so a test can tell the stub recorded a nullptr argument.
  const char* dlc_create_destination_dir = reinterpret_cast<const char*>(1);
  const uint8_t* dlc_create_buffer = nullptr;
  uint64_t dlc_create_buffer_size = 0;
  QnnSystemDlc_Handle_t dlc_freed_handle = nullptr;

  Qnn_ErrorHandle_t add_to_dlc_result = QNN_SUCCESS;
  Qnn_ContextHandle_t add_to_dlc_context = nullptr;
  QnnSystemDlc_Handle_t add_to_dlc_dlc = nullptr;

  std::vector<uint8_t> dlc_binary;
  Qnn_ErrorHandle_t get_binary_size_result = QNN_SUCCESS;
  Qnn_ErrorHandle_t get_binary_result = QNN_SUCCESS;
  uint64_t get_binary_written_size_override = 0;  // 0 => report dlc_binary.size()

  Qnn_ErrorHandle_t get_records_result = QNN_SUCCESS;
  Qnn_ErrorHandle_t read_record_result = QNN_SUCCESS;
#ifdef QNN_SYSTEM_DLC_API_ENABLED
  QnnSystemDlc_RecordType_t get_records_type = QNN_SYSTEM_DLC_RECORD_NAME_UNKNOWN;
#endif  // QNN_SYSTEM_DLC_API_ENABLED
  uint8_t get_records_most_optimal = 0xFF;

  Qnn_ErrorHandle_t sys_ctx_create_result = QNN_SUCCESS;
  Qnn_ErrorHandle_t get_binary_info_result = QNN_SUCCESS;
  std::vector<const void*> get_binary_info_buffers;
  std::vector<QnnSystemContext_Handle_t> get_binary_info_sys_ctx;

  const void* RecordData(size_t idx) const { return records_[idx].data(); }

 private:
  // ---- QNN System API stubs ----

  static Qnn_ErrorHandle_t SystemLogCreate(QnnLog_Callback_t callback, QnnLog_Level_t level,
                                           Qnn_LogHandle_t* logger) {
    current_->calls.push_back("systemLogCreate");
    current_->log_create_callback = callback;
    current_->log_create_level = level;
    if (current_->log_create_result != QNN_SUCCESS) return current_->log_create_result;
    *logger = current_->FakeLogHandle();
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t SystemLogFree(Qnn_LogHandle_t /*logger*/) {
    current_->calls.push_back("systemLogFree");
    return current_->log_free_result;
  }

  static Qnn_ErrorHandle_t SystemContextCreate(QnnSystemContext_Handle_t* handle) {
    current_->calls.push_back("systemContextCreate");
    if (current_->sys_ctx_create_result != QNN_SUCCESS) return current_->sys_ctx_create_result;
    *handle = current_->FakeSysCtxHandle();
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t SystemContextFree(QnnSystemContext_Handle_t /*handle*/) {
    current_->calls.push_back("systemContextFree");
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t SystemContextGetBinaryInfo(QnnSystemContext_Handle_t sys_ctx, void* buffer,
                                                      uint64_t /*buffer_size*/,
                                                      const QnnSystemContext_BinaryInfo_t** binary_info,
                                                      Qnn_ContextBinarySize_t* binary_info_size) {
    current_->calls.push_back("systemContextGetBinaryInfo");
    current_->get_binary_info_sys_ctx.push_back(sys_ctx);
    current_->get_binary_info_buffers.push_back(buffer);
    if (current_->get_binary_info_result != QNN_SUCCESS) return current_->get_binary_info_result;
    for (size_t idx = 0; idx < current_->records_.size(); ++idx) {
      if (current_->records_[idx].data() == buffer) {
        *binary_info = &current_->binary_infos_[idx];
        *binary_info_size = sizeof(QnnSystemContext_BinaryInfo_t);
        return QNN_SUCCESS;
      }
    }
    return kStubError;
  }

#ifdef QNN_SYSTEM_DLC_API_ENABLED
  static Qnn_ErrorHandle_t SystemDlcCreateWithDestinationDir(Qnn_LogHandle_t logger, const char* destination_dir,
                                                             QnnSystemDlc_Handle_t* dlc_handle) {
    current_->calls.push_back("systemDlcCreateWithDestinationDir");
    current_->dlc_create_logger = logger;
    current_->dlc_create_destination_dir = destination_dir;
    if (current_->dlc_create_result != QNN_SUCCESS) return current_->dlc_create_result;
    *dlc_handle = current_->FakeDlcHandle();
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t SystemDlcCreateFromBinary(Qnn_LogHandle_t logger, const uint8_t* buffer,
                                                     const Qnn_ContextBinarySize_t buffer_size,
                                                     QnnSystemDlc_Handle_t* dlc_handle) {
    current_->calls.push_back("systemDlcCreateFromBinary");
    current_->dlc_create_logger = logger;
    current_->dlc_create_buffer = buffer;
    current_->dlc_create_buffer_size = buffer_size;
    if (current_->dlc_create_result != QNN_SUCCESS) return current_->dlc_create_result;
    *dlc_handle = current_->FakeDlcHandle();
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t SystemDlcFree(QnnSystemDlc_Handle_t dlc_handle) {
    current_->calls.push_back("systemDlcFree");
    current_->dlc_freed_handle = dlc_handle;
    return current_->dlc_free_result;
  }

  static Qnn_ErrorHandle_t SystemDlcGetBinarySize(QnnSystemDlc_Handle_t /*dlc_handle*/,
                                                  Qnn_SystemDlcBinarySize_t* size) {
    current_->calls.push_back("systemDlcGetBinarySize");
    if (current_->get_binary_size_result != QNN_SUCCESS) return current_->get_binary_size_result;
    *size = current_->dlc_binary.size();
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t SystemDlcGetBinary(QnnSystemDlc_Handle_t /*dlc_handle*/, uint8_t* buffer,
                                              Qnn_SystemDlcBinarySize_t buffer_size,
                                              Qnn_SystemDlcBinarySize_t* written_size) {
    current_->calls.push_back("systemDlcGetBinary");
    if (current_->get_binary_result != QNN_SUCCESS) return current_->get_binary_result;
    const std::vector<uint8_t>& binary = current_->dlc_binary;
    std::memcpy(buffer, binary.data(), std::min<uint64_t>(buffer_size, binary.size()));
    *written_size = current_->get_binary_written_size_override != 0 ? current_->get_binary_written_size_override
                                                                    : binary.size();
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t SystemDlcGetRecordsByType(QnnSystemDlc_Handle_t /*dlc_handle*/,
                                                     QnnSystemDlc_RecordType_t record_type,
                                                     uint8_t most_optimal_only,
                                                     QnnSystemDlc_RecordHandle_t** record_handles,
                                                     uint32_t* num_record_handles) {
    current_->calls.push_back("systemDlcGetRecordsByType");
    current_->get_records_type = record_type;
    current_->get_records_most_optimal = most_optimal_only;
    if (current_->get_records_result != QNN_SUCCESS) return current_->get_records_result;

    std::vector<QnnSystemDlc_RecordHandle_t>& handles = current_->record_handles_;
    handles.clear();
    for (std::vector<uint8_t>& record : current_->records_) {
      handles.push_back(&record);
    }
    *record_handles = handles.data();
    *num_record_handles = static_cast<uint32_t>(handles.size());
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t SystemDlcReadRecordDataMemoryMapped(QnnSystemDlc_RecordHandle_t record_handle,
                                                               const uint8_t** data, uint64_t* data_size) {
    current_->calls.push_back("systemDlcReadRecordDataMemoryMapped");
    if (current_->read_record_result != QNN_SUCCESS) return current_->read_record_result;
    const auto* record = static_cast<const std::vector<uint8_t>*>(record_handle);
    *data = record->data();
    *data_size = record->size();
    return QNN_SUCCESS;
  }

  // ---- QNN (backend) API stubs ----

  static Qnn_ErrorHandle_t ContextAddToDlc(Qnn_ContextHandle_t context, QnnSystemDlc_Handle_t dlc_handle) {
    current_->calls.push_back("contextAddToDlc");
    current_->add_to_dlc_context = context;
    current_->add_to_dlc_dlc = dlc_handle;
    return current_->add_to_dlc_result;
  }
#endif  // QNN_SYSTEM_DLC_API_ENABLED

  static Qnn_ErrorHandle_t ErrorGetMessage(Qnn_ErrorHandle_t /*error*/, const char** message) {
    *message = "stub error";
    return QNN_SUCCESS;
  }

  static QnnUnit_BackendSystemDlcPluginTest* current_;

  // Declared before backend_manager_ so the manager's stored logger pointer stays valid.
  StubApiEnv env_;
  StubBackendManager backend_manager_{env_.api_ptrs, env_.logger};

  int log_sentinel_ = 0;
  int dlc_sentinel_ = 0;
  int sys_ctx_sentinel_ = 0;
  int context_sentinel_ = 0;

  std::deque<std::vector<uint8_t>> records_;
  std::deque<std::vector<QnnSystemContext_GraphInfo_t>> graph_infos_;
  std::deque<QnnHtpSystemContext_GraphBlobInfo_t> blobs_;
  std::deque<QnnSystemContext_BinaryInfo_t> binary_infos_;
  std::vector<QnnSystemDlc_RecordHandle_t> record_handles_;

 protected:
  // Declared last so it is destroyed before the backend manager it points to.
  std::unique_ptr<qnn::QnnBackendSystemDlcPlugin> plugin_;
};

QnnUnit_BackendSystemDlcPluginTest* QnnUnit_BackendSystemDlcPluginTest::current_ = nullptr;

// ---------------------------------------------------------------------------
// Release / destructor — nothing created
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendSystemDlcPluginTest, Release_NothingCreated_ReturnsOkWithoutFreeCalls) {
  ASSERT_TRUE(Plugin().Release().IsOK());
  EXPECT_TRUE(calls.empty());
}

#ifdef QNN_SYSTEM_DLC_API_ENABLED

// ---------------------------------------------------------------------------
// SetupDlc (CreateSystemLog + CreateEmptyDlc)
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlc_Succeeds_CreatesSystemLogThenEmptyDlc) {
  auto status = Plugin().SetupDlc();
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogCreate", "systemDlcCreateWithDestinationDir"}));
  EXPECT_NE(log_create_callback, nullptr);
  // The stub logger caches FATAL severity, which maps to QNN_LOG_LEVEL_ERROR.
  EXPECT_EQ(log_create_level, QNN_LOG_LEVEL_ERROR);
  // SystemDlc must be given the SystemLog handle, not the backend log handle.
  EXPECT_EQ(dlc_create_logger, FakeLogHandle());
  EXPECT_EQ(dlc_create_destination_dir, nullptr);
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlc_CalledTwice_ReusesExistingHandles) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  calls.clear();

  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  EXPECT_TRUE(calls.empty());
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlc_SystemLogCreateMissing_ReturnsError) {
  SysInterface().systemLogCreate = nullptr;

  ExpectErrorContains(Plugin().SetupDlc(), "QnnSystemLog_create");
  EXPECT_TRUE(calls.empty());
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlc_SystemLogCreateError_ReturnsErrorWithoutCreatingDlc) {
  log_create_result = kStubError;

  ExpectErrorContains(Plugin().SetupDlc(), "Failed to create SystemLog");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogCreate"}));
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlc_CreateWithDestinationDirMissing_ReleasesSystemLog) {
  SysInterface().systemDlcCreateWithDestinationDir = nullptr;

  ExpectErrorContains(Plugin().SetupDlc(), "QnnSystemDlc_createWithDestinationDir");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogCreate", "systemLogFree"}));
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlc_CreateWithDestinationDirError_ReleasesSystemLog) {
  dlc_create_result = kStubError;

  ExpectErrorContains(Plugin().SetupDlc(), "Failed to create DLC");
  EXPECT_EQ(calls,
            (std::vector<std::string>{"systemLogCreate", "systemDlcCreateWithDestinationDir", "systemLogFree"}));
}

// Rollback failure is only logged; the original setup error is returned.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlc_CreateErrorAndRollbackError_ReturnsOriginalError) {
  dlc_create_result = kStubError;
  log_free_result = kStubError;

  ExpectErrorContains(Plugin().SetupDlc(), "Failed to create DLC");
  EXPECT_EQ(calls,
            (std::vector<std::string>{"systemLogCreate", "systemDlcCreateWithDestinationDir", "systemLogFree"}));
}

// SetupDlc after Release builds fresh handles.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlc_AfterRelease_RecreatesHandles) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  ASSERT_TRUE(Plugin().Release().IsOK());
  calls.clear();

  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogCreate", "systemDlcCreateWithDestinationDir"}));
}

// ---------------------------------------------------------------------------
// SetupDlcFromBinary (CreateSystemLog + CreateDlcFromBinary)
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlcFromBinary_Succeeds_PassesBufferAndSystemLog) {
  const uint8_t buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  auto status = Plugin().SetupDlcFromBinary(buffer, sizeof(buffer));
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogCreate", "systemDlcCreateFromBinary"}));
  EXPECT_EQ(dlc_create_logger, FakeLogHandle());
  EXPECT_EQ(dlc_create_buffer, buffer);
  EXPECT_EQ(dlc_create_buffer_size, sizeof(buffer));
}

// A second from-binary setup is rejected, and the rollback tears down the DLC and
// SystemLog created by the first call.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlcFromBinary_CalledTwice_ReturnsErrorAndReleases) {
  const uint8_t buffer[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(buffer, sizeof(buffer)).IsOK());
  calls.clear();

  ExpectErrorContains(Plugin().SetupDlcFromBinary(buffer, sizeof(buffer)), "DLC created already");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemDlcFree", "systemLogFree"}));
  EXPECT_EQ(dlc_freed_handle, FakeDlcHandle());
}

// An empty DLC is reused by SetupDlc, but cannot be replaced by a from-binary DLC.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlcFromBinary_AfterSetupDlc_ReturnsError) {
  const uint8_t buffer[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());

  ExpectErrorContains(Plugin().SetupDlcFromBinary(buffer, sizeof(buffer)), "DLC created already");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlcFromBinary_CreateFromBinaryMissing_ReleasesSystemLog) {
  const uint8_t buffer[4] = {0};
  SysInterface().systemDlcCreateFromBinary = nullptr;

  ExpectErrorContains(Plugin().SetupDlcFromBinary(buffer, sizeof(buffer)), "QnnSystemDlc_createFromBinary");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogCreate", "systemLogFree"}));
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlcFromBinary_CreateFromBinaryError_ReleasesSystemLog) {
  const uint8_t buffer[4] = {0};
  dlc_create_result = kStubError;

  ExpectErrorContains(Plugin().SetupDlcFromBinary(buffer, sizeof(buffer)), "Failed to create DLC from binary");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogCreate", "systemDlcCreateFromBinary", "systemLogFree"}));
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlcFromBinary_SystemLogCreateError_DoesNotCreateDlc) {
  const uint8_t buffer[4] = {0};
  log_create_result = kStubError;

  ExpectErrorContains(Plugin().SetupDlcFromBinary(buffer, sizeof(buffer)), "Failed to create SystemLog");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogCreate"}));
}

// ---------------------------------------------------------------------------
// Release / destructor (ReleaseDlc + ReleaseSystemLog)
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendSystemDlcPluginTest, Release_AfterSetup_FreesDlcThenSystemLog) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  calls.clear();

  ASSERT_TRUE(Plugin().Release().IsOK());
  EXPECT_EQ(calls, (std::vector<std::string>{"systemDlcFree", "systemLogFree"}));
  EXPECT_EQ(dlc_freed_handle, FakeDlcHandle());
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, Release_CalledTwice_SecondCallIsNoOp) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  ASSERT_TRUE(Plugin().Release().IsOK());
  calls.clear();

  ASSERT_TRUE(Plugin().Release().IsOK());
  EXPECT_TRUE(calls.empty());
}

// A DLC release failure is reported, but the SystemLog is still released.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, Release_DlcFreeMissing_ReturnsErrorAndStillFreesSystemLog) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  calls.clear();
  SysInterface().systemDlcFree = nullptr;

  ExpectErrorContains(Plugin().Release(), "Failed to release");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogFree"}));
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, Release_DlcFreeError_ReturnsErrorAndStillFreesSystemLog) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  calls.clear();
  dlc_free_result = kStubError;

  ExpectErrorContains(Plugin().Release(), "Failed to release");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemDlcFree", "systemLogFree"}));
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, Release_SystemLogFreeMissing_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  calls.clear();
  SysInterface().systemLogFree = nullptr;

  ExpectErrorContains(Plugin().Release(), "Failed to release");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemDlcFree"}));
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, Release_SystemLogFreeError_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  calls.clear();
  log_free_result = kStubError;

  ExpectErrorContains(Plugin().Release(), "Failed to release");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemDlcFree", "systemLogFree"}));
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, Destructor_AfterSetup_ReleasesHandles) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  calls.clear();

  plugin_.reset();
  EXPECT_EQ(calls, (std::vector<std::string>{"systemDlcFree", "systemLogFree"}));
}

// A failing release in the destructor is only logged — no throw / crash.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, Destructor_ReleaseError_DoesNotCrash) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  dlc_free_result = kStubError;
  log_free_result = kStubError;

  plugin_.reset();
  EXPECT_EQ(plugin_, nullptr);
}

// ---------------------------------------------------------------------------
// AddContextToDlc
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendSystemDlcPluginTest, AddContextToDlc_Succeeds_PassesContextAndDlcHandle) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  Qnn_ContextHandle_t context = FakeContextHandle();

  auto status = Plugin().AddContextToDlc(context);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(add_to_dlc_context, context);
  EXPECT_EQ(add_to_dlc_dlc, FakeDlcHandle());
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, AddContextToDlc_ApiMissing_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  QnnInterface().contextAddToDlc = nullptr;
  Qnn_ContextHandle_t context = FakeContextHandle();

  ExpectErrorContains(Plugin().AddContextToDlc(context), "QnnContext_addToDlc");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, AddContextToDlc_ApiError_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  add_to_dlc_result = kStubError;
  Qnn_ContextHandle_t context = FakeContextHandle();

  ExpectErrorContains(Plugin().AddContextToDlc(context), "Failed to add context to DLC");
}

// ---------------------------------------------------------------------------
// GetDlcBinaryBuffer
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryBuffer_NullOutPointer_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  uint64_t size = 0;

  ExpectErrorContains(Plugin().GetDlcBinaryBuffer(nullptr, size), "Null dlc_buffer");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryBuffer_BeforeSetup_ReturnsError) {
  unsigned char* buffer = nullptr;
  uint64_t size = 0;

  ExpectErrorContains(Plugin().GetDlcBinaryBuffer(&buffer, size), "No QNN DLC");
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryBuffer_GetBinarySizeMissing_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  SysInterface().systemDlcGetBinarySize = nullptr;
  unsigned char* buffer = nullptr;
  uint64_t size = 0;

  ExpectErrorContains(Plugin().GetDlcBinaryBuffer(&buffer, size), "QnnSystemDlc_getBinarySize");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryBuffer_GetBinaryMissing_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  SysInterface().systemDlcGetBinary = nullptr;
  unsigned char* buffer = nullptr;
  uint64_t size = 0;

  ExpectErrorContains(Plugin().GetDlcBinaryBuffer(&buffer, size), "QnnSystemDlc_getBinary API");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryBuffer_GetBinarySizeError_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  get_binary_size_result = kStubError;
  unsigned char* buffer = nullptr;
  uint64_t size = 0;

  ExpectErrorContains(Plugin().GetDlcBinaryBuffer(&buffer, size), "Failed to get QNN DLC binary size");
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryBuffer_GetBinaryError_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  dlc_binary = {1, 2, 3};
  get_binary_result = kStubError;
  unsigned char* buffer = nullptr;
  uint64_t size = 0;

  ExpectErrorContains(Plugin().GetDlcBinaryBuffer(&buffer, size), "Failed to get QNN DLC binary.");
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryBuffer_WrittenSizeExceedsAllocated_ReturnsError) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  dlc_binary = {1, 2, 3};
  get_binary_written_size_override = 4;
  unsigned char* buffer = nullptr;
  uint64_t size = 0;

  ExpectErrorContains(Plugin().GetDlcBinaryBuffer(&buffer, size), "exceeds allocated buffer size");
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(size, 0u);
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryBuffer_Succeeds_ReturnsOwnedCopy) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  dlc_binary = {0xde, 0xad, 0xbe, 0xef, 0x01};
  unsigned char* raw_buffer = nullptr;
  uint64_t size = 0;

  auto status = Plugin().GetDlcBinaryBuffer(&raw_buffer, size);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  std::unique_ptr<unsigned char[]> buffer(raw_buffer);  // caller owns the buffer
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(size, dlc_binary.size());
  EXPECT_EQ(std::memcmp(buffer.get(), dlc_binary.data(), dlc_binary.size()), 0);
}

// Fewer bytes written than reported is accepted; the written size is returned.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryBuffer_WrittenSizeSmaller_ReturnsWrittenSize) {
  ASSERT_TRUE(Plugin().SetupDlc().IsOK());
  dlc_binary = {1, 2, 3, 4};
  get_binary_written_size_override = 2;
  unsigned char* raw_buffer = nullptr;
  uint64_t size = 0;

  ASSERT_TRUE(Plugin().GetDlcBinaryBuffer(&raw_buffer, size).IsOK());
  std::unique_ptr<unsigned char[]> buffer(raw_buffer);
  EXPECT_EQ(size, 2u);
}

// ---------------------------------------------------------------------------
// GetDlcBinaryInfo (+ GetDlcRecordBuffers with most_optimal_only = true)
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryInfo_BeforeSetup_ReturnsError) {
  Qnn_Version_t blob_version{};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;

  ExpectErrorContains(Plugin().GetDlcBinaryInfo(FakeSysCtxHandle(), blob_version, graph_count, &graphs_info),
                      "Expecting DLC is created from binary first");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryInfo_GetRecordsByTypeMissing_ReturnsError) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  SysInterface().systemDlcGetRecordsByType = nullptr;
  Qnn_Version_t blob_version{};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;

  ExpectErrorContains(Plugin().GetDlcBinaryInfo(FakeSysCtxHandle(), blob_version, graph_count, &graphs_info),
                      "QnnSystemDlc_getRecordsByType");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryInfo_ReadRecordDataMissing_ReturnsError) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  SysInterface().systemDlcReadRecordDataMemoryMapped = nullptr;
  Qnn_Version_t blob_version{};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;

  ExpectErrorContains(Plugin().GetDlcBinaryInfo(FakeSysCtxHandle(), blob_version, graph_count, &graphs_info),
                      "QnnSystemDlc_readRecordDataMemoryMapped");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryInfo_GetRecordsByTypeError_ReturnsError) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  get_records_result = kStubError;
  Qnn_Version_t blob_version{};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;

  ExpectErrorContains(Plugin().GetDlcBinaryInfo(FakeSysCtxHandle(), blob_version, graph_count, &graphs_info),
                      "Failed to get record from QNN DLC by type");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryInfo_NoRecords_ReturnsError) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  Qnn_Version_t blob_version{};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;

  ExpectErrorContains(Plugin().GetDlcBinaryInfo(FakeSysCtxHandle(), blob_version, graph_count, &graphs_info),
                      "Expecting at least one record handle");
  EXPECT_TRUE(get_binary_info_buffers.empty());
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryInfo_ReadRecordDataError_ReturnsError) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  AddRecord({GraphSpec{}});
  read_record_result = kStubError;
  Qnn_Version_t blob_version{};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;

  ExpectErrorContains(Plugin().GetDlcBinaryInfo(FakeSysCtxHandle(), blob_version, graph_count, &graphs_info),
                      "Failed to read record data");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryInfo_GetBinaryInfoError_ReturnsError) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  AddRecord({GraphSpec{}});
  get_binary_info_result = kStubError;
  Qnn_Version_t blob_version{};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;

  ExpectErrorContains(Plugin().GetDlcBinaryInfo(FakeSysCtxHandle(), blob_version, graph_count, &graphs_info),
                      "Failed to get context binary info");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcBinaryInfo_Succeeds_UsesMostOptimalHtpRecord) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  AddRecord({GraphSpec{}, GraphSpec{}}, Qnn_Version_t{3, 2, 1});
  Qnn_Version_t blob_version{};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;

  auto status = Plugin().GetDlcBinaryInfo(FakeSysCtxHandle(), blob_version, graph_count, &graphs_info);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  EXPECT_EQ(get_records_type, QNN_SYSTEM_DLC_RECORD_PREFIX_HTP_CACHE_RECORD);
  EXPECT_EQ(get_records_most_optimal, 1u);
  ASSERT_EQ(get_binary_info_buffers.size(), 1u);
  EXPECT_EQ(get_binary_info_buffers[0], RecordData(0));
  EXPECT_EQ(get_binary_info_sys_ctx[0], FakeSysCtxHandle());
  EXPECT_EQ(graph_count, 2u);
  EXPECT_NE(graphs_info, nullptr);
  EXPECT_EQ(blob_version.major, 3u);
  EXPECT_EQ(blob_version.minor, 2u);
  EXPECT_EQ(blob_version.patch, 1u);
}

// ---------------------------------------------------------------------------
// GetDlcMaxSpillFillBufferSize (+ GetDlcRecordBuffers with most_optimal_only = false)
// ---------------------------------------------------------------------------

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcMaxSpillFillBufferSize_BeforeSetup_ReturnsError) {
  uint64_t max_size = 0;

  ExpectErrorContains(Plugin().GetDlcMaxSpillFillBufferSize(max_size), "No DLC to get max spill-fill buffer size");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcMaxSpillFillBufferSize_RecordBuffersFail_ReturnsError) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  get_records_result = kStubError;
  uint64_t max_size = 0;

  ExpectErrorContains(Plugin().GetDlcMaxSpillFillBufferSize(max_size), "Failed to get record from QNN DLC by type");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcMaxSpillFillBufferSize_SystemContextCreateError_ReturnsError) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  AddRecord({GraphSpec{}});
  sys_ctx_create_result = kStubError;
  uint64_t max_size = 0;

  ExpectErrorContains(Plugin().GetDlcMaxSpillFillBufferSize(max_size), "without system context");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcMaxSpillFillBufferSize_GetBinaryInfoError_ReturnsError) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  AddRecord({GraphSpec{}});
  get_binary_info_result = kStubError;
  uint64_t max_size = 0;

  ExpectErrorContains(Plugin().GetDlcMaxSpillFillBufferSize(max_size), "Failed to get context binary info");
  // The system context created for the query is still freed.
  EXPECT_EQ(calls.back(), "systemContextFree");
}

// Max is taken across every record and every V3 graph with an HTP V1 blob; V1/V2
// graphs, unknown graph versions, and unknown HTP blob versions are skipped.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcMaxSpillFillBufferSize_MultipleRecords_ReturnsMaximum) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());

  GraphSpec v3_small;
  v3_small.spill_fill_buffer_size = 100;
  GraphSpec v1_graph;
  v1_graph.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;
  GraphSpec v2_graph;
  v2_graph.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2;
  AddRecord({v3_small, v1_graph, v2_graph});

  GraphSpec v3_large;
  v3_large.spill_fill_buffer_size = 300;
  GraphSpec v3_mid;
  v3_mid.spill_fill_buffer_size = 200;
  GraphSpec v3_unknown_blob;
  v3_unknown_blob.blob_version = QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_UNDEFINED;
  v3_unknown_blob.spill_fill_buffer_size = 1000;  // must be ignored
  GraphSpec unknown_graph;
  unknown_graph.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_UNDEFINED;
  AddRecord({v3_mid, v3_large, v3_unknown_blob, unknown_graph});

  uint64_t max_size = 0;
  auto status = Plugin().GetDlcMaxSpillFillBufferSize(max_size);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  EXPECT_EQ(max_size, 300u);
  EXPECT_EQ(get_records_type, QNN_SYSTEM_DLC_RECORD_PREFIX_HTP_CACHE_RECORD);
  EXPECT_EQ(get_records_most_optimal, 0u);
  ASSERT_EQ(get_binary_info_buffers.size(), 2u);
  EXPECT_EQ(get_binary_info_buffers[0], RecordData(0));
  EXPECT_EQ(get_binary_info_buffers[1], RecordData(1));
  EXPECT_EQ(get_binary_info_sys_ctx[0], FakeSysCtxHandle());
  EXPECT_EQ(calls.back(), "systemContextFree");
}

// The out value is reset even when no graph carries spill-fill information.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, GetDlcMaxSpillFillBufferSize_NoHtpGraphInfo_ReturnsZero) {
  const uint8_t dlc[4] = {0};
  ASSERT_TRUE(Plugin().SetupDlcFromBinary(dlc, sizeof(dlc)).IsOK());
  GraphSpec v1_graph;
  v1_graph.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;
  AddRecord({v1_graph});

  uint64_t max_size = 12345;
  ASSERT_TRUE(Plugin().GetDlcMaxSpillFillBufferSize(max_size).IsOK());
  EXPECT_EQ(max_size, 0u);
}

#else  // !QNN_SYSTEM_DLC_API_ENABLED

// ---------------------------------------------------------------------------
// Pre-DLC SDKs: every DLC entry point reports unsupported.
// ---------------------------------------------------------------------------

// SystemLog is still created, then rolled back when empty-DLC creation is rejected.
TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlc_DlcApiUnavailable_ReturnsErrorAndReleasesSystemLog) {
  ExpectErrorContains(Plugin().SetupDlc(), "only supported in QAIRT 2.48+");
  EXPECT_EQ(calls, (std::vector<std::string>{"systemLogCreate", "systemLogFree"}));
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, SetupDlcFromBinary_DlcApiUnavailable_ReturnsError) {
  const uint8_t buffer[4] = {0};
  ExpectErrorContains(Plugin().SetupDlcFromBinary(buffer, sizeof(buffer)), "only supported in QAIRT 2.48+");
}

TEST_F(QnnUnit_BackendSystemDlcPluginTest, DlcQueries_DlcApiUnavailable_ReturnError) {
  Qnn_ContextHandle_t context = FakeContextHandle();
  ExpectErrorContains(Plugin().AddContextToDlc(context), "only supported in QAIRT 2.48+");

  unsigned char* buffer = nullptr;
  uint64_t size = 0;
  ExpectErrorContains(Plugin().GetDlcBinaryBuffer(&buffer, size), "only supported in QAIRT 2.48+");

  Qnn_Version_t blob_version{};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
  ExpectErrorContains(Plugin().GetDlcBinaryInfo(FakeSysCtxHandle(), blob_version, graph_count, &graphs_info),
                      "only supported in QAIRT 2.48+");

  uint64_t max_size = 0;
  ExpectErrorContains(Plugin().GetDlcMaxSpillFillBufferSize(max_size), "only supported in QAIRT 2.48+");
}

#endif  // QNN_SYSTEM_DLC_API_ENABLED

// ===========================================================================
// Real-HTP-backend tests
//
// Load libQnnHtp.so + libQnnSystem.so and exercise the plugin against the real
// QNN System API (SystemLog / SystemDlc). No ORT session is created.
// ===========================================================================

#ifdef QNN_SYSTEM_DLC_API_ENABLED

class QnnUnit_BackendSystemDlcPluginHtpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!htp_.IsValid()) {
      GTEST_SKIP() << "QNN HTP backend (libQnnHtp.so / libQnnSystem.so) is not available! Skipping test.";
    }
  }

  QnnRealHtpBackendManagerContext htp_{/*need_load_system_lib=*/true};
};

TEST_F(QnnUnit_BackendSystemDlcPluginHtpTest, SetupDlc_HTP_SucceedsAndReleases) {
  qnn::QnnBackendSystemDlcPlugin plugin(htp_.Manager());

  auto status = plugin.SetupDlc();
  ASSERT_TRUE(status.IsOK()) << "SetupDlc failed: " << status.GetErrorMessage();
  // Idempotent while the DLC exists.
  ASSERT_TRUE(plugin.SetupDlc().IsOK());

  status = plugin.Release();
  EXPECT_TRUE(status.IsOK()) << "Release failed: " << status.GetErrorMessage();
}

// Garbage bytes are rejected by the real SystemDlc API without crashing, and the
// plugin rolls back to a clean state.
TEST_F(QnnUnit_BackendSystemDlcPluginHtpTest, SetupDlcFromBinary_HTP_InvalidBuffer_ReturnsError) {
  qnn::QnnBackendSystemDlcPlugin plugin(htp_.Manager());
  const uint8_t garbage[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                               0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};

  ExpectErrorContains(plugin.SetupDlcFromBinary(garbage, sizeof(garbage)), "Failed to create DLC from binary");

  uint64_t max_size = 0;
  ExpectErrorContains(plugin.GetDlcMaxSpillFillBufferSize(max_size), "No DLC");
  EXPECT_TRUE(plugin.Release().IsOK());
}

#endif  // QNN_SYSTEM_DLC_API_ENABLED

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
