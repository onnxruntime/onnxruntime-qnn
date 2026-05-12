// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <stdexcept>
#include <string>
#include <vector>

// Include qnn_test_utils before genie headers so that core/common/safeint.h
// (pulled in transitively) sets up SafeIntDefaultExceptionHandler first. This
// lets ort_api.h (included via genie_backend_manager.h) skip its own handler
// definition and avoid a redefinition conflict.
#include "test/providers/qnn/qnn_test_utils.h"

#include "core/providers/qnn/genie/genie_node.h"
#include "core/providers/qnn/genie/genie_api_loader.h"
#include "core/providers/qnn/genie/genie_backend_manager.h"

namespace onnxruntime {
namespace test {

// ==============================================================================
// GenieNodeStateDeleterTest
//
// Unit tests for GenieNodeStateDeleter. These verify the resource cleanup
// behavior using mock GenieApi function pointers. No hardware is required.
//
// The GenieApi struct is a plain POD of function pointers, so mock functions
// with matching signatures can be assigned directly to its fields for testing.
// ==============================================================================

namespace {

// Tracks which mock free functions were called and in what order.
struct MockFreeCallTracker {
  int node_free_count = 0;
  int log_free_count = 0;
  int config_free_count = 0;
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
  int dlc_free_count = 0;
  int dlc_config_free_count = 0;
#endif
  std::vector<std::string> call_order;

  void Reset() {
    node_free_count = 0;
    log_free_count = 0;
    config_free_count = 0;
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
    dlc_free_count = 0;
    dlc_config_free_count = 0;
#endif
    call_order.clear();
  }
};

// Global tracker — reset by GenieNodeStateDeleterTest::SetUp() before each test.
static MockFreeCallTracker g_mock_tracker;

// Mock free functions that record calls to the global tracker.
// All Genie free functions return Genie_Status_t (int32_t), so mock functions
// return GENIE_STATUS_SUCCESS (0).
//
// Note: In C++, top-level const on function parameters is stripped from the
// function type, so these signatures are compatible with decltype(&GenieNode_free)
// etc., which declare `const GenieNode_Handle_t` parameters.

static Genie_Status_t MockNodeFree(GenieNode_Handle_t) {
  ++g_mock_tracker.node_free_count;
  g_mock_tracker.call_order.push_back("node");
  return GENIE_STATUS_SUCCESS;
}

static Genie_Status_t MockLogFree(GenieLog_Handle_t) {
  ++g_mock_tracker.log_free_count;
  g_mock_tracker.call_order.push_back("log");
  return GENIE_STATUS_SUCCESS;
}

static Genie_Status_t MockNodeConfigFree(GenieNodeConfig_Handle_t) {
  ++g_mock_tracker.config_free_count;
  g_mock_tracker.call_order.push_back("config");
  return GENIE_STATUS_SUCCESS;
}

#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
static Genie_Status_t MockDlcFree(GenieDlc_Handle_t) {
  ++g_mock_tracker.dlc_free_count;
  g_mock_tracker.call_order.push_back("dlc");
  return GENIE_STATUS_SUCCESS;
}

static Genie_Status_t MockDlcConfigFree(GenieDlcConfig_Handle_t) {
  ++g_mock_tracker.dlc_config_free_count;
  g_mock_tracker.call_order.push_back("dlc_config");
  return GENIE_STATUS_SUCCESS;
}
#endif

// Sentinel handle values: non-null but never dereferenced by mock free functions.
const GenieNode_Handle_t kFakeNode = reinterpret_cast<GenieNode_Handle_t>(uintptr_t{1});
const GenieLog_Handle_t kFakeLog = reinterpret_cast<GenieLog_Handle_t>(uintptr_t{2});
const GenieNodeConfig_Handle_t kFakeConfig = reinterpret_cast<GenieNodeConfig_Handle_t>(uintptr_t{3});
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
const GenieDlc_Handle_t kFakeDlc = reinterpret_cast<GenieDlc_Handle_t>(uintptr_t{4});
const GenieDlcConfig_Handle_t kFakeDlcCfg = reinterpret_cast<GenieDlcConfig_Handle_t>(uintptr_t{5});
#endif

// Returns a GenieApi with all mock free functions assigned.
GenieApi MakeMockGenieApi() {
  GenieApi api{};  // zero-initialize all function pointers to null
  api.Node_free = MockNodeFree;
  api.Log_free = MockLogFree;
  api.NodeConfig_free = MockNodeConfigFree;
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
  api.Dlc_free = MockDlcFree;
  api.DlcConfig_free = MockDlcConfigFree;
#endif
  return api;
}

}  // namespace

// Resets g_mock_tracker before each test so no manual Reset() calls are needed in test bodies.
class GenieNodeStateDeleterTest : public GenieBackendTests {
 protected:
  void SetUp() override {
    GenieBackendTests::SetUp();
    g_mock_tracker.Reset();
  }
};

// No extra setup needed; GenieApiLoader construction behavior is self-contained.
class GenieApiLoaderTest : public GenieBackendTests {};

// "Unit" suffix distinguishes from the integration fixture in genie_integration_test.cc.
class GenieBackendManagerUnitTest : public GenieBackendTests {};

// Passing nullptr to the deleter should not crash.
TEST_F(GenieNodeStateDeleterTest, NullState_DoesNothing) {
  GenieNodeStateDeleter deleter;
  EXPECT_NO_FATAL_FAILURE(deleter(nullptr));
}

// A state with a null api pointer should be deleted without invoking any free functions.
TEST_F(GenieNodeStateDeleterTest, NullApi_NoFreeFunctionsCalled) {
  auto* st = new GenieNodeState();
  st->api = nullptr;
  // Set non-null handles to ensure they would be freed if api were valid.
  st->node = kFakeNode;
  st->genie_logger = kFakeLog;
  st->config = kFakeConfig;

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.node_free_count, 0);
  EXPECT_EQ(g_mock_tracker.log_free_count, 0);
  EXPECT_EQ(g_mock_tracker.config_free_count, 0);
}

// When all handles in the state are null, no free functions should be called.
TEST_F(GenieNodeStateDeleterTest, AllHandlesNull_NoFreeFunctionsCalled) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  // All handles remain null (default-initialized by GenieNodeState constructor).

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.node_free_count, 0);
  EXPECT_EQ(g_mock_tracker.log_free_count, 0);
  EXPECT_EQ(g_mock_tracker.config_free_count, 0);
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
  EXPECT_EQ(g_mock_tracker.dlc_free_count, 0);
  EXPECT_EQ(g_mock_tracker.dlc_config_free_count, 0);
#endif
}

// When only the node handle is set, only Node_free should be called.
TEST_F(GenieNodeStateDeleterTest, NodeSet_NodeFreeCalled) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->node = kFakeNode;

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.node_free_count, 1);
  EXPECT_EQ(g_mock_tracker.log_free_count, 0);
  EXPECT_EQ(g_mock_tracker.config_free_count, 0);
}

// When only the logger handle is set, only Log_free should be called.
TEST_F(GenieNodeStateDeleterTest, LoggerSet_LogFreeCalled) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->genie_logger = kFakeLog;

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.node_free_count, 0);
  EXPECT_EQ(g_mock_tracker.log_free_count, 1);
  EXPECT_EQ(g_mock_tracker.config_free_count, 0);
}

// When only the config handle is set, only NodeConfig_free should be called.
TEST_F(GenieNodeStateDeleterTest, ConfigSet_NodeConfigFreeCalled) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->config = kFakeConfig;

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.node_free_count, 0);
  EXPECT_EQ(g_mock_tracker.log_free_count, 0);
  EXPECT_EQ(g_mock_tracker.config_free_count, 1);
}

// When api->Log_free is null, the logger handle should not be freed.
// genie_node.cc guards: if (st->genie_logger && api->Log_free) api->Log_free(...)
TEST_F(GenieNodeStateDeleterTest, LogFreeNullInApi_LoggerSkipped) {
  GenieApi mock_api = MakeMockGenieApi();
  mock_api.Log_free = nullptr;  // Explicitly null — should be guarded by the deleter.
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->genie_logger = kFakeLog;

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.log_free_count, 0);
}

// When api->Node_free is null, the node handle should not be freed.
// genie_node.cc guards: if (st->node && api->Node_free) api->Node_free(...)
TEST_F(GenieNodeStateDeleterTest, NodeFreeNullInApi_NodeNotFreed) {
  GenieApi mock_api = MakeMockGenieApi();
  mock_api.Node_free = nullptr;  // Explicitly null — should be guarded by the deleter.
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->node = kFakeNode;

  GenieNodeStateDeleter deleter;
  EXPECT_NO_FATAL_FAILURE(deleter(st));

  EXPECT_EQ(g_mock_tracker.node_free_count, 0);
}

// When api->NodeConfig_free is null, the config handle should not be freed.
// genie_node.cc guards: if (st->config && api->NodeConfig_free) api->NodeConfig_free(...)
TEST_F(GenieNodeStateDeleterTest, ConfigFreeNullInApi_ConfigNotFreed) {
  GenieApi mock_api = MakeMockGenieApi();
  mock_api.NodeConfig_free = nullptr;  // Explicitly null — should be guarded by the deleter.
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->config = kFakeConfig;

  GenieNodeStateDeleter deleter;
  EXPECT_NO_FATAL_FAILURE(deleter(st));

  EXPECT_EQ(g_mock_tracker.config_free_count, 0);
}

// When all core handles (node, logger, config) are set, all three free functions
// should each be called exactly once.
TEST_F(GenieNodeStateDeleterTest, AllCoreHandlesSet_CoreFreesCalled) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->node = kFakeNode;
  st->genie_logger = kFakeLog;
  st->config = kFakeConfig;

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.node_free_count, 1);
  EXPECT_EQ(g_mock_tracker.log_free_count, 1);
  EXPECT_EQ(g_mock_tracker.config_free_count, 1);
}

// Core handles must be freed in the order documented in genie_node.cc:
// node → log → config
TEST_F(GenieNodeStateDeleterTest, AllCoreHandlesSet_FreeOrderIsNodeLogConfig) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->node = kFakeNode;
  st->genie_logger = kFakeLog;
  st->config = kFakeConfig;

  GenieNodeStateDeleter deleter;
  deleter(st);

  ASSERT_EQ(g_mock_tracker.call_order.size(), 3u);
  EXPECT_EQ(g_mock_tracker.call_order[0], "node");
  EXPECT_EQ(g_mock_tracker.call_order[1], "log");
  EXPECT_EQ(g_mock_tracker.call_order[2], "config");
}

#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
// The DLC-related tests below require Genie API version >= 1.17 (QAIRT >= 2.45.0),
// which is when GenieDlc.h and DLC-related GenieNodeState fields were introduced.

// When only the DLC handle is set, only Dlc_free should be called.
TEST_F(GenieNodeStateDeleterTest, DlcHandleSet_DlcFreeCalled) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->dlc_handle = kFakeDlc;

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.dlc_free_count, 1);
  EXPECT_EQ(g_mock_tracker.dlc_config_free_count, 0);
}

// When only the DLC config handle is set, only DlcConfig_free should be called.
TEST_F(GenieNodeStateDeleterTest, DlcConfigSet_DlcConfigFreeCalled) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->dlc_config_handle = kFakeDlcCfg;

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.dlc_free_count, 0);
  EXPECT_EQ(g_mock_tracker.dlc_config_free_count, 1);
}

// When all five handles are set, all five free functions should each be called once.
TEST_F(GenieNodeStateDeleterTest, AllHandlesSet_AllFiveHandlesFreesCalled) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->node = kFakeNode;
  st->genie_logger = kFakeLog;
  st->config = kFakeConfig;
  st->dlc_handle = kFakeDlc;
  st->dlc_config_handle = kFakeDlcCfg;

  GenieNodeStateDeleter deleter;
  deleter(st);

  EXPECT_EQ(g_mock_tracker.node_free_count, 1);
  EXPECT_EQ(g_mock_tracker.log_free_count, 1);
  EXPECT_EQ(g_mock_tracker.config_free_count, 1);
  EXPECT_EQ(g_mock_tracker.dlc_free_count, 1);
  EXPECT_EQ(g_mock_tracker.dlc_config_free_count, 1);
}

// All five handles must be freed in the order documented in genie_node.cc:
// node → log → config → dlc → dlc_config
TEST_F(GenieNodeStateDeleterTest, AllHandlesSet_FreeOrderIsCorrect) {
  GenieApi mock_api = MakeMockGenieApi();
  auto* st = new GenieNodeState();
  st->api = &mock_api;
  st->node = kFakeNode;
  st->genie_logger = kFakeLog;
  st->config = kFakeConfig;
  st->dlc_handle = kFakeDlc;
  st->dlc_config_handle = kFakeDlcCfg;

  GenieNodeStateDeleter deleter;
  deleter(st);

  ASSERT_EQ(g_mock_tracker.call_order.size(), 5u);
  EXPECT_EQ(g_mock_tracker.call_order[0], "node");
  EXPECT_EQ(g_mock_tracker.call_order[1], "log");
  EXPECT_EQ(g_mock_tracker.call_order[2], "config");
  EXPECT_EQ(g_mock_tracker.call_order[3], "dlc");
  EXPECT_EQ(g_mock_tracker.call_order[4], "dlc_config");
}

#endif  // GENIE_API_VERSION >= 1.17

// ==============================================================================
// GenieApiLoaderTest
//
// Unit tests for GenieApiLoader construction behavior. Symbol loading is lazy
// (deferred to the first Get() call), so construction with a non-null handle
// is safe regardless of whether the handle points to a real library.
// ==============================================================================

// Constructing with a null handle should throw std::runtime_error.
// See genie_api_loader.cc: throws "GenieApiLoader: Null library handle".
TEST_F(GenieApiLoaderTest, NullHandle_ThrowsRuntimeError) {
  EXPECT_THROW(GenieApiLoader(nullptr), std::runtime_error);
}

// Constructing with a non-null handle should not throw.
// dlsym is not called until Get() is invoked (lazy initialization via std::call_once).
TEST_F(GenieApiLoaderTest, NonNullHandle_ConstructsWithoutThrowing) {
  EXPECT_NO_THROW(GenieApiLoader(reinterpret_cast<void*>(uintptr_t{1})));
}

// ==============================================================================
// GenieBackendManagerUnitTest
//
// Unit tests for GenieBackendManager, verifying initialization state and
// error handling. These tests exercise failure paths that do not require
// the Genie library to be present on the system.
//
// A valid Ort::Logger is required because logging is performed during
// SetupBackend(). We obtain one from the test environment's default
// logging manager (initialized by ortenv_setup() in test_main.cc).
// ==============================================================================

namespace {

// Returns an Ort::Logger suitable for use in unit tests that exercise error paths.
// Uses the null-sink default constructor (same pattern as GenieBackendTests::SetUp()),
// avoiding any dependency on the unexported onnxruntime::GetStackTrace symbol.
Ort::Logger GetTestLogger() {
  return Ort::Logger();
}

}  // namespace

// GetGenieBackendHandle() should return null before SetupBackend() is called.
TEST_F(GenieBackendManagerUnitTest, InitialState_HandleIsNull) {
  Ort::Logger logger = GetTestLogger();
  auto mgr = qnn::GenieBackendManager::Create(
      qnn::GenieBackendManagerConfig{"nonexistent_genie.so"}, logger);
  EXPECT_EQ(mgr->GetGenieBackendHandle(), nullptr);
}

// SetupBackend() with a nonexistent library path should return a failure status.
TEST_F(GenieBackendManagerUnitTest, SetupBackend_InvalidPath_ReturnsError) {
  Ort::Logger logger = GetTestLogger();
  auto mgr = qnn::GenieBackendManager::Create(
      qnn::GenieBackendManagerConfig{"nonexistent_genie.so"}, logger);

  Ort::Status status = mgr->SetupBackend();

  EXPECT_FALSE(status.IsOK());
}

// The error message from a failed SetupBackend() should be non-empty.
TEST_F(GenieBackendManagerUnitTest, SetupBackend_InvalidPath_ErrorMessageNotEmpty) {
  Ort::Logger logger = GetTestLogger();
  auto mgr = qnn::GenieBackendManager::Create(
      qnn::GenieBackendManagerConfig{"nonexistent_genie.so"}, logger);

  Ort::Status status = mgr->SetupBackend();

  ASSERT_FALSE(status.IsOK());
  EXPECT_FALSE(std::string(status.GetErrorMessage()).empty());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
