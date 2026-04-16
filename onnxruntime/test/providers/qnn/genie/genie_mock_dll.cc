// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Compiled into MockGenie.dll ONLY.
//
// Plain stub implementations of all Genie C API symbols. No gmock dependency.
// All stubs return GENIE_STATUS_SUCCESS and increment a call counter so tests
// can verify which functions were invoked.
//
// The test EXE retrieves the counters via GetProcAddress("GetMockGenieCalls")
// after session creation.

#include "../Genie/GenieCommon.h"
#include "../Genie/GenieDlc.h"
#include "../Genie/GenieLog.h"
#include "../Genie/GenieNode.h"

#include <cstring>
#include <unordered_map>
#include <atomic>
#include <string>
#include <mutex>

// ---------------------------------------------------------------------------
// Platform-portable export macro.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
#  define MOCK_EXPORT __declspec(dllexport)
#else
#  define MOCK_EXPORT __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// Call tracking — one counter per Genie API function.
// Exported so the test EXE can read them after session creation.
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, std::atomic<int>> g_calls;
static std::mutex g_call_mutex;

extern "C" {

MOCK_EXPORT
int GetMockGenieCallCount(const char* api_name) {
  if (!api_name) return 0;
  std::lock_guard<std::mutex> lock(g_call_mutex);
  auto it = g_calls.find(api_name);
  if (it == g_calls.end()) return 0;
  return it->second.load();
}

MOCK_EXPORT
void ResetMockGenieCalls() {
  std::lock_guard<std::mutex> lock(g_call_mutex);
  g_calls.clear();
}

// ---------------------------------------------------------------------------
// Stub implementations — return plausible fake handles and SUCCESS.
// ---------------------------------------------------------------------------

Genie_Status_t GenieNodeConfig_createFromJson(const char* /*json_config*/,
                                              GenieNodeConfig_Handle_t* config_handle) {
  ++g_calls["NodeConfig_createFromJson"];
  if (config_handle) *config_handle = reinterpret_cast<GenieNodeConfig_Handle_t>(0x1000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlcConfig_create(const char* /*dlcSource*/,
                                     const char* /*jsonStr*/,
                                     GenieDlcConfig_Handle_t* configHandle) {
  ++g_calls["DlcConfig_create"];
  if (configHandle) *configHandle = reinterpret_cast<GenieDlcConfig_Handle_t>(0x2000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlcConfig_free(const GenieDlcConfig_Handle_t /*configHandle*/) {
  ++g_calls["DlcConfig_free"];
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlc_create(const GenieDlcConfig_Handle_t /*configHandle*/,
                                GenieDlc_Handle_t* dlcHandle) {
  ++g_calls["Dlc_create"];
  if (dlcHandle) *dlcHandle = reinterpret_cast<GenieDlc_Handle_t>(0x3000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlc_free(const GenieDlc_Handle_t /*dlcHandle*/) {
  ++g_calls["Dlc_free"];
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlc_getUseCases(const GenieDlc_Handle_t /*dlcHandle*/,
                                    Genie_AllocCallback_t /*callback*/,
                                    const char** /*useCases*/) {
  ++g_calls["Dlc_getUseCases"];
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNodeConfig_createFromDlc(GenieDlc_Handle_t /*dlcHandle*/,
                                             const char* /*useCaseName*/,
                                             const char* /*configStr*/,
                                             GenieNodeConfig_Handle_t* configHandle) {
  ++g_calls["NodeConfig_createFromDlc"];
  if (configHandle) *configHandle = reinterpret_cast<GenieNodeConfig_Handle_t>(0x4000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_create(const GenieNodeConfig_Handle_t /*nodeConfigHandle*/,
                                GenieNode_Handle_t* nodeHandle) {
  ++g_calls["Node_create"];
  if (nodeHandle) *nodeHandle = reinterpret_cast<GenieNode_Handle_t>(0x5000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_setData(const GenieNode_Handle_t /*nodeHandle*/,
                                 const GenieNode_IOName_t /*nodeIOName*/,
                                 const void* /*data*/,
                                 const size_t /*dataSize*/,
                                 const char* /*dataConfig*/) {
  ++g_calls["Node_setData"];
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_getData(const GenieNode_Handle_t /*nodeHandle*/,
                                 const GenieNode_IOName_t /*nodeIOName*/,
                                 const char* /*ioConfig*/,
                                 GenieNode_IOCallback_t ioCallback,
                                 const void* userData) {
  ++g_calls["Node_getData"];
  // Invoke the callback with a minimal fake float payload so that ComputeImpl
  // can populate the ORT output tensor without error.
  //
  // Shape encoding: ComputeImpl's OutputCallback parses the "dimensions" array
  // from outputConfig and then inserts a 1 at index 1, so:
  //   outputConfig "[1,1]"  →  parsed [1,1]  →  after insert [1,1,1]
  // This matches the graph's declared output shape {1, 1, 1} (float32).
  if (ioCallback) {
    static const float kFakeOutput = 0.0f;
    ioCallback(&kFakeOutput, sizeof(float),
               "{\"dimensions\": [1,1],\"data-type\": \"float32\"}",
               userData);
  }
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_execute(const GenieNode_Handle_t /*nodeHandle*/,
                                 const char* /*executionConfig*/,
                                 void* /*userData*/) {
  ++g_calls["Node_execute"];
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_free(const GenieNode_Handle_t /*nodeHandle*/) {
  ++g_calls["Node_free"];
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_reset(const GenieNode_Handle_t /*nodeHandle*/) {
  ++g_calls["Node_reset"];
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNodeConfig_free(const GenieNodeConfig_Handle_t /*configHandle*/) {
  ++g_calls["NodeConfig_free"];
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieLog_create(const GenieLogConfig_Handle_t /*configHandle*/,
                               const GenieLog_Callback_t /*callback*/,
                               const GenieLog_Level_t /*logLevel*/,
                               GenieLog_Handle_t* logHandle) {
  ++g_calls["Log_create"];
  if (logHandle) *logHandle = reinterpret_cast<GenieLog_Handle_t>(0x6000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNodeConfig_bindLogger(const GenieNodeConfig_Handle_t /*configHandle*/,
                                          const GenieLog_Handle_t /*logHandle*/) {
  ++g_calls["NodeConfig_bindLogger"];
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieLog_free(GenieLog_Handle_t /*logHandle*/) {
  ++g_calls["Log_free"];
  return GENIE_STATUS_SUCCESS;
}

}  // extern "C"

