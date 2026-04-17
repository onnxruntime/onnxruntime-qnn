// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Compiled into MockGenie.dll ONLY.
//
// Plain stub implementations of all Genie C API symbols. No gmock dependency.
// All stubs return GENIE_STATUS_SUCCESS and increment a call counter so tests
// can verify which functions were invoked.
//
// The test EXE retrieves the counters via GetProcAddress("GetMockGenieCallCount")
// after session creation.

#include "../Genie/GenieCommon.h"
#include "../Genie/GenieDlc.h"
#include "../Genie/GenieLog.h"
#include "../Genie/GenieNode.h"

#include <atomic>
#include <cstring>
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Platform-portable export macro.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
#define MOCK_EXPORT __declspec(dllexport)
#else
#define MOCK_EXPORT __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// Call tracking — one pre-allocated atomic counter per Genie API function.
// Pre-allocation avoids map mutations at call time, so no mutex is needed.
// Exported so the test EXE can read and reset them after session creation.
// ---------------------------------------------------------------------------

static std::atomic<int> g_NodeConfig_createFromJson{0};
static std::atomic<int> g_DlcConfig_create{0};
static std::atomic<int> g_DlcConfig_free{0};
static std::atomic<int> g_Dlc_create{0};
static std::atomic<int> g_Dlc_free{0};
static std::atomic<int> g_Dlc_getUseCases{0};
static std::atomic<int> g_NodeConfig_createFromDlc{0};
static std::atomic<int> g_Node_create{0};
static std::atomic<int> g_Node_setData{0};
static std::atomic<int> g_Node_getData{0};
static std::atomic<int> g_Node_execute{0};
static std::atomic<int> g_Node_free{0};
static std::atomic<int> g_Node_reset{0};
static std::atomic<int> g_NodeConfig_free{0};
static std::atomic<int> g_Log_create{0};
static std::atomic<int> g_NodeConfig_bindLogger{0};
static std::atomic<int> g_Log_free{0};

extern "C" {

MOCK_EXPORT
int GetMockGenieCallCount(const char* api_name) {
  if (!api_name) return 0;
  static const std::unordered_map<std::string, std::atomic<int>*> kCounters = {
      {"NodeConfig_createFromJson", &g_NodeConfig_createFromJson},
      {"DlcConfig_create", &g_DlcConfig_create},
      {"DlcConfig_free", &g_DlcConfig_free},
      {"Dlc_create", &g_Dlc_create},
      {"Dlc_free", &g_Dlc_free},
      {"Dlc_getUseCases", &g_Dlc_getUseCases},
      {"NodeConfig_createFromDlc", &g_NodeConfig_createFromDlc},
      {"Node_create", &g_Node_create},
      {"Node_setData", &g_Node_setData},
      {"Node_getData", &g_Node_getData},
      {"Node_execute", &g_Node_execute},
      {"Node_free", &g_Node_free},
      {"Node_reset", &g_Node_reset},
      {"NodeConfig_free", &g_NodeConfig_free},
      {"Log_create", &g_Log_create},
      {"NodeConfig_bindLogger", &g_NodeConfig_bindLogger},
      {"Log_free", &g_Log_free},
  };
  auto it = kCounters.find(api_name);
  return it != kCounters.end() ? it->second->load() : 0;
}

MOCK_EXPORT
void ResetMockGenieCallCounts() {
  g_NodeConfig_createFromJson = 0;
  g_DlcConfig_create = 0;
  g_DlcConfig_free = 0;
  g_Dlc_create = 0;
  g_Dlc_free = 0;
  g_Dlc_getUseCases = 0;
  g_NodeConfig_createFromDlc = 0;
  g_Node_create = 0;
  g_Node_setData = 0;
  g_Node_getData = 0;
  g_Node_execute = 0;
  g_Node_free = 0;
  g_Node_reset = 0;
  g_NodeConfig_free = 0;
  g_Log_create = 0;
  g_NodeConfig_bindLogger = 0;
  g_Log_free = 0;
}

// ---------------------------------------------------------------------------
// Stub implementations — return plausible fake handles and SUCCESS.
// ---------------------------------------------------------------------------

Genie_Status_t GenieNodeConfig_createFromJson(const char* /*json_config*/,
                                              GenieNodeConfig_Handle_t* config_handle) {
  ++g_NodeConfig_createFromJson;
  if (config_handle) *config_handle = reinterpret_cast<GenieNodeConfig_Handle_t>(0x1000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlcConfig_create(const char* /*dlcSource*/,
                                     const char* /*jsonStr*/,
                                     GenieDlcConfig_Handle_t* configHandle) {
  ++g_DlcConfig_create;
  if (configHandle) *configHandle = reinterpret_cast<GenieDlcConfig_Handle_t>(0x2000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlcConfig_free(const GenieDlcConfig_Handle_t /*configHandle*/) {
  ++g_DlcConfig_free;
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlc_create(const GenieDlcConfig_Handle_t /*configHandle*/,
                               GenieDlc_Handle_t* dlcHandle) {
  ++g_Dlc_create;
  if (dlcHandle) *dlcHandle = reinterpret_cast<GenieDlc_Handle_t>(0x3000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlc_free(const GenieDlc_Handle_t /*dlcHandle*/) {
  ++g_Dlc_free;
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieDlc_getUseCases(const GenieDlc_Handle_t /*dlcHandle*/,
                                    Genie_AllocCallback_t /*callback*/,
                                    const char** /*useCases*/) {
  ++g_Dlc_getUseCases;
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNodeConfig_createFromDlc(GenieDlc_Handle_t /*dlcHandle*/,
                                             const char* /*useCaseName*/,
                                             const char* /*configStr*/,
                                             GenieNodeConfig_Handle_t* configHandle) {
  ++g_NodeConfig_createFromDlc;
  if (configHandle) *configHandle = reinterpret_cast<GenieNodeConfig_Handle_t>(0x4000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_create(const GenieNodeConfig_Handle_t /*nodeConfigHandle*/,
                                GenieNode_Handle_t* nodeHandle) {
  ++g_Node_create;
  if (nodeHandle) *nodeHandle = reinterpret_cast<GenieNode_Handle_t>(0x5000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_setData(const GenieNode_Handle_t /*nodeHandle*/,
                                 const GenieNode_IOName_t /*nodeIOName*/,
                                 const void* /*data*/,
                                 const size_t /*dataSize*/,
                                 const char* /*dataConfig*/) {
  ++g_Node_setData;
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_getData(const GenieNode_Handle_t /*nodeHandle*/,
                                 const GenieNode_IOName_t /*nodeIOName*/,
                                 const char* /*ioConfig*/,
                                 GenieNode_IOCallback_t ioCallback,
                                 const void* userData) {
  ++g_Node_getData;
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
  ++g_Node_execute;
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_free(const GenieNode_Handle_t /*nodeHandle*/) {
  ++g_Node_free;
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNode_reset(const GenieNode_Handle_t /*nodeHandle*/) {
  ++g_Node_reset;
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNodeConfig_free(const GenieNodeConfig_Handle_t /*configHandle*/) {
  ++g_NodeConfig_free;
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieLog_create(const GenieLogConfig_Handle_t /*configHandle*/,
                               const GenieLog_Callback_t /*callback*/,
                               const GenieLog_Level_t /*logLevel*/,
                               GenieLog_Handle_t* logHandle) {
  ++g_Log_create;
  if (logHandle) *logHandle = reinterpret_cast<GenieLog_Handle_t>(0x6000);
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieNodeConfig_bindLogger(const GenieNodeConfig_Handle_t /*configHandle*/,
                                          const GenieLog_Handle_t /*logHandle*/) {
  ++g_NodeConfig_bindLogger;
  return GENIE_STATUS_SUCCESS;
}

Genie_Status_t GenieLog_free(GenieLog_Handle_t /*logHandle*/) {
  ++g_Log_free;
  return GENIE_STATUS_SUCCESS;
}

}  // extern "C"
