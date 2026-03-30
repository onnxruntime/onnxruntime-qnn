// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <stdexcept>
#include <vector>

#include "../Genie/GenieCommon.h"
#include "../Genie/GenieLog.h"
#include "../Genie/GenieNode.h"
#include "../Genie/GenieDlc.h"

// GenieApi: holds all resolved function pointers
struct GenieApi {
  decltype(&GenieNodeConfig_createFromJson) NodeConfig_createFromJson;
  decltype(&GenieDlcConfig_create) DlcConfig_create;
  decltype(&GenieDlcConfig_free) DlcConfig_free;
  decltype(&GenieDlc_create) Dlc_create;
  decltype(&GenieDlc_free) Dlc_free;
  decltype(&GenieDlc_getUseCases) Dlc_getUseCases;
  decltype(&GenieNodeConfig_createFromDlc) NodeConfig_createFromDlc;
  decltype(&GenieNode_create) Node_create;
  decltype(&GenieNode_setData) Node_setData;
  decltype(&GenieNode_getData) Node_getData;
  decltype(&GenieNode_execute) Node_execute;
  decltype(&GenieNode_free) Node_free;
  decltype(&GenieNode_reset) Node_reset;
  decltype(&GenieNodeConfig_free) NodeConfig_free;
  decltype(&GenieLog_create) Log_create;
  decltype(&GenieNodeConfig_bindLogger) NodeConfig_bindLogger;
  decltype(&GenieLog_free) Log_free;
};

// GenieApiLoader: resolves and owns symbol table
class GenieApiLoader {
 public:
  explicit GenieApiLoader(void* shared_library_handle);

  // Lazy initialize & return reference
  const GenieApi& Get();

 private:
  void Init();                // loads symbols
  GenieApi api_;              // resolved symbols
  void* handle_;              // dlopen/LoadLibrary handle
  std::once_flag init_flag_;  // ensures Init() runs once
};

// Per-node runtime state used by ONNX Runtime during execution
struct GenieNodeState {
  const GenieApi* api = nullptr;
  GenieNodeConfig_Handle_t config = nullptr;
  GenieNode_Handle_t node = nullptr;
  GenieLog_Handle_t genieLogger = nullptr;
  GenieDlc_Handle_t dlcHandle = nullptr;
  GenieDlcConfig_Handle_t dlcConfigHandle = nullptr;
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  std::mutex mu;
};

struct GenieNodeBuilder {
  const GenieApi* api;
  std::string dlc_path;
  size_t num_inputs;
  size_t num_outputs;
};
