// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <stdexcept>
#include <vector>

#include "../Genie/GenieCommon.h"
#include "../Genie/GenieLog.h"
#include "../Genie/GenieNode.h"
// Genie DLC support requires Genie API version in GenieCommon.h to be >= 1.17 (QAIRT >= 2.45.0)
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
#include "../Genie/GenieDlc.h"
#endif

// GenieApi: holds all resolved function pointers
struct GenieApi {
  decltype(&GenieNodeConfig_createFromJson) NodeConfig_createFromJson;
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
  decltype(&GenieDlcConfig_create) DlcConfig_create;
  decltype(&GenieDlcConfig_free) DlcConfig_free;
  decltype(&GenieDlc_create) Dlc_create;
  decltype(&GenieDlc_free) Dlc_free;
  decltype(&GenieDlc_getUseCases) Dlc_getUseCases;
  decltype(&GenieNodeConfig_createFromDlc) NodeConfig_createFromDlc;
  decltype(&GenieNode_getData) Node_getData;
  decltype(&GenieNode_execute) Node_execute;
  decltype(&GenieNode_reset) Node_reset;
#endif
  decltype(&GenieNode_create) Node_create;
  decltype(&GenieNode_setData) Node_setData;
  decltype(&GenieNode_free) Node_free;
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
