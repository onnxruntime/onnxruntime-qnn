// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/providers/qnn/genie/genie_api_loader.h"

// Per-node runtime state used by ONNX Runtime during execution
struct GenieNodeState {
  const GenieApi* api = nullptr;
  GenieNodeConfig_Handle_t config = nullptr;
  GenieNode_Handle_t node = nullptr;
  GenieLog_Handle_t genie_logger = nullptr;
  GenieDlc_Handle_t dlc_handle = nullptr;
  GenieDlcConfig_Handle_t dlc_config_handle = nullptr;
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  std::mutex mu;
};

// Custom deleter for GenieNodeState
struct GenieNodeStateDeleter {
  void operator()(GenieNodeState* st);
};

struct GenieNodeBuilder {
  const GenieApi* api = nullptr;
  std::string dlc_path;
  size_t num_inputs = 0;
  size_t num_outputs = 0;
};
