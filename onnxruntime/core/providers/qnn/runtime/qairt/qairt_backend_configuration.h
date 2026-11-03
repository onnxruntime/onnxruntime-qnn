// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once
#ifdef USE_QAIRT_API

#include <functional>
#include "QairtCpp/QairtBackend.hpp"
#include "QairtCpp/QairtDevice.hpp"
#include "QairtCpp/QairtContext.hpp"
#include "QairtCpp/QairtGraph.hpp"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime::qnn {

// Optional configuration supplied by accelerator integrations. Empty callbacks
// preserve SDK defaults; SDK-specific configuration stays below the EP seam.
// Captured resources must remain valid until this configuration is destroyed.
struct QairtBackendConfiguration {
  std::function<qairt::BackendConfiguration(const qairt::Api&)> backend;
  std::function<qairt::DeviceConfiguration(const qairt::Api&)> device;
  std::function<qairt::ContextConfiguration(const qairt::Api&)> context;
  std::function<qairt::GraphConfiguration(const qairt::Api&)> graph;
  std::function<Ort::Status(qairt::Graph&)> after_retrieve;
};

}  // namespace onnxruntime::qnn
#endif
