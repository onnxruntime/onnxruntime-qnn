// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
#ifdef USE_QAIRT_API
#include <stdexcept>
#include "core/providers/qnn/runtime/qairt/qairt_backend_manager.h"
#include "core/providers/qnn/runtime/qairt/qairt_backend_configuration.h"
#include "core/providers/qnn/runtime/qairt/qairt_graph_emitter.h"
#include "core/providers/qnn/runtime/qairt/qairt_type_convert.h"
#include "QairtCpp/QairtApi.hpp"
#include "QairtCpp/QairtBackend.hpp"
#include "QairtCpp/QairtContext.hpp"
#include "QairtCpp/QairtLog.hpp"

namespace onnxruntime::qnn {
QairtBackendManager::QairtBackendManager(Config config) : config_(std::move(config)) {}
QairtBackendManager::~QairtBackendManager() = default;

std::unique_ptr<IGraphEmitter> QairtBackendManager::CreateGraphEmitter() {
  auto owner = shared_from_this();
  return std::make_unique<QairtGraphEmitter>(QairtGraphEmitter::Dependencies{
      GetApi(),
      [owner] { return owner->GetContext(); },
      [owner] { return owner->CreateGraphConfiguration(); },
      [owner](qairt::Graph& graph) { return owner->ConfigureRetrievedGraph(graph); },
      [owner](const Qnn_OpConfig_t& op) { return owner->ValidateOp(op); }});
}

Ort::Status QairtBackendManager::Setup(SetupFailure* failure) {
  std::lock_guard<std::mutex> lock(initialization_mutex_);
  if (failure != nullptr) {
    *failure = SetupFailure::None;
  }
  if (qairt_backend_) {
    return Ort::Status();
  }

  std::unique_ptr<qairt::Api> api;
  try {
    api = std::make_unique<qairt::Api>(config_.backend_lib_path.c_str());
  } catch (const std::exception& e) {
    if (failure != nullptr) {
      *failure = SetupFailure::LibraryLoad;
    }
    return MAKE_EP_FAIL(("Unable to load QAIRT backend " + config_.backend_lib_path + ": " + e.what()).c_str());
  }

  try {
    auto log = api->makeShared<qairt::Log>(static_cast<QairtLog_CallbackFn_t>(nullptr), QAIRT_LOG_LEVEL_WARN);
    std::shared_ptr<qairt::Backend> backend;
    if (config_.configuration && config_.configuration->backend) {
      auto backend_config = config_.configuration->backend(*api);
      backend = api->makeShared<qairt::Backend>(*log, backend_config);
    } else {
      backend = api->makeShared<qairt::Backend>(*log);
    }
    std::shared_ptr<qairt::Device> device;
    if (config_.configuration && config_.configuration->device) {
      auto device_config = config_.configuration->device(*api);
      device = api->makeShared<qairt::Device>(*log, device_config);
    }
    qairt_api_ = std::move(api);
    qairt_log_ = std::move(log);
    qairt_backend_ = std::move(backend);
    qairt_device_ = std::move(device);
    return Ort::Status();
  } catch (const std::exception& e) {
    if (failure != nullptr) {
      *failure = SetupFailure::BackendInitialization;
    }
    return MAKE_EP_FAIL(("Failed to initialize QAIRT backend " + config_.backend_lib_path + ": " + e.what()).c_str());
  }
}
qairt::Api& QairtBackendManager::GetApi() {
  if (!qairt_api_) {
    throw std::logic_error("QAIRT Setup must complete before API access.");
  }
  return *qairt_api_;
}
qairt::Backend& QairtBackendManager::GetBackend() {
  if (!qairt_backend_) {
    throw std::logic_error("QAIRT Setup must complete before backend access.");
  }
  return *qairt_backend_;
}
std::shared_ptr<qairt::Context> QairtBackendManager::GetContext() {
  std::lock_guard<std::mutex> lock(initialization_mutex_);
  if (!qairt_backend_) {
    throw std::logic_error("QAIRT Setup must complete before context creation.");
  }
  if (!qairt_context_) {
    auto config = CreateContextConfiguration();
    const auto device = qairt_device_ ? qairt::ApiTypeRef<const qairt::Device&>(*qairt_device_)
                                      : qairt::ApiTypeRef<const qairt::Device&>{};
    qairt_context_ = std::make_shared<qairt::Context>(qairt_backend_->createContext(device, config));
  }
  return qairt_context_;
}
qairt::ContextConfiguration QairtBackendManager::CreateContextConfiguration() {
  return config_.configuration && config_.configuration->context
             ? config_.configuration->context(*qairt_api_)
             : qairt_api_->make<qairt::ContextConfiguration>();
}
qairt::GraphConfiguration QairtBackendManager::CreateGraphConfiguration() {
  return config_.configuration && config_.configuration->graph
             ? config_.configuration->graph(*qairt_api_)
             : qairt_api_->make<qairt::GraphConfiguration>();
}
Ort::Status QairtBackendManager::ConfigureRetrievedGraph(qairt::Graph& graph) {
  return config_.configuration && config_.configuration->after_retrieve
             ? config_.configuration->after_retrieve(graph)
             : Ort::Status();
}
Ort::Status QairtBackendManager::ValidateOp(const Qnn_OpConfig_t& op) {
  try {
    qairt::OpConfig converted;
    RETURN_IF_ERROR(qairt_convert::FromQnnOpConfig(op, *qairt_api_, converted));
    qairt_backend_->validateOpConfig(converted);
    return Ort::Status();
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(e.what());
  }
}
Ort::Status QairtBackendManager::ExportContext(std::vector<unsigned char>& buffer) {
  try {
    RETURN_IF(!qairt_context_, "QAIRT context has not been created.");
    buffer.resize(qairt_context_->getBinarySize());
    if (!buffer.empty()) {
      buffer.resize(qairt_context_->getBinary(buffer.data(), buffer.size()));
    }
    return Ort::Status();
  } catch (const std::exception& e) {
    buffer.clear();
    return MAKE_EP_FAIL(e.what());
  }
}
}  // namespace onnxruntime::qnn
#endif
