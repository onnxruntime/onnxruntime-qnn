// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#ifdef USE_QAIRT_API
#include "core/providers/qnn/builder/qairt_backend_manager.h"
#include "core/providers/qnn/builder/qairt_type_convert.h"
#include "QairtCpp/QairtApi.hpp"
#include "QairtCpp/QairtBackend.hpp"
#include "QairtCpp/QairtContext.hpp"
#include "QairtCpp/QairtLog.hpp"

namespace onnxruntime::qnn {
QairtBackendManager::QairtBackendManager(Config config) : config_(std::move(config)) {}
QairtBackendManager::~QairtBackendManager() = default;

Ort::Status QairtBackendManager::Setup() {
  if (backend_) return Ort::Status();
  try {
    auto api = std::make_unique<qairt::Api>(config_.backend_lib_path.c_str());
    auto log = api->makeShared<qairt::Log>(static_cast<QairtLog_CallbackFn_t>(nullptr), QAIRT_LOG_LEVEL_WARN);
    auto backend = api->makeShared<qairt::Backend>(*log);
    api_ = std::move(api);
    log_ = std::move(log);
    backend_ = std::move(backend);
    return Ort::Status();
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(("QAIRT backend setup failed for " + config_.backend_lib_path + ": " + e.what()).c_str());
  }
}
qairt::Api& QairtBackendManager::GetApi() { return *api_; }
qairt::Backend& QairtBackendManager::GetBackend() { return *backend_; }
std::shared_ptr<qairt::Context> QairtBackendManager::GetContext() {
  if (!context_) {
    auto config = api_->make<qairt::ContextConfiguration>();
    context_ = std::make_shared<qairt::Context>(backend_->createContext(config));
  }
  return context_;
}
Ort::Status QairtBackendManager::ValidateOp(const Qnn_OpConfig_t& op) {
  try {
    qairt::OpConfig converted;
    RETURN_IF_ERROR(qairt_convert::FromQnnOpConfig(op, *api_, converted));
    backend_->validateOpConfig(converted);
    return Ort::Status();
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(e.what());
  }
}
Ort::Status QairtBackendManager::ExportContext(std::vector<unsigned char>& buffer) {
  try {
    RETURN_IF(!context_, "QAIRT context has not been created.");
    buffer.resize(context_->getBinarySize());
    if (!buffer.empty()) buffer.resize(context_->getBinary(buffer.data(), buffer.size()));
    return Ort::Status();
  } catch (const std::exception& e) {
    buffer.clear();
    return MAKE_EP_FAIL(e.what());
  }
}
}  // namespace onnxruntime::qnn
#endif
