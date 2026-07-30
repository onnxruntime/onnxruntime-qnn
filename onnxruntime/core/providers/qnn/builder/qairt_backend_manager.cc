// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifdef USE_QAIRT_API

#include "core/providers/qnn/builder/qairt_backend_manager.h"

#include "QairtCpp/QairtApi.hpp"

namespace onnxruntime {
namespace qnn {

// Null sentinels for C-handle accessors. Never dereferenced on the QAIRT path.
const QNN_INTERFACE_VER_TYPE QairtBackendManager::null_interface_{};
const Qnn_BackendHandle_t QairtBackendManager::null_backend_handle_ = nullptr;
const Qnn_ContextHandle_t QairtBackendManager::null_context_handle_ = nullptr;
const Qnn_ProfileHandle_t QairtBackendManager::null_profile_handle_ = nullptr;

std::unique_ptr<QairtBackendManager> QairtBackendManager::Create(const Config& config,
                                                                  Ort::Status& status) {
  std::unique_ptr<QairtBackendManager> mgr(new QairtBackendManager());
  status = mgr->Initialize(config);
  if (!status.IsOK()) {
    return nullptr;
  }
  return mgr;
}

Ort::Status QairtBackendManager::Initialize(const Config& config) {
  backend_type_ = config.backend_type;

  try {
    api_ = std::make_unique<qairt::Api>(config.backend_lib_path.c_str());
  } catch (const std::runtime_error& e) {
    return MAKE_EP_FAIL(("QairtBackendManager: failed to load backend library `" +
                         config.backend_lib_path + "`: " + e.what())
                            .c_str());
  }

  try {
    // Migration guide pattern: makeShared for Log and Backend (shared across components).
    log_ = api_->makeShared<qairt::Log>(static_cast<QairtLog_CallbackFn_t>(nullptr),
                                        QAIRT_LOG_LEVEL_WARN);
    backend_ = api_->makeShared<qairt::Backend>(*log_);

    auto context_config = api_->make<qairt::ContextConfiguration>();
    auto context = backend_->createContext(context_config);
    contexts_.push_back(std::make_unique<qairt::Context>(std::move(context)));

    if (config.enable_profiling) {
      profile_ = std::make_unique<qairt::Profile>(backend_->createProfile(QAIRT_PROFILE_LEVEL_BASIC));
    }
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtBackendManager::Initialize failed: " +
                         std::string(e.what()))
                            .c_str());
  }

  return Ort::Status();
}

const QNN_INTERFACE_VER_TYPE& QairtBackendManager::GetQnnInterface() {
  return null_interface_;
}

const QNN_INTERFACE_VER_TYPE& QairtBackendManager::GetQnnValidatorInterface() {
  return null_interface_;
}

const Qnn_BackendHandle_t& QairtBackendManager::GetQnnBackendHandle() {
  return null_backend_handle_;
}

const Qnn_BackendHandle_t& QairtBackendManager::GetQnnValidatorBackendHandle() {
  return null_backend_handle_;
}

const Qnn_ContextHandle_t& QairtBackendManager::GetQnnContext(int /*index*/) {
  return null_context_handle_;
}

QnnBackendType QairtBackendManager::GetQnnBackendType() {
  return backend_type_;
}

const Qnn_ProfileHandle_t& QairtBackendManager::GetQnnProfileHandle() {
  return null_profile_handle_;
}

std::unique_ptr<unsigned char[]> QairtBackendManager::GetContextBinaryBuffer(
    uint64_t& written_buffer_size) {
  // ponytail: context binary serialization via QAIRT C++ API. Implement when
  // context caching is wired up. For now return null — EP skips caching gracefully.
  written_buffer_size = 0;
  return nullptr;
}

Ort::Status QairtBackendManager::LoadCachedQnnContextFromBuffer(
    char* /*buffer*/,
    uint64_t /*buffer_length*/,
    const std::string& /*context_bin_filepath*/,
    std::string /*node_name*/,
    std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& /*qnn_models*/,
    int64_t /*max_spill_fill_size*/) {
  // ponytail: context-from-binary via QAIRT C++ API (Backend::createContextFromBinary).
  // Implement when cached context path is tested end-to-end.
  return MAKE_EP_FAIL("QairtBackendManager::LoadCachedQnnContextFromBuffer not yet implemented");
}

Ort::Status QairtBackendManager::SetHtpPowerConfigs(uint32_t /*htp_power_config_client_id*/,
                                                    HtpPerformanceMode /*htp_performance_mode*/,
                                                    uint32_t /*rpc_polling_time*/,
                                                    uint32_t /*rpc_control_latency*/) {
  // ponytail: QAIRT C++ API has no HTP power config equivalent yet.
  // Return OK — power config is best-effort (EP logs warning but doesn't fail session).
  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime

#endif  // USE_QAIRT_API
