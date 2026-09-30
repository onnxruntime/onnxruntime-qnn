// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifdef USE_QAIRT_API

#include "core/providers/qnn/builder/qairt_backend_manager.h"

#include "QairtCpp/QairtApi.hpp"
#include <cstdio>

namespace onnxruntime {
namespace qnn {

// Null sentinels for C-handle accessors. Never dereferenced on the QAIRT path.
const QNN_INTERFACE_VER_TYPE QairtBackendManager::null_interface_{};
const Qnn_BackendHandle_t QairtBackendManager::null_backend_handle_ = nullptr;
const Qnn_ContextHandle_t QairtBackendManager::null_context_handle_ = nullptr;

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
  fprintf(stderr, "[QAIRT C++ API] QairtBackendManager::Initialize — loading backend: %s\n", config.backend_lib_path.c_str());
  fflush(stderr);
  backend_type_ = config.backend_type;

  try {
    api_ = std::make_unique<qairt::Api>(config.backend_lib_path.c_str());
  } catch (const std::runtime_error& e) {
    return MAKE_EP_FAIL(("QairtBackendManager: failed to load backend library `" +
                         config.backend_lib_path + "`: " + e.what())
                            .c_str());
  }

  try {
    log_ = api_->makeShared<qairt::Log>(static_cast<QairtLog_CallbackFn_t>(nullptr),
                                        QAIRT_LOG_LEVEL_WARN);
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtBackendManager: makeShared<Log> failed: " +
                         std::string(e.what()))
                            .c_str());
  }

  try {
    backend_ = api_->makeShared<qairt::Backend>(*log_);
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtBackendManager: makeShared<Backend> failed: " +
                         std::string(e.what()))
                            .c_str());
  }

  try {
    auto context_config = api_->make<qairt::ContextConfiguration>();
    auto context = backend_->createContext(context_config);
    contexts_.push_back(std::make_unique<qairt::Context>(std::move(context)));
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtBackendManager: createContext failed: " +
                         std::string(e.what()))
                            .c_str());
  }

  try {
    if (config.enable_profiling) {
      profile_ = std::make_unique<qairt::Profile>(backend_->createProfile(QAIRT_PROFILE_LEVEL_BASIC));
    }
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtBackendManager: createProfile failed: " +
                         std::string(e.what()))
                            .c_str());
  }

  fprintf(stderr, "[QAIRT C++ API] QairtBackendManager::Initialize — SUCCESS (backend=%s, context created)\n",
          config.backend_lib_path.c_str());
  fflush(stderr);
  return Ort::Status();
}

const QNN_INTERFACE_VER_TYPE& QairtBackendManager::GetQnnInterface() const {
  return null_interface_;
}

const QNN_INTERFACE_VER_TYPE& QairtBackendManager::GetQnnValidatorInterface() const {
  return null_interface_;
}

const Qnn_BackendHandle_t& QairtBackendManager::GetQnnBackendHandle() const {
  return null_backend_handle_;
}

const Qnn_BackendHandle_t& QairtBackendManager::GetQnnValidatorBackendHandle() const {
  return null_backend_handle_;
}

const Qnn_ContextHandle_t& QairtBackendManager::GetQnnContext(int /*index*/) {
  return null_context_handle_;
}

QnnBackendType QairtBackendManager::GetQnnBackendType() const {
  return backend_type_;
}

Ort::Status QairtBackendManager::GetContextBinaryBuffer(bool /*is_multi_soc_buffer*/,
                                                        unsigned char** context_buffer,
                                                        uint64_t& buffer_size) {
  buffer_size = 0;
  *context_buffer = nullptr;
  if (contexts_.empty()) {
    return Ort::Status();
  }
  try {
    uint64_t binary_size = contexts_[0]->getBinarySize();
    if (binary_size == 0) {
      return Ort::Status();
    }
    auto buffer = new unsigned char[binary_size];
    buffer_size = contexts_[0]->getBinary(buffer, binary_size);
    *context_buffer = buffer;
    return Ort::Status();
  } catch (const qairt::Exception& e) {
    buffer_size = 0;
    *context_buffer = nullptr;
    return MAKE_EP_FAIL(("QairtBackendManager::GetContextBinaryBuffer failed: " +
                         std::string(e.what()))
                            .c_str());
  }
}

Ort::Status QairtBackendManager::LoadCachedQnnContextFromBuffer(
    char* /*buffer*/,
    uint64_t /*buffer_length*/,
    const std::string& /*context_bin_filepath*/,
    std::string /*node_name*/,
    std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& /*qnn_models*/,
    int64_t /*max_spill_fill_size*/,
    const qnn::EpContextIoDispatch& /*io_dispatch*/,
    bool /*is_multi_soc_buffer*/) {
  return MAKE_EP_FAIL("QairtBackendManager::LoadCachedQnnContextFromBuffer not yet implemented");
}

}  // namespace qnn
}  // namespace onnxruntime

#endif  // USE_QAIRT_API
