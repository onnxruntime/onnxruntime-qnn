// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef USE_QAIRT_API

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "QairtCpp/QairtApi.hpp"
#include "QairtCpp/QairtBackend.hpp"
#include "QairtCpp/QairtContext.hpp"
#include "QairtCpp/QairtLog.hpp"
#include "QairtCpp/QairtProfile.hpp"

#include "core/providers/qnn/builder/i_backend_manager.h"

namespace onnxruntime {
namespace qnn {

// IBackendManager implementation using QAIRT C++ API.
// Manages: Api → Log → Backend → Device → Context lifecycle.
// C-handle accessors return null sentinels — callers on the QAIRT path
// use QairtGraphEmitter (which holds direct references to api_/context_).
class QairtBackendManager final : public IBackendManager {
 public:
  struct Config {
    std::string backend_lib_path;
    QnnBackendType backend_type = QnnBackendType::HTP;
    bool enable_profiling = false;
  };

  static std::unique_ptr<QairtBackendManager> Create(const Config& config, Ort::Status& status);

  ~QairtBackendManager() override = default;
  QairtBackendManager(const QairtBackendManager&) = delete;
  QairtBackendManager& operator=(const QairtBackendManager&) = delete;

  // IBackendManager — C handle accessors return null sentinels.
  const QNN_INTERFACE_VER_TYPE& GetQnnInterface() const override;
  const QNN_INTERFACE_VER_TYPE& GetQnnValidatorInterface() const override;
  const Qnn_BackendHandle_t& GetQnnBackendHandle() const override;
  const Qnn_BackendHandle_t& GetQnnValidatorBackendHandle() const override;
  const Qnn_ContextHandle_t& GetQnnContext(int index = 0) override;
  QnnBackendType GetQnnBackendType() const override;

  Ort::Status GetContextBinaryBuffer(bool is_multi_soc_buffer,
                                     unsigned char** context_buffer,
                                     uint64_t& buffer_size) override;
  Ort::Status LoadCachedQnnContextFromBuffer(
      char* buffer,
      uint64_t buffer_length,
      const std::string& context_bin_filepath,
      std::string node_name,
      std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models,
      int64_t max_spill_fill_size,
      const qnn::EpContextIoDispatch& io_dispatch,
      bool is_multi_soc_buffer = false) override;

  // QAIRT-specific accessors for QairtGraphEmitter construction.
  qairt::Api& GetApi() { return *api_; }
  qairt::Context& GetContext(int index = 0) { return *contexts_[index]; }

 private:
  QairtBackendManager() = default;
  Ort::Status Initialize(const Config& config);

  std::unique_ptr<qairt::Api> api_;
  std::shared_ptr<qairt::Log> log_;
  std::shared_ptr<qairt::Backend> backend_;
  std::vector<std::unique_ptr<qairt::Context>> contexts_;
  std::unique_ptr<qairt::Profile> profile_;

  QnnBackendType backend_type_ = QnnBackendType::HTP;

  // Null sentinels returned by C-handle accessors.
  static const QNN_INTERFACE_VER_TYPE null_interface_;
  static const Qnn_BackendHandle_t null_backend_handle_;
  static const Qnn_ContextHandle_t null_context_handle_;
};

}  // namespace qnn
}  // namespace onnxruntime

#endif  // USE_QAIRT_API
