// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#ifdef USE_QAIRT_API
#include <memory>
#include "HTP/QnnHtpDevice.h"
#include <string>
#include <vector>
#include "core/providers/qnn/builder/backend_manager_interface.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace qairt {
class Api;
class Log;
class Backend;
class Context;
class Profile;
}  // namespace qairt
namespace onnxruntime::qnn {

// Owns only QAIRT C++ objects. No QNN function table or C backend handles.
class QairtBackendManager final : public IBackendManager,
                                  public std::enable_shared_from_this<QairtBackendManager> {
 public:
  struct Config {
    std::string backend_lib_path;
    QnnBackendType backend_type = QnnBackendType::HTP;
    QnnHtpDevice_Arch_t htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
    uint32_t soc_model = 0;
  };
  explicit QairtBackendManager(Config config);
  ~QairtBackendManager();
  QairtBackendManager(const QairtBackendManager&) = delete;
  QairtBackendManager& operator=(const QairtBackendManager&) = delete;

  struct CachedGraph {
    std::shared_ptr<qairt::Context> context;
    std::unique_ptr<GraphInfo> info;
  };
  Ort::Status LoadContext(std::vector<char>& buffer, std::vector<CachedGraph>& graphs, qairt::Profile* profile = nullptr);
  std::string GetSdkVersion();
  Ort::Status Setup();
  Ort::Status ValidateOp(const Qnn_OpConfig_t& op);
  Ort::Status ExportContext(std::vector<unsigned char>& buffer);
  std::string GetSystemLibraryPath() const;
  qairt::Api& GetApi();
  qairt::Backend& GetBackend();
  std::shared_ptr<qairt::Context> GetContext();
  BackendRuntimeKind GetRuntimeKind() const override { return BackendRuntimeKind::Qairt; }
  QnnBackendType GetBackendType() const override { return config_.backend_type; }
  QnnHtpDevice_Arch_t GetHtpArch() const override { return config_.htp_arch; }
  uint32_t GetSocModel() const override { return config_.soc_model; }
  std::unique_ptr<IGraphEmitter> CreateGraphEmitter() override;

 private:
  Config config_;
  std::unique_ptr<qairt::Api> qairt_api_;
  std::shared_ptr<qairt::Log> qairt_log_;
  std::shared_ptr<qairt::Backend> qairt_backend_;
  std::shared_ptr<qairt::Context> qairt_context_;
};
}  // namespace onnxruntime::qnn
#endif
