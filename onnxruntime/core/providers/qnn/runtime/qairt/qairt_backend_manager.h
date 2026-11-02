// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
#pragma once

#ifdef USE_QAIRT_API
#include <memory>
#include <mutex>
#include "HTP/QnnHtpDevice.h"
#include <string>
#include <vector>
#include "core/providers/qnn/runtime/interfaces/backend_manager_interface.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace qairt {
class Api;
class Log;
class Backend;
class Device;
class Context;
class ContextConfiguration;
class Graph;
class GraphConfiguration;
class Profile;
}  // namespace qairt
namespace onnxruntime::qnn {
struct QairtBackendConfiguration;

// Owns only QAIRT C++ objects. No QNN function table or C backend handles.
// Accelerator-specific behavior (HTP, GPU, future LPAI) should be composed as
// focused QAIRT extension services instead of adding it to IBackendManager.
class QairtBackendManager final : public IBackendManager,
                                  public std::enable_shared_from_this<QairtBackendManager> {
 public:
  struct Config {
    std::string backend_lib_path;
    QnnBackendType backend_type = QnnBackendType::HTP;
    QnnHtpDevice_Arch_t htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
    uint32_t soc_model = 0;
    std::shared_ptr<const QairtBackendConfiguration> configuration;
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
  enum class SetupFailure { None,
                            LibraryLoad,
                            BackendInitialization };
  Ort::Status Setup(SetupFailure* failure = nullptr);
  Ort::Status ValidateOp(const Qnn_OpConfig_t& op);
  Ort::Status ExportContext(std::vector<unsigned char>& buffer);
  std::string GetSystemLibraryPath() const;
  // Setup must succeed before publishing the manager to graph/profile callers.
  // Only Setup and lazy context creation may run concurrently during initialization.
  qairt::Api& GetApi();
  qairt::Backend& GetBackend();
  std::shared_ptr<qairt::Context> GetContext();
  qairt::ContextConfiguration CreateContextConfiguration();
  qairt::GraphConfiguration CreateGraphConfiguration();
  Ort::Status ConfigureRetrievedGraph(qairt::Graph& graph);
  const std::shared_ptr<qairt::Device>& GetDevice() const { return qairt_device_; }
  BackendRuntimeKind GetRuntimeKind() const override { return BackendRuntimeKind::Qairt; }
  QnnBackendType GetBackendType() const override { return config_.backend_type; }
  QnnHtpDevice_Arch_t GetHtpArch() const override { return config_.htp_arch; }
  uint32_t GetSocModel() const override { return config_.soc_model; }
  std::unique_ptr<IGraphEmitter> CreateGraphEmitter() override;

 private:
  // Configuration callbacks may retain SDK objects. Release them after contexts
  // and graphs, but before unloading the API library.
  std::unique_ptr<qairt::Api> qairt_api_;
  Config config_;
  // Initialization is serialized; inference never holds this mutex.
  std::mutex initialization_mutex_;
  std::shared_ptr<qairt::Log> qairt_log_;
  std::shared_ptr<qairt::Backend> qairt_backend_;
  std::shared_ptr<qairt::Device> qairt_device_;
  std::shared_ptr<qairt::Context> qairt_context_;
};
}  // namespace onnxruntime::qnn
#endif
