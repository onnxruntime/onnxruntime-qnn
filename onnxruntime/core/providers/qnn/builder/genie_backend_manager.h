// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include "core/providers/qnn/builder/qnn_model.h"

namespace onnxruntime {
namespace qnn {

// configuration values for GenieBackendManager creation
// TODO - There may be values here that we want to the values inside of the GenieEngine? or node config?
struct GenieBackendManagerConfig {
    std::string backend_path;
};

class GenieBackendManager : public std::enable_shared_from_this<GenieBackendManager> {
 private:
  // private tag to pass to constructor to ensure that constructor cannot be directly called externally
  struct PrivateConstructorTag {};

 public:
  static std::shared_ptr<GenieBackendManager> Create(const GenieBackendManagerConfig& config) {
    return std::make_shared<GenieBackendManager>(config, PrivateConstructorTag{});
  }

  // Note: Creation should be done via Create(). This constructor is public so that it can be called from
  // std::make_shared().
  GenieBackendManager(const GenieBackendManagerConfig& config, PrivateConstructorTag);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GenieBackendManager);

  ~GenieBackendManager();

  // Initializes handles to Genie resources (device, logger, etc.).
  Status SetupBackend(const logging::Logger& logger);

  void* getGenieBackendHandle() {return backend_lib_handle_;}

  Status GetZipContextPath(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                          onnxruntime::PathString model_path,
                          std::filesystem::path& zip_extract_path);
  Status GetGenieConfig(std::filesystem::path zip_extracted_path, std::string& genieConfigJsonText);
 private:
  Status LoadBackend();

  void ReleaseResources();

  void* LoadLib(const char* file_name, int flags, std::string& error_msg);

  Status UnloadLib(void* handle);

 private:
  std::recursive_mutex logger_recursive_mutex_;
  const logging::Logger* logger_ = nullptr;

  const std::string backend_path_;
  void* genie_interface_ = nullptr;
  bool backend_initialized_ = false;
  bool backend_setup_completed_ = false;
  void* backend_lib_handle_ = nullptr;

#ifdef _WIN32
  std::set<HMODULE> mod_handles_;
#endif
};

}  // namespace qnn
}  // namespace onnxruntime
