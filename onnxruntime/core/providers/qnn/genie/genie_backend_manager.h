// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <libloaderapi.h>
#include <set>
#else
#include <dlfcn.h>
#endif

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

struct GenieBackendManagerConfig {
  std::string backend_path;
};

class GenieBackendManager : public std::enable_shared_from_this<GenieBackendManager> {
 private:
  // private tag to pass to constructor to ensure that constructor cannot be directly called externally
  struct PrivateConstructorTag {};

 public:
  static std::shared_ptr<GenieBackendManager> Create(const GenieBackendManagerConfig& config, const Ort::Logger& logger) {
    return std::make_shared<GenieBackendManager>(config, logger, PrivateConstructorTag{});
  }

  // Note: Creation should be done via Create(). This constructor is public so that it can be called from
  // std::make_shared().
  GenieBackendManager(const GenieBackendManagerConfig& config, const Ort::Logger& logger, PrivateConstructorTag);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GenieBackendManager);

  ~GenieBackendManager();

  // Initializes handles to Genie resources (device, logger, etc.).
  Ort::Status SetupBackend();

  void* GetGenieBackendHandle() { return backend_lib_handle_; }

 private:
  Ort::Status LoadBackend();

  void ReleaseResources();

  void* LoadLib(const char* file_name, int flags, std::string& error_msg);

  Ort::Status UnloadLib(void* handle);

 private:
  std::recursive_mutex logger_recursive_mutex_;
  const Ort::Logger* logger_ptr_;

  const std::string backend_path_;
  bool backend_setup_completed_ = false;
  void* backend_lib_handle_ = nullptr;

#ifdef _WIN32
  std::unordered_set<HMODULE> mod_handles_;
#endif
};

}  // namespace qnn
}  // namespace onnxruntime
