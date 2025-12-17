//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <libloaderapi.h>
#include <set>
#else
#include <dlfcn.h>
#endif

#include "genie_backend_manager.h"
#include "qnn_model.h"
#include <filesystem>
#include <fstream>
#include <string>
#include "QnnOpDef.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/qnn_telemetry.h"
#include "core/providers/qnn/shared_context.h"
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

GenieBackendManager::GenieBackendManager(const GenieBackendManagerConfig& config, PrivateConstructorTag)
    : backend_path_(config.backend_path) {
}

GenieBackendManager::~GenieBackendManager() {
  ReleaseResources();
}

Status GenieBackendManager::SetupBackend(const logging::Logger& logger) {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);
  if (backend_setup_completed_) {
    LOGS(logger, VERBOSE) << "Backend setup already!";
    return Status::OK();
  }

  logger_ = &logger;
  LOGS(logger, INFO) << "Setting up Genie backend";
  
  // Load the Genie backend library
  Status status = LoadBackend();
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Failed to load Genie backend: " << status.ErrorMessage();
    return status;
  }

  LOGS(logger, VERBOSE) << "Genie SetupBackend succeed";
  backend_setup_completed_ = true;
  return Status::OK();
}

void* GenieBackendManager::LoadLib(const char* file_name, int flags, std::string& error_msg) {
#ifdef _WIN32
  DWORD as_is, to_be;
  bool loaded_before = false;

  if (!file_name || ::strlen(file_name) == 0) {
    error_msg = "filename is null or empty";
    return nullptr;
  }

  // POSIX asks one of symbol resolving approaches:
  // NOW or LAZY must be specified
  if (!(flags & static_cast<int>(DlOpenFlag::DL_NOW))) {
    error_msg = "flags must include DL_NOW";
    return nullptr;
  }

  HANDLE cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, nullptr, 0, &as_is) == 0) {
    error_msg = "enumerate modules failed before loading module";
    return nullptr;
  }

  HMODULE mod;
  auto file_path = std::filesystem::path(file_name);
  if (!file_path.is_absolute()) {
    // construct an absolute path from ORT runtime path + file_name and check whether it exists.
    const Env& env = GetDefaultEnv();
    auto pathstring = env.GetRuntimePath() + ToPathString(file_name);
    auto absolute_path = pathstring.c_str();
    if (std::filesystem::exists(std::filesystem::path(absolute_path))) {
      // load library from absolute path and search for dependencies there.
      mod = LoadLibraryExW(absolute_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    } else {
      // use default dll search order for file_name.
      mod = LoadLibraryExA(file_name, nullptr, 0);
    }
  } else {
    // file_name represents an absolute path.
    // load library from absolute path and search for dependencies there.
    mod = LoadLibraryExA(file_name, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  }
  if (!mod) {
    error_msg = "load library failed";
    return nullptr;
  }

  if (EnumProcessModules(cur_proc, nullptr, 0, &to_be) == 0) {
    error_msg = "enumerate modules failed after loading module";
    FreeLibrary(mod);
    return nullptr;
  }

  if (as_is == to_be) {
    loaded_before = true;
  }

  // (not loaded_before) and DL_LOCAL means this lib was not loaded yet
  // add it into the local set
  //
  // If loaded_before and DL_LOCAL, means this lib was already loaded
  // 2 cases here for how it was loaded before:
  // a. with DL_LOCAL, just ignore since it was already in local set
  // b. with DL_GLOBAL, POSIX asks it in global, ignore it, too
  if ((!loaded_before) && (flags & static_cast<int>(DlOpenFlag::DL_LOCAL))) {
    mod_handles_.insert(mod);
  }

  // once callers ask for global, needs to be in global thereafter
  // so the lib should be removed from local set
  if (flags & static_cast<int>(DlOpenFlag::DL_GLOBAL)) {
    mod_handles_.erase(mod);
  }

  return static_cast<void*>(mod);
#else
  ORT_UNUSED_PARAMETER(error_msg);
  int real_flags = 0;

  if (flags & static_cast<int>(DlOpenFlag::DL_NOW)) {
    real_flags |= RTLD_NOW;
  }

  if (flags & static_cast<int>(DlOpenFlag::DL_LOCAL)) {
    real_flags |= RTLD_LOCAL;
  }

  if (flags & static_cast<int>(DlOpenFlag::DL_GLOBAL)) {
    real_flags |= RTLD_GLOBAL;
  }

  return ::dlopen(file_name, real_flags);
#endif
}

Status GenieBackendManager::LoadBackend() {
  std::string error_msg;
  backend_lib_handle_ = LoadLib(backend_path_.c_str(),
                                static_cast<int>(DlOpenFlag::DL_NOW) | static_cast<int>(DlOpenFlag::DL_GLOBAL),
                                error_msg);
  ORT_RETURN_IF(nullptr == backend_lib_handle_, "Unable to load Genie backend, error: ", error_msg);

  // TODO: Initialize Genie interface here
  // This would involve getting the appropriate function pointers from the loaded library

  return Status::OK();
}

void GenieBackendManager::ReleaseResources() {
  if (backend_lib_handle_) {
    auto result = UnloadLib(backend_lib_handle_);
    if (Status::OK() != result) {
      LOGS_DEFAULT(ERROR) << "Failed to unload backend library: " << result.ErrorMessage();
    }
  }

  backend_setup_completed_ = false;

  return;
}

Status GenieBackendManager::UnloadLib(void* handle) {
  if (!handle) {
    return Status::OK();
  }

#ifdef _WIN32
  HMODULE mod = static_cast<HMODULE>(handle);

  if (FreeLibrary(mod) == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to free library.");
  }
  mod_handles_.erase(mod);
#else
  auto rt = ::dlclose(handle);
  if (rt != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to free library.");
  }
#endif  // defined(_WIN32)

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
