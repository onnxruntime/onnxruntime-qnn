// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/genie/genie_backend_manager.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <gsl/gsl>
#include <memory>
#include <string>

namespace onnxruntime {
namespace qnn {

GenieBackendManager::GenieBackendManager(const GenieBackendManagerConfig& config, const Ort::Logger& logger, PrivateConstructorTag)
    : logger_ptr_(&logger), backend_path_(config.backend_path) {
}

GenieBackendManager::~GenieBackendManager() {
  ReleaseResources();
}

Ort::Status GenieBackendManager::SetupBackend() {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);
  if (backend_setup_completed_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Backend setup already!");
    return Ort::Status();
  }

  // Load the Genie backend library
  Ort::Status status = LoadBackend();
  if (!status.IsOK()) {
    std::ostringstream oss;
    oss << "Failed to load Genie backend library: "
        << status.GetErrorMessage();
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
    return status;
  }
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Genie SetupBackend succeeded");
  backend_setup_completed_ = true;
  return Ort::Status();
}

Ort::Status GenieBackendManager::LoadBackend() {
  std::ostringstream oss;
  oss << "Loading Genie backend library from: "
      << backend_path_.c_str();
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, oss.str().c_str());

  std::string error_msg;
  backend_lib_handle_ = LoadLib(backend_path_.c_str(),
                                static_cast<int>(DlOpenFlag::DL_NOW) | static_cast<int>(DlOpenFlag::DL_GLOBAL),
                                error_msg);

  if (nullptr == backend_lib_handle_) {
    std::ostringstream ossMsg;
    ossMsg << "Unable to load Genie backend, error: " << error_msg.c_str();
    return MAKE_EP_FAIL(ossMsg.str().c_str());
  }

  return Ort::Status();
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
  std::filesystem::path file_path(file_name);
  if (!file_path.is_absolute()) {
    // construct an absolute path from ORT runtime path + file_name and check whether it exists.

    auto absolute_path = std::filesystem::path(OrtGetRuntimePath()) / file_path;
    auto absolute_path_str = absolute_path.c_str();

    if (std::filesystem::exists(absolute_path)) {
      // load library from absolute path and search for dependencies there.
      mod = LoadLibraryExW(absolute_path_str, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
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

  UniqueLibraryHandle library_handle(static_cast<void*>(mod));
  backend_libraries_->push_back(std::move(library_handle));
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

  UniqueLibraryHandle library_handle(::dlopen(file_name, real_flags));
  void* handle = library_handle.get();
  if (!handle) {
    error_msg = ::dlerror();
  } else {
    backend_libraries_->push_back(std::move(library_handle));
  }
  return handle;
#endif
}

void GenieBackendManager::LibraryDeleter::operator()(void* handle) const noexcept {
  if (handle == nullptr) {
    return;
  }
#ifdef _WIN32
  const bool unloaded = ::FreeLibrary(static_cast<HMODULE>(handle)) != 0;
#else
  const bool unloaded = ::dlclose(handle) == 0;
#endif
  if (!unloaded && OrtLoggingManager::HasDefaultLogger()) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR,
                "Failed to unload Genie backend library.");
  }
}

void GenieBackendManager::ReleaseResources() {
  backend_setup_completed_ = false;
}

}  // namespace qnn
}  // namespace onnxruntime
