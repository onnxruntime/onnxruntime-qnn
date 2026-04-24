// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/genie/genie_api_loader.h"
#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
inline void* dynlib_sym(void* h, const char* name) {
  HMODULE hmod = reinterpret_cast<HMODULE>(h);
  return reinterpret_cast<void*>(::GetProcAddress(hmod, name));
}
#else
#include <dlfcn.h>
using dynlib_handle = void*;
inline dynlib_handle dynlib_open(const char* path) { return ::dlopen(path, RTLD_NOW); }
inline void* dynlib_sym(dynlib_handle h, const char* name) { return ::dlsym(h, name); }
inline void dynlib_close(dynlib_handle h) {
  if (h) ::dlclose(h);
}
inline const char* dynlib_error() { return ::dlerror(); }
#endif

template <typename T>
static T must_dlsym(void* h, const char* name) {
  void* p = dynlib_sym(h, name);
  if (!p) {
    throw std::runtime_error(std::string("dlsym failed for symbol: ") + name);
  }
  return reinterpret_cast<T>(p);
}

GenieApiLoader::GenieApiLoader(void* shared_library_handle)
    : handle_(shared_library_handle) {
  if (!handle_) {
    throw std::runtime_error("GenieApiLoader: Null library handle");
  }
}

const GenieApi& GenieApiLoader::Get() {
  std::call_once(init_flag_, &GenieApiLoader::Init, this);
  return api_;
}

// Macro to simplify symbol loading
#define STRINGIFY(x) #x
#define LOAD_GENIE_SYMBOL(name) \
  api_.name = must_dlsym<decltype(api_.name)>(handle_, STRINGIFY(Genie##name))

void GenieApiLoader::Init() {
  LOAD_GENIE_SYMBOL(NodeConfig_createFromJson);
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
  LOAD_GENIE_SYMBOL(DlcConfig_create);
  LOAD_GENIE_SYMBOL(DlcConfig_free);
  LOAD_GENIE_SYMBOL(Dlc_create);
  LOAD_GENIE_SYMBOL(Dlc_free);
  LOAD_GENIE_SYMBOL(Dlc_getUseCases);
  LOAD_GENIE_SYMBOL(NodeConfig_createFromDlc);
  LOAD_GENIE_SYMBOL(Node_getData);
  LOAD_GENIE_SYMBOL(Node_execute);
  LOAD_GENIE_SYMBOL(Node_reset);
#endif
  LOAD_GENIE_SYMBOL(Node_create);
  LOAD_GENIE_SYMBOL(Node_setData);
  LOAD_GENIE_SYMBOL(Node_free);
  LOAD_GENIE_SYMBOL(NodeConfig_free);
  LOAD_GENIE_SYMBOL(Log_create);
  LOAD_GENIE_SYMBOL(NodeConfig_bindLogger);
  LOAD_GENIE_SYMBOL(Log_free);
}
