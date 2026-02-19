
#include "core/providers/qnn/builder/genie_api_loader.h"
#include <iostream>
#include <stdexcept>
#include <cstring>

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
  inline void dynlib_close(dynlib_handle h) { if (h) ::dlclose(h); }
  inline const char* dynlib_error() { return ::dlerror(); }
#endif

static void* must_dlsym(void* h, const char* name) {
    void* p = dynlib_sym(h, name);
    if (!p) throw std::runtime_error(std::string("dlsym failed for symbol: ") + name);
    return p;
}

GenieApiLoader::GenieApiLoader(void* shared_library_handle)
    : handle_(shared_library_handle) 
{
    if (!handle_) {
        throw std::runtime_error("GenieApiLoader: Null library handle");
    }
}

const GenieApi& GenieApiLoader::Get() {
    std::call_once(init_flag_, &GenieApiLoader::Init, this);
    return api_;
}

// Init(): Resolve all symbols exactly once
void GenieApiLoader::Init() {
    try {
        api_.NodeConfig_createFromJson =
            (TYPE_GenieNodeConfig_createFromJson)must_dlsym(handle_, "GenieNodeConfig_createFromJson");

        api_.Node_create =
            (TYPE_GenieNode_create)must_dlsym(handle_, "GenieNode_create");

        api_.Node_setData =
            (TYPE_GenieNode_setData)must_dlsym(handle_, "GenieNode_setData");

        api_.Node_getData =
            (TYPE_GenieNode_getData)must_dlsym(handle_, "GenieNode_getData");

        api_.Node_execute =
            (TYPE_GenieNode_execute)must_dlsym(handle_, "GenieNode_execute");

        api_.Node_free =
            (TYPE_GenieNode_free)must_dlsym(handle_, "GenieNode_free");

        api_.NodeConfig_free =
            (TYPE_GenieNodeConfig_free)must_dlsym(handle_, "GenieNodeConfig_free");


    } catch (const std::exception& ex) {
        std::cerr << "[GenieApiLoader] Initialization failed: " << ex.what() << std::endl;
        throw;
    }
}
