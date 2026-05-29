// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <filesystem>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#endif
namespace onnxruntime {
namespace test {
#if defined(_WIN32)

bool ReadEnvironmentVariable(const wchar_t* name, std::wstring& value_out) {
  const DWORD value_size = ::GetEnvironmentVariableW(name, nullptr, 0);
  if (value_size == 0) {
    return false;
  }

  std::vector<wchar_t> value(value_size);

  if (::GetEnvironmentVariableW(name, value.data(), value_size) == 0) {
    return false;
  }

  value_out = std::wstring{value.data()};
  return true;
}

bool GetServiceBinaryDirectoryPath(const wchar_t* service_name,
                                   std::filesystem::path& service_binary_directory_path_out) {
  struct ServiceHandleDeleter {
    void operator()(SC_HANDLE handle) { ::CloseServiceHandle(handle); }
  };

  using UniqueServiceHandle = std::unique_ptr<std::remove_pointer_t<SC_HANDLE>, ServiceHandleDeleter>;

  SC_HANDLE scm_handle_raw = ::OpenSCManagerW(nullptr,  // local computer
                                              nullptr,  // SERVICES_ACTIVE_DATABASE
                                              STANDARD_RIGHTS_READ);
  if (scm_handle_raw == nullptr) {
    return false;
  }

  auto scm_handle = UniqueServiceHandle{scm_handle_raw};

  SC_HANDLE service_handle_raw = ::OpenServiceW(scm_handle.get(),
                                                service_name,
                                                SERVICE_QUERY_CONFIG);
  if (service_handle_raw == nullptr) {
    return false;
  }

  auto service_handle = UniqueServiceHandle{service_handle_raw};

  // get service config required buffer size
  DWORD service_config_buffer_size{};
  if (!::QueryServiceConfigW(service_handle.get(), nullptr, 0, &service_config_buffer_size) &&
      ::GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return false;
  }

  // get the service config
  std::vector<std::byte> service_config_buffer(service_config_buffer_size);
  QUERY_SERVICE_CONFIGW* service_config = reinterpret_cast<QUERY_SERVICE_CONFIGW*>(service_config_buffer.data());
  if (!::QueryServiceConfigW(service_handle.get(), service_config, service_config_buffer_size,
                             &service_config_buffer_size)) {
    return false;
  }

  std::wstring service_binary_path_name = service_config->lpBinaryPathName;

  // replace system root placeholder with the value of the SYSTEMROOT environment variable
  const std::wstring system_root_placeholder = L"\\SystemRoot";

  if (service_binary_path_name.find(system_root_placeholder, 0) != 0) {
    return false;
  }

  std::wstring system_root{};
  if (!ReadEnvironmentVariable(L"SYSTEMROOT", system_root)) return false;
  service_binary_path_name.replace(0, system_root_placeholder.size(), system_root);

  const auto service_binary_path = std::filesystem::path{service_binary_path_name};
  auto service_binary_directory_path = service_binary_path.parent_path();

  if (!std::filesystem::exists(service_binary_directory_path)) {
    return false;
  }

  service_binary_directory_path_out = std::move(service_binary_directory_path);
  return true;
}

#endif  // defined(_WIN32)

bool GetRpcMemDynamicLibraryPath(std::filesystem::path& path_out) {
#if defined(_WIN32)

  std::filesystem::path qcnspmcdm_dir_path{};
  if (!GetServiceBinaryDirectoryPath(L"qcnspmcdm", qcnspmcdm_dir_path)) return false;
  path_out = qcnspmcdm_dir_path / L"libcdsprpc.dll";
  return true;

#else  // ^^^ defined(_WIN32) / vvv !defined(_WIN32)

  path_out = "libcdsprpc.so";
  return true;

#endif  // !defined(_WIN32)
}

void TriggerPDReset() {
#if defined(_WIN32)
  std::filesystem::path rpcmem_library_path{};
  GetRpcMemDynamicLibraryPath(rpcmem_library_path);
  HMODULE lib_handle = LoadLibraryW(rpcmem_library_path.c_str());
  if (!lib_handle) {
    return;  // Failed to load library
  }
  typedef int (*RscFnHandleType_t)(uint32_t, void*, uint32_t);
  FARPROC addr = GetProcAddress(lib_handle, "remote_session_control");
  if (!addr) {
    FreeLibrary(lib_handle);
    return;  // Failed to get procedure address
  }
  RscFnHandleType_t rsc_call = reinterpret_cast<RscFnHandleType_t>(addr);
  typedef struct {
    int domain;
  } remote_rpc_process_clean_params;
  remote_rpc_process_clean_params scdata;
  scdata.domain = 3; /*CDSP_DOMAIN_ID*/
  rsc_call(/*FASTRPC_REMOTE_PROCESS_KILL*/ 6, &scdata, sizeof(remote_rpc_process_clean_params));
  if (lib_handle) {
    FreeLibrary(lib_handle);
  }
  lib_handle = nullptr;
#endif  // !defined(_WIN32)
}
}  // namespace test
}  // namespace onnxruntime
