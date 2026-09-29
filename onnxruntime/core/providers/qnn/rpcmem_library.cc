// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/rpcmem_library.h"

#if defined(_WIN32)
#include <filesystem>

#include <Windows.h>
#include <cfgmgr32.h>
// INITGUID must precede <devpkey.h> so the DEVPKEY_* keys are emitted as data symbols in this
// translation unit; without it they are only declared and the link fails with unresolved externals.
#define INITGUID
#include <devpropdef.h>
#include <devpkey.h>
#undef INITGUID
// Configuration Manager (CM_*) APIs used to resolve the MCDM driver service from its interface GUID.
#pragma comment(lib, "cfgmgr32.lib")
#endif  // defined(_WIN32)

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime::qnn {

// Unload the dynamic library referenced by `library_handle`.
// Avoid throwing because this may run from a dtor.
void DynamicLibraryHandleDeleter::operator()(void* library_handle) noexcept {
  if (library_handle == nullptr) {
    return;
  }

  const auto unload_status = OrtUnloadDynamicLibrary(library_handle);

  if (!unload_status.IsOK()) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                ORT_LOGGING_LEVEL_WARNING,
                ("Failed to unload dynamic library. Error: " + unload_status.GetErrorMessage()).c_str());
  }
}

namespace {

#if defined(_WIN32)

struct HKeyDeleter {
  void operator()(HKEY handle) { ::RegCloseKey(handle); }
};

using UniqueHKey = std::unique_ptr<std::remove_pointer_t<HKEY>, HKeyDeleter>;

// Read a REG_SZ / REG_EXPAND_SZ value from an open registry key into `value_out`.
Ort::Status ReadRegistryStringValue(HKEY key, const wchar_t* value_name, std::wstring& value_out) {
  DWORD value_size_bytes = 0;
  DWORD value_type = 0;
  RETURN_IF(::RegQueryValueExW(key, value_name, nullptr, &value_type, nullptr, &value_size_bytes) != ERROR_SUCCESS,
            ("Failed to query registry value size for '" +
             std::filesystem::path(value_name).string() + "'.")
                .c_str());
  RETURN_IF(value_type != REG_SZ && value_type != REG_EXPAND_SZ,
            ("Registry value '" + std::filesystem::path(value_name).string() +
             "' has unexpected type " + std::to_string(value_type) + ".")
                .c_str());

  // value_size_bytes includes the terminating null if it was stored; allocate an extra
  // wchar_t so the result is always null-terminated regardless of how it was written.
  std::vector<wchar_t> buffer(value_size_bytes / sizeof(wchar_t) + 1, L'\0');
  RETURN_IF(::RegQueryValueExW(key, value_name, nullptr, nullptr,
                               reinterpret_cast<LPBYTE>(buffer.data()), &value_size_bytes) != ERROR_SUCCESS,
            ("Failed to read registry value '" +
             std::filesystem::path(value_name).string() + "'.")
                .c_str());

  // RegQueryValueExW never auto-expands, so a REG_EXPAND_SZ value comes back with its %VAR%
  // tokens intact. Expand them here. Note this does not touch the NT-namespace "\SystemRoot"
  // prefix (not a %VAR% form), which the caller resolves separately.
  if (value_type == REG_EXPAND_SZ) {
    const DWORD expanded_size = ::ExpandEnvironmentStringsW(buffer.data(), nullptr, 0);
    RETURN_IF(expanded_size == 0,
              ("Failed to expand registry value '" + std::filesystem::path(value_name).string() +
               "'. ExpandEnvironmentStringsW error: " + std::to_string(::GetLastError()))
                  .c_str());
    std::vector<wchar_t> expanded(expanded_size, L'\0');
    RETURN_IF(::ExpandEnvironmentStringsW(buffer.data(), expanded.data(), expanded_size) == 0,
              ("Failed to expand registry value '" + std::filesystem::path(value_name).string() +
               "'. ExpandEnvironmentStringsW error: " + std::to_string(::GetLastError()))
                  .c_str());
    value_out = std::wstring{expanded.data()};
    return Ort::Status();
  }

  value_out = std::wstring{buffer.data()};
  return Ort::Status();
}

// Resolve the MCDM driver service name from its device-interface class GUID via the
// Configuration Manager (CfgMgr32), which is AppContainer-safe unlike SetupAPI or a direct
// read of the ACL-locked HKLM\SYSTEM\CurrentControlSet\Enum subtree:
//   interface GUID -> instance id (DEVPKEY_Device_InstanceId) -> devnode -> DEVPKEY_Device_Service.
Ort::Status GetMcdmServiceName(std::wstring& service_name_out) {
  // MCDM device-interface class GUID: {171b1d2d-1466-4c42-a65d-623455547fa1}
  const GUID mcdm_interface_class_guid = {
      0x171b1d2d, 0x1466, 0x4c42, {0xa6, 0x5d, 0x62, 0x34, 0x55, 0x54, 0x7f, 0xa1}};

  // Retrieve the double-null-terminated list of present device interfaces for the class.
  ULONG interface_list_size = 0;
  CONFIGRET cr = ::CM_Get_Device_Interface_List_SizeW(
      &interface_list_size, const_cast<LPGUID>(&mcdm_interface_class_guid), nullptr,
      CM_GET_DEVICE_INTERFACE_LIST_PRESENT);
  RETURN_IF(cr != CR_SUCCESS,
            ("Failed to get MCDM device interface list size. CM_Get_Device_Interface_List_SizeW CONFIGRET: " +
             std::to_string(cr))
                .c_str());
  // The size counts the double-null-terminated list, so an empty list is 1 wchar_t (the extra
  // terminator); fewer than 2 means no MCDM device interface is present.
  RETURN_IF(interface_list_size < 2,
            "No MCDM device interface found. Is the QNN MCDM driver installed?");

  std::vector<wchar_t> interface_list(interface_list_size, L'\0');
  cr = ::CM_Get_Device_Interface_ListW(
      const_cast<LPGUID>(&mcdm_interface_class_guid), nullptr, interface_list.data(), interface_list_size,
      CM_GET_DEVICE_INTERFACE_LIST_PRESENT);
  RETURN_IF(cr != CR_SUCCESS,
            ("Failed to get MCDM device interface list. CM_Get_Device_Interface_ListW CONFIGRET: " +
             std::to_string(cr))
                .c_str());

  // Walk each interface symbolic link until one resolves to a service.
  for (const wchar_t* interface_symlink = interface_list.data();
       *interface_symlink != L'\0';
       interface_symlink += ::wcslen(interface_symlink) + 1) {
    DEVPROPTYPE property_type = 0;
    wchar_t device_instance_id[MAX_DEVICE_ID_LEN]{};
    ULONG device_instance_id_size = sizeof(device_instance_id);
    cr = ::CM_Get_Device_Interface_PropertyW(
        interface_symlink, &DEVPKEY_Device_InstanceId, &property_type,
        reinterpret_cast<PBYTE>(device_instance_id), &device_instance_id_size, 0);
    if (cr != CR_SUCCESS || property_type != DEVPROP_TYPE_STRING) {
      continue;
    }

    DEVINST devinst = 0;
    cr = ::CM_Locate_DevNodeW(&devinst, device_instance_id, CM_LOCATE_DEVNODE_NORMAL);
    if (cr != CR_SUCCESS) {
      continue;
    }

    property_type = 0;
    wchar_t service_name[256]{};
    ULONG service_name_size = sizeof(service_name);
    cr = ::CM_Get_DevNode_PropertyW(
        devinst, &DEVPKEY_Device_Service, &property_type,
        reinterpret_cast<PBYTE>(service_name), &service_name_size, 0);
    if (cr != CR_SUCCESS || property_type != DEVPROP_TYPE_STRING || service_name[0] == L'\0') {
      continue;
    }

    service_name_out = std::wstring{service_name};
    return Ort::Status();
  }

  return MAKE_FAIL("Failed to resolve MCDM service name from the device interface class GUID.");
}

Ort::Status ReadEnvironmentVariable(const wchar_t* name, std::wstring& value_out) {
  const DWORD value_size = ::GetEnvironmentVariableW(name, nullptr, 0);
  RETURN_IF(value_size == 0,
            ("Failed to get environment variable length. GetEnvironmentVariableW error: " +
             std::to_string(::GetLastError()))
                .c_str());

  std::vector<wchar_t> value(value_size);

  RETURN_IF(::GetEnvironmentVariableW(name, value.data(), value_size) == 0,
            ("Failed to get environment variable value. GetEnvironmentVariableW error: " +
             std::to_string(::GetLastError()))
                .c_str());

  value_out = std::wstring{value.data()};
  return Ort::Status();
}

Ort::Status GetServiceBinaryDirectoryPath(const wchar_t* service_name,
                                          std::filesystem::path& service_binary_directory_path_out) {
  // Read ImagePath directly from the registry instead of via the Service Control Manager,
  // which is restricted in sandboxed environments. KEY_READ is sufficient.
  const std::wstring reg_key_path =
      std::wstring(L"SYSTEM\\CurrentControlSet\\Services\\") + service_name;

  HKEY service_key_raw = nullptr;
  RETURN_IF(::RegOpenKeyExW(HKEY_LOCAL_MACHINE, reg_key_path.c_str(),
                            0, KEY_READ, &service_key_raw) != ERROR_SUCCESS,
            ("Failed to open registry key 'HKLM\\" +
             std::filesystem::path(reg_key_path).string() +
             "'. RegOpenKeyExW error: " + std::to_string(::GetLastError()))
                .c_str());
  auto service_key = UniqueHKey{service_key_raw};

  std::wstring service_binary_path_name{};
  RETURN_IF_ERROR(ReadRegistryStringValue(service_key.get(), L"ImagePath", service_binary_path_name));

  // replace system root placeholder with the value of the SYSTEMROOT environment variable
  const std::wstring system_root_placeholder = L"\\SystemRoot";

  RETURN_IF(service_binary_path_name.find(system_root_placeholder, 0) != 0,
            ("Service binary path '" + std::filesystem::path(service_binary_path_name).string() +
             "' does not start with expected system root placeholder value '" +
             std::filesystem::path(system_root_placeholder).string() + "'.")
                .c_str());

  std::wstring system_root{};
  RETURN_IF_ERROR(ReadEnvironmentVariable(L"SYSTEMROOT", system_root));
  service_binary_path_name.replace(0, system_root_placeholder.size(), system_root);

  const auto service_binary_path = std::filesystem::path{service_binary_path_name};
  auto service_binary_directory_path = service_binary_path.parent_path();

  RETURN_IF(!std::filesystem::exists(service_binary_directory_path),
            ("Service binary directory path does not exist: " + service_binary_directory_path.string()).c_str());

  service_binary_directory_path_out = std::move(service_binary_directory_path);
  return Ort::Status();
}

// Resolve the MCDM driver package directory (contains libcdsprpc.dll and the htp\ subfolder),
// falling back to the well-known "qcnspmcdm" service name if GUID resolution fails.
Ort::Status GetMcdmDriverDirectoryPath(std::filesystem::path& driver_directory_path_out) {
  std::wstring service_name{};
  if (!GetMcdmServiceName(service_name).IsOK() || service_name.empty()) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_WARNING,
                "QNN driver lookup: MCDM service name discovery failed; falling back to 'qcnspmcdm'.");
    service_name = L"qcnspmcdm";
  }

  return GetServiceBinaryDirectoryPath(service_name.c_str(), driver_directory_path_out);
}

#endif  // defined(_WIN32)

Ort::Status GetRpcMemDynamicLibraryPath(std::basic_string<ORTCHAR_T>& path_out) {
#if defined(_WIN32)

  std::filesystem::path mcdm_dir_path{};
  RETURN_IF_ERROR(GetMcdmDriverDirectoryPath(mcdm_dir_path));
  path_out = (mcdm_dir_path / L"libcdsprpc.dll").wstring();
  return Ort::Status();

#else  // ^^^ defined(_WIN32) / vvv !defined(_WIN32)

  path_out = ORT_TSTR("libcdsprpc.so");
  return Ort::Status();

#endif  // !defined(_WIN32)
}

Ort::Status LoadDynamicLibrary(const std::basic_string<ORTCHAR_T>& path, bool global_symbols,
                               UniqueDynamicLibraryHandle& library_handle_out) {
  void* library_handle_raw = nullptr;
  RETURN_IF_ERROR(OrtLoadDynamicLibrary(path, global_symbols, &library_handle_raw));

  library_handle_out = UniqueDynamicLibraryHandle{library_handle_raw};
  return Ort::Status();
}

UniqueDynamicLibraryHandle GetRpcMemDynamicLibraryHandle() {
  const std::string error_message_prefix = "Failed to initialize RPCMEM dynamic library handle: ";

  std::basic_string<ORTCHAR_T> rpcmem_library_path{};
  auto status = GetRpcMemDynamicLibraryPath(rpcmem_library_path);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW(error_message_prefix + status.GetErrorMessage(), ORT_RUNTIME_EXCEPTION);
  }

  UniqueDynamicLibraryHandle library_handle{};
  status = LoadDynamicLibrary(rpcmem_library_path, /* global_symbols */ false, library_handle);
  if (!status.IsOK()) {
    ORT_CXX_API_THROW(error_message_prefix + status.GetErrorMessage(), ORT_RUNTIME_EXCEPTION);
  }

  return library_handle;
}

RpcMemApi CreateApi(void* library_handle) {
  RpcMemApi api{};

  if (!OrtGetSymbolFromLibrary(library_handle, "rpcmem_alloc", (void**)&api.alloc).IsOK()) {
    ORT_CXX_API_THROW("Failed to get symbol rpcmem_alloc.", ORT_RUNTIME_EXCEPTION);
  }

  if (!OrtGetSymbolFromLibrary(library_handle, "rpcmem_free", (void**)&api.free).IsOK()) {
    ORT_CXX_API_THROW("Failed to get symbol rpcmem_free.", ORT_RUNTIME_EXCEPTION);
  }

  if (!OrtGetSymbolFromLibrary(library_handle, "rpcmem_to_fd", (void**)&api.to_fd).IsOK()) {
    ORT_CXX_API_THROW("Failed to get symbol rpcmem_to_fd.", ORT_RUNTIME_EXCEPTION);
  }

  if (!OrtGetSymbolFromLibrary(library_handle, "remote_register_buf_attr2", (void**)&api.register_buf).IsOK()) {
    ORT_CXX_API_THROW("Failed to get symbol remote_register_buf_attr2.", ORT_RUNTIME_EXCEPTION);
  }

  return api;
}

}  // namespace

RpcMemLibrary::RpcMemLibrary()
    : library_handle_(GetRpcMemDynamicLibraryHandle()),
      api_{CreateApi(library_handle_.get())} {
}

}  // namespace onnxruntime::qnn
