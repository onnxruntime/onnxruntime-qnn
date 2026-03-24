// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

#include <wil/filesystem.h>

#include <utility>

#include <QnnContext.h>
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

WindowsFileMapper::WindowsFileMapper(const Ort::Logger& logger)
    : logger_(logger) {
}

WindowsFileMapper::~WindowsFileMapper() {
}

static void UnmapFile(void* addr) noexcept {
  bool successful = UnmapViewOfFile(addr);
  if (!successful) {
    const auto error_code = GetLastError();
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                ORT_LOGGING_LEVEL_ERROR,
                ("Failed to unmap view of file with ptr: " + utils::PtrToString(addr) + ", Error code: " + std::to_string(error_code) + ", \"" + std::system_category().message(error_code) + "\"").c_str());
  }
}

Ort::Status WindowsFileMapper::GetContextBinMappedMemoryPtr(const std::string& bin_filepath,
                                                            void** mapped_data_ptr) {
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, ("Creating context bin file mapping for " + bin_filepath).c_str());

  RETURN_IF(bin_filepath.empty(), "Context bin file path is empty");

  std::lock_guard<std::mutex> lock(map_mutex_);
  auto map_it = mapped_memory_ptrs_.find(bin_filepath);
  if (map_it != mapped_memory_ptrs_.end()) {
    *mapped_data_ptr = map_it->second.get();
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_INFO,
                ("Found existing mapview memory pointer (" + utils::PtrToString(mapped_data_ptr) + ") for context bin file: " + bin_filepath).c_str());
    return Ort::Status();
  }

  std::wstring bin_filepath_wstr(bin_filepath.begin(), bin_filepath.end());
  wil::unique_hfile file_handle{CreateFile2(bin_filepath_wstr.c_str(),
                                            GENERIC_READ,
                                            FILE_SHARE_READ,
                                            OPEN_EXISTING,
                                            NULL)};
  if (file_handle.get() == INVALID_HANDLE_VALUE) {
    const auto error_code = GetLastError();
    return MAKE_FAIL(("Failed to create file handle for context bin" + bin_filepath + ". Error code: " + std::to_string(error_code) + ", \"" + std::system_category().message(error_code) + "\"").c_str());
  }

  ORT_CXX_LOG(logger_,
              ORT_LOGGING_LEVEL_VERBOSE, ("Created file handle (" + utils::PtrToString(file_handle.get()) + ") for context bin: " + bin_filepath).c_str());

  wil::unique_hfile file_mapping_handle{CreateFileMappingW(file_handle.get(),
                                                           nullptr,
                                                           PAGE_READONLY,
                                                           0x00,
                                                           0x00,
                                                           nullptr)};
  if (file_mapping_handle.get() == INVALID_HANDLE_VALUE) {
    const auto error_code = GetLastError();
    return MAKE_FAIL(("Failed to create file mapping handle for context bin" + bin_filepath + ". Error code: " + std::to_string(error_code) + ", \"" + std::system_category().message(error_code) + "\"").c_str());
  }

  ORT_CXX_LOG(logger_,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("Created file mapping with handle (" + utils::PtrToString(file_mapping_handle.get()) + ") for context bin:" + bin_filepath).c_str());

  void* const mapped_base_ptr = MapViewOfFile(file_mapping_handle.get(),
                                              FILE_MAP_READ,
                                              0, 0, 0);

  if (mapped_base_ptr == nullptr) {
    const auto error_code = GetLastError();
    return MAKE_FAIL(("Failed to retrieve mapview pointer for context bin" + bin_filepath + ". Error code: " + std::to_string(error_code) + ", \"" + std::system_category().message(error_code) + "\"").c_str());
  }

  ORT_CXX_LOG(logger_,
              ORT_LOGGING_LEVEL_INFO,
              ("Created mapview pointer with address " + utils::PtrToString(mapped_base_ptr) + " for context bin " + bin_filepath).c_str());

  MappedMemoryPtr mapped_memory_ptr{reinterpret_cast<char*>(mapped_base_ptr),
                                    [mapped_base_ptr](void*) {
                                      UnmapFile(mapped_base_ptr);
                                    }};

  *mapped_data_ptr = mapped_memory_ptr.get();
  mapped_memory_ptrs_.emplace(bin_filepath, std::move(mapped_memory_ptr));

  return Ort::Status();
}
}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
