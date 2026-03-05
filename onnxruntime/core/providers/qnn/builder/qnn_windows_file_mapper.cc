// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

#include <wil/filesystem.h>

#include <sstream>
#include <utility>

#include "QnnContext.h"

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

static void UnmapFile(void* addr, const Ort::Logger& logger) noexcept {
  bool successful = UnmapViewOfFile(addr);
  if (!successful) {
    const auto error_code = GetLastError();
    std::ostringstream oss;
    oss << "Failed to unmap view of file with ptr: " << addr
        << ", Error code: " << error_code << ", \""
        << std::system_category().message(error_code) << "\"";
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
  }
}

Ort::Status WindowsFileMapper::GetContextBinMappedMemoryPtr(const std::string& bin_filepath, void** mapped_data_ptr) {
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, ("Creating context bin file mapping for " + bin_filepath).c_str());
  RETURN_IF(bin_filepath.empty(), "Context bin file path is empty");

  std::lock_guard<std::mutex> lock(map_mutex_);
  std::ostringstream oss;

  auto map_it = mapped_memory_ptrs_.find(bin_filepath);
  if (map_it != mapped_memory_ptrs_.end()) {
    *mapped_data_ptr = map_it->second.get();

    oss.clear();
    oss << "Found existing mapview memory pointer (" << mapped_data_ptr << ") for context bin file: " << bin_filepath;
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, oss.str().c_str());

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
    return MAKE_EP_FAIL(("Failed to create file handle for context bin" + bin_filepath +
                         ". Error code: " + std::to_string(error_code) + ", \"" +
                         std::system_category().message(error_code) + "\"")
                            .c_str());
  }

  oss.clear();
  oss << "Created file handle (" << file_handle.get() << ") for context bin:" << bin_filepath;
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, oss.str().c_str());

  wil::unique_hfile file_mapping_handle{CreateFileMappingW(file_handle.get(),
                                                           nullptr,
                                                           PAGE_READONLY,
                                                           0x00,
                                                           0x00,
                                                           nullptr)};
  if (file_mapping_handle.get() == INVALID_HANDLE_VALUE) {
    const auto error_code = GetLastError();
    return MAKE_EP_FAIL(("Failed to create file mapping handle for context bin" +
                         bin_filepath + ". Error code: " + std::to_string(error_code) + ", \"" +
                         std::system_category().message(error_code) + "\"")
                            .c_str());
  }

  oss.clear();
  oss << "Created file mapping with handle (" << file_mapping_handle.get() << ") for context bin:" << bin_filepath;
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, oss.str().c_str());

  void* const mapped_base_ptr = MapViewOfFile(file_mapping_handle.get(),
                                              FILE_MAP_READ,
                                              0, 0, 0);

  if (mapped_base_ptr == nullptr) {
    const auto error_code = GetLastError();
    return MAKE_EP_FAIL(("Failed to retrieve mapview pointer for context bin" +
                         bin_filepath + ". Error code: " + std::to_string(error_code) + ", \"" +
                         std::system_category().message(error_code) + "\"")
                            .c_str());
  }

  oss.clear();
  oss << "Created mapview pointer with address " << mapped_base_ptr << " for context bin " << bin_filepath;
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, oss.str().c_str());

  MappedMemoryPtr mapped_memory_ptr{reinterpret_cast<char*>(mapped_base_ptr),
                                    [mapped_base_ptr, &logger = logger_](void*) {
                                      UnmapFile(mapped_base_ptr, logger);
                                    }};

  *mapped_data_ptr = mapped_memory_ptr.get();
  mapped_memory_ptrs_.emplace(bin_filepath, std::move(mapped_memory_ptr));

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime

#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
