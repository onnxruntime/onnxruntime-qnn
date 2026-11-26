// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/ep_context_io_dispatch.h"
#include "core/providers/qnn/builder/qnn_def.h"

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION && defined(_WIN32)
#define QNN_EP_DECRYPT_SOURCE_IMPLEMENTATION
#include "core/providers/qnn/builder/qnn_context_decrypt_source.h"
#undef QNN_EP_DECRYPT_SOURCE_IMPLEMENTATION
#if defined(QNN_FILE_MAPPED_WEIGHTS_AVAILABLE)
#include <filesystem>
#include <memory>
#include <windows.h>
#include <wil/resource.h>
#endif

struct QnnEpDecryptSourceBinding {
  const OrtReadNamedBufferFunc legacy_read;
  const QnnEpDecryptFromSourceFunc decrypt_from_source;
  void* const app_state;
};

namespace {
OrtStatus* SourceBindingError(const char* message, OrtErrorCode code = ORT_FAIL) noexcept {
  const auto& ort_api = Ort::GetApi();
  return ort_api.CreateStatus(code, message);
}

OrtStatus* ORT_API_CALL ReadDecryptSourceBridge(void* state, const char* name,
                                                OrtAllocator* allocator, void** output,
                                                size_t* output_bytes) {
  try {
    const auto& binding = *static_cast<const QnnEpDecryptSourceBinding*>(state);
    return binding.legacy_read(binding.app_state, name, allocator, output, output_bytes);
  } catch (const std::exception& e) {
    return SourceBindingError(e.what());
  } catch (...) {
    return SourceBindingError("Legacy decrypt callback threw an exception.");
  }
}
#if defined(QNN_FILE_MAPPED_WEIGHTS_AVAILABLE)
struct UnmapSourceView {
  void operator()(const char* address) const noexcept {
    UnmapViewOfFile(address);
  }
};

using SourceView = std::unique_ptr<const char, UnmapSourceView>;

Ort::Status MapSource(const std::string& validated_utf8_path, SourceView& view, size_t& source_bytes) {
  RETURN_IF(validated_utf8_path.empty() || validated_utf8_path.find('\0') != std::string::npos,
            "Decrypt source path is empty or contains a null character.");
  try {
    const auto path = std::filesystem::u8path(validated_utf8_path);
    wil::unique_hfile file{CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)};
    if (!file) {
      const auto error = GetLastError();
      return Ort::Status(("Failed to open decrypt source: " + validated_utf8_path +
                          ", Windows error: " + std::to_string(error))
                             .c_str(),
                         ORT_FAIL);
    }
    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file.get(), &file_size)) {
      const auto error = GetLastError();
      return Ort::Status(("Failed to get decrypt source size: " + validated_utf8_path +
                          ", Windows error: " + std::to_string(error))
                             .c_str(),
                         ORT_FAIL);
    }
    RETURN_IF(file_size.QuadPart <= 0, "Decrypt source file is empty.");

    wil::unique_handle mapping{CreateFileMappingW(file.get(), nullptr, PAGE_READONLY, 0, 0, nullptr)};
    if (!mapping) {
      const auto error = GetLastError();
      return Ort::Status(("Failed to map decrypt source: " + validated_utf8_path +
                          ", Windows error: " + std::to_string(error))
                             .c_str(),
                         ORT_FAIL);
    }
    SourceView mapped{static_cast<const char*>(MapViewOfFile(mapping.get(), FILE_MAP_READ, 0, 0, 0))};
    if (!mapped) {
      const auto error = GetLastError();
      return Ort::Status(("Failed to view decrypt source: " + validated_utf8_path +
                          ", Windows error: " + std::to_string(error))
                             .c_str(),
                         ORT_FAIL);
    }
    view = std::move(mapped);
    source_bytes = static_cast<size_t>(file_size.QuadPart);
    return Ort::Status();
  } catch (const std::exception& e) {
    return Ort::Status(e.what(), ORT_FAIL);
  }
}
#endif
}  // namespace

extern "C" OrtStatus* ORT_API_CALL QnnEpCreateDecryptSourceBinding(
    OrtReadNamedBufferFunc legacy_read, QnnEpDecryptFromSourceFunc decrypt_from_source,
    void* app_state, QnnEpDecryptSourceBinding** binding,
    OrtReadNamedBufferFunc* read_bridge) noexcept {
  if (!binding || !read_bridge) {
    return SourceBindingError("Source binding outputs are required.", ORT_INVALID_ARGUMENT);
  }
  *binding = nullptr;
  *read_bridge = nullptr;
  if (!legacy_read || !decrypt_from_source) {
    return SourceBindingError("Both legacy and source decrypt callbacks are required.", ORT_INVALID_ARGUMENT);
  }
  try {
    *binding = new QnnEpDecryptSourceBinding{legacy_read, decrypt_from_source, app_state};
    *read_bridge = ReadDecryptSourceBridge;
    return nullptr;
  } catch (const std::exception& e) {
    return SourceBindingError(e.what());
  }
}

extern "C" void ORT_API_CALL QnnEpReleaseDecryptSourceBinding(
    QnnEpDecryptSourceBinding* binding) noexcept {
  delete binding;
}
#endif

namespace onnxruntime {
namespace qnn {

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION

EpContextIoDispatch::EpContextIoDispatch() noexcept : config_{nullptr} {}

EpContextIoDispatch::EpContextIoDispatch(const OrtSessionOptions& session_options,
                                         const OrtApiRuntimeCapabilities& runtime_capabilities,
                                         bool map_decrypt_source,
                                         const Ort::Logger* logger) noexcept
    : config_{nullptr}, map_decrypt_source_(map_decrypt_source) {
  // The named capability is negotiated with the actual runtime, not inferred from the
  // compile-time ORT_API_VERSION. Skip APIs absent from an older runtime's OrtApi struct.
  if (!runtime_capabilities.SupportsEpContextEncryption()) {
    ORT_CXX_LOG_PTR(logger, ORT_LOGGING_LEVEL_WARNING,
                    "ORT runtime API version is below 28 (v1.28). EPContext encryption "
                    "callbacks are unavailable; upgrade to onnxruntime >= 1.28 to enable.");
    return;
  }

  try {
    config_ = Ort::Experimental::EpContextConfig(Ort::ConstSessionOptions(&session_options));
  } catch (const Ort::Exception& e) {
    // Can't distinguish "app didn't ask" from "runtime/build API skew" here, so only warn.
    ORT_CXX_LOG_PTR(logger, ORT_LOGGING_LEVEL_WARNING,
                    ("EPContext encryption config unavailable (" + std::string(e.what()) +
                     "); EPContext read/write callbacks will not be used for this session.")
                        .c_str());
    return;
  }
  try {
    config_.GetReadFunc(read_fn_, read_state_);
  } catch (const Ort::Exception& e) {
    read_fn_ = nullptr;
    read_state_ = nullptr;
    ORT_CXX_LOG_PTR(logger, ORT_LOGGING_LEVEL_WARNING,
                    ("Failed to resolve EPContext read callback (" + std::string(e.what()) +
                     "); falling back to plaintext disk read for this session.")
                        .c_str());
  }
  try {
    config_.GetWriteFunc(write_fn_, write_state_);
  } catch (const Ort::Exception& e) {
    write_fn_ = nullptr;
    write_state_ = nullptr;
    ORT_CXX_LOG_PTR(logger, ORT_LOGGING_LEVEL_WARNING,
                    ("Failed to resolve EPContext write callback (" + std::string(e.what()) +
                     "); falling back to plaintext disk write for this session.")
                        .c_str());
  }
}

bool EpContextIoDispatch::HasReadCallback() const noexcept {
  return read_fn_ != nullptr;
}

bool EpContextIoDispatch::HasWriteCallback() const noexcept {
  return write_fn_ != nullptr;
}

bool EpContextIoDispatch::HasSourceReadCallback() const noexcept {
#if defined(_WIN32) && QNN_ARCH_ARM64
  return read_fn_ == ReadDecryptSourceBridge && read_state_ != nullptr;
#else
  return false;
#endif
}

Ort::Status EpContextIoDispatch::Read(const std::string& file_name, EpContextBuffer& out,
                                      [[maybe_unused]] const std::string& validated_external_path) const {
  RETURN_IF_NOT(HasReadCallback(), "EPContext read callback is unavailable.");

  const auto& ort_api = Ort::GetApi();
  OrtAllocator* allocator = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetAllocatorWithDefaultOptions(&allocator));

  void* data = nullptr;
  size_t data_size = 0;
  OrtStatus* s = nullptr;
#if defined(QNN_FILE_MAPPED_WEIGHTS_AVAILABLE)
  SourceView source;
#endif
  if (map_decrypt_source_) {
    RETURN_IF_NOT(HasSourceReadCallback(), "Source mapping requires the private source-aware decrypt binding.");
#if defined(QNN_FILE_MAPPED_WEIGHTS_AVAILABLE)
    const auto& binding = *static_cast<const QnnEpDecryptSourceBinding*>(read_state_);
    size_t source_bytes = 0;
    RETURN_IF_ERROR(MapSource(validated_external_path, source, source_bytes));
    try {
      s = binding.decrypt_from_source(binding.app_state, file_name.c_str(), source.get(), source_bytes,
                                      allocator, &data, &data_size);
    } catch (const std::exception& e) {
      return Ort::Status(e.what(), ORT_FAIL);
    } catch (...) {
      return Ort::Status("Source decrypt callback threw an exception.", ORT_FAIL);
    }
#else
    return Ort::Status("Decrypt source mapping requires Windows ARM64/HTP mapping support.", ORT_NOT_IMPLEMENTED);
#endif
  } else {
    s = read_fn_(read_state_, file_name.c_str(), allocator, &data, &data_size);
  }
  if (s != nullptr) {
    return Ort::Status(s);
  }
  if (data == nullptr || data_size == 0) {
    if (data != nullptr) {
      allocator->Free(allocator, data);
    }
    return MAKE_FAIL(
        ("OrtReadNamedBufferFunc returned invalid buffer for " + file_name).c_str());
  }
  out.Adopt(data, data_size, allocator);
  return Ort::Status();
}

Ort::Status EpContextIoDispatch::Write(const std::string& name,
                                       const void* buffer,
                                       size_t buffer_size,
                                       const Ort::Logger& logger) const {
  RETURN_IF_NOT(HasWriteCallback(), "EPContext write callback is unavailable.");

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
              ("EPContext data forwarded to App write callback: " + name).c_str());
  if (auto* s = write_fn_(write_state_, name.c_str(), buffer, buffer_size)) {
    return Ort::Status(s);
  }
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
              ("EPContext data written by App write callback: " +
               std::to_string(buffer_size) + " bytes")
                  .c_str());
  return Ort::Status();
}

#else  // ORT_API_HAS_EPCONTEXT_ENCRYPTION — stub for pre-v28 ORT

EpContextIoDispatch::EpContextIoDispatch() noexcept {}

EpContextIoDispatch::EpContextIoDispatch(const OrtSessionOptions& /*session_options*/,
                                         const OrtApiRuntimeCapabilities& /*runtime_capabilities*/,
                                         bool /*map_decrypt_source*/,
                                         const Ort::Logger* /*logger*/) noexcept {}

bool EpContextIoDispatch::HasReadCallback() const noexcept { return false; }
bool EpContextIoDispatch::HasWriteCallback() const noexcept { return false; }
bool EpContextIoDispatch::HasSourceReadCallback() const noexcept { return false; }

Ort::Status EpContextIoDispatch::Read(const std::string& /*file_name*/,
                                      EpContextBuffer& /*out*/,
                                      const std::string& /*validated_external_path*/) const {
  return MAKE_EP_FAIL("EPContext encryption unavailable in this build.");
}

Ort::Status EpContextIoDispatch::Write(const std::string& /*name*/,
                                       const void* /*buffer*/,
                                       size_t /*buffer_size*/,
                                       const Ort::Logger& /*logger*/) const {
  return MAKE_EP_FAIL("EPContext encryption unavailable in this build.");
}

#endif  // ORT_API_HAS_EPCONTEXT_ENCRYPTION

}  // namespace qnn
}  // namespace onnxruntime
