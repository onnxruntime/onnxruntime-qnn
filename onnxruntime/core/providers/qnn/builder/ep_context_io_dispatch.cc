// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/ep_context_io_dispatch.h"
#ifdef _WIN32
#include <windows.h>
#endif

namespace onnxruntime {
namespace qnn {

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION

#ifdef _WIN32
// Anonymous mapped storage keeps plaintext out of an application-created file.
// The application decrypts directly into this allocation using the existing ABI.
struct MappedContextAllocator : OrtAllocator {
  Ort::AllocatorWithDefaultOptions info_allocator;
  MappedContextAllocator() : OrtAllocator{} {
    version = ORT_API_VERSION;
    Alloc = [](OrtAllocator*, size_t size) -> void* {
      if (size == 0) return nullptr;
      const uint64_t length = size;
      HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                          static_cast<DWORD>(length >> 32), static_cast<DWORD>(length), nullptr);
      if (!mapping) return nullptr;
      void* data = MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, size);
      CloseHandle(mapping);
      return data;
    };
    Free = [](OrtAllocator*, void* data) { if (data) UnmapViewOfFile(data); };
    Info = [](const OrtAllocator* self) -> const OrtMemoryInfo* {
      return static_cast<const MappedContextAllocator*>(self)->info_allocator.GetInfo();
    };
    Reserve = nullptr;
  }
};
#endif

EpContextIoDispatch::EpContextIoDispatch() noexcept : config_{nullptr} {}

EpContextIoDispatch::EpContextIoDispatch(const OrtSessionOptions& session_options,
                                         const OrtApiRuntimeCapabilities& runtime_capabilities,
                                         const Ort::Logger* logger) noexcept
    : config_{nullptr} {
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

Ort::Status EpContextIoDispatch::Read(const std::string& file_name, EpContextBuffer& out, bool mapped_allocation) const {
  RETURN_IF_NOT(HasReadCallback(), "EPContext read callback is unavailable.");

  std::shared_ptr<OrtAllocator> allocator;
#ifdef _WIN32
  if (mapped_allocation) allocator = std::make_shared<MappedContextAllocator>();
#else
  (void)mapped_allocation;
#endif
  if (!allocator) {
    auto default_allocator = std::make_shared<Ort::AllocatorWithDefaultOptions>();
    allocator = std::shared_ptr<OrtAllocator>(default_allocator, *default_allocator);
  }
  void* data = nullptr;
  size_t data_size = 0;
  OrtStatus* s = read_fn_(read_state_, file_name.c_str(), allocator.get(), &data, &data_size);
  // Preserve the existing callback contract: outputs are transferred only on
  // success. On error the application remains responsible for its allocations.
  if (s != nullptr) {
    return Ort::Status(s);
  }
  EpContextBuffer owned;
  if (data != nullptr) owned.Adopt(data, data_size, allocator);
  if (data == nullptr || data_size == 0) {
    return Ort::Status(
        ("OrtReadNamedBufferFunc returned invalid buffer for " + file_name).c_str(), ORT_FAIL);
  }
  out = std::move(owned);
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
                                         const Ort::Logger* /*logger*/) noexcept {}

bool EpContextIoDispatch::HasReadCallback() const noexcept { return false; }
bool EpContextIoDispatch::HasWriteCallback() const noexcept { return false; }

Ort::Status EpContextIoDispatch::Read(const std::string& /*file_name*/,
                                      EpContextBuffer& /*out*/, bool /*mapped_allocation*/) const {
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
