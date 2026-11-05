// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/ep_context_io_dispatch.h"

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION && defined(_WIN32)
#define QNN_EP_DECRYPT_SOURCE_POC_IMPLEMENTATION
#include "core/providers/qnn/builder/qnn_context_decrypt_source_poc.h"
#undef QNN_EP_DECRYPT_SOURCE_POC_IMPLEMENTATION
#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"

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
                                         const Ort::Logger* logger,
                                         bool map_decrypt_source) noexcept
    : map_decrypt_source_(map_decrypt_source), logger_(logger), config_{nullptr} {
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
#if defined(_WIN32)
  return read_fn_ == ReadDecryptSourceBridge && read_state_ != nullptr;
#else
  return false;
#endif
}

Ort::Status EpContextIoDispatch::Read(const std::string& file_name, EpContextBuffer& out,
                                      const std::string& validated_external_path) const {
  RETURN_IF_NOT(HasReadCallback(), "EPContext read callback is unavailable.");

  auto allocator = std::make_shared<Ort::AllocatorWithDefaultOptions>();
  void* data = nullptr;
  size_t data_size = 0;
  OrtStatus* s = nullptr;
  if (map_decrypt_source_) {
    RETURN_IF_NOT(HasSourceReadCallback(), "Source mapping requires the private source-aware decrypt binding.");
#if defined(QNN_FILE_MAPPED_WEIGHTS_AVAILABLE)
    const auto& binding = *static_cast<const QnnEpDecryptSourceBinding*>(read_state_);
    RETURN_IF_NOT(logger_ || OrtLoggingManager::GetDefaultLoggerPtr(),
                  "Decrypt source mapping requires a session or registered default logger.");
    WindowsFileMapper mapper(logger_ ? *logger_ : OrtLoggingManager::GetDefaultLogger());
    std::shared_ptr<char> source;
    size_t source_bytes = 0;
    RETURN_IF_ERROR(mapper.MapContextBinary(validated_external_path, source, source_bytes));
    try {
      s = binding.decrypt_from_source(binding.app_state, file_name.c_str(), source.get(), source_bytes,
                                      *allocator, &data, &data_size);
    } catch (const std::exception& e) {
      return Ort::Status(e.what(), ORT_FAIL);
    } catch (...) {
      return Ort::Status("Source decrypt callback threw an exception.", ORT_FAIL);
    }
    // All callback worker reads must finish before return; release ciphertext now.
#else
    return Ort::Status("Decrypt source mapping requires Windows ARM64/HTP mapping support.", ORT_NOT_IMPLEMENTED);
#endif
  } else {
    s = read_fn_(read_state_, file_name.c_str(), *allocator, &data, &data_size);
  }
  if (s != nullptr) {
    return Ort::Status(s);
  }
  if (data == nullptr || data_size == 0) {
    if (data != nullptr) {
      allocator->Free(data);
    }
    return Ort::Status(
        ("OrtReadNamedBufferFunc returned invalid buffer for " + file_name).c_str(), ORT_FAIL);
  }
  out.Adopt(data, data_size, std::shared_ptr<OrtAllocator>(allocator, *allocator));
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
                                         const Ort::Logger* logger,
                                         bool map_decrypt_source) noexcept
    : map_decrypt_source_(map_decrypt_source), logger_(logger) {}

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
