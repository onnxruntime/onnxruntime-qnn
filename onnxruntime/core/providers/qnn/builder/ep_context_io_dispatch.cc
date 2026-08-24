// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/ep_context_io_dispatch.h"

namespace onnxruntime {
namespace qnn {

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION

EpContextIoDispatch::EpContextIoDispatch(const OrtSessionOptions* session_options,
                                         const Ort::Logger* logger) noexcept
    : config_{nullptr} {
  try {
    config_ = Ort::Experimental::EpContextConfig(Ort::ConstSessionOptions(session_options));
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

Ort::Status EpContextIoDispatch::Read(const std::string& file_name, std::vector<char>& out) const {
  Ort::AllocatorWithDefaultOptions allocator;
  void* data = nullptr;
  size_t data_size = 0;
  OrtStatus* s = read_fn_(read_state_, file_name.c_str(), allocator, &data, &data_size);
  if (s != nullptr) {
    return Ort::Status(s);
  }
  if (data == nullptr || data_size == 0) {
    if (data != nullptr) {
      allocator.Free(data);
    }
    return Ort::Status(
        ("OrtReadNamedBufferFunc returned invalid buffer for " + file_name).c_str(), ORT_FAIL);
  }
  out.assign(static_cast<char*>(data), static_cast<char*>(data) + data_size);
  allocator.Free(data);
  return Ort::Status();
}

Ort::Status EpContextIoDispatch::Write(const std::string& name,
                                       const void* buffer,
                                       size_t buffer_size,
                                       const Ort::Logger& logger) const {
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

EpContextIoDispatch::EpContextIoDispatch(const OrtSessionOptions* /*session_options*/,
                                         const Ort::Logger* /*logger*/) noexcept {}

bool EpContextIoDispatch::HasReadCallback() const noexcept { return false; }
bool EpContextIoDispatch::HasWriteCallback() const noexcept { return false; }

Ort::Status EpContextIoDispatch::Read(const std::string& /*file_name*/,
                                      std::vector<char>& /*out*/) const {
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
