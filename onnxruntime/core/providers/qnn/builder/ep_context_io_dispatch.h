// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <memory>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/ort_api_runtime_capabilities.h"

namespace onnxruntime {
namespace qnn {

// Owns default-allocator output without copying; the allocator is owned by ORT.
class EpContextBuffer {
  using OwnedBufferPtr = std::unique_ptr<void, Ort::detail::AllocatedFree>;

 public:
  char* data() const noexcept { return static_cast<char*>(data_.get()); }

  size_t size() const noexcept { return data_ ? size_ : 0; }

  void Adopt(void* data, size_t size, OrtAllocator* allocator) noexcept {
    data_ = OwnedBufferPtr(data, Ort::detail::AllocatedFree(allocator));
    size_ = size;
  }

 private:
  OwnedBufferPtr data_{nullptr, Ort::detail::AllocatedFree(nullptr)};
  size_t size_ = 0;
};

// Owns the App-provided EPContext callbacks and the session's file-mapping setting.
// Constructed once per session by QnnEp; not thread-safe (callers must serialize concurrent
// Read/Write calls).
//
// On pre-v28 ORT, or when the runtime is older than v28, this is a no-op stub;
// HasReadCallback()/HasWriteCallback() always return false. Callers can use this class
// unconditionally without macro guards.
//
// Scope: only the routed EPContext binary ("_qnn.bin"/EP_CACHE_CONTEXT, and multi-SoC's
// context-binary-list buffers) is covered. The QNN IR-serializer ".dlc" (IR serializer backend,
// Genie's DlcConfig_create) is read/written directly against disk and is not encrypted here.
class EpContextIoDispatch {
 public:
  // Creates a dispatch without application callbacks.
  // Session-bound callers must use the constructor below and supply the runtime-negotiated
  // capabilities explicitly.
  EpContextIoDispatch() noexcept;

  // Resolves the callbacks from `session_options`. Never throws.
  //
  // `runtime_capabilities` is negotiated with the loaded runtime by QnnEpFactory. The EPContext
  // encryption APIs require ORT API v28; calling them against an older runtime reads past the
  // end of that runtime's OrtApi struct instead of cleanly throwing. This constructor asks the
  // shared named capability before touching Ort::Experimental::EpContextConfig, so feature code
  // does not duplicate a raw API-version guard. There is intentionally no default value: a
  // session-bound caller must supply the factory's negotiated capabilities.
  //
  // `logger` (optional) receives a WARNING if resolution genuinely fails rather than the app
  // simply not registering anything.
  explicit EpContextIoDispatch(const OrtSessionOptions& session_options,
                               const OrtApiRuntimeCapabilities& runtime_capabilities,
                               const Ort::Logger* logger = nullptr,
                               bool enable_file_mapped_weights = true) noexcept;
  ~EpContextIoDispatch() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(EpContextIoDispatch);

  bool HasReadCallback() const noexcept;
  bool HasWriteCallback() const noexcept;
  bool IsFileMappingEnabled() const noexcept { return file_mapping_enabled_; }

  // Precondition: HasReadCallback(). Adopts valid successful output without copying.
  // On callback error, the callback cleans up and the previous output is preserved.
  Ort::Status Read(const std::string& file_name, EpContextBuffer& out) const;

  // Precondition: HasWriteCallback(). Logs INFO with byte count on success.
  Ort::Status Write(const std::string& name,
                    const void* buffer,
                    size_t buffer_size,
                    const Ort::Logger& logger) const;

 private:
#if ORT_API_HAS_EPCONTEXT_ENCRYPTION
  Ort::Experimental::EpContextConfig config_;
  OrtReadNamedBufferFunc read_fn_ = nullptr;
  void* read_state_ = nullptr;
  OrtWriteNamedBufferFunc write_fn_ = nullptr;
  void* write_state_ = nullptr;
#endif
  bool file_mapping_enabled_ = true;
};

}  // namespace qnn
}  // namespace onnxruntime
