// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Owns the App-provided EPContext read/write callbacks resolved from a session options object.
// Constructed once per session by QnnEp; not thread-safe (callers must serialize concurrent
// Read/Write calls).
//
// On pre-v28 ORT the class is a no-op stub; HasReadCallback()/HasWriteCallback() always
// return false. Callers can use this class unconditionally without macro guards.
//
// Scope: only the routed EPContext binary ("_qnn.bin"/EP_CACHE_CONTEXT, and multi-SoC's
// context-binary-list buffers) is covered. The QNN IR-serializer ".dlc" (IR serializer backend,
// Genie's DlcConfig_create) is read/written directly against disk and is not encrypted here.
class EpContextIoDispatch {
 public:
  // Resolves the callbacks from `session_options`. Never throws; on pre-v28 ORT or no
  // registered callback, HasReadCallback()/HasWriteCallback() return false. `logger`
  // (optional) receives a WARNING if resolution genuinely fails (e.g. runtime/build API
  // version skew) rather than the app simply not registering anything.
  explicit EpContextIoDispatch(const OrtSessionOptions* session_options,
                               const Ort::Logger* logger = nullptr) noexcept;
  ~EpContextIoDispatch() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(EpContextIoDispatch);

  bool HasReadCallback() const noexcept;
  bool HasWriteCallback() const noexcept;

  // Precondition: HasReadCallback(). Fills `out` with the decrypted bytes.
  Ort::Status Read(const std::string& file_name, std::vector<char>& out) const;

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
};

}  // namespace qnn
}  // namespace onnxruntime
