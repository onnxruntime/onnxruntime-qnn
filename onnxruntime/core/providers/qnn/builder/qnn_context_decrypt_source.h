// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "onnxruntime_experimental_c_api.h"

#if defined(QNN_EP_DECRYPT_SOURCE_IMPLEMENTATION) && defined(_WIN32)
#define QNN_EP_DECRYPT_SOURCE_EXPORT __declspec(dllexport)
#else
#define QNN_EP_DECRYPT_SOURCE_EXPORT
#endif

// Private Windows binding. Not an ORT public API; requires explicit application opt-in.
// Register the provider before calling these helpers. Retain the provider module,
// binding and application state until all referencing options and sessions are released.
struct QnnEpDecryptSourceBinding;

using QnnEpDecryptFromSourceFunc =
    OrtStatus*(ORT_API_CALL*)(void* app_state, const char* name,
                              const void* ciphertext, size_t ciphertext_bytes,
                              OrtAllocator* allocator, void** plaintext,
                              size_t* plaintext_bytes);

// Register the returned bridge and binding with the existing ORT named-buffer read callback.
// The source is the encrypted external context file mapped read-only by the provider.
// A source callback borrows those bytes; a legacy read callback receives only the name.
// Both callbacks return independent plaintext allocated with the ORT allocator,
// which EpContextBuffer owns after a successful dispatch Read().
// The returned bridge delegates to legacy_read with the original app_state when
// the source policy is OFF. ON requires this exact bridge and maps the validated
// external path before invoking decrypt_from_source. Ciphertext and plaintext
// sizes may differ. The decrypt function must finish all source reads before
// returning, including its workers. Source is borrowed read-only memory. Successful
// output must be an independent allocation from allocator, never source or a subrange.
// The application cleans up its own failed output.
extern "C" QNN_EP_DECRYPT_SOURCE_EXPORT OrtStatus* ORT_API_CALL QnnEpCreateDecryptSourceBinding(
    OrtReadNamedBufferFunc legacy_read, QnnEpDecryptFromSourceFunc decrypt_from_source,
    void* app_state, QnnEpDecryptSourceBinding** binding,
    OrtReadNamedBufferFunc* read_bridge) noexcept;

// Release the binding after all referencing options and sessions have been released.
extern "C" QNN_EP_DECRYPT_SOURCE_EXPORT void ORT_API_CALL QnnEpReleaseDecryptSourceBinding(
    QnnEpDecryptSourceBinding* binding) noexcept;

#undef QNN_EP_DECRYPT_SOURCE_EXPORT
