// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "onnxruntime_experimental_c_api.h"

#if defined(QNN_EP_DECRYPT_SOURCE_POC_IMPLEMENTATION) && defined(_WIN32)
#define QNN_EP_DECRYPT_SOURCE_POC_EXPORT __declspec(dllexport)
#else
#define QNN_EP_DECRYPT_SOURCE_POC_EXPORT
#endif

// Private Windows PoC ABI. Not an ORT public API or a production QNN EP option.
// Register the provider before calling these helpers. Retain the provider module,
// binding and application state until all referencing options and sessions are released.
struct QnnEpDecryptSourceBinding;

using QnnEpDecryptFromSourceFunc = OrtStatus*(ORT_API_CALL*)(void* app_state, const char* name, const void* ciphertext, size_t ciphertext_bytes,
                                                             OrtAllocator* allocator, void** plaintext, size_t* plaintext_bytes);

inline constexpr const char* kQnnEpDecryptSourceMappingOption =
    "ep.qnn.context_binary_decrypt_source_mapping";

// The returned bridge delegates to legacy_read with the original app_state when
// the source policy is OFF. ON requires this exact bridge and maps the validated
// external path before invoking decrypt_from_source. Ciphertext and plaintext
// sizes may differ. The decrypt function must finish all source reads before
// returning, allocate output with allocator, and clean up its own failed output.
extern "C" QNN_EP_DECRYPT_SOURCE_POC_EXPORT OrtStatus* ORT_API_CALL QnnEpCreateDecryptSourceBinding(
    OrtReadNamedBufferFunc legacy_read, QnnEpDecryptFromSourceFunc decrypt_from_source,
    void* app_state, QnnEpDecryptSourceBinding** binding,
    OrtReadNamedBufferFunc* read_bridge) noexcept;

extern "C" QNN_EP_DECRYPT_SOURCE_POC_EXPORT void ORT_API_CALL QnnEpReleaseDecryptSourceBinding(
    QnnEpDecryptSourceBinding* binding) noexcept;

#undef QNN_EP_DECRYPT_SOURCE_POC_EXPORT
