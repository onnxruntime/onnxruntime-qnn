// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace onnxruntime {

// Runtime capabilities negotiated between this EP and the loaded ORT runtime.
// Keep raw API-version comparisons here so feature users ask a named capability
// question rather than duplicating version guards.
class RuntimeCapabilities {
 public:
  static constexpr uint32_t kEpContextEncryptionMinApiVersion = 28;

  constexpr explicit RuntimeCapabilities(uint32_t effective_api_version) noexcept
      : effective_api_version_{effective_api_version} {}

  static constexpr RuntimeCapabilities Unavailable() noexcept {
    return RuntimeCapabilities{0};
  }

  constexpr bool SupportsEpContextEncryption() const noexcept {
    return effective_api_version_ >= kEpContextEncryptionMinApiVersion;
  }

 private:
  uint32_t effective_api_version_;
};

}  // namespace onnxruntime
