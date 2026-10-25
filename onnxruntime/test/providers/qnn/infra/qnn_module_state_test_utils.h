// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Test-only helpers that keep module-local QNN EP state synchronized between
// onnxruntime_provider_test and the provider shared library on Windows.
// This header is not self-gated on QNN_EP_INTERNAL_SYMBOL_ACCESS because the
// provider DLL bridge is compiled only for that build mode but does not inherit
// the test executable's compile definition.

#pragma once

#include <cstdint>

#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/infra/qnn_test_windows_compat.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/shared_context.h"

#ifdef _WIN32
extern "C" const OrtApi* ORT_API_CALL QnnUnit_SetOrtApiForTesting(const OrtApi* api) noexcept;
extern "C" void ORT_API_CALL QnnUnit_RestoreOrtApiForTesting(const OrtApi* previous) noexcept;
extern "C" const OrtLogger* ORT_API_CALL QnnUnit_SetDefaultLoggerForTesting(
    const OrtLogger* logger) noexcept;
extern "C" bool ORT_API_CALL QnnUnit_SetSharedBackendManagerAllocatorForTesting(
    uint32_t allocator_type) noexcept;
extern "C" void ORT_API_CALL QnnUnit_ResetSharedBackendManagerForTesting() noexcept;
#endif

namespace onnxruntime {
namespace test {

// Updates only the provider module's OrtApi state. This is a no-op when the
// executable and provider share the same definition.
class ScopedProviderModuleOrtApiOverride {
 public:
  explicit ScopedProviderModuleOrtApiOverride(const OrtApi* api) {
#ifdef _WIN32
    previous_ = QnnUnit_SetOrtApiForTesting(api);
#else
    (void)api;
#endif
  }

  ~ScopedProviderModuleOrtApiOverride() {
#ifdef _WIN32
    QnnUnit_RestoreOrtApiForTesting(previous_);
#endif
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedProviderModuleOrtApiOverride);

 private:
#ifdef _WIN32
  const OrtApi* previous_ = nullptr;
#endif
};

// Updates and restores both copies of the global OrtApi state.
class OrtGlobalApiOverride {
 public:
  explicit OrtGlobalApiOverride(const OrtApi* api)
      : previous_(OrtGetApiBase()->GetApi(ORT_API_VERSION)),
        provider_override_(api) {
    Ort::detail::Global::Api(api);
  }

  ~OrtGlobalApiOverride() {
    Ort::detail::Global::Api(previous_);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtGlobalApiOverride);

 private:
  const OrtApi* previous_ = nullptr;
  ScopedProviderModuleOrtApiOverride provider_override_;
};

// Updates and restores both copies of the default logger state.
class ScopedDefaultLoggerOverride {
 public:
  explicit ScopedDefaultLoggerOverride(const OrtLogger* logger)
      : previous_(OrtLoggingManager::GetDefaultLoggerPtr()) {
    OrtLoggingManager::SetDefaultLogger(logger);
#ifdef _WIN32
    provider_previous_ = QnnUnit_SetDefaultLoggerForTesting(logger);
#endif
  }

  ~ScopedDefaultLoggerOverride() {
#ifdef _WIN32
    QnnUnit_SetDefaultLoggerForTesting(provider_previous_);
#endif
    OrtLoggingManager::SetDefaultLogger(previous_);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedDefaultLoggerOverride);

 private:
  const OrtLogger* previous_ = nullptr;
#ifdef _WIN32
  const OrtLogger* provider_previous_ = nullptr;
#endif
};

// These helpers select the provider module's SharedContext copy on Windows and
// the process-wide definition on platforms with ELF symbol interposition.
inline void ResetSharedBackendManagerForTest() {
#ifdef _WIN32
  QnnUnit_ResetSharedBackendManagerForTesting();
#else
  SharedContext::GetInstance().ResetSharedQnnBackendManager();
#endif
}

inline bool SetSharedBackendManagerAllocatorForTest(qnn::QnnAllocatorType allocator_type) {
#ifdef _WIN32
  return QnnUnit_SetSharedBackendManagerAllocatorForTesting(static_cast<uint32_t>(allocator_type));
#else
  auto manager = SharedContext::GetInstance().GetSharedQnnBackendManager();
  if (manager == nullptr) {
    return false;
  }

  manager->SetQnnAllocatorType(allocator_type);
  return true;
#endif
}

}  // namespace test
}  // namespace onnxruntime
