// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Windows internal-symbol unit-test bridge.
//
// The ORT C++ wrapper stores its OrtApi pointer in a header-only function-local
// static. On Windows, onnxruntime_provider_test.exe and
// onnxruntime_providers_qnn.dll each get their own copy. Fake OrtNode/FakeGraph
// tests therefore need a small entry point compiled into the provider DLL so the
// DLL's copy can be pointed at the fake OrtApi table for the duration of a test.
// OrtLoggingManager and the SharedContext singleton have the same per-module
// behavior, so tests must also update and restore the provider DLL's copies.

#include "test/providers/qnn/infra/qnn_module_state_test_utils.h"

extern "C" const OrtApi* ORT_API_CALL QnnUnit_SetOrtApiForTesting(const OrtApi* api) noexcept {
  const OrtApi* previous = Ort::detail::Global::Api();
  Ort::detail::Global::Api(api);
  return previous;
}

extern "C" void ORT_API_CALL QnnUnit_RestoreOrtApiForTesting(const OrtApi* previous) noexcept {
  Ort::detail::Global::Api(previous);
}

extern "C" const OrtLogger* ORT_API_CALL QnnUnit_SetDefaultLoggerForTesting(
    const OrtLogger* logger) noexcept {
  const OrtLogger* previous = onnxruntime::OrtLoggingManager::GetDefaultLoggerPtr();
  onnxruntime::OrtLoggingManager::SetDefaultLogger(logger);
  return previous;
}

extern "C" bool ORT_API_CALL QnnUnit_SetSharedBackendManagerAllocatorForTesting(
    uint32_t allocator_type) noexcept {
  auto manager = onnxruntime::SharedContext::GetInstance().GetSharedQnnBackendManager();
  if (manager == nullptr) {
    return false;
  }

  manager->SetQnnAllocatorType(static_cast<onnxruntime::qnn::QnnAllocatorType>(allocator_type));
  return true;
}

extern "C" void ORT_API_CALL QnnUnit_ResetSharedBackendManagerForTesting() noexcept {
  onnxruntime::SharedContext::GetInstance().ResetSharedQnnBackendManager();
}
