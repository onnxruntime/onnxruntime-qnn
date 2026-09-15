// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <shared_mutex>

#include "core/providers/qnn/ort_api.h"

#include "core/providers/qnn/common/inlined_containers.h"

#ifdef _WIN32

#include <d3d12.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

namespace onnxruntime {

using ImportResourceCleanUpFn_t = std::function<void(void* handle)>;

/**
 * @brief Derived handle for imported external memory from D3D12 to QNN.
 *
 * Derives from OrtExternalMemoryHandle (base struct) and adds QNN-specific fields.
 */
struct QnnExternalMemoryHandle : OrtExternalMemoryHandle {
  ComPtr<ID3D12Resource> d3d12_resource_;

  InlinedVector<ImportResourceCleanUpFn_t, 1> cleanup_callbacks_;

  QnnExternalMemoryHandle();

  static void ORT_API_CALL ReleaseCallback(_In_ OrtExternalMemoryHandle* handle) noexcept;
};

/**
 * @brief Derived handle for imported external semaphore from D3D12 fence to QNN.
 *
 * Derives from OrtExternalSemaphoreHandle (base struct) and adds QNN-specific fields.
 */
struct QnnExternalSemaphoreHandle : OrtExternalSemaphoreHandle {
  ComPtr<ID3D12Fence> d3d12_fence_;

  QnnExternalSemaphoreHandle();

  static void ORT_API_CALL ReleaseCallback(_In_ OrtExternalSemaphoreHandle* handle) noexcept;
};

/**
 * @brief Implementation of OrtExternalResourceImporterImpl for QNN EP.
 *
 * This struct implements the external resource importer interface using QNN Runtime APIs
 * to import D3D12 shared resources and timeline fences for zero-copy import.
 *
 * Currently Supported handle types:
 * - ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
 * Currently UnSupported handle types:
 * - ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
 * - ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE
 */
struct QnnExternalResourceImporterImpl : OrtExternalResourceImporterImpl {
  using ImportResourceCleanUpFn = ImportResourceCleanUpFn_t;

  QnnExternalResourceImporterImpl(int device_id, const OrtApi& ort_api_in);

  // Memory operations
  static bool ORT_API_CALL CanImportMemoryImpl(
      _In_ const OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalMemoryHandleType handle_type) noexcept;

  static OrtStatus* ORT_API_CALL ImportMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalMemoryDescriptor* desc,
      _Outptr_ OrtExternalMemoryHandle** out_handle) noexcept;

  static void ORT_API_CALL ReleaseMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalMemoryHandle* handle) noexcept;

  static OrtStatus* ORT_API_CALL CreateTensorFromMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalMemoryHandle* mem_handle,
      _In_ const OrtExternalTensorDescriptor* tensor_desc,
      _Outptr_ OrtValue** out_tensor) noexcept;

  // Semaphore operations
  static bool ORT_API_CALL CanImportSemaphoreImpl(
      _In_ const OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreType type) noexcept;

  static OrtStatus* ORT_API_CALL ImportSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalSemaphoreDescriptor* desc,
      _Outptr_ OrtExternalSemaphoreHandle** out_handle) noexcept;

  static void ORT_API_CALL ReleaseSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandle* handle) noexcept;

  static OrtStatus* ORT_API_CALL WaitSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandle* handle,
      _In_ OrtSyncStream* stream,
      _In_ uint64_t value) noexcept;

  static OrtStatus* ORT_API_CALL SignalSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandle* handle,
      _In_ OrtSyncStream* stream,
      _In_ uint64_t value) noexcept;

  static void ORT_API_CALL ReleaseImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr) noexcept;

  static bool FindImportMemory(void* handle) {
    std::shared_lock read_lock{mutex_mem_handle_registry_};
    return mem_handle_registry_.contains((QnnExternalMemoryHandle*)handle);
  }

  static Ort::Status AddImportMemoryCleanUp(void* handle, ImportResourceCleanUpFn&& cleanup) {
    if (FindImportMemory(handle)) {
      ((QnnExternalMemoryHandle*)handle)->cleanup_callbacks_.emplace_back(std::move(cleanup));
    } else {
      return MAKE_EP_FAIL("Import memory handle not found");
    }
    return Ort::Status();
  }

  int device_id_;
  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  ComPtr<ID3D12Device> d3d12_device_;

  static InlinedHashSet<QnnExternalMemoryHandle*> mem_handle_registry_;
  static std::shared_mutex mutex_mem_handle_registry_;
};

/**
 * @brief SyncStream implementation for QNN EP.
 *
 * This enables WaitSemaphore and SignalSemaphore operations on a QNN stream.
 */
struct QnnSyncStreamImpl : OrtSyncStreamImpl {
  QnnSyncStreamImpl(int device_id, const OrtApi& ort_api_in);
  ~QnnSyncStreamImpl();

  static void* ORT_API_CALL GetHandleImpl(
      _In_ OrtSyncStreamImpl* this_ptr) noexcept;

  static void ORT_API_CALL ReleaseImpl(
      _In_ OrtSyncStreamImpl* this_ptr) noexcept;

  int device_id_;
  const OrtApi& ort_api_;
};

}  // namespace onnxruntime

#endif  // _WIN32
