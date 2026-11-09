// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

#include "core/providers/qnn/ort_api.h"

#include "core/providers/qnn/common/inlined_containers.h"
#include "core/providers/qnn/rpcmem_library.h"

#ifdef _WIN32

#include <d3d12.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

namespace onnxruntime {

using ImportResourceCleanUpFn_t = std::function<void(void* handle)>;

// Page granularity required for host allocations mapped into the NPU address space by FastRPC.
inline constexpr size_t kHostAllocationImportAlignment = 4096;

enum class QnnExternalMemoryKind {
  kD3D12Resource,   // ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
  kHostAllocation,  // ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION
};

/**
 * @brief Derived handle for external memory imported into QNN.
 *
 * Derives from OrtExternalMemoryHandle (base struct) and adds QNN-specific fields.
 * - D3D12 resources are identified by the handle pointer itself.
 * - Host allocations are app-owned CPU buffers registered with FastRPC once at import time. QNN mem handles
 *   created for addresses inside the buffer are retained across runs and deregistered when the handle is released.
 */
struct QnnExternalMemoryHandle : OrtExternalMemoryHandle {
  QnnExternalMemoryKind kind_ = QnnExternalMemoryKind::kD3D12Resource;

  ComPtr<ID3D12Resource> d3d12_resource_;

  // Host allocation fields.
  void* host_base_ = nullptr;
  size_t host_size_ = 0;
  int host_fd_ = -1;
  // Keeps the RPCMEM library loaded until the FastRPC registration is undone.
  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library_;

  struct CleanUpRecord {
    ImportResourceCleanUpFn_t callback;
    // Hooks tied to an owner (e.g. a QNN context) are dropped once the owner has expired.
    std::optional<std::weak_ptr<void>> owner;
  };

  std::mutex cleanup_mutex_;  // synchronize access to cleanup_callbacks_
  InlinedVector<CleanUpRecord, 1> cleanup_callbacks_;

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
 * - ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION (NPU/HTP devices only)
 * Currently UnSupported handle types:
 * - ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
 * - ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE
 */
struct QnnExternalResourceImporterImpl : OrtExternalResourceImporterImpl {
  using ImportResourceCleanUpFn = ImportResourceCleanUpFn_t;

  // `supports_host_allocation` enables HOST_ALLOCATION import (NPU devices).
  // `rpcmem_library` overrides the process-wide RPCMEM library (used by tests); when null it is loaded on first use.
  QnnExternalResourceImporterImpl(int device_id, const OrtApi& ort_api_in,
                                  bool supports_host_allocation = false,
                                  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library = nullptr);

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

  // Returns true if `handle` is an imported D3D12 memory handle.
  static bool FindImportMemory(void* handle) {
    std::shared_lock read_lock{mutex_mem_handle_registry_};
    return mem_handle_registry_.contains((QnnExternalMemoryHandle*)handle);
  }

  struct ImportedHostMemoryInfo {
    void* base_address;
    size_t size;
    int fd;
  };

  // Looks up the imported host allocation containing `address`.
  static std::optional<ImportedHostMemoryInfo> FindImportedHostMemory(const void* address);

  // Adds a clean up callback that runs when the import handle is released.
  // `handle_or_address` is a D3D12 import handle or an address within an imported host allocation.
  static Ort::Status AddImportMemoryCleanUp(void* handle_or_address, ImportResourceCleanUpFn&& cleanup,
                                            std::optional<std::weak_ptr<void>> owner = std::nullopt);

  int device_id_;
  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  std::mutex d3d12_device_mutex_;
  ComPtr<ID3D12Device> d3d12_device_;
  const bool supports_host_allocation_;
  std::mutex rpcmem_library_mutex_;
  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library_;

  // Imported D3D12 memory handles.
  static InlinedHashSet<QnnExternalMemoryHandle*> mem_handle_registry_;
  // Imported host allocations: base address -> handle. Ranges never overlap.
  static std::map<uintptr_t, QnnExternalMemoryHandle*> host_memory_registry_;
  // Guards both registries.
  static std::shared_mutex mutex_mem_handle_registry_;

 private:
  OrtStatus* ImportHostAllocation(const OrtExternalMemoryDescriptor& desc, QnnExternalMemoryHandle& handle);
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
