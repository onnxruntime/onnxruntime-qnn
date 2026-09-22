// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "core/providers/qnn/common/inlined_containers.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/rpcmem_library.h"

#ifdef _WIN32
#include <d3d12.h>
#include <wrl/client.h>
#endif

namespace onnxruntime::qnn {

class HtpSharedMemoryAllocator : public OrtAllocator {
 public:
  using RpcMemLibraryProvider = std::function<std::shared_ptr<RpcMemLibrary>(std::string& error_message)>;

  HtpSharedMemoryAllocator(const OrtMemoryInfo* mem_info,
                           std::shared_ptr<RpcMemLibrary> rpcmem_lib)
      : memory_info_(mem_info),
        rpcmem_lib_{std::move(rpcmem_lib)},
        logger_(OrtLoggingManager::GetDefaultLogger()) {
    if (rpcmem_lib_ == nullptr) {
      ORT_CXX_API_THROW("rpcmem_lib should not be nullptr.", ORT_EP_FAIL);
    }

    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;
  }

  // Creates an allocator without loading RPCMEM. The provider is invoked on
  // the first allocation and the resulting library handle is cached. This is
  // required for environment-level allocators because ORT creates advertised
  // allocators while registering the EP library.
  HtpSharedMemoryAllocator(const OrtMemoryInfo* mem_info,
                           RpcMemLibraryProvider rpcmem_library_provider)
      : memory_info_(mem_info),
        rpcmem_library_provider_{std::move(rpcmem_library_provider)},
        logger_(OrtLoggingManager::GetDefaultLogger()) {
    if (rpcmem_library_provider_ == nullptr) {
      ORT_CXX_API_THROW("rpcmem_library_provider should not be empty.", ORT_EP_FAIL);
    }

    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(HtpSharedMemoryAllocator);

  // OrtAllocator implementations.
  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size);

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p);

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const HtpSharedMemoryAllocator& impl = *static_cast<const HtpSharedMemoryAllocator*>(this_);
    return impl.memory_info_;
  }

  struct SharedMemoryInfo {
    int fd;
    uint64_t offset;
    uint64_t total_size;
  };

  // Gets an allocation's shared memory info.
  // `address_within_allocation` identifies the allocation. It must be an address within an allocation returned by
  // Alloc() which has not yet been freed.
  static Ort::Status GetAllocationSharedMemoryInfo(void* address_within_allocation,
                                                   SharedMemoryInfo& allocation_info);

  // Allocation clean up callback signature.
  // For a given allocation, any added clean up callbacks will be called with the allocation's base address when the
  // allocation is freed.
  using AllocationCleanUpFn = std::function<void(void* allocation_base_address)>;

  // Adds allocation clean up callback to call when the allocation is freed.
  // `address_within_allocation` identifies the allocation. It must be an address within an allocation returned by
  // Alloc() which has not yet been freed.
  // `allocation_clean_up` is the clean up callback. The associated allocator takes ownership of the callback.
  static Ort::Status AddAllocationCleanUp(void* address_within_allocation, AllocationCleanUpFn&& allocation_clean_up);

 private:
  std::shared_ptr<RpcMemLibrary> GetOrCreateRpcMemLibrary();

  Ort::Status GetAllocationSharedMemoryInfoForThisAllocator(void* allocation_base_address,
                                                            SharedMemoryInfo& allocation_info);

  Ort::Status AddAllocationCleanUpForThisAllocator(void* allocation_base_address,
                                                   AllocationCleanUpFn&& allocation_clean_up);

  struct AllocationRecord {
    SharedMemoryInfo shared_memory_info;
    InlinedVector<AllocationCleanUpFn, 1> clean_up_fns;
    std::shared_ptr<RpcMemLibrary> rpcmem_library;
  };

  // allocation address -> corresponding allocation record
  InlinedHashMap<const void*, AllocationRecord> allocations_;
  std::mutex allocations_mutex_;  // synchronize access to allocations_

  const OrtMemoryInfo* memory_info_;
  std::shared_ptr<RpcMemLibrary> rpcmem_lib_;
  RpcMemLibraryProvider rpcmem_library_provider_;
  std::mutex rpcmem_library_mutex_;
  const Ort::Logger& logger_;
};

#ifdef _WIN32
// Allocator that allocates DX12 GPU buffers (D3D12_HEAP_TYPE_UPLOAD) for use with the QNN GPU backend.
// Buffers are CPU-mappable (host accessible) and GPU-readable, enabling zero-copy tensor I/O.
class Dx12SharedMemoryAllocator : public OrtAllocator {
 public:
  Dx12SharedMemoryAllocator(const OrtMemoryInfo* mem_info, OrtStatus*& status)
      : memory_info_(mem_info),
        dx12_device_{nullptr},
        logger_(OrtLoggingManager::GetDefaultLogger()) {
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dx12_device_));
    if (FAILED(hr) || dx12_device_ == nullptr) {
      status = MAKE_EP_FAIL("D3D12CreateDevice failed. DX12 allocator will not be available.");
    }

    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Dx12SharedMemoryAllocator);

  // OrtAllocator implementations.
  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size);

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p);

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const Dx12SharedMemoryAllocator& impl = *static_cast<const Dx12SharedMemoryAllocator*>(this_);
    return impl.memory_info_;
  }

  struct Dx12AllocationInfo {
    ID3D12Resource* resource;  // the D3D12 resource (not AddRef'd; lifetime tied to allocator record)
    uint64_t offset;
    uint64_t total_size;
  };

  // Gets an allocation's DX12 resource info.
  // `address_within_allocation` must be an address within an allocation returned by Alloc() that has not been freed.
  static Ort::Status GetAllocationDx12Info(void* address_within_allocation,
                                           Dx12AllocationInfo& allocation_info);

  // Allocation clean up callback signature.
  using AllocationCleanUpFn = std::function<void(void* allocation_base_address)>;

  // Adds a clean up callback to call when the allocation is freed.
  static Ort::Status AddAllocationCleanUp(void* address_within_allocation, AllocationCleanUpFn&& allocation_clean_up);

 private:
  Ort::Status GetAllocationDx12InfoForThisAllocator(void* allocation_base_address,
                                                    Dx12AllocationInfo& allocation_info);

  Ort::Status AddAllocationCleanUpForThisAllocator(void* allocation_base_address,
                                                   AllocationCleanUpFn&& allocation_clean_up);

  struct AllocationRecord {
    Dx12AllocationInfo dx12_info;
    InlinedVector<AllocationCleanUpFn, 1> clean_up_fns;
  };

  // allocation (mapped CPU) address -> corresponding allocation record
  InlinedHashMap<const void*, AllocationRecord> allocations_;
  std::mutex allocations_mutex_;

  const OrtMemoryInfo* memory_info_;
  Microsoft::WRL::ComPtr<ID3D12Device> dx12_device_;
  const Ort::Logger& logger_;
};
#endif  // _WIN32

}  // namespace onnxruntime::qnn
