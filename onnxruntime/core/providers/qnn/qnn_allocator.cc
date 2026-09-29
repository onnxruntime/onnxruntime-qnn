// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/qnn_allocator.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <gsl/gsl>
#include <optional>
#include <shared_mutex>

#include "SafeInt.hpp"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/rpcmem_library.h"

namespace onnxruntime::qnn {

namespace {

size_t AllocationAlignment() {
  constexpr size_t min_allocation_alignment = 64;  // Equal to MlasGetPreferredBufferAlignment()
  return min_allocation_alignment;
}

bool IsAligned(const void* address, size_t alignment) {
  assert((alignment & (alignment - 1)) == 0);  // alignment must be a power of two
  return (reinterpret_cast<uintptr_t>(address) & (alignment - 1)) == 0;
}

std::unique_ptr<void, void (*)(void*)> WrapSharedMemoryWithUniquePtr(void* shared_memory_raw,
                                                                     const RpcMemApi& rpcmem_api) {
  return {shared_memory_raw, rpcmem_api.free};
}

// This class tracks information about allocations made by allocator (e.g. HtpSharedMemoryAllocator) instances.
// Given an address within a tracked allocation, we can look up information about it like the base address and the
// allocating allocator instance.
template <typename Allocator>
class AllocationTracker {
 public:
  struct Record {
    void* base_address;
    size_t size_in_bytes;
    gsl::not_null<Allocator*> allocator;
  };

  // Starts tracking an allocation.
  // Returns true if successful, or false if there is already a tracked allocation at `base_address`.
  bool RegisterAllocation(void* base_address, size_t size_in_bytes, Allocator& allocator);

  // Stops tracking an allocation.
  // Returns true if successful, or false if there is no tracked allocation at `base_address`.
  bool UnregisterAllocation(void* base_address);

  // Looks up a tracked allocation's record.
  // Returns the record associated with the tracked allocation containing `address_within_allocation`,
  // or `std::nullopt` if there is no such tracked allocation.
  std::optional<Record> LookUp(void* address_within_allocation);

 private:
  std::map<const void*, Record> records_;
  std::shared_mutex records_mutex_;
};

template <typename Allocator>
bool AllocationTracker<Allocator>::RegisterAllocation(void* base_address, size_t size_in_bytes,
                                                      Allocator& allocator) {
  Record record{base_address, size_in_bytes, &allocator};

  std::unique_lock write_lock{records_mutex_};
  const bool registered = records_.emplace(base_address, std::move(record)).second;
  return registered;
}

template <typename Allocator>
bool AllocationTracker<Allocator>::UnregisterAllocation(void* base_address) {
  std::unique_lock write_lock{records_mutex_};
  const bool unregistered = records_.erase(base_address) == 1;
  return unregistered;
}

template <typename Allocator>
std::optional<typename AllocationTracker<Allocator>::Record> AllocationTracker<Allocator>::LookUp(
    void* address_within_allocation) {
  std::shared_lock read_lock{records_mutex_};

  // Look for a record where `address_within_allocation` falls within the range:
  //   [`record.base_address`, `record.base_address` + `record.size_in_bytes`)

  // First, find the first record with a base address greater than `address_within_allocation`, or the end of the
  // container if no such record exists.
  const auto first_record_with_larger_base_address_it = records_.upper_bound(address_within_allocation);

  // The previous record should have the greatest base address that is not greater than `address_within_allocation`.
  // Make sure it exists.
  if (first_record_with_larger_base_address_it == records_.begin()) {
    return std::nullopt;
  }

  const auto record_it = std::prev(first_record_with_larger_base_address_it);

  const auto record = record_it->second;
  assert(address_within_allocation >= record.base_address);

  // Verify that `address_within_allocation` is within the upper end of the range.
  if (reinterpret_cast<std::byte*>(address_within_allocation) >=
      reinterpret_cast<std::byte*>(record.base_address) + record.size_in_bytes) {
    return std::nullopt;
  }

  return record;
}

AllocationTracker<HtpSharedMemoryAllocator>& GlobalHtpSharedMemoryAllocationTracker() {
  static AllocationTracker<HtpSharedMemoryAllocator> allocation_tracker{};
  return allocation_tracker;
}

#ifdef _WIN32
AllocationTracker<Dx12SharedMemoryAllocator>& GlobalDx12SharedMemoryAllocationTracker() {
  static AllocationTracker<Dx12SharedMemoryAllocator> allocation_tracker{};
  return allocation_tracker;
}
#endif

}  // namespace

void* ORT_API_CALL HtpSharedMemoryAllocator::AllocImpl(struct OrtAllocator* this_, size_t requested_size) {
  HtpSharedMemoryAllocator* allocator = static_cast<HtpSharedMemoryAllocator*>(this_);

  const size_t shared_memory_block_size_in_bytes = requested_size;

  // rpcmem_alloc() has an int size parameter. make sure we don't overflow.
  // TODO switch to rpcmem_alloc2() which has size_t size parameter.
  // need to verify that rpcmem_alloc2() is available in all environments we care about.
  const SafeInt<int> shared_memory_block_size_in_bytes_int = shared_memory_block_size_in_bytes;

  // allocate shared memory
  void* shared_memory_raw = allocator->rpcmem_lib_->Api().alloc(rpcmem::RPCMEM_HEAP_ID_SYSTEM,
                                                                rpcmem::RPCMEM_DEFAULT_FLAGS,
                                                                shared_memory_block_size_in_bytes_int);
  if (shared_memory_raw == nullptr) {
    ORT_CXX_API_THROW("rpcmem_alloc() failed to allocate and returned nullptr.", ORT_EP_FAIL);
  }
  auto shared_memory = WrapSharedMemoryWithUniquePtr(shared_memory_raw, allocator->rpcmem_lib_->Api());

  const size_t allocation_alignment = AllocationAlignment();
  if (!IsAligned(shared_memory_raw, allocation_alignment)) {
    ORT_CXX_API_THROW(
        ("Shared memory address does not have required alignment (" +
         std::to_string(allocation_alignment) + " bytes)."),
        ORT_EP_FAIL);
  }

  // get shared memory fd
  const auto shared_memory_fd = allocator->rpcmem_lib_->Api().to_fd(shared_memory.get());
  if (shared_memory_fd == -1) {
    ORT_CXX_API_THROW("rpcmem_to_fd() returned invalid file descriptor.", ORT_EP_FAIL);
  }

  std::byte* allocation_address = reinterpret_cast<std::byte*>(shared_memory_raw);

  // store allocation record
  {
    SharedMemoryInfo shared_memory_info{};
    shared_memory_info.fd = shared_memory_fd;
    shared_memory_info.offset = 0;
    shared_memory_info.total_size = shared_memory_block_size_in_bytes;

    AllocationRecord allocation_record{};
    allocation_record.shared_memory_info = std::move(shared_memory_info);

    std::scoped_lock g{allocator->allocations_mutex_};
    const bool inserted = allocator->allocations_.emplace(allocation_address, std::move(allocation_record)).second;
    if (!inserted) {
      ORT_CXX_API_THROW("Allocation record already exists for address.", ORT_EP_FAIL);
    }
  }

  // register with global allocation tracker
  {
    const bool registered = GlobalHtpSharedMemoryAllocationTracker().RegisterAllocation(allocation_address,
                                                                                        shared_memory_block_size_in_bytes,
                                                                                        *allocator);

    if (!registered) {
      ORT_CXX_API_THROW("Attempted to register allocation but it is already tracked for address.", ORT_EP_FAIL);
    }
  }

  shared_memory.release();
  return allocation_address;
}

void ORT_API_CALL HtpSharedMemoryAllocator::FreeImpl(struct OrtAllocator* this_, void* allocation_address) {
  HtpSharedMemoryAllocator* allocator = static_cast<HtpSharedMemoryAllocator*>(this_);

  if (allocation_address == nullptr) {
    return;
  }

  const auto allocation_node = [allocator, allocation_address]() {
    std::scoped_lock g{allocator->allocations_mutex_};
    return allocator->allocations_.extract(allocation_address);
  }();

  if (allocation_node.empty()) {
    ORT_CXX_API_THROW("Failed to get allocation info for address.", ORT_EP_FAIL);
  }

  // At this point, we have a valid allocation to free.
  // Avoid throwing exceptions as this may be running from a destructor.
  try {
    // take ownership of shared memory and free at end of scope
    auto shared_memory = WrapSharedMemoryWithUniquePtr(allocation_address, allocator->rpcmem_lib_->Api());

    // unregister with global allocation tracker
    {
      const bool unregistered = GlobalHtpSharedMemoryAllocationTracker().UnregisterAllocation(allocation_address);
      if (!unregistered) {
        std::ostringstream oss;
        oss << "Attempted to deregister allocation but it is untracked for address (" << allocation_address << ").";
        ORT_CXX_LOG(allocator->logger_, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
      }
    }

    // clean up allocation record
    const auto& allocation_record = allocation_node.mapped();
    for (auto& clean_up_fn : allocation_record.clean_up_fns) {
      // attempt to run each clean_up_fn even if exceptions are thrown
      try {
        clean_up_fn(allocation_address);
      } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Caught exception while running clean up callback for address (" << allocation_address << "): "
            << e.what();
        ORT_CXX_LOG(allocator->logger_, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
      }
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Caught exception while freeing address (" << allocation_address << "): " << e.what();
    ORT_CXX_LOG(allocator->logger_, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
  }
}

Ort::Status HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfo(void* address_within_allocation,
                                                                    SharedMemoryInfo& shared_memory_info_out) {
  const auto tracked_record = GlobalHtpSharedMemoryAllocationTracker().LookUp(address_within_allocation);
  RETURN_IF_NOT(tracked_record.has_value(), "Failed to look up tracked allocation.");

  void* const base_address = tracked_record->base_address;
  SharedMemoryInfo shared_memory_info{};
  RETURN_IF_ERROR(tracked_record->allocator->GetAllocationSharedMemoryInfoForThisAllocator(
      base_address, shared_memory_info));

  // adjust `shared_memory_info.offset` for `address_within_allocation`
  const auto offset_from_base = std::distance(reinterpret_cast<std::byte*>(base_address),
                                              reinterpret_cast<std::byte*>(address_within_allocation));

  shared_memory_info.offset += offset_from_base;

  shared_memory_info_out = std::move(shared_memory_info);
  return Ort::Status();
}

Ort::Status HtpSharedMemoryAllocator::AddAllocationCleanUp(void* address_within_allocation,
                                                           AllocationCleanUpFn&& allocation_clean_up) {
  const auto tracked_record = GlobalHtpSharedMemoryAllocationTracker().LookUp(address_within_allocation);
  RETURN_IF_NOT(tracked_record.has_value(), "Failed to look up tracked allocation.");

  void* const base_address = tracked_record->base_address;
  return tracked_record->allocator->AddAllocationCleanUpForThisAllocator(base_address,
                                                                         std::move(allocation_clean_up));
}

Ort::Status HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfoForThisAllocator(void* allocation_base_address,
                                                                                    SharedMemoryInfo& allocation_info) {
  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_base_address);
  RETURN_IF(allocation_it == allocations_.end(), "Failed to get allocation info for address.");

  allocation_info = allocation_it->second.shared_memory_info;
  return Ort::Status();
}

Ort::Status HtpSharedMemoryAllocator::AddAllocationCleanUpForThisAllocator(void* allocation_base_address,
                                                                           AllocationCleanUpFn&& allocation_clean_up) {
  RETURN_IF(allocation_clean_up == nullptr, "allocation_clean_up should not be empty.");

  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_base_address);
  RETURN_IF(allocation_it == allocations_.end(), "Failed to get allocation info for address.");

  auto& clean_up_fns = allocation_it->second.clean_up_fns;
  clean_up_fns.emplace_back(std::move(allocation_clean_up));
  return Ort::Status();
}

#ifdef _WIN32

void* ORT_API_CALL Dx12SharedMemoryAllocator::AllocImpl(struct OrtAllocator* this_, size_t requested_size) {
  Dx12SharedMemoryAllocator* allocator = static_cast<Dx12SharedMemoryAllocator*>(this_);

  if (requested_size == 0) {
    ORT_CXX_API_THROW("Dx12SharedMemoryAllocator: requested_size must be > 0.", ORT_EP_FAIL);
  }

  // Create a D3D12 buffer resource in an UPLOAD heap (CPU-writable, GPU-readable).
  D3D12_RESOURCE_DESC buffer_desc = {};
  buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  buffer_desc.Width = requested_size;
  buffer_desc.Height = 1;
  buffer_desc.DepthOrArraySize = 1;
  buffer_desc.MipLevels = 1;
  buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
  buffer_desc.SampleDesc.Count = 1;
  buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  buffer_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

  D3D12_HEAP_PROPERTIES heap_props = {};
  heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
  heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_props.CreationNodeMask = 1;
  heap_props.VisibleNodeMask = 1;

  ID3D12Resource* d3d12_resource = nullptr;
  HRESULT hr = allocator->dx12_device_->CreateCommittedResource(
      &heap_props,
      D3D12_HEAP_FLAG_NONE,
      &buffer_desc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(&d3d12_resource));
  if (FAILED(hr) || d3d12_resource == nullptr) {
    ORT_CXX_API_THROW("Dx12SharedMemoryAllocator: CreateCommittedResource failed.", ORT_EP_FAIL);
  }

  // Map the resource to get a CPU-visible pointer (persistent mapping for UPLOAD heap).
  void* mapped_ptr = nullptr;
  hr = d3d12_resource->Map(0, nullptr, &mapped_ptr);
  if (FAILED(hr) || mapped_ptr == nullptr) {
    d3d12_resource->Release();
    ORT_CXX_API_THROW("Dx12SharedMemoryAllocator: Map failed.", ORT_EP_FAIL);
  }

  // Store allocation record.
  {
    Dx12AllocationInfo dx12_info{};
    dx12_info.resource = d3d12_resource;
    dx12_info.offset = 0;
    dx12_info.total_size = requested_size;

    AllocationRecord allocation_record{};
    allocation_record.dx12_info = std::move(dx12_info);

    ORT_CXX_LOGF(allocator->logger_,
                 ORT_LOGGING_LEVEL_INFO,
                 "\nMaking DX12 allocation:"
                 "\n  resource   = %p"
                 "\n  offset     = %llu"
                 "\n  total_size = %llu"
                 "\n  mapped_ptr = %p\n",
                 allocation_record.dx12_info.resource,
                 allocation_record.dx12_info.offset,
                 allocation_record.dx12_info.total_size,
                 mapped_ptr);

    std::scoped_lock g{allocator->allocations_mutex_};
    const bool inserted = allocator->allocations_.emplace(mapped_ptr, std::move(allocation_record)).second;
    if (!inserted) {
      d3d12_resource->Unmap(0, nullptr);
      d3d12_resource->Release();
      ORT_CXX_API_THROW("Dx12SharedMemoryAllocator: Allocation record already exists for address.", ORT_EP_FAIL);
    }
  }

  // Register with global DX12 allocation tracker.
  {
    const bool registered = GlobalDx12SharedMemoryAllocationTracker().RegisterAllocation(mapped_ptr,
                                                                                         requested_size,
                                                                                         *allocator);
    if (!registered) {
      d3d12_resource->Unmap(0, nullptr);
      d3d12_resource->Release();
      ORT_CXX_API_THROW("Dx12SharedMemoryAllocator: Attempted to register allocation but it is already tracked.", ORT_EP_FAIL);
    }
  }

  return mapped_ptr;
}

void ORT_API_CALL Dx12SharedMemoryAllocator::FreeImpl(struct OrtAllocator* this_, void* allocation_address) {
  Dx12SharedMemoryAllocator* allocator = static_cast<Dx12SharedMemoryAllocator*>(this_);

  if (allocation_address == nullptr) {
    return;
  }

  const auto allocation_node = [allocator, allocation_address]() {
    std::scoped_lock g{allocator->allocations_mutex_};
    return allocator->allocations_.extract(allocation_address);
  }();

  if (allocation_node.empty()) {
    ORT_CXX_API_THROW("Dx12SharedMemoryAllocator: Failed to get allocation info for address.", ORT_EP_FAIL);
  }

  try {
    const auto& allocation_record = allocation_node.mapped();
    ID3D12Resource* resource = allocation_record.dx12_info.resource;

    ORT_CXX_LOGF(allocator->logger_,
                 ORT_LOGGING_LEVEL_INFO,
                 "\nFreeing DX12 allocation:"
                 "\n  resource           = %p"
                 "\n  offset             = %llu"
                 "\n  total_size         = %llu"
                 "\n  allocation_address = %p\n",
                 allocation_record.dx12_info.resource,
                 allocation_record.dx12_info.offset,
                 allocation_record.dx12_info.total_size,
                 allocation_address);

    // Unregister from global tracker.
    {
      const bool unregistered = GlobalDx12SharedMemoryAllocationTracker().UnregisterAllocation(allocation_address);
      if (!unregistered) {
        std::ostringstream oss;
        oss << "Dx12SharedMemoryAllocator: Attempted to deregister allocation but it is untracked for address ("
            << allocation_address << ").";
        ORT_CXX_LOG(allocator->logger_, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
      }
    }

    // Run cleanup callbacks (e.g., QNN mem handle deregistration).
    for (const auto& clean_up_fn : allocation_record.clean_up_fns) {
      try {
        clean_up_fn(allocation_address);
      } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Dx12SharedMemoryAllocator: Exception in clean up callback for address ("
            << allocation_address << "): " << e.what();
        ORT_CXX_LOG(allocator->logger_, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
      }
    }

    // Unmap and release the D3D12 resource.
    if (resource != nullptr) {
      resource->Unmap(0, nullptr);
      resource->Release();
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Dx12SharedMemoryAllocator: Exception while freeing address (" << allocation_address << "): " << e.what();
    ORT_CXX_LOG(allocator->logger_, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
  }
}

Ort::Status Dx12SharedMemoryAllocator::GetAllocationDx12Info(void* address_within_allocation,
                                                             Dx12AllocationInfo& allocation_info_out) {
  const auto tracked_record = GlobalDx12SharedMemoryAllocationTracker().LookUp(address_within_allocation);
  RETURN_IF_NOT(tracked_record.has_value(), "Dx12SharedMemoryAllocator: Failed to look up tracked allocation.");

  void* const base_address = tracked_record->base_address;
  Dx12AllocationInfo dx12_info{};
  RETURN_IF_ERROR(tracked_record->allocator->GetAllocationDx12InfoForThisAllocator(base_address, dx12_info));

  // Adjust offset for address_within_allocation.
  const auto offset_from_base = std::distance(reinterpret_cast<std::byte*>(base_address),
                                              reinterpret_cast<std::byte*>(address_within_allocation));

  // Due to QNN API limitations (Qnn_MemDx12BufInfo_t does not include an offset field),
  // the DX12 memory allocator does not support nonzero offsets. The offset tracking is implemented
  // throughout this class, though, so that it is ready to go if API support arrives in the future,
  // at which point this check can be removed.
  RETURN_IF_NOT(offset_from_base == 0,
                "Dx12SharedMemoryAllocator: Offset tensors currently unsuppported."
                " Tensor data must begin at allocation base address.");

  dx12_info.offset += offset_from_base;

  allocation_info_out = std::move(dx12_info);
  return Ort::Status();
}

Ort::Status Dx12SharedMemoryAllocator::AddAllocationCleanUp(void* address_within_allocation,
                                                            AllocationCleanUpFn&& allocation_clean_up) {
  const auto tracked_record = GlobalDx12SharedMemoryAllocationTracker().LookUp(address_within_allocation);
  RETURN_IF_NOT(tracked_record.has_value(), "Dx12SharedMemoryAllocator: Failed to look up tracked allocation.");

  void* const base_address = tracked_record->base_address;
  return tracked_record->allocator->AddAllocationCleanUpForThisAllocator(base_address,
                                                                         std::move(allocation_clean_up));
}

Ort::Status Dx12SharedMemoryAllocator::GetAllocationDx12InfoForThisAllocator(void* allocation_base_address,
                                                                             Dx12AllocationInfo& allocation_info) {
  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_base_address);
  RETURN_IF(allocation_it == allocations_.end(), "Dx12SharedMemoryAllocator: Failed to get allocation info for address.");

  allocation_info = allocation_it->second.dx12_info;
  return Ort::Status();
}

Ort::Status Dx12SharedMemoryAllocator::AddAllocationCleanUpForThisAllocator(void* allocation_base_address,
                                                                            AllocationCleanUpFn&& allocation_clean_up) {
  RETURN_IF(allocation_clean_up == nullptr, "allocation_clean_up should not be empty.");

  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_base_address);
  RETURN_IF(allocation_it == allocations_.end(), "Dx12SharedMemoryAllocator: Failed to get allocation info for address.");

  allocation_it->second.clean_up_fns.emplace_back(std::move(allocation_clean_up));
  return Ort::Status();
}

#endif  // _WIN32

}  // namespace onnxruntime::qnn
