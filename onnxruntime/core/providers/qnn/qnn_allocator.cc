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

OrtStatus* HtpSharedMemoryAllocator::Create(const OrtApi& ort_api,
                                            const OrtMemoryInfo* memory_info,
                                            OrtAllocator** allocator) noexcept {
  *allocator = nullptr;

  try {
    auto htp_allocator = std::make_unique<HtpSharedMemoryAllocator>(memory_info);
    *allocator = htp_allocator.release();
  } catch (const std::exception& e) {
    return ort_api.CreateStatus(ORT_FAIL, e.what());
  }

  return nullptr;
}

void* ORT_API_CALL HtpSharedMemoryAllocator::AllocImpl(struct OrtAllocator* this_, size_t requested_size) {
  HtpSharedMemoryAllocator* allocator = static_cast<HtpSharedMemoryAllocator*>(this_);
  if (requested_size == 0) {
    return nullptr;
  }

  {
    std::scoped_lock g{allocator->allocations_mutex_};
    for (auto& [address, record] : allocator->allocations_) {
      // Exact-size reuse keeps registrations for tensor views stable and avoids
      // retaining an oversized backing allocation for a smaller request.
      if (!record.in_use && record.shared_memory_info.total_size == requested_size) {
        // Context destruction already deregisters its QNN handles. Drop
        // expired hooks so factory allocators can outlive many sessions.
        auto& hooks = record.clean_up_fns;
        hooks.erase(std::remove_if(hooks.begin(), hooks.end(), [](const auto& hook) {
                      return hook.owner.has_value() && hook.owner->expired();
                    }),
                    hooks.end());
        record.in_use = true;
        allocator->cached_bytes_ -= requested_size;
        --allocator->cached_allocations_;
        return const_cast<void*>(address);
      }
    }
  }

  std::string error_message;
  auto rpcmem_library = allocator->rpcmem_library_;
  if (rpcmem_library == nullptr) {
    rpcmem_library = RpcMemLibraryManager::GetOrCreate(error_message);
  }
  if (rpcmem_library == nullptr) {
    ORT_CXX_API_THROW("Unable to load RPCMEM for QnnHtpShared allocator: " + error_message, ORT_EP_FAIL);
  }

  const size_t shared_memory_block_size_in_bytes = requested_size;

  // rpcmem_alloc() has an int size parameter. make sure we don't overflow.
  // TODO switch to rpcmem_alloc2() which has size_t size parameter.
  // need to verify that rpcmem_alloc2() is available in all environments we care about.
  const SafeInt<int> shared_memory_block_size_in_bytes_int = shared_memory_block_size_in_bytes;

  // allocate shared memory
  void* shared_memory_raw = rpcmem_library->Api().alloc(rpcmem::RPCMEM_HEAP_ID_SYSTEM,
                                                        rpcmem::RPCMEM_DEFAULT_FLAGS,
                                                        shared_memory_block_size_in_bytes_int);
  if (shared_memory_raw == nullptr) {
    ORT_CXX_API_THROW("rpcmem_alloc() failed to allocate and returned nullptr.", ORT_EP_FAIL);
  }
  auto shared_memory = WrapSharedMemoryWithUniquePtr(shared_memory_raw, rpcmem_library->Api());

  const size_t allocation_alignment = AllocationAlignment();
  if (!IsAligned(shared_memory_raw, allocation_alignment)) {
    ORT_CXX_API_THROW(
        ("Shared memory address does not have required alignment (" +
         std::to_string(allocation_alignment) + " bytes)."),
        ORT_EP_FAIL);
  }

  // get shared memory fd
  const auto shared_memory_fd = rpcmem_library->Api().to_fd(shared_memory.get());
  if (shared_memory_fd == -1) {
    ORT_CXX_API_THROW("rpcmem_to_fd() returned invalid file descriptor.", ORT_EP_FAIL);
  }

  std::byte* allocation_address = reinterpret_cast<std::byte*>(shared_memory_raw);

  // Publish both records before returning the buffer. Roll back the local
  // record if global tracking fails; shared_memory still owns the RPC buffer.
  {
    AllocationRecord allocation_record{};
    allocation_record.shared_memory_info.fd = shared_memory_fd;
    allocation_record.shared_memory_info.offset = 0;
    allocation_record.shared_memory_info.total_size = shared_memory_block_size_in_bytes;
    allocation_record.rpcmem_library = rpcmem_library;

    std::scoped_lock g{allocator->allocations_mutex_};
    const bool inserted = allocator->allocations_.emplace(allocation_address, std::move(allocation_record)).second;
    if (!inserted) {
      ORT_CXX_API_THROW("Allocation record already exists for address.", ORT_EP_FAIL);
    }
    try {
      if (!GlobalHtpSharedMemoryAllocationTracker().RegisterAllocation(allocation_address,
                                                                       shared_memory_block_size_in_bytes,
                                                                       *allocator)) {
        ORT_CXX_API_THROW("Attempted to register allocation but it is already tracked for address.", ORT_EP_FAIL);
      }
    } catch (...) {
      allocator->allocations_.erase(allocation_address);
      throw;
    }
  }

  shared_memory.release();
  return allocation_address;
}

HtpSharedMemoryAllocator::~HtpSharedMemoryAllocator() {
  ReleaseCachedAllocations(/*release_in_use=*/true);
}

void ORT_API_CALL HtpSharedMemoryAllocator::FreeImpl(struct OrtAllocator* this_, void* allocation_address) {
  auto* allocator = static_cast<HtpSharedMemoryAllocator*>(this_);
  if (allocation_address == nullptr) {
    return;
  }

  auto oversized_allocation = [allocator, allocation_address]() {
    std::scoped_lock g{allocator->allocations_mutex_};
    const auto it = allocator->allocations_.find(allocation_address);
    if (it == allocator->allocations_.end() || !it->second.in_use) {
      ORT_CXX_API_THROW("Failed to get live allocation info for address.", ORT_EP_FAIL);
    }
    auto& record = it->second;
    if (record.shared_memory_info.total_size > kMaxCachedBytes) {
      return allocator->allocations_.extract(it);
    }
    record.in_use = false;
    record.last_free = ++allocator->free_sequence_;
    allocator->cached_bytes_ += record.shared_memory_info.total_size;
    ++allocator->cached_allocations_;
    return decltype(allocator->allocations_)::node_type{};
  }();
  if (!oversized_allocation.empty()) {
    allocator->ReleaseAllocation(allocation_address, std::move(oversized_allocation.mapped()));
    return;
  }

  // Evict the oldest idle buffers until both limits are satisfied. Cleanup can
  // call into QNN, so never run it while holding allocations_mutex_. Alloc/Free
  // on other buffers and Shrink may proceed while an eviction is cleaned up.
  for (;;) {
    auto allocation_node = [allocator]() {
      std::scoped_lock g{allocator->allocations_mutex_};
      decltype(allocator->allocations_)::node_type node;
      if (allocator->cached_bytes_ <= kMaxCachedBytes &&
          allocator->cached_allocations_ <= kMaxCachedAllocations) {
        return node;
      }
      auto oldest = allocator->allocations_.end();
      for (auto it = allocator->allocations_.begin(); it != allocator->allocations_.end(); ++it) {
        if (!it->second.in_use &&
            (oldest == allocator->allocations_.end() || it->second.last_free < oldest->second.last_free)) {
          oldest = it;
        }
      }
      if (oldest != allocator->allocations_.end()) {
        allocator->cached_bytes_ -= oldest->second.shared_memory_info.total_size;
        --allocator->cached_allocations_;
        node = allocator->allocations_.extract(oldest);
      }
      return node;
    }();
    if (allocation_node.empty()) {
      return;
    }
    allocator->ReleaseAllocation(const_cast<void*>(allocation_node.key()), std::move(allocation_node.mapped()));
  }
}

OrtStatus* ORT_API_CALL HtpSharedMemoryAllocator::ShrinkImpl(struct OrtAllocator* this_) noexcept {
  auto* allocator = static_cast<HtpSharedMemoryAllocator*>(this_);
  try {
    allocator->ReleaseCachedAllocations();
    return nullptr;
  } catch (const std::exception& e) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, e.what());
  } catch (...) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "Failed to shrink HTP shared memory pool.");
  }
}

void HtpSharedMemoryAllocator::ReleaseCachedAllocations(bool release_in_use) {
  for (;;) {
    auto allocation_node = [this, release_in_use]() {
      std::scoped_lock g{allocations_mutex_};
      for (auto it = allocations_.begin(); it != allocations_.end(); ++it) {
        if (release_in_use || !it->second.in_use) {
          if (!it->second.in_use) {
            cached_bytes_ -= it->second.shared_memory_info.total_size;
            --cached_allocations_;
          }
          return allocations_.extract(it);
        }
      }
      return decltype(allocations_)::node_type{};
    }();
    if (allocation_node.empty()) {
      return;
    }
    ReleaseAllocation(const_cast<void*>(allocation_node.key()), std::move(allocation_node.mapped()));
  }
}

void HtpSharedMemoryAllocator::ReleaseAllocation(void* allocation_address, AllocationRecord&& allocation_record) noexcept {
  try {
    // Run every registration cleanup before freeing the backing memory. The
    // record keeps RPCMEM loaded until the buffer has actually been released.
    auto shared_memory = WrapSharedMemoryWithUniquePtr(allocation_address, allocation_record.rpcmem_library->Api());
    if (!GlobalHtpSharedMemoryAllocationTracker().UnregisterAllocation(allocation_address)) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Attempted to deregister an untracked shared allocation.");
    }
    for (auto& clean_up_fn : allocation_record.clean_up_fns) {
      try {
        clean_up_fn.callback(allocation_address);
      } catch (const std::exception& e) {
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, e.what());
      } catch (...) {
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Caught exception while cleaning up shared allocation.");
      }
    }
  } catch (const std::exception& e) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, e.what());
  } catch (...) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Caught exception while releasing shared allocation.");
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

bool HtpSharedMemoryAllocator::IsAllocationTracked(void* address_within_allocation) {
  const auto tracked_record = GlobalHtpSharedMemoryAllocationTracker().LookUp(address_within_allocation);
  if (!tracked_record.has_value()) {
    return false;
  }
  auto* allocator = tracked_record->allocator.get();
  std::scoped_lock g{allocator->allocations_mutex_};
  const auto it = allocator->allocations_.find(tracked_record->base_address);
  return it != allocator->allocations_.end() && it->second.in_use;
}

Ort::Status HtpSharedMemoryAllocator::AddAllocationCleanUp(void* address_within_allocation,
                                                           AllocationCleanUpFn&& allocation_clean_up,
                                                           std::optional<std::weak_ptr<void>> owner) {
  const auto tracked_record = GlobalHtpSharedMemoryAllocationTracker().LookUp(address_within_allocation);
  RETURN_IF_NOT(tracked_record.has_value(), "Failed to look up tracked allocation.");

  void* const base_address = tracked_record->base_address;
  return tracked_record->allocator->AddAllocationCleanUpForThisAllocator(base_address,
                                                                         std::move(allocation_clean_up), std::move(owner));
}

Ort::Status HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfoForThisAllocator(void* allocation_base_address,
                                                                                    SharedMemoryInfo& allocation_info) {
  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_base_address);
  RETURN_IF(allocation_it == allocations_.end() || !allocation_it->second.in_use,
            "Failed to get live allocation info for address.");

  allocation_info = allocation_it->second.shared_memory_info;
  return Ort::Status();
}

Ort::Status HtpSharedMemoryAllocator::AddAllocationCleanUpForThisAllocator(void* allocation_base_address,
                                                                           AllocationCleanUpFn&& allocation_clean_up,
                                                                           std::optional<std::weak_ptr<void>> owner) {
  RETURN_IF(allocation_clean_up == nullptr, "allocation_clean_up should not be empty.");

  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_base_address);
  RETURN_IF(allocation_it == allocations_.end() || !allocation_it->second.in_use,
            "Failed to get live allocation info for address.");

  auto& clean_up_fns = allocation_it->second.clean_up_fns;
  clean_up_fns.push_back({std::move(allocation_clean_up), std::move(owner)});
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

bool Dx12SharedMemoryAllocator::IsAllocationTracked(void* address_within_allocation) {
  return GlobalDx12SharedMemoryAllocationTracker().LookUp(address_within_allocation).has_value();
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
