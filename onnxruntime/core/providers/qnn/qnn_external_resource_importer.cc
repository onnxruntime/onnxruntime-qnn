// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32

#include "core/providers/qnn/qnn_external_resource_importer.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/qnn_data_transfer.h"

#include <algorithm>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace onnxruntime {

// ============================================================================
// QnnExternalMemoryHandle Implementation
// ============================================================================

size_t GetHostAllocationImportAlignment() {
  static const size_t page_size = [] {
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    return static_cast<size_t>(system_info.dwPageSize);
  }();
  return page_size;
}

QnnExternalMemoryHandle::QnnExternalMemoryHandle()
    : d3d12_resource_(nullptr) {
  // Initialize base struct fields
  version = ORT_API_VERSION;
  ep_device = nullptr;
  descriptor.version = ORT_API_VERSION;
  descriptor.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  descriptor.native_handle = nullptr;
  descriptor.size_bytes = 0;
  descriptor.offset_bytes = 0;
  Release = ReleaseCallback;
}

void ORT_API_CALL QnnExternalMemoryHandle::ReleaseCallback(
    _In_ OrtExternalMemoryHandle* handle) noexcept {
  if (handle == nullptr) return;
  QnnExternalMemoryHandle* derived = static_cast<QnnExternalMemoryHandle*>(handle);

  // Unpublish the handle first so no new lookups or clean up hooks can reach it.
  {
    std::unique_lock write_lock{QnnExternalResourceImporterImpl::mutex_mem_handle_registry_};
    if (derived->IsHostAllocation()) {
      QnnExternalResourceImporterImpl::host_memory_registry_.erase(reinterpret_cast<uintptr_t>(derived->HostBase()));
    } else {
      QnnExternalResourceImporterImpl::mem_handle_registry_.erase(derived);
    }
  }

  // Run cleanup callbacks (e.g., QNN mem handle deregistration) before the backing memory is unmapped.
  InlinedVector<QnnExternalMemoryHandle::CleanUpRecord, 1> cleanup_callbacks;
  {
    std::scoped_lock lock{derived->cleanup_mutex_};
    cleanup_callbacks = std::move(derived->cleanup_callbacks_);
  }
  for (const auto& cleanup : cleanup_callbacks) {
    if (cleanup.owner.expired()) {
      continue;  // e.g. the QNN context already deregistered and dropped its mem handle registration
    }
    try {
      cleanup.callback(derived);
    } catch (const std::exception& e) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, e.what());
    } catch (...) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR,
                  "Caught exception while running import memory clean up callback.");
    }
  }

  // Undo the FastRPC registration.
  if (derived->IsHostAllocation() && derived->rpcmem_library_ != nullptr) {
    if (!qnn::DeregisterFastRpcBuffer(derived->rpcmem_library_->Api(), derived->HostBase(), derived->HostSize(),
                                      qnn::rpcmem::RPCMEM_ATTR_IMPORT_BUFFER)) {
      std::ostringstream oss;
      oss << "Failed to deregister imported host allocation from RPCMEM: " << derived->HostBase();
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
    }
  }

  delete derived;
}

// ============================================================================
// QnnExternalSemaphoreHandle Implementation
// ============================================================================

QnnExternalSemaphoreHandle::QnnExternalSemaphoreHandle() {
  // Initialize base struct fields
  version = ORT_API_VERSION;
  ep_device = nullptr;
  descriptor.version = ORT_API_VERSION;
  descriptor.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
  descriptor.native_handle = nullptr;
  Release = ReleaseCallback;
}

void ORT_API_CALL QnnExternalSemaphoreHandle::ReleaseCallback(
    _In_ OrtExternalSemaphoreHandle* handle) noexcept {
  if (handle == nullptr) return;
  auto* derived = static_cast<QnnExternalSemaphoreHandle*>(handle);
  delete derived;
}

// ============================================================================
// QnnExternalResourceImporterImpl Implementation
// ============================================================================

InlinedHashSet<QnnExternalMemoryHandle*> QnnExternalResourceImporterImpl::mem_handle_registry_;
std::map<uintptr_t, QnnExternalMemoryHandle*> QnnExternalResourceImporterImpl::host_memory_registry_;
std::shared_mutex QnnExternalResourceImporterImpl::mutex_mem_handle_registry_;

QnnExternalResourceImporterImpl::QnnExternalResourceImporterImpl(
    int device_id, const OrtApi& ort_api_in, OrtHardwareDeviceType device_type,
    std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library)
    : device_id_{device_id},
      ort_api_{ort_api_in},
      ep_api_{*ort_api_in.GetEpApi()},
      d3d12_device_{nullptr},
      supports_d3d12_resource_{device_type == OrtHardwareDeviceType_GPU},
      supports_host_allocation_{device_type == OrtHardwareDeviceType_NPU},
      rpcmem_library_override_{std::move(rpcmem_library)} {
  ort_version_supported = ORT_API_VERSION;

  // Memory operations
  CanImportMemory = CanImportMemoryImpl;
  ImportMemory = ImportMemoryImpl;
  ReleaseMemory = ReleaseMemoryImpl;
  CreateTensorFromMemory = CreateTensorFromMemoryImpl;

  // Semaphore operations
  CanImportSemaphore = CanImportSemaphoreImpl;
  ImportSemaphore = ImportSemaphoreImpl;
  ReleaseSemaphore = ReleaseSemaphoreImpl;
  WaitSemaphore = WaitSemaphoreImpl;
  SignalSemaphore = SignalSemaphoreImpl;

  // Release
  Release = ReleaseImpl;
}

bool ORT_API_CALL QnnExternalResourceImporterImpl::CanImportMemoryImpl(
    _In_ const OrtExternalResourceImporterImpl* this_ptr,
    _In_ OrtExternalMemoryHandleType handle_type) noexcept {
  const auto& impl = *static_cast<const QnnExternalResourceImporterImpl*>(this_ptr);
  // D3D12 resources (not heaps) are registered as QNN_MEM_TYPE_DX12, which only the GPU backend accepts. Host
  // allocations are mapped through FastRPC, which only the HTP backend uses.
  return (handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE && impl.supports_d3d12_resource_) ||
         (handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION && impl.supports_host_allocation_);
}

QnnExternalMemoryHandle* QnnExternalResourceImporterImpl::FindHostHandleLocked(uintptr_t address) {
  auto it = host_memory_registry_.upper_bound(address);
  if (it == host_memory_registry_.begin()) {
    return nullptr;
  }
  --it;
  return address - it->first < it->second->HostSize() ? it->second : nullptr;
}

std::optional<QnnExternalResourceImporterImpl::ImportedHostMemoryInfo>
QnnExternalResourceImporterImpl::FindImportedHostMemory(const void* address) {
  std::shared_lock read_lock{mutex_mem_handle_registry_};
  const QnnExternalMemoryHandle* handle = FindHostHandleLocked(reinterpret_cast<uintptr_t>(address));
  if (handle == nullptr) {
    return std::nullopt;
  }
  return ImportedHostMemoryInfo{handle->HostBase(), handle->HostSize(), handle->host_fd_};
}

Ort::Status QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(void* handle_or_address,
                                                                    ImportResourceCleanUpFn&& cleanup,
                                                                    std::weak_ptr<void> owner) {
  RETURN_IF(cleanup == nullptr, "cleanup should not be empty.");

  // Hold the registry lock so the handle cannot be released while the hook is added.
  std::shared_lock read_lock{mutex_mem_handle_registry_};
  QnnExternalMemoryHandle* handle = nullptr;
  if (const auto d3d12_it = mem_handle_registry_.find(static_cast<QnnExternalMemoryHandle*>(handle_or_address));
      d3d12_it != mem_handle_registry_.end()) {
    handle = *d3d12_it;
  } else {
    handle = FindHostHandleLocked(reinterpret_cast<uintptr_t>(handle_or_address));
  }
  RETURN_IF(handle == nullptr, "Import memory handle not found");

  std::scoped_lock lock{handle->cleanup_mutex_};
  // A long-lived import may be used by many sessions; drop hooks whose owner has gone away.
  auto& hooks = handle->cleanup_callbacks_;
  hooks.erase(std::remove_if(hooks.begin(), hooks.end(), [](const auto& hook) { return hook.owner.expired(); }),
              hooks.end());
  hooks.push_back({std::move(cleanup), std::move(owner)});
  return Ort::Status();
}

size_t QnnExternalResourceImporterImpl::ReleaseOutstandingHostImports() {
  std::vector<QnnExternalMemoryHandle*> outstanding;
  {
    std::shared_lock read_lock{mutex_mem_handle_registry_};
    outstanding.reserve(host_memory_registry_.size());
    for (const auto& [base, handle] : host_memory_registry_) {
      outstanding.push_back(handle);
    }
  }
  for (auto* handle : outstanding) {
    std::ostringstream oss;
    oss << "Releasing HOST_ALLOCATION import that the app did not release (address: " << handle->HostBase()
        << ", size: " << handle->HostSize() << "). Release import handles before unregistering the EP library.";
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
    QnnExternalMemoryHandle::ReleaseCallback(handle);
  }
  return outstanding.size();
}

OrtStatus* QnnExternalResourceImporterImpl::ImportHostAllocation(const OrtExternalMemoryDescriptor& desc,
                                                                 QnnExternalMemoryHandle& handle) {
  if (!supports_host_allocation_) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 "HOST_ALLOCATION import is only supported for QNN NPU (HTP) devices");
  }

  const auto base = reinterpret_cast<uintptr_t>(desc.native_handle);
  if (desc.native_handle == nullptr || desc.size_bytes == 0) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, "HOST_ALLOCATION import requires a non-null address and size");
  }
  const size_t alignment = GetHostAllocationImportAlignment();
  if (base % alignment != 0 || desc.size_bytes % alignment != 0) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 ("HOST_ALLOCATION import requires an address and size aligned to the " +
                                  std::to_string(alignment) + "-byte page size")
                                     .c_str());
  }
  if (desc.offset_bytes >= desc.size_bytes) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, "HOST_ALLOCATION import offset must be less than its size");
  }
  if (desc.size_bytes > UINTPTR_MAX - base) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, "HOST_ALLOCATION import range overflows the address space");
  }
  if (qnn::HtpSharedMemoryAllocator::IsAllocationTracked(desc.native_handle)) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Memory from the QnnHtpShared allocator is already shared; bind it directly instead");
  }

  // RpcMemLibraryManager caches the library process-wide, so this only loads it on the first import.
  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library = rpcmem_library_override_;
  if (rpcmem_library == nullptr) {
    std::string error_message;
    rpcmem_library = qnn::RpcMemLibraryManager::GetOrCreate(error_message);
    if (rpcmem_library == nullptr) {
      return ort_api_.CreateStatus(ORT_FAIL, ("Unable to load RPCMEM for HOST_ALLOCATION import: " +
                                              error_message)
                                                 .c_str());
    }
  }
  const auto& api = rpcmem_library->Api();
  if (api.register_buf == nullptr || api.to_fd == nullptr) {
    return ort_api_.CreateStatus(ORT_FAIL, "RPCMEM library does not support buffer registration");
  }

  // Keep the registry locked across registration so overlapping imports cannot race.
  std::unique_lock write_lock{mutex_mem_handle_registry_};
  const auto next = host_memory_registry_.lower_bound(base);
  const bool overlaps_next = next != host_memory_registry_.end() && next->first - base < desc.size_bytes;
  const bool overlaps_prev = next != host_memory_registry_.begin() &&
                             base - std::prev(next)->first < std::prev(next)->second->HostSize();
  if (overlaps_next || overlaps_prev) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 "HOST_ALLOCATION import overlaps an already imported host allocation");
  }

  // FastRPC mappings are process-wide. Refuse memory that is already mapped (e.g. the app's own rpcmem_alloc
  // buffer): releasing this import would otherwise tear down a mapping the EP does not own.
  if (api.to_fd(desc.native_handle) != -1) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 "HOST_ALLOCATION import address is already mapped by FastRPC; bind it directly instead");
  }

  // Map the existing pages into the NPU address space once. The buffer is written by graph outputs, so it is not
  // registered read-only. The import commits when it is published in the registry; any failure before that,
  // including an exception, undoes the mapping.
  try {
    const int fd = qnn::RegisterFastRpcBuffer(api, desc.native_handle, desc.size_bytes,
                                              qnn::rpcmem::RPCMEM_ATTR_IMPORT_BUFFER);
    if (fd == -1) {
      return ort_api_.CreateStatus(ORT_FAIL, "Failed to register host allocation with FastRPC");
    }
    handle.host_fd_ = fd;
    handle.rpcmem_library_ = rpcmem_library;
    host_memory_registry_.emplace(base, &handle);
  } catch (...) {
    handle.rpcmem_library_.reset();
    try {
      qnn::DeregisterFastRpcBuffer(api, desc.native_handle, desc.size_bytes, qnn::rpcmem::RPCMEM_ATTR_IMPORT_BUFFER);
    } catch (...) {
    }
    return ort_api_.CreateStatus(ORT_FAIL, "Failed to import host allocation");
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL QnnExternalResourceImporterImpl::ImportMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalMemoryDescriptor* desc,
    _Outptr_ OrtExternalMemoryHandle** out_handle) noexcept {
  auto& impl = *static_cast<QnnExternalResourceImporterImpl*>(this_ptr);

  if (desc == nullptr || out_handle == nullptr) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                      "desc and out_handle cannot be nullptr");
  }

  *out_handle = nullptr;

  // ImportMemory is noexcept: report exceptions (e.g. allocation failures) as errors instead of terminating. Nothing
  // is published or handed out until the import has fully succeeded.
  try {
    // Create and return the derived handle
    std::unique_ptr<QnnExternalMemoryHandle> handle{
        new (std::nothrow) QnnExternalMemoryHandle()};
    if (!handle) {
      return impl.ort_api_.CreateStatus(ORT_FAIL, "Failed to allocate external memory handle");
    }

    if (desc->handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION) {
      handle->ep_device = nullptr;
      handle->descriptor = *desc;
      // On success the handle is published in the host registry; ownership passes to the caller.
      if (OrtStatus* status = impl.ImportHostAllocation(*desc, *handle); status != nullptr) {
        return status;
      }
      *out_handle = handle.release();
      return nullptr;
    }

    switch (desc->handle_type) {
      case ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: {
        if (!impl.supports_d3d12_resource_) {
          return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                            "D3D12_RESOURCE import is only supported for QNN GPU devices");
        }
        if (desc->offset_bytes != 0) {
          return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                            "Not supported importing resource with non-zero offset");
        }
        std::lock_guard<std::mutex> lock{impl.d3d12_device_mutex_};
        if (!impl.d3d12_device_) {
          HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&impl.d3d12_device_));
          if (FAILED(hr) || impl.d3d12_device_ == nullptr) {
            return impl.ort_api_.CreateStatus(ORT_FAIL, "D3D12CreateDevice failed.");
          }
        }

        HRESULT hr = impl.d3d12_device_->OpenSharedHandle(
            desc->native_handle,
            IID_PPV_ARGS(&(handle->d3d12_resource_)));

        if (!SUCCEEDED(hr)) {
          return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                            "Invalid external memory handle");
        }

        break;
      }
      default:
        return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "Unsupported external memory handle type");
    }

    handle->ep_device = nullptr;
    handle->descriptor = *desc;

    // Publish before handing out ownership, so a failure here leaves nothing behind.
    {
      std::unique_lock write_lock{mutex_mem_handle_registry_};
      mem_handle_registry_.insert(handle.get());
    }
    *out_handle = handle.release();

    return nullptr;
  } catch (const std::exception& e) {
    return impl.ort_api_.CreateStatus(ORT_FAIL, (std::string("Failed to import external memory: ") + e.what()).c_str());
  } catch (...) {
    return impl.ort_api_.CreateStatus(ORT_FAIL, "Failed to import external memory");
  }
}

void ORT_API_CALL QnnExternalResourceImporterImpl::ReleaseMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalMemoryHandle* handle) noexcept {
  // The handle has a Release callback that does the actual cleanup
  if (handle) {
    handle->Release(handle);
  }
}

OrtStatus* ORT_API_CALL QnnExternalResourceImporterImpl::CreateTensorFromMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalMemoryHandle* mem_handle,
    _In_ const OrtExternalTensorDescriptor* tensor_desc,
    _Outptr_ OrtValue** out_tensor) noexcept {
  auto& impl = *static_cast<QnnExternalResourceImporterImpl*>(this_ptr);

  if (mem_handle == nullptr || tensor_desc == nullptr || out_tensor == nullptr) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                      "mem_handle, tensor_desc, and out_tensor cannot be nullptr");
  }

  *out_tensor = nullptr;

  const auto* qnn_handle = static_cast<const QnnExternalMemoryHandle*>(mem_handle);

  if (qnn_handle->IsHostAllocation()) {
    size_t tensor_size_bytes = 0;
    try {
      tensor_size_bytes = qnn::utils::GetOnnxTensorDataSizeInBytes(gsl::span{tensor_desc->shape, tensor_desc->rank},
                                                                   tensor_desc->element_type);
    } catch (...) {
      return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported tensor element type");
    }

    // Tensors live in [base + import offset, base + import size).
    const size_t import_offset = qnn_handle->descriptor.offset_bytes;
    const size_t data_size = qnn_handle->HostSize() - import_offset;
    if (tensor_desc->offset_bytes > data_size || tensor_size_bytes > data_size - tensor_desc->offset_bytes) {
      std::ostringstream oss;
      oss << "Tensor size (" << tensor_size_bytes << " bytes) at offset (" << tensor_desc->offset_bytes
          << " bytes) exceeds imported host allocation (" << data_size << " bytes after the import offset)";
      return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
    }

    // The tensor points at the real CPU address so CPU consumers can read it directly. The HOST_ACCESSIBLE memory
    // info matches the QnnHtpShared allocator so ORT treats it as CPU-compatible and does not copy it.
    void* data_ptr = static_cast<char*>(qnn_handle->HostBase()) + import_offset + tensor_desc->offset_bytes;
    OrtMemoryInfo* memory_info = nullptr;
    OrtStatus* status = qnn::CreateHtpSharedMemoryInfo(impl.ort_api_, &memory_info);
    if (status != nullptr) {
      return status;
    }

    status = impl.ort_api_.CreateTensorWithDataAsOrtValue(memory_info, data_ptr, tensor_size_bytes,
                                                          tensor_desc->shape, tensor_desc->rank,
                                                          tensor_desc->element_type, out_tensor);
    impl.ort_api_.ReleaseMemoryInfo(memory_info);
    return status;
  }

  if (tensor_desc->offset_bytes != 0) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Not supported create tensor with non-zero offset");
  }

  size_t tensor_size_bytes = 0;
  try {
    tensor_size_bytes = qnn::utils::GetOnnxTensorDataSizeInBytes(gsl::span{tensor_desc->shape, tensor_desc->rank}, tensor_desc->element_type);
  } catch (...) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported tensor element type");
  }

  size_t available_size = qnn_handle->descriptor.size_bytes - qnn_handle->descriptor.offset_bytes - tensor_desc->offset_bytes;

  if (tensor_size_bytes > available_size) {
    std::ostringstream oss;
    oss << "Tensor size (" << tensor_size_bytes << " bytes) exceeds available memory ("
        << available_size << " bytes)";
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
  }

  // Calculate the data pointer with offset
  void* data_ptr = static_cast<char*>((void*)qnn_handle) + tensor_desc->offset_bytes;

  // Create memory info for the GPU device. Using the existing name DML works for us.
  OrtMemoryInfo* memory_info = nullptr;
  const uint32_t vendor_id{'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24)};
  OrtStatus* status = impl.ort_api_.CreateMemoryInfo_V2(
      "D3D12_RESOURCE_IMPORT",
      OrtMemoryInfoDeviceType_GPU,
      vendor_id,
      impl.device_id_,
      OrtDeviceMemoryType_DEFAULT,
      0,
      OrtDeviceAllocator,
      &memory_info);
  if (status != nullptr) {
    return status;
  }

  // Create tensor from the external data
  status = impl.ort_api_.CreateTensorWithDataAsOrtValue(
      memory_info,
      data_ptr,
      tensor_size_bytes,
      tensor_desc->shape,
      tensor_desc->rank,
      tensor_desc->element_type,
      out_tensor);

  impl.ort_api_.ReleaseMemoryInfo(memory_info);

  return status;
}

// ============================================================================
// Semaphore Operations
// ============================================================================

bool ORT_API_CALL QnnExternalResourceImporterImpl::CanImportSemaphoreImpl(
    _In_ const OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalSemaphoreType /*type*/) noexcept {
  // Currently not implemented.
  return false;
  /*
  // Supports D3D12 fence only.
  return type == ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
  */
}

OrtStatus* ORT_API_CALL QnnExternalResourceImporterImpl::ImportSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalSemaphoreDescriptor* /* desc*/,
    _Outptr_ OrtExternalSemaphoreHandle** /*out_handle*/) noexcept {
  auto& impl = *static_cast<QnnExternalResourceImporterImpl*>(this_ptr);

  // Currently not implemented.
  return impl.ort_api_.CreateStatus(ORT_NOT_IMPLEMENTED, "Not implemented");

  /*

  if (desc == nullptr || out_handle == nullptr) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "desc and out_handle cannot be nullptr");
  }

  *out_handle = nullptr;

  // Create and return the derived handle
  std::unique_ptr<QnnExternalSemaphoreHandle> handle{
      new (std::nothrow) QnnExternalSemaphoreHandle()};
  if (!handle) {
    return impl.ort_api_.CreateStatus(ORT_FAIL, "Failed to allocate external semaphore handle");
  }

  switch (desc->type) {
    case ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE: {
      std::lock_guard<std::mutex> lock{impl.d3d12_device_mutex_};
      if (!impl.d3d12_device_) {
        HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&impl.d3d12_device_));
        if (FAILED(hr) || impl.d3d12_device_ == nullptr) {
          return impl.ort_api_.CreateStatus(ORT_FAIL, "D3D12CreateDevice failed.");
        }
      }

      HRESULT hr = impl.d3d12_device_->OpenSharedHandle(
          desc->native_handle,
          IID_PPV_ARGS(&(handle->d3d12_fence_)));

      if (!SUCCEEDED(hr)) {
        return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "Invalid external D3D12 fence handle");
      }

      break;
    }
    case ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_WIN32:
    case ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_OPAQUE_FD:
    default:
      return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                       "Invalid external semaphore type");
  }
  handle->ep_device = nullptr;
  handle->descriptor = *desc;

  *out_handle = handle.release();
  return nullptr;
  */
}

void ORT_API_CALL QnnExternalResourceImporterImpl::ReleaseSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalSemaphoreHandle* handle) noexcept {
  // The handle has a Release callback that does the actual cleanup
  if (handle) {
    handle->Release(handle);
  }
}

OrtStatus* ORT_API_CALL QnnExternalResourceImporterImpl::WaitSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ OrtExternalSemaphoreHandle* /* handle */,
    _In_ OrtSyncStream* /*sync_stream*/,
    _In_ uint64_t /*value*/) noexcept {
  auto& impl = *static_cast<QnnExternalResourceImporterImpl*>(this_ptr);

  // Currently not implemented.
  return impl.ort_api_.CreateStatus(ORT_NOT_IMPLEMENTED, "Not implemented");

  /*

  if (handle == nullptr || sync_stream == nullptr) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "handle and sync_stream cannot be nullptr");
  }

  auto* sem_handle = static_cast<QnnExternalSemaphoreHandle*>(handle);

  return nullptr;
  */
}

OrtStatus* ORT_API_CALL QnnExternalResourceImporterImpl::SignalSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ OrtExternalSemaphoreHandle* /* handle */,
    _In_ OrtSyncStream* /*sync_stream*/,
    _In_ uint64_t /*value*/) noexcept {
  auto& impl = *static_cast<QnnExternalResourceImporterImpl*>(this_ptr);

  // Currently not implemented.
  return impl.ort_api_.CreateStatus(ORT_NOT_IMPLEMENTED, "Not implemented");

  /*

  if (handle == nullptr || sync_stream == nullptr) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "handle and sync_stream cannot be nullptr");
  }

  auto* sem_handle = static_cast<QnnExternalSemaphoreHandle*>(handle);

  return nullptr;
  */
}

void ORT_API_CALL QnnExternalResourceImporterImpl::ReleaseImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr) noexcept {
  if (this_ptr == nullptr) {
    return;
  }
  delete static_cast<QnnExternalResourceImporterImpl*>(this_ptr);
}

// ============================================================================
// QnnSyncStreamImpl Implementation
// ============================================================================

QnnSyncStreamImpl::QnnSyncStreamImpl(int device_id, const OrtApi& ort_api_in)
    : device_id_(device_id), ort_api_{ort_api_in} {
  ort_version_supported = ORT_API_VERSION;

  // Wire up base struct function pointers
  Release = ReleaseImpl;
  GetHandle = GetHandleImpl;
  CreateNotification = nullptr;  // Not implemented
}

QnnSyncStreamImpl::~QnnSyncStreamImpl() {
}

void* ORT_API_CALL QnnSyncStreamImpl::GetHandleImpl(
    _In_ OrtSyncStreamImpl* /*this_ptr*/) noexcept {
  // auto& impl = *static_cast<QnnSyncStreamImpl*>(this_ptr);

  // Currently not implemented.
  return static_cast<void*>(nullptr);
}

void ORT_API_CALL QnnSyncStreamImpl::ReleaseImpl(
    _In_ OrtSyncStreamImpl* this_ptr) noexcept {
  if (this_ptr == nullptr) {
    return;
  }
  delete static_cast<QnnSyncStreamImpl*>(this_ptr);
}

}  // namespace onnxruntime

#endif  // _WIN32
