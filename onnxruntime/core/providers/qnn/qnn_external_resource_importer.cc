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

namespace onnxruntime {

// ============================================================================
// QnnExternalMemoryHandle Implementation
// ============================================================================

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
    if (derived->kind_ == QnnExternalMemoryKind::kHostAllocation) {
      QnnExternalResourceImporterImpl::host_memory_registry_.erase(reinterpret_cast<uintptr_t>(derived->host_base_));
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
    try {
      cleanup.callback(derived);
    } catch (const std::exception& e) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, e.what());
    } catch (...) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR,
                  "Caught exception while running import memory clean up callback.");
    }
  }

  // Undo the FastRPC registration. A -1 fd requests deregistration.
  if (derived->kind_ == QnnExternalMemoryKind::kHostAllocation && derived->rpcmem_library_ != nullptr) {
    const auto& api = derived->rpcmem_library_->Api();
    api.register_buf(derived->host_base_, derived->host_size_, -1, qnn::rpcmem::RPCMEM_ATTR_IMPORT_BUFFER);
    if (api.to_fd(derived->host_base_) != -1) {
      std::ostringstream oss;
      oss << "Failed to deregister imported host allocation from RPCMEM: " << derived->host_base_;
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
    int device_id, const OrtApi& ort_api_in,
    bool supports_host_allocation,
    std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library) : device_id_{device_id},
                                                          ort_api_{ort_api_in},
                                                          ep_api_{*ort_api_in.GetEpApi()},
                                                          d3d12_device_{nullptr},
                                                          supports_host_allocation_{supports_host_allocation},
                                                          rpcmem_library_{std::move(rpcmem_library)} {
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
  // Supports D3D12 resource handles (not heap handles), and host allocations on NPU devices.
  return handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE ||
         (handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION && impl.supports_host_allocation_);
}

std::optional<QnnExternalResourceImporterImpl::ImportedHostMemoryInfo>
QnnExternalResourceImporterImpl::FindImportedHostMemory(const void* address) {
  const auto addr = reinterpret_cast<uintptr_t>(address);
  std::shared_lock read_lock{mutex_mem_handle_registry_};
  auto it = host_memory_registry_.upper_bound(addr);
  if (it == host_memory_registry_.begin()) {
    return std::nullopt;
  }
  --it;
  const QnnExternalMemoryHandle& handle = *it->second;
  if (addr - it->first >= handle.host_size_) {
    return std::nullopt;
  }
  return ImportedHostMemoryInfo{handle.host_base_, handle.host_size_, handle.host_fd_};
}

Ort::Status QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(void* handle_or_address,
                                                                    ImportResourceCleanUpFn&& cleanup,
                                                                    std::optional<std::weak_ptr<void>> owner) {
  RETURN_IF(cleanup == nullptr, "cleanup should not be empty.");

  // Hold the registry lock so the handle cannot be released while the hook is added.
  std::shared_lock read_lock{mutex_mem_handle_registry_};
  QnnExternalMemoryHandle* handle = nullptr;
  if (const auto d3d12_it = mem_handle_registry_.find(static_cast<QnnExternalMemoryHandle*>(handle_or_address));
      d3d12_it != mem_handle_registry_.end()) {
    handle = *d3d12_it;
  } else {
    const auto addr = reinterpret_cast<uintptr_t>(handle_or_address);
    auto it = host_memory_registry_.upper_bound(addr);
    if (it != host_memory_registry_.begin()) {
      --it;
      if (addr - it->first < it->second->host_size_) {
        handle = it->second;
      }
    }
  }
  RETURN_IF(handle == nullptr, "Import memory handle not found");

  std::scoped_lock lock{handle->cleanup_mutex_};
  // A long-lived import may be used by many sessions; drop hooks whose owner has gone away.
  auto& hooks = handle->cleanup_callbacks_;
  hooks.erase(std::remove_if(hooks.begin(), hooks.end(),
                             [](const auto& hook) { return hook.owner.has_value() && hook.owner->expired(); }),
              hooks.end());
  hooks.push_back({std::move(cleanup), std::move(owner)});
  return Ort::Status();
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
  if (base % kHostAllocationImportAlignment != 0 || desc.size_bytes % kHostAllocationImportAlignment != 0) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 "HOST_ALLOCATION import requires a 4096-byte aligned address and size");
  }
  if (desc.size_bytes > UINTPTR_MAX - base) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, "HOST_ALLOCATION import range overflows the address space");
  }
  if (qnn::HtpSharedMemoryAllocator::IsAllocationTracked(desc.native_handle)) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Memory from the QnnHtpShared allocator is already shared; bind it directly instead");
  }

  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library;
  {
    std::lock_guard<std::mutex> lock{rpcmem_library_mutex_};
    if (rpcmem_library_ == nullptr) {
      std::string error_message;
      rpcmem_library_ = qnn::RpcMemLibraryManager::GetOrCreate(error_message);
      if (rpcmem_library_ == nullptr) {
        return ort_api_.CreateStatus(ORT_FAIL, ("Unable to load RPCMEM for HOST_ALLOCATION import: " +
                                                error_message)
                                                   .c_str());
      }
    }
    rpcmem_library = rpcmem_library_;
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
                             base - std::prev(next)->first < std::prev(next)->second->host_size_;
  if (overlaps_next || overlaps_prev) {
    return ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 "HOST_ALLOCATION import overlaps an already imported host allocation");
  }

  // Map the existing pages into the NPU address space once. The buffer is written by graph outputs, so it is not
  // registered read-only. register_buf does not return a status; rpcmem_to_fd verifies the registration.
  api.register_buf(desc.native_handle, desc.size_bytes, /*fd*/ 0, qnn::rpcmem::RPCMEM_ATTR_IMPORT_BUFFER);
  const int fd = api.to_fd(desc.native_handle);
  if (fd == -1) {
    return ort_api_.CreateStatus(ORT_FAIL, "Failed to register host allocation with FastRPC");
  }

  handle.kind_ = QnnExternalMemoryKind::kHostAllocation;
  handle.host_base_ = desc.native_handle;
  handle.host_size_ = desc.size_bytes;
  handle.host_fd_ = fd;
  handle.rpcmem_library_ = std::move(rpcmem_library);
  host_memory_registry_.emplace(base, &handle);
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

  if (desc->offset_bytes != 0) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Not supported importing resource with non-zero offset");
  }

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

  auto raw_handle = handle.release();
  *out_handle = raw_handle;

  std::unique_lock write_lock{mutex_mem_handle_registry_};
  mem_handle_registry_.insert(raw_handle);

  return nullptr;
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

  if (qnn_handle->kind_ == QnnExternalMemoryKind::kHostAllocation) {
    size_t tensor_size_bytes = 0;
    try {
      tensor_size_bytes = qnn::utils::GetOnnxTensorDataSizeInBytes(gsl::span{tensor_desc->shape, tensor_desc->rank},
                                                                   tensor_desc->element_type);
    } catch (...) {
      return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported tensor element type");
    }

    if (tensor_desc->offset_bytes > qnn_handle->host_size_ ||
        tensor_size_bytes > qnn_handle->host_size_ - tensor_desc->offset_bytes) {
      std::ostringstream oss;
      oss << "Tensor size (" << tensor_size_bytes << " bytes) at offset (" << tensor_desc->offset_bytes
          << " bytes) exceeds imported host allocation (" << qnn_handle->host_size_ << " bytes)";
      return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
    }

    // The tensor points at the real CPU address so CPU consumers can read it directly. The HOST_ACCESSIBLE memory
    // info matches the QnnHtpShared allocator so ORT treats it as CPU-compatible and does not copy it.
    void* data_ptr = static_cast<char*>(qnn_handle->host_base_) + tensor_desc->offset_bytes;
    OrtMemoryInfo* memory_info = nullptr;
    OrtStatus* status = impl.ort_api_.CreateMemoryInfo_V2("QnnHtpShared",
                                                          OrtMemoryInfoDeviceType_CPU,
                                                          qnn::kQualcommPciVendorId,
                                                          /*device_id*/ 0,
                                                          OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                                          /*alignment*/ 0,
                                                          OrtDeviceAllocator,
                                                          &memory_info);
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
