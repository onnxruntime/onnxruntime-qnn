// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32

#include "core/providers/qnn/qnn_external_resource_importer.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include <new>
#include <sstream>
#include <string>

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
  QNN_EP_API_IMPL_BEGIN
  if (handle == nullptr) return;
  QnnExternalMemoryHandle* derived = static_cast<QnnExternalMemoryHandle*>(handle);

  // Run cleanup callbacks (e.g., QNN mem handle deregistration).
  for (const auto& cleanup : derived->cleanup_callbacks_) {
    cleanup(derived);
  }

  std::unique_lock write_lock{QnnExternalResourceImporterImpl::mutex_mem_handle_registry_};
  QnnExternalResourceImporterImpl::mem_handle_registry_.erase(derived);

  delete derived;
  QNN_EP_API_IMPL_END_VOID
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
  QNN_EP_API_IMPL_BEGIN
  if (handle == nullptr) return;
  auto* derived = static_cast<QnnExternalSemaphoreHandle*>(handle);
  delete derived;
  QNN_EP_API_IMPL_END_VOID
}

// ============================================================================
// QnnExternalResourceImporterImpl Implementation
// ============================================================================

InlinedHashSet<QnnExternalMemoryHandle*> QnnExternalResourceImporterImpl::mem_handle_registry_;
std::shared_mutex QnnExternalResourceImporterImpl::mutex_mem_handle_registry_;

QnnExternalResourceImporterImpl::QnnExternalResourceImporterImpl(
    int device_id, const OrtApi& ort_api_in) : device_id_{device_id},
                                               ort_api_{ort_api_in},
                                               ep_api_{*ort_api_in.GetEpApi()},
                                               d3d12_device_{nullptr} {
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

  HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d12_device_));
  if (FAILED(hr) || d3d12_device_ == nullptr) {
    throw std::runtime_error("D3D12CreateDevice failed.");
  }
}

bool ORT_API_CALL QnnExternalResourceImporterImpl::CanImportMemoryImpl(
    _In_ const OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalMemoryHandleType handle_type) noexcept {
  // Supports D3D12 resource handles only, and not heap handles
  return handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
}

OrtStatus* ORT_API_CALL QnnExternalResourceImporterImpl::ImportMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalMemoryDescriptor* desc,
    _Outptr_ OrtExternalMemoryHandle** out_handle) noexcept {
  QNN_EP_API_IMPL_BEGIN
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

  switch (desc->handle_type) {
    case ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: {
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
  QNN_EP_API_IMPL_END
}

void ORT_API_CALL QnnExternalResourceImporterImpl::ReleaseMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalMemoryHandle* handle) noexcept {
  QNN_EP_API_IMPL_BEGIN
  // The handle has a Release callback that does the actual cleanup
  if (handle) {
    handle->Release(handle);
  }
  QNN_EP_API_IMPL_END_VOID
}

OrtStatus* ORT_API_CALL QnnExternalResourceImporterImpl::CreateTensorFromMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalMemoryHandle* mem_handle,
    _In_ const OrtExternalTensorDescriptor* tensor_desc,
    _Outptr_ OrtValue** out_tensor) noexcept {
  QNN_EP_API_IMPL_BEGIN
  auto& impl = *static_cast<QnnExternalResourceImporterImpl*>(this_ptr);

  if (mem_handle == nullptr || tensor_desc == nullptr || out_tensor == nullptr) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                      "mem_handle, tensor_desc, and out_tensor cannot be nullptr");
  }

  *out_tensor = nullptr;

  if (tensor_desc->offset_bytes != 0) {
    return impl.ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Not supported create tensor with non-zero offset");
  }

  const auto* qnn_handle = static_cast<const QnnExternalMemoryHandle*>(mem_handle);

  size_t tensor_size_bytes = 0;
  tensor_size_bytes = qnn::utils::GetOnnxTensorDataSizeInBytes(gsl::span{tensor_desc->shape, tensor_desc->rank}, tensor_desc->element_type);

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
  QNN_EP_API_IMPL_END
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
  QNN_EP_API_IMPL_BEGIN
  // The handle has a Release callback that does the actual cleanup
  if (handle) {
    handle->Release(handle);
  }
  QNN_EP_API_IMPL_END_VOID
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
  QNN_EP_API_IMPL_BEGIN
  if (this_ptr == nullptr) {
    return;
  }
  delete static_cast<QnnExternalResourceImporterImpl*>(this_ptr);
  QNN_EP_API_IMPL_END_VOID
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
  QNN_EP_API_IMPL_BEGIN
  if (this_ptr == nullptr) {
    return;
  }
  delete static_cast<QnnSyncStreamImpl*>(this_ptr);
  QNN_EP_API_IMPL_END_VOID
}

}  // namespace onnxruntime

#endif  // _WIN32
