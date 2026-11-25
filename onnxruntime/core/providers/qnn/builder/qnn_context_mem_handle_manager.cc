// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_context_mem_handle_manager.h"

#include <cstddef>

#include "HTP/QnnHtpMem.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/qnn_external_resource_importer.h"

namespace onnxruntime::qnn {

QnnMemHandleRegistration::QnnMemHandleRegistration(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                   Qnn_MemHandle_t mem_handle,
                                                   size_t registered_tensor_data_size)
    : qnn_interface_{qnn_interface},
      registered_tensor_data_size_{registered_tensor_data_size},
      mem_handle_{mem_handle} {
}

void QnnMemHandleRegistration::Deregister() noexcept {
  std::scoped_lock lock{mutex_};
  if (mem_handle_ == nullptr) {
    return;
  }

  // NOTE: Use the default ORT logger. A session logger may already be gone when this runs:
  //   ~InferenceSession() -> ~Logger() -> ~QnnExecutionProvider() -> ~QnnBackendManager() ->
  //   ~QnnContextMemHandleManager() -> Deregister()
  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE, "Unregistering QNN mem handle.");
  const auto unregister_result = qnn_interface_.memDeRegister(&mem_handle_, 1);
  if (unregister_result != QNN_SUCCESS) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                ORT_LOGGING_LEVEL_ERROR,
                ("qnn_interface.memDeRegister() failed: " +
                 utils::GetVerboseQnnErrorMessage(qnn_interface_, unregister_result))
                    .c_str());
  }
  // QnnMem_deRegister invalidates the handle even on failure.
  mem_handle_ = nullptr;
}

QnnMemHandleRegistration::~QnnMemHandleRegistration() {
  // Normally already deregistered by the record or the clean up hook; this only matters if adopting the handle into
  // a record failed.
  Deregister();
}

bool QnnMemHandleRegistration::IsRegistered() const {
  std::scoped_lock lock{mutex_};
  return mem_handle_ != nullptr;
}

Qnn_MemHandle_t QnnMemHandleRegistration::Handle() const {
  std::scoped_lock lock{mutex_};
  return mem_handle_;
}

QnnContextMemHandleManager::QnnContextMemHandleManager(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                       Qnn_ContextHandle_t context,
                                                       QnnBackendType qnn_backend_type,
                                                       QnnAllocatorType qnn_allocator_type)
    : qnn_interface_{qnn_interface},
      context_{context},
      qnn_backend_type_{qnn_backend_type},
      qnn_allocator_type_{qnn_allocator_type} {
}

QnnContextMemHandleManager::~QnnContextMemHandleManager() {
  Clear();
}

Ort::Status QnnContextMemHandleManager::GetOrRegister(void* memory_address,
                                                      const Qnn_Tensor_t& qnn_tensor,
                                                      Qnn_MemHandle_t& qnn_mem_handle,
                                                      bool& did_register,
                                                      const Ort::Logger& logger) {
  const auto qnn_tensor_rank = GetQnnTensorRank(qnn_tensor);
  auto* const qnn_tensor_dims = GetQnnTensorDims(qnn_tensor);
  const auto qnn_tensor_data_type = GetQnnTensorDataType(qnn_tensor);

  const size_t qnn_tensor_data_size =
      utils::GetQnnTensorDataSizeInBytes(gsl::span{qnn_tensor_dims, size_t{qnn_tensor_rank}}, qnn_tensor_data_type);

  {
    std::scoped_lock g{mem_handles_mutex_};

    // find existing mem handle
    if (auto mem_handles_it = mem_handles_.find(memory_address);
        mem_handles_it != mem_handles_.end() && !mem_handles_it->second.registration->IsRegistered()) {
      // The backing memory's clean up hook already deregistered it (e.g. the import was released). Register again
      // below for whatever memory is now at this address.
      mem_handles_.erase(mem_handles_it);
    }
    if (const auto mem_handles_it = mem_handles_.find(memory_address);
        mem_handles_it != mem_handles_.end()) {
      const auto& registration = *mem_handles_it->second.registration;

      // Re-bind policy: the first tensor bound at an address fixes its registration in this context. A later tensor
      // at the same address reuses that registration if it is no larger, whatever its shape or data type, and is
      // rejected otherwise (the existing handle may still be bound by other tensors, so it is not replaced).
      //
      // QnnMem_register takes the first tensor's shape and data type, but HTP binds a QNN_HTP_MEM_SHARED_BUFFER handle
      // by {fd, offset}: executing it with a tensor of another shape, data type or a smaller size works on HTP, for
      // graph inputs and outputs (host_allocation_import_registration_reused_across_shapes_and_types). Byte size is
      // the bound checked here.
      RETURN_IF_NOT(qnn_tensor_data_size <= registration.RegisteredTensorDataSize(),
                    ("Actual tensor data size (" + std::to_string(qnn_tensor_data_size) +
                     ") is larger than registered tensor data size (" +
                     std::to_string(registration.RegisteredTensorDataSize()) +
                     "). A tensor bound at this address must not be larger than the first tensor bound there; bind it "
                     "at a different address, or release and re-import the buffer.")
                        .c_str());

      qnn_mem_handle = registration.Handle();
      did_register = false;
      return Ort::Status();
    }

    // register a new mem handle
    Qnn_MemDescriptor_t mem_descriptor = QNN_MEM_DESCRIPTOR_INIT;
    // customInfo must remain valid until memRegister returns.
    QnnMemHtp_Descriptor_t htp_mem_descriptor{};
    mem_descriptor.memShape.dimSize = qnn_tensor_dims;
    mem_descriptor.memShape.numDim = qnn_tensor_rank;
    mem_descriptor.memShape.shapeConfig = nullptr;
    mem_descriptor.dataType = qnn_tensor_data_type;
#ifdef _WIN32
    if (QnnExternalResourceImporterImpl::FindImportMemory(memory_address)) {
      auto imp_mem_handle = static_cast<QnnExternalMemoryHandle*>(memory_address);

      mem_descriptor.memType = QNN_MEM_TYPE_DX12;
      mem_descriptor.dx12BufInfo.resourceHandle =
          static_cast<Qnn_Dx12ResourceHandle_t>(imp_mem_handle->d3d12_resource_.Get());

      std::ostringstream oss1;
      oss1 << "Registering QNN mem handle for context: " << context_
           << ", Imported memory (handle: " << memory_address
           << ", resource: " << imp_mem_handle->d3d12_resource_
           << ", offset: " << 0
           << ")";
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, oss1.str().c_str());
    } else if (const auto imported_host_memory = QnnExternalResourceImporterImpl::FindImportedHostMemory(memory_address);
               imported_host_memory.has_value()) {
      // App-owned host allocation already registered with FastRPC at import time.
      const size_t offset = static_cast<size_t>(static_cast<const std::byte*>(memory_address) -
                                                static_cast<const std::byte*>(imported_host_memory->base_address));
      RETURN_IF_NOT(qnn_tensor_data_size <= imported_host_memory->size - offset,
                    "Tensor exceeds its imported host allocation.");

      mem_descriptor.memType = QNN_MEM_TYPE_CUSTOM;
      htp_mem_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;
      htp_mem_descriptor.size = imported_host_memory->size;
      htp_mem_descriptor.sharedBufferConfig.fd = imported_host_memory->fd;
      htp_mem_descriptor.sharedBufferConfig.offset = offset;
      mem_descriptor.customInfo = &htp_mem_descriptor;

      std::ostringstream oss1;
      oss1 << "Registering QNN mem handle for context: " << context_
           << ", imported host memory (address: " << memory_address
           << ", offset: " << offset
           << ", fd: " << imported_host_memory->fd
           << ")";
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, oss1.str().c_str());
    } else if (IsDx12SharedMemoryAllocator(qnn_allocator_type_)) {
      // DX12 path: QNN_MEM_TYPE_DX12 with Qnn_MemDx12BufInfo_t
      Dx12SharedMemoryAllocator::Dx12AllocationInfo dx12_info{};
      RETURN_IF_ERROR(Dx12SharedMemoryAllocator::GetAllocationDx12Info(memory_address, dx12_info));

      mem_descriptor.memType = QNN_MEM_TYPE_DX12;
      mem_descriptor.dx12BufInfo.resourceHandle =
          static_cast<Qnn_Dx12ResourceHandle_t>(dx12_info.resource);

      std::ostringstream oss1;
      oss1 << "Registering QNN mem handle for context: " << context_
           << ", DX12 shared memory (address: " << memory_address
           << ", resource: " << dx12_info.resource
           << ", offset: " << dx12_info.offset
           << ")";
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, oss1.str().c_str());
    } else
#endif  // _WIN32
      if (IsHtpSharedMemoryAllocator(qnn_allocator_type_)) {
        mem_descriptor.memType = QNN_MEM_TYPE_CUSTOM;

        HtpSharedMemoryAllocator::SharedMemoryInfo shared_memory_info{};
        RETURN_IF_ERROR(HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfo(memory_address, shared_memory_info));

        htp_mem_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;
        htp_mem_descriptor.size = shared_memory_info.total_size;
        htp_mem_descriptor.sharedBufferConfig.fd = shared_memory_info.fd;
        htp_mem_descriptor.sharedBufferConfig.offset = shared_memory_info.offset;

        mem_descriptor.customInfo = &htp_mem_descriptor;

        std::ostringstream oss1;
        oss1 << "Registering QNN mem handle for context: " << context_
             << ", shared memory (address: " << memory_address
             << ", offset: " << shared_memory_info.offset
             << ", fd: " << shared_memory_info.fd
             << ")";
        ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, oss1.str().c_str());
      } else {
        return MAKE_EP_FAIL("No HTP or DX12 allocation found for shared memory address.");
      }

    std::ostringstream oss2;
    oss2 << "Registering QNN mem handle. context: " << context_;
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, oss2.str().c_str());

    Qnn_MemHandle_t raw_mem_handle{};
    const auto register_result = qnn_interface_.memRegister(context_, &mem_descriptor, 1, &raw_mem_handle);
#ifdef _WIN32
    if (IsGpuBackend(qnn_backend_type_) &&
        IsDx12SharedMemoryAllocator(qnn_allocator_type_) &&
        register_result == QNN_MEM_ERROR_MAPPING) {
      ORT_CXX_LOG(logger,
                  ORT_LOGGING_LEVEL_ERROR,
                  "QnnMem_register failed with QNN_MEM_ERROR_MAPPING when using the DX12 shared memory allocator with the GPU"
                  " backend on Windows. This is likely due to outdated graphics drivers on the device. Please try installing"
                  " new drivers from https://softwarecenter.qualcomm.com/catalog/item/Windows_Graphics_Driver.");
    }
#else
    std::ignore = qnn_backend_type_;
#endif
    RETURN_IF_NOT(register_result == QNN_SUCCESS,
                  ("qnn_interface.memRegister() failed: " +
                   utils::GetVerboseQnnErrorMessage(qnn_interface_, register_result))
                      .c_str());

    // Adopt the handle at once, so any later failure (including an allocation failure) deregisters it.
    std::shared_ptr<QnnMemHandleRegistration> registration;
    try {
      registration = std::make_shared<QnnMemHandleRegistration>(qnn_interface_, raw_mem_handle, qnn_tensor_data_size);
    } catch (...) {
      QnnMemHandleRegistration{qnn_interface_, raw_mem_handle, qnn_tensor_data_size}.Deregister();
      return MAKE_EP_FAIL("Failed to allocate a QNN mem handle registration.");
    }

    try {
      std::ostringstream oss3;
      oss3 << "Registered QNN mem handle. mem_handle: " << raw_mem_handle;
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, oss3.str().c_str());

      // Retire records whose memory has gone away before adding one, so a long-lived context that binds many
      // distinct, short-lived buffers keeps no more records than it has live registrations plus those retired since
      // its last registration.
      RetireDeregisteredLocked();
      mem_handles_.emplace(memory_address, MemHandleRecord{registration});
    } catch (...) {
      // `registration` deregisters the handle when it goes out of scope.
      return MAKE_EP_FAIL("Failed to record a QNN mem handle registration.");
    }

    qnn_mem_handle = raw_mem_handle;
    did_register = true;
    return Ort::Status();
  }
}

void QnnContextMemHandleManager::RetireDeregisteredLocked() {
  for (auto it = mem_handles_.begin(); it != mem_handles_.end();) {
    if (!it->second.registration->IsRegistered()) {
      mem_handles_.erase(it++);
    } else {
      ++it;
    }
  }
}

std::shared_ptr<QnnMemHandleRegistration> QnnContextMemHandleManager::GetRegistration(const void* memory_address) {
  std::scoped_lock g{mem_handles_mutex_};
  const auto mem_handles_it = mem_handles_.find(memory_address);
  return mem_handles_it != mem_handles_.end() ? mem_handles_it->second.registration : nullptr;
}

Ort::Status QnnContextMemHandleManager::Unregister(void* memory_address) {
  std::scoped_lock g{mem_handles_mutex_};

  auto mem_handles_it = mem_handles_.find(memory_address);
  RETURN_IF_NOT(mem_handles_it != mem_handles_.end(), "No mem handle found for address.");

  mem_handles_.erase(mem_handles_it);

  return Ort::Status();
}

void QnnContextMemHandleManager::Discard(const void* memory_address) noexcept {
  std::scoped_lock g{mem_handles_mutex_};
  if (const auto mem_handles_it = mem_handles_.find(memory_address); mem_handles_it != mem_handles_.end()) {
    mem_handles_.erase(mem_handles_it);
  }
}

void QnnContextMemHandleManager::Clear() {
  std::scoped_lock g{mem_handles_mutex_};
  mem_handles_.clear();
}

}  // namespace onnxruntime::qnn
