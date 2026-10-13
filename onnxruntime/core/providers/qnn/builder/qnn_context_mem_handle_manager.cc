// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_context_mem_handle_manager.h"

#include "HTP/QnnHtpMem.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/qnn_external_resource_importer.h"

namespace onnxruntime::qnn {

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
    if (const auto mem_handles_it = mem_handles_.find(memory_address);
        mem_handles_it != mem_handles_.end()) {
      const auto& mem_handle_record = mem_handles_it->second;

      // check that actual tensor size is less than or equal to registered tensor size
      RETURN_IF_NOT(qnn_tensor_data_size <= mem_handle_record.registered_tensor_data_size,
                    ("Actual tensor data size (" + std::to_string(qnn_tensor_data_size) +
                     ") is larger than registered tensor data size (" +
                     std::to_string(mem_handle_record.registered_tensor_data_size) + ").")
                        .c_str());

      qnn_mem_handle = mem_handle_record.mem_handle.get();
      did_register = false;
      return Ort::Status();
    }

    // register a new mem handle
    Qnn_MemDescriptor_t mem_descriptor = QNN_MEM_DESCRIPTOR_INIT;
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

        QnnMemHtp_Descriptor_t htp_mem_descriptor{};
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

    std::ostringstream oss3;
    oss3 << "Registered QNN mem handle. mem_handle: " << raw_mem_handle;
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, oss3.str().c_str());

    // NOTE: Must use the default ORT logger inside this lambda. Don't capture logger because it may be deleted
    // by the time we need to unregister all memory handles. This happens when logger is a session logger:
    //   ~InferenceSession() -> ~Logger() -> ~QnnExecutionProvider() -> ~QnnBackendManager() ->
    //   ~QnnContextMemHandleManager() -> unregister_mem_handle() segfault
    const auto unregister_mem_handle = [&qnn_interface = this->qnn_interface_](Qnn_MemHandle_t raw_mem_handle) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE, "Unregistering QNN mem handle.");

      const auto unregister_result = qnn_interface.memDeRegister(&raw_mem_handle, 1);
      if (unregister_result != QNN_SUCCESS) {
        ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                    ORT_LOGGING_LEVEL_ERROR,
                    ("qnn_interface.memDeRegister() failed: " +
                     utils::GetVerboseQnnErrorMessage(qnn_interface, unregister_result))
                        .c_str());
      }
    };

    UniqueQnnMemHandle mem_handle(raw_mem_handle, unregister_mem_handle);
    MemHandleRecord mem_handle_record{qnn_tensor_data_size, std::move(mem_handle)};
    mem_handles_.emplace(memory_address, std::move(mem_handle_record));

    qnn_mem_handle = raw_mem_handle;
    did_register = true;
    return Ort::Status();
  }
}

Ort::Status QnnContextMemHandleManager::Unregister(void* memory_address) {
  std::scoped_lock g{mem_handles_mutex_};

  auto mem_handles_it = mem_handles_.find(memory_address);
  RETURN_IF_NOT(mem_handles_it != mem_handles_.end(), "No mem handle found for address.");

  mem_handles_.erase(mem_handles_it);

  return Ort::Status();
}

void QnnContextMemHandleManager::Clear() {
  std::scoped_lock g{mem_handles_mutex_};
  mem_handles_.clear();
}

}  // namespace onnxruntime::qnn
