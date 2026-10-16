// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/qnn_data_transfer.h"

#include <algorithm>
#include <cstring>
#include <exception>

namespace onnxruntime {

namespace {

// QNN shared-memory allocators expose CPU-addressable memory with the
// Qualcomm vendor ID and HOST_ACCESSIBLE memory type. ORT still treats this
// as a distinct memory device, so it needs an EP data transfer to stage
// tensors between the default CPU allocator and QNN shared memory.
struct QnnHostAccessibleDataTransfer final : OrtDataTransferImpl {
  QnnHostAccessibleDataTransfer(const OrtApi& ort_api_in, const OrtEpApi& ep_api_in)
      : ort_api{ort_api_in}, ep_api{ep_api_in} {
    ort_version_supported = ORT_API_VERSION;
    Release = ReleaseImpl;
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
  }

  static bool IsQnnHostAccessibleMemory(const QnnHostAccessibleDataTransfer& impl,
                                        const OrtMemoryDevice* memory_device) noexcept {
    return memory_device != nullptr &&
           impl.ep_api.MemoryDevice_GetDeviceType(memory_device) == OrtMemoryInfoDeviceType_CPU &&
           impl.ep_api.MemoryDevice_GetVendorId(memory_device) == kQualcommPciVendorId &&
           impl.ep_api.MemoryDevice_GetMemoryType(memory_device) == OrtDeviceMemoryType_HOST_ACCESSIBLE;
  }

  static bool IsDefaultCpuMemory(const QnnHostAccessibleDataTransfer& impl,
                                 const OrtMemoryDevice* memory_device) noexcept {
    return memory_device != nullptr &&
           impl.ep_api.MemoryDevice_GetDeviceType(memory_device) == OrtMemoryInfoDeviceType_CPU &&
           impl.ep_api.MemoryDevice_GetMemoryType(memory_device) == OrtDeviceMemoryType_DEFAULT;
  }

  static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                       const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept {
    const auto& impl = *static_cast<const QnnHostAccessibleDataTransfer*>(this_ptr);
    const bool src_is_qnn_shared = IsQnnHostAccessibleMemory(impl, src_memory_device);
    const bool dst_is_qnn_shared = IsQnnHostAccessibleMemory(impl, dst_memory_device);

    // Own exactly the two routes required by the shared allocator. In
    // particular, leave shared-to-shared and unrelated CPU memory routes to
    // ORT's other data transfers.
    return (src_is_qnn_shared && IsDefaultCpuMemory(impl, dst_memory_device)) ||
           (dst_is_qnn_shared && IsDefaultCpuMemory(impl, src_memory_device));
  }

  static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                 const OrtValue** src_tensors,
                                                 OrtValue** dst_tensors,
                                                 OrtSyncStream** /*streams*/,
                                                 size_t num_tensors) noexcept {
    auto& impl = *static_cast<QnnHostAccessibleDataTransfer*>(this_ptr);

    if (num_tensors == 0) {
      return nullptr;
    }

    if (src_tensors == nullptr || dst_tensors == nullptr) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                       "QNN data transfer received null tensor arrays.");
    }

    for (size_t i = 0; i < num_tensors; ++i) {
      const OrtValue* src_tensor = src_tensors[i];
      OrtValue* dst_tensor = dst_tensors[i];
      if (src_tensor == nullptr || dst_tensor == nullptr) {
        return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "QNN data transfer received a null tensor.");
      }

      const OrtMemoryDevice* src_device = impl.ep_api.Value_GetMemoryDevice(src_tensor);
      const OrtMemoryDevice* dst_device = impl.ep_api.Value_GetMemoryDevice(dst_tensor);
      if (!CanCopyImpl(this_ptr, src_device, dst_device)) {
        return impl.ort_api.CreateStatus(
            ORT_INVALID_ARGUMENT,
            "QNN data transfer only supports copies between default CPU and QNN host-accessible memory.");
      }

      ONNXTensorElementDataType src_element_type;
      ONNXTensorElementDataType dst_element_type;
      const int64_t* src_shape = nullptr;
      const int64_t* dst_shape = nullptr;
      size_t src_rank = 0;
      size_t dst_rank = 0;
      RETURN_IF_NOT_NULL(impl.ort_api.GetTensorElementTypeAndShapeDataReference(
          src_tensor, &src_element_type, &src_shape, &src_rank));
      RETURN_IF_NOT_NULL(impl.ort_api.GetTensorElementTypeAndShapeDataReference(
          dst_tensor, &dst_element_type, &dst_shape, &dst_rank));

      if (src_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING ||
          dst_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "QNN data transfer does not support string tensors.");
      }

      if (src_element_type != dst_element_type || src_rank != dst_rank ||
          (src_rank != 0 && !std::equal(src_shape, src_shape + src_rank, dst_shape))) {
        return impl.ort_api.CreateStatus(
            ORT_INVALID_ARGUMENT,
            "QNN data transfer source and destination types or shapes differ.");
      }

      size_t src_bytes = 0;
      size_t dst_bytes = 0;
      RETURN_IF_NOT_NULL(impl.ort_api.GetTensorSizeInBytes(src_tensor, &src_bytes));
      RETURN_IF_NOT_NULL(impl.ort_api.GetTensorSizeInBytes(dst_tensor, &dst_bytes));
      if (src_bytes != dst_bytes) {
        return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "QNN data transfer source and destination sizes differ.");
      }

      const void* src_data = nullptr;
      void* dst_data = nullptr;
      RETURN_IF_NOT_NULL(impl.ort_api.GetTensorData(src_tensor, &src_data));
      RETURN_IF_NOT_NULL(impl.ort_api.GetTensorMutableData(dst_tensor, &dst_data));

      if (src_bytes != 0 && (src_data == nullptr || dst_data == nullptr)) {
        return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "QNN data transfer received null tensor data.");
      }

      if (src_data != dst_data && src_bytes != 0) {
        std::memcpy(dst_data, src_data, src_bytes);
      }
    }

    return nullptr;
  }

  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
    delete static_cast<QnnHostAccessibleDataTransfer*>(this_ptr);
  }

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

}  // namespace

OrtStatus* CreateQnnHostAccessibleDataTransfer(const OrtApi& ort_api,
                                               const OrtEpApi& ep_api,
                                               OrtDataTransferImpl** data_transfer) noexcept {
  *data_transfer = nullptr;

  try {
    *data_transfer = new QnnHostAccessibleDataTransfer(ort_api, ep_api);
  } catch (const std::exception& e) {
    return ort_api.CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return ort_api.CreateStatus(ORT_FAIL, "Failed to create QNN data transfer.");
  }

  return nullptr;
}

}  // namespace onnxruntime
