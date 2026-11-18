// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/qnn_data_transfer.h"

#include <cstring>

namespace onnxruntime {
namespace qnn {

QnnHostAccessibleDataTransfer::QnnHostAccessibleDataTransfer(const OrtApi& ort_api_in,
                                                             const OrtEpApi& ep_api_in)
    : ort_api{ort_api_in}, ep_api{ep_api_in} {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
}

bool QnnHostAccessibleDataTransfer::IsQnnHostAccessibleMemory(
    const QnnHostAccessibleDataTransfer& impl,
    const OrtMemoryDevice* memory_device) noexcept {
  return memory_device != nullptr &&
         impl.ep_api.MemoryDevice_GetDeviceType(memory_device) == OrtMemoryInfoDeviceType_CPU &&
         impl.ep_api.MemoryDevice_GetVendorId(memory_device) == kQualcommPciVendorId &&
         impl.ep_api.MemoryDevice_GetMemoryType(memory_device) == OrtDeviceMemoryType_HOST_ACCESSIBLE;
}

bool QnnHostAccessibleDataTransfer::IsDefaultCpuMemory(
    const QnnHostAccessibleDataTransfer& impl,
    const OrtMemoryDevice* memory_device) noexcept {
  return memory_device != nullptr &&
         impl.ep_api.MemoryDevice_GetDeviceType(memory_device) == OrtMemoryInfoDeviceType_CPU &&
         impl.ep_api.MemoryDevice_GetMemoryType(memory_device) == OrtDeviceMemoryType_DEFAULT;
}

bool ORT_API_CALL QnnHostAccessibleDataTransfer::CanCopyImpl(
    const OrtDataTransferImpl* this_ptr,
    const OrtMemoryDevice* src_memory_device,
    const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto& impl = *static_cast<const QnnHostAccessibleDataTransfer*>(this_ptr);
  const bool src_is_qnn_shared = IsQnnHostAccessibleMemory(impl, src_memory_device);
  const bool dst_is_qnn_shared = IsQnnHostAccessibleMemory(impl, dst_memory_device);

  return (src_is_qnn_shared && IsDefaultCpuMemory(impl, dst_memory_device)) ||
         (dst_is_qnn_shared && IsDefaultCpuMemory(impl, src_memory_device));
}

OrtStatus* ORT_API_CALL QnnHostAccessibleDataTransfer::CopyTensorsImpl(
    OrtDataTransferImpl* this_ptr,
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

    ONNXTensorElementDataType src_element_type;
    ONNXTensorElementDataType dst_element_type;
    const int64_t* shape = nullptr;
    size_t rank = 0;
    RETURN_IF_NOT_NULL(impl.ort_api.GetTensorElementTypeAndShapeDataReference(
        src_tensor, &src_element_type, &shape, &rank));
    RETURN_IF_NOT_NULL(impl.ort_api.GetTensorElementTypeAndShapeDataReference(
        dst_tensor, &dst_element_type, &shape, &rank));

    if (src_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING ||
        dst_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                       "QNN data transfer does not support string tensors.");
    }

    if (src_element_type != dst_element_type) {
      return impl.ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT,
          "QNN data transfer source and destination types differ.");
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

void ORT_API_CALL QnnHostAccessibleDataTransfer::ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
  delete static_cast<QnnHostAccessibleDataTransfer*>(this_ptr);
}

}  // namespace qnn
}  // namespace onnxruntime
