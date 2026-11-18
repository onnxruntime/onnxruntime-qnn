// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Qualcomm's PCI vendor ID. This identifies QNN memory devices and is distinct
// from QnnEpFactory::vendor_id_, which is the ACPI "QCOM" code.
inline constexpr uint32_t kQualcommPciVendorId = 0x5143;

// QNN shared-memory allocators expose CPU-addressable memory with the
// Qualcomm vendor ID and HOST_ACCESSIBLE memory type. ORT still treats this
// as a distinct memory device, so it needs an EP data transfer to stage
// tensors between the default CPU allocator and QNN shared memory.
class QnnHostAccessibleDataTransfer final : public OrtDataTransferImpl {
 public:
  QnnHostAccessibleDataTransfer(const OrtApi& ort_api_in, const OrtEpApi& ep_api_in);

 private:
  static bool IsQnnHostAccessibleMemory(const QnnHostAccessibleDataTransfer& impl,
                                        const OrtMemoryDevice* memory_device) noexcept;
  static bool IsDefaultCpuMemory(const QnnHostAccessibleDataTransfer& impl,
                                 const OrtMemoryDevice* memory_device) noexcept;
  static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                       const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept;
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                 const OrtValue** src_tensors,
                                                 OrtValue** dst_tensors,
                                                 OrtSyncStream** streams,
                                                 size_t num_tensors) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

}  // namespace qnn
}  // namespace onnxruntime
