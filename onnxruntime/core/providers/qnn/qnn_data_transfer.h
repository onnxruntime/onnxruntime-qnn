// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {

// Qualcomm's PCI vendor ID. This identifies QNN memory devices and is distinct
// from QnnEpFactory::vendor_id_, which is the ACPI "QCOM" code.
inline constexpr uint32_t kQualcommPciVendorId = 0x5143;

OrtStatus* CreateQnnHostAccessibleDataTransfer(const OrtApi& ort_api,
                                               const OrtEpApi& ep_api,
                                               OrtDataTransferImpl** data_transfer) noexcept;

}  // namespace onnxruntime
