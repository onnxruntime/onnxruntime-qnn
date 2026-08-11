// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// True only for an initializer-backed standalone per-channel DQ. QNN cannot represent that
// node directly, so this narrow compatibility case remains foldable. All representable Q/DQ
// nodes follow the pre-#339 path and stay as QNN operations.
bool CanFoldInitializerPerChannelDequantize(const QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit);

// Fold the initializer-backed per-channel DQ and register its output as a STATIC tensor.
// Caller MUST first verify with `CanFoldInitializerPerChannelDequantize`.
Ort::Status FoldInitializerPerChannelDequantize(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit) ORT_MUST_USE_RESULT;

// Reads the raw bytes of a real initializer or a previously-folded STATIC tensor.
Ort::Status GetEffectivelyConstantTensorBytes(QnnModelWrapper& qnn_model_wrapper,
                                              const std::string& tensor_name,
                                              /*out*/ std::vector<uint8_t>& bytes) ORT_MUST_USE_RESULT;

}  // namespace qnn
}  // namespace onnxruntime
