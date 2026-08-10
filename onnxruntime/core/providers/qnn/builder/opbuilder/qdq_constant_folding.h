// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// True if `node_unit` is a standalone Q/DQ with an effectively-constant input
// (real initializer or previously-folded tensor) eligible for compile-time folding.
bool CanFoldConstantQdq(const QnnModelWrapper& qnn_model_wrapper,
                        const OrtNodeUnit& node_unit);

// Fold the Q/DQ statically and register its output as a STATIC tensor. Caller
// MUST first verify with `CanFoldConstantQdq`.
Ort::Status TryFoldConstantQDQ(QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit) ORT_MUST_USE_RESULT;

// Reads the raw bytes of a real initializer or a previously-folded STATIC tensor.
Ort::Status GetEffectivelyConstantTensorBytes(QnnModelWrapper& qnn_model_wrapper,
                                              const std::string& tensor_name,
                                              /*out*/ std::vector<uint8_t>& bytes) ORT_MUST_USE_RESULT;

}  // namespace qnn
}  // namespace onnxruntime
