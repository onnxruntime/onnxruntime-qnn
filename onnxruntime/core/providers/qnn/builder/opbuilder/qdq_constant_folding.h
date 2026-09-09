// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// True if `node_unit` is a standalone Q/DQ with an effectively-constant input
// (real initializer or previously-folded tensor) eligible for compile-time folding.
bool CanFoldConstantQdq(const QnnModelWrapper& qnn_model_wrapper,
                        const OrtNodeUnit& node_unit);

// Budget for one folded DequantizeLinear output, in FP32 bytes. Folding converts
// compact quantized weights into FP32 STATIC tensors stored in the DLC; beyond
// this size the DLC cost outweighs the saved runtime op.
inline constexpr size_t kQdqFoldMaxFp32Bytes = 1024 * 1024;  // 1 MiB

// True when a constant-DQ fold must be skipped in favor of a runtime QNN
// Dequantize: the folded FP32 blob exceeds the DLC budget. Callers must first
// establish via CanSubstituteRuntimeDequantize that a runtime Dequantize is a
// faithful substitute (per-channel and sub-byte inputs always fold).
// Pure function of its inputs: covered by unit tests.
bool ShouldSkipConstantDQFold(size_t num_elems);

// Fold the Q/DQ statically and register its output as a STATIC tensor. Caller
// MUST first verify with `CanFoldConstantQdq`.
Ort::Status TryFoldConstantQDQ(QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit) ORT_MUST_USE_RESULT;

// Reads the bytes of a real initializer or a previously-folded STATIC tensor as plain
// two's-complement integers, one element per byte for sub-byte types. Unlike
// QnnModelWrapper::UnpackInitializerData(), sub-byte elements are sign-extended, not left masked.
Ort::Status GetEffectivelyConstantTensorBytes(QnnModelWrapper& qnn_model_wrapper,
                                              const std::string& tensor_name,
                                              /*out*/ std::vector<uint8_t>& bytes) ORT_MUST_USE_RESULT;

}  // namespace qnn
}  // namespace onnxruntime
