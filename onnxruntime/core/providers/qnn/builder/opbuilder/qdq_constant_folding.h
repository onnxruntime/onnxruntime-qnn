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

// True when the constant DQ on `input_def` may be left as a runtime QNN Dequantize instead
// of being folded, i.e. when the runtime op is a faithful substitute for the folded constant.
// False for per-channel and INT4/INT2 inputs, which must always fold; see the definition for
// why. Gates ShouldSkipConstantDQFold, so a "must fold" input ignores the size budget.
Ort::Status CanSubstituteRuntimeDequantize(const QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnitIODef& input_def,
                                           /*out*/ bool& can_substitute) ORT_MUST_USE_RESULT;

// True when a foldable constant DQ exceeds the DLC budget and is better left as a runtime
// QNN Dequantize. Only meaningful for inputs CanSubstituteRuntimeDequantize() admits.
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
