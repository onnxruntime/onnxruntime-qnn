// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// Returns true if `node_unit` is a standalone DequantizeLinear or QuantizeLinear
// whose only data input is effectively a compile-time constant (a real graph
// initializer or a previously folded constant). Such nodes can be evaluated at
// build time and emitted as QNN STATIC tensors instead of producing a runtime
// QNN op + APP_WRITE input.
bool CanFoldConstantQdq(const QnnModelWrapper& qnn_model_wrapper,
                        const OrtNodeUnit& node_unit);

// Constant-fold a standalone DequantizeLinear / QuantizeLinear node whose input
// is effectively constant. On success, the node's output is registered as a
// QNN_TENSOR_TYPE_STATIC tensor wrapper and is marked as a folded constant so
// that downstream Q/DQ hops can continue folding. The Q/DQ op itself is not
// added to the QNN graph.
//
// Caller MUST first verify with `CanFoldConstantQdq`. Behavior on a non-fold
// candidate is an error status.
Ort::Status TryFoldConstantQDQ(QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit) ORT_MUST_USE_RESULT;

}  // namespace qnn
}  // namespace onnxruntime
