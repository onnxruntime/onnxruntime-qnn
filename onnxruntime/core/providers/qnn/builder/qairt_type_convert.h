// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef USE_QAIRT_API

#include <string>
#include <vector>

#include "QairtCpp/QairtApi.hpp"
#include "QairtCpp/QairtOpConfig.hpp"
#include "QairtCpp/QairtTensor.hpp"
#include "QnnTypes.h"

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace qairt_convert {

// Translation helpers between the QNN C structs the EP builds (Qnn_Tensor_t,
// Qnn_OpConfig_t) and the QAIRT C++ objects the QAIRT API consumes.
//
// The EP's ~50 op builders and ~36 fusions all emit QnnTensorWrapper /
// QnnOpConfigWrapper, which own Qnn_Tensor_t / Qnn_OpConfig_t. Rather than fork
// that entire layer, QairtGraphEmitter converts at the IGraphEmitter boundary
// using these helpers. That keeps every op builder untouched.
//
// Buffer ownership: qairt::ClientBuffer::setData() stores the pointer without
// copying, so the Qnn_Tensor_t's clientBuf.data must outlive the qairt::Tensor.
// In practice both live inside the same emit/execute call, and the underlying
// storage is owned by QnnTensorWrapper (compose) or ORT (execute).

// Maps Qnn_DataType_t to qairt::DataType. The two enums share numeric values
// (both derive from the same backend ABI), but cast through explicitly so a
// future divergence surfaces here rather than as silent data corruption.
inline qairt::DataType ToQairtDataType(Qnn_DataType_t dt) {
  return static_cast<qairt::DataType>(dt);
}

// Copies the quantization encoding from a Qnn_QuantizeParams_t onto a
// qairt::QuantizeParams. Only the encodings the EP actually emits are handled
// (verified against the QNN_QUANTIZATION_ENCODING_* values used across
// onnxruntime/core/providers/qnn/): SCALE_OFFSET, AXIS_SCALE_OFFSET,
// BW_SCALE_OFFSET, BW_AXIS_SCALE_OFFSET, BLOCK, BW_FLOAT_BLOCK,
// BLOCKWISE_EXPANSION, and UNDEFINED.
//
// Returns an error for anything else so an unhandled encoding fails loudly at
// graph-emit time instead of producing a silently mis-quantized graph.
Ort::Status ApplyQuantizeParams(const Qnn_QuantizeParams_t& qnn_qp,
                                const qairt::Api& api,
                                qairt::Tensor& out);

// Builds a qairt::Tensor mirroring `qnn_tensor`. Does not copy tensor data --
// the client buffer points at the same memory (see ownership note above).
Ort::Status FromQnnTensor(const Qnn_Tensor_t& qnn_tensor,
                          const qairt::Api& api,
                          qairt::Tensor& out);

// Builds a qairt::OpConfig mirroring `qnn_op`, converting its params and its
// input/output tensors.
Ort::Status FromQnnOpConfig(const Qnn_OpConfig_t& qnn_op,
                            const qairt::Api& api,
                            qairt::OpConfig& out);

}  // namespace qairt_convert
}  // namespace qnn
}  // namespace onnxruntime

#endif  // USE_QAIRT_API
