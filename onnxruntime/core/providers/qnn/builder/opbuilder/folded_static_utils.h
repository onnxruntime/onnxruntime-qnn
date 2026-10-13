// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <gsl/gsl>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Cap folded STATIC outputs (DLC bloat). DQQ scales are scalar, so a large fold only bloats the DLC.
inline constexpr size_t kFoldedStaticMaxBytes = 1024 * 1024;  // 1 MiB

// Chaining without a global set: a STATIC wrapper is initializer-backed or fold-derived.
inline bool IsFoldedStaticTensor(const QnnModelWrapper& qnn_model_wrapper, const std::string& tensor_name) {
  return qnn_model_wrapper.IsQnnTensorWrapperExist(tensor_name) &&
         qnn_model_wrapper.GetQnnTensorWrapper(tensor_name).GetTensorType() == QNN_TENSOR_TYPE_STATIC;
}

inline bool IsConstantOrFoldedStatic(const QnnModelWrapper& qnn_model_wrapper, const std::string& tensor_name) {
  return qnn_model_wrapper.IsConstantInput(tensor_name) || IsFoldedStaticTensor(qnn_model_wrapper, tensor_name);
}

inline Ort::Status GetConstantOrFoldedBytes(QnnModelWrapper& qnn_model_wrapper,
                                            const std::string& tensor_name,
                                            /*out*/ std::vector<uint8_t>& bytes) {
  if (qnn_model_wrapper.IsConstantInput(tensor_name)) {
    const OrtValueInfo* init = qnn_model_wrapper.GetConstantTensor(tensor_name);
    RETURN_IF(init == nullptr, "Constant initializer not found for tensor.");
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(init, bytes));
    ONNXTensorElementDataType onnx_data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(utils::GetOnnxTensorElemDataType(init, onnx_data_type));
    utils::SignExtendUnpackedSubByteData(onnx_data_type, gsl::make_span(bytes));
    return Ort::Status();
  }
  RETURN_IF(!IsFoldedStaticTensor(qnn_model_wrapper, tensor_name),
            "Tensor is not a constant initializer or folded static tensor.");
  const QnnTensorWrapper& wrapper = qnn_model_wrapper.GetQnnTensorWrapper(tensor_name);
  const Qnn_ClientBuffer_t& buf = GetQnnTensorClientBuf(wrapper.GetQnnTensor());
  const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(buf.data);
  RETURN_IF(data_ptr == nullptr && buf.dataSize != 0, "Folded static tensor has null data.");
  bytes.assign(data_ptr, data_ptr + buf.dataSize);
  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
