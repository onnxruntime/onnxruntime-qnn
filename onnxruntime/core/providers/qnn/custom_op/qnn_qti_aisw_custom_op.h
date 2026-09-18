// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/custom_op/qnn_custom_op.h"

namespace onnxruntime {

// Custom Qualcomm block-op domain.
inline constexpr const char* kQtiAiswDomain = "qti_aisw";
inline constexpr const char* kQtiAiswBufferOpType = "Buffer";
inline constexpr const char* kQtiAiswStatefulLstmOpType = "StatefulLstm";
inline constexpr const char* kQtiAiswStatefulGruOpType = "StatefulGru";

inline constexpr size_t kQtiAiswBufferResetInputIndex = 1;
inline constexpr size_t kQtiAiswStatefulLstmResetInputIndex = 8;
inline constexpr size_t kQtiAiswStatefulGruResetInputIndex = 6;

// The qti_aisw block ops that the QNN EP supports. Extend this list when adding a new block op:
// the factory registers a QtiAiswPlaceholderOp per op type, and the matching op builder must be
// registered in op_builder_factory.cc.
inline constexpr std::array<const char*, 3> kQtiAiswBlockOpTypes = {
    kQtiAiswBufferOpType, kQtiAiswStatefulLstmOpType, kQtiAiswStatefulGruOpType};

namespace qnn {

// Placeholder schema for qti_aisw block ops. Unlike UDOs, these models retain empty inputs to
// preserve the standard ONNX RNN optional-slot positions. ORT's generic custom-op type inference
// accepts absent inputs only when their formal parameters are OPTIONAL, so these cannot use the
// heterogeneous VARIADIC UDO schema.
struct QtiAiswPlaceholderOp
    : Ort::CustomOpBase<QtiAiswPlaceholderOp, QnnUdoPlaceholderKernel, /*WithStatus=*/true> {
  QtiAiswPlaceholderOp(std::string op_type, std::string ep_type)
      : op_type_(std::move(op_type)), ep_type_(std::move(ep_type)) {}

  const char* GetName() const { return op_type_.c_str(); }
  const char* GetExecutionProviderType() const { return ep_type_.c_str(); }

  // OPTIONAL inputs to accept empty interior slots (e.g. sequence_lens, B, initial_h).
  // Most are UNDEFINED (any type); reset slot is BOOL to prevent InferOutputTypes from
  // propagating bool to outputs (see GetInputType for slot indices).
  static constexpr size_t kMaxInputs = 24;  // generous upper bound; covers every block op's arity
  static constexpr size_t kMaxOutputs = 3;  // Y, Y_h, Y_c

  size_t GetInputTypeCount() const { return kMaxInputs; }
  // Reset slot is BOOL (single-type) so InferOutputTypes doesn't propagate bool to outputs.
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if ((op_type_ == kQtiAiswStatefulGruOpType && index == kQtiAiswStatefulGruResetInputIndex) ||
        (op_type_ == kQtiAiswStatefulLstmOpType && index == kQtiAiswStatefulLstmResetInputIndex) ||
        (op_type_ == kQtiAiswBufferOpType && index == kQtiAiswBufferResetInputIndex)) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  }

  // OPTIONAL outputs: InferOutputShapeFn sets only output 0's shape. Outputs 1+ are left unshapen —
  // SetOutputTypeShape on an out-of-range index crashes (unchecked OOB in ORT's InferenceContext).
  size_t GetOutputTypeCount() const { return kMaxOutputs; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  }

  OrtStatusPtr CreateKernelV2(const OrtApi& /*api*/, const OrtKernelInfo* /*info*/,
                              void** op_kernel) const {
    *op_kernel = std::make_unique<QnnUdoPlaceholderKernel>(op_type_).release();
    return nullptr;
  }

  OrtStatusPtr KernelComputeV2(void* op_kernel, OrtKernelContext* context) const {
    return static_cast<QnnUdoPlaceholderKernel*>(op_kernel)->ComputeV2(context);
  }

 private:
  std::string op_type_;
  std::string ep_type_;
};

}  // namespace qnn

}  // namespace onnxruntime
