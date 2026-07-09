// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstring>
#include <memory>
#include <string>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {

// The qti_aisw block ops that the QNN EP supports via placeholder custom ops (see
// QtiAiswPlaceholderOp below). Extend this list when adding a new block op — the factory
// registers one placeholder per name, and the matching op builder must be registered in
// op_builder_factory.cc.
inline constexpr std::array<const char*, 3> kQtiAiswBlockOpNames = {
    "Buffer", "StatefulLstm", "StatefulGru"};

// Placeholder ("schema-only") custom op for the qti_aisw block ops.
//
// The qti_aisw ops have no registered ONNX schema, so ORT's Graph::Resolve() rejects any model
// that contains them with "qti_aisw:<Op>(-1) is not a registered function/op" — this happens
// during model load, before EP partitioning, so the QNN EP never gets a chance to claim the node.
//
// Following the ORT plugin-EP "placeholder op" pattern (see onnxruntime_ep_c_api.h
// GetCustomOpDomains docs), the QNN EP factory supplies these placeholder ops via
// GetCustomOpDomains(). ORT registers them into the session when the EP is appended, which
// satisfies model validation. The op's kernel Compute() is a no-op and is never invoked for
// HTP-assigned nodes: GetCapability() reports these nodes as fused/compiled, so QNN runs them.
//
// The ops declare fully variadic, heterogeneous inputs and outputs so validation accepts the
// varying arities of the ops without pinning exact type/count. A type/shape inference function
// (InferOutputShapeImpl) sets output 0's shape; see the members below and its definition for why.

// Minimal kernel — no-op on the QNN path. If a rejected node falls back to CPU, Compute fails
// explicitly (no CPU implementation for these ops).
struct QtiAiswPlaceholderKernel {
  explicit QtiAiswPlaceholderKernel(const char* op_name) : op_name_(op_name) {}

  OrtStatusPtr ComputeV2(OrtKernelContext* /*context*/) {
    const std::string msg = std::string("QNN EP: qti_aisw op '") + (op_name_ ? op_name_ : "?") +
                            "' was not claimed by the QNN backend and has no CPU implementation; "
                            "it cannot be executed on a fallback provider.";
    return Ort::GetApi().CreateStatus(ORT_FAIL, msg.c_str());
  }

 private:
  const char* op_name_ = nullptr;
};

struct QtiAiswPlaceholderOp
    : Ort::CustomOpBase<QtiAiswPlaceholderOp, QtiAiswPlaceholderKernel, /*WithStatus*/ true> {
  explicit QtiAiswPlaceholderOp(const char* op_name) : name_(op_name) {
    // Use InferOutputShapeImpl for type/shape inference — built-in path unsuitable (see members).
    OrtCustomOp::InferOutputShapeFn = InferOutputShapeImpl;
  }

  const char* GetName() const { return name_; }
  // nullptr = CPU provider. Rejected nodes fall back here so session creation doesn't fail;
  // Compute errors explicitly if such a node is ever executed (see QtiAiswPlaceholderKernel).
  const char* GetExecutionProviderType() const { return nullptr; }

  // OPTIONAL inputs to accept empty interior slots (e.g. sequence_lens, B, initial_h).
  // Most are UNDEFINED (any type); reset slot is BOOL to prevent InferOutputTypes from
  // propagating bool to outputs (see GetInputType for slot indices).
  static constexpr size_t kMaxInputs = 24;  // generous upper bound; covers every block op's arity
  static constexpr size_t kMaxOutputs = 3;  // Y, Y_h, Y_c

  size_t GetInputTypeCount() const { return kMaxInputs; }
  // Reset slot is BOOL (single-type) so InferOutputTypes doesn't propagate bool to outputs.
  //   StatefulGru: in[6], StatefulLstm: in[8], Buffer: in[1]
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (name_ != nullptr) {
      if ((std::strcmp(name_, "StatefulGru") == 0 && index == 6) ||
          (std::strcmp(name_, "StatefulLstm") == 0 && index == 8) ||
          (std::strcmp(name_, "Buffer") == 0 && index == 1)) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
      }
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
    *op_kernel = std::make_unique<QtiAiswPlaceholderKernel>(name_).release();
    return nullptr;
  }

  OrtStatusPtr KernelComputeV2(void* op_kernel, OrtKernelContext* context) const {
    return static_cast<QtiAiswPlaceholderKernel*>(op_kernel)->ComputeV2(context);
  }

 private:
  const char* name_ = nullptr;

  static int64_t ReadIntAttr(const OrtApi& api, OrtShapeInferContext* ctx,
                             const char* name, int64_t fallback);
  static bool ReadStringAttr(const OrtApi& api, OrtShapeInferContext* ctx,
                             const char* name, std::string& out_str);
  static OrtStatusPtr InferOutputShapeImpl(const OrtCustomOp* op, OrtShapeInferContext* ctx);
};

}  // namespace onnxruntime
