// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "core/providers/qnn/ort_api.h"  // Ort::CustomOpBase, Ort::CustomOpDomain

namespace onnxruntime {
namespace qnn {

// A placeholder kernel whose Compute() always returns an error.
//
// QNN EP is a compile-based EP: custom-domain nodes are fused and compiled via
// EpGraphSupportInfo_AddNodesToFuse(). The kernel is registered only to satisfy
// ORT's model-load schema validation; it must never actually execute. Returning
// an explicit error (rather than a silent no-op) prevents wrong results if a UDO
// node is accidentally not fused.
struct QnnUdoPlaceholderKernel {
  OrtStatusPtr ComputeV2(OrtKernelContext* /*context*/) {
    return Ort::GetApi().CreateStatus(
        ORT_FAIL,
        "QnnUdoPlaceholderOp kernel must never execute; "
        "the node should be fused and compiled by QNN EP.");
  }
};

// A variadic-input/variadic-output placeholder OrtCustomOp used by QnnEpFactory to
// register a UDO op schema so ORT can load and validate ONNX models that contain
// custom-domain nodes.  The factory builds one of these per op-type entry found in
// ORT_QNN_CUSTOM_OP_DOMAINS and keeps it alive for the factory's lifetime.
struct QnnUdoPlaceholderOp
    : Ort::CustomOpBase<QnnUdoPlaceholderOp, QnnUdoPlaceholderKernel, /*WithStatus=*/true> {
  QnnUdoPlaceholderOp(std::string op_type, std::string ep_type)
      : op_type_(std::move(op_type)), ep_type_(std::move(ep_type)) {}
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnUdoPlaceholderOp);

  OrtStatusPtr CreateKernelV2(const OrtApi& /*api*/,
                              const OrtKernelInfo* /*info*/,
                              void** op_kernel) const {
    *op_kernel = new QnnUdoPlaceholderKernel();
    return nullptr;
  }

  OrtStatusPtr KernelComputeV2(void* op_kernel, OrtKernelContext* context) const {
    return static_cast<QnnUdoPlaceholderKernel*>(op_kernel)->ComputeV2(context);
  }

  const char* GetName() const { return op_type_.c_str(); }
  const char* GetExecutionProviderType() const { return ep_type_.c_str(); }

  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
  bool GetVariadicInputHomogeneity() const { return false; }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
  bool GetVariadicOutputHomogeneity() const { return false; }

  // ORT calls InferOutputShapeFn unconditionally for every custom-op node during
  // Graph::PerformTypeAndShapeInferencing, even when the model already carries
  // output value_info. Without this function, loading an ONNX model whose
  // custom-domain node uses UNDEFINED input/output types fails with a type
  // inference error ("output arg ... type inference failed") because ORT cannot
  // resolve the output type from the variadic heterogeneous schema alone.
  //
  // This implementation propagates input[0]'s shape to all outputs with type
  // FLOAT (the default of SetOutputShape). For the placeholder op the kernel
  // never executes (the node is fused and compiled by QNN EP), so the type
  // is only needed to satisfy model-load validation. Current UDO usage is float
  // at the ONNX graph level (QDQ wrapping is stripped by UDOQDQFusion before
  // QNN sees the node), so FLOAT matches the model's declared type. A UDO whose
  // ONNX-level output shape genuinely differs from input[0], or whose type is
  // non-FLOAT, needs per-op type/shape configuration — deferred to a follow-up.
  static OrtStatusPtr InferOutputShape(Ort::ShapeInferContext& ctx) {
    if (ctx.GetInputCount() == 0) {
      return nullptr;
    }
    const auto& input_shape = ctx.GetInputShape(0);
    for (size_t i = 0; i < 1 /* placeholder has exactly 1 output */; ++i) {
      ctx.SetOutputShape(i, input_shape);
    }
    return nullptr;
  }

 private:
  std::string op_type_;
  std::string ep_type_;
};

}  // namespace qnn
}  // namespace onnxruntime
