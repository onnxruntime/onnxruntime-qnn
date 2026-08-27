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

 private:
  std::string op_type_;
  std::string ep_type_;
};

}  // namespace qnn
}  // namespace onnxruntime
