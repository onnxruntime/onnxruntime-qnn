// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "QnnGraph.h"
#include "QnnInterface.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime::qnn {

// Owns the QNN graph handle and confines QNN graph API dispatch to one class.
// Backend, context, profiling, and memory lifecycles remain owned by
// QnnBackendManager and QnnModel.
class QnnGraphEmitter final {
 public:
  explicit QnnGraphEmitter(const QNN_INTERFACE_VER_TYPE& qnn_interface);

  void SetCreateInfo(Qnn_ContextHandle_t context,
                     const QnnGraph_Config_t** graph_configs);
  Ort::Status CreateGraph(const char* name);
  Ort::Status RetrieveGraph(Qnn_ContextHandle_t context, const std::string& name);
  Ort::Status Finalize(Qnn_ProfileHandle_t profile);
  Qnn_ErrorHandle_t Execute(Qnn_Tensor_t* inputs, uint32_t num_inputs,
                            Qnn_Tensor_t* outputs, uint32_t num_outputs,
                            Qnn_ProfileHandle_t profile, Qnn_SignalHandle_t signal) const;
  Ort::Status SetConfig(const QnnGraph_Config_t** graph_configs);

  // Compatibility adapters for existing qnn_def entry points. Keeping the SDK
  // calls here preserves one graph API boundary without changing their callers.
  static Qnn_ErrorHandle_t CreateTensor(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                        Qnn_GraphHandle_t graph, Qnn_Tensor_t& tensor);
  static Qnn_ErrorHandle_t ValidateOp(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                      Qnn_BackendHandle_t backend, const Qnn_OpConfig_t& op);
  static Qnn_ErrorHandle_t AddNode(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                   Qnn_GraphHandle_t graph, const Qnn_OpConfig_t& op);
  static Qnn_ErrorHandle_t RetrieveGraphHandle(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                               Qnn_ContextHandle_t context,
                                               const char* name,
                                               Qnn_GraphHandle_t* graph);

  Qnn_GraphHandle_t Graph() const { return graph_; }
  Qnn_ContextHandle_t Context() const { return context_; }

 private:
  const QNN_INTERFACE_VER_TYPE& qnn_interface_;
  Qnn_ContextHandle_t context_ = nullptr;
  const QnnGraph_Config_t** graph_configs_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::string name_;
};

}  // namespace onnxruntime::qnn
