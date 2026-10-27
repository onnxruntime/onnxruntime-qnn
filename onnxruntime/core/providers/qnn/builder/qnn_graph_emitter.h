// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "QnnGraph.h"
#include "QnnInterface.h"
#include "core/providers/qnn/builder/graph_emitter_interface.h"

namespace onnxruntime::qnn {

class QnnBackendManager;

class QnnGraphProfile final : public IGraphProfile {
 public:
  explicit QnnGraphProfile(Qnn_ProfileHandle_t handle) : handle_(handle) {}
  GraphRuntimeKind RuntimeKind() const override { return GraphRuntimeKind::Qnn; }
  Qnn_ProfileHandle_t Handle() const { return handle_; }

 private:
  Qnn_ProfileHandle_t handle_ = nullptr;
};

// Owns the QNN graph handle and confines QNN graph API dispatch to one class.
// Backend, context, profiling, and memory lifecycles remain owned by
// QnnBackendManager and QnnModel.
class QnnGraphEmitter final : public IGraphEmitter {
 public:
  explicit QnnGraphEmitter(const QnnBackendManager& manager);
  ~QnnGraphEmitter() override = default;

  void SetCreateInfo(Qnn_ContextHandle_t context, const QnnGraph_Config_t** graph_configs);

  Ort::Status CreateGraph(const char* name) override;
  Ort::Status CreateTensor(Qnn_Tensor_t& tensor) override;
  Ort::Status AddNode(const Qnn_OpConfig_t& op) override;
  Ort::Status ValidateOp(const Qnn_OpConfig_t& op) override;

  Ort::Status RetrieveGraph(Qnn_ContextHandle_t context, const std::string& name);
  Ort::Status Finalize(const IGraphProfile* profile) override;
  Ort::Status Execute(Qnn_Tensor_t* inputs, uint32_t num_inputs,
                      Qnn_Tensor_t* outputs, uint32_t num_outputs,
                      const IGraphProfile* profile,
                      GraphExecutionResult& result) override;
  Ort::Status SetConfig(const QnnGraph_Config_t** graph_configs);

  // Compatibility adapters for existing qnn_def unit-test entry points.
  // Keeping the SDK calls here preserves a single graph API boundary.
  static Ort::Status CreateTensor(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                  Qnn_GraphHandle_t graph, Qnn_Tensor_t& tensor);
  static Ort::Status ValidateOp(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                Qnn_BackendHandle_t backend, const Qnn_OpConfig_t& op);
  static Ort::Status AddNode(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                             Qnn_GraphHandle_t graph, const Qnn_OpConfig_t& op);

  Qnn_GraphHandle_t Graph() const { return graph_; }
  Qnn_ContextHandle_t Context() const { return context_; }
  const std::string& Name() const { return name_; }

 private:
  const QnnBackendManager& manager_;
  Qnn_ContextHandle_t context_ = nullptr;
  const QnnGraph_Config_t** graph_configs_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::string name_;
};

}  // namespace onnxruntime::qnn
