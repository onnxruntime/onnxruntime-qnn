// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_graph_emitter.h"

#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime::qnn {

QnnGraphEmitter::QnnGraphEmitter(const QNN_INTERFACE_VER_TYPE& qnn_interface)
    : qnn_interface_(qnn_interface) {}

void QnnGraphEmitter::SetCreateInfo(Qnn_ContextHandle_t context,
                                    const QnnGraph_Config_t** graph_configs) {
  context_ = context;
  graph_configs_ = graph_configs;
  graph_ = nullptr;
  name_.clear();
}

Ort::Status QnnGraphEmitter::CreateGraph(const char* name) {
  RETURN_IF(context_ == nullptr, "Invalid Qnn context.");
  RETURN_IF(name == nullptr || name[0] == '\0', "Empty graph name.");
  RETURN_IF(graph_ != nullptr, "A QNN emitter already owns a graph.");

  auto status = qnn_interface_.graphCreate(context_, name, graph_configs_, &graph_);
  if (status != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
    status = RetrieveGraphHandle(qnn_interface_, context_, name, &graph_);
    if (status != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
      return MAKE_EP_FAIL(("Failed to create Qnn graph: " + std::string{name} + ". " +
                           utils::FormatQnnError(qnn_interface_, status))
                              .c_str());
    }
  }

  name_ = name;
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::RetrieveGraph(Qnn_ContextHandle_t context, const std::string& name) {
  RETURN_IF(context == nullptr, "Invalid QNN context.");
  RETURN_IF(name.empty(), "Empty QNN graph name.");

  Qnn_GraphHandle_t graph = nullptr;
  const auto status = RetrieveGraphHandle(qnn_interface_, context, name.c_str(), &graph);
  RETURN_IF(status != QNN_GRAPH_NO_ERROR || graph == nullptr,
            ("Failed to retrieve QNN graph: " + name + ". " +
             utils::FormatQnnError(qnn_interface_, status))
                .c_str());

  context_ = context;
  graph_ = graph;
  name_ = name;
  graph_configs_ = nullptr;
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::Finalize(Qnn_ProfileHandle_t profile) {
  RETURN_IF(graph_ == nullptr, "QNN graph is not initialized.");
  const auto status = qnn_interface_.graphFinalize(graph_, profile, nullptr);
  RETURN_IF(status != QNN_GRAPH_NO_ERROR,
            ("Failed to finalize QNN graph. " + utils::FormatQnnError(qnn_interface_, status)).c_str());
  return Ort::Status();
}

Qnn_ErrorHandle_t QnnGraphEmitter::Execute(Qnn_Tensor_t* inputs, uint32_t num_inputs,
                                           Qnn_Tensor_t* outputs, uint32_t num_outputs,
                                           Qnn_ProfileHandle_t profile, Qnn_SignalHandle_t signal) const {
  if (graph_ == nullptr) {
    return QNN_COMMON_ERROR_INVALID_ARGUMENT;
  }
  return qnn_interface_.graphExecute(graph_, inputs, num_inputs, outputs, num_outputs, profile, signal);
}

Ort::Status QnnGraphEmitter::SetConfig(const QnnGraph_Config_t** graph_configs) {
  RETURN_IF(graph_ == nullptr, "QNN graph is not initialized.");
  RETURN_IF(qnn_interface_.graphSetConfig == nullptr,
            "Invalid function pointer for graphSetConfig; cannot apply runtime graph configs.");
  const auto status = qnn_interface_.graphSetConfig(graph_, graph_configs);
  RETURN_IF(status != QNN_SUCCESS,
            ("Failed to apply runtime graph configs for graph: " + name_ + ". " +
             utils::FormatQnnError(qnn_interface_, status))
                .c_str());
  return Ort::Status();
}

Qnn_ErrorHandle_t QnnGraphEmitter::CreateTensor(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                Qnn_GraphHandle_t graph, Qnn_Tensor_t& tensor) {
  return qnn_interface.tensorCreateGraphTensor(graph, &tensor);
}

Qnn_ErrorHandle_t QnnGraphEmitter::ValidateOp(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                              Qnn_BackendHandle_t backend, const Qnn_OpConfig_t& op) {
  return qnn_interface.backendValidateOpConfig(backend, op);
}

Qnn_ErrorHandle_t QnnGraphEmitter::AddNode(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                           Qnn_GraphHandle_t graph, const Qnn_OpConfig_t& op) {
  return qnn_interface.graphAddNode(graph, op);
}

Qnn_ErrorHandle_t QnnGraphEmitter::RetrieveGraphHandle(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                       Qnn_ContextHandle_t context,
                                                       const char* name,
                                                       Qnn_GraphHandle_t* graph) {
  return qnn_interface.graphRetrieve(context, name, graph);
}

}  // namespace onnxruntime::qnn
