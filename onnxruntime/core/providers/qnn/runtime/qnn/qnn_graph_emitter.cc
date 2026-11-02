// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/runtime/qnn/qnn_graph_emitter.h"

#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime::qnn {

QnnGraphEmitter::QnnGraphEmitter(const QNN_INTERFACE_VER_TYPE& qnn_interface, OpValidator validator)
    : qnn_interface_(qnn_interface), validator_(std::move(validator)) {}

Ort::Status QnnGraphEmitter::InitializeGraph(Qnn_ContextHandle_t context,
                                             const QnnGraph_Config_t** graph_configs) {
  RETURN_IF(context == nullptr, "Invalid QNN context.");
  context_ = context;
  graph_configs_ = graph_configs;
  graph_ = nullptr;
  name_.clear();
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::CreateGraph(const char* name) {
  RETURN_IF(context_ == nullptr, "Invalid QNN context.");
  RETURN_IF(name == nullptr || name[0] == '\0', "Empty QNN graph name.");
  RETURN_IF(graph_ != nullptr, "A QNN emitter already owns a graph.");

  const auto& qnn_interface = qnn_interface_;
  auto status = qnn_interface.graphCreate(context_, name, graph_configs_, &graph_);
  if (status != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
    status = qnn_interface.graphRetrieve(context_, name, &graph_);
    if (status != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
      return MAKE_EP_FAIL(("Failed to create QNN graph: " + std::string{name} + ". " +
                           utils::FormatQnnError(qnn_interface, status))
                              .c_str());
    }
  }

  name_ = name;
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::CreateTensor(Qnn_Tensor_t& tensor) {
  RETURN_IF(graph_ == nullptr, "QNN graph is not initialized.");
  const auto status = CreateTensor(qnn_interface_, graph_, tensor);
  RETURN_IF(status != QNN_TENSOR_NO_ERROR, ("QNN tensorCreateGraphTensor failed with error code " + std::to_string(status)).c_str());
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::AddNode(const Qnn_OpConfig_t& op) {
  RETURN_IF(graph_ == nullptr, "QNN graph is not initialized.");
  const auto status = AddNode(qnn_interface_, graph_, op);
  RETURN_IF(status != QNN_GRAPH_NO_ERROR, ("QNN graphAddNode failed with error code " + std::to_string(status)).c_str());
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::ValidateOp(const Qnn_OpConfig_t& op) {
  RETURN_IF(!validator_, "QNN op validator is not initialized.");
  return validator_(op);
}

Ort::Status QnnGraphEmitter::RetrieveGraph(Qnn_ContextHandle_t context, const std::string& name) {
  RETURN_IF(context == nullptr, "Invalid QNN context.");
  RETURN_IF(name.empty(), "Empty QNN graph name.");

  Qnn_GraphHandle_t graph = nullptr;
  const auto& qnn_interface = qnn_interface_;
  const auto status = RetrieveGraphHandle(qnn_interface, context, name.c_str(), &graph);
  RETURN_IF(status != QNN_GRAPH_NO_ERROR || graph == nullptr,
            ("Failed to retrieve QNN graph: " + name + ". " +
             utils::FormatQnnError(qnn_interface, status))
                .c_str());

  context_ = context;
  graph_ = graph;
  name_ = name;
  graph_configs_ = nullptr;
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::Finalize(const IGraphProfile* profile) {
  RETURN_IF(graph_ == nullptr, "QNN graph is not initialized.");
  RETURN_IF(profile != nullptr && profile->RuntimeKind() != GraphRuntimeKind::Qnn,
            "QNN graph received a profile for a different runtime.");
  const auto profile_handle = profile == nullptr
                                  ? nullptr
                                  : static_cast<const QnnGraphProfile*>(profile)->Handle();
  const auto& qnn_interface = qnn_interface_;
  const auto status = qnn_interface.graphFinalize(graph_, profile_handle, nullptr);
  RETURN_IF(status != QNN_GRAPH_NO_ERROR,
            ("Failed to finalize QNN graph. " + utils::FormatQnnError(qnn_interface, status)).c_str());
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::Execute(Qnn_Tensor_t* inputs, uint32_t num_inputs,
                                     Qnn_Tensor_t* outputs, uint32_t num_outputs,
                                     const IGraphProfile* profile,
                                     GraphExecutionResult& result) {
  RETURN_IF(graph_ == nullptr, "QNN graph is not initialized.");
  RETURN_IF(profile != nullptr && profile->RuntimeKind() != GraphRuntimeKind::Qnn,
            "QNN graph received a profile for a different runtime.");
  const auto profile_handle = profile == nullptr
                                  ? nullptr
                                  : static_cast<const QnnGraphProfile*>(profile)->Handle();
  result.native_error_code = qnn_interface_.graphExecute(
      graph_, inputs, num_inputs, outputs, num_outputs, profile_handle, nullptr);
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::SetConfig(const QnnGraph_Config_t** graph_configs) {
  RETURN_IF(graph_ == nullptr, "QNN graph is not initialized.");
  const auto& qnn_interface = qnn_interface_;
  RETURN_IF(qnn_interface.graphSetConfig == nullptr,
            "Invalid function pointer for graphSetConfig; cannot apply runtime graph configs.");
  const auto status = qnn_interface.graphSetConfig(graph_, graph_configs);
  RETURN_IF(status != QNN_GRAPH_NO_ERROR,
            ("Failed to set QNN graph configuration. " + utils::FormatQnnError(qnn_interface, status)).c_str());
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
