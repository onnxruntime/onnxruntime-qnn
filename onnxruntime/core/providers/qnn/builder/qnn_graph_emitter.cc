// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_graph_emitter.h"

#include "QnnGraph.h"
#include "QnnTypes.h"

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

Ort::Status QnnGraphEmitter::CreateGraph(Qnn_ContextHandle_t ctx,
                                          const char* name,
                                          const QnnGraph_Config_t** configs,
                                          Qnn_GraphHandle_t& out) {
  auto rt = qnn_interface_.graphCreate(ctx, name, configs, &out);
  if (rt != QNN_GRAPH_NO_ERROR || out == nullptr) {
    return MAKE_EP_FAIL(("QnnGraphEmitter::CreateGraph failed for graph `" +
                         std::string(name) + "` with error code " +
                         std::to_string(rt))
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::RetrieveGraph(Qnn_ContextHandle_t ctx,
                                            const char* name,
                                            Qnn_GraphHandle_t& out) {
  auto rt = qnn_interface_.graphRetrieve(ctx, name, &out);
  if (rt != QNN_GRAPH_NO_ERROR || out == nullptr) {
    return MAKE_EP_FAIL(("QnnGraphEmitter::RetrieveGraph failed for graph `" +
                         std::string(name) + "` with error code " +
                         std::to_string(rt))
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::CreateTensor(Qnn_GraphHandle_t graph,
                                           Qnn_Tensor_t& tensor) {
  auto rt = qnn_interface_.tensorCreateGraphTensor(graph, &tensor);
  if (rt != QNN_TENSOR_NO_ERROR) {
    return MAKE_EP_FAIL(("QnnGraphEmitter::CreateTensor failed with error code " +
                         std::to_string(rt))
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::AddNode(Qnn_GraphHandle_t graph,
                                      const Qnn_OpConfig_t& op) {
  auto rt = qnn_interface_.graphAddNode(graph, op);
  if (rt != QNN_GRAPH_NO_ERROR) {
    return MAKE_EP_FAIL(("QnnGraphEmitter::AddNode failed with error code " +
                         std::to_string(rt))
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::ValidateOp(Qnn_BackendHandle_t backend,
                                         const Qnn_OpConfig_t& op) {
  auto rt = qnn_interface_.backendValidateOpConfig(backend, op);
  if (rt != QNN_SUCCESS) {
    return MAKE_EP_FAIL(("QnnGraphEmitter::ValidateOp failed with error code " +
                         std::to_string(rt))
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::FinalizeGraph(Qnn_GraphHandle_t graph,
                                            Qnn_ProfileHandle_t profile) {
  auto rt = qnn_interface_.graphFinalize(graph, profile, nullptr);
  if (rt != QNN_GRAPH_NO_ERROR) {
    return MAKE_EP_FAIL(("QnnGraphEmitter::FinalizeGraph failed with error code " +
                         std::to_string(rt))
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QnnGraphEmitter::ExecuteGraph(Qnn_GraphHandle_t graph,
                                           Qnn_Tensor_t* inputs,
                                           uint32_t n_inputs,
                                           Qnn_Tensor_t* outputs,
                                           uint32_t n_outputs,
                                           Qnn_ProfileHandle_t profile,
                                           Qnn_SignalHandle_t signal) {
  auto rt = qnn_interface_.graphExecute(graph, inputs, n_inputs, outputs, n_outputs, profile, signal);
  if (rt != QNN_GRAPH_NO_ERROR) {
    return MAKE_EP_FAIL(("QnnGraphEmitter::ExecuteGraph failed with error code " +
                         std::to_string(rt))
                            .c_str());
  }
  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
