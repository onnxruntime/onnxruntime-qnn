// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "QnnGraph.h"
#include "QnnTypes.h"

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Pure virtual interface for QNN graph-emit operations: create tensors/nodes,
// validate ops, finalize, and execute. Concrete implementations:
//   - QnnGraphEmitter  : delegates to QNN_INTERFACE_VER_TYPE C function pointers (Phase 1)
//   - QairtGraphEmitter: delegates to QAIRT C++ API (Phase 2, USE_QAIRT_API=ON)
class IGraphEmitter {
 public:
  virtual ~IGraphEmitter() = default;

  // Create a new QNN graph inside `ctx`. On success `out` is set.
  virtual Ort::Status CreateGraph(Qnn_ContextHandle_t ctx,
                                  const char* name,
                                  const QnnGraph_Config_t** configs,
                                  Qnn_GraphHandle_t& out) = 0;

  // Retrieve an existing graph by name (used during context-from-binary path).
  virtual Ort::Status RetrieveGraph(Qnn_ContextHandle_t ctx,
                                    const char* name,
                                    Qnn_GraphHandle_t& out) = 0;

  // Register a tensor in the graph (tensorCreateGraphTensor).
  virtual Ort::Status CreateTensor(Qnn_GraphHandle_t graph,
                                   Qnn_Tensor_t& tensor) = 0;

  // Add a node to the graph (graphAddNode).
  virtual Ort::Status AddNode(Qnn_GraphHandle_t graph,
                              const Qnn_OpConfig_t& op) = 0;

  // Validate an op config against the backend (backendValidateOpConfig).
  virtual Ort::Status ValidateOp(Qnn_BackendHandle_t backend,
                                 const Qnn_OpConfig_t& op) = 0;

  // Finalize the graph (graphFinalize).
  virtual Ort::Status FinalizeGraph(Qnn_GraphHandle_t graph,
                                    Qnn_ProfileHandle_t profile) = 0;

  // Execute the graph (graphExecute). Returns raw Qnn_ErrorHandle_t so callers
  // can detect specific error codes (e.g. QNN_COMMON_ERROR_SYSTEM_COMMUNICATION for SSR).
  virtual Qnn_ErrorHandle_t ExecuteGraph(Qnn_GraphHandle_t graph,
                                         Qnn_Tensor_t* inputs,
                                         uint32_t n_inputs,
                                         Qnn_Tensor_t* outputs,
                                         uint32_t n_outputs,
                                         Qnn_ProfileHandle_t profile,
                                         Qnn_SignalHandle_t signal) = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
