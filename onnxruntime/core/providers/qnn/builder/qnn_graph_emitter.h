// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "QnnInterface.h"

#include "core/providers/qnn/builder/i_graph_emitter.h"

namespace onnxruntime {
namespace qnn {

// Concrete IGraphEmitter implementation that delegates every operation to the
// QNN C function-pointer interface (QNN_INTERFACE_VER_TYPE). This is the
// default path (USE_QAIRT_API=OFF). It holds a non-owning reference to the
// interface struct that QnnBackendManager owns.
class QnnGraphEmitter final : public IGraphEmitter {
 public:
  // qnn_interface must outlive this object (owned by QnnBackendManager).
  explicit QnnGraphEmitter(const QNN_INTERFACE_VER_TYPE& qnn_interface)
      : qnn_interface_(qnn_interface) {}

  ~QnnGraphEmitter() override = default;

  // Disallow copy/move — the interface reference is not rebindable.
  QnnGraphEmitter(const QnnGraphEmitter&) = delete;
  QnnGraphEmitter& operator=(const QnnGraphEmitter&) = delete;

  Ort::Status CreateGraph(Qnn_ContextHandle_t ctx,
                          const char* name,
                          const QnnGraph_Config_t** configs,
                          Qnn_GraphHandle_t& out) override;

  Ort::Status RetrieveGraph(Qnn_ContextHandle_t ctx,
                             const char* name,
                             Qnn_GraphHandle_t& out) override;

  Ort::Status CreateTensor(Qnn_GraphHandle_t graph,
                           Qnn_Tensor_t& tensor) override;

  Ort::Status AddNode(Qnn_GraphHandle_t graph,
                      const Qnn_OpConfig_t& op) override;

  Ort::Status ValidateOp(Qnn_BackendHandle_t backend,
                         const Qnn_OpConfig_t& op) override;

  Ort::Status FinalizeGraph(Qnn_GraphHandle_t graph,
                            Qnn_ProfileHandle_t profile) override;

  Qnn_ErrorHandle_t ExecuteGraph(Qnn_GraphHandle_t graph,
                                 Qnn_Tensor_t* inputs,
                                 uint32_t n_inputs,
                                 Qnn_Tensor_t* outputs,
                                 uint32_t n_outputs,
                                 Qnn_ProfileHandle_t profile,
                                 Qnn_SignalHandle_t signal) override;

 private:
  const QNN_INTERFACE_VER_TYPE& qnn_interface_;
};

}  // namespace qnn
}  // namespace onnxruntime
