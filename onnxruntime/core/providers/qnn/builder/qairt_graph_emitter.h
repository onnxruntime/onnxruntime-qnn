// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef USE_QAIRT_API

#include <memory>
#include <string>
#include <unordered_map>

#include "QairtCpp/QairtApi.hpp"
#include "QairtCpp/QairtContext.hpp"
#include "QairtCpp/QairtGraph.hpp"

#include "core/providers/qnn/builder/graph_emitter_interface.h"

namespace onnxruntime {
namespace qnn {

// IGraphEmitter implementation backed by QAIRT C++ API.
// Accepts QNN C types at the interface boundary (to keep callers unchanged),
// converts internally, and delegates to qairt::Graph methods.
//
// Lifecycle: created by QairtBackendManager, which owns the Api/Context objects.
// The emitter holds non-owning references — the manager must outlive it.
class QairtGraphEmitter final : public IGraphEmitter {
 public:
  QairtGraphEmitter(qairt::Api& api, qairt::Context& context);
  ~QairtGraphEmitter() override = default;

  QairtGraphEmitter(const QairtGraphEmitter&) = delete;
  QairtGraphEmitter& operator=(const QairtGraphEmitter&) = delete;

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
  qairt::Api& api_;
  qairt::Context& context_;
  // ponytail: single-graph for now; multi-graph support via map if needed.
  std::shared_ptr<qairt::Graph> current_graph_;
  // ponytail: tensors registered via createGraphTensor, keyed by ID.
  // addNode must reference these objects (shallowCopy), not fresh tensors —
  // QairtHtp.dll validates by handle identity, not ID matching.
  std::unordered_map<uint32_t, qairt::Tensor> registered_tensors_;
};

}  // namespace qnn
}  // namespace onnxruntime

#endif  // USE_QAIRT_API
