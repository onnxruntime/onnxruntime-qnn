// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifdef USE_QAIRT_API

#include "core/providers/qnn/builder/qairt_graph_emitter.h"
#include "core/providers/qnn/builder/qairt_type_convert.h"
#include "core/providers/qnn/builder/qnn_def.h"

#include "QairtCpp/QairtApi.hpp"

#define QAIRT_RETURN_IF_ERROR(expr) \
  do {                              \
    auto _s = (expr);               \
    if (!_s.IsOK()) return _s;      \
  } while (0)

namespace onnxruntime {
namespace qnn {

QairtGraphEmitter::QairtGraphEmitter(qairt::Api& api, qairt::Context& context)
    : api_(api), context_(context) {}

Ort::Status QairtGraphEmitter::CreateGraph(Qnn_ContextHandle_t /*ctx*/,
                                           const char* name,
                                           const QnnGraph_Config_t** /*configs*/,
                                           Qnn_GraphHandle_t& out) {
  try {
    auto graph_config = api_.make<qairt::GraphConfiguration>();
    auto graph = context_.createGraph(name, graph_config);
    current_graph_ = std::make_shared<qairt::Graph>(std::move(graph));
    // Return a non-null sentinel so callers' null-checks pass.
    // The handle is never dereferenced on the QAIRT path.
    out = reinterpret_cast<Qnn_GraphHandle_t>(current_graph_.get());
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::CreateGraph failed for `" +
                         std::string(name) + "`: " + e.what())
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QairtGraphEmitter::RetrieveGraph(Qnn_ContextHandle_t /*ctx*/,
                                             const char* name,
                                             Qnn_GraphHandle_t& out) {
  try {
    current_graph_ = context_.retrieveGraph(name);
    out = reinterpret_cast<Qnn_GraphHandle_t>(current_graph_.get());
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::RetrieveGraph failed for `" +
                         std::string(name) + "`: " + e.what())
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QairtGraphEmitter::CreateTensor(Qnn_GraphHandle_t /*graph*/,
                                            Qnn_Tensor_t& tensor) {
  try {
    qairt::Tensor qairt_tensor;
    QAIRT_RETURN_IF_ERROR(qairt_convert::FromQnnTensor(tensor, api_, qairt_tensor));
    current_graph_->createGraphTensor(qairt_tensor);
    // Write back the assigned tensor ID so the caller can reference it later.
    SetQnnTensorID(tensor, static_cast<uint32_t>(qairt_tensor.getId()));
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::CreateTensor failed: " +
                         std::string(e.what()))
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QairtGraphEmitter::AddNode(Qnn_GraphHandle_t /*graph*/,
                                       const Qnn_OpConfig_t& op) {
  try {
    qairt::OpConfig qairt_op;
    QAIRT_RETURN_IF_ERROR(qairt_convert::FromQnnOpConfig(op, api_, qairt_op));
    current_graph_->addNode(qairt_op);
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::AddNode failed: " +
                         std::string(e.what()))
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QairtGraphEmitter::ValidateOp(Qnn_BackendHandle_t /*backend*/,
                                          const Qnn_OpConfig_t& /*op*/) {
  // ponytail: QAIRT C++ API has no standalone backendValidateOpConfig equivalent.
  // Validation happens implicitly at addNode(). Return OK — if the op is invalid,
  // addNode will throw. Add explicit validation when QAIRT exposes it.
  return Ort::Status();
}

Ort::Status QairtGraphEmitter::FinalizeGraph(Qnn_GraphHandle_t /*graph*/,
                                             Qnn_ProfileHandle_t /*profile*/) {
  try {
    // ponytail: profile handle ignored — QAIRT profile is set via GraphConfiguration
    // or passed to execute(). Add profile forwarding when perf infra wires up.
    current_graph_->finalize();
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::FinalizeGraph failed: " +
                         std::string(e.what()))
                            .c_str());
  }
  return Ort::Status();
}

Ort::Status QairtGraphEmitter::ExecuteGraph(Qnn_GraphHandle_t /*graph*/,
                                            Qnn_Tensor_t* inputs,
                                            uint32_t n_inputs,
                                            Qnn_Tensor_t* outputs,
                                            uint32_t n_outputs,
                                            Qnn_ProfileHandle_t /*profile*/,
                                            Qnn_SignalHandle_t /*signal*/) {
  try {
    std::vector<qairt::Tensor> qairt_inputs;
    qairt_inputs.reserve(n_inputs);
    for (uint32_t i = 0; i < n_inputs; ++i) {
      qairt::Tensor t;
      QAIRT_RETURN_IF_ERROR(qairt_convert::FromQnnTensor(inputs[i], api_, t));
      qairt_inputs.push_back(std::move(t));
    }

    std::vector<qairt::Tensor> qairt_outputs;
    qairt_outputs.reserve(n_outputs);
    for (uint32_t i = 0; i < n_outputs; ++i) {
      qairt::Tensor t;
      QAIRT_RETURN_IF_ERROR(qairt_convert::FromQnnTensor(outputs[i], api_, t));
      qairt_outputs.push_back(std::move(t));
    }

    current_graph_->execute(qairt_inputs, qairt_outputs);

    // Write back output buffer sizes (backend may have updated them for dynamic shapes).
    for (uint32_t i = 0; i < n_outputs; ++i) {
      auto& cb = qairt_outputs[i].getClientBuffer();
      SetQnnTensorClientBufSize(outputs[i], cb.getDataSize());
    }
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::ExecuteGraph failed: " +
                         std::string(e.what()))
                            .c_str());
  }
  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime

#endif  // USE_QAIRT_API
