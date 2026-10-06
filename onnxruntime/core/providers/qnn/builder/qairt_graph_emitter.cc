// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifdef USE_QAIRT_API

#include "core/providers/qnn/builder/qairt_graph_emitter.h"
#include "core/providers/qnn/builder/qairt_type_convert.h"
#include "core/providers/qnn/builder/qnn_def.h"

#include "QairtCpp/QairtApi.hpp"

#include <stdexcept>
#include <cstdio>

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
  fprintf(stderr, "[QAIRT C++ API] QairtGraphEmitter::CreateGraph — graph: %s\n", name);
  fflush(stderr);
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
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::CreateGraph std::exception for `" +
                         std::string(name) + "`: " + e.what())
                            .c_str());
  } catch (...) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::CreateGraph unknown exception for `" +
                         std::string(name) + "`")
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
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::RetrieveGraph std::exception for `" +
                         std::string(name) + "`: " + e.what())
                            .c_str());
  } catch (...) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::RetrieveGraph unknown exception for `" +
                         std::string(name) + "`")
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
    uint32_t id = static_cast<uint32_t>(qairt_tensor.getId());
    SetQnnTensorID(tensor, id);
    // Store registered tensor — addNode must use shallowCopy of these.
    registered_tensors_.emplace(id, std::move(qairt_tensor));
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::CreateTensor failed: " +
                         std::string(e.what()))
                            .c_str());
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::CreateTensor std::exception: " +
                         std::string(e.what()))
                            .c_str());
  } catch (...) {
    return MAKE_EP_FAIL("QairtGraphEmitter::CreateTensor unknown exception");
  }
  return Ort::Status();
}

Ort::Status QairtGraphEmitter::AddNode(Qnn_GraphHandle_t /*graph*/,
                                       const Qnn_OpConfig_t& op) {
  try {
    qairt::OpConfig qairt_op = api_.make<qairt::OpConfig>();
    const auto& v1 = op.v1;

    if (v1.name) qairt_op.setName(v1.name);
    if (v1.packageName) qairt_op.setPackageName(v1.packageName);
    if (v1.typeName) qairt_op.setTypeName(v1.typeName);

    // Convert params
    std::vector<qairt::Param> params;
    params.reserve(v1.numOfParams);
    for (uint32_t i = 0; i < v1.numOfParams; ++i) {
      const auto& qp = v1.params[i];
      auto param = api_.make<qairt::Param>();
      if (qp.name) param.setName(qp.name);

      if (qp.paramType == QNN_PARAMTYPE_SCALAR) {
        qairt::Scalar scalar;
        QAIRT_RETURN_IF_ERROR(qairt_convert::ConvertScalarPublic(qp.scalarParam, api_, scalar));
        param.setScalar(scalar);
      } else if (qp.paramType == QNN_PARAMTYPE_TENSOR) {
        uint32_t id = GetQnnTensorID(qp.tensorParam);
        auto it = registered_tensors_.find(id);
        if (it == registered_tensors_.end()) {
          return MAKE_EP_FAIL(("QairtGraphEmitter::AddNode: param tensor id=" +
                               std::to_string(id) + " not registered")
                                  .c_str());
        }
        param.setTensor(it->second.shallowCopy());
      }
      params.push_back(std::move(param));
    }
    qairt_op.setParams(params);

    // ponytail: OpConfig tensors MUST be shallowCopy of registered tensors.
    // shallowCopy creates a new handle with properties (isNative) already set from
    // createGraphTensor time. Fresh tensors default to APP_WRITE=0 at C level —
    // QairtHtp.dll rejects that in addNode validation.
    std::vector<qairt::Tensor> inputs;
    inputs.reserve(v1.numOfInputs);
    for (uint32_t i = 0; i < v1.numOfInputs; ++i) {
      uint32_t id = GetQnnTensorID(v1.inputTensors[i]);
      auto it = registered_tensors_.find(id);
      if (it == registered_tensors_.end()) {
        return MAKE_EP_FAIL(("QairtGraphEmitter::AddNode: input tensor id=" +
                             std::to_string(id) + " not registered")
                                .c_str());
      }
      inputs.push_back(it->second.shallowCopy());
    }
    qairt_op.setInputs(inputs);

    std::vector<qairt::Tensor> outputs;
    outputs.reserve(v1.numOfOutputs);
    for (uint32_t i = 0; i < v1.numOfOutputs; ++i) {
      uint32_t id = GetQnnTensorID(v1.outputTensors[i]);
      auto it = registered_tensors_.find(id);
      if (it == registered_tensors_.end()) {
        return MAKE_EP_FAIL(("QairtGraphEmitter::AddNode: output tensor id=" +
                             std::to_string(id) + " not registered")
                                .c_str());
      }
      outputs.push_back(it->second.shallowCopy());
    }
    qairt_op.setOutputs(outputs);

    current_graph_->addNode(qairt_op);
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::AddNode failed: " +
                         std::string(e.what()))
                            .c_str());
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::AddNode std::exception: " +
                         std::string(e.what()))
                            .c_str());
  } catch (...) {
    return MAKE_EP_FAIL("QairtGraphEmitter::AddNode unknown exception");
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
  fprintf(stderr, "[QAIRT C++ API] QairtGraphEmitter::FinalizeGraph\n");
  fflush(stderr);
  try {
    // ponytail: profile handle ignored — QAIRT profile is set via GraphConfiguration
    // or passed to execute(). Add profile forwarding when perf infra wires up.
    current_graph_->finalize();
  } catch (const qairt::Exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::FinalizeGraph failed: " +
                         std::string(e.what()))
                            .c_str());
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(("QairtGraphEmitter::FinalizeGraph std::exception: " +
                         std::string(e.what()))
                            .c_str());
  } catch (...) {
    return MAKE_EP_FAIL("QairtGraphEmitter::FinalizeGraph unknown exception");
  }
  return Ort::Status();
}

Qnn_ErrorHandle_t QairtGraphEmitter::ExecuteGraph(Qnn_GraphHandle_t /*graph*/,
                                                  Qnn_Tensor_t* inputs,
                                                  uint32_t n_inputs,
                                                  Qnn_Tensor_t* outputs,
                                                  uint32_t n_outputs,
                                                  Qnn_ProfileHandle_t /*profile*/,
                                                  Qnn_SignalHandle_t /*signal*/) {
  fprintf(stderr, "[QAIRT C++ API] QairtGraphEmitter::ExecuteGraph (inputs=%u, outputs=%u)\n", n_inputs, n_outputs);
  fflush(stderr);
  try {
    std::vector<qairt::Tensor> qairt_inputs;
    qairt_inputs.reserve(n_inputs);
    for (uint32_t i = 0; i < n_inputs; ++i) {
      qairt::Tensor t;
      auto s = qairt_convert::FromQnnTensor(inputs[i], api_, t);
      if (!s.IsOK()) return QNN_GRAPH_ERROR_GENERAL;
      qairt_inputs.push_back(std::move(t));
    }

    std::vector<qairt::Tensor> qairt_outputs;
    qairt_outputs.reserve(n_outputs);
    for (uint32_t i = 0; i < n_outputs; ++i) {
      qairt::Tensor t;
      auto s = qairt_convert::FromQnnTensor(outputs[i], api_, t);
      if (!s.IsOK()) return QNN_GRAPH_ERROR_GENERAL;
      qairt_outputs.push_back(std::move(t));
    }

    current_graph_->execute(qairt_inputs, qairt_outputs);

    for (uint32_t i = 0; i < n_outputs; ++i) {
      auto& cb = qairt_outputs[i].getClientBuffer();
      SetQnnTensorClientBufSize(outputs[i], cb.getDataSize());
    }
  } catch (const qairt::Exception&) {
    return QNN_GRAPH_ERROR_GENERAL;
  } catch (...) {
    return QNN_GRAPH_ERROR_GENERAL;
  }
  return QNN_GRAPH_NO_ERROR;
}

}  // namespace qnn
}  // namespace onnxruntime

#endif  // USE_QAIRT_API
