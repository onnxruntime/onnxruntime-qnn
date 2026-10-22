// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifdef USE_QAIRT_API

#include "core/providers/qnn/builder/qairt_graph_emitter.h"
#include "core/providers/qnn/builder/qairt_type_convert.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qairt_backend_manager.h"

#include "QairtCpp/QairtApi.hpp"

#include <stdexcept>

#define QAIRT_RETURN_IF_ERROR(expr) \
  do {                              \
    auto _s = (expr);               \
    if (!_s.IsOK()) return _s;      \
  } while (0)

namespace onnxruntime {
namespace qnn {

QairtGraphEmitter::QairtGraphEmitter(std::shared_ptr<QairtBackendManager> manager)
    : manager_(std::move(manager)) {}

Ort::Status QairtGraphEmitter::RetrieveGraph(std::shared_ptr<qairt::Context> context, const std::string& name) {
  try {
    current_graph_ = context->retrieveGraph(name.c_str());
    context_ = std::move(context);
    return Ort::Status();
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(e.what());
  }
}

Ort::Status QairtGraphEmitter::CreateGraph(const char* name) {
  try {
    RETURN_IF(current_graph_, "A QAIRT emitter already owns a graph.");
    context_ = manager_->GetContext();
    auto config = manager_->GetApi().make<qairt::GraphConfiguration>();
    current_graph_ = std::make_shared<qairt::Graph>(context_->createGraph(name, config));
    return Ort::Status();
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(e.what());
  }
}

Ort::Status QairtGraphEmitter::CreateTensor(Qnn_Tensor_t& tensor) {
  try {
    qairt::Tensor qairt_tensor;
    QAIRT_RETURN_IF_ERROR(qairt_convert::FromQnnTensor(tensor, manager_->GetApi(), qairt_tensor));
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

Ort::Status QairtGraphEmitter::AddNode(const Qnn_OpConfig_t& op) {
  try {
    qairt::OpConfig qairt_op = manager_->GetApi().make<qairt::OpConfig>();
    const auto& v1 = op.v1;

    if (v1.name) qairt_op.setName(v1.name);
    if (v1.packageName) qairt_op.setPackageName(v1.packageName);
    if (v1.typeName) qairt_op.setTypeName(v1.typeName);

    // Convert params
    std::vector<qairt::Param> params;
    params.reserve(v1.numOfParams);
    for (uint32_t i = 0; i < v1.numOfParams; ++i) {
      const auto& qp = v1.params[i];
      auto param = manager_->GetApi().make<qairt::Param>();
      if (qp.name) param.setName(qp.name);

      if (qp.paramType == QNN_PARAMTYPE_SCALAR) {
        qairt::Scalar scalar;
        QAIRT_RETURN_IF_ERROR(qairt_convert::ConvertScalarPublic(qp.scalarParam, manager_->GetApi(), scalar));
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

    // OpConfig tensors retain the properties of their registered tensors.
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

Ort::Status QairtGraphEmitter::ValidateOp(const Qnn_OpConfig_t& op) {
  return manager_->ValidateOp(op);
}

Ort::Status QairtGraphEmitter::Finalize(qairt::Profile* profile) {
  try {
    if (profile)
      current_graph_->finalize(*profile);
    else
      current_graph_->finalize();
    return Ort::Status();
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(e.what());
  }
}

Ort::Status QairtGraphEmitter::Execute(Qnn_Tensor_t* inputs, uint32_t n_inputs,
                                       Qnn_Tensor_t* outputs, uint32_t n_outputs,
                                       std::shared_ptr<qairt::Profile> profile) {
  try {
    std::vector<qairt::Tensor> in(n_inputs), out(n_outputs);
    for (uint32_t i = 0; i < n_inputs; ++i) {
      RETURN_IF_ERROR(qairt_convert::FromQnnTensor(inputs[i], manager_->GetApi(), in[i]));
    }
    for (uint32_t i = 0; i < n_outputs; ++i) {
      RETURN_IF_ERROR(qairt_convert::FromQnnTensor(outputs[i], manager_->GetApi(), out[i]));
    }
    current_graph_->execute(in, out, profile);
    for (uint32_t i = 0; i < n_outputs; ++i) {
      SetQnnTensorClientBufSize(outputs[i], out[i].getClientBuffer().getDataSize());
    }
    return Ort::Status();
  } catch (const qairt::Exception& e) {
    if (e == qairt::CommonError::SystemCommunication || e == qairt::CommonError::SystemCommunicationFatal) {
      return Ort::Status("QAIRT device communication failed; QAIRT context recovery is not implemented.",
                         QNN_SSR_UNRECOVERABLE_ERROR_CODE);
    }
    return MAKE_EP_FAIL(e.what());
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(e.what());
  }
}
}  // namespace qnn
}  // namespace onnxruntime
#endif
