// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#ifdef USE_QAIRT_API
#include <memory>
#include <unordered_map>
#include "QairtCpp/QairtGraph.hpp"
#include "QairtCpp/QairtContext.hpp"
#include "core/providers/qnn/builder/graph_emitter_interface.h"
namespace onnxruntime::qnn {
class QairtBackendManager;
class QairtProfileAdapter;
class QairtGraphEmitter final : public IGraphEmitter {
 public:
  explicit QairtGraphEmitter(std::shared_ptr<QairtBackendManager> manager);
  ~QairtGraphEmitter() override = default;
  Ort::Status RetrieveGraph(std::shared_ptr<qairt::Context> context, const std::string& name);
  Ort::Status CreateGraph(const char* name) override;
  Ort::Status CreateTensor(Qnn_Tensor_t& tensor) override;
  Ort::Status AddNode(const Qnn_OpConfig_t& op) override;
  Ort::Status ValidateOp(const Qnn_OpConfig_t& op) override;
  Ort::Status Finalize(qairt::Profile* profile);
  Ort::Status Execute(Qnn_Tensor_t* inputs, uint32_t n_inputs,
                      Qnn_Tensor_t* outputs, uint32_t n_outputs, std::shared_ptr<qairt::Profile> profile);

 private:
  // Declaration order ensures tensors/graph/context are freed before the backend/API.
  std::shared_ptr<QairtBackendManager> manager_;
  std::shared_ptr<qairt::Context> context_;
  std::shared_ptr<qairt::Graph> current_graph_;
  std::unordered_map<uint32_t, qairt::Tensor> registered_tensors_;
};
}  // namespace onnxruntime::qnn
#endif
