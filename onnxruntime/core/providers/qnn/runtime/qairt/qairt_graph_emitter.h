// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
#pragma once
#ifdef USE_QAIRT_API
#include <functional>
#include <memory>
#include <unordered_map>
#include "QairtCpp/QairtGraph.hpp"
#include "QairtCpp/QairtContext.hpp"
#include "core/providers/qnn/runtime/interfaces/graph_emitter_interface.h"
namespace qairt {
class Api;
}
namespace onnxruntime::qnn {
class QairtProfileAdapter;

class QairtGraphProfile final : public IGraphProfile {
 public:
  explicit QairtGraphProfile(std::shared_ptr<qairt::Profile> profile) : profile_(std::move(profile)) {}
  GraphRuntimeKind RuntimeKind() const override { return GraphRuntimeKind::Qairt; }
  const std::shared_ptr<qairt::Profile>& Profile() const { return profile_; }

 private:
  std::shared_ptr<qairt::Profile> profile_;
};

class QairtGraphEmitter final : public IGraphEmitter {
 public:
  struct Dependencies {
    const qairt::Api& api;
    std::function<std::shared_ptr<qairt::Context>()> get_context;
    std::function<qairt::GraphConfiguration()> create_graph_config;
    std::function<Ort::Status(qairt::Graph&)> configure_retrieved_graph;
    std::function<Ort::Status(const Qnn_OpConfig_t&)> validate_op;
  };
  explicit QairtGraphEmitter(Dependencies dependencies);
  ~QairtGraphEmitter() override = default;
  Ort::Status RetrieveGraph(std::shared_ptr<qairt::Context> context, const std::string& name);
  Ort::Status InitializeGraph(Qnn_ContextHandle_t context,
                              const QnnGraph_Config_t** graph_configs) override;
  Ort::Status CreateGraph(const char* name) override;
  Ort::Status CreateTensor(Qnn_Tensor_t& tensor) override;
  Ort::Status AddNode(const Qnn_OpConfig_t& op) override;
  Ort::Status ValidateOp(const Qnn_OpConfig_t& op) override;
  Ort::Status Finalize(const IGraphProfile* profile) override;
  Ort::Status Execute(Qnn_Tensor_t* inputs, uint32_t n_inputs,
                      Qnn_Tensor_t* outputs, uint32_t n_outputs,
                      const IGraphProfile* profile,
                      GraphExecutionResult& result) override;

 private:
  // Factory callbacks retain runtime ownership. Declaration order destroys SDK
  // graph objects before those callbacks release the backend/API lifetime.
  Dependencies dependencies_;
  std::shared_ptr<qairt::Context> qairt_context_;
  std::shared_ptr<qairt::Graph> qairt_graph_;
  std::unordered_map<uint32_t, qairt::Tensor> registered_tensors_;
  // Metadata is immutable after graph initialization. Execute is serialized by
  // QnnModel's per-graph mutex; only client buffers change between runs.
  std::vector<qairt::Tensor> input_tensors_;
  std::vector<qairt::Tensor> output_tensors_;
  bool io_metadata_initialized_ = false;
};
}  // namespace onnxruntime::qnn
#endif
