// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

#include "QnnGraph.h"
#include "QnnTypes.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime::qnn {

enum class GraphRuntimeKind {
  Qnn,
  Qairt,
};

class IGraphProfile {
 public:
  virtual ~IGraphProfile() = default;
  virtual GraphRuntimeKind RuntimeKind() const = 0;
};

struct GraphExecutionResult {
  uint64_t native_error_code = 0;
};

class IGraphEmitter {
 public:
  virtual ~IGraphEmitter() = default;

  virtual Ort::Status InitializeGraph(Qnn_ContextHandle_t context,
                                      const QnnGraph_Config_t** graph_configs) = 0;
  virtual Ort::Status CreateGraph(const char* name) = 0;
  virtual Ort::Status CreateTensor(Qnn_Tensor_t& tensor) = 0;
  virtual Ort::Status AddNode(const Qnn_OpConfig_t& op) = 0;
  virtual Ort::Status ValidateOp(const Qnn_OpConfig_t& op) = 0;
  virtual Ort::Status Finalize(const IGraphProfile* profile) = 0;
  virtual Ort::Status Execute(Qnn_Tensor_t* inputs, uint32_t num_inputs,
                              Qnn_Tensor_t* outputs, uint32_t num_outputs,
                              const IGraphProfile* profile,
                              GraphExecutionResult& result) = 0;
};

}  // namespace onnxruntime::qnn
