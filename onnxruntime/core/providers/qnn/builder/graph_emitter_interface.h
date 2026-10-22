// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#include "QnnTypes.h"
#include "core/providers/qnn/ort_api.h"
namespace onnxruntime::qnn {
class IGraphEmitter {
 public:
  virtual ~IGraphEmitter() = default;
  virtual Ort::Status CreateGraph(const char* name) = 0;
  virtual Ort::Status CreateTensor(Qnn_Tensor_t& tensor) = 0;
  virtual Ort::Status AddNode(const Qnn_OpConfig_t& op) = 0;
  virtual Ort::Status ValidateOp(const Qnn_OpConfig_t& op) = 0;
};
}  // namespace onnxruntime::qnn
