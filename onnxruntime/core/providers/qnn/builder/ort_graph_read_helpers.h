// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Graph-structure reads through an injected OrtApi rather than the process-global Ort::Const*
// wrappers, so snapshot handles resolve to the shim accessors at background compose time.
namespace ort_read {

inline std::string NodeOpType(const OrtApi& ort_api, const OrtNode* node) {
  const char* op_type = nullptr;
  Ort::ThrowOnError(ort_api.Node_GetOperatorType(node, &op_type));
  return op_type != nullptr ? std::string(op_type) : std::string();
}

inline std::string NodeName(const OrtApi& ort_api, const OrtNode* node) {
  const char* name = nullptr;
  Ort::ThrowOnError(ort_api.Node_GetName(node, &name));
  return name != nullptr ? std::string(name) : std::string();
}

inline std::string GraphName(const OrtApi& ort_api, const OrtGraph* graph) {
  const char* name = nullptr;
  Ort::ThrowOnError(ort_api.Graph_GetName(graph, &name));
  return name != nullptr ? std::string(name) : std::string();
}

inline std::vector<const OrtValueInfo*> GraphInputs(const OrtApi& ort_api, const OrtGraph* graph) {
  size_t num = 0;
  Ort::ThrowOnError(ort_api.Graph_GetNumInputs(graph, &num));
  std::vector<const OrtValueInfo*> inputs(num);
  if (num > 0) {
    Ort::ThrowOnError(ort_api.Graph_GetInputs(graph, inputs.data(), num));
  }
  return inputs;
}

inline std::vector<const OrtNode*> GraphNodes(const OrtApi& ort_api, const OrtGraph* graph) {
  size_t num = 0;
  Ort::ThrowOnError(ort_api.Graph_GetNumNodes(graph, &num));
  std::vector<const OrtNode*> nodes(num);
  if (num > 0) {
    Ort::ThrowOnError(ort_api.Graph_GetNodes(graph, nodes.data(), num));
  }
  return nodes;
}

inline std::vector<const OrtValueInfo*> GraphInitializers(const OrtApi& ort_api, const OrtGraph* graph) {
  size_t num = 0;
  Ort::ThrowOnError(ort_api.Graph_GetNumInitializers(graph, &num));
  std::vector<const OrtValueInfo*> initializers(num);
  if (num > 0) {
    Ort::ThrowOnError(ort_api.Graph_GetInitializers(graph, initializers.data(), num));
  }
  return initializers;
}

inline std::string ValueInfoName(const OrtApi& ort_api, const OrtValueInfo* value_info) {
  const char* name = nullptr;
  Ort::ThrowOnError(ort_api.GetValueInfoName(value_info, &name));
  return name != nullptr ? std::string(name) : std::string();
}

inline bool IsGraphOutput(const OrtApi& ort_api, const OrtValueInfo* value_info) {
  bool is_graph_output = false;
  Ort::ThrowOnError(ort_api.ValueInfo_IsGraphOutput(value_info, &is_graph_output));
  return is_graph_output;
}

inline std::vector<const OrtValueInfo*> NodeInputs(const OrtApi& ort_api, const OrtNode* node) {
  size_t num = 0;
  Ort::ThrowOnError(ort_api.Node_GetNumInputs(node, &num));
  std::vector<const OrtValueInfo*> inputs(num);
  if (num > 0) {
    Ort::ThrowOnError(ort_api.Node_GetInputs(node, inputs.data(), num));
  }
  return inputs;
}

inline std::vector<const OrtValueInfo*> NodeOutputs(const OrtApi& ort_api, const OrtNode* node) {
  size_t num = 0;
  Ort::ThrowOnError(ort_api.Node_GetNumOutputs(node, &num));
  std::vector<const OrtValueInfo*> outputs(num);
  if (num > 0) {
    Ort::ThrowOnError(ort_api.Node_GetOutputs(node, outputs.data(), num));
  }
  return outputs;
}

inline const OrtNode* ProducerNode(const OrtApi& ort_api, const OrtValueInfo* value_info) {
  const OrtNode* producer = nullptr;
  Ort::ThrowOnError(ort_api.ValueInfo_GetValueProducer(value_info, &producer, nullptr));
  return producer;
}

inline std::vector<const OrtNode*> ConsumerNodes(const OrtApi& ort_api, const OrtValueInfo* value_info) {
  size_t num = 0;
  Ort::ThrowOnError(ort_api.ValueInfo_GetValueNumConsumers(value_info, &num));
  std::vector<const OrtNode*> nodes(num);
  if (num > 0) {
    std::vector<int64_t> indices(num);
    Ort::ThrowOnError(ort_api.ValueInfo_GetValueConsumers(value_info, nodes.data(), indices.data(), num));
  }
  return nodes;
}

inline ONNXTensorElementDataType ElemType(const OrtApi& ort_api, const OrtValueInfo* value_info) {
  const OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ort_api.GetValueInfoTypeInfo(value_info, &type_info));
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  Ort::ThrowOnError(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info));
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::ThrowOnError(ort_api.GetTensorElementType(tensor_info, &elem_type));
  return elem_type;
}

inline std::vector<int64_t> Shape(const OrtApi& ort_api, const OrtValueInfo* value_info) {
  const OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ort_api.GetValueInfoTypeInfo(value_info, &type_info));
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  Ort::ThrowOnError(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info));
  size_t num_dims = 0;
  Ort::ThrowOnError(ort_api.GetDimensionsCount(tensor_info, &num_dims));
  std::vector<int64_t> shape(num_dims);
  if (num_dims > 0) {
    Ort::ThrowOnError(ort_api.GetDimensions(tensor_info, shape.data(), num_dims));
  }
  return shape;
}

}  // namespace ort_read
}  // namespace qnn
}  // namespace onnxruntime
