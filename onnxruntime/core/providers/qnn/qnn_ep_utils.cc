// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/qnn_ep_utils.h"

#include <iostream>
#include <string>

#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/inlined_containers.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace QDQ {

void OrtSelectors::RegisterSelector(const OrtOpVersionsAndSelector::OpVersionsMap& ops_and_versions_in,
                                    std::unique_ptr<OrtNodeGroupSelector> selector_in) {
  auto entry = std::make_unique<OrtOpVersionsAndSelector>(
      ops_and_versions_in,
      std::move(selector_in));

  selectors_set_.push_back(std::move(entry));
}

namespace {

// =============================================================================
// 1. ValueInfo / node metadata helpers
// =============================================================================

// 1.1 Extract the ONNX element type from a ValueInfo.
std::optional<ONNXTensorElementDataType> GetDataTypeFromValueInfo(const OrtApi& ort_api,
                                                                  const OrtValueInfo* value_info) {
  const OrtTypeInfo* type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(value_info, &type_info), ort_api, std::nullopt);

  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info), ort_api, std::nullopt);
  if (tensor_info == nullptr) {
    return std::nullopt;
  }

  ONNXTensorElementDataType element_type;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetTensorElementType(tensor_info, &element_type), ort_api, std::nullopt);
  return element_type;
}

// 1.2 Element type of `node`'s input[index].
std::optional<ONNXTensorElementDataType> GetNodeInputDataType(const OrtNode* node, const OrtApi& ort_api, int index) {
  size_t num_defs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(node, &num_defs), ort_api, std::nullopt);
  if (index >= static_cast<int>(num_defs)) {
    return std::nullopt;
  }
  std::vector<const OrtValueInfo*> inputs(num_defs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(node, inputs.data(), inputs.size()), ort_api, std::nullopt);
  return GetDataTypeFromValueInfo(ort_api, inputs[index]);
}

// 1.3 Element type of `node`'s output[index].
std::optional<ONNXTensorElementDataType> GetNodeOutputDataType(const OrtNode* node, const OrtApi& ort_api, int index) {
  size_t num_defs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(node, &num_defs), ort_api, std::nullopt);
  if (index >= static_cast<int>(num_defs)) {
    return std::nullopt;
  }
  std::vector<const OrtValueInfo*> outputs(num_defs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(node, outputs.data(), outputs.size()), ort_api, std::nullopt);
  return GetDataTypeFromValueInfo(ort_api, outputs[index]);
}

// 1.4 Shape for a ValueInfo. Returns false on API failure.
bool GetValueInfoShape(const OrtApi& ort_api, const OrtValueInfo* value_info, std::vector<int64_t>& shape) {
  const OrtTypeInfo* type_info = nullptr;
  if (ort_api.GetValueInfoTypeInfo(value_info, &type_info) != nullptr) return false;
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  if (ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info) != nullptr || tensor_info == nullptr) return false;
  size_t rank = 0;
  if (ort_api.GetDimensionsCount(tensor_info, &rank) != nullptr) return false;
  shape.resize(rank);
  return rank == 0 || ort_api.GetDimensions(tensor_info, shape.data(), rank) == nullptr;
}

// =============================================================================
// 2. Initializer / scalar readers
// =============================================================================

// 2.0 Forward decl — defined further below with the QDQ-pair helpers.
const OrtValue* GetConstantInitializer(const OrtGraph* graph, const OrtApi& ort_api, const char* name);

// 2.1 Look up the constant initializer OrtValue backing a ValueInfo by name.
const OrtValue* GetInitializerFromValueInfo(const OrtGraph* graph, const OrtApi& ort_api,
                                            const OrtValueInfo* value_info) {
  const char* name = nullptr;
  if (ort_api.GetValueInfoName(value_info, &name) != nullptr || name == nullptr) {
    return nullptr;
  }
  return GetConstantInitializer(graph, ort_api, name);
}

// 2.2 Read a scalar of type T from an initializer.
template <typename T>
bool GetScalarValue(const OrtApi& ort_api, const OrtValue* initializer, T& value) {
  T* data = nullptr;
  if (ort_api.GetTensorMutableData(const_cast<OrtValue*>(initializer), (void**)&data) != nullptr) {
    return false;
  }
  value = *data;
  return true;
}

// 2.3 Read a scalar zero_point → (int64 value, matching QNN datatype).
bool GetZeroPointValue(const OrtApi& ort_api, const OrtValue* zp_init,
                       int64_t& zero_point, Qnn_DataType_t& qnn_data_type) {
  OrtTensorTypeAndShapeInfo* zp_info = nullptr;
  if (ort_api.GetTensorTypeAndShape(zp_init, &zp_info) != nullptr) return false;

  ONNXTensorElementDataType zp_type;
  OrtStatus* status = ort_api.GetTensorElementType(zp_info, &zp_type);
  ort_api.ReleaseTensorTypeAndShapeInfo(zp_info);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  void* zp_data = nullptr;
  if (ort_api.GetTensorMutableData(const_cast<OrtValue*>(zp_init), &zp_data) != nullptr || zp_data == nullptr) {
    return false;
  }

  switch (zp_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      zero_point = static_cast<int64_t>(*reinterpret_cast<uint8_t*>(zp_data));
      qnn_data_type = QNN_DATATYPE_UFIXED_POINT_8;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      zero_point = static_cast<int64_t>(*reinterpret_cast<uint16_t*>(zp_data));
      qnn_data_type = QNN_DATATYPE_UFIXED_POINT_16;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      zero_point = static_cast<int64_t>(*reinterpret_cast<int8_t*>(zp_data));
      qnn_data_type = QNN_DATATYPE_SFIXED_POINT_8;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      zero_point = static_cast<int64_t>(*reinterpret_cast<int16_t*>(zp_data));
      qnn_data_type = QNN_DATATYPE_SFIXED_POINT_16;
      return true;
    default:
      return false;
  }
}

// 2.4 Read scale (input[1]) and zero_point (input[2]) from a Q/DQ node.
bool GetQNodeScaleAndZeroPoint(const OrtGraph* graph, const OrtApi& ort_api,
                               const OrtNode* q_node,
                               float& scale, int64_t& zero_point,
                               Qnn_DataType_t& qnn_data_type) {
  size_t num_inputs = 0;
  if (ort_api.Node_GetNumInputs(q_node, &num_inputs) != nullptr || num_inputs < 3) return false;

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  if (ort_api.Node_GetInputs(q_node, inputs.data(), inputs.size()) != nullptr) return false;

  const OrtValue* scale_init = GetInitializerFromValueInfo(graph, ort_api, inputs[1]);
  if (scale_init == nullptr || !GetScalarValue(ort_api, scale_init, scale)) return false;

  const OrtValue* zp_init = GetInitializerFromValueInfo(graph, ort_api, inputs[2]);
  if (zp_init == nullptr) return false;

  return GetZeroPointValue(ort_api, zp_init, zero_point, qnn_data_type);
}

// 2.5 Read Clip's [min, max]. Opset 6 stores them as attrs; opset 11+ as inputs[1..2] initializers.
void GetClipMinMax(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* clip_node,
                   float& clip_min, float& clip_max) {
  clip_min = std::numeric_limits<float>::lowest();
  clip_max = std::numeric_limits<float>::max();

  // 2.5.1 Opset 6: attribute-carried min/max.
  OrtNodeAttrHelper clip_helper(*clip_node);
  if (clip_helper.HasAttr("min") || clip_helper.HasAttr("max")) {
    clip_min = clip_helper.Get("min", clip_min);
    clip_max = clip_helper.Get("max", clip_max);
    return;
  }

  // 2.5.2 Opset 11+: initializer-carried min (input[1]) / max (input[2]).
  size_t clip_num_inputs = 0;
  if (ort_api.Node_GetNumInputs(clip_node, &clip_num_inputs) != nullptr || clip_num_inputs < 2) return;
  std::vector<const OrtValueInfo*> clip_inputs(clip_num_inputs);
  if (ort_api.Node_GetInputs(clip_node, clip_inputs.data(), clip_inputs.size()) != nullptr) return;

  if (clip_num_inputs >= 2 && clip_inputs[1] != nullptr) {
    if (const OrtValue* min_init = GetInitializerFromValueInfo(graph, ort_api, clip_inputs[1])) {
      GetScalarValue(ort_api, min_init, clip_min);
    }
  }
  if (clip_num_inputs >= 3 && clip_inputs[2] != nullptr) {
    if (const OrtValue* max_init = GetInitializerFromValueInfo(graph, ort_api, clip_inputs[2])) {
      GetScalarValue(ort_api, max_init, clip_max);
    }
  }
}

// =============================================================================
// 3. QDQ-unit fold helpers
//
//   Two related transforms that expand what the vanilla `DQ -> op -> Q` selector
//   would otherwise pick up:
//
//     3a. TryFoldRedundantActivation:  DQ -> op -> Relu/Clip -> Q
//         Fold the activation into the QDQ unit iff the terminal Q's encoding
//         cannot represent values outside the activation's clamp range (else
//         HTP would silently skip the clamp).
//
//     3b. TryAbsorbTrailingReshape:    Gemm -> Reshape -> [Relu/Clip ->] Q
//         (MatMulAddFusion sandwich) Absorb the trailing Reshape (+ optionally
//         Relu/Clip) so the group's output IODef inherits Q's encoding; the op
//         builder then emits FC + Reshape.
// =============================================================================

// 3.0 Encoding-safe check for a Relu/Clip whose output is quantised by `q_node`.
// Fuse is safe iff [encoding_min, encoding_max] ⊆ [activation_min, activation_max].
bool IsActivationEncodingSafe(const OrtGraph* graph, const OrtApi& ort_api,
                              const OrtNode* activation_node, const std::string& op_type,
                              const OrtNode* q_node) {
  float scale_val = 0.0f;
  int64_t zero_point = 0;
  Qnn_DataType_t qnn_dt = QNN_DATATYPE_UNDEFINED;
  if (!GetQNodeScaleAndZeroPoint(graph, ort_api, q_node, scale_val, zero_point, qnn_dt)) return false;

  int64_t qmin = 0, qmax = 0;
  if (!qnn::utils::GetQminQmax(qnn_dt, qmin, qmax).IsOK()) return false;

  const float encoding_min = scale_val * static_cast<float>(qmin - zero_point);
  const float encoding_max = scale_val * static_cast<float>(qmax - zero_point);

  float activation_min = 0.0f;  // Relu: [0, +inf)
  float activation_max = std::numeric_limits<float>::max();
  if (op_type == "Clip") {
    GetClipMinMax(graph, ort_api, activation_node, activation_min, activation_max);
  }
  return encoding_min >= activation_min && encoding_max <= activation_max;
}

// 3.0.1 Convenience: does `value_info` have exactly one consumer AND is not a graph output?
// Populates `sole_consumer` on success. Any API error → false.
bool GetSoleNonOutputConsumer(const OrtApi& ort_api, const OrtValueInfo* value_info,
                              const OrtNode*& sole_consumer) {
  bool is_graph_output = false;
  if (ort_api.ValueInfo_IsGraphOutput(value_info, &is_graph_output) != nullptr || is_graph_output) return false;
  size_t num_consumers = 0;
  if (ort_api.ValueInfo_GetValueNumConsumers(value_info, &num_consumers) != nullptr || num_consumers != 1) return false;
  int64_t unused_idx = 0;
  if (ort_api.ValueInfo_GetValueConsumers(value_info, &sole_consumer, &unused_idx, 1) != nullptr) return false;
  return sole_consumer != nullptr;
}

// 3.0.2 Convenience: get the single output ValueInfo of `node`. Returns nullptr on any deviation.
const OrtValueInfo* GetSingleOutput(const OrtApi& ort_api, const OrtNode* node) {
  size_t n = 0;
  if (ort_api.Node_GetNumOutputs(node, &n) != nullptr || n != 1) return nullptr;
  std::vector<const OrtValueInfo*> outs(1);
  if (ort_api.Node_GetOutputs(node, outs.data(), 1) != nullptr) return nullptr;
  return outs[0];
}

// 3a. Detect  node -> Relu/Clip -> Q  and, if encoding-safe, return the Relu/Clip so it
//     folds into the QDQ unit. Returns nullptr on any mismatch.
//
// The encoding-safety gate matters: if the Q encoding can represent values outside the
// activation's clamp range (e.g. Relu output → Q with zp != 0 permits negatives), then
// HTP would NOT re-clamp under fusion and the model would be semantically wrong.
const OrtNode* TryFoldRedundantActivation(const OrtGraph* graph, const OrtApi& ort_api,
                                          const OrtNode* node) {
  // 3a.1 node must have exactly one output; that output must have exactly one consumer
  //      and must not be a graph output.
  const OrtValueInfo* node_out = GetSingleOutput(ort_api, node);
  if (node_out == nullptr) return nullptr;
  const OrtNode* activation = nullptr;
  if (!GetSoleNonOutputConsumer(ort_api, node_out, activation)) return nullptr;

  // 3a.2 The consumer must be Relu or Clip, with exactly one output whose sole
  //      consumer is a QuantizeLinear.
  const std::string act_op = Ort::ConstNode(activation).GetOperatorType();
  if (act_op != "Relu" && act_op != "Clip") return nullptr;
  const OrtValueInfo* act_out = GetSingleOutput(ort_api, activation);
  if (act_out == nullptr) return nullptr;
  const OrtNode* q_node = nullptr;
  if (!GetSoleNonOutputConsumer(ort_api, act_out, q_node)) return nullptr;
  if (Ort::ConstNode(q_node).GetOperatorType() != "QuantizeLinear") return nullptr;

  // 3a.3 Only fold when Q's encoding cannot represent values outside the activation range.
  return IsActivationEncodingSafe(graph, ort_api, activation, act_op, q_node) ? activation : nullptr;
}

// 3b.0.0 Reject block-quantized weight (DQ scale is rank-2). BQ Gemm is handled by a
//         separate path in the builder that emits FC → FP16 → Quantize; composing it with
//         absorbed-Reshape isn't supported, so gate at the selector to keep the pre-existing
//         BQ path + standalone Reshape (both QNN-supported).
bool IsGemmWeightBlockQuantized(const OrtApi& ort_api, const OrtValueInfo* weight_vi) {
  const OrtNode* dq = nullptr;
  if (ort_api.ValueInfo_GetValueProducer(weight_vi, &dq, nullptr) != nullptr || dq == nullptr) return false;
  if (Ort::ConstNode(dq).GetOperatorType() != "DequantizeLinear") return false;
  size_t dq_num_inputs = 0;
  if (ort_api.Node_GetNumInputs(dq, &dq_num_inputs) != nullptr || dq_num_inputs < 2) return false;
  std::vector<const OrtValueInfo*> dq_inputs(dq_num_inputs);
  if (ort_api.Node_GetInputs(dq, dq_inputs.data(), dq_inputs.size()) != nullptr) return false;
  std::vector<int64_t> scale_shape;
  return GetValueInfoShape(ort_api, dq_inputs[1], scale_shape) && scale_shape.size() == 2;
}

// 3b.0 Guard for the absorb-Reshape path. Reject configurations that require another builder path:
// transposed B, non-FC bias shapes, NATIVE bias, and BQ weight.
bool IsGemmSafeForAbsorbedReshape(const OrtApi& ort_api, const OrtNode* gemm_node) {
  OrtNodeAttrHelper attrs(*gemm_node);
  if (attrs.Get("transB", static_cast<int64_t>(0)) != 0) return false;

  size_t num_inputs = 0;
  if (ort_api.Node_GetNumInputs(gemm_node, &num_inputs) != nullptr) return false;

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  if (ort_api.Node_GetInputs(gemm_node, inputs.data(), inputs.size()) != nullptr) return false;
  if (num_inputs < 2 || inputs[1] == nullptr || IsGemmWeightBlockQuantized(ort_api, inputs[1])) return false;

  if (num_inputs < 3) return true;  // No bias requires no separate Add.

  const OrtValueInfo* bias_vi = inputs[2];
  if (bias_vi == nullptr) return true;

  const float beta = attrs.Get("beta", 1.0f);
  if (beta == 0.0f) return true;

  // 3b.0.1 Shape gate: scalar C for N=1, C=[N], or C=[1,N] can be passed to FullyConnected.
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> bias_shape;
  if (!GetValueInfoShape(ort_api, inputs[1], weight_shape) ||
      !GetValueInfoShape(ort_api, bias_vi, bias_shape) ||
      weight_shape.size() != 2) {
    return false;
  }
  for (int64_t dim : weight_shape) {
    if (dim < 0) return false;
  }
  for (int64_t dim : bias_shape) {
    if (dim < 0) return false;
  }
  if (!qnn::utils::IsCompatibleFcBiasShape(bias_shape, weight_shape[1])) return false;

  // 3b.0.2 NATIVE-bias gate: walk through a DQ (if any) to the true producer. If that
  //         producer is another op, the bias is NATIVE (intermediate) and the builder
  //         would require a separate Add.
  const OrtNode* bias_producer = nullptr;
  if (ort_api.ValueInfo_GetValueProducer(bias_vi, &bias_producer, nullptr) != nullptr) return false;
  if (bias_producer == nullptr) return true;  // graph input / initializer

  if (Ort::ConstNode(bias_producer).GetOperatorType() != "DequantizeLinear") {
    return false;  // Bias directly produced by another op.
  }
  size_t dq_num_inputs = 0;
  if (ort_api.Node_GetNumInputs(bias_producer, &dq_num_inputs) != nullptr || dq_num_inputs == 0) return false;
  std::vector<const OrtValueInfo*> dq_inputs(dq_num_inputs);
  if (ort_api.Node_GetInputs(bias_producer, dq_inputs.data(), dq_inputs.size()) != nullptr) return false;
  const OrtNode* upstream = nullptr;
  if (ort_api.ValueInfo_GetValueProducer(dq_inputs[0], &upstream, nullptr) != nullptr) return false;
  return upstream == nullptr;  // DQ's own input must originate at a graph input / initializer.
}

// 3b. Detect  Gemm -> Reshape -> [Relu/Clip ->] Q  (MatMulAddFusion sandwich) and, if
//     safe, absorb the Reshape (+ optional activation) into the Gemm's QDQ unit.
//
// "Absorbed" means: the Reshape becomes part of the Gemm's QDQ group instead of a
// standalone QNN op, and `gemm_op_builder` emits the pair as FC (rank-2, encoded) +
// QNN Reshape (rank-N, same encoding).
//
// Reachability: ORT's L1 `QDQPropagationTransformer` propagates Q backwards across
// Reshape (but not Relu/Clip), so `Gemm -> [Reshape] -> Q` is rewritten to
// `Gemm -> Q -> DQ -> [Reshape] -> Q` before QNN EP runs — only the Relu/Clip chain
// (3b.3.2) reaches this code today;
// 3b.3.1 is a defensive fallback for the `QDQPropagationTransformer`.
bool TryAbsorbTrailingReshape(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* gemm_node,
                              const OrtNode*& absorbed_reshape, const OrtNode*& folded_activation) {
  // 3b.1 Op-type and safety gate on the Gemm attrs/inputs.
  if (Ort::ConstNode(gemm_node).GetOperatorType() != "Gemm") return false;
  if (!IsGemmSafeForAbsorbedReshape(ort_api, gemm_node)) return false;

  // 3b.2 Gemm -> single-consumer output feeding a Reshape.
  const OrtValueInfo* gemm_out = GetSingleOutput(ort_api, gemm_node);
  if (gemm_out == nullptr) return false;
  const OrtNode* reshape_node = nullptr;
  if (!GetSoleNonOutputConsumer(ort_api, gemm_out, reshape_node)) return false;
  if (Ort::ConstNode(reshape_node).GetOperatorType() != "Reshape") return false;

  // 3b.3 Reshape -> single-consumer output feeding Q or Relu/Clip.
  const OrtValueInfo* reshape_out = GetSingleOutput(ort_api, reshape_node);
  if (reshape_out == nullptr) return false;
  const OrtNode* next_node = nullptr;
  if (!GetSoleNonOutputConsumer(ort_api, reshape_out, next_node)) return false;

  const std::string next_op = Ort::ConstNode(next_node).GetOperatorType();

  // 3b.3.1 Reshape -> Q: unconditional absorb.
  if (next_op == "QuantizeLinear") {
    absorbed_reshape = reshape_node;
    return true;
  }
  if (next_op != "Relu" && next_op != "Clip") return false;

  // 3b.3.2 Reshape -> Relu/Clip -> Q: run the encoding-safe fold.
  const OrtValueInfo* act_out = GetSingleOutput(ort_api, next_node);
  if (act_out == nullptr) return false;
  const OrtNode* q_node = nullptr;
  if (!GetSoleNonOutputConsumer(ort_api, act_out, q_node)) return false;
  if (Ort::ConstNode(q_node).GetOperatorType() != "QuantizeLinear") return false;
  if (!IsActivationEncodingSafe(graph, ort_api, next_node, next_op, q_node)) return false;

  absorbed_reshape = reshape_node;
  folded_activation = next_node;
  return true;
}

// Helper function to get a constant initializer from a node's input
const OrtValue* GetConstantInitializer(const OrtGraph* graph, const OrtApi& ort_api, const char* name) {
  const OrtValue* initializer = nullptr;

  // Get all initializers in the graph
  size_t num_initializers = 0;
  OrtStatus* status = ort_api.Graph_GetNumInitializers(graph, &num_initializers);
  if (status == nullptr) {
    std::vector<const OrtValueInfo*> initializers(num_initializers);
    status = ort_api.Graph_GetInitializers(graph, initializers.data(), num_initializers);
    if (status == nullptr) {
      // Find the initializer with the given name
      for (size_t i = 0; i < num_initializers; ++i) {
        const OrtValueInfo* value_info = initializers[i];
        const char* initializer_name = nullptr;
        status = ort_api.GetValueInfoName(value_info, &initializer_name);
        if (status == nullptr && strcmp(initializer_name, name) == 0) {
          // Found the initializer, get its value
          status = ort_api.ValueInfo_GetInitializerValue(value_info, &initializer);
          if (status == nullptr) {
            break;
          }
        }
        if (status != nullptr) {
          ort_api.ReleaseStatus(status);
          status = nullptr;
        }
      }
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
      }
    } else {
      ort_api.ReleaseStatus(status);
    }
  }

  return initializer;
}

// Helper function to check if a Q or DQ node's scale is a positive constant scalar
bool IsQOrDQScalePositiveConstantScalar(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* q_node) {
  // Get the scale input (index 1) of the Q/DQ node
  size_t num_inputs = 0;
  OrtStatus* status = nullptr;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetNumInputs(q_node, &num_inputs), ort_api);
  if (num_inputs < 2) {
    return false;
  }

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetInputs(q_node, inputs.data(), inputs.size()), ort_api);

  // Get the scale input name
  const OrtValueInfo* scale_value_info = inputs[1];
  const char* scale_name = nullptr;
  // Use the correct API function to get the name of a value info
  ORT_RETURN_FALSE_ON_ERROR(ort_api.GetValueInfoName(scale_value_info, &scale_name), ort_api);

  // Get the scale initializer
  const OrtValue* scale_initializer = GetConstantInitializer(graph, ort_api, scale_name);
  if (scale_initializer == nullptr) {
    return false;
  }

  // Check if the scale is a scalar
  OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.GetTensorTypeAndShape(scale_initializer, &tensor_info), ort_api);

  size_t num_dims = 0;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.GetDimensionsCount(tensor_info, &num_dims), ort_api);
  if (num_dims != 0) {  // Scalar has 0 dimensions
    return false;
  }

  // Check if the scale is positive
  ONNXTensorElementDataType element_type;
  status = ort_api.GetTensorElementType(tensor_info, &element_type);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);
    return false;
  }

  ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);

  // Check the value based on the data type
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    float* scale_data = nullptr;
    ORT_RETURN_FALSE_ON_ERROR(ort_api.GetTensorMutableData(const_cast<OrtValue*>(scale_initializer), (void**)&scale_data), ort_api);
    return *scale_data > 0.0f;
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    double* scale_data = nullptr;
    ORT_RETURN_FALSE_ON_ERROR(ort_api.GetTensorMutableData(const_cast<OrtValue*>(scale_initializer), (void**)&scale_data), ort_api);
    return *scale_data > 0.0;
  }

  return false;
}

// Helper function to check if a node group can be created
bool CanCreateNodeGroup(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                        const OrtNode* redundant_clip_node,
                        const std::vector<const OrtNode*>& dq_nodes,
                        const std::vector<const OrtNode*>& q_nodes) {
  // Avoid unused parameter warnings
  ORT_UNUSED_PARAMETER(redundant_clip_node);
  ORT_UNUSED_PARAMETER(graph);

  if (dq_nodes.empty()) {
    return false;
  }

  // Check if the number of DQ inputs matches the number of inputs that exist
  size_t num_inputs = 0;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetNumInputs(node, &num_inputs), ort_api);
  if (num_inputs < dq_nodes.size()) {
    return false;
  }

  // Check if Q nodes are allowed to be empty
  if (q_nodes.empty()) {
    return false;
  }

  // Check if the number of Q outputs matches the number of outputs that exist
  size_t num_outputs = 0;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetNumOutputs(node, &num_outputs), ort_api);
  if (num_outputs < q_nodes.size()) {
    return false;
  }

  std::vector<const OrtValueInfo*> outputs(num_outputs);
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetOutputs(node, outputs.data(), outputs.size()), ort_api);

  // Check if any of the outputs are graph outputs
  bool produces_graph_output = false;
  for (size_t i = 0; i < num_outputs; i++) {
    const OrtValueInfo* value_info = outputs[i];
    bool is_graph_output = false;
    ORT_CONTINUE_ON_ERROR(ort_api.ValueInfo_IsGraphOutput(value_info, &is_graph_output), ort_api);

    if (is_graph_output) {
      produces_graph_output = true;
      break;
    }
  }

  // Count the total number of consumers for all outputs
  size_t total_consumers = 0;
  for (size_t i = 0; i < num_outputs; i++) {
    const OrtValueInfo* value_info = outputs[i];
    size_t num_consumers = 0;
    ORT_CONTINUE_ON_ERROR(ort_api.ValueInfo_GetValueNumConsumers(value_info, &num_consumers), ort_api);

    total_consumers += num_consumers;
  }

  return (num_outputs == q_nodes.size()) &&
         (q_nodes.size() == total_consumers) &&
         !produces_graph_output;
}

// Helper function to check if a QDQ pair is supported
bool IsQDQPairSupported(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* q_node, const OrtNode* dq_node) {
  // Check if both nodes have the same scale
  size_t q_num_inputs = 0;
  OrtStatus* status = nullptr;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetNumInputs(q_node, &q_num_inputs), ort_api);
  if (q_num_inputs < 2) {
    return false;
  }

  std::vector<const OrtValueInfo*> q_inputs(q_num_inputs);
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetInputs(q_node, q_inputs.data(), q_inputs.size()), ort_api);

  size_t dq_num_inputs = 0;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetNumInputs(dq_node, &dq_num_inputs), ort_api);
  if (dq_num_inputs < 2) {
    return false;
  }

  std::vector<const OrtValueInfo*> dq_inputs(dq_num_inputs);
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetInputs(dq_node, dq_inputs.data(), dq_inputs.size()), ort_api);

  // Get the scale input names
  const OrtValueInfo* q_scale_value_info = q_inputs[1];
  const OrtValueInfo* dq_scale_value_info = dq_inputs[1];

  const char* q_scale_name = nullptr;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.GetValueInfoName(q_scale_value_info, &q_scale_name), ort_api);

  const char* dq_scale_name = nullptr;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.GetValueInfoName(dq_scale_value_info, &dq_scale_name), ort_api);

  // Check if the scale names are the same (indicating they're the same initializer)
  bool same_scale = (strcmp(q_scale_name, dq_scale_name) == 0);

  // If the scales are different, check if they have the same value
  if (!same_scale) {
    const OrtValue* q_scale_initializer = GetConstantInitializer(graph, ort_api, q_scale_name);
    const OrtValue* dq_scale_initializer = GetConstantInitializer(graph, ort_api, dq_scale_name);

    if (q_scale_initializer == nullptr || dq_scale_initializer == nullptr) {
      return false;
    }

    // Check if both scales have the same data type and shape
    OrtTensorTypeAndShapeInfo* q_tensor_info = nullptr;
    ORT_RETURN_FALSE_ON_ERROR(ort_api.GetTensorTypeAndShape(q_scale_initializer, &q_tensor_info), ort_api);

    OrtTensorTypeAndShapeInfo* dq_tensor_info = nullptr;
    status = ort_api.GetTensorTypeAndShape(dq_scale_initializer, &dq_tensor_info);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
      return false;
    }

    ONNXTensorElementDataType q_element_type, dq_element_type;
    status = ort_api.GetTensorElementType(q_tensor_info, &q_element_type);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
      return false;
    }

    status = ort_api.GetTensorElementType(dq_tensor_info, &dq_element_type);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
      return false;
    }

    if (q_element_type != dq_element_type) {
      ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
      return false;
    }

    // Compare the scale values
    if (q_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      float* q_scale_data = nullptr;
      float* dq_scale_data = nullptr;
      status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(q_scale_initializer), (void**)&q_scale_data);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
        ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
        return false;
      }

      status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(dq_scale_initializer), (void**)&dq_scale_data);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
        ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
        return false;
      }

      same_scale = (*q_scale_data == *dq_scale_data);
    } else if (q_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      double* q_scale_data = nullptr;
      double* dq_scale_data = nullptr;
      status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(q_scale_initializer), (void**)&q_scale_data);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
        ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
        return false;
      }

      status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(dq_scale_initializer), (void**)&dq_scale_data);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
        ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
        return false;
      }

      same_scale = (*q_scale_data == *dq_scale_data);
    }

    ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
    ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
  }

  return same_scale;
}

}  // namespace

bool OrtNodeGroupSelector::CheckQDQNodes(const OrtGraph* /*graph*/, const OrtApi& ort_api, const OrtNode* node,
                                         const OrtNode* /*redundant_clip_node*/,
                                         const std::vector<const OrtNode*>& dq_nodes,
                                         const std::vector<const OrtNode*>& q_nodes,
                                         int num_dq_inputs,
                                         bool is_empty_q_nodes_allowed) const {
  if (num_dq_inputs == -1) {
    size_t num_inputs = 0;
    ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetNumInputs(node, &num_inputs), ort_api);
    num_dq_inputs = static_cast<int>(num_inputs);
  }

  // Check if the number of DQ inputs matches the expected number
  if (num_dq_inputs != static_cast<int>(dq_nodes.size())) {
    return false;
  }

  // Check if Q nodes are allowed to be empty
  if (q_nodes.empty()) {
    return is_empty_q_nodes_allowed;
  }

  // Check if the number of Q outputs matches the number of outputs that exist
  size_t num_outputs = 0;
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetNumOutputs(node, &num_outputs), ort_api);

  std::vector<const OrtValueInfo*> outputs(num_outputs);
  ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetOutputs(node, outputs.data(), outputs.size()), ort_api);

  // Check if any of the outputs are graph outputs
  bool produces_graph_output = false;

  for (size_t i = 0; i < num_outputs; i++) {
    const OrtValueInfo* value_info = outputs[i];
    bool is_graph_output = false;
    ORT_CONTINUE_ON_ERROR(ort_api.ValueInfo_IsGraphOutput(value_info, &is_graph_output), ort_api);

    if (is_graph_output) {
      produces_graph_output = true;
      break;
    }
  }

  // Count the total number of consumers for all outputs
  size_t total_consumers = 0;
  for (size_t i = 0; i < num_outputs; i++) {
    const OrtValueInfo* value_info = outputs[i];
    size_t num_consumers = 0;
    ORT_CONTINUE_ON_ERROR(ort_api.ValueInfo_GetValueNumConsumers(value_info, &num_consumers), ort_api);

    total_consumers += num_consumers;
  }

  return (num_outputs == q_nodes.size()) &&
         (q_nodes.size() == total_consumers) &&
         !produces_graph_output;
}

bool OrtDropQDQNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                        const OrtNode* redundant_clip_node,
                                        const std::vector<const OrtNode*>& dq_nodes,
                                        const std::vector<const OrtNode*>& q_nodes) const {
  if (redundant_clip_node) {
    return false;
  }

  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dt_input.value() != dt_output.value()) {
    return false;
  }

  const OrtNode* dq_node = dq_nodes.front();
  const OrtNode* q_node = q_nodes.front();

  if (!allow_nonpositive_scale_) {
    // Check if the Q node's scale is a positive constant scalar
    if (!IsQOrDQScalePositiveConstantScalar(graph, ort_api, q_node)) {
      return false;
    }
  }

  // Check if the QDQ pair is supported (same scale)
  return IsQDQPairSupported(graph, ort_api, q_node, dq_node);
}

// Implementation of Check() for OrtDropDQNodeGroupSelector
bool OrtDropDQNodeGroupSelector::Check(const OrtGraph* /*graph*/, const OrtApi& ort_api, const OrtNode* /*node*/,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& /*q_nodes*/) const {
  // For drop DQ operations, we check if the node has exactly one DQ input
  if (redundant_clip_node) {
    return false;
  }

  constexpr int num_dq_inputs = 1;
  if (num_dq_inputs != static_cast<int>(dq_nodes.size())) {
    return false;
  }

  // Check if the DQ input has the expected data type
  const OrtNode* dq_node = dq_nodes.front();
  auto dt_input = GetNodeInputDataType(dq_node, ort_api, 0);

  if (!dt_input.has_value()) {
    return false;
  }

  return true;
}

// Implementation of Check() for OrtUnaryNodeGroupSelector
bool OrtUnaryNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                      const OrtNode* redundant_clip_node,
                                      const std::vector<const OrtNode*>& dq_nodes,
                                      const std::vector<const OrtNode*>& q_nodes) const {
  // For unary operations, we check if the node has exactly one DQ input and one Q output
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  // Check if the input and output data types match
  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dt_input.value() != dt_output.value()) {
    return false;
  }

  return true;
}

bool OrtClipNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                     const OrtNode* redundant_clip_node,
                                     const std::vector<const OrtNode*>& dq_nodes,
                                     const std::vector<const OrtNode*>& q_nodes) const {
  // Clip can have 1, 2, or 3 DQ inputs:
  // - 1 DQ: only data input is quantized
  // - 2 DQ: data and min or max are quantized
  // - 3 DQ: data, min, and max are all quantized
  const size_t num_dq_nodes = dq_nodes.size();
  if (num_dq_nodes < 1 || num_dq_nodes > 3) {
    return false;
  }

  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, static_cast<int>(num_dq_nodes))) {
    return false;
  }

  // If Clip feeds a Q node, require the data input[0] to come from a DQ node.
  // DQ -> Clip -> Q can form Clip ORT Unit, but DQ -> Op -> Clip -> Q is not allowed as Clip here is redundant.
  if (!q_nodes.empty()) {
    // 1. get num of inputs
    size_t clip_input_count = 0;
    ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetNumInputs(node, &clip_input_count), ort_api);

    // 2. get inputs as OrtValueInfo instances
    std::vector<const OrtValueInfo*> clip_inputs(clip_input_count);
    ORT_RETURN_FALSE_ON_ERROR(ort_api.Node_GetInputs(node, clip_inputs.data(), clip_inputs.size()), ort_api);

    // 3. get the producer/parent of the Clip first input
    const OrtNode* data_producer = nullptr;
    ORT_RETURN_FALSE_ON_ERROR(ort_api.ValueInfo_GetValueProducer(clip_inputs[0], &data_producer, nullptr), ort_api);

    // 4. check if the Clip first input producer is a DQ node
    if (data_producer == nullptr || Ort::ConstNode(data_producer).GetOperatorType() != "DequantizeLinear") {
      return false;
    }

    // 5. check if DQ node in the same group
    if (std::find(dq_nodes.begin(), dq_nodes.end(), data_producer) == dq_nodes.end()) {
      return false;
    }
  }

  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dt_input.value() != dt_output.value()) {
    return false;
  }

  return true;
}

// Implementation of Check() for OrtBinaryNodeGroupSelector
bool OrtBinaryNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& q_nodes) const {
  // For binary operations, we check if the node has exactly two DQ inputs and one Q output
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 2)) {
    return false;
  }

  // Check if the input and output data types match
  auto dt_input_1 = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_input_2 = GetNodeInputDataType(dq_nodes[1], ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input_1.has_value() || !dt_input_2.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dt_input_1.value() != dt_input_2.value() || dt_input_1.value() != dt_output.value()) {
    return false;
  }

  return true;
}

// Implementation of Check() for OrtVariadicNodeGroupSelector
bool OrtVariadicNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                         const OrtNode* redundant_clip_node,
                                         const std::vector<const OrtNode*>& dq_nodes,
                                         const std::vector<const OrtNode*>& q_nodes) const {
  // For variadic operations, we check if the node has at least one DQ input and one Q output
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes)) {
    return false;
  }

  // Check if all DQ inputs have the same data type
  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  if (!dt_input.has_value()) {
    return false;
  }
  for (size_t i = 1; i < dq_nodes.size(); ++i) {
    auto dt_i = GetNodeInputDataType(dq_nodes[i], ort_api, 0);
    if (!dt_i.has_value() || dt_input.value() != dt_i.value()) {
      return false;
    }
  }

  // Check if all Q outputs have the same data type
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);
  if (!dt_output.has_value()) {
    return false;
  }
  for (size_t i = 1; i < q_nodes.size(); ++i) {
    auto dt_o = GetNodeOutputDataType(q_nodes[i], ort_api, 0);
    if (!dt_o.has_value() || dt_output.value() != dt_o.value()) {
      return false;
    }
  }

  if (dt_input.value() != dt_output.value()) {
    return false;
  }

  return true;
}

bool OrtSplitNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                      const OrtNode* redundant_clip_node,
                                      const std::vector<const OrtNode*>& dq_nodes,
                                      const std::vector<const OrtNode*>& q_nodes) const {
  if (redundant_clip_node) {
    return false;
  }

  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  const OrtNode* dq_node = dq_nodes.front();
  auto dt_input = GetNodeInputDataType(dq_node, ort_api, 0);

  if (!dt_input.has_value()) {
    return false;
  }

  // All Q outputs should have same data type and (optionally) equal quantization parameters as the input.
  for (size_t q_idx = 0; q_idx < q_nodes.size(); q_idx++) {
    const OrtNode* q_node = q_nodes[q_idx];

    auto dt_output = GetNodeOutputDataType(q_node, ort_api, 0);
    if (!dt_output.has_value() || dt_input.value() != dt_output.value()) {
      return false;
    }

    if (req_equal_quant_params_ && !IsQDQPairSupported(graph, ort_api, q_node, dq_node)) {
      return false;
    }
  }

  return true;
}

bool OrtConvNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                     const OrtNode* redundant_clip_node,
                                     const std::vector<const OrtNode*>& dq_nodes,
                                     const std::vector<const OrtNode*>& q_nodes) const {
  // Conv allows the bias (input[2]) to lack a DQ node; inputs[0] (data) and inputs[1] (weight)
  // must always be DQ-produced. Unlike ORT-core ConvNodeGroupSelector (which requires all inputs
  // to be quantized), we relax the count to [2,3] to support a float bias at input[2].
  const size_t num_dq_nodes = dq_nodes.size();
  if (num_dq_nodes < 2 || num_dq_nodes > 3) {
    return false;
  }
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, static_cast<int>(num_dq_nodes))) {
    return false;
  }

  // DQ nodes are positional. Verify explicitly that both inputs[0] (data) and inputs[1] (weight) are DQ-produced.
  {
    size_t num_inputs = 0;
    if (ort_api.Node_GetNumInputs(node, &num_inputs) != nullptr || num_inputs < 2) {
      return false;
    }
    std::vector<const OrtValueInfo*> inputs(num_inputs);
    if (ort_api.Node_GetInputs(node, inputs.data(), inputs.size()) != nullptr) {
      return false;
    }
    for (int slot : {0, 1}) {
      if (inputs[slot] == nullptr) {
        return false;
      }
      const OrtNode* producer = nullptr;
      if (ort_api.ValueInfo_GetValueProducer(inputs[slot], &producer, nullptr) != nullptr) {
        return false;
      }
      if (producer == nullptr ||
          Ort::ConstNode(producer).GetOperatorType() != "DequantizeLinear") {
        return false;  // inputs[0] and inputs[1] must be DQ-produced; only inputs[2] (bias) may be float.
      }
    }
  }

  // Input and output types need to be same
  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_weight = GetNodeInputDataType(dq_nodes[1], ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input.has_value() || !dt_weight.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dt_input.value() != dt_output.value()) {
    return false;
  }

  if (dt_input.value() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 &&
      dt_weight.value() != dt_input.value()) {
    return false;
  }

  if (dq_nodes.size() == 3) {
    // Bias has a DQ node: it must be INT32.
    auto dt_bias = GetNodeInputDataType(dq_nodes[2], ort_api, 0);
    if (!dt_bias.has_value() || dt_bias.value() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      return false;
    }
  }

  return true;
}

bool OrtEinsumNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, /*num_dq_inputs=*/-1,
                     /*is_empty_q_nodes_allowed=*/true)) {
    return false;
  }
  size_t num_dq_inputs = dq_nodes.size();
  for (size_t i = 0; i < num_dq_inputs; ++i) {
    auto dt_input = GetNodeInputDataType(dq_nodes[i], ort_api, 0);

    if (!dt_input.has_value()) {
      return false;
    }
  }

  if (!q_nodes.empty()) {
    auto dt_input0 = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
    auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

    if (!dt_input0.has_value() || !dt_output.has_value()) {
      return false;
    }

    // Check if input and output data types match
    if (dt_input0.value() != dt_output.value()) {
      return false;
    }
  }

  return true;
}

bool OrtReciprocalNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                           const OrtNode* redundant_clip_node,
                                           const std::vector<const OrtNode*>& dq_nodes,
                                           const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, /*num_dq_inputs=*/-1,
                     /*is_empty_q_nodes_allowed=*/true)) {
    return false;
  }
  size_t num_dq_inputs = dq_nodes.size();
  for (size_t i = 0; i < num_dq_inputs; ++i) {
    auto dt_input = GetNodeInputDataType(dq_nodes[i], ort_api, 0);
    if (!dt_input.has_value()) {
      return false;
    }
  }
  if (!q_nodes.empty()) {
    auto dt_input0 = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
    auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);
    if (!dt_input0.has_value() || !dt_output.has_value()) {
      return false;
    }
    if (dt_input0.value() != dt_output.value()) {
      return false;
    }
  }
  return true;
}

bool OrtMatMulNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& q_nodes) const {
  if (dq_nodes.size() != 2) {
    return false;
  }

  // Get input data types
  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_weight = GetNodeInputDataType(dq_nodes[1], ort_api, 0);

  if (!dt_input.has_value() || !dt_weight.has_value()) {
    return false;
  }

  if (dt_input.value() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 &&
      dt_weight.value() != dt_input.value()) {
    return false;
  }

  // Without a trailing Q this would be a MatMulIntegerToFloat, which QNN EP does not build.
  if (q_nodes.empty()) {
    return false;
  }

  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes)) {
    return false;
  }

  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);
  return dt_output.has_value() && dt_input.value() == dt_output.value();
}

bool OrtGemmNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                     const OrtNode* redundant_clip_node,
                                     const std::vector<const OrtNode*>& dq_nodes,
                                     const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, -1 /*num_dq_inputs*/,
                     true /*is_empty_q_nodes_allowed*/)) {
    return false;
  }

  // Check if we have at least 2 DQ nodes (A and B inputs)
  if (dq_nodes.size() < 2) {
    return false;
  }

  // Get input data types for A and B
  auto dt_A = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_B = GetNodeInputDataType(dq_nodes[1], ort_api, 0);

  if (!dt_A.has_value() || !dt_B.has_value()) {
    return false;
  }

  // If A is INT8, B must also be INT8
  if (dt_A.value() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
    if (dt_A.value() != dt_B.value()) {  // if A is signed int, B must be signed int
      return false;
    }
  }

  // If there are Q nodes, check if activation and output have the same type
  if (!q_nodes.empty()) {
    auto dt_Y = GetNodeOutputDataType(q_nodes[0], ort_api, 0);
    if (!dt_Y.has_value() || dt_A.value() != dt_Y.value()) {  // activation and output must be same type
      return false;
    }
  }

  // If there's no bias (less than 3 DQ nodes), we're done
  if (dq_nodes.size() < 3) {
    return true;
  }

  // Check if beta attribute is 1.0 (required for bias)
  OrtNodeAttrHelper attr_helper(*node);
  float beta_value = attr_helper.Get("beta", 0.0f);

  // Beta needs to be 1.0 for bias
  if (beta_value != 1.0f) {
    return false;
  }

  // Check if bias has the correct data type (INT32)
  auto dt_bias = GetNodeInputDataType(dq_nodes[2], ort_api, 0);
  return dt_bias.has_value() && dt_bias.value() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

bool OrtWhereNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                      const OrtNode* redundant_clip_node,
                                      const std::vector<const OrtNode*>& dq_nodes,
                                      const std::vector<const OrtNode*>& q_nodes) const {
  // Where has 1 boolean input and 2 dq inputs
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 2)) {
    return false;
  }

  // Check if all DQ inputs have the same data type
  const auto dt_input_1 = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  const auto dt_input_2 = GetNodeInputDataType(dq_nodes[1], ort_api, 0);

  // Check if all Q outputs have the same data type
  const auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input_1.has_value() || !dt_input_2.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dt_input_1.value() != dt_input_2.value() || dt_input_1.value() != dt_output.value()) {
    return false;
  }

  return true;
}

bool OrtPadNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                    const OrtNode* redundant_clip_node,
                                    const std::vector<const OrtNode*>& dq_nodes,
                                    const std::vector<const OrtNode*>& q_nodes) const {
  // Pad can have 1 or 2 dq input, the optional input constant_value can be quantized or non-quantized.
  // QNN supports data input quantized with constant_value input non-quantized.
  int num_dq_inputs = static_cast<int>(dq_nodes.size());
  // Data input (dq_nodes[0] below) must be quantized; reject 0 here since CheckQDQNodes
  // only checks dq_nodes.size() against num_dq_inputs, which is derived from it.
  if (num_dq_inputs < 1 || num_dq_inputs > 2) {
    return false;
  }

  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, num_dq_inputs)) {
    return false;
  }

  const auto dt_input_1 = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  const auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input_1.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dq_nodes.size() > 1) {
    const auto dt_input_2 = GetNodeInputDataType(dq_nodes[1], ort_api, 0);
    return dt_input_2.has_value() && dt_input_1.value() == dt_input_2.value() && dt_input_1.value() == dt_output.value();
  } else {
    return dt_input_1.value() == dt_output.value();
  }
}

bool OrtInstanceAndLayerNormalizationNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                                              const OrtNode* redundant_clip_node,
                                                              const std::vector<const OrtNode*>& dq_nodes,
                                                              const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes)) {
    return false;
  }

  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input.has_value() || !dt_output.has_value()) {
    return false;
  }

  bool has_bias = false;
  std::optional<int32_t> dt_bias;

  // bias is optional for LayerNorm
  if (dq_nodes.size() > 2) {
    has_bias = true;
    dt_bias = GetNodeInputDataType(dq_nodes[2], ort_api, 0);
    if (!dt_bias.has_value()) {
      return false;
    }
  }

  // Input, output, need to be the same type. The bias is int32.
  // Scale can be different with input for a16w8 case
  return (dt_input.value() == dt_output.value()) &&
         (has_bias ? dt_bias.value() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 : true);
}

bool OrtBatchNormalizationNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                                   const OrtNode* redundant_clip_node,
                                                   const std::vector<const OrtNode*>& dq_nodes,
                                                   const std::vector<const OrtNode*>& q_nodes) const {
  // BatchNormalization has 5 inputs: x, scale, bias, mean, var.
  // Require DQ on x and scale (indices 0,1). mean, var may optionally have DQ.
  const int num_dq_nodes = gsl::narrow_cast<int>(dq_nodes.size());
  if (num_dq_nodes < 2 || num_dq_nodes > 5) {
    return false;
  }

  // No output Q means BN produces a float output and runs in float, so allow the empty-Q group.
  const bool has_float_output = q_nodes.empty();
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, num_dq_nodes,
                     /*is_empty_q_nodes_allowed=*/true)) {
    return false;
  }

  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_scale = GetNodeInputDataType(dq_nodes[1], ort_api, 0);

  if (!dt_input.has_value() || !dt_scale.has_value()) {
    return false;
  }

  // The input/output dtype match only applies when the output is quantized.
  if (!has_float_output) {
    auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);
    if (!dt_output.has_value() || dt_input.value() != dt_output.value()) {
      return false;
    }
  }

  if (dt_input.value() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 &&
      dt_scale.value() != dt_input.value()) {
    return false;
  }

  return true;
}

bool OrtLogicalComparisonNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                                  const OrtNode* redundant_clip_node,
                                                  const std::vector<const OrtNode*>& dq_nodes,
                                                  const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, -1, true)) {
    return false;
  }

  auto dt_input_1 = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_input_2 = GetNodeInputDataType(dq_nodes[1], ort_api, 0);
  return dt_input_1.has_value() && dt_input_2.has_value() && dt_input_1.value() == dt_input_2.value();
}

bool OrtTopKNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                     const OrtNode* redundant_clip_node,
                                     const std::vector<const OrtNode*>& dq_nodes,
                                     const std::vector<const OrtNode*>& q_nodes) const {
  // Not support for now. Need to handle the indices output if we want to support it.
  if (redundant_clip_node) {
    return false;
  }

  constexpr int num_dq_inputs = 1;
  constexpr int num_q_outputs = 1;
  if (num_dq_inputs != gsl::narrow_cast<int>(dq_nodes.size())) {
    return false;
  }

  if (!CanCreateNodeGroup(graph, ort_api, node, nullptr, dq_nodes, q_nodes)) {
    return false;
  }

  if (num_q_outputs != gsl::narrow_cast<int>(q_nodes.size())) {
    return false;
  }

  const OrtNode* dq_node = dq_nodes.front();
  const OrtNode* q_node = q_nodes.front();

  auto dt_input = GetNodeInputDataType(dq_node, ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_node, ort_api, 0);

  if (!dt_input.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dt_input.value() != dt_output.value()) {
    return false;
  }

  // Check if the QDQ pair is supported (same scale)
  return IsQDQPairSupported(graph, ort_api, q_node, dq_node);
}

bool OrtCumSumNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& q_nodes) const {
  // Only the first input has DQ node
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dt_input.value() != dt_output.value()) {
    return false;
  }

  return true;
}

bool OrtScatterElementsNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                                const OrtNode* redundant_clip_node,
                                                const std::vector<const OrtNode*>& dq_nodes,
                                                const std::vector<const OrtNode*>& q_nodes) const {
  // ScatterElements has 1 INT32 input and 2 dq inputs
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 2)) {
    return false;
  }

  const auto dt_input_1 = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  const auto dt_input_2 = GetNodeInputDataType(dq_nodes[1], ort_api, 0);
  const auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input_1.has_value() || !dt_input_2.has_value() || !dt_output.has_value()) {
    return false;
  }

  // All input and output types must match.
  if (dt_input_1.value() != dt_input_2.value() || dt_input_1.value() != dt_output.value()) {
    return false;
  }

  return true;
}

bool OrtRMSNormalizationNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                                 const OrtNode* redundant_clip_node,
                                                 const std::vector<const OrtNode*>& dq_nodes,
                                                 const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes)) {
    return false;
  }

  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input.has_value() || !dt_output.has_value()) {
    return false;
  }

  // input and output need to be the same type.
  return (dt_input.value() == dt_output.value());
}

bool OrtMatMulNBitsNodeGroupSelector::Check(const OrtGraph* graph,
                                            const OrtApi& ort_api,
                                            const OrtNode* node,
                                            const OrtNode* redundant_clip_node,
                                            const std::vector<const OrtNode*>& dq_nodes,
                                            const std::vector<const OrtNode*>& q_nodes) const {
  // MatMulNBits has exactly 1 DQ input (activation).
  // Weight, scales, and zero_points are plain initializer inputs, not DQ nodes.
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, /*num_dq_inputs*/ 1)) {
    return false;
  }

  auto dt_input = GetNodeInputDataType(dq_nodes[0], ort_api, 0);
  auto dt_output = GetNodeOutputDataType(q_nodes[0], ort_api, 0);

  if (!dt_input.has_value() || !dt_output.has_value()) {
    return false;
  }

  if (dt_input.value() != dt_output.value()) {
    return false;
  }

  return true;
}

// =============================================================================
// GetOrtQDQSelection — attempt to form a QDQ node group anchored at `node`.
//
//   1. Collect DQ producers of `node`'s inputs.
//   2. Try to extend the group forward:
//      2a.  node -> Relu/Clip -> Q                    (encoding-safe fold)
//      2b.  Gemm -> Reshape -> [Relu/Clip ->] Q       (MatMulAddFusion sandwich)
//   3. Collect Q consumers off the resolved anchor (clip > reshape > node).
//   4. Delegate the final accept/reject to the op-specific selector.
// =============================================================================
std::optional<OrtNodeGroup> GetOrtQDQSelection(const OrtGraph* graph, const OrtApi& ort_api,
                                               const OrtNode* node, const OrtNodeGroupSelector* selector) {
  // 1. Collect DQ producers feeding `node`.
  std::vector<const OrtNode*> dq_nodes;
  size_t num_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(node, &num_inputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> inputs(num_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(node, inputs.data(), inputs.size()), ort_api, std::nullopt);
  for (size_t i = 0; i < num_inputs; ++i) {
    if (inputs[i] == nullptr) continue;
    const OrtNode* producer = nullptr;
    ORT_CONTINUE_ON_ERROR(ort_api.ValueInfo_GetValueProducer(inputs[i], &producer, nullptr), ort_api);
    if (producer != nullptr && Ort::ConstNode(producer).GetOperatorType() == "DequantizeLinear") {
      dq_nodes.push_back(producer);
    }
  }

  // 2. Extend the group forward.
  size_t output_count = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(node, &output_count), ort_api, std::nullopt);

  // 2a. Redundant Relu/Clip fold (see TryFoldRedundantActivation).
  const OrtNode* clip_node = nullptr;
  if (output_count == 1) {
    clip_node = TryFoldRedundantActivation(graph, ort_api, node);
  }

  // 2b. MatMulAddFusion post-Gemm Reshape absorb (see TryAbsorbTrailingReshape).
  //     Skipped when 2a already folded — the two paths are mutually exclusive.
  const OrtNode* output_reshape_node = nullptr;
  if (clip_node == nullptr && output_count == 1) {
    const OrtNode* folded_activation = nullptr;
    if (TryAbsorbTrailingReshape(graph, ort_api, node, output_reshape_node, folded_activation) &&
        folded_activation != nullptr) {
      clip_node = folded_activation;
    }
  }

  // 3. Collect Q consumers from the extended anchor.
  //    Anchor precedence: clip > absorbed reshape > node.
  const OrtNode* q_anchor = clip_node ? clip_node : (output_reshape_node ? output_reshape_node : node);
  std::vector<const OrtNode*> q_nodes;
  size_t num_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(q_anchor, &num_outputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> outputs(num_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(q_anchor, outputs.data(), outputs.size()), ort_api, std::nullopt);
  for (size_t i = 0; i < num_outputs; ++i) {
    if (outputs[i] == nullptr) continue;
    size_t num_consumers = 0;
    ORT_CONTINUE_ON_ERROR(ort_api.ValueInfo_GetValueNumConsumers(outputs[i], &num_consumers), ort_api);
    if (num_consumers == 0) continue;
    std::vector<const OrtNode*> consumers(num_consumers);
    std::vector<int64_t> input_indices(num_consumers);
    ORT_CONTINUE_ON_ERROR(
        ort_api.ValueInfo_GetValueConsumers(outputs[i], consumers.data(), input_indices.data(), num_consumers),
        ort_api);
    for (const OrtNode* consumer : consumers) {
      if (Ort::ConstNode(consumer).GetOperatorType() == "QuantizeLinear") {
        q_nodes.push_back(consumer);
      }
    }
  }

  // 4. Delegate accept/reject to the op-specific selector.
  if (!selector->Check(graph, ort_api, node, clip_node, dq_nodes, q_nodes)) {
    return std::nullopt;
  }
  OrtNodeGroup node_group;
  node_group.target_node = node;
  node_group.redundant_clip_node = clip_node;
  node_group.output_reshape_node = output_reshape_node;
  node_group.dq_nodes = std::move(dq_nodes);
  node_group.q_nodes = std::move(q_nodes);
  return node_group;
}

// Implementation of OrtSelectorManager constructor and related functions
OrtSelectorManager::OrtSelectorManager() {
  CreateSelectors();
  InitializeSelectorsMap();
}

void OrtSelectorManager::CreateSelectors() {
  // Register selectors for different op types

  // Register misc ops
  OrtOpVersionsAndSelector::OpVersionsMap misc_ops = {
      {"Expand", {}},
      {"Flatten", {}},
      {"Gather", {}},
      {"GatherElements", {}},
      {"MaxPool", {12}},
      {"Reshape", {}},
      {"Resize", {}},
      {"Squeeze", {}},
      {"Tile", {}},
      {"Transpose", {}},
      {"Unsqueeze", {}}};
  ort_selectors_.RegisterSelector(misc_ops, std::make_unique<OrtDropQDQNodeGroupSelector>());

  // Register drop DQ ops
  OrtOpVersionsAndSelector::OpVersionsMap drop_dq_ops = {
      {"ArgMax", {}},
      {"ArgMin", {}},
      {"NonZero", {}}};
  ort_selectors_.RegisterSelector(drop_dq_ops, std::make_unique<OrtDropDQNodeGroupSelector>());

  // Register unary ops
  OrtOpVersionsAndSelector::OpVersionsMap unary_ops = {
      {"Abs", {}},
      {"Asin", {}},
      {"Atan", {}},
      {"AveragePool", {}},
      {"Ceil", {}},
      {"Cos", {}},
      {"DepthToSpace", {}},
      {"Elu", {}},
      {"Erf", {}},
      {"Exp", {}},
      {"Floor", {}},
      {"Gelu", {}},
      {"GlobalAveragePool", {}},
      {"GlobalMaxPool", {}},
      {"HardSigmoid", {}},
      {"HardSwish", {}},
      {"LRN", {}},
      {"LeakyRelu", {}},
      {"Log", {}},
      {"LogSoftmax", {}},
      {"LpNormalization", {}},
      {"Neg", {}},
      {"ReduceL2", {}},
      {"ReduceLogSumExp", {}},
      {"ReduceMax", {}},
      {"ReduceMean", {}},
      {"ReduceMin", {}},
      {"ReduceProd", {}},
      {"ReduceSum", {}},
      {"Relu", {}},
      {"Round", {}},
      {"Sigmoid", {}},
      {"Sign", {}},
      {"Sin", {}},
      {"Slice", {}},
      {"Softmax", {}},
      {"Softplus", {}},
      {"SpaceToDepth", {}},
      {"Sqrt", {}},
      {"Swish", {}},
      {"Tanh", {}}};
  ort_selectors_.RegisterSelector(unary_ops, std::make_unique<OrtUnaryNodeGroupSelector>());

  // Register clip ops
  OrtOpVersionsAndSelector::OpVersionsMap clip_ops = {
      {"Clip", {}}};
  ort_selectors_.RegisterSelector(clip_ops, std::make_unique<OrtClipNodeGroupSelector>());

  // Register binary ops
  OrtOpVersionsAndSelector::OpVersionsMap binary_ops = {
      {"Add", {}},
      {"Div", {}},
      {"GridSample", {}},
      {"Mul", {}},
      {"Pow", {}},
      {"PRelu", {}},
      {"Sub", {}}};
  ort_selectors_.RegisterSelector(binary_ops, std::make_unique<OrtBinaryNodeGroupSelector>());

  // Register variadic ops
  OrtOpVersionsAndSelector::OpVersionsMap variadic_ops = {
      {"Concat", {}},
      {"Max", {}},
      {"Min", {}}};
  ort_selectors_.RegisterSelector(variadic_ops, std::make_unique<OrtVariadicNodeGroupSelector>());

  // Register split ops
  OrtOpVersionsAndSelector::OpVersionsMap split_ops = {
      {"Split", {}}};
  ort_selectors_.RegisterSelector(split_ops, std::make_unique<OrtSplitNodeGroupSelector>());

  // Register conv ops
  OrtOpVersionsAndSelector::OpVersionsMap conv_ops = {
      {"Conv", {}}};
  ort_selectors_.RegisterSelector(conv_ops, std::make_unique<OrtConvNodeGroupSelector>());

  // Register conv transpose ops
  OrtOpVersionsAndSelector::OpVersionsMap conv_transpose_ops = {
      {"ConvTranspose", {}}};
  ort_selectors_.RegisterSelector(conv_transpose_ops, std::make_unique<OrtConvNodeGroupSelector>());

  // Register einsum ops
  OrtOpVersionsAndSelector::OpVersionsMap einsum_ops = {
      {"Einsum", {}}};
  ort_selectors_.RegisterSelector(einsum_ops, std::make_unique<OrtEinsumNodeGroupSelector>());

  // Register reciprocal ops
  OrtOpVersionsAndSelector::OpVersionsMap reciprocal_ops = {
      {"Reciprocal", {}}};
  ort_selectors_.RegisterSelector(reciprocal_ops, std::make_unique<OrtReciprocalNodeGroupSelector>());

  // Register matmul ops
  OrtOpVersionsAndSelector::OpVersionsMap matmul_ops = {
      {"MatMul", {}}};
  ort_selectors_.RegisterSelector(matmul_ops, std::make_unique<OrtMatMulNodeGroupSelector>());

  // Register gemm ops
  OrtOpVersionsAndSelector::OpVersionsMap gemm_ops = {
      {"Gemm", {}}};
  ort_selectors_.RegisterSelector(gemm_ops, std::make_unique<OrtGemmNodeGroupSelector>());

  // Register instance and layer normalization ops
  OrtOpVersionsAndSelector::OpVersionsMap instance_layer_norm_ops = {
      {"InstanceNormalization", {}},
      {"LayerNormalization", {}}};
  ort_selectors_.RegisterSelector(instance_layer_norm_ops, std::make_unique<OrtInstanceAndLayerNormalizationNodeGroupSelector>());

  // Register batch normalization ops
  OrtOpVersionsAndSelector::OpVersionsMap batch_norm_ops = {
      {"BatchNormalization", {}}};
  ort_selectors_.RegisterSelector(batch_norm_ops, std::make_unique<OrtBatchNormalizationNodeGroupSelector>());

  // Register logical comparison ops
  OrtOpVersionsAndSelector::OpVersionsMap logical_comparison_ops = {
      {"Equal", {}},
      {"Greater", {}},
      {"GreaterOrEqual", {}},
      {"Less", {}},
      {"LessOrEqual", {}}};
  ort_selectors_.RegisterSelector(logical_comparison_ops, std::make_unique<OrtLogicalComparisonNodeGroupSelector>());

  // Register where ops
  OrtOpVersionsAndSelector::OpVersionsMap where_ops = {
      {"Where", {}}};
  ort_selectors_.RegisterSelector(where_ops, std::make_unique<OrtWhereNodeGroupSelector>());

  // Register pad ops
  OrtOpVersionsAndSelector::OpVersionsMap pad_ops = {
      {"Pad", {}}};
  ort_selectors_.RegisterSelector(pad_ops, std::make_unique<OrtPadNodeGroupSelector>());

  // Register topk ops
  OrtOpVersionsAndSelector::OpVersionsMap topk_ops = {
      {"TopK", {}}};
  ort_selectors_.RegisterSelector(topk_ops, std::make_unique<OrtTopKNodeGroupSelector>());

  // Register cumsum ops
  OrtOpVersionsAndSelector::OpVersionsMap cumsum_ops = {
      {"CumSum", {}}};
  ort_selectors_.RegisterSelector(cumsum_ops, std::make_unique<OrtCumSumNodeGroupSelector>());

  // Register scatter_elements ops
  OrtOpVersionsAndSelector::OpVersionsMap scatter_elements_ops = {
      {"ScatterElements", {}}};
  ort_selectors_.RegisterSelector(scatter_elements_ops, std::make_unique<OrtScatterElementsNodeGroupSelector>());

  // Register rmsnormalization ops
  OrtOpVersionsAndSelector::OpVersionsMap rmsnorm_ops = {
      {"RMSNormalization", {}},
      {"SimplifiedLayerNormalization", {}}};
  ort_selectors_.RegisterSelector(rmsnorm_ops, std::make_unique<OrtRMSNormalizationNodeGroupSelector>());

  // Register MatMulNBits ops
  OrtOpVersionsAndSelector::OpVersionsMap matmulnbits_ops = {
      {"MatMulNBits", {}}};
  ort_selectors_.RegisterSelector(matmulnbits_ops, std::make_unique<OrtMatMulNBitsNodeGroupSelector>());
}

void OrtSelectorManager::InitializeSelectorsMap() {
  for (const auto& entry : ort_selectors_.SelectorsSet()) {
    for (const auto& op_info : entry->op_versions_map) {
      op_type_to_selectors_map_.insert({op_info.first, &*entry});
    }
  }
}

// Implementation of GetQDQSelections for OrtGraph
std::vector<OrtNodeGroup> OrtSelectorManager::GetOrtQDQSelections(const OrtGraph* graph,
                                                                  const OrtApi& ort_api,
                                                                  const Ort::Logger& logger) const {
  std::vector<OrtNodeGroup> qdq_selections;

  // Get all nodes from the graph
  size_t num_nodes = 0;
  auto status = ort_api.Graph_GetNumNodes(graph, &num_nodes);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return qdq_selections;
  }
  std::vector<const OrtNode*> nodes(num_nodes);
  status = ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return qdq_selections;
  }

  // Process each node
  for (size_t i = 0; i < num_nodes; ++i) {
    const OrtNode* node = nodes[i];

    // Get node op type
    std::string op_type = Ort::ConstNode(node).GetOperatorType();

    // Get node domain
    const char* domain = nullptr;
    ORT_CONTINUE_ON_ERROR(ort_api.Node_GetDomain(node, &domain), ort_api);

    // Check domain (similar to the GraphViewer version)
    std::string domain_str(domain);
    if (domain_str != kOnnxDomain && domain_str != kMSInternalNHWCDomain && domain_str != kMSDomain && domain_str != kMLOnnxDomain) {
      continue;
    }

    // Find selector for this op type
    auto op_rule = op_type_to_selectors_map_.find(op_type);
    if (op_rule == op_type_to_selectors_map_.cend()) {
      continue;
    }

    const auto& op_versions_and_selector = *op_rule->second;

    // Check the supported versions if specified
    const auto& versions = op_versions_and_selector.op_versions_map.find(op_type)->second;
    if (!versions.empty()) {
      // Get node version
      int since_version = 0;
      ORT_CONTINUE_ON_ERROR(ort_api.Node_GetSinceVersion(node, &since_version), ort_api);

      if (std::find(versions.cbegin(), versions.cend(), since_version) == versions.cend()) {
        ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Op version is not supported for " + op_type).c_str());
        continue;
      }
    }

    // Get QDQ selection for this node
    const auto qdq_node_group_selection = GetOrtQDQSelection(graph, ort_api, node, op_versions_and_selector.selector.get());
    if (qdq_node_group_selection.has_value()) {
      const auto& qdq_group = *qdq_node_group_selection;
      qdq_selections.push_back(qdq_group);
    }
  }
  return qdq_selections;
}

}  // namespace QDQ

namespace utils {

// QNN-EP COPY START
// Below implementations are directly copied from "core/common/common.h"
// Returns whether `key` is in `container`.
// Like C++20's map/set contains() member function.
template <typename Key, typename... OtherContainerArgs,
          template <typename...> typename AssociativeContainer,
          typename LookupKey>
inline bool Contains(const AssociativeContainer<Key, OtherContainerArgs...>& container, LookupKey&& key) {
  return container.find(std::forward<LookupKey>(key)) != container.end();
}
// QNN-EP COPY END

std::vector<std::vector<const OrtNode*>> CreateSupportedPartitionNodeGroups(
    const OrtGraph* graph,
    const OrtApi& ort_api,
    const std::vector<const OrtNode*>& supported_nodes,
    const std::string& ep_type,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map) {
  std::vector<std::vector<const OrtNode*>> supported_groups{};

  size_t num_nodes = 0;
  auto status = ort_api.Graph_GetNumNodes(graph, &num_nodes);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return {};
  }
  std::vector<const OrtNode*> graph_nodes(num_nodes);
  status = ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return {};
  }

  // #inputs from unprocessed nodes (in-degree) per node.
  std::unordered_map<size_t, size_t> in_degree{};
  // Nodes that are ready to process.
  std::deque<const OrtNode*> nodes_to_process{};
  // Nodes that will be processed when considering the next partition node group.
  std::deque<const OrtNode*> nodes_to_process_with_next_group{};

  // Initialize in-degrees and find root nodes.
  for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const OrtNode* node = graph_nodes[node_idx];
    const OrtNodeUnit* node_unit = node_unit_map.at(node);

    if (&node_unit->GetNode() != node) {
      // Only process the target node.
      continue;
    }

    size_t degree = node_unit->GetInputEdgesCount(ort_api);
    in_degree.insert({node_unit->Index(), degree});
    if (degree == 0) {
      nodes_to_process.push_back(node);
    }
  }

  std::vector<const OrtNode*> supported_group{};
  // The partition node group's border is the aggregate of its nodes' output nodes.
  InlinedHashSet<const OrtNode*> supported_group_border{};

  auto close_group = [&]() {
    if (!supported_group.empty()) {
      supported_groups.emplace_back(std::move(supported_group));
      supported_group.clear();
      supported_group_border.clear();
    }
  };

  size_t num_nodes_processed = 0;

  while (!nodes_to_process.empty() || !nodes_to_process_with_next_group.empty()) {
    if (nodes_to_process.empty()) {
      // We have processed all the nodes that we can while building this partition node group, start a new one.
      close_group();
      nodes_to_process.swap(nodes_to_process_with_next_group);
      continue;
    }

    const OrtNode* node = nodes_to_process.front();
    nodes_to_process.pop_front();

    const OrtNodeUnit* node_unit = node_unit_map.at(node);
    const bool is_qdq_node_unit = node_unit->UnitType() == OrtNodeUnit::Type::QDQGroup;

    // A node that is already assigned to an EP other than current EP is unsupported.
    const char* node_ep_name;
    ORT_CONTINUE_ON_ERROR(ort_api.Node_GetEpName(node, &node_ep_name), ort_api);
    const bool is_node_supported = ((std::string(node_ep_name).empty() || node_ep_name == ep_type) &&
                                    std::find(supported_nodes.cbegin(), supported_nodes.cend(), node) != supported_nodes.cend());

    if (!is_node_supported && Contains(supported_group_border, node)) {
      // An unsupported node on the border will be processed after the current partition node group.
      nodes_to_process_with_next_group.push_back(node);
      continue;
    }

    if (is_node_supported) {
      if (is_qdq_node_unit) {
        // Add DQ -> node -> Q for the node unit and must be in topological order.
        for (const OrtNode* dq : node_unit->GetDQNodes()) {
          supported_group.push_back(dq);
        }

        supported_group.push_back(node);
        const OrtNode* output_reshape_node = node_unit->GetOutputReshapeNode();
        if (output_reshape_node) {
          supported_group.push_back(output_reshape_node);
          supported_group_border.erase(output_reshape_node);
        }
        const OrtNode* redundent_clip_node = node_unit->GetRedundantClipNode();
        if (redundent_clip_node) {
          supported_group.push_back(redundent_clip_node);
          supported_group_border.erase(redundent_clip_node);
        }

        for (const OrtNode* q : node_unit->GetQNodes()) {
          supported_group.push_back(q);
        }
      } else {
        supported_group.push_back(node);
      }

      // Remove node from the border.
      supported_group_border.erase(node);
    }

    // For each downstream node:
    //   1: Add the downstream node to the border if the current node is supported.
    //   2: Adjust in-degrees of the nodes consuming the current node's outputs, and add any new nodes to process.
    for (const OrtNode* output_node : node_unit->GetOutputNodes(ort_api)) {
      const OrtNodeUnit* downstream_node_unit = node_unit_map.at(output_node);
      const OrtNode* downstream_node = &downstream_node_unit->GetNode();

      if (is_node_supported) {
        supported_group_border.insert(downstream_node);
      }

      auto& downstream_node_in_degree = in_degree[downstream_node_unit->Index()];
      --downstream_node_in_degree;

      if (downstream_node_in_degree == 0) {
        nodes_to_process.push_back(downstream_node);
      }
    }

    ++num_nodes_processed;
  }

  close_group();

  if (num_nodes_processed != in_degree.size()) {
    ORT_CXX_API_THROW("Processed " + std::to_string(num_nodes_processed) +
                          " nodes. Expected to process " + std::to_string(in_degree.size()),
                      ORT_EP_FAIL);
  }

  return supported_groups;
}

}  // namespace utils

// Implementation of GetQDQNodeUnits for OrtGraph
std::pair<std::vector<std::unique_ptr<OrtNodeUnit>>, std::unordered_map<const OrtNode*, const OrtNodeUnit*>>
GetAllOrtNodeUnits(OrtApi ort_api, const OrtGraph* graph, const Ort::Logger& logger) {
  std::vector<std::unique_ptr<OrtNodeUnit>> node_unit_holder;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map;

  // Get all nodes from the graph
  size_t num_nodes = 0;
  auto status = ort_api.Graph_GetNumNodes(graph, &num_nodes);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return std::make_pair(std::move(node_unit_holder), std::move(node_unit_map));
  }
  std::vector<const OrtNode*> nodes(num_nodes);
  status = ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return std::make_pair(std::move(node_unit_holder), std::move(node_unit_map));
  }

  const auto add_node_unit_to_map = [&](const std::vector<const OrtNode*>& _nodes, const OrtNodeUnit* node_unit) {
    for (const OrtNode* node : _nodes) {
      node_unit_map[node] = node_unit;
    }
  };

  // Get QDQ NodeUnits first
  QDQ::OrtSelectorManager selector_mgr;

  const auto qdq_selections = selector_mgr.GetOrtQDQSelections(graph, ort_api, logger);
  for (const auto& qdq_selection : qdq_selections) {
    auto qdq_unit = std::make_unique<OrtNodeUnit>(graph, qdq_selection, ort_api);

    // Fill the node to node_unit map for all nodes in the QDQ Group
    add_node_unit_to_map(qdq_selection.dq_nodes, qdq_unit.get());
    add_node_unit_to_map(qdq_selection.q_nodes, qdq_unit.get());
    add_node_unit_to_map({qdq_selection.target_node}, qdq_unit.get());
    if (qdq_selection.redundant_clip_node) {
      add_node_unit_to_map({qdq_selection.redundant_clip_node}, qdq_unit.get());
    }
    if (qdq_selection.output_reshape_node) {
      add_node_unit_to_map({qdq_selection.output_reshape_node}, qdq_unit.get());
    }

    node_unit_holder.push_back(std::move(qdq_unit));
  }

  // Get the left over single-node OrtNodeUnit.
  for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const OrtNode* node = nodes[node_idx];

    // This is already part of a QDQ OrtNodeUnit.
    if (node_unit_map.find(node) != node_unit_map.cend())
      continue;

    auto node_unit = std::make_unique<OrtNodeUnit>(node, ort_api);
    node_unit_map[node] = node_unit.get();
    node_unit_holder.push_back(std::move(node_unit));
  }

  return std::make_pair(std::move(node_unit_holder), std::move(node_unit_map));
}

}  // namespace onnxruntime
