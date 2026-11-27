// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

#ifdef QNN_QMOE_AVAILABLE
namespace {

// com.microsoft QMoE input indices consumed by the QNN GPU kernels.
constexpr size_t kInputIdx = 0;
constexpr size_t kRouterProbsIdx = 1;
constexpr size_t kFc1WeightsIdx = 2;
constexpr size_t kFc1ScalesIdx = 3;
constexpr size_t kFc1BiasIdx = 4;
constexpr size_t kFc2WeightsIdx = 5;
constexpr size_t kFc2ScalesIdx = 6;
constexpr size_t kFc2BiasIdx = 7;
constexpr size_t kNumConsumedInputs = 8;

// QNN QMoE inputs are positional through router_weights (QNN slot 14); QNN validation rejects a
// truncated input array, so absent optional slots are passed as null tensors. Only the expert
// tensors are reordered: ONNX lists fc1 weights/scales/bias before fc2 weights, QNN lists both
// weight tensors first.
constexpr std::array<size_t, 15> kQnnToOnnxInputIdx = {
    kInputIdx, kRouterProbsIdx, kFc1WeightsIdx, kFc2WeightsIdx, kFc1ScalesIdx, kFc1BiasIdx,
    kFc2ScalesIdx, kFc2BiasIdx, 8, 9, 10, 11, 12, 13, 14};

// Configuration supported by the QNN GPU QMoE kernels (see the GPU QMoE validator).
constexpr int64_t kSupportedExpertWeightBits = 4;
constexpr int64_t kSupportedBlockSize = 32;
constexpr uint32_t kWeightPackFactor = 2;    // Two 4-bit weights per uint8.
constexpr uint32_t kSwigluFusionFactor = 2;  // Fused gate/up projection doubles fc1 rows.
constexpr uint32_t kHiddenSizeAlignment = 64;

Ort::Status GetIODataType(const OrtNodeUnitIODef& io_def, QnnBackendType backend_type,
                          Qnn_DataType_t& data_type) {
  return utils::GetQnnDataType(io_def.quant_param.has_value(), io_def.type, data_type, backend_type);
}

bool IsFloat(Qnn_DataType_t data_type) {
  return data_type == QNN_DATATYPE_FLOAT_32 || data_type == QNN_DATATYPE_FLOAT_16;
}

}  // namespace

class QMoEOpBuilder final : public BaseOpBuilder {
 public:
  QMoEOpBuilder() : BaseOpBuilder("QMoEOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QMoEOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status QMoEOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger) const {
  const QnnBackendType backend_type = qnn_model_wrapper.GetQnnBackendType();
  RETURN_IF_NOT(IsGpuBackend(backend_type), "QMoE: GPU backend only");

  // Attributes. Defaults follow the com.microsoft QMoE schema.
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t k = node_helper.Get("k", static_cast<int64_t>(1));
  RETURN_IF_NOT(node_helper.Get("expert_weight_bits", static_cast<int64_t>(4)) == kSupportedExpertWeightBits,
                "QMoE: only expert_weight_bits == 4 is supported");
  RETURN_IF_NOT(node_helper.Get("block_size", static_cast<int64_t>(0)) == kSupportedBlockSize,
                "QMoE: only block_size == 32 is supported");
  RETURN_IF_NOT(node_helper.Get("activation_type", std::string("relu")) == "swiglu",
                "QMoE: only activation_type == swiglu is supported");
  RETURN_IF_NOT(node_helper.Get("swiglu_fusion", static_cast<int64_t>(0)) ==
                    QNN_OP_Q_MO_E_SWIGLU_FUSION_FUSED_INTERLEAVED,
                "QMoE: only swiglu_fusion == 1 (fused, interleaved) is supported");
  // ONNX treats an absent swiglu_limit as "no clamp", which the QNN GPU kernel cannot express.
  RETURN_IF_NOT(node_helper.HasAttr("swiglu_limit"), "QMoE: swiglu_limit must be set");
  RETURN_IF_NOT(node_helper.Get("use_sparse_mixer", static_cast<int64_t>(0)) == 0,
                "QMoE: use_sparse_mixer is not supported");

  // Inputs. The QNN GPU kernels consume only inputs 0-7; any other optional input (fc3_*, zero
  // points, router_weights, global/activation scales) would otherwise be silently dropped.
  const auto& inputs = node_unit.Inputs();
  RETURN_IF_NOT(inputs.size() > kFc2ScalesIdx, "QMoE: expects at least 7 inputs");
  for (size_t idx : {kInputIdx, kRouterProbsIdx, kFc1WeightsIdx, kFc1ScalesIdx, kFc2WeightsIdx, kFc2ScalesIdx}) {
    RETURN_IF_NOT(inputs[idx].Exists(), ("QMoE: input " + std::to_string(idx) + " is required").c_str());
  }
  for (size_t idx = kNumConsumedInputs; idx < inputs.size(); ++idx) {
    RETURN_IF(inputs[idx].Exists(), ("QMoE: optional input " + std::to_string(idx) + " is not supported").c_str());
  }
  const bool has_fc1_bias = inputs[kFc1BiasIdx].Exists();
  const bool has_fc2_bias = inputs.size() > kFc2BiasIdx && inputs[kFc2BiasIdx].Exists();
  RETURN_IF(has_fc1_bias != has_fc2_bias,
            "QMoE: fc1_experts_bias and fc2_experts_bias must both be present or both be absent");

  // Data types.
  Qnn_DataType_t input_type{};
  RETURN_IF_ERROR(GetIODataType(inputs[kInputIdx], backend_type, input_type));
  RETURN_IF_NOT(IsFloat(input_type), "QMoE: input must be float32 or float16");
  for (size_t idx : {kRouterProbsIdx, kFc1BiasIdx, kFc2BiasIdx}) {
    if (!inputs[idx].Exists()) {
      continue;
    }
    Qnn_DataType_t data_type{};
    RETURN_IF_ERROR(GetIODataType(inputs[idx], backend_type, data_type));
    RETURN_IF_NOT(data_type == input_type,
                  ("QMoE: input " + std::to_string(idx) + " must have the same data type as input 0").c_str());
  }
  for (size_t idx : {kFc1WeightsIdx, kFc2WeightsIdx}) {
    Qnn_DataType_t data_type{};
    RETURN_IF_ERROR(GetIODataType(inputs[idx], backend_type, data_type));
    RETURN_IF_NOT(data_type == QNN_DATATYPE_UINT_8, "QMoE: expert weights must be packed uint8");
  }
  for (size_t idx : {kFc1ScalesIdx, kFc2ScalesIdx}) {
    Qnn_DataType_t data_type{};
    RETURN_IF_ERROR(GetIODataType(inputs[idx], backend_type, data_type));
    RETURN_IF_NOT(IsFloat(data_type), "QMoE: expert scales must be float32 or float16");
  }

  // Shapes.
  TensorInfo input_info{};
  TensorInfo router_info{};
  TensorInfo fc1_weights_info{};
  TensorInfo fc1_scales_info{};
  TensorInfo fc2_weights_info{};
  TensorInfo fc2_scales_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kInputIdx], input_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kRouterProbsIdx], router_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kFc1WeightsIdx], fc1_weights_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kFc1ScalesIdx], fc1_scales_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kFc2WeightsIdx], fc2_weights_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kFc2ScalesIdx], fc2_scales_info));

  RETURN_IF_NOT(input_info.shape.size() == 3,
                "QMoE: input must be 3D (batch_size, sequence_length, hidden_size)");
  const uint32_t num_tokens = input_info.shape[0] * input_info.shape[1];
  const uint32_t hidden_size = input_info.shape[2];
  RETURN_IF_NOT(num_tokens > 0, "QMoE: num_tokens must be greater than 0");
  RETURN_IF_NOT(hidden_size > 0 && hidden_size % kHiddenSizeAlignment == 0,
                "QMoE: hidden_size must be a positive multiple of 64");

  RETURN_IF_NOT(router_info.shape.size() == 2 && router_info.shape[0] == num_tokens,
                "QMoE: router_probs must have shape (num_tokens, num_experts)");
  const uint32_t num_experts = router_info.shape[1];
  RETURN_IF_NOT(num_experts > 1, "QMoE: num_experts must be greater than 1");
  RETURN_IF_NOT(k >= 1 && k <= static_cast<int64_t>(num_experts), "QMoE: k must be in [1, num_experts]");

  RETURN_IF_NOT(fc1_weights_info.shape.size() == 3 && fc1_weights_info.shape[1] % kSwigluFusionFactor == 0,
                "QMoE: fc1_experts_weights must be 3D with an even fused intermediate dimension");
  const uint32_t fused_inter_size = fc1_weights_info.shape[1];
  const uint32_t inter_size = fused_inter_size / kSwigluFusionFactor;
  const uint32_t block_size = static_cast<uint32_t>(kSupportedBlockSize);
  RETURN_IF_NOT(inter_size > 0 && inter_size % block_size == 0,
                "QMoE: inter_size must be a positive multiple of block_size");

  RETURN_IF_NOT(fc1_weights_info.shape ==
                    std::vector<uint32_t>({num_experts, fused_inter_size, hidden_size / kWeightPackFactor}),
                "QMoE: fc1_experts_weights must have shape (num_experts, 2 * inter_size, hidden_size / 2)");
  RETURN_IF_NOT(fc2_weights_info.shape ==
                    std::vector<uint32_t>({num_experts, hidden_size, inter_size / kWeightPackFactor}),
                "QMoE: fc2_experts_weights must have shape (num_experts, hidden_size, inter_size / 2)");
  RETURN_IF_NOT(fc1_scales_info.shape ==
                    std::vector<uint32_t>({num_experts, fused_inter_size, hidden_size / block_size}),
                "QMoE: fc1_scales must have shape (num_experts, 2 * inter_size, hidden_size / block_size)");
  RETURN_IF_NOT(fc2_scales_info.shape ==
                    std::vector<uint32_t>({num_experts, hidden_size, inter_size / block_size}),
                "QMoE: fc2_scales must have shape (num_experts, hidden_size, inter_size / block_size)");
  if (has_fc1_bias) {
    TensorInfo fc1_bias_info{};
    TensorInfo fc2_bias_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kFc1BiasIdx], fc1_bias_info));
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[kFc2BiasIdx], fc2_bias_info));
    RETURN_IF_NOT(fc1_bias_info.shape == std::vector<uint32_t>({num_experts, fused_inter_size}),
                  "QMoE: fc1_experts_bias must have shape (num_experts, 2 * inter_size)");
    RETURN_IF_NOT(fc2_bias_info.shape == std::vector<uint32_t>({num_experts, hidden_size}),
                  "QMoE: fc2_experts_bias must have shape (num_experts, hidden_size)");
  }

  // Device- and memory-dependent limits (batched path on 8xx-tier GPUs only, route-table size,
  // buffer sizes) are left to QNN backend validation.
  return BaseOpBuilder::IsOpSupported(qnn_model_wrapper, node_unit, logger);
}

Ort::Status QMoEOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();
  for (size_t onnx_idx : kQnnToOnnxInputIdx) {
    if (onnx_idx >= inputs.size() || !inputs[onnx_idx].Exists()) {
      input_names.emplace_back();  // Null tensor for an absent optional input.
      continue;
    }
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[onnx_idx], logger, input_names));
  }
  return Ort::Status();
}

Ort::Status QMoEOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  // IsOpSupported has already restricted the attributes to the configuration QNN GPU supports.
  OrtNodeAttrHelper node_helper(node_unit);
  const auto node_index = node_unit.Index();
  const std::string& node_name = node_unit.Name();
  std::vector<std::string> param_tensor_names;

  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_index, node_name,
                                         static_cast<uint32_t>(node_helper.Get("k", static_cast<int64_t>(1))),
                                         QNN_OP_Q_MO_E_PARAM_K, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_index, node_name,
                                         static_cast<uint32_t>(kSupportedExpertWeightBits),
                                         QNN_OP_Q_MO_E_PARAM_EXPERT_WEIGHT_BITS, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_index, node_name,
                                         static_cast<uint32_t>(kSupportedBlockSize),
                                         QNN_OP_Q_MO_E_PARAM_BLOCK_SIZE, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_index, node_name,
                                         QNN_OP_Q_MO_E_ACTIVATION_TYPE_SWIGLU,
                                         QNN_OP_Q_MO_E_PARAM_ACTIVATION_TYPE, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_index, node_name,
                                         QNN_OP_Q_MO_E_SWIGLU_FUSION_FUSED_INTERLEAVED,
                                         QNN_OP_Q_MO_E_PARAM_SWIGLU_FUSION, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_index, node_name,
                                      node_helper.Get("swiglu_limit", 0.0f),
                                      QNN_OP_Q_MO_E_PARAM_SWIGLU_LIMIT, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_index, node_name,
                                     node_helper.Get("normalize_routing_weights", static_cast<int64_t>(0)) != 0,
                                     QNN_OP_Q_MO_E_PARAM_NORMALIZE_ROUTING_WEIGHTS, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_index, node_name, false,
                                     QNN_OP_Q_MO_E_PARAM_USE_SPARSE_MIXER, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_index, node_name,
                                      node_helper.Get("activation_alpha", 1.0f),
                                      QNN_OP_Q_MO_E_PARAM_ACTIVATION_ALPHA, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_index, node_name,
                                      node_helper.Get("activation_beta", 0.0f),
                                      QNN_OP_Q_MO_E_PARAM_ACTIVATION_BETA, param_tensor_names));

  return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                        logger, do_op_validation, QNN_OP_Q_MO_E);
}

void CreateQMoEOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QMoEOpBuilder>());
}
#else
void CreateQMoEOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  ORT_UNUSED_PARAMETER(op_type);
  ORT_UNUSED_PARAMETER(op_registrations);
}
#endif  // QNN_QMOE_AVAILABLE

}  // namespace qnn
}  // namespace onnxruntime
