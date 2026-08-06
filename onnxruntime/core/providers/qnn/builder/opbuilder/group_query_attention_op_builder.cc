// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <numeric>

#include "QnnOpDef.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// GQA not available until opset version 2.12.0 (QAIRT 2.48). TODO: Remove this check once the EP uplevels to 2.48.
#if !(QNN_OPSET_VERSION_MAJOR < 2 || (QNN_OPSET_VERSION_MAJOR == 2 && QNN_OPSET_VERSION_MINOR <= 11))
class GroupQueryAttentionOpBuilder : public BaseOpBuilder {
 public:
  GroupQueryAttentionOpBuilder() : BaseOpBuilder("GroupQueryAttentionOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GroupQueryAttentionOpBuilder);

 protected:
  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

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

Ort::Status GroupQueryAttentionOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  RETURN_IF_NOT(IsGpuBackend(qnn_model_wrapper.GetQnnBackendType()),
                "GroupQueryAttention is only supported with the GPU backend");

  const size_t num_inputs = node_unit.Inputs().size();
  const auto& inputs = node_unit.Inputs();

  const size_t num_outputs = node_unit.Outputs().size();
  const auto& outputs = node_unit.Outputs();

  TensorInfo present_key_tensor_info = {};
  RETURN_IF_NOT(num_outputs > 1 && outputs[1].Exists(), "Required output tensor present_key not provided");
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[1], present_key_tensor_info));
  RETURN_IF_NOT(present_key_tensor_info.shape.size() == 4, "Unexpected rank for present_key");
  const auto max_sequence_length = present_key_tensor_info.shape[2];

  // At time of writing, the com.microsoft.GroupQueryAttention op def has 14 inputs and 4 outputs.
  const size_t max_num_inputs = 14;
  const size_t max_num_outputs = 4;

  for (size_t i = 10; i < std::min(num_inputs, max_num_inputs); i++) {
    RETURN_IF(inputs[i].Exists(), "attention_bias, head_sink, k_scale, and v_scale inputs are not supported");
  }
  RETURN_IF(num_inputs > max_num_inputs,
            ("More than " + std::to_string(max_num_inputs) + " inputs provided, which is unsupported").c_str());

  RETURN_IF(num_outputs > 3 && outputs[3].Exists(), "output_qk output is not supported");
  RETURN_IF(num_outputs > max_num_outputs,
            ("More than " + std::to_string(max_num_outputs) + " outputs provided, which is unsupported").c_str());

  OrtNodeAttrHelper node_helper(node_unit);

  RETURN_IF_NOT(node_helper.HasAttr("num_heads"), "required attribute num_heads not provided");
  RETURN_IF_NOT(node_helper.HasAttr("kv_num_heads"), "required attribute kv_num_heads not provided");

  std::string k_quant_type = node_helper.Get("k_quant_type", std::string("NONE"));
  RETURN_IF(k_quant_type != "NONE", "k_quant_type != NONE not supported");
  std::string v_quant_type = node_helper.Get("v_quant_type", std::string("NONE"));
  RETURN_IF(v_quant_type != "NONE", "v_quant_type != NONE not supported");
  RETURN_IF(node_helper.HasAttr("kv_cache_bit_width"),
            "kv_cache_bit_width attribute not supported");

  int32_t local_window_size = node_helper.Get("local_window_size", -1);
  RETURN_IF(local_window_size < -1, ("unsupported value for local_window_size: " + std::to_string(local_window_size)).c_str());
  RETURN_IF(local_window_size != -1 && SafeInt<uint32_t>(local_window_size) < max_sequence_length,
            "Local attention through local_window_size not supported");

  int32_t qk_output = node_helper.Get("qk_output", 0);
  RETURN_IF(qk_output != 0, "qk_output != 0 not supported");

  // note: QNN RotaryEmbedding supports the interleaved attribute, but QNN GroupQueryAttention does not.
  //       We could support ONNX GQA w/ rotary_interleaved by decomposing it into QNN RotaryEmbeddings (w/ interleaved)
  //       + QNN GQA.
  int32_t rotary_interleaved = node_helper.Get("rotary_interleaved", 0);
  RETURN_IF(rotary_interleaved != 0, "rotary_interleaved != 0 not supported");

  int32_t smooth_softmax = node_helper.Get("smooth_softmax", -1);
  RETURN_IF(smooth_softmax != -1, "smooth_softmax != -1 not supported");

  float softcap = node_helper.Get("softcap", 0.0f);
  RETURN_IF(softcap != 0.0f, "softcap != 0 not supported");

  // Validate OpConfig with backend
  std::vector<std::string> input_names;
  RETURN_IF_ERROR(ProcessInputs(qnn_model_wrapper, node_unit, logger, input_names, true));
  RETURN_IF_ERROR(ProcessAttributesAndOutputs(qnn_model_wrapper, node_unit, std::move(input_names), logger, true));
  return Ort::Status();
}

Ort::Status GroupQueryAttentionOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        const Ort::Logger& logger,
                                                        std::vector<std::string>& input_names,
                                                        bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& onnx_inputs = node_unit.Inputs();

  constexpr auto qnn_idx_to_onnx = std::array{
      0u,  // query
      5u,  // seqlens_K
      6u,  // total_sequence_length
      1u,  // key
      2u,  // value
      3u,  // past_key
      4u,  // past_value
      7u,  // cos_cache
      8u,  // sin_cache
      9u   // position_ids
  };
  constexpr size_t kQnnTotalSeqLenIdx = 2;  // index of total_sequence_length in qnn_idx_to_onnx

  for (size_t qnn_idx = 0; qnn_idx < qnn_idx_to_onnx.size(); ++qnn_idx) {
    const auto onnx_idx = qnn_idx_to_onnx[qnn_idx];
    if (onnx_inputs.size() > onnx_idx && onnx_inputs[onnx_idx].Exists()) {
      // QNN requires total_sequence_length as a 0D scalar, but ONNX provides it as shape [1].
      // Build the tensor wrapper directly with an empty shape to avoid a Reshape.
      if (qnn_idx == kQnnTotalSeqLenIdx) {
        const std::string& input_name = onnx_inputs[onnx_idx].name;
        if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
          TensorInfo tensor_info = {};
          RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[onnx_idx], tensor_info));
          tensor_info.shape = {};  // override to 0D scalar
          QnnTensorWrapper tensor_wrapper;
          RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(tensor_info, input_name, tensor_wrapper));
          RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper)), "Failed to add tensor.");
        }
        input_names.push_back(input_name);
      } else {
        RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[onnx_idx], logger, input_names));
      }
    } else {
      std::string null_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_null_tensor");
      input_names.emplace_back(null_tensor_name);
      QnnTensorWrapper null_tensor_wrapper = QnnTensorWrapper::MakeNull(null_tensor_name);
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(null_tensor_wrapper)),
                    ("Failed to add null tensor: " + null_tensor_name).c_str());
    }
  }
  return Ort::Status();
}

Ort::Status GroupQueryAttentionOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                      const OrtNodeUnit& node_unit,
                                                                      std::vector<std::string>&& input_names,
                                                                      const Ort::Logger& logger,
                                                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  OrtNodeAttrHelper node_helper(node_unit);

  std::vector<std::string> param_names;

  const size_t num_outputs = node_unit.Outputs().size();
  const auto& outputs = node_unit.Outputs();

  // num_heads
  std::optional<int64_t> num_heads = node_helper.GetInt64("num_heads");
  RETURN_IF_NOT(num_heads.has_value(), "required attribute num_heads not provided");
  uint32_t num_heads_u32 = SafeInt<uint32_t>(num_heads.value());
  RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper,
                               node_unit.Index(),
                               node_unit.Name(),
                               num_heads_u32,
                               QNN_OP_GROUP_QUERY_ATTENTION_PARAM_NUM_HEADS,
                               param_names));

  // kv_num_heads
  const std::optional<int64_t> kv_num_heads = node_helper.GetInt64("kv_num_heads");
  RETURN_IF_NOT(kv_num_heads.has_value(), "required attribute kv_num_heads not provided");
  const uint32_t kv_num_heads_u32 = SafeInt<uint32_t>(kv_num_heads.value());
  RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper,
                               node_unit.Index(),
                               node_unit.Name(),
                               kv_num_heads_u32,
                               QNN_OP_GROUP_QUERY_ATTENTION_PARAM_KV_NUM_HEADS,
                               param_names));

  // do_rotary
  const int64_t do_rotary = node_helper.Get("do_rotary", static_cast<int64_t>(0));
  const uint32_t do_rotary_u32 = SafeInt<uint32_t>(do_rotary);
  RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper,
                               node_unit.Index(),
                               node_unit.Name(),
                               do_rotary_u32,
                               QNN_OP_GROUP_QUERY_ATTENTION_PARAM_DO_ROTARY,
                               param_names));

  // scale
  RETURN_IF_NOT(num_outputs > 0 && outputs[0].Exists(), "Required output out[0] not provided");
  TensorInfo output_tensor_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_tensor_info));
  RETURN_IF_NOT(output_tensor_info.shape.size() == 3, "Unexpected rank for output out[0]");
  const size_t head_size = output_tensor_info.shape[2] / num_heads.value();
  RETURN_IF(head_size == 0, "head_size can't be zero!");

  const float scale_default = 1.0f / std::sqrt(static_cast<float>(head_size));
  const float scale = node_helper.Get("scale", scale_default);
  RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper,
                               node_unit.Index(),
                               node_unit.Name(),
                               scale,
                               QNN_OP_GROUP_QUERY_ATTENTION_PARAM_SCALE,
                               param_names));

  std::vector<std::string> output_names;
  for (size_t output_idx = 0; output_idx < num_outputs; ++output_idx) {
    if (outputs[output_idx].Exists()) {
      const std::string& output_name = outputs[output_idx].name;
      output_names.push_back(output_name);

      TensorInfo output_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[output_idx], output_info));

      bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
      Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

      QnnTensorWrapper output_tensorwrapper(output_name,
                                            tensor_type,
                                            output_info.qnn_data_type,
                                            std::move(output_info.quant_param),
                                            std::move(output_info.shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
    } else {
      std::string null_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_null_tensor");
      output_names.emplace_back(null_tensor_name);
      QnnTensorWrapper null_tensor_wrapper = QnnTensorWrapper::MakeNull(null_tensor_name);
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(null_tensor_wrapper)),
                    ("Failed to add null tensor: " + null_tensor_name).c_str());
    }
  }

  const std::string node_name = utils::UniqueNameGenerator().New(node_unit);

  const size_t num_inputs = node_unit.Inputs().size();
  const auto& inputs = node_unit.Inputs();
  RETURN_IF_NOT(num_inputs > 5 && inputs[5].Exists(), "Required input tensor seqlens_k not provided");

  TensorInfo seqlens_k_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[5], seqlens_k_info));
  if (seqlens_k_info.shape.size() > 1) {
    // The op def for com.microsoft.GroupQueryAttention requires seqlens_k to have rank 1,
    // but ORT does not enforce it. And the Olive AttentionMaskToSequenceLengths graph
    // surgery generates a seqlens_k with shape [batch_size, 1]. QNN's GQA op def does enforce
    // seqlens_k to have rank 1 and will fail during op validation otherwise. So handle
    // the technically out-of-spec rank > 1 seqlens_k by inserting a reshape to [batch_size].
    uint32_t seqlens_k_total_size = std::accumulate(seqlens_k_info.shape.begin(),
                                                    seqlens_k_info.shape.end(),
                                                    1u,
                                                    std::multiplies<uint32_t>());

    RETURN_IF_NOT(seqlens_k_info.shape[0] == seqlens_k_total_size, "Unexpected shape for seqlens_k");
    const auto& seqlens_k_input_name = input_names[1];
    bool is_seqlens_k_graph_input = qnn_model_wrapper.IsGraphInput(seqlens_k_input_name);
    std::string seqlens_k_reshaped = onnxruntime::qnn::utils::UniqueNameGenerator().New(seqlens_k_input_name, "_reshaped");
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(seqlens_k_input_name,
                                                     seqlens_k_reshaped,
                                                     seqlens_k_info.shape,
                                                     {seqlens_k_total_size},
                                                     seqlens_k_info.qnn_data_type,
                                                     seqlens_k_info.quant_param.Copy(),
                                                     do_op_validation,
                                                     is_seqlens_k_graph_input));
    input_names[1] = seqlens_k_reshaped;
  }

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_GROUP_QUERY_ATTENTION,
                                                std::move(input_names),
                                                std::move(output_names),
                                                std::move(param_names),
                                                do_op_validation),
                "Failed to add node.");

  return Ort::Status();
}

void CreateGroupQueryAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GroupQueryAttentionOpBuilder>());
}
#else
void CreateGroupQueryAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  ORT_UNUSED_PARAMETER(op_type);
  ORT_UNUSED_PARAMETER(op_registrations);
}
#endif  // !(QNN_OPSET_VERSION_MAJOR < 2 || (QNN_OPSET_VERSION_MAJOR == 2 && QNN_OPSET_VERSION_MINOR <= 11))

}  // namespace qnn
}  // namespace onnxruntime
