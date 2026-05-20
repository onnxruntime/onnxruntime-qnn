// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

// GQA not available until opset version 2.12.0 (QAIRT 2.48). Until the EP upgrades to default to 2.48,
// manually define the necessary values.
// TODO: Remove once the EP upgrades to 2.48.
#if QNN_OPSET_VERSION_MAJOR < 2 || (QNN_OPSET_VERSION_MAJOR == 2 && QNN_OPSET_VERSION_MINOR <= 11)
#define QNN_OP_GROUP_QUERY_ATTENTION "GroupQueryAttention"
#define QNN_OP_GROUP_QUERY_ATTENTION_PARAM_NUM_HEADS "num_heads"
#define QNN_OP_GROUP_QUERY_ATTENTION_PARAM_KV_NUM_HEADS "kv_num_heads"
#define QNN_OP_GROUP_QUERY_ATTENTION_PARAM_DO_ROTARY "do_rotary"
#define QNN_OP_GROUP_QUERY_ATTENTION_PARAM_SCALE "scale"
#endif  // QNN_OPSET_VERSION_MAJOR < 2 || (QNN_OPSET_VERSION_MAJOR == 2 && QNN_OPSET_VERSION_MINOR <= 11)

namespace onnxruntime {
namespace qnn {

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

  const size_t num_inputs = node_unit.Inputs().size();
  const auto& inputs = node_unit.Inputs();

  const size_t num_outputs = node_unit.Outputs().size();
  const auto& outputs = node_unit.Outputs();

  TensorInfo present_key_tensor_info = {};
  RETURN_IF_NOT(outputs.size() > 1 && outputs[1].Exists(), "Required output tensor present_key not provided");
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[1], present_key_tensor_info));
  RETURN_IF_NOT(present_key_tensor_info.shape.size() == 4, "Unexpected rank for present_key");
  const auto max_sequence_length = present_key_tensor_info.shape[2];

  if (num_inputs > 3 && inputs[3].Exists()) {
    TensorInfo past_key_tensor_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[3], past_key_tensor_info));
    RETURN_IF_NOT(past_key_tensor_info.shape.size() == 4, "Unexpected rank for past_key");
    RETURN_IF_NOT(past_key_tensor_info.shape[2] == max_sequence_length,
                  "QNN GroupQueryAttention requires past_key_shape[2] == present_key_shape[2] == max_sequence_length");
  }

  if (num_inputs > 4 && inputs[4].Exists()) {
    TensorInfo past_value_tensor_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[4], past_value_tensor_info));
    RETURN_IF_NOT(past_value_tensor_info.shape.size() == 4, "Unexpected rank for past_value");
    RETURN_IF_NOT(past_value_tensor_info.shape[2] == max_sequence_length,
                  "QNN GroupQueryAttention requires past_value_shape[2] == present_value_shape[2] == max_sequence_length");
  }

  // At time of writing, the com.micorosoft.GroupQueryAttention op def has 14 inputs and 4 outputs.
  const size_t max_num_inputs = 14;
  const size_t max_num_outputs = 4;

  for (size_t i = 10; i < std::min(num_inputs, max_num_inputs); i++) {
    RETURN_IF(inputs[i].Exists(), "attention_bias, head_sink, k_scale, and v_scale inputs are not supported");
  }
  RETURN_IF(num_inputs > max_num_inputs,
            ("More than " + std::to_string(max_num_inputs) + " inputs provided, which is unsupported").c_str());

  RETURN_IF(num_outputs > 3 && outputs[3].Exists(), "output_qk output is not supported");
  RETURN_IF(num_outputs > 4,
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
  RETURN_IF(local_window_size != -1 && SafeInt<uint32_t>(local_window_size) < max_sequence_length,
            "Local attention through local_window_size not supported");

  int32_t qk_output = node_helper.Get("qk_output", 0);
  RETURN_IF(qk_output != 0, "qk_output != 0 not supported");

  float smooth_softmax = node_helper.Get("smooth_softmax", 1.0f);
  RETURN_IF(smooth_softmax != 1.0f, "smooth_softmax != 1 not supported");

  float softcap = node_helper.Get("softcap", 0.0f);
  RETURN_IF(softcap != 0.0f, "softcap != 0 not supported");

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

  for (const auto onnx_idx : qnn_idx_to_onnx) {
    if (onnx_inputs.size() > onnx_idx && onnx_inputs[onnx_idx].Exists()) {
      RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[onnx_idx], logger, input_names));
    } else {
      std::string null_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_null_tensor");
      input_names.emplace_back(null_tensor_name);
      QnnTensorWrapper null_tensor_wrapper(null_tensor_name, QNN_TENSOR_TYPE_NULL, QNN_DATATYPE_UNDEFINED,
                                           QnnQuantParamsWrapper(), std::vector<uint32_t>{0});
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

  // num_heads
  std::optional<int64_t> num_heads = node_helper.GetInt64("num_heads");
  RETURN_IF_NOT(num_heads.has_value(), "required attribute num_heads not provided");
  Qnn_Scalar_t num_heads_scalar = QNN_SCALAR_INIT;
  num_heads_scalar.dataType = QNN_DATATYPE_UINT_32;
  num_heads_scalar.uint32Value = SafeInt<uint32_t>(num_heads.value());

  QnnParamWrapper num_heads_param_wrapper(node_unit.Index(),
                                          node_unit.Name(),
                                          QNN_OP_GROUP_QUERY_ATTENTION_PARAM_NUM_HEADS,
                                          num_heads_scalar);
  param_names.emplace_back(num_heads_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(num_heads_param_wrapper));

  // kv_num_heads
  std::optional<int64_t> kv_num_heads = node_helper.GetInt64("kv_num_heads");
  RETURN_IF_NOT(kv_num_heads.has_value(), "required attribute kv_num_heads not provided");
  Qnn_Scalar_t kv_num_heads_scalar = QNN_SCALAR_INIT;
  kv_num_heads_scalar.dataType = QNN_DATATYPE_UINT_32;
  kv_num_heads_scalar.uint32Value = SafeInt<uint32_t>(kv_num_heads.value());

  QnnParamWrapper kv_num_heads_param_wrapper(node_unit.Index(),
                                             node_unit.Name(),
                                             QNN_OP_GROUP_QUERY_ATTENTION_PARAM_KV_NUM_HEADS,
                                             kv_num_heads_scalar);
  param_names.emplace_back(kv_num_heads_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(kv_num_heads_param_wrapper));

  // do_rotary
  std::optional<int64_t> do_rotary = node_helper.GetInt64("do_rotary");
  if (do_rotary.has_value()) {
    Qnn_Scalar_t do_rotary_scalar = QNN_SCALAR_INIT;
    do_rotary_scalar.dataType = QNN_DATATYPE_UINT_32;
    do_rotary_scalar.uint32Value = SafeInt<uint32_t>(do_rotary.value());

    QnnParamWrapper do_rotary_param_wrapper(node_unit.Index(),
                                            node_unit.Name(),
                                            QNN_OP_GROUP_QUERY_ATTENTION_PARAM_DO_ROTARY,
                                            do_rotary_scalar);
    param_names.emplace_back(do_rotary_param_wrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(do_rotary_param_wrapper));
  }

  // scale
  std::optional<float> scale = node_helper.GetFloat("scale");
  if (scale.has_value()) {
    Qnn_Scalar_t scale_scalar = QNN_SCALAR_INIT;
    scale_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    scale_scalar.floatValue = scale.value();

    QnnParamWrapper scale_param_wrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        QNN_OP_GROUP_QUERY_ATTENTION_PARAM_SCALE,
                                        scale_scalar);
    param_names.emplace_back(scale_param_wrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(scale_param_wrapper));
  }

  std::vector<std::string> output_names;
  const auto& outputs = node_unit.Outputs();
  for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
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
  }

  const std::string node_name = utils::UniqueNameGenerator().New(node_unit);

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

}  // namespace qnn
}  // namespace onnxruntime
