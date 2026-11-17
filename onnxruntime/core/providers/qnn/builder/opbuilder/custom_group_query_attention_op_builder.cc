// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

#include <array>
#include <functional>
#include <limits>
#include <numeric>

namespace onnxruntime {
namespace qnn {

// Separate translation for the experimental gqa_op_package contract.
class CustomGroupQueryAttentionOpBuilder final : public BaseOpBuilder {
 public:
  CustomGroupQueryAttentionOpBuilder() : BaseOpBuilder("CustomGroupQueryAttentionOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomGroupQueryAttentionOpBuilder);

 protected:
  Ort::Status IsOpSupported(QnnModelWrapper& qmw, const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override {
    RETURN_IF_NOT(qmw.GetQnnBackendType() == QnnBackendType::HTP, "Custom GQA requires HTP");
    RETURN_IF_NOT(qmw.GetModelSettings().use_custom_gqa, "Custom GQA package is not configured");
    const auto& inputs = node_unit.Inputs();
    const auto& outputs = node_unit.Outputs();
    for (size_t i : {0u, 3u, 4u, 5u, 6u}) {
      RETURN_IF_NOT(inputs.size() > i && inputs[i].Exists(), "Custom GQA requires query, past KV and sequence lengths");
    }
    RETURN_IF_NOT(outputs.size() == 3 && outputs[0].Exists() && outputs[1].Exists() && outputs[2].Exists(),
                  "Custom GQA requires output, present_key and present_value");
    const auto present = [&](size_t i) { return inputs.size() > i && inputs[i].Exists(); };
    RETURN_IF(present(1) != present(2), "Custom GQA requires both key and value, or packed QKV");
    RETURN_IF(present(7) != present(8), "Custom GQA requires both cos_cache and sin_cache");
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].Exists()) continue;
      RETURN_IF(i > 10, "Custom GQA does not support head_sink or Q/K norm weights");
      TensorInfo info{};
      RETURN_IF_ERROR(qmw.GetTensorInfo(inputs[i], info));
      const bool is_id = i == 5 || i == 6 || i == 9;
      RETURN_IF_NOT(is_id ? (info.qnn_data_type == QNN_DATATYPE_INT_32 ||
                             (i == 9 && info.qnn_data_type == QNN_DATATYPE_INT_64))
                          : info.qnn_data_type == QNN_DATATYPE_FLOAT_16,
                    "Custom GQA requires FP16 activations/cache and INT32 sequence lengths");
    }
    for (const auto& output : outputs) {
      TensorInfo info{};
      RETURN_IF_ERROR(qmw.GetTensorInfo(output, info));
      RETURN_IF_NOT(info.qnn_data_type == QNN_DATATYPE_FLOAT_16, "Custom GQA requires FP16 outputs");
    }
    return AddToModelBuilder(qmw, node_unit, logger, /*do_op_validation=*/true);
  }

  Ort::Status ProcessInputs(QnnModelWrapper& qmw, const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger, std::vector<std::string>& input_names,
                            bool do_op_validation) const override {
    const auto& inputs = node_unit.Inputs();
    const bool rotary = OrtNodeAttrHelper(node_unit).Get("do_rotary", int64_t{0}) != 0;
    // Same ONNX-to-QNN order as the native builder; this package omits absent slots, not null tensors.
    constexpr std::array<size_t, 11> order{0, 5, 6, 1, 2, 3, 4, 7, 8, 9, 10};
    for (size_t onnx_idx : order) {
      if (inputs.size() <= onnx_idx || !inputs[onnx_idx].Exists()) continue;
      if (!rotary && (onnx_idx == 7 || onnx_idx == 8)) continue;
      RETURN_IF_ERROR(ProcessInput(qmw, inputs[onnx_idx], logger, input_names));
      TensorInfo info{};
      RETURN_IF_ERROR(qmw.GetTensorInfo(inputs[onnx_idx], info));
      if (onnx_idx == 5 && info.shape != std::vector<uint32_t>{1}) {
        // Accept the [1, 1] sequence-length shape used by model surgery, as the native builder does.
        RETURN_IF_NOT(!info.shape.empty() && info.shape[0] == 1 &&
                          std::accumulate(info.shape.begin(), info.shape.end(), uint64_t{1},
                                          std::multiplies<uint64_t>()) == 1,
                      "Custom GQA seqlens_k must contain one element for batch 1");
        const std::string reshaped = utils::UniqueNameGenerator().New(input_names.back(), "_custom_gqa_rank1");
        RETURN_IF_ERROR(qmw.AddReshapeNode(input_names.back(), reshaped, info.shape, {1},
                                           info.qnn_data_type, info.quant_param.Copy(), do_op_validation,
                                           qmw.IsGraphInput(input_names.back())));
        input_names.back() = reshaped;
      } else if (onnx_idx == 9 && info.qnn_data_type == QNN_DATATYPE_INT_64) {
        // Cast before the base pipeline: reordered/compacted indices no longer match ONNX indices.
        const std::string cast_output = utils::UniqueNameGenerator().New(input_names.back(), "_custom_gqa_int32");
        RETURN_IF_ERROR(qmw.AddCastNode(cast_output, input_names.back(), cast_output,
                                        QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_INT_32,
                                        QnnQuantParamsWrapper(), std::move(info.shape), do_op_validation));
        input_names.back() = cast_output;
      }
    }
    return Ort::Status();
  }

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qmw, const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger, bool do_op_validation) const override {
    ORT_UNUSED_PARAMETER(logger);
    OrtNodeAttrHelper attrs(node_unit);
    const auto heads = attrs.GetInt64("num_heads");
    const auto kv_heads = attrs.GetInt64("kv_num_heads");
    RETURN_IF_NOT(heads && kv_heads && *heads > 0 && *kv_heads > 0 && *heads % *kv_heads == 0,
                  "Custom GQA requires positive num_heads divisible by kv_num_heads");
    RETURN_IF(*heads > std::numeric_limits<uint32_t>::max() || *kv_heads > std::numeric_limits<uint32_t>::max(),
              "Custom GQA head counts exceed uint32 range");
    const int64_t rotary = attrs.Get("do_rotary", int64_t{0});
    const int32_t causal = attrs.Get("causal", 1);
    const int32_t window = attrs.Get("local_window_size", -1);
    const int32_t compact = attrs.Get("sliding_window_cache", 0);
    const int32_t smooth = attrs.Get("smooth_softmax", -1);
    const float softcap = attrs.Get("softcap", 0.0f);
    const int32_t qk_output = attrs.Get("qk_output", 0);
    RETURN_IF(rotary != 0 && rotary != 1, "Custom GQA do_rotary must be 0 or 1");
    RETURN_IF(attrs.Get("rotary_interleaved", 0) != 0, "Custom GQA does not support interleaved rotary");
    RETURN_IF(attrs.Get("k_quant_type", std::string("NONE")) != "NONE" ||
                  attrs.Get("v_quant_type", std::string("NONE")) != "NONE" || attrs.HasAttr("kv_cache_bit_width"),
              "Custom GQA does not support quantized KV cache");
    RETURN_IF(causal != 1 || (smooth != -1 && smooth != 0) || softcap != 0.0f || qk_output != 0,
              "Custom GQA requires causal attention without smooth_softmax, softcap or output_qk");
    RETURN_IF((compact != 0 && compact != 1) || (compact == 1 ? window <= 0 : window != -1),
              "Custom GQA requires a positive window for compact cache, or -1 for full cache");

    std::vector<std::string> param_names;
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), static_cast<uint32_t>(*heads), "num_heads", param_names));
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), static_cast<uint32_t>(*kv_heads), "kv_num_heads", param_names));
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), static_cast<uint32_t>(rotary), "do_rotary", param_names));
    // Package scale=0 means 1/sqrt(D), matching the ONNX sentinel.
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), attrs.Get("scale", 0.0f), "scale", param_names));
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), uint32_t{1}, "causal", param_names));
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), window, "local_window_size", param_names));
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), static_cast<uint32_t>(compact), "sliding_window_cache", param_names));
    // ONNX -1 disables smooth_softmax; the package uses 0.
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), uint32_t{0}, "smooth_softmax", param_names));
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), softcap, "softcap", param_names));
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), uint32_t{0}, "qk_output", param_names));
    RETURN_IF_ERROR(AddQnnScalar(qmw, node_unit.Index(), node_unit.Name(), attrs.Get("qk_norm_epsilon", 1.0e-6f), "qk_norm_epsilon", param_names));

    std::vector<std::string> output_names;
    for (const auto& output : node_unit.Outputs()) {
      TensorInfo info{};
      RETURN_IF_ERROR(qmw.GetTensorInfo(output, info));
      const auto type = qmw.IsGraphOutput(output.name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
      QnnTensorWrapper tensor(output.name, type, info.qnn_data_type, std::move(info.quant_param), std::move(info.shape));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tensor)), "Failed to add custom GQA output tensor");
      output_names.push_back(output.name);
    }
    RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit), "gqa_op_package", "GroupQueryAttention",
                                    std::move(input_names), std::move(output_names), std::move(param_names), do_op_validation),
                  "Failed to add custom GQA node");
    return Ort::Status();
  }
};

void CreateCustomGroupQueryAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& registrations) {
  registrations.AddOpBuilder(op_type, std::make_unique<CustomGroupQueryAttentionOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
