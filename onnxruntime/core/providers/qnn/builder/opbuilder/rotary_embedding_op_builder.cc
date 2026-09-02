// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

class RotaryEmbeddingOpBuilder final : public BaseOpBuilder {
 public:
  RotaryEmbeddingOpBuilder() : BaseOpBuilder("RotaryEmbeddingOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RotaryEmbeddingOpBuilder);

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

 private:
  Ort::Status ValidateInputShapes(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const Ort::Logger& logger) const ORT_MUST_USE_RESULT;
};

Ort::Status RotaryEmbeddingOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                    const OrtNodeUnit& node_unit,
                                                    const Ort::Logger& logger) const {
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Validating RotaryEmbedding op for QNN EP");

  RETURN_IF_NOT(node_unit.Domain() != kMSDomain,
                "QNN EP only supports standard ONNX RotaryEmbedding (opset 23+), "
                "not com.microsoft.RotaryEmbedding. "
                "The native QNN op does not support the contrib op's input layout or attributes.");

  RETURN_IF_NOT(IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()),
                "QNN RotaryEmbedding is only supported on HTP backend");

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= 4,
                "RotaryEmbedding requires 3-4 inputs (X, cos_cache, sin_cache, [position_ids])");
  RETURN_IF_NOT(outputs.size() == 1, "RotaryEmbedding requires exactly 1 output");

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));

  RETURN_IF_NOT(input_info.qnn_data_type == QNN_DATATYPE_FLOAT_16 ||
                    input_info.qnn_data_type == QNN_DATATYPE_FLOAT_32,
                "RotaryEmbedding only supports FP16 and FP32 data types");

  TensorInfo cos_cache_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], cos_cache_info));
  RETURN_IF_NOT(cos_cache_info.qnn_data_type == input_info.qnn_data_type,
                "cos_cache must have same data type as input");

  TensorInfo sin_cache_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], sin_cache_info));
  RETURN_IF_NOT(sin_cache_info.qnn_data_type == input_info.qnn_data_type,
                "sin_cache must have same data type as input");

  RETURN_IF_ERROR(ValidateInputShapes(qnn_model_wrapper, node_unit, logger));

  OrtNodeAttrHelper node_helper(node_unit);

  int64_t rotary_embedding_dim = node_helper.Get("rotary_embedding_dim", static_cast<int64_t>(0));

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape),
                "Cannot get input shape");

  const size_t input_rank = input_shape.size();
  RETURN_IF_NOT(input_rank == 3 || input_rank == 4,
                "RotaryEmbedding input must be rank 3 or 4");

  if (input_rank == 3) {
    int64_t num_heads = node_helper.Get("num_heads", static_cast<int64_t>(0));
    RETURN_IF_NOT(num_heads > 0,
                  "num_heads attribute is required for 3D input and must be > 0");
    const uint32_t hidden_size = input_shape[2];
    RETURN_IF_NOT(hidden_size % num_heads == 0,
                  "hidden_size must be divisible by num_heads");
  }

  uint32_t head_size = 0;
  if (input_rank == 4) {
    head_size = input_shape[3];  // [B, NH, S, HS]
  } else {
    int64_t num_heads = node_helper.Get("num_heads", static_cast<int64_t>(0));
    head_size = input_shape[2] / static_cast<uint32_t>(num_heads);  // [B, S, NH*HS]
  }

  if (rotary_embedding_dim == 0) {
    rotary_embedding_dim = head_size;
  }
  RETURN_IF_NOT(rotary_embedding_dim % 2 == 0,
                "rotary_embedding_dim must be even");
  RETURN_IF_NOT(rotary_embedding_dim >= 2 && rotary_embedding_dim <= static_cast<int64_t>(head_size),
                "rotary_embedding_dim must be in [2, head_size]");

  std::vector<uint32_t> cos_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].shape, cos_shape),
                "Cannot get cos_cache shape");
  std::vector<uint32_t> sin_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[2].shape, sin_shape),
                "Cannot get sin_cache shape");

  const bool has_position_ids = (inputs.size() > 3 && inputs[3].Exists());

  if (has_position_ids) {
    RETURN_IF_NOT(cos_shape.size() == 2,
                  "cos_cache must be rank 2 [max_pos, rotary_dim/2] when position_ids is provided");
    RETURN_IF_NOT(sin_shape.size() == 2,
                  "sin_cache must be rank 2 [max_pos, rotary_dim/2] when position_ids is provided");
  } else {
    RETURN_IF_NOT(cos_shape.size() == 3,
                  "cos_cache must be rank 3 [B, S, rotary_dim/2] when position_ids is absent");
    RETURN_IF_NOT(sin_shape.size() == 3,
                  "sin_cache must be rank 3 [B, S, rotary_dim/2] when position_ids is absent");
  }

  const uint32_t expected_cache_dim = static_cast<uint32_t>(rotary_embedding_dim / 2);
  RETURN_IF_NOT(cos_shape.back() == expected_cache_dim,
                "cos_cache last dimension must equal rotary_embedding_dim/2");
  RETURN_IF_NOT(sin_shape.back() == expected_cache_dim,
                "sin_cache last dimension must equal rotary_embedding_dim/2");

  if (has_position_ids) {
    std::vector<uint32_t> pos_ids_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[3].shape, pos_ids_shape),
                  "Cannot get position_ids shape");
    RETURN_IF_NOT(pos_ids_shape.size() == 2,
                  "position_ids must be rank 2 [B, S]");

    const uint32_t batch_size = input_shape[0];
    const uint32_t seq_len = (input_rank == 4) ? input_shape[2] : input_shape[1];
    RETURN_IF_NOT(pos_ids_shape[0] == batch_size && pos_ids_shape[1] == seq_len,
                  "position_ids shape must match [batch_size, seq_len]");
  }

  return BaseOpBuilder::IsOpSupported(qnn_model_wrapper, node_unit, logger);
}

Ort::Status RotaryEmbeddingOpBuilder::ValidateInputShapes(QnnModelWrapper& qnn_model_wrapper,
                                                          const OrtNodeUnit& node_unit,
                                                          const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape),
                "Cannot get input shape");

  const size_t input_rank = input_shape.size();
  RETURN_IF_NOT(input_rank == 3 || input_rank == 4,
                "RotaryEmbedding input must be rank 3 [B,S,NH*HS] or rank 4 [B,NH,S,HS]");

  return Ort::Status();
}

// Only process inputs that exist — position_ids (input[3]) is optional.
Ort::Status RotaryEmbeddingOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const OrtNodeUnit& node_unit,
                                                    const Ort::Logger& logger,
                                                    std::vector<std::string>& input_names,
                                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].Exists()) {
      RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[i], logger, input_names));
    }
  }
  return Ort::Status();
}

// Rank-3 input [B, S, NH*HS] is reshaped/transposed to 4D [B, NH, S, HS] for the native op.
Ort::Status RotaryEmbeddingOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                  const OrtNodeUnit& node_unit,
                                                                  std::vector<std::string>&& input_names,
                                                                  const Ort::Logger& logger,
                                                                  bool do_op_validation) const {
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Processing RotaryEmbedding op for QNN (native op)");

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  OrtNodeAttrHelper node_helper(node_unit);

  const bool interleaved = node_helper.Get("interleaved", false);
  const int64_t rotary_embedding_dim = node_helper.Get("rotary_embedding_dim", static_cast<int64_t>(0));
  const int64_t num_heads = node_helper.Get("num_heads", static_cast<int64_t>(0));

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape),
                "Cannot get input shape");

  const size_t input_rank = input_shape.size();
  const bool is_4d_input = (input_rank == 4);

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
  const Qnn_DataType_t qnn_data_type = input_info.qnn_data_type;

  uint32_t batch_size = input_shape[0];
  uint32_t seq_len = 0;
  uint32_t num_heads_val = 0;
  uint32_t head_size = 0;

  if (is_4d_input) {
    num_heads_val = input_shape[1];
    seq_len = input_shape[2];
    head_size = input_shape[3];
  } else {
    seq_len = input_shape[1];
    RETURN_IF_NOT(num_heads > 0, "num_heads required for 3D input");
    num_heads_val = static_cast<uint32_t>(num_heads);
    head_size = input_shape[2] / num_heads_val;
  }

  const std::vector<uint32_t> native_shape = {batch_size, num_heads_val, seq_len, head_size};

  std::string native_input = input_names[0];
  if (!is_4d_input) {
    std::string reshaped = utils::UniqueNameGenerator().New(node_unit, "_reshape_input");
    std::vector<uint32_t> reshaped_bsnh_shape = {batch_size, seq_len, num_heads_val, head_size};
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        input_names[0], reshaped,
        input_shape, reshaped_bsnh_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));

    native_input = utils::UniqueNameGenerator().New(node_unit, "_transpose_input");
    std::vector<uint32_t> transpose_perm = {0, 2, 1, 3};
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        node_unit.Index(), reshaped, native_input,
        reshaped_bsnh_shape, transpose_perm, native_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));
  }

  std::vector<std::string> rope_input_names = {native_input, input_names[1], input_names[2]};
  const bool has_position_ids = (inputs.size() > 3 && inputs[3].Exists());
  if (has_position_ids) {
    rope_input_names.push_back(input_names[3]);
  }

  std::vector<std::string> rope_param_names;
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name() + "_interleaved",
                                     interleaved, QNN_OP_ROTARY_EMBEDDING_PARAM_INTERLEAVED, rope_param_names));

  const uint32_t resolved_rotary_dim = (rotary_embedding_dim != 0)
                                           ? static_cast<uint32_t>(rotary_embedding_dim)
                                           : head_size;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(),
                                         node_unit.Name() + "_rotary_embedding_dim",
                                         resolved_rotary_dim,
                                         QNN_OP_ROTARY_EMBEDDING_PARAM_ROTARY_EMBEDDING_DIM,
                                         rope_param_names));

  const std::string& output_name = outputs[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  std::string native_output = output_name;
  if (!is_4d_input) {
    native_output = utils::UniqueNameGenerator().New(node_unit, "_rope_output");
  }

  QnnTensorWrapper native_output_tensor(native_output,
                                        (!is_4d_input) ? QNN_TENSOR_TYPE_NATIVE
                                                       : (is_graph_output ? QNN_TENSOR_TYPE_APP_READ
                                                                          : QNN_TENSOR_TYPE_NATIVE),
                                        qnn_data_type, input_info.quant_param.Copy(),
                                        std::vector<uint32_t>(native_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(native_output_tensor)),
                "Failed to add RotaryEmbedding output tensor");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::UniqueNameGenerator().New(node_unit, "_rope"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_ROTARY_EMBEDDING,
                    std::move(rope_input_names),
                    {native_output},
                    std::move(rope_param_names),
                    do_op_validation),
                "Failed to create native RotaryEmbedding node");

  if (!is_4d_input) {
    std::string transposed_back = utils::UniqueNameGenerator().New(node_unit, "_transpose_output");
    std::vector<uint32_t> transpose_perm = {0, 2, 1, 3};
    std::vector<uint32_t> bsnh_shape = {batch_size, seq_len, num_heads_val, head_size};
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        node_unit.Index(), native_output, transposed_back,
        native_shape, transpose_perm, bsnh_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, false));

    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        transposed_back, output_name,
        bsnh_shape, input_shape,
        qnn_data_type, input_info.quant_param, do_op_validation,
        false, is_graph_output));
  }

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Successfully built RotaryEmbedding QNN Op");
  return Ort::Status();
}

void CreateRotaryEmbeddingOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RotaryEmbeddingOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
