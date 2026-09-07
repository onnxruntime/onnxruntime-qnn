// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/opbuilder/rnn_op_utils.h"

#include <utility>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
namespace rnn_details {

Ort::Status DeriveNumDirectionsConcatAxis(const std::vector<uint32_t>& shape, uint32_t& axis) {
  RETURN_IF_NOT(shape.size() >= 3,
                "QNN EP: bidirectional RNN Concat output must have rank >= 3 "
                "(Y is [seq, num_dir, batch, hidden]; Y_h/Y_c are [num_dir, batch, hidden]).");
  axis = static_cast<uint32_t>(shape.size() - 3);
  return Ort::Status();
}

Ort::Status AddStridedSlice(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const std::string& input_name,
                            const std::string& output_name,
                            const std::vector<uint32_t>& input_shape,
                            const std::vector<uint32_t>& output_shape,
                            const std::vector<std::vector<int32_t>>& ranges,
                            const uint32_t& begin_mask,
                            const uint32_t& end_mask,
                            const uint32_t& shrink_axes,
                            const uint32_t& new_axes_mask,
                            const Qnn_DataType_t& tensor_data_type,
                            const QnnQuantParamsWrapper& quantize_param,
                            bool do_op_validation,
                            bool is_for_input,
                            bool is_for_output) {
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(output_name)) {
    return Ort::Status();
  }

  QnnTensorWrapper input_tensorwrapper(input_name, is_for_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_NATIVE,
                                       tensor_data_type, quantize_param.Copy(), std::vector<uint32_t>(input_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                "Failed to add input tensor for inserted StridedSlice.");

  const std::string node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_STRIDED_SLICE);
  std::vector<uint32_t> ranges_data;
  for (size_t i = 0; i < ranges.size(); i++) {
    for (size_t j = 0; j < 3; j++) {
      ranges_data.emplace_back(SafeInt<uint32_t>(ranges[i][j]));
    }
  }
  QnnParamWrapper ranges_param_wrapper(node_unit.Index(), node_name, QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                       {static_cast<uint32_t>(ranges.size()), 3}, std::move(ranges_data), true);
  std::vector<std::string> param_names = {ranges_param_wrapper.GetParamTensorName()};
  qnn_model_wrapper.AddParamWrapper(std::move(ranges_param_wrapper));

  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, begin_mask,
                                         QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK, param_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, end_mask,
                                         QNN_OP_STRIDED_SLICE_PARAM_END_MASK, param_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, shrink_axes,
                                         QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES, param_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, new_axes_mask,
                                         QNN_OP_STRIDED_SLICE_PARAM_NEW_AXES_MASK, param_names));

  QnnTensorWrapper output_tensorwrapper(output_name, is_for_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                        tensor_data_type, quantize_param.Copy(), std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                "Failed to add output tensor for inserted StridedSlice.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE,
                                                {input_name}, {output_name}, std::move(param_names), do_op_validation),
                "Failed to create manually inserted Qnn StridedSlice node.");
  return Ort::Status();
}

Ort::Status AddStridedSliceOrReshape(QnnModelWrapper& qnn_model_wrapper,
                                     const OrtNodeUnit& node_unit,
                                     const std::string& input_name,
                                     const std::string& output_name,
                                     const std::vector<uint32_t>& input_shape,
                                     const std::vector<uint32_t>& output_shape,
                                     const std::vector<std::vector<int32_t>>& ranges,
                                     const uint32_t& begin_mask,
                                     const uint32_t& end_mask,
                                     const uint32_t& shrink_axes,
                                     const uint32_t& new_axes_mask,
                                     const Qnn_DataType_t& tensor_data_type,
                                     const QnnQuantParamsWrapper& quantize_param,
                                     bool do_op_validation,
                                     bool is_for_input,
                                     bool is_for_output) {
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(output_name)) {
    return Ort::Status();
  }
  size_t minSize = std::min(input_shape.size(), output_shape.size());
  if (input_shape[0] == 1 && std::equal(output_shape.rbegin(), output_shape.rbegin() + minSize, input_shape.rbegin())) {
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input_name, output_name, input_shape, output_shape,
                                                     tensor_data_type, quantize_param.Copy(), quantize_param.Copy(),
                                                     do_op_validation, is_for_input, is_for_output));
  } else {
    QnnTensorWrapper input_tensorwrapper(input_name, is_for_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_NATIVE,
                                         tensor_data_type, quantize_param.Copy(), std::vector<uint32_t>(input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                  "Failed to add input tensor for inserted StridedSlice or Reshape.");

    const std::string node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_STRIDED_SLICE);
    std::vector<uint32_t> ranges_data;
    for (size_t i = 0; i < ranges.size(); i++) {
      for (size_t j = 0; j < 3; j++) {
        ranges_data.emplace_back(SafeInt<uint32_t>(ranges[i][j]));
      }
    }
    QnnParamWrapper ranges_param_wrapper(node_unit.Index(), node_name, QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                         {static_cast<uint32_t>(ranges.size()), 3}, std::move(ranges_data), true);
    std::vector<std::string> param_names = {ranges_param_wrapper.GetParamTensorName()};
    qnn_model_wrapper.AddParamWrapper(std::move(ranges_param_wrapper));

    // begin_mask
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, begin_mask,
                                           QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK, param_names));

    // end_mask
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, end_mask,
                                           QNN_OP_STRIDED_SLICE_PARAM_END_MASK, param_names));

    // shrink_axes
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, shrink_axes,
                                           QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES, param_names));

    // new_axes_mask
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, new_axes_mask,
                                           QNN_OP_STRIDED_SLICE_PARAM_NEW_AXES_MASK, param_names));

    // outputs
    QnnTensorWrapper output_tensorwrapper(output_name, is_for_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                          tensor_data_type, quantize_param.Copy(), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                  "Failed to add output tensor for inserted StridedSlice.");
    // addNode
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE,
                                                  {input_name}, {output_name}, std::move(param_names), do_op_validation),
                  "Failed to create manually inserted Qnn StridedSlice node.");
  }
  return Ort::Status();
}

// Manually unrolls the GRU across time steps only (time_major=true).
// Each QNN GRU node processes a single time step with input shape [1, batch, input] (seq=1, time_major=true).
// The full batch is processed together in each step, avoiding the batch-element loop entirely.
//
// qnn_input_count: size of the QNN GRU input vector (14 for standard GRU, 15 for StatefulGru with reset).
// reset: optional stateful reset input; pass kNoReset() for the standard GRU builder.
Ort::Status AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit,
                               const std::string& direction,
                               const std::vector<std::string>& input_names,
                               const Ort::Logger& logger,
                               bool do_op_validation,
                               bool is_bidirection,
                               size_t qnn_input_count,
                               const ResetInput& reset,
                               std::vector<std::string>& uni_gru_output_names) {
  ORT_UNUSED_PARAMETER(logger);
  const auto& onnx_inputs = node_unit.Inputs();
  const auto& onnx_outputs = node_unit.Outputs();
  std::vector<TensorInfo> input_tensor_infos(onnx_inputs.size());
  for (size_t i = 0; i < onnx_inputs.size(); i++) {
    if (onnx_inputs[i].Exists()) {
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[i], input_tensor_infos[i]));
    }
  }
  std::vector<TensorInfo> output_tensor_infos(2);
  for (size_t i = 0; i < 2; i++) {
    if (onnx_outputs.size() > i && onnx_outputs[i].Exists()) {
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_outputs[i], output_tensor_infos[i]));
    } else {
      output_tensor_infos[i].qnn_data_type = input_tensor_infos[0].qnn_data_type;
    }
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t hidden_size_i64 = node_helper.Get("hidden_size", static_cast<int64_t>(0));
  RETURN_IF_NOT(hidden_size_i64 > 0, "hidden_size is not set for GRU");
  const uint32_t hidden_size = SafeInt<uint32_t>(hidden_size_i64);
  const int32_t hidden_size_sign = SafeInt<int32_t>(hidden_size_i64);
  const int64_t linear_before_reset = node_helper.Get("linear_before_reset", static_cast<int64_t>(0));
  const int64_t layout = node_helper.Get("layout", static_cast<int64_t>(0));

  const uint32_t input_size = input_tensor_infos[0].shape[2];
  const uint32_t batch_size = layout == 0 ? input_tensor_infos[0].shape[1] : input_tensor_infos[0].shape[0];
  const uint32_t seq_length = layout == 0 ? input_tensor_infos[0].shape[0] : input_tensor_infos[0].shape[1];
  const int32_t direction_idx = input_tensor_infos[1].shape[0] < 2 || direction == "forward" ? 0 : 1;

  // GRU parameters - shared by all unrolled cells. Direction is the ONNX direction; reverse
  // time-step ordering is handled by the unrolling loop, not by the QNN direction param.
  std::vector<std::string> param_names;
  const uint32_t qnn_direction = (direction == "reverse") ? QNN_OP_GRU_DIRECTION_REVERSE : QNN_OP_GRU_DIRECTION_FORWARD;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name() + "_" + direction,
                                         qnn_direction, QNN_OP_GRU_PARAM_DIRECTION, param_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(linear_before_reset),
                                         QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET, param_names));
  // TODO: Remove CPU time_major workaround once QNN CPU batch-dim accuracy issue is fixed.
  // time_major: CPU always false (batch dim bug); HTP follows layout attr (0->true, 1->false).
  const bool is_cpu_backend = qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::CPU;
  const bool time_major = is_cpu_backend ? false : (layout == 0);
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), time_major,
                                     QNN_OP_GRU_PARAM_TIME_MAJOR, param_names));

  const std::string null_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_null_tensor_" + direction);
  QnnTensorWrapper null_tensor_wrapper = QnnTensorWrapper::MakeNull(null_tensor_name);
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(null_tensor_wrapper)),
                "Failed to add null tensor for GRU.");

  // Base GRU input template (weights, biases) - shared across all unrolled cells.
  // Size qnn_input_count to accommodate optional reset slot (e.g., 15 for StatefulGru vs 14 for GRU).
  std::vector<std::string> qnn_gru_input_names(qnn_input_count, null_tensor_name);

  // Slice W, R, B weights (same as before - shared across all cells)
  // W: ONNX in[1] [num_directions, 3*hidden_size, input_size]
  {
    std::vector<uint32_t> qnn_idx = {1, 2, 3};
    std::vector<int32_t> begins = {0, 1, 2};
    std::vector<std::string> names = {
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_update_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_reset_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_new_gate_weight_" + direction)};
    for (size_t i = 0; i < 3; i++) {
      RETURN_IF_ERROR(AddStridedSlice(qnn_model_wrapper, node_unit, input_names[1], names[i],
                                      input_tensor_infos[1].shape, {hidden_size, input_size},
                                      {{direction_idx, direction_idx + 1, 1},
                                       {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
                                       {0, SafeInt<int32_t>(input_size), 1}},
                                      0, 0, 0b001U, 0, input_tensor_infos[1].qnn_data_type,
                                      input_tensor_infos[1].quant_param, do_op_validation, false, false));
      qnn_gru_input_names[qnn_idx[i]] = names[i];
    }
  }
  // R: ONNX in[2] [num_directions, 3*hidden_size, hidden_size]
  {
    std::vector<uint32_t> qnn_idx = {4, 5, 6};
    std::vector<int32_t> begins = {0, 1, 2};
    std::vector<std::string> names = {
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_update_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_reset_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_new_gate_weight_" + direction)};
    for (size_t i = 0; i < 3; i++) {
      RETURN_IF_ERROR(AddStridedSlice(qnn_model_wrapper, node_unit, input_names[2], names[i],
                                      input_tensor_infos[2].shape, {hidden_size, hidden_size},
                                      {{direction_idx, direction_idx + 1, 1},
                                       {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
                                       {0, hidden_size_sign, 1}},
                                      0, 0, 0b001U, 0, input_tensor_infos[2].qnn_data_type,
                                      input_tensor_infos[2].quant_param, do_op_validation, false, false));
      qnn_gru_input_names[qnn_idx[i]] = names[i];
    }
  }
  // B: ONNX in[3] [num_directions, 6*hidden_size]
  {
    std::vector<uint32_t> qnn_idx = {7, 8, 9, 10, 11, 12};
    if (onnx_inputs.size() > 3 && onnx_inputs[3].Exists()) {
      std::vector<int32_t> begins = {0, 1, 2, 3, 4, 5};
      std::vector<std::string> names = {
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_update_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_reset_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_new_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_update_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_reset_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_new_gate_bias_" + direction)};
      for (size_t i = 0; i < 6; i++) {
        RETURN_IF_ERROR(AddStridedSlice(qnn_model_wrapper, node_unit, input_names[3], names[i],
                                        input_tensor_infos[3].shape, {hidden_size},
                                        {{direction_idx, direction_idx + 1, 1},
                                         {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1}},
                                        0, 0, 0b01U, 0, input_tensor_infos[3].qnn_data_type,
                                        input_tensor_infos[3].quant_param, do_op_validation, false, false));
        qnn_gru_input_names[qnn_idx[i]] = names[i];
      }
    } else {
      std::string zero_bias_name = utils::UniqueNameGenerator().New(node_unit, "_zero_bias");
      QnnTensorWrapper zero_tw(zero_bias_name, QNN_TENSOR_TYPE_STATIC, input_tensor_infos[0].qnn_data_type,
                               QnnQuantParamsWrapper(), std::vector<uint32_t>{hidden_size},
                               std::vector<uint8_t>(utils::GetElementSizeByType(input_tensor_infos[0].qnn_data_type) * hidden_size, 0));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_tw)), "Failed to add zero bias.");
      for (size_t i = 0; i < 6; i++) qnn_gru_input_names[qnn_idx[i]] = zero_bias_name;
    }
  }
  // Optional stateful reset input: wire into the designated QNN slot if provided.
  if (!reset.onnx_name.empty()) {
    qnn_gru_input_names[reset.qnn_slot] = reset.onnx_name;
  }
  // initial_h: ONNX in[5] [num_directions, batch_size, hidden_size] -> [1, batch_size, hidden_size]
  std::string initial_h_name;
  {
    std::vector<uint32_t> h_shape = {1, batch_size, hidden_size};
    if (onnx_inputs.size() > 5 && onnx_inputs[5].Exists()) {
      if (input_tensor_infos[5].shape == h_shape) {
        // num_directions=1: already [1, batch, hidden], use directly
        initial_h_name = input_names[5];
      } else {
        // num_directions=2 (bidirectional): slice one direction
        initial_h_name = utils::UniqueNameGenerator().New(input_names[5], direction);
        RETURN_IF_ERROR(AddStridedSlice(qnn_model_wrapper, node_unit, input_names[5], initial_h_name,
                                        input_tensor_infos[5].shape, h_shape,
                                        {{direction_idx, direction_idx + 1, 1},
                                         {0, SafeInt<int32_t>(batch_size), 1},
                                         {0, hidden_size_sign, 1}},
                                        0, 0, 0, 0, input_tensor_infos[5].qnn_data_type,
                                        input_tensor_infos[5].quant_param, do_op_validation, false, false));
      }
    } else {
      initial_h_name = utils::UniqueNameGenerator().New(node_unit.Name(), "_GRU_initial_h");
      QnnTensorWrapper zero_h(initial_h_name, QNN_TENSOR_TYPE_STATIC, input_tensor_infos[0].qnn_data_type,
                              QnnQuantParamsWrapper(), std::vector<uint32_t>(h_shape),
                              std::vector<uint8_t>(utils::GetElementSizeByType(input_tensor_infos[0].qnn_data_type) * batch_size * hidden_size, 0));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_h)), "Failed to add initial hidden state.");
    }
  }

  // Unroll across time steps only.
  // Each step feeds the full batch with seq=1.
  // time_major=true:  input [1, batch, input], output Y [1, batch, hidden]
  // time_major=false: input [batch, 1, input], output Y [batch, 1, hidden]

  // For CPU (time_major=false), transpose X upfront from [seq, batch, input] to [batch, seq, input]
  // so that per-step slicing produces [batch, 1, input] directly without per-step Reshapes.
  std::string x_source = input_names[0];                               // [seq, batch, input] for HTP, [batch, seq, input] for CPU
  std::vector<uint32_t> x_source_shape = input_tensor_infos[0].shape;  // [seq, batch, input]
  if (is_cpu_backend) {
    std::string x_transposed = utils::UniqueNameGenerator().New(input_names[0], "_transposed_" + direction);
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        node_unit.Index(), input_names[0], x_transposed,
        input_tensor_infos[0].shape, {1, 0, 2}, {batch_size, seq_length, input_size},
        input_tensor_infos[0].qnn_data_type, input_tensor_infos[0].quant_param,
        do_op_validation, false, false));
    x_source = x_transposed;
    x_source_shape = {batch_size, seq_length, input_size};
  }

  std::vector<std::string> qnn_all_step_hidden_names(seq_length);  // Y tensors, indexed by t
  std::string prev_h_name = initial_h_name;                        // [1, batch, hidden]

  // If Y_h is a direct graph output for unidirectional, the last step can write it directly.
  const bool needs_y_h_output = !is_bidirection && onnx_outputs.size() > 1 && onnx_outputs[1].Exists();
  const std::string y_h_out_name = needs_y_h_output ? onnx_outputs[1].name : "";
  const bool y_h_is_graph_output = needs_y_h_output && qnn_model_wrapper.IsGraphOutput(y_h_out_name);

  for (uint32_t step = 0; step < seq_length; step++) {
    const bool is_last_step = (step == seq_length - 1);
    // For reverse direction iterate from the last time step to the first.
    const uint32_t t = (direction == "reverse") ? (seq_length - step - 1) : step;
    const std::string sfx = "_t" + std::to_string(t) + "_" + direction;

    std::vector<std::string> cell_inputs = qnn_gru_input_names;

    // Slice one time step from x_source:
    // time_major=true:  x_source[t:t+1, :, :] from [seq, batch, input] -> [1, batch, input]
    // time_major=false: x_source[:, t:t+1, :] from [batch, seq, input] -> [batch, 1, input]
    const std::vector<uint32_t> x_step_shape = time_major ? std::vector<uint32_t>{1, batch_size, input_size}
                                                          : std::vector<uint32_t>{batch_size, 1, input_size};
    const int32_t seq_dim = time_major ? 0 : 1;
    const int32_t batch_dim = time_major ? 1 : 0;
    std::vector<std::vector<int32_t>> x_ranges(3);
    x_ranges[seq_dim] = {SafeInt<int32_t>(t), SafeInt<int32_t>(t + 1), 1};
    x_ranges[batch_dim] = {0, SafeInt<int32_t>(batch_size), 1};
    x_ranges[2] = {0, SafeInt<int32_t>(input_size), 1};

    std::string x_step = utils::UniqueNameGenerator().New(input_names[0], "_xs" + sfx);
    RETURN_IF_ERROR(AddStridedSlice(qnn_model_wrapper, node_unit, x_source, x_step,
                                    x_source_shape, x_step_shape, x_ranges,
                                    0, 0, 0, 0, input_tensor_infos[0].qnn_data_type,
                                    input_tensor_infos[0].quant_param, do_op_validation, false, false));
    cell_inputs[0] = x_step;

    // initial_h for this step
    cell_inputs[13] = prev_h_name;

    // GRU outputs:
    // time_major=true:  Y [1, batch, hidden], Y_h [1, batch, hidden]
    // time_major=false: Y [batch, 1, hidden], Y_h [1, batch, hidden]
    const std::vector<uint32_t> y_shape = time_major ? std::vector<uint32_t>{1, batch_size, hidden_size}
                                                     : std::vector<uint32_t>{batch_size, 1, hidden_size};
    std::string y_name = utils::UniqueNameGenerator().New(node_unit, "_Y" + sfx);

    // For the last step on HTP: name Y_h after the ONNX output directly.
    // CPU derives h from Y via Transpose instead (Y_h is unreliable on CPU).
    const bool write_yh_directly = is_last_step && needs_y_h_output && time_major;
    std::string yh_name = write_yh_directly ? y_h_out_name
                                            : utils::UniqueNameGenerator().New(node_unit, "_Yh" + sfx);
    {
      QnnTensorWrapper y_tw(y_name, QNN_TENSOR_TYPE_NATIVE, output_tensor_infos[0].qnn_data_type,
                            output_tensor_infos[0].quant_param.Copy(), std::vector<uint32_t>(y_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(y_tw)), "Failed to add GRU Y output.");
      Qnn_TensorType_t yh_type = (write_yh_directly && y_h_is_graph_output) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
      QnnTensorWrapper yh_tw(yh_name, yh_type, output_tensor_infos[1].qnn_data_type,
                             output_tensor_infos[1].quant_param.Copy(), std::vector<uint32_t>{1, batch_size, hidden_size});
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(yh_tw)), "Failed to add GRU Y_h output.");
    }
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, "_cell" + sfx),
                      QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GRU,
                      std::move(cell_inputs), {y_name, yh_name},
                      std::vector<std::string>(param_names), do_op_validation),
                  "Failed to create GRU node.");

    // Derive next step's initial_h.
    // QNN CPU's Y_h (out[1]) is unreliable, so on CPU we derive h from Y (out[0]) via Transpose.
    // On HTP (time_major=true), Y_h (out[1]) is already [1, batch, hidden] and can be used directly.
    if (!is_cpu_backend) {
      // HTP: use Y_h directly as next step's initial_h
      prev_h_name = yh_name;
    } else {
      // CPU: Y is [batch, 1, hidden] - Transpose to [1, batch, hidden].
      // On the last step for unidirectional, write directly to the ONNX output (is_for_output=true).
      const bool write_as_output = is_last_step && needs_y_h_output;
      const std::string y_as_h = write_as_output ? y_h_out_name
                                                 : utils::UniqueNameGenerator().New(node_unit, "_Y_as_h" + sfx);
      RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
          node_unit.Index(), y_name, y_as_h,
          y_shape, {1, 0, 2}, {1, batch_size, hidden_size},
          output_tensor_infos[0].qnn_data_type, output_tensor_infos[0].quant_param,
          do_op_validation, false, write_as_output && y_h_is_graph_output));
      prev_h_name = y_as_h;
    }

    // Collect Y for Concat (no per-step Reshape needed).
    qnn_all_step_hidden_names[t] = y_name;
  }

  // Concat all per-step Y tensors into [seq, batch, hidden].
  // HTP (time_major=true):  Y [1,batch,hidden] * seq, Concat axis=0 -> [seq, batch, hidden]
  // CPU (time_major=false): Y [batch,1,hidden] * seq, Concat axis=1 -> [batch, seq, hidden]
  //                         then Transpose -> [seq, batch, hidden]
  const uint32_t concat_axis = time_major ? 0 : 1;
  const std::string y_concat = utils::UniqueNameGenerator().New(node_unit, "_Y_concat_" + direction);
  {
    std::vector<uint32_t> concat_out_shape = time_major
                                                 ? std::vector<uint32_t>{seq_length, batch_size, hidden_size}
                                                 : std::vector<uint32_t>{batch_size, seq_length, hidden_size};
    std::vector<std::string> cp;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), y_concat,
                                           concat_axis, QNN_OP_CONCAT_PARAM_AXIS, cp));
    QnnTensorWrapper tw(y_concat, QNN_TENSOR_TYPE_NATIVE, output_tensor_infos[0].qnn_data_type,
                        output_tensor_infos[0].quant_param.Copy(), std::move(concat_out_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "Failed to add Y Concat output.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(y_concat, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                                  std::move(qnn_all_step_hidden_names), {y_concat}, std::move(cp), do_op_validation),
                  "Failed to create Y Concat node.");
  }

  // For CPU (time_major=false), Transpose [batch, seq, hidden] -> [seq, batch, hidden].
  std::string y_all;
  if (is_cpu_backend) {
    y_all = utils::UniqueNameGenerator().New(node_unit, "_Y_all_" + direction);
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        node_unit.Index(), y_concat, y_all,
        {batch_size, seq_length, hidden_size}, {1, 0, 2}, {seq_length, batch_size, hidden_size},
        output_tensor_infos[0].qnn_data_type, output_tensor_infos[0].quant_param,
        do_op_validation, false, false));
  } else {
    y_all = y_concat;
  }

  // Map to ONNX output shapes:
  //   Y:   [seq, batch, hidden] -> ONNX [seq, 1, batch, hidden]  (Reshape to insert num_directions dim)
  //   Y_h: [1, batch, hidden]   -> ONNX [1, batch, hidden]       (already correct shape, use prev_h_name)
  for (size_t i = 0; i < 2; i++) {
    if (onnx_outputs.size() > i && onnx_outputs[i].Exists()) {
      if (i == 0) {
        // Y: Reshape [seq, batch, hidden] -> [seq, 1, batch, hidden]
        const std::string out_name = is_bidirection
                                         ? utils::UniqueNameGenerator().New(y_all, "_unsqueeze_" + direction)
                                         : onnx_outputs[i].name;
        RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(y_all, out_name,
                                                         {seq_length, batch_size, hidden_size},
                                                         {seq_length, 1, batch_size, hidden_size},
                                                         output_tensor_infos[0].qnn_data_type,
                                                         output_tensor_infos[0].quant_param, do_op_validation, false,
                                                         qnn_model_wrapper.IsGraphOutput(out_name)));
        uni_gru_output_names.emplace_back(out_name);
      } else {
        // Y_h: already written under onnx_outputs[1].name in the last step (APP_READ or NATIVE).
        if (!is_bidirection) {
          uni_gru_output_names.emplace_back(y_h_out_name);
        } else {
          // Bidirectional: pass directly to Concat.
          uni_gru_output_names.emplace_back(prev_h_name);
        }
      }
    } else {
      uni_gru_output_names.emplace_back("");
    }
  }
  return Ort::Status();
}

// reset: optional stateful reset input; pass kNoReset() for the standard LSTM builder.
Ort::Status AddUnidirectionLSTM(QnnModelWrapper& qnn_model_wrapper,
                                const OrtNodeUnit& node_unit,
                                const std::string& direction,
                                const std::vector<std::string>& input_names,
                                const Ort::Logger& logger,
                                bool do_op_validation,
                                bool is_bidirection,
                                const ResetInput& reset,
                                std::vector<std::string>& uni_lstm_output_names) {
  ORT_UNUSED_PARAMETER(logger);
  const auto& onnx_inputs = node_unit.Inputs();
  const auto& onnx_outputs = node_unit.Outputs();
  const std::string& node_name = node_unit.Name();
  std::vector<TensorInfo> input_tensor_infos(onnx_inputs.size());
  for (size_t i = 0; i < onnx_inputs.size(); i++) {
    if (onnx_inputs[i].Exists()) {
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[i], input_tensor_infos[i]));
    }
  }
  // because QNN LSTM three outputs are mandatory, we should provide them tensor info
  std::vector<TensorInfo> output_tensor_infos(3);
  for (size_t i = 0; i < 3; i++) {
    if (onnx_outputs.size() > i && onnx_outputs[i].Exists()) {
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_outputs[i], output_tensor_infos[i]));
    } else {
      output_tensor_infos[i].qnn_data_type = input_tensor_infos[0].qnn_data_type;
    }
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const uint32_t hidden_size = node_helper.Get("hidden_size", 0);
  const int32_t hidden_size_sign = SafeInt<int32_t>(hidden_size);
  RETURN_IF_NOT(hidden_size > 0, "hidden size is not set for LSTM");
  const int64_t layout = node_helper.Get("layout", static_cast<int64_t>(0));

  const uint32_t input_size = input_tensor_infos[0].shape[2];
  const uint32_t batch_size = layout == 0 ? input_tensor_infos[0].shape[1] : input_tensor_infos[0].shape[0];
  const uint32_t seq_length = layout == 0 ? input_tensor_infos[0].shape[0] : input_tensor_infos[0].shape[1];
  const int32_t direction_idx = input_tensor_infos[1].shape[0] < 2 || direction == "forward" ? 0 : 1;

  // When false (default), lower LSTM as ORT-QNN-EP side per-timestep unrolled QNN_OP_LSTM cells + Pack.
  // When true, lower as a single monolithic QNN_OP_LSTM node and rely on HTP's native
  // monolithic LSTM kernel (enabled via enable_htp_monolithic_lstm provider option, which also
  // toggles the QNN_HTP_GRAPH_CONFIG_OPTION_MONOLITHIC_LSTM graph config).
  //
  // The monolithic path is only meaningful for the HTP/NPU backend.
  // The IR backend is also allowed to honor the flag
  // so that a serialized DLC can faithfully mirror the HTP monolithic graph — otherwise an
  // IR-produced DLC (always unrolled) could not be compared against the HTP monolithic one.
  // All other backends (CPU/GPU) have no monolithic kernel and always use the unrolled lowering.
  // A stateful reset forces monolithic because in[24] only exists in the 3D monolithic op (per QAIRT MasterOpDef).
  const bool has_reset = !reset.onnx_name.empty();
  const bool enable_htp_monolithic_lstm = has_reset ||
                                          ((IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) ||
                                            IsIrBackend(qnn_model_wrapper.GetQnnBackendType())) &&
                                           qnn_model_wrapper.GetModelSettings().enable_htp_monolithic_lstm);

  // params
  std::vector<std::string> param_names;

  // direction
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name() + "_" + direction,
                                         direction == "forward" ? QNN_OP_LSTM_DIRECTION_FORWARD : QNN_OP_LSTM_DIRECTION_REVERSE,
                                         QNN_OP_LSTM_PARAM_DIRECTION, param_names));

  // cell_clip_threshold
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                      QNN_OP_LSTM_PARAM_CELL_CLIP_THRESHOLD, param_names));

  // output_clip_threshold
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                      QNN_OP_LSTM_PARAM_OUTPUT_CLIP_THRESHOLD, param_names));

  // time_major
  // monolithic QNN_OP_LSTM consumes the whole 3D sequence at once (time_major=true);
  // the unrolled path feeds one 2D timestep per cell node (time_major=false).
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), enable_htp_monolithic_lstm,
                                     QNN_OP_LSTM_PARAM_TIME_MAJOR, param_names));

  // input_gate_qscale
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                      QNN_OP_LSTM_PARAM_INPUT_GATE_QSCALE, param_names));

  // forget_gate_qscale
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                      QNN_OP_LSTM_PARAM_FORGET_GATE_QSCALE, param_names));

  // cell_gate_qscale
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                      QNN_OP_LSTM_PARAM_CELL_GATE_QSCALE, param_names));

  // output_gate_qscale
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                      QNN_OP_LSTM_PARAM_OUTPUT_GATE_QSCALE, param_names));

  // hidden_state_offset
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                      QNN_OP_LSTM_PARAM_HIDDEN_STATE_OFFSET, param_names));

  // hidden_state_qscale
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                      QNN_OP_LSTM_PARAM_HIDDEN_STATE_QSCALE, param_names));

  // Common LSTM cell inputs
  const std::string null_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_null_tensor_" + direction);
  QnnTensorWrapper null_tensor_wrapper = QnnTensorWrapper::MakeNull(null_tensor_name);

  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(null_tensor_wrapper)),
                "Failed to add null tensor for LSTM.");

  std::vector<std::string> qnn_lstm_input_names(enable_htp_monolithic_lstm ? 25 : 24, null_tensor_name);
  if (enable_htp_monolithic_lstm) {
    qnn_lstm_input_names[0] = input_names[0];
  }
  // input W
  {
    // QNN in[1] = ONNX in[1][direction, 2*hidden_size:3*hidden_size, :]
    // QNN in[2] = ONNX in[1][direction, 3*hidden_size:4*hidden_size, :]
    // QNN in[3] = ONNX in[1][direction, 1*hidden_size:2*hidden_size, :]
    // QNN in[16] = ONNX in[1][direction, 0*hidden_size:1*hidden_size, :]
    uint32_t begin_mask = 0b000U;
    uint32_t end_mask = 0b000U;
    uint32_t shrink_axes = 0b001U;
    uint32_t new_axes_mask = 0b000U;
    std::vector<uint32_t> qnn_input_indices = {1, 2, 3, 16};
    std::vector<int32_t> begins = {2, 3, 1, 0};
    std::vector<std::string> qnn_lstm_weight_name = {
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_forget_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_cell_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_output_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_input_gate_weight_" + direction),
    };
    for (size_t i = 0; i < 4; i++) {
      std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                  {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
                                                  {0, SafeInt<int32_t>(input_size), 1}};
      std::vector<uint32_t> output_shape = {hidden_size, input_size};
      RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                               /*node_unit=*/node_unit,
                                               /*input_name=*/input_names[1],
                                               /*output_name=*/qnn_lstm_weight_name[i],
                                               /*input_shape=*/input_tensor_infos[1].shape,
                                               /*output_shape=*/output_shape,
                                               /*ranges=*/ranges,
                                               /*begin_mask=*/begin_mask,
                                               /*end_mask=*/end_mask,
                                               /*shrink_axes=*/shrink_axes,
                                               /*new_axes_mask=*/new_axes_mask,
                                               /*tensor_data_type=*/input_tensor_infos[1].qnn_data_type,
                                               /*QnnQuantParamsWrapper=*/input_tensor_infos[1].quant_param,
                                               /*do_op_validation=*/do_op_validation,
                                               /*is_for_input=*/false,
                                               /*is_for_output=*/false));
      qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_weight_name[i];
    }
  }

  // input R
  {
    // QNN in[4] = ONNX in[2][direction, 2*hidden_size:3*hidden_size, :]
    // QNN in[5] = ONNX in[2][direction, 3*hidden_size:4*hidden_size, :]
    // QNN in[6] = ONNX in[2][direction, 1*hidden_size:2*hidden_size, :]
    // QNN in[17] = ONNX in[2][direction, 0*hidden_size:1*hidden_size, :]
    uint32_t begin_mask = 0b000U;
    uint32_t end_mask = 0b000U;
    uint32_t shrink_axes = 0b001U;
    uint32_t new_axes_mask = 0b000U;
    std::vector<uint32_t> qnn_input_indices = {4, 5, 6, 17};
    std::vector<int32_t> begins = {2, 3, 1, 0};
    std::vector<std::string> qnn_lstm_weight_name = {
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_forget_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_cell_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_output_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_input_gate_weight_" + direction)};
    for (size_t i = 0; i < 4; i++) {
      std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                  {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
                                                  {0, hidden_size_sign, 1}};
      std::vector<uint32_t> output_shape = {hidden_size, hidden_size};
      RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                               /*node_unit=*/node_unit,
                                               /*input_name=*/input_names[2],
                                               /*output_name=*/qnn_lstm_weight_name[i],
                                               /*input_shape=*/input_tensor_infos[2].shape,
                                               /*output_shape=*/output_shape,
                                               /*ranges=*/ranges,
                                               /*begin_mask=*/begin_mask,
                                               /*end_mask=*/end_mask,
                                               /*shrink_axes=*/shrink_axes,
                                               /*new_axes_mask=*/new_axes_mask,
                                               /*tensor_data_type=*/input_tensor_infos[2].qnn_data_type,
                                               /*QnnQuantParamsWrapper=*/input_tensor_infos[2].quant_param,
                                               /*do_op_validation=*/do_op_validation,
                                               /*is_for_input=*/false,
                                               /*is_for_output=*/false));
      qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_weight_name[i];
    }
  }

  // input B
  {
    // QNN in[7] = ONNX in[3][direction, 2*hidden_size:3*hidden_size] + ONNX in[3][direction, 6*hidden_size:7*hidden_size]
    // QNN in[8] = ONNX in[3][direction, 3*hidden_size:4*hidden_size] + ONNX in[3][direction, 7*hidden_size:8*hidden_size]
    // QNN in[9] = ONNX in[3][direction, 1*hidden_size:2*hidden_size] + ONNX in[3][direction, 5*hidden_size:6*hidden_size]
    // QNN in[21] = ONNX in[3][direction, 0*hidden_size:1*hidden_size] + ONNX in[3][direction, 4*hidden_size:5*hidden_size]
    uint32_t begin_mask = 0b00U;
    uint32_t end_mask = 0b00U;
    uint32_t shrink_axes = 0b01U;
    uint32_t new_axes_mask = 0b00U;
    std::vector<uint32_t> output_shape = {hidden_size};
    std::vector<std::string> qnn_lstm_bias_name = {
        utils::UniqueNameGenerator().New(node_unit, "_forget_gate_bias_" + direction),
        utils::UniqueNameGenerator().New(node_unit, "_cell_gate_bias_" + direction),
        utils::UniqueNameGenerator().New(node_unit, "_output_gate_bias_" + direction),
        utils::UniqueNameGenerator().New(node_unit, "_input_gate_bias_" + direction)};
    std::vector<uint32_t> qnn_input_indices = {7, 8, 9, 21};
    if (onnx_inputs.size() > 3 && onnx_inputs[3].Exists()) {
      std::vector<int32_t> begins = {2, 3, 1, 0, 6, 7, 5, 4};
      std::vector<std::string> onnx_lstm_bias_name = {
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_forget_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_cell_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_output_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_input_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_forget_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_cell_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_output_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_input_gate_bias_" + direction)};
      for (size_t i = 0; i < 8; i++) {
        std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                    {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1}};
        RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                 /*node_unit=*/node_unit,
                                                 /*input_name=*/input_names[3],
                                                 /*output_name=*/onnx_lstm_bias_name[i],
                                                 /*input_shape=*/input_tensor_infos[3].shape,
                                                 /*output_shape=*/output_shape,
                                                 /*ranges=*/ranges,
                                                 /*begin_mask=*/begin_mask,
                                                 /*end_mask=*/end_mask,
                                                 /*shrink_axes=*/shrink_axes,
                                                 /*new_axes_mask=*/new_axes_mask,
                                                 /*tensor_data_type=*/input_tensor_infos[3].qnn_data_type,
                                                 /*QnnQuantParamsWrapper=*/input_tensor_infos[3].quant_param,
                                                 /*do_op_validation=*/do_op_validation,
                                                 /*is_for_input=*/false,
                                                 /*is_for_output=*/false));
      }
      for (size_t i = 0; i < 4; i++) {
        std::vector<std::string> add_input_names = {onnx_lstm_bias_name[i], onnx_lstm_bias_name[i + 4]};
        // TODO: The quantize_param should not be used directly, we should calculate an approximate quant_param here.
        QnnTensorWrapper add_output_tensorwrapper(qnn_lstm_bias_name[i], QNN_TENSOR_TYPE_NATIVE, input_tensor_infos[3].qnn_data_type,
                                                  input_tensor_infos[3].quant_param.Copy(), std::vector<uint32_t>(output_shape));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(add_output_tensorwrapper)),
                      "QNN EP: Failed to add output tensor for inserted ElementWiseAdd node.");
        std::string add_node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_BINARY);
        std::vector<std::string> add_param_names;
        RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), add_node_name,
                                               static_cast<uint32_t>(QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD),
                                               QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, add_param_names));
        RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(add_node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_BINARY,
                                                      std::move(add_input_names),
                                                      {qnn_lstm_bias_name[i]},
                                                      std::move(add_param_names),
                                                      do_op_validation),
                      "Failed to create manually inserted ElementWiseAdd node.");
        qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_bias_name[i];
      }
    } else {
      // prepare zero bias
      std::string zero_bias_name = utils::UniqueNameGenerator().New(node_unit, "_zero_bias");
      QnnTensorWrapper zero_bias_tensor_wrapper(zero_bias_name,
                                                QNN_TENSOR_TYPE_STATIC,
                                                input_tensor_infos[0].qnn_data_type,
                                                QnnQuantParamsWrapper(),
                                                std::vector<uint32_t>(output_shape),
                                                std::vector<uint8_t>(
                                                    utils::GetElementSizeByType(input_tensor_infos[0].qnn_data_type) * hidden_size,
                                                    0));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_bias_tensor_wrapper)),
                    "Failed to add additional zero bias for QNN LSTM node.");
      for (size_t i = 0; i < 4; i++) {
        qnn_lstm_input_names[qnn_input_indices[i]] = zero_bias_name;
      }
    }
  }

  // input P
  if (onnx_inputs.size() > 7 && onnx_inputs[7].Exists()) {
    // QNN in[18] = ONNX in[7][direction, 0*hidden_size:1*hidden_size]
    // QNN in[19] = ONNX in[7][direction, 2*hidden_size:3*hidden_size]
    // QNN in[20] = ONNX in[7][direction, 1*hidden_size:2*hidden_size]
    uint32_t begin_mask = 0b00U;
    uint32_t end_mask = 0b00U;
    uint32_t shrink_axes = 0b01U;
    uint32_t new_axes_mask = 0b00U;
    std::vector<uint32_t> output_shape = {hidden_size};
    std::vector<uint32_t> qnn_input_indices = {18, 19, 20};
    std::vector<int32_t> begins = {0, 2, 1};
    std::vector<std::string> qnn_lstm_weight_name = {
        utils::UniqueNameGenerator().New(input_names[7], "_cell_to_input_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[7], "_cell_to_forget_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[7], "_cell_to_output_gate_weight_" + direction)};
    for (size_t i = 0; i < 3; i++) {
      std::vector<std::vector<int32_t>> ranges = {
          {direction_idx, direction_idx + 1, 1},
          {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
      };
      RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                               /*node_unit=*/node_unit,
                                               /*input_name=*/input_names[7],
                                               /*output_name=*/qnn_lstm_weight_name[i],
                                               /*input_shape=*/input_tensor_infos[7].shape,
                                               /*output_shape=*/output_shape,
                                               /*ranges=*/ranges,
                                               /*begin_mask=*/begin_mask,
                                               /*end_mask=*/end_mask,
                                               /*shrink_axes=*/shrink_axes,
                                               /*new_axes_mask=*/new_axes_mask,
                                               /*tensor_data_type=*/input_tensor_infos[7].qnn_data_type,
                                               /*QnnQuantParamsWrapper=*/input_tensor_infos[7].quant_param,
                                               /*do_op_validation=*/do_op_validation,
                                               /*is_for_input=*/false,
                                               /*is_for_output=*/false));
      qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_weight_name[i];
    }
  }

  // input initial h, c
  {
    // QNN in[10] = ONNX in[5][direction_idx, :, :]
    // QNN in[11] = ONNX in[6][direction_idx, :, :]
    uint32_t begin_mask = 0b000U;
    uint32_t end_mask = 0b000U;
    uint32_t shrink_axes = 0b001U;
    uint32_t new_axes_mask = 0b000U;
    std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                {0, SafeInt<int32_t>(batch_size), 1},
                                                {0, hidden_size_sign, 1}};
    std::vector<uint32_t> src_indices = {5, 6};
    std::vector<uint32_t> qnn_input_indices = {10, 11};
    std::vector<uint32_t> output_shape = {batch_size, hidden_size};
    for (size_t i = 0; i < 2; i++) {
      if (onnx_inputs.size() > src_indices[i] && onnx_inputs[src_indices[i]].Exists()) {
        const std::string qnn_lstm_input_name = utils::UniqueNameGenerator().New(input_names[src_indices[i]], direction);
        RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                 /*node_unit=*/node_unit,
                                                 /*input_name=*/input_names[src_indices[i]],
                                                 /*output_name=*/qnn_lstm_input_name,
                                                 /*input_shape=*/input_tensor_infos[src_indices[i]].shape,
                                                 /*output_shape=*/output_shape,
                                                 /*ranges=*/ranges,
                                                 /*begin_mask=*/begin_mask,
                                                 /*end_mask=*/end_mask,
                                                 /*shrink_axes=*/shrink_axes,
                                                 /*new_axes_mask=*/new_axes_mask,
                                                 /*tensor_data_type=*/input_tensor_infos[src_indices[i]].qnn_data_type,
                                                 /*QnnQuantParamsWrapper=*/input_tensor_infos[src_indices[i]].quant_param,
                                                 /*do_op_validation=*/do_op_validation,
                                                 /*is_for_input=*/false,
                                                 /*is_for_output=*/false));
        qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_input_name;
      } else {
        // prepare zero initial values
        std::string zero_initial_values_name = utils::UniqueNameGenerator().New(node_name, std::string("_LSTM_initial_values_") + (i == 0 ? "h" : "c"));
        QnnTensorWrapper zero_bias_tensor_wrapper(zero_initial_values_name,
                                                  QNN_TENSOR_TYPE_STATIC,
                                                  input_tensor_infos[0].qnn_data_type,
                                                  QnnQuantParamsWrapper(),
                                                  std::vector<uint32_t>(output_shape),
                                                  std::vector<uint8_t>(
                                                      utils::GetElementSizeByType(input_tensor_infos[0].qnn_data_type) * batch_size * hidden_size,
                                                      0));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_bias_tensor_wrapper)),
                      "Failed to add additional initial values for QNN LSTM node.");
        qnn_lstm_input_names[qnn_input_indices[i]] = zero_initial_values_name;
      }
    }
  }

  // Optional stateful reset input: wire into the designated QNN slot if provided. A non-empty reset
  // forces enable_htp_monolithic_lstm above, so qnn_lstm_input_names is the 25-slot monolithic array
  // here and reset.qnn_slot (24) is in range. The unrolled path is never reached with a reset present.
  if (has_reset) {
    RETURN_IF_NOT(reset.qnn_slot < qnn_lstm_input_names.size(),
                  "QNN EP: stateful reset slot out of range for LSTM input vector.");
    qnn_lstm_input_names[reset.qnn_slot] = reset.onnx_name;
  }

  // outputs
  std::vector<std::vector<uint32_t>> qnn_lstm_output_shapes = {
      {seq_length, batch_size, hidden_size},
      {batch_size, hidden_size},
      {batch_size, hidden_size}};

  // reshape_input_names ends up holding, in ONNX output order [Y, Y_h, Y_c], the QNN tensor
  // name that feeds the final reshape-to-ONNX-shape step below.
  std::vector<std::string> reshape_input_names(3);

  // QNN output order: all_hidden, last_cell, last_hidden
  // ONNX output order: all_hidden, last_hidden, last_cell
  // so output_tensor_infos (ONNX order) must be mapped to QNN output position.
  static constexpr int kOutputPositionMap[] = {0, 2, 1};
  auto emit_lstm_outputs = [&](const std::vector<std::string>& out_names,
                               const std::vector<std::vector<uint32_t>>& out_shapes) -> Ort::Status {
    for (size_t j = 0; j < 3; j++) {
      QnnTensorWrapper output_tensorwrapper(out_names[j],
                                            QNN_TENSOR_TYPE_NATIVE,
                                            output_tensor_infos[kOutputPositionMap[j]].qnn_data_type,
                                            output_tensor_infos[kOutputPositionMap[j]].quant_param.Copy(),
                                            std::vector<uint32_t>(out_shapes[j]));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                    ("QNN EP: Failed to add " + std::to_string(j) + "th output tensor for QNN LSTM.").c_str());
    }
    return Ort::Status();
  };

  if (!enable_htp_monolithic_lstm) {
    // Unrolled path: HTP does not have a fast native path for a single 3D QNN_OP_LSTM node,
    // so emit one QNN_OP_LSTM cell per timestep (2D in/out) and chain hidden/cell state
    // between cells, then Pack all per-timestep hidden states back into a 3D tensor.
    std::vector<std::string> qnn_all_hidden_state_names(seq_length);
    for (uint32_t i = 0; i < seq_length; i++) {
      uint32_t sequence_idx = direction == "forward" ? i : seq_length - i - 1;
      std::vector<std::string> qnn_lstm_input_names_i = qnn_lstm_input_names;

      // input X: QNN in[0] = ONNX in[0][sequence_idx, :, :]
      {
        uint32_t begin_mask = 0b000U;
        uint32_t end_mask = 0b000U;
        uint32_t shrink_axes = 0b001U;
        uint32_t new_axes_mask = 0b000U;
        std::vector<std::vector<int32_t>> ranges = {{SafeInt<int32_t>(sequence_idx), SafeInt<int32_t>(sequence_idx + 1), 1},
                                                    {0, SafeInt<int32_t>(batch_size), 1},
                                                    {0, SafeInt<int32_t>(input_size), 1}};
        std::string qnn_lstm_input_name = utils::UniqueNameGenerator().New(input_names[0], "_cell_" + std::to_string(sequence_idx) + "_input");
        std::vector<uint32_t> output_shape = {batch_size, input_size};
        RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                 /*node_unit=*/node_unit,
                                                 /*input_name=*/input_names[0],
                                                 /*output_name=*/qnn_lstm_input_name,
                                                 /*input_shape=*/input_tensor_infos[0].shape,
                                                 /*output_shape=*/output_shape,
                                                 /*ranges=*/ranges,
                                                 /*begin_mask=*/begin_mask,
                                                 /*end_mask=*/end_mask,
                                                 /*shrink_axes=*/shrink_axes,
                                                 /*new_axes_mask=*/new_axes_mask,
                                                 /*tensor_data_type=*/input_tensor_infos[0].qnn_data_type,
                                                 /*QnnQuantParamsWrapper=*/input_tensor_infos[0].quant_param,
                                                 /*do_op_validation=*/do_op_validation,
                                                 /*is_for_input=*/false,
                                                 /*is_for_output=*/false));
        qnn_lstm_input_names_i[0] = qnn_lstm_input_name;
      }

      std::vector<uint32_t> qnn_lstm_output_shape = {batch_size, hidden_size};
      std::vector<std::string> qnn_lstm_output_names = {
          utils::UniqueNameGenerator().New(node_unit, "_QNN_LSTM_output_all_hidden_state_" + std::to_string(sequence_idx) + "_" + direction),
          utils::UniqueNameGenerator().New(node_unit, "_QNN_LSTM_output_cell_state_" + std::to_string(sequence_idx) + "_" + direction),
          utils::UniqueNameGenerator().New(node_unit, "_QNN_LSTM_output_hidden_state_" + std::to_string(sequence_idx) + "_" + direction)};
      qnn_lstm_input_names[10] = qnn_lstm_output_names[2];  // update initial_h for next cell
      qnn_lstm_input_names[11] = qnn_lstm_output_names[1];  // update initial_c for next cell
      // Single-timestep cell, so out[0] (all_hidden) == out[2] (final hidden state); reuse out[2].
      qnn_all_hidden_state_names[sequence_idx] = qnn_lstm_output_names[2];

      RETURN_IF_ERROR(emit_lstm_outputs(qnn_lstm_output_names,
                                        {qnn_lstm_output_shape, qnn_lstm_output_shape, qnn_lstm_output_shape}));
      const std::string lstm_node_name = utils::UniqueNameGenerator().New(node_unit, "_cell" + std::to_string(sequence_idx) + "_" + direction);
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(lstm_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_LSTM,
                                                    std::move(qnn_lstm_input_names_i), std::move(qnn_lstm_output_names),
                                                    std::vector<std::string>(param_names), do_op_validation),
                    "QNN EP: Failed to create Qnn LSTM node.");
    }

    // Pack all per-timestep hidden states together for ONNX output[0] (Y).
    const std::string qnn_pack_output_name = utils::UniqueNameGenerator().New(node_unit, "_QNN_LSTM_output_hidden_state_all_" + direction);
    std::vector<std::string> pack_param_names;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), qnn_pack_output_name, 0,
                                           QNN_OP_PACK_PARAM_AXIS, pack_param_names));

    QnnTensorWrapper pack_output_tensorwrapper(qnn_pack_output_name,
                                               QNN_TENSOR_TYPE_NATIVE,
                                               output_tensor_infos[0].qnn_data_type,
                                               output_tensor_infos[0].quant_param.Copy(),
                                               {seq_length, batch_size, hidden_size});
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(pack_output_tensorwrapper)),
                  "QNN EP: Failed to add output tensor for QNN Pack.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(qnn_pack_output_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PACK,
                                                  std::move(qnn_all_hidden_state_names), {qnn_pack_output_name},
                                                  std::move(pack_param_names), do_op_validation),
                  "QNN EP: Failed to create Qnn Pack node.");

    // qnn_lstm_input_names[10]/[11] now hold the final timestep's hidden/cell state (Y_h, Y_c).
    reshape_input_names = {qnn_pack_output_name, qnn_lstm_input_names[10], qnn_lstm_input_names[11]};
  } else {
    // Monolithic path: a single QNN_OP_LSTM node consumes/produces the whole 3D sequence
    // at once, relying on HTP's native monolithic LSTM kernel (QNN_HTP_GRAPH_CONFIG_OPTION_MONOLITHIC_LSTM).
    // This is also the path taken whenever a stateful reset input is present (reset lives at in[24]).
    std::vector<std::string> qnn_lstm_output_names = {
        utils::UniqueNameGenerator().New(node_unit, "_QNN_LSTM_output_all_hidden_state_" + direction),
        utils::UniqueNameGenerator().New(node_unit, "_QNN_LSTM_output_cell_state_" + direction),
        utils::UniqueNameGenerator().New(node_unit, "_QNN_LSTM_output_hidden_state_" + direction)};

    // QNN output order: all_hidden, last_cell, last_hidden
    // ONNX output order: all_hidden, last_hidden, last_cell
    RETURN_IF_ERROR(emit_lstm_outputs(qnn_lstm_output_names, qnn_lstm_output_shapes));
    const std::string lstm_node_name = utils::UniqueNameGenerator().New(node_unit, "_" + direction);
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(lstm_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_LSTM,
                                                  std::move(qnn_lstm_input_names), std::vector<std::string>(qnn_lstm_output_names),
                                                  std::vector<std::string>(param_names), do_op_validation),
                  "QNN EP: Failed to create Qnn LSTM node.");
    for (size_t i = 0; i < 3; i++) {
      reshape_input_names[i] = qnn_lstm_output_names[kOutputPositionMap[i]];
    }
  }

  // in the output shapes below, the value of 1 indicates unidirectional
  std::vector<std::vector<uint32_t>> onnx_lstm_output_shapes = {
      {seq_length, 1, batch_size, hidden_size},
      {1, batch_size, hidden_size},
      {1, batch_size, hidden_size}};
  for (size_t i = 0; i < 3; i++) {
    // if bidirection, return output names for concat
    if (onnx_outputs.size() > i && onnx_outputs[i].Exists()) {
      const std::string reshape_output_name = is_bidirection ? utils::UniqueNameGenerator().New(reshape_input_names[i], "_unsqueeze_" + direction) : onnx_outputs[i].name;
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(/*input_name=*/reshape_input_names[i],
                                                       /*output_name=*/reshape_output_name,
                                                       /*input_shape=*/qnn_lstm_output_shapes[i],
                                                       /*output_shape=*/onnx_lstm_output_shapes[i],
                                                       /*tensor_data_type=*/output_tensor_infos[i].qnn_data_type,
                                                       /*quantize_param=*/output_tensor_infos[i].quant_param,
                                                       /*do_op_validation=*/do_op_validation,
                                                       /*is_for_input=*/false,
                                                       /*is_for_output=*/qnn_model_wrapper.IsGraphOutput(reshape_output_name)));
      uni_lstm_output_names.emplace_back(reshape_output_name);
    } else {
      uni_lstm_output_names.emplace_back("");
    }
  }
  return Ort::Status();
}

}  // namespace rnn_details
}  // namespace qnn
}  // namespace onnxruntime
