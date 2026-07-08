// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class GRUOpBuilder : public BaseOpBuilder {
 public:
  GRUOpBuilder() : BaseOpBuilder("GRUOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GRUOpBuilder);

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

 private:
  Ort::Status AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                                 const OrtNodeUnit& node_unit,
                                 const std::string& direction,
                                 const std::vector<std::string>& input_names,
                                 const Ort::Logger& logger,
                                 const bool& do_op_validation,
                                 const bool& is_bidirection,
                                 std::vector<std::string>& uni_gru_output_names) const;
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
                              bool is_for_output) const;
};

Ort::Status GRUOpBuilder::AddStridedSlice(QnnModelWrapper& qnn_model_wrapper,
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
                                          bool is_for_output) const {
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

Ort::Status GRUOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  if (node_unit.Inputs().size() > 4 && node_unit.Inputs()[4].Exists()) {
    TensorInfo tensor_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[4], tensor_info));
    RETURN_IF_NOT(tensor_info.is_initializer, "QNN EP: dynamic sequence_length is not supported.");
    std::vector<uint8_t> sequence_lens_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(tensor_info.initializer_tensor, sequence_lens_bytes));
    const size_t num_elems = sequence_lens_bytes.size() / sizeof(int32_t);
    gsl::span<const int32_t> sequence_lens{reinterpret_cast<const int32_t*>(sequence_lens_bytes.data()), num_elems};
    RETURN_IF(std::any_of(sequence_lens.begin(), sequence_lens.end(),
                          [sequence_lens](int i) { return i != sequence_lens[0]; }),
              "QNN EP: Only support GRU with same sequence length.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  RETURN_IF(node_helper.Get("layout", static_cast<int64_t>(0)) != 0,
            "QNN EP doesn't support layout=1 for GRU (ORT CPU EP cannot provide a reference for accuracy validation).");
  RETURN_IF(node_helper.HasAttr("clip"), "QNN EP doesn't support clip for GRU.");
  const std::vector<std::string> activations = node_helper.Get("activations", std::vector<std::string>{});
  RETURN_IF((activations.size() >= 2 && (activations[0] != "sigmoid" || activations[1] != "tanh")) ||
                (activations.size() == 4 && (activations[2] != "sigmoid" || activations[3] != "tanh")),
            "QNN EP doesn't support non-default activations for GRU.");
  return Ort::Status();
}

Ort::Status GRUOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const Ort::Logger& logger,
                                        std::vector<std::string>& input_names,
                                        bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& onnx_inputs = node_unit.Inputs();
  for (size_t i = 0; i < onnx_inputs.size(); i++) {
    if (onnx_inputs[i].Exists()) {
      RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[i], logger, input_names));
    } else {
      input_names.emplace_back("");
    }
  }
  return Ort::Status();
}

// Manually unrolls the GRU across time steps only (time_major=true).
// Each QNN GRU node processes a single time step with input shape [1, batch, input] (seq=1, time_major=true).
// The full batch is processed together in each step, avoiding the batch-element loop entirely.
Ort::Status GRUOpBuilder::AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const std::string& direction,
                                             const std::vector<std::string>& input_names,
                                             const Ort::Logger& logger,
                                             const bool& do_op_validation,
                                             const bool& is_bidirection,
                                             std::vector<std::string>& uni_gru_output_names) const {
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

  // GRU parameters - shared by all unrolled time-step cells.
  // Use the actual ONNX direction so QNN sees the correct gate ordering.
  // Time step ordering for reverse is handled by the unrolling loop (iterating t from seq-1 down to 0).
  std::vector<std::string> param_names;
  const uint32_t qnn_direction = (direction == "reverse") ? QNN_OP_GRU_DIRECTION_REVERSE : QNN_OP_GRU_DIRECTION_FORWARD;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name() + "_" + direction,
                                         qnn_direction, QNN_OP_GRU_PARAM_DIRECTION, param_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(linear_before_reset),
                                         QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET, param_names));
  // TODO: Once the QNN CPU backend accuracy issue with time_major=true is resolved, remove the CPU
  // workaround below and the associated Transpose nodes (X pre-transpose, per-step Y-to-h transpose,
  // and post-concat transpose), then let CPU use time_major=true uniformly with layout=0.
  // time_major: on CPU, always use false to work around its batch dimension bug with time_major=true.
  // On other backends (HTP), follow the ONNX layout attribute: layout=0 -> true, layout=1 -> false.
  const bool is_cpu_backend = qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::CPU;
  const bool time_major = is_cpu_backend ? false : (layout == 0);
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), time_major,
                                     QNN_OP_GRU_PARAM_TIME_MAJOR, param_names));

  const std::string null_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_null_tensor_" + direction);
  QnnTensorWrapper null_tensor_wrapper = QnnTensorWrapper::MakeNull(null_tensor_name);
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(null_tensor_wrapper)),
                "Failed to add null tensor for GRU.");

  // Base GRU input template (weights, biases) - shared across all unrolled cells
  std::vector<std::string> qnn_gru_input_names(14, null_tensor_name);

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

Ort::Status GRUOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      std::vector<std::string>&& input_names,
                                                      const Ort::Logger& logger,
                                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  OrtNodeAttrHelper node_helper(node_unit);
  std::string direction = node_helper.Get("direction", "forward");
  RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= 6, "GRU should receive inputs ranging from 3 to 6!");

  if (direction == "bidirectional") {
    std::vector<std::string> fwd_out, rev_out;
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, "forward", input_names, logger, do_op_validation, true, fwd_out));
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, "reverse", input_names, logger, do_op_validation, true, rev_out));
    for (size_t i = 0; i < 2; i++) {
      TensorInfo output_info = {};
      if (node_unit.Outputs().size() > i && node_unit.Outputs()[i].Exists()) {
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[i], output_info));
        std::string name = node_unit.Outputs()[i].name;
        std::vector<std::string> cp;
        RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), name,
                                               static_cast<uint32_t>(output_info.shape.size() - 3),
                                               QNN_OP_CONCAT_PARAM_AXIS, cp));
        Qnn_TensorType_t tt = qnn_model_wrapper.IsGraphOutput(name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
        QnnTensorWrapper tw(name, tt, output_info.qnn_data_type, output_info.quant_param.Copy(),
                            std::vector<uint32_t>(output_info.shape));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "Failed to add Concat output.");
        RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_CONCAT),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                                      {fwd_out[i], rev_out[i]}, {name}, std::move(cp), do_op_validation),
                      "Failed to create Concat node.");
      }
    }
  } else {
    std::vector<std::string> uni_out;
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, direction, input_names, logger, do_op_validation, false, uni_out));
  }
  return Ort::Status();
}

void CreateGRUOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GRUOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
