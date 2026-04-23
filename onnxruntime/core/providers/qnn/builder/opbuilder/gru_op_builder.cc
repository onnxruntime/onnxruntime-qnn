// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
                                       bool is_for_output) const;
};

Ort::Status GRUOpBuilder::AddStridedSliceOrReshape(QnnModelWrapper& qnn_model_wrapper,
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
  size_t minSize = std::min(input_shape.size(), output_shape.size());
  uint32_t in_elems = 1, out_elems = 1;
  for (auto s : input_shape) in_elems *= s;
  for (auto s : output_shape) out_elems *= s;
  if (in_elems == out_elems && input_shape[0] == 1 &&
      std::equal(output_shape.rbegin(), output_shape.rbegin() + minSize, input_shape.rbegin())) {
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
  }
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
  const float clip = node_helper.Get("clip", (float)0.0);
  RETURN_IF(clip != 0, "QNN EP doesn't support non-default clip for GRU.");
  const std::vector<std::string> activations = node_helper.Get("activations", std::vector<std::string>{});
  RETURN_IF((activations.size() >= 2 && (activations[0] != "sigmoid" || activations[1] != "tanh")) ||
                (activations.size() == 4 && (activations[2] != "sigmoid" || activations[3] != "tanh")),
            "QNN EP doesn't support non-default activations for GRU.");
  const int64_t layout = node_helper.Get("layout", static_cast<int64_t>(0));
  RETURN_IF_NOT(layout == 0,
                ("QNN EP: Unsupport layout mode" + std::to_string(layout) + " for " + node_unit.Name()).c_str());
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

// Manually unrolls the GRU across time steps and batch elements.
// Each QNN GRU node processes a single (time_step, batch_element) with batch_size=1, seq_length=1.
// This works around QNN CPU/HTP backend bugs where batch_size > 1 produces incorrect results.
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
  const uint32_t hidden_size = node_helper.Get("hidden_size", 0);
  const int32_t hidden_size_sign = SafeInt<int32_t>(hidden_size);
  RETURN_IF_NOT(hidden_size > 0, "hidden size is not set for GRU");
  const int64_t linear_before_reset = node_helper.Get("linear_before_reset", static_cast<int64_t>(0));

  const uint32_t input_size = input_tensor_infos[0].shape[2];
  const uint32_t batch_size = input_tensor_infos[0].shape[1];
  const uint32_t seq_length = input_tensor_infos[0].shape[0];
  const int32_t direction_idx = input_tensor_infos[1].shape[0] < 2 || direction == "forward" ? 0 : 1;

  // GRU parameters - shared by all unrolled cells
  std::vector<std::string> param_names;
  // Always use forward direction for individual cells; time step ordering handled by unrolling loop
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name() + "_" + direction,
                                         QNN_OP_GRU_DIRECTION_FORWARD, QNN_OP_GRU_PARAM_DIRECTION, param_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(linear_before_reset),
                                         QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET, param_names));
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), false,
                                     QNN_OP_GRU_PARAM_TIME_MAJOR, param_names));

  // Null tensor for optional inputs
  const std::string null_tensor_name = "null_tensor";
  QnnTensorWrapper null_tensor_wrapper(null_tensor_name, QNN_TENSOR_TYPE_NULL, QNN_DATATYPE_FLOAT_32,
                                       QnnQuantParamsWrapper(), std::vector<uint32_t>{0});
  qnn_model_wrapper.AddTensorWrapper(std::move(null_tensor_wrapper));

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
      RETURN_IF_ERROR(AddStridedSliceOrReshape(qnn_model_wrapper, node_unit, input_names[1], names[i],
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
      RETURN_IF_ERROR(AddStridedSliceOrReshape(qnn_model_wrapper, node_unit, input_names[2], names[i],
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
        RETURN_IF_ERROR(AddStridedSliceOrReshape(qnn_model_wrapper, node_unit, input_names[3], names[i],
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
      initial_h_name = utils::UniqueNameGenerator().New(input_names[5], direction);
      RETURN_IF_ERROR(AddStridedSliceOrReshape(qnn_model_wrapper, node_unit, input_names[5], initial_h_name,
                                               input_tensor_infos[5].shape, h_shape,
                                               {{direction_idx, direction_idx + 1, 1},
                                                {0, SafeInt<int32_t>(batch_size), 1}, {0, hidden_size_sign, 1}},
                                               0, 0, 0, 0, input_tensor_infos[5].qnn_data_type,
                                               input_tensor_infos[5].quant_param, do_op_validation, false, false));
    } else {
      initial_h_name = utils::UniqueNameGenerator().New(node_unit.Name(), "_GRU_initial_h");
      QnnTensorWrapper zero_h(initial_h_name, QNN_TENSOR_TYPE_STATIC, input_tensor_infos[0].qnn_data_type,
                              QnnQuantParamsWrapper(), std::vector<uint32_t>(h_shape),
                              std::vector<uint8_t>(utils::GetElementSizeByType(input_tensor_infos[0].qnn_data_type) * batch_size * hidden_size, 0));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_h)), "Failed to add initial hidden state.");
    }
  }

  // Split initial_h [1, batch, hidden] -> per-batch [hidden_size] tensors
  std::vector<std::string> prev_h_names(batch_size);
  for (uint32_t b = 0; b < batch_size; b++) {
    prev_h_names[b] = utils::UniqueNameGenerator().New(node_unit, "_h_init_b" + std::to_string(b) + "_" + direction);
    RETURN_IF_ERROR(AddStridedSliceOrReshape(qnn_model_wrapper, node_unit, initial_h_name, prev_h_names[b],
                                             {1, batch_size, hidden_size}, {hidden_size},
                                             {{0, 1, 1}, {SafeInt<int32_t>(b), SafeInt<int32_t>(b + 1), 1}, {0, hidden_size_sign, 1}},
                                             0, 0, 0b011U, 0, input_tensor_infos[0].qnn_data_type,
                                             output_tensor_infos[1].quant_param, do_op_validation, false, false));
  }

  // Unroll across time steps and batch elements
  std::vector<std::string> qnn_all_hidden_state_names(seq_length);

  for (uint32_t step = 0; step < seq_length; step++) {
    uint32_t t = direction == "forward" ? step : seq_length - step - 1;
    std::vector<std::string> per_batch_h(batch_size);

    for (uint32_t b = 0; b < batch_size; b++) {
      std::vector<std::string> cell_inputs = qnn_gru_input_names;
      const std::string sfx = "_t" + std::to_string(t) + "_b" + std::to_string(b) + "_" + direction;

      // Slice X[t, b, :] -> [input_size], reshape to [1, 1, input_size]
      std::string x_flat = utils::UniqueNameGenerator().New(input_names[0], "_xf" + sfx);
      RETURN_IF_ERROR(AddStridedSliceOrReshape(qnn_model_wrapper, node_unit, input_names[0], x_flat,
                                               input_tensor_infos[0].shape, {input_size},
                                               {{SafeInt<int32_t>(t), SafeInt<int32_t>(t + 1), 1},
                                                {SafeInt<int32_t>(b), SafeInt<int32_t>(b + 1), 1},
                                                {0, SafeInt<int32_t>(input_size), 1}},
                                               0, 0, 0b011U, 0, input_tensor_infos[0].qnn_data_type,
                                               input_tensor_infos[0].quant_param, do_op_validation, false, false));
      std::string x_3d = utils::UniqueNameGenerator().New(input_names[0], "_x3d" + sfx);
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(x_flat, x_3d, {input_size}, {1, 1, input_size},
                                                       input_tensor_infos[0].qnn_data_type,
                                                       input_tensor_infos[0].quant_param, do_op_validation, false, false));
      cell_inputs[0] = x_3d;

      // Reshape prev_h [hidden_size] -> [1, 1, hidden_size] for initial_h
      std::string h_3d = utils::UniqueNameGenerator().New(node_unit, "_h3d" + sfx);
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(prev_h_names[b], h_3d, {hidden_size}, {1, 1, hidden_size},
                                                       input_tensor_infos[0].qnn_data_type,
                                                       output_tensor_infos[1].quant_param, do_op_validation, false, false));
      cell_inputs[13] = h_3d;

      // GRU outputs: Y [1, 1, hidden], Y_h [1, 1, hidden]
      std::string y_name = utils::UniqueNameGenerator().New(node_unit, "_Y" + sfx);
      std::string yh_name = utils::UniqueNameGenerator().New(node_unit, "_Yh" + sfx);
      for (const auto& [name, idx] : std::vector<std::pair<std::string, size_t>>{{y_name, 0}, {yh_name, 1}}) {
        QnnTensorWrapper tw(name, QNN_TENSOR_TYPE_NATIVE, output_tensor_infos[idx].qnn_data_type,
                            output_tensor_infos[idx].quant_param.Copy(), std::vector<uint32_t>{1, 1, hidden_size});
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "Failed to add GRU output tensor.");
      }
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        utils::UniqueNameGenerator().New(node_unit, "_cell" + sfx),
                        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GRU,
                        std::move(cell_inputs), {y_name, yh_name},
                        std::vector<std::string>(param_names), do_op_validation),
                    "Failed to create GRU node.");

      // Reshape Y [1, 1, hidden] -> [hidden_size]
      std::string h_flat = utils::UniqueNameGenerator().New(node_unit, "_hf" + sfx);
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(y_name, h_flat, {1, 1, hidden_size}, {hidden_size},
                                                       output_tensor_infos[0].qnn_data_type,
                                                       output_tensor_infos[0].quant_param, do_op_validation, false, false));
      per_batch_h[b] = h_flat;
      prev_h_names[b] = h_flat;
    }

    // Pack per-batch [hidden] -> [batch, hidden]
    std::string packed = utils::UniqueNameGenerator().New(node_unit, "_pk_t" + std::to_string(t) + "_" + direction);
    {
      std::vector<std::string> pp;
      RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), packed, 0, QNN_OP_PACK_PARAM_AXIS, pp));
      QnnTensorWrapper tw(packed, QNN_TENSOR_TYPE_NATIVE, output_tensor_infos[0].qnn_data_type,
                          output_tensor_infos[0].quant_param.Copy(), {batch_size, hidden_size});
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "Failed to add Pack output.");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(packed, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PACK,
                                                    std::move(per_batch_h), {packed}, std::move(pp), do_op_validation),
                    "Failed to create Pack node.");
    }
    qnn_all_hidden_state_names[t] = packed;
  }

  // Pack time steps [batch, hidden] * seq -> [seq, batch, hidden]
  const std::string y_all = utils::UniqueNameGenerator().New(node_unit, "_Y_all_" + direction);
  {
    std::vector<std::string> pp;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), y_all, 0, QNN_OP_PACK_PARAM_AXIS, pp));
    QnnTensorWrapper tw(y_all, QNN_TENSOR_TYPE_NATIVE, output_tensor_infos[0].qnn_data_type,
                        output_tensor_infos[0].quant_param.Copy(), {seq_length, batch_size, hidden_size});
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "Failed to add Y Pack output.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(y_all, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PACK,
                                                  std::move(qnn_all_hidden_state_names), {y_all}, std::move(pp), do_op_validation),
                  "Failed to create Y Pack node.");
  }

  // Pack final hidden states [hidden] * batch -> [batch, hidden]
  const std::string y_h_packed = utils::UniqueNameGenerator().New(node_unit, "_Yh_pk_" + direction);
  {
    std::vector<std::string> pp;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), y_h_packed, 0, QNN_OP_PACK_PARAM_AXIS, pp));
    QnnTensorWrapper tw(y_h_packed, QNN_TENSOR_TYPE_NATIVE, output_tensor_infos[1].qnn_data_type,
                        output_tensor_infos[1].quant_param.Copy(), {batch_size, hidden_size});
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "Failed to add Y_h Pack output.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(y_h_packed, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PACK,
                                                  std::vector<std::string>(prev_h_names), {y_h_packed}, std::move(pp), do_op_validation),
                  "Failed to create Y_h Pack node.");
  }

  // Map to ONNX output shapes
  std::vector<std::vector<uint32_t>> onnx_shapes = {{seq_length, 1, batch_size, hidden_size}, {1, batch_size, hidden_size}};
  for (size_t i = 0; i < 2; i++) {
    if (onnx_outputs.size() > i && onnx_outputs[i].Exists()) {
      const std::string out_name = is_bidirection
                                       ? utils::UniqueNameGenerator().New(y_all, "_unsqueeze_" + direction)
                                       : onnx_outputs[i].name;
      const std::string& src = (i == 0) ? y_all : y_h_packed;
      const std::vector<uint32_t> src_shape = (i == 0) ? std::vector<uint32_t>{seq_length, batch_size, hidden_size}
                                                       : std::vector<uint32_t>{batch_size, hidden_size};
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(src, out_name, src_shape, onnx_shapes[i],
                                                       output_tensor_infos[i].qnn_data_type,
                                                       output_tensor_infos[i].quant_param, do_op_validation, false,
                                                       qnn_model_wrapper.IsGraphOutput(out_name)));
      uni_gru_output_names.emplace_back(out_name);
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
