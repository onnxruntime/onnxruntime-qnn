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
  /*
  ONNX GRU inputs:
  in[0]: X [seq_length, batch_size, input_size]
  in[1]: W [num_directions, 3*hidden_size, input_size], gate order: z (update), r (reset), h (new)
  in[2]: R [num_directions, 3*hidden_size, hidden_size], gate order: z (update), r (reset), h (new)

  ONNX GRU optional inputs:
  in[3]: B [num_directions, 6*hidden_size], order: [Wb[zrh], Rb[zrh]]
  in[4]: sequence_lens  ---> not supported
  in[5]: initial_h [num_directions, batch_size, hidden_size]

  ONNX GRU Parameters:
  - activations      ---> not supported by QNN
  - activation_alpha ---> not supported by QNN
  - activation_beta  ---> not supported by QNN
  - clip             ---> not supported by QNN
  - direction
  - hidden_size
  - linear_before_reset
  - layout: The shape format of inputs X, initial_h and outputs Y, Y_h.
            If 0, the following shapes are expected:
                X.shape = [seq_length, batch_size, input_size],
                Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
            If 1, the following shapes are expected:
                X.shape = [batch_size, seq_length, input_size],
                Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].

  ONNX GRU optional outputs:
  out[0]: Y [seq_length, num_directions, batch_size, hidden_size]
  out[1]: Y_h [num_directions, batch_size, hidden_size]

  QNN GRU inputs:
  in[0]: x_t: 3D of shape [seq_length, batch_size, input_size] if time_major
                           [batch_size, seq_length, input_size] else
  in[1]: W_xz: input-to-update weights [hidden_size, input_size]   = ONNX in[1][direction, 0*hidden_size:1*hidden_size, :]
  in[2]: W_xr: input-to-reset weights [hidden_size, input_size]    = ONNX in[1][direction, 1*hidden_size:2*hidden_size, :]
  in[3]: W_xn: input-to-new weights [hidden_size, input_size]      = ONNX in[1][direction, 2*hidden_size:3*hidden_size, :]
  in[4]: W_hz: recurrent-to-update weights [hidden_size, hidden_size] = ONNX in[2][direction, 0*hidden_size:1*hidden_size, :]
  in[5]: W_hr: recurrent-to-reset weights [hidden_size, hidden_size]  = ONNX in[2][direction, 1*hidden_size:2*hidden_size, :]
  in[6]: W_hn: recurrent-to-new weights [hidden_size, hidden_size]    = ONNX in[2][direction, 2*hidden_size:3*hidden_size, :]

  # optional inputs
  in[7]:  b_xz: input-to-update gate bias [hidden_size]     = ONNX in[3][direction, 0*hidden_size:1*hidden_size]
  in[8]:  b_xr: input-to-reset gate bias [hidden_size]      = ONNX in[3][direction, 1*hidden_size:2*hidden_size]
  in[9]:  b_xn: input-to-new gate bias [hidden_size]        = ONNX in[3][direction, 2*hidden_size:3*hidden_size]
  in[10]: b_hz: recurrent-to-update gate bias [hidden_size] = ONNX in[3][direction, 3*hidden_size:4*hidden_size]
  in[11]: b_hr: recurrent-to-reset gate bias [hidden_size]  = ONNX in[3][direction, 4*hidden_size:5*hidden_size]
  in[12]: b_hn: recurrent-to-new gate bias [hidden_size]    = ONNX in[3][direction, 5*hidden_size:6*hidden_size]
  in[13]: initial_h [1, batch_size, hidden_size]             = ONNX in[5][direction:direction+1, :, :] as [1, batch_size, hidden_size]
  in[14]: reset ---> not used

  QNN GRU Parameters:
  - direction
  - linear_before_reset
  - time_major

  QNN GRU outputs:
  out[0]: Y 3D of shape [seq_length, batch_size, hidden_size] if time_major
                        [batch_size, seq_length, hidden_size] else
  out[1]: Y_h [1, batch_size, hidden_size]
  */

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
  // add strided_slice or reshape
  // this is not general condition, only limited to caller in this builder
  size_t minSize = std::min(input_shape.size(), output_shape.size());
  if (input_shape[0] == 1 && std::equal(output_shape.rbegin(), output_shape.rbegin() + minSize, input_shape.rbegin())) {
    // add Reshape
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input_name,
                                                     output_name,
                                                     input_shape,
                                                     output_shape,
                                                     tensor_data_type,
                                                     quantize_param.Copy(),
                                                     quantize_param.Copy(),
                                                     do_op_validation,
                                                     is_for_input,
                                                     is_for_output));
  } else {
    // add StridedSlice
    // inputs
    QnnTensorWrapper input_tensorwrapper(input_name, is_for_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_NATIVE,
                                         tensor_data_type, quantize_param.Copy(),
                                         std::vector<uint32_t>(input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                  "Failed to add input tensor for inserted StridedSlice or Reshape.");

    // params
    const std::string node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_STRIDED_SLICE);

    // ranges
    std::vector<uint32_t> ranges_data;
    for (size_t i = 0; i < ranges.size(); i++) {
      for (size_t j = 0; j < 3; j++) {
        ranges_data.emplace_back(SafeInt<uint32_t>(ranges[i][j]));
      }
    }
    QnnParamWrapper ranges_param_wrapper(node_unit.Index(), node_name, QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                         {static_cast<uint32_t>(ranges.size()), 3}, std::move(ranges_data), true);
    std::vector<std::string> param_names = {
        ranges_param_wrapper.GetParamTensorName(),
    };
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
    QnnTensorWrapper output_tensorwrapper(output_name,
                                          is_for_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                          tensor_data_type,
                                          quantize_param.Copy(),
                                          std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                  "Failed to add output tensor for inserted StridedSlice.");
    // addNode
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, {input_name},
                                                  {output_name}, std::move(param_names), do_op_validation),
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
    RETURN_IF(std::any_of(sequence_lens.begin(),
                          sequence_lens.end(),
                          [sequence_lens](int i) { return i != sequence_lens[0]; }),
              "QNN EP: Only support GRU with same sequence length.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const float clip = node_helper.Get("clip", (float)0.0);
  RETURN_IF(clip != 0,
            "QNN EP doesn't support non-default clip for GRU.");
  const std::vector<std::string> activations = node_helper.Get("activations", std::vector<std::string>{});
  RETURN_IF((activations.size() >= 2 && (activations[0] != "sigmoid" || activations[1] != "tanh")) ||
                (activations.size() == 4 && (activations[2] != "sigmoid" || activations[3] != "tanh")),
            "QNN EP doesn't support non-default activations for GRU.");
  // TODO: Add support for layout==1
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
  // QNN GRU has 2 mandatory outputs
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

  // params
  std::vector<std::string> param_names;

  // direction
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name() + "_" + direction,
                                         direction == "forward" ? QNN_OP_GRU_DIRECTION_FORWARD : QNN_OP_GRU_DIRECTION_REVERSE,
                                         QNN_OP_GRU_PARAM_DIRECTION, param_names));

  // linear_before_reset
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(linear_before_reset),
                                         QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET, param_names));

  // time_major: set to true since current builder only supports time major GRU (layout=0)
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), true,
                                     QNN_OP_GRU_PARAM_TIME_MAJOR, param_names));

  // Common null tensor for optional inputs
  const std::string null_tensor_name = "null_tensor";
  QnnTensorWrapper null_tensor_wrapper(null_tensor_name, QNN_TENSOR_TYPE_NULL, QNN_DATATYPE_FLOAT_32,
                                       QnnQuantParamsWrapper(), std::vector<uint32_t>{0});
  qnn_model_wrapper.AddTensorWrapper(std::move(null_tensor_wrapper));

  std::vector<std::string> qnn_gru_input_names(15, null_tensor_name);
  qnn_gru_input_names[0] = input_names[0];

  // input W: ONNX in[1] [num_directions, 3*hidden_size, input_size], gate order: z, r, h
  {
    // QNN in[1] (W_xz) = ONNX in[1][direction, 0*hidden_size:1*hidden_size, :]
    // QNN in[2] (W_xr) = ONNX in[1][direction, 1*hidden_size:2*hidden_size, :]
    // QNN in[3] (W_xn) = ONNX in[1][direction, 2*hidden_size:3*hidden_size, :]
    uint32_t begin_mask = 0b000U;
    uint32_t end_mask = 0b000U;
    uint32_t shrink_axes = 0b001U;
    uint32_t new_axes_mask = 0b000U;
    std::vector<uint32_t> qnn_input_indices = {1, 2, 3};
    std::vector<int32_t> begins = {0, 1, 2};
    std::vector<std::string> qnn_gru_weight_name = {
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_update_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_reset_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[1], "_input_to_new_gate_weight_" + direction),
    };
    for (size_t i = 0; i < 3; i++) {
      std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                  {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
                                                  {0, SafeInt<int32_t>(input_size), 1}};
      std::vector<uint32_t> output_shape = {hidden_size, input_size};
      RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                               /*node_unit=*/node_unit,
                                               /*input_name=*/input_names[1],
                                               /*output_name=*/qnn_gru_weight_name[i],
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
      qnn_gru_input_names[qnn_input_indices[i]] = qnn_gru_weight_name[i];
    }
  }

  // input R: ONNX in[2] [num_directions, 3*hidden_size, hidden_size], gate order: z, r, h
  {
    // QNN in[4] (W_hz) = ONNX in[2][direction, 0*hidden_size:1*hidden_size, :]
    // QNN in[5] (W_hr) = ONNX in[2][direction, 1*hidden_size:2*hidden_size, :]
    // QNN in[6] (W_hn) = ONNX in[2][direction, 2*hidden_size:3*hidden_size, :]
    uint32_t begin_mask = 0b000U;
    uint32_t end_mask = 0b000U;
    uint32_t shrink_axes = 0b001U;
    uint32_t new_axes_mask = 0b000U;
    std::vector<uint32_t> qnn_input_indices = {4, 5, 6};
    std::vector<int32_t> begins = {0, 1, 2};
    std::vector<std::string> qnn_gru_weight_name = {
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_update_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_reset_gate_weight_" + direction),
        utils::UniqueNameGenerator().New(input_names[2], "_recurrent_to_new_gate_weight_" + direction),
    };
    for (size_t i = 0; i < 3; i++) {
      std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                  {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
                                                  {0, hidden_size_sign, 1}};
      std::vector<uint32_t> output_shape = {hidden_size, hidden_size};
      RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                               /*node_unit=*/node_unit,
                                               /*input_name=*/input_names[2],
                                               /*output_name=*/qnn_gru_weight_name[i],
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
      qnn_gru_input_names[qnn_input_indices[i]] = qnn_gru_weight_name[i];
    }
  }

  // input B: ONNX in[3] [num_directions, 6*hidden_size], order: [Wb[zrh], Rb[zrh]]
  {
    // QNN in[7]  (b_xz) = ONNX in[3][direction, 0*hidden_size:1*hidden_size]
    // QNN in[8]  (b_xr) = ONNX in[3][direction, 1*hidden_size:2*hidden_size]
    // QNN in[9]  (b_xn) = ONNX in[3][direction, 2*hidden_size:3*hidden_size]
    // QNN in[10] (b_hz) = ONNX in[3][direction, 3*hidden_size:4*hidden_size]
    // QNN in[11] (b_hr) = ONNX in[3][direction, 4*hidden_size:5*hidden_size]
    // QNN in[12] (b_hn) = ONNX in[3][direction, 5*hidden_size:6*hidden_size]
    uint32_t begin_mask = 0b00U;
    uint32_t end_mask = 0b00U;
    uint32_t shrink_axes = 0b01U;
    uint32_t new_axes_mask = 0b00U;
    std::vector<uint32_t> output_shape = {hidden_size};
    std::vector<uint32_t> qnn_input_indices = {7, 8, 9, 10, 11, 12};
    if (onnx_inputs.size() > 3 && onnx_inputs[3].Exists()) {
      std::vector<int32_t> begins = {0, 1, 2, 3, 4, 5};
      std::vector<std::string> qnn_gru_bias_name = {
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_update_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_reset_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_input_to_new_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_update_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_reset_gate_bias_" + direction),
          utils::UniqueNameGenerator().New(input_names[3], "_recurrent_to_new_gate_bias_" + direction),
      };
      for (size_t i = 0; i < 6; i++) {
        std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                    {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1}};
        RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                 /*node_unit=*/node_unit,
                                                 /*input_name=*/input_names[3],
                                                 /*output_name=*/qnn_gru_bias_name[i],
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
        qnn_gru_input_names[qnn_input_indices[i]] = qnn_gru_bias_name[i];
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
                    "Failed to add additional zero bias for QNN GRU node.");
      for (size_t i = 0; i < 6; i++) {
        qnn_gru_input_names[qnn_input_indices[i]] = zero_bias_name;
      }
    }
  }

  // input initial_h: ONNX in[5] [num_directions, batch_size, hidden_size]
  {
    // QNN in[13] = ONNX in[5][direction_idx:direction_idx+1, :, :] as [1, batch_size, hidden_size]
    // shrink_axes must be 0 so the direction dim is kept as 1 (not squeezed), matching QNN in[13]'s
    // expected shape [1, batch_size, hidden_size]. For unidirectional (num_directions=1), the
    // Reshape path is taken instead since input already has shape [1, batch, hidden].
    uint32_t begin_mask = 0b000U;
    uint32_t end_mask = 0b000U;
    uint32_t shrink_axes = 0b000U;
    uint32_t new_axes_mask = 0b000U;
    std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                {0, SafeInt<int32_t>(batch_size), 1},
                                                {0, hidden_size_sign, 1}};
    std::vector<uint32_t> output_shape = {1, batch_size, hidden_size};
    if (onnx_inputs.size() > 5 && onnx_inputs[5].Exists()) {
      const std::string qnn_gru_initial_h_name = utils::UniqueNameGenerator().New(input_names[5], direction);
      RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                               /*node_unit=*/node_unit,
                                               /*input_name=*/input_names[5],
                                               /*output_name=*/qnn_gru_initial_h_name,
                                               /*input_shape=*/input_tensor_infos[5].shape,
                                               /*output_shape=*/output_shape,
                                               /*ranges=*/ranges,
                                               /*begin_mask=*/begin_mask,
                                               /*end_mask=*/end_mask,
                                               /*shrink_axes=*/shrink_axes,
                                               /*new_axes_mask=*/new_axes_mask,
                                               /*tensor_data_type=*/input_tensor_infos[5].qnn_data_type,
                                               /*QnnQuantParamsWrapper=*/input_tensor_infos[5].quant_param,
                                               /*do_op_validation=*/do_op_validation,
                                               /*is_for_input=*/false,
                                               /*is_for_output=*/false));
      qnn_gru_input_names[13] = qnn_gru_initial_h_name;
    } else {
      // prepare zero initial_h
      const std::string& node_name = node_unit.Name();
      std::string zero_initial_h_name = utils::UniqueNameGenerator().New(node_name, "_GRU_initial_h");
      QnnTensorWrapper zero_initial_h_wrapper(zero_initial_h_name,
                                              QNN_TENSOR_TYPE_STATIC,
                                              input_tensor_infos[0].qnn_data_type,
                                              QnnQuantParamsWrapper(),
                                              std::vector<uint32_t>(output_shape),
                                              std::vector<uint8_t>(
                                                  utils::GetElementSizeByType(input_tensor_infos[0].qnn_data_type) * batch_size * hidden_size,
                                                  0));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_initial_h_wrapper)),
                    "Failed to add initial hidden state for QNN GRU node.");
      qnn_gru_input_names[13] = zero_initial_h_name;
    }
  }

  // outputs
  // QNN out[0]: Y [seq_length, batch_size, hidden_size]
  // QNN out[1]: Y_h [1, batch_size, hidden_size] - QNN CPU may not give the correct final hidden
  //              state in out[1], so we derive Y_h from the appropriate time step of out[0] instead.
  std::vector<std::vector<uint32_t>> qnn_gru_output_shapes = {
      {seq_length, batch_size, hidden_size},
      {1, batch_size, hidden_size}};

  std::vector<std::string> qnn_gru_output_names = {
      utils::UniqueNameGenerator().New(node_unit, "_QNN_GRU_output_all_hidden_state_" + direction),
      utils::UniqueNameGenerator().New(node_unit, "_QNN_GRU_output_last_hidden_state_" + direction)};

  for (size_t j = 0; j < qnn_gru_output_names.size(); j++) {
    QnnTensorWrapper output_tensorwrapper(qnn_gru_output_names[j],
                                          QNN_TENSOR_TYPE_NATIVE,
                                          output_tensor_infos[j].qnn_data_type,
                                          output_tensor_infos[j].quant_param.Copy(),
                                          std::vector<uint32_t>(qnn_gru_output_shapes[j]));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                  ("QNN EP: Failed to add " + std::to_string(j) + "th output tensor for QNN GRU.").c_str());
  }
  const std::string gru_node_name = utils::UniqueNameGenerator().New(node_unit, "_" + direction);
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(gru_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GRU,
                                                std::move(qnn_gru_input_names), std::vector<std::string>(qnn_gru_output_names),
                                                std::vector<std::string>(param_names), do_op_validation),
                "QNN EP: Failed to create Qnn GRU node.");

  // Map QNN outputs to ONNX outputs with appropriate reshapes/slices.
  //
  // QNN out[0]: Y [seq_length, batch_size, hidden_size]
  //   -> ONNX out[0]: Y [seq_length, 1, batch_size, hidden_size] (add num_directions dim via Reshape)
  //
  // QNN out[1] is unreliable on some backends (may return first-step state instead of final state).
  // Instead, derive Y_h by slicing the correct time step from out[0]:
  //   - forward:  Y[seq_length-1, :, :] = final hidden state
  //   - reverse:  Y[0, :, :] = final hidden state (last processed in reverse order)
  // Then reshape [batch_size, hidden_size] -> [1, batch_size, hidden_size] for ONNX Y_h.
  std::vector<std::vector<uint32_t>> onnx_gru_output_shapes = {
      {seq_length, 1, batch_size, hidden_size},
      {1, batch_size, hidden_size}};

  for (size_t i = 0; i < 2; i++) {
    if (onnx_outputs.size() > i && onnx_outputs[i].Exists()) {
      // For bidirectional: use a unique intermediate name that will be consumed by a Concat op.
      // For unidirectional: use the ONNX output name directly.
      const std::string reshape_output_name = is_bidirection
                                                  ? utils::UniqueNameGenerator().New(qnn_gru_output_names[i], "_unsqueeze_" + direction)
                                                  : onnx_outputs[i].name;

      if (i == 0) {
        // Y: Reshape [seq, batch, hidden] -> [seq, 1, batch, hidden]
        RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(/*input_name=*/qnn_gru_output_names[0],
                                                         /*output_name=*/reshape_output_name,
                                                         /*input_shape=*/qnn_gru_output_shapes[0],
                                                         /*output_shape=*/onnx_gru_output_shapes[0],
                                                         /*tensor_data_type=*/output_tensor_infos[0].qnn_data_type,
                                                         /*quantize_param=*/output_tensor_infos[0].quant_param,
                                                         /*do_op_validation=*/do_op_validation,
                                                         /*is_for_input=*/false,
                                                         /*is_for_output=*/qnn_model_wrapper.IsGraphOutput(reshape_output_name)));
      } else {
        // Y_h: slice the final hidden state from Y (out[0]) and reshape to [1, batch, hidden].
        // For forward direction: take Y[seq_length-1, :, :] (last time step).
        // For reverse direction: take Y[0, :, :] (first index = last processed in reverse).
        const int32_t y_h_t = (direction == "forward") ? SafeInt<int32_t>(seq_length) - 1 : 0;
        const std::string y_h_slice_name = utils::UniqueNameGenerator().New(
            qnn_gru_output_names[0], "_y_h_slice_" + direction);
        std::vector<std::vector<int32_t>> y_h_ranges = {{y_h_t, y_h_t + 1, 1},
                                                         {0, SafeInt<int32_t>(batch_size), 1},
                                                         {0, hidden_size_sign, 1}};
        std::vector<uint32_t> y_h_slice_shape = {batch_size, hidden_size};
        RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                 /*node_unit=*/node_unit,
                                                 /*input_name=*/qnn_gru_output_names[0],
                                                 /*output_name=*/y_h_slice_name,
                                                 /*input_shape=*/qnn_gru_output_shapes[0],
                                                 /*output_shape=*/y_h_slice_shape,
                                                 /*ranges=*/y_h_ranges,
                                                 /*begin_mask=*/0b000U,
                                                 /*end_mask=*/0b000U,
                                                 /*shrink_axes=*/0b001U,
                                                 /*new_axes_mask=*/0b000U,
                                                 /*tensor_data_type=*/output_tensor_infos[1].qnn_data_type,
                                                 /*QnnQuantParamsWrapper=*/output_tensor_infos[1].quant_param,
                                                 /*do_op_validation=*/do_op_validation,
                                                 /*is_for_input=*/false,
                                                 /*is_for_output=*/false));
        // Reshape [batch, hidden] -> [1, batch, hidden] for ONNX Y_h format.
        RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(/*input_name=*/y_h_slice_name,
                                                         /*output_name=*/reshape_output_name,
                                                         /*input_shape=*/y_h_slice_shape,
                                                         /*output_shape=*/onnx_gru_output_shapes[1],
                                                         /*tensor_data_type=*/output_tensor_infos[1].qnn_data_type,
                                                         /*quantize_param=*/output_tensor_infos[1].quant_param,
                                                         /*do_op_validation=*/do_op_validation,
                                                         /*is_for_input=*/false,
                                                         /*is_for_output=*/qnn_model_wrapper.IsGraphOutput(reshape_output_name)));
      }
      uni_gru_output_names.emplace_back(reshape_output_name);
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
    std::vector<std::string> uni_gru_output_names_forward, uni_gru_output_names_reverse;
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, "forward", input_names, logger, do_op_validation, true,
                                       uni_gru_output_names_forward));
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, "reverse", input_names, logger, do_op_validation, true,
                                       uni_gru_output_names_reverse));

    // Concat forward and reverse output along the num_directions axis
    for (size_t i = 0; i < 2; i++) {
      TensorInfo output_info = {};
      if (node_unit.Outputs().size() > i && node_unit.Outputs()[i].Exists()) {
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[i], output_info));
        std::string onnx_output_name = node_unit.Outputs()[i].name;

        // param
        std::vector<std::string> concat_param_names;
        RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), onnx_output_name,
                                               static_cast<uint32_t>(output_info.shape.size() - 3),
                                               QNN_OP_CONCAT_PARAM_AXIS, concat_param_names));

        // create tensor and add op
        Qnn_TensorType_t output_tensor_type = qnn_model_wrapper.IsGraphOutput(onnx_output_name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
        QnnTensorWrapper concat_output_tensorwrapper(onnx_output_name,
                                                     output_tensor_type,
                                                     output_info.qnn_data_type,
                                                     output_info.quant_param.Copy(),
                                                     std::vector<uint32_t>(output_info.shape));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(concat_output_tensorwrapper)),
                      "QNN EP: Failed to add output tensor for QNN Concat.");
        RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_CONCAT),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_CONCAT,
                                                      {uni_gru_output_names_forward[i], uni_gru_output_names_reverse[i]},
                                                      {onnx_output_name},
                                                      std::move(concat_param_names), do_op_validation),
                      "QNN EP: Failed to create Qnn Concat node.");
      }
    }
  } else {
    // not used, just a placeholder
    std::vector<std::string> uni_gru_output_names;
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, direction, input_names, logger, do_op_validation, false,
                                       uni_gru_output_names));
  }
  return Ort::Status();
}

void CreateGRUOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GRUOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
