// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/rnn_op_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// NOTE: The ONNX->QNN decomposition (gate slicing, ifoc reordering, bias summation, bidirectional
// Concat) is shared with the qti_aisw "StatefulLstm" builder via rnn_details::AddUnidirectionLSTM
// in rnn_op_utils.cc. Both builders delegate to that function — changes there affect both.
class LSTMOpBuilder : public BaseOpBuilder {
 public:
  LSTMOpBuilder() : BaseOpBuilder("LSTMOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LSTMOpBuilder);

 protected:
  /*
  ONNX LSTM inputs:
  in[0]: X [seq_length, batch_size, input_size], the input sequences packed
  in[1]: W [num_directions, 4*hidden_size, input_size], the weight tensor for the gates. Concatenation of W[iofc] and WB[iofc]
  in[2]: R [num_directions, 4*hidden_size, hidden_size], the recurrence weight tensor. Concatenation of R[iofc] and RB[iofc]

  ONNX LSTM optional inputs:
  in[3]: B [num_directions, 8*hidden_size], the bias tensor for input gate. Concatenation of [Wb[iofc], Rb[iofc]], and [WBb[iofc], RBb[iofc]] (if bidirectional)
  in[4]: sequence_lens
  in[5]: initial_h [num_directions, batch_size, hidden_size].
  in[6]: initial_c [num_directions, batch_size, hidden_size].
  in[7]: P [num_directions, 3*hidde_size], the weight tensor for peepholes. Concatenation of P[iof] and PB[iof]

  ONNX LSTM Parameters:
  - activation_alpha ---> Not supported by QNN.
  - activation_beta  ---> Not supported by QNN.
  - activations      ---> Not supported by QNN.
  - clip             ---> Not supported by QNN since the clip in ONNX applied to iofc while QNN only apply to c. Refer
                          https://github.com/microsoft/onnxruntime/blob/v1.21.0/onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc
  - direction
  - hidden_size
  - input_forget     ---> Not supported by QNN
  - layout: The shape format of inputs X, initial_h, initial_c and outputs Y, Y_h, Y_c.
            If 0, the following shapes are expected:
                X.shape = [seq_length, batch_size, input_size],
                Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [num_directions, batch_size, hidden_size].
            If 1, the following shapes are expected:
                X.shape = [batch_size, seq_length, input_size],
                Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [batch_size, num_directions, hidden_size].

  ONNX LSTM optional outputs:
  out[0]: Y [seq_length, num_directions, batch_size, hidden_size] = stack of out[0] from QNN_LSTM with varient directions
  out[1]: Y_h [num_directions, batch_size, hidden_size] = stack of out[2] from QNN_LSTM with varient directions
  out[2]: Y_c [num_directions, batch_size, hidden_size] = stack of out[1] from QNN_LSTM with varient directions

  QNN LSTM inputs:
  in[0]: x_t: 2D of shape [batch_size, input_size] or
              3D of shape [time_steps, batch_size, input_size] if time_major
                          [batch_size, time_steps, input_size] else
  in[1]: W_xf: input-to-forget weights [num_units, input_size]      = ONNX in[1][direction, 2*hidden_size:3*hidden_size, :]
  in[2]: W_xc: input-to-cell weights [num_units, input_size]        = ONNX in[1][direction, 3*hidden_size:4*hidden_size, :]
  in[3]: W_xo: input-to-output weights [num_units, input_size]      = ONNX in[1][direction, 1*hidden_size:2*hidden_size, :]
  in[4]: W_hf: recurrent-to-forget weights [num_units, output_size] = ONNX in[2][direction, 2*hidden_size:3*hidden_size, :]
  in[5]: W_hc: recurrent-to-cell weights [num_units, output_size]   = ONNX in[2][direction, 3*hidden_size:4*hidden_size, :]
  in[6]: W_ho: recurrent-to-output weights [num_units, output_size] = ONNX in[2][direction, 1*hidden_size:2*hidden_size, :]
  in[7]: b_f: forget gate bias [num_units]                          = ONNX in[3][direction, 2*hidden_size:3*hidden_size] + in[3][direction, 6*hidden_size:7*hidden_size]
  in[8]: b_c: cell bias [num_units]                                 = ONNX in[3][direction, 3*hidden_size:4*hidden_size] + in[3][direction, 7*hidden_size:8*hidden_size]
  in[9]: b_o: output gate bias [num_units]                          = ONNX in[3][direction, 1*hidden_size:4*hidden_size] + in[3][direction, 5*hidden_size:6*hidden_size]

  # optional inputs
  in[10]: h_t_init: hidden state init [batch_size, output_size]     = ONNX in[5][direction]
  in[11]: c_t_init: cell state init [batch_size, num_units]         = ONNX in[6][direction]
  in[12]: The input layer normalization weights  ---> not supported on fp16 yet.
  in[13]: The forget layer normalization weights ---> not supported on fp16 yet.
  in[14]: The cell layer normalization weights   ---> not supported on fp16 yet.
  in[15]: The output layer normalization weights ---> not supported on fp16 yet.
  in[16]: W_xi: input-to-input weights [num_units, input_size]      = ONNX in[1][direction, 0*hidden_size:1*hidden_size, :]
  in[17]: W_hi: recurrent-to-input weights [num_units, output_size] = ONNX in[2][direction, 0*hidden_size:1*hidden_size, :]
  in[18]: W_ci: cell-to-input weights [num_units]                   = ONNX in[7][direction, 0*hidden_size:1*hidden_size]
  in[19]: W_cf: cell-to-forget weights [num_units]                  = ONNX in[7][direction, 2*hidden_size:3*hidden_size]
  in[20]: W_co: cell-to-output weights [num_units]                  = ONNX in[7][direction, 1*hidden_size:2*hidden_size]
  in[21]: b_i: input gate bias [num_units]                          = ONNX in[3][direction, 0*hidden_size:1*hidden_size] + in[3][direction, 4*hidden_size:5*hidden_size]
  in[22]: W_proj: projection weights [output_size, num_units]     ---> not used
  in[23]: b_proj: projection bias [output_size]                   ---> not used
  in[24]: reset: Determines if the internal state should be reset ---> not used

  QNN LSTM Parameters:
  - direction
  - cell_clip_threshold   ---> not used
  - output_clip_threshold ---> not used
  - time_major
  - input_gate_qscale     ---> not used since we fallback to fp16.
  - forget_gate_qscale    ---> not used since we fallback to fp16.
  - cell_gate_qscale      ---> not used since we fallback to fp16.
  - output_gate_qscale    ---> not used since we fallback to fp16.
  - hidden_state_offset   ---> not used since we fallback to fp16.
 -  hidden_state_qscale   ---> not used since we fallback to fp16.

  QNN LSTM outputs:
  out[0]: h_t 2D of shape [batch_size, output_size] or
              3D of shape [time_steps, batch_size, output_size] if time_major
                          [batch_size, time_steps, output_size] else
  out[1]: c_t [batch_size, num_unit]
  out[2]: o_t [batch_size, output_size]

  QNN LSTM optional outputs:
  out[3]: input_gate [batch_size, num_unit]      ---> not used
  out[4]: forget_gate [batch_size, num_unit]     ---> not used
  out[5]: cell_gate [batch_size, num_unit]       ---> not used
  out[6]: output_gate [batch_size, num_unit]     ---> not used
  out[7]: hidden_state [batch_size, output_size] ---> not used
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
  Ort::Status AddUnidirectionLSTM(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const std::string& direction,
                                  const std::vector<std::string>& input_names,
                                  const Ort::Logger& logger,
                                  const bool& do_op_validation,
                                  const bool& is_bidirection,
                                  std::vector<std::string>& uni_lstm_output_names) const;
};

Ort::Status LSTMOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  // Both unrolled/monolithic paths need X's shape to read seq_length/batch_size/input_size at compile time
  // Reject a dynamic/symbolic X shape here rather than hard-failing later in ComposeGraph.
  TensorInfo x_tensor_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], x_tensor_info));

  if (node_unit.Inputs().size() > 4 && node_unit.Inputs()[4].Exists()) {
    TensorInfo tensor_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[4], tensor_info));

    RETURN_IF_NOT(tensor_info.is_initializer, "QNN EP: dynamic sequence_length is not supported.");

    std::vector<uint8_t> sequence_lens_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(tensor_info.initializer_tensor, sequence_lens_bytes));
    const size_t num_elems = sequence_lens_bytes.size() / sizeof(int32_t);
    // Copy into an aligned int32_t buffer; sequence_lens_bytes.data() is only 1-byte aligned, so a
    // reinterpret_cast<const int32_t*> + deref would be undefined behavior.
    std::vector<int32_t> sequence_lens(num_elems);
    if (num_elems > 0) {
      std::memcpy(sequence_lens.data(), sequence_lens_bytes.data(), num_elems * sizeof(int32_t));
    }
    RETURN_IF(std::any_of(sequence_lens.begin(),
                          sequence_lens.end(),
                          [&sequence_lens](int32_t i) { return i != sequence_lens[0]; }),
              "QNN EP: Only support LSTM with same sequence length.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const float clip = node_helper.Get("clip", 0.0f);
  RETURN_IF(clip != 0,
            "QNN EP: LSTM 'clip' (gate-input activation clamp) has no equivalent QNN LSTM "
            "parameter — QNN LSTM exposes cell_clip_threshold/output_clip_threshold which clip "
            "different quantities and cannot represent ONNX gate-input clamping.");
  const std::vector<std::string> activations = node_helper.Get("activations", std::vector<std::string>{});
  RETURN_IF((activations.size() >= 3 && (activations[0] != "sigmoid" || activations[1] != "tanh" || activations[2] != "tanh")) ||
                (activations.size() == 6 && (activations[3] != "sigmoid" || activations[4] != "tanh" || activations[5] != "tanh")),
            "QNN EP doesn't support non-default activations for LSTM.");
  // TODO: Add support for layout==1
  const int64_t layout = node_helper.Get("layout", static_cast<int64_t>(0));
  RETURN_IF_NOT(layout == 0,
                ("QNN EP: Unsupported layout mode" + std::to_string(layout) + " for " + node_unit.Name()).c_str());
  return Ort::Status();
}

Ort::Status LSTMOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
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

Ort::Status LSTMOpBuilder::AddUnidirectionLSTM(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
                                               const std::string& direction,
                                               const std::vector<std::string>& input_names,
                                               const Ort::Logger& logger,
                                               const bool& do_op_validation,
                                               const bool& is_bidirection,
                                               std::vector<std::string>& uni_lstm_output_names) const {
  return rnn_details::AddUnidirectionLSTM(qnn_model_wrapper, node_unit, direction, input_names,
                                          logger, do_op_validation, is_bidirection,
                                          rnn_details::kNoReset(),
                                          uni_lstm_output_names);
}

Ort::Status LSTMOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  OrtNodeAttrHelper node_helper(node_unit);
  std::string direction = node_helper.Get("direction", "forward");
  RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= 8, "LSTM should receive inputs ranging from 3 to 8!");

  if (direction == "bidirectional") {
    std::vector<std::string> uni_lstm_output_names_forward, uni_lstm_output_names_reverse;
    RETURN_IF_ERROR(AddUnidirectionLSTM(qnn_model_wrapper, node_unit, "forward", input_names, logger, do_op_validation, true,
                                        uni_lstm_output_names_forward));
    RETURN_IF_ERROR(AddUnidirectionLSTM(qnn_model_wrapper, node_unit, "reverse", input_names, logger, do_op_validation, true,
                                        uni_lstm_output_names_reverse));

    // Concat forward and reverse output
    for (size_t i = 0; i < 3; i++) {
      TensorInfo output_info = {};
      if (node_unit.Outputs().size() > i && node_unit.Outputs()[i].Exists()) {
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[i], output_info));
        std::string onnx_output_name = node_unit.Outputs()[i].name;

        // param
        std::vector<std::string> concat_param_names;
        uint32_t concat_axis = 0;
        RETURN_IF_ERROR(rnn_details::DeriveNumDirectionsConcatAxis(output_info.shape, concat_axis));
        RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), onnx_output_name,
                                               concat_axis,
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
                                                      {uni_lstm_output_names_forward[i], uni_lstm_output_names_reverse[i]},
                                                      {onnx_output_name},
                                                      std::move(concat_param_names), do_op_validation),
                      "QNN EP: Failed to create Qnn Concat node.");
      }
    }
  } else {
    // not used, just a placeholder
    std::vector<std::string> uni_lstm_output_names;
    RETURN_IF_ERROR(AddUnidirectionLSTM(qnn_model_wrapper, node_unit, direction, input_names, logger, do_op_validation, false,
                                        uni_lstm_output_names));
  }
  return Ort::Status();
}

void CreateLSTMOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<LSTMOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
