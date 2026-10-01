// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/rnn_op_builder_base.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

// NOTE: The ONNX->QNN decomposition (gate slicing, ifoc reordering, bias summation, bidirectional
// Concat) is shared with the qti_aisw "StatefulLstm" builder via LstmBaseOpBuilder and
// rnn_utils::AddUnidirectionLSTM. Changes in that shared path affect both builders.
class LSTMOpBuilder : public LstmBaseOpBuilder {
 public:
  LSTMOpBuilder() : LstmBaseOpBuilder("LSTMOpBuilder") {}
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

  const char* OpDisplayName() const override { return "LSTM"; }
  size_t MaxOnnxInputCount() const override { return 8; }

  Ort::Status ValidateAdditionalSupport(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  rnn_utils::ResetInput GetResetInput(const OrtNodeUnit& node_unit,
                                      const std::vector<std::string>& input_names) const override;
};

Ort::Status LSTMOpBuilder::ValidateAdditionalSupport(QnnModelWrapper& qnn_model_wrapper,
                                                     const OrtNodeUnit& node_unit,
                                                     const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);
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

rnn_utils::ResetInput LSTMOpBuilder::GetResetInput(const OrtNodeUnit& node_unit,
                                                   const std::vector<std::string>& input_names) const {
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(input_names);
  return rnn_utils::NoQnnResetInput();
}

void CreateLSTMOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<LSTMOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
