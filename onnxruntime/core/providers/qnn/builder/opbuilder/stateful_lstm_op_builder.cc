// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <cstring>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/rnn_op_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Builder for the qti_aisw custom "StatefulLstm" block op.
//
// StatefulLstm has the exact standard ONNX LSTM input signature (in[0..7]: X, W, R, B, sequence_lens,
// initial_h, initial_c, P) plus a trailing "reset" input appended at ONNX index 8. It maps to the same
// QNN_OP_LSTM ("Lstm") op as the standard LSTM builder, with one difference: the reset input is wired
// into QNN LSTM in[24] (which the standard LSTM builder leaves as null_tensor).
//
// The ONNX->QNN decomposition (per-direction StridedSlice gate slicing, ifoc gate reordering, bias
// summation, zero-bias / zero-initial-state stubs, bidirectional Concat, and the QNN LSTM input
// vector of size 25) is shared with the standard LSTM builder via rnn_details::AddUnidirectionLSTM
// in rnn_op_utils.cc. This builder passes a reset ResetInput; the standard LSTM passes kNoReset().
//
// Bidirectional is supported for all dtypes (float and quantized) via a forward + reverse unroll
// joined by Concat, matching HtpOpDefSupplement, which places no direction constraint on
// the Lstm op.
class StatefulLstmOpBuilder : public BaseOpBuilder {
 public:
  StatefulLstmOpBuilder() : BaseOpBuilder("StatefulLstmOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StatefulLstmOpBuilder);

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
  // ONNX index of the trailing stateful "reset" input.
  static constexpr size_t kOnnxResetInputIndex = 8;
  // QNN LSTM input slot for the reset signal (per QAIRT MasterOpDef Lstm in[24]).
  static constexpr size_t kQnnLstmResetInputIndex = 24;

  Ort::Status AddUnidirectionLSTM(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const std::string& direction,
                                  const std::vector<std::string>& input_names,
                                  const Ort::Logger& logger,
                                  const bool& do_op_validation,
                                  const bool& is_bidirection,
                                  std::vector<std::string>& uni_lstm_output_names) const;
};

Ort::Status StatefulLstmOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
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
    // Copy into an aligned int32_t buffer; sequence_lens_bytes.data() is only 1-byte aligned, so a
    // reinterpret_cast<const int32_t*> + deref would be undefined behavior.
    std::vector<int32_t> sequence_lens(num_elems);
    if (num_elems > 0) {
      std::memcpy(sequence_lens.data(), sequence_lens_bytes.data(), num_elems * sizeof(int32_t));
    }
    RETURN_IF(std::any_of(sequence_lens.begin(),
                          sequence_lens.end(),
                          [&sequence_lens](int32_t i) { return i != sequence_lens[0]; }),
              "QNN EP: Only support StatefulLstm with same sequence length.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const float clip = node_helper.Get("clip", 0.0f);
  RETURN_IF(clip != 0,
            "QNN EP: StatefulLstm 'clip' (gate-input activation clamp) has no equivalent QNN LSTM "
            "parameter — QNN LSTM exposes cell_clip_threshold/output_clip_threshold which clip "
            "different quantities and cannot represent ONNX gate-input clamping.");
  const int64_t input_forget = node_helper.Get("input_forget", static_cast<int64_t>(0));
  RETURN_IF(input_forget != 0,
            "QNN EP doesn't support input_forget=1 for StatefulLstm.");

  // No direction restriction: HtpOpDefSupplement places no direction constraint on the
  // Lstm op for any dtype, and this builder implements bidirectional codegen (forward + reverse
  // unroll joined by Concat) for both float and quantized inputs.
  return Ort::Status();
}

Ort::Status StatefulLstmOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const OrtNodeUnit& node_unit,
                                                 const Ort::Logger& logger,
                                                 std::vector<std::string>& input_names,
                                                 bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& onnx_inputs = node_unit.Inputs();
  // Iterate every ONNX input including the trailing reset input (ONNX index 8). Non-existent optional
  // inputs are represented by an empty name so downstream index math stays aligned.
  for (size_t i = 0; i < onnx_inputs.size(); i++) {
    if (onnx_inputs[i].Exists()) {
      RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[i], logger, input_names));
    } else {
      input_names.emplace_back("");
    }
  }
  return Ort::Status();
}

Ort::Status StatefulLstmOpBuilder::AddUnidirectionLSTM(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const std::string& direction,
                                                       const std::vector<std::string>& input_names,
                                                       const Ort::Logger& logger,
                                                       const bool& do_op_validation,
                                                       const bool& is_bidirection,
                                                       std::vector<std::string>& uni_lstm_output_names) const {
  const auto& onnx_inputs = node_unit.Inputs();
  const std::string reset_name = (onnx_inputs.size() > kOnnxResetInputIndex &&
                                  onnx_inputs[kOnnxResetInputIndex].Exists())
                                     ? input_names[kOnnxResetInputIndex]
                                     : "";
  return rnn_details::AddUnidirectionLSTM(qnn_model_wrapper, node_unit, direction, input_names,
                                          logger, do_op_validation, is_bidirection,
                                          {reset_name, kQnnLstmResetInputIndex},
                                          uni_lstm_output_names);
}

Ort::Status StatefulLstmOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                               const OrtNodeUnit& node_unit,
                                                               std::vector<std::string>&& input_names,
                                                               const Ort::Logger& logger,
                                                               bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  OrtNodeAttrHelper node_helper(node_unit);
  std::string direction = node_helper.Get("direction", "forward");
  // Standard LSTM inputs (3..8) plus the trailing stateful reset at ONNX index 8 -> up to 9 inputs.
  RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= 9, "StatefulLstm should receive inputs ranging from 3 to 9!");

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

void CreateStatefulLstmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<StatefulLstmOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
