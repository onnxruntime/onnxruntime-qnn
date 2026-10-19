// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/opbuilder/rnn_op_builder_base.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

Ort::Status ValidateSequenceLens(QnnModelWrapper& qnn_model_wrapper,
                                 const OrtNodeUnit& node_unit,
                                 const char* op_display_name) {
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
    RETURN_IF(std::any_of(sequence_lens.begin(), sequence_lens.end(),
                          [&sequence_lens](int32_t i) { return i != sequence_lens[0]; }),
              (std::string("QNN EP: Only support ") + op_display_name + " with same sequence length.").c_str());
  }

  return Ort::Status();
}

Ort::Status ValidateUnsupportedClip(const OrtNodeUnit& node_unit,
                                    const char* op_display_name,
                                    const char* qnn_op_name,
                                    const char* unsupported_reason,
                                    bool allows_explicit_zero) {
  OrtNodeAttrHelper node_helper(node_unit);
  const float clip = node_helper.Get("clip", 0.0f);
  RETURN_IF(clip != 0 || (!allows_explicit_zero && node_helper.HasAttr("clip")),
            (std::string("QNN EP: ") + op_display_name +
             " 'clip' (gate-input activation clamp) has no equivalent " + qnn_op_name + " " +
             unsupported_reason)
                .c_str());
  return Ort::Status();
}

Ort::Status ValidateRnnArity(const OrtNodeUnit& node_unit,
                             const char* op_display_name,
                             size_t max_input_count,
                             size_t max_output_count) {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= max_input_count,
                (std::string(op_display_name) + " should receive inputs ranging from 3 to " +
                 std::to_string(max_input_count) + "!")
                    .c_str());
  RETURN_IF_NOT(outputs.size() <= max_output_count,
                (std::string(op_display_name) + " should produce at most " +
                 std::to_string(max_output_count) + " outputs!")
                    .c_str());
  return Ort::Status();
}

Ort::Status ValidateDirection(const OrtNodeUnit& node_unit, const char* op_display_name) {
  OrtNodeAttrHelper node_helper(node_unit);
  const std::string direction = node_helper.Get("direction", "forward");
  RETURN_IF_NOT(direction == "forward" || direction == "reverse" || direction == "bidirectional",
                (std::string("QNN EP: ") + op_display_name +
                 " direction must be one of forward, reverse, or bidirectional.")
                    .c_str());
  return Ort::Status();
}

}  // namespace

Ort::Status GruBaseOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit,
                                            const Ort::Logger& logger) const {
  RETURN_IF_ERROR(ValidateRnnArity(node_unit, OpDisplayName(), MaxOnnxInputCount(), /*max_output_count=*/2));
  RETURN_IF_ERROR(ValidateDirection(node_unit, OpDisplayName()));
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape),
                (std::string("QNN EP: dynamic ") + OpDisplayName() + " input shape is not supported.").c_str());

  RETURN_IF_ERROR(ValidateSequenceLens(qnn_model_wrapper, node_unit, OpDisplayName()));
  RETURN_IF_ERROR(ValidateUnsupportedClip(node_unit, OpDisplayName(), "QNN GRU",
                                          "parameter and cannot be expressed in the QNN graph.",
                                          AllowsExplicitZeroClip()));
  RETURN_IF_ERROR(ValidateAdditionalSupport(qnn_model_wrapper, node_unit, logger));
  return Ort::Status();
}

Ort::Status GruBaseOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit,
                                            const Ort::Logger& logger,
                                            std::vector<std::string>& input_names,
                                            bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& onnx_inputs = node_unit.Inputs();
  // Iterate every ONNX input, including trailing stateful reset inputs. Non-existent optional
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

Ort::Status GruBaseOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                          const OrtNodeUnit& node_unit,
                                                          std::vector<std::string>&& input_names,
                                                          const Ort::Logger& logger,
                                                          bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  OrtNodeAttrHelper node_helper(node_unit);
  std::string direction = node_helper.Get("direction", "forward");
  RETURN_IF_ERROR(ValidateRnnArity(node_unit, OpDisplayName(), MaxOnnxInputCount(), /*max_output_count=*/2));

  const bool is_qdq = node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup;
  const int64_t linear_before_reset = node_helper.Get("linear_before_reset", static_cast<int64_t>(0));

  std::vector<TensorInfo> input_infos(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].Exists()) {
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[i], input_infos[i]));
    }
  }

  // The selector enforces structural QDQ well-formedness and folds the DQ -> GRU/StatefulGru -> Q
  // group; ShouldFpDegradeQdqGru() makes the op-semantic decision of whether this group runs as a
  // genuine quantized Gru or is fp-degraded (explicit Dequantize -> fp32 GRU -> Quantize, all on QNN).
  const bool use_fp_fallback =
      is_qdq && rnn_utils::ShouldFpDegradeQdqGru(input_infos, inputs, outputs, direction, linear_before_reset);

  // fp-degrade input side: Dequantize each present quantized input to fp32 once (shared by both
  // directions in the bidirectional case) and rewrite input_names in place. seq_lens and the BOOL_8
  // reset signal have no quantization parameters, so the IsQuantized() guard leaves them unchanged.
  if (use_fp_fallback) {
    for (size_t i = 0; i < input_names.size(); i++) {
      if (!inputs[i].Exists() || input_names[i].empty() || !input_infos[i].quant_param.IsQuantized()) {
        continue;
      }
      const std::string dq_name = utils::UniqueNameGenerator().New(input_names[i], FpFallbackInputSuffix());
      RETURN_IF_ERROR(qnn_model_wrapper.AddDequantizeNode(input_names[i], dq_name, QNN_DATATYPE_FLOAT_32,
                                                          input_infos[i].shape, do_op_validation));
      input_names[i] = dq_name;
    }
  }

  auto add_unidirection_gru = [&](const std::string& uni_direction,
                                  bool is_bidirection,
                                  std::vector<std::string>& uni_gru_output_names) -> Ort::Status {
    const rnn_utils::ResetInput reset = GetResetInput(node_unit, input_names);
    return rnn_utils::AddUnidirectionGRU(qnn_model_wrapper, node_unit, uni_direction, input_names,
                                         logger, do_op_validation, is_bidirection,
                                         use_fp_fallback,
                                         GetQnnGruInputCount(reset), reset,
                                         uni_gru_output_names);
  };

  if (direction == "bidirectional") {
    // Unroll each direction independently (is_bidirection=true makes AddUnidirectionGRU emit
    // per-direction intermediate outputs), then Concat the forward and reverse results along the
    // num_directions axis. GRU/StatefulGru has up to two outputs: Y and Y_h.
    std::vector<std::string> fwd_out, rev_out;
    RETURN_IF_ERROR(add_unidirection_gru("forward", true, fwd_out));
    RETURN_IF_ERROR(add_unidirection_gru("reverse", true, rev_out));

    for (size_t i = 0; i < 2; i++) {
      TensorInfo output_info = {};
      if (outputs.size() > i && outputs[i].Exists()) {
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[i], output_info));
        const std::string& name = outputs[i].name;

        // Concat axis = the num_directions dimension (rank - 3 for Y [seq, num_dir, batch, hidden]
        // and Y_h [num_dir, batch, hidden]).
        std::vector<std::string> cp;
        uint32_t concat_axis = 0;
        RETURN_IF_ERROR(rnn_utils::DeriveNumDirectionsConcatAxis(output_info.shape, concat_axis));
        const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(name);
        if (use_fp_fallback) {
          // Concat the two fp32 direction temps into an fp32 temp, then Quantize to the ONNX output.
          const std::string concat_fp = utils::UniqueNameGenerator().New(name, "_gru_concat_f32");
          RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), concat_fp,
                                                 concat_axis, QNN_OP_CONCAT_PARAM_AXIS, cp));
          QnnTensorWrapper tw(concat_fp, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                              QnnQuantParamsWrapper(), std::vector<uint32_t>(output_info.shape));
          RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "Failed to add fp32 Concat output.");
          RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_CONCAT),
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                                        {fwd_out[i], rev_out[i]}, {concat_fp}, std::move(cp), do_op_validation),
                        "Failed to create fp32 Concat node.");
          RETURN_IF_ERROR(qnn_model_wrapper.AddQuantizeNode(
              concat_fp, name, is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
              output_info.qnn_data_type, output_info.quant_param.Copy(), output_info.shape, do_op_validation));
        } else {
          RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), name,
                                                 concat_axis, QNN_OP_CONCAT_PARAM_AXIS, cp));
          Qnn_TensorType_t tt = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
          QnnTensorWrapper tw(name, tt, output_info.qnn_data_type, output_info.quant_param.Copy(),
                              std::vector<uint32_t>(output_info.shape));
          RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "Failed to add Concat output.");
          RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_CONCAT),
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                                        {fwd_out[i], rev_out[i]}, {name}, std::move(cp), do_op_validation),
                        "Failed to create Concat node.");
        }
      }
    }
  } else {
    std::vector<std::string> uni_out;
    RETURN_IF_ERROR(add_unidirection_gru(direction, false, uni_out));
    if (use_fp_fallback) {
      // uni_out holds fp32 temps (or "" for an absent output). Quantize each present output to its
      // u8 ONNX name.
      for (size_t i = 0; i < 2; i++) {
        if (outputs.size() > i && outputs[i].Exists()) {
          TensorInfo output_info = {};
          RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[i], output_info));
          const std::string& name = outputs[i].name;
          const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(name);
          RETURN_IF_ERROR(qnn_model_wrapper.AddQuantizeNode(
              uni_out[i], name, is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
              output_info.qnn_data_type, output_info.quant_param.Copy(), output_info.shape, do_op_validation));
        }
      }
    }
  }
  return Ort::Status();
}

Ort::Status LstmBaseOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const Ort::Logger& logger) const {
  RETURN_IF_ERROR(ValidateRnnArity(node_unit, OpDisplayName(), MaxOnnxInputCount(), /*max_output_count=*/3));
  RETURN_IF_ERROR(ValidateDirection(node_unit, OpDisplayName()));
  RETURN_IF_ERROR(ValidateInputShapeSupport(qnn_model_wrapper, node_unit, logger));
  RETURN_IF_ERROR(ValidateSequenceLens(qnn_model_wrapper, node_unit, OpDisplayName()));
  RETURN_IF_ERROR(ValidateUnsupportedClip(node_unit, OpDisplayName(), "QNN LSTM",
                                          "parameter — QNN LSTM exposes cell_clip_threshold/output_clip_threshold which "
                                          "clip different quantities and cannot represent ONNX gate-input clamping.",
                                          /*allows_explicit_zero=*/true));
  RETURN_IF_ERROR(ValidateAdditionalSupport(qnn_model_wrapper, node_unit, logger));
  return Ort::Status();
}

Ort::Status LstmBaseOpBuilder::ValidateInputShapeSupport(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  // Both unrolled/monolithic paths need X's shape to read seq_length/batch_size/input_size at compile time.
  // Reject a dynamic/symbolic X shape during support validation rather than hard-failing later in ComposeGraph.
  TensorInfo x_tensor_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], x_tensor_info));
  return Ort::Status();
}

Ort::Status LstmBaseOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const Ort::Logger& logger,
                                             std::vector<std::string>& input_names,
                                             bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& onnx_inputs = node_unit.Inputs();
  // Iterate every ONNX input, including trailing stateful reset inputs. Non-existent optional
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

Ort::Status LstmBaseOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                           const OrtNodeUnit& node_unit,
                                                           std::vector<std::string>&& input_names,
                                                           const Ort::Logger& logger,
                                                           bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  std::string direction = node_helper.Get("direction", "forward");
  RETURN_IF_ERROR(ValidateRnnArity(node_unit, OpDisplayName(), MaxOnnxInputCount(), /*max_output_count=*/3));

  auto add_unidirection_lstm = [&](const std::string& uni_direction,
                                   bool is_bidirection,
                                   std::vector<std::string>& uni_lstm_output_names) -> Ort::Status {
    return rnn_utils::AddUnidirectionLSTM(qnn_model_wrapper, node_unit, uni_direction, input_names,
                                          logger, do_op_validation, is_bidirection,
                                          GetResetInput(node_unit, input_names),
                                          uni_lstm_output_names);
  };

  if (direction == "bidirectional") {
    std::vector<std::string> uni_lstm_output_names_forward, uni_lstm_output_names_reverse;
    RETURN_IF_ERROR(add_unidirection_lstm("forward", true, uni_lstm_output_names_forward));
    RETURN_IF_ERROR(add_unidirection_lstm("reverse", true, uni_lstm_output_names_reverse));

    // Concat forward and reverse output.
    for (size_t i = 0; i < 3; i++) {
      TensorInfo output_info = {};
      if (node_unit.Outputs().size() > i && node_unit.Outputs()[i].Exists()) {
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[i], output_info));
        std::string onnx_output_name = node_unit.Outputs()[i].name;

        // param
        std::vector<std::string> concat_param_names;
        uint32_t concat_axis = 0;
        RETURN_IF_ERROR(rnn_utils::DeriveNumDirectionsConcatAxis(output_info.shape, concat_axis));
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
    RETURN_IF_ERROR(add_unidirection_lstm(direction, false, uni_lstm_output_names));
  }
  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
