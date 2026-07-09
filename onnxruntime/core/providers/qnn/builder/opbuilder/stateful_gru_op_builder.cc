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

// Builder for the qti_aisw custom "StatefulGru" block op.
//
// StatefulGru has the exact standard ONNX GRU input signature (in[0..5]: X, W, R, B, sequence_lens,
// initial_h) plus a trailing "reset" input appended at ONNX index 6. It maps to the same QNN_OP_GRU
// ("Gru") op as the standard GRU builder, with one difference: the reset input is wired into QNN GRU
// in[14] (the standard GRU builder uses a 14-slot input vector, indices 0..13; the reset needs slot
// 14. When reset is present, the vector grows to 15 and the native multi-time-step QNN GRU path
// is used so the reset is applied once per inference rather than once per unrolled timestep.
//
// The ONNX->QNN decomposition (per-timestep unroll with a single QNN GRU cell per step, W/R/B gate
// slicing, zero-bias / zero-initial-state stubs, per-step hidden-state chaining, and CPU vs HTP
// time_major handling) is shared with the standard GRU builder via rnn_details::AddUnidirectionGRU
// in rnn_op_utils.cc. This builder passes qnn_input_count=15 and a reset ResetInput when reset
// exists; otherwise it matches standard GRU with qnn_input_count=14 and kNoReset().
//
// Bidirectional is supported for float inputs (forward + reverse unroll joined by Concat). Native
// quantized (INT8/INT16) Gru is forward-only per HtpOpDefSupplement, so a bidirectional QDQ group
// is fp-degraded. A non-QDQ quantized StatefulGru remains rejected (the float "FP16" configuration
// covers both FLOAT_16 and FLOAT_32 and has no direction restriction).
class StatefulGruOpBuilder : public BaseOpBuilder {
 public:
  StatefulGruOpBuilder() : BaseOpBuilder("StatefulGruOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StatefulGruOpBuilder);

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
  static constexpr size_t kOnnxResetInputIndex = 6;
  // QNN GRU input slot for the reset signal (per QAIRT MasterOpDef Gru in[14]). in[13] = initial_h.
  static constexpr size_t kQnnGruResetInputIndex = 14;
  // QNN GRU input vector size including the reset slot (14 base slots 0..13 + reset at 14).
  static constexpr size_t kQnnGruInputCount = 15;

  Ort::Status AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                                 const OrtNodeUnit& node_unit,
                                 const std::string& direction,
                                 const std::vector<std::string>& input_names,
                                 const Ort::Logger& logger,
                                 const bool& do_op_validation,
                                 const bool& is_bidirection,
                                 const bool& use_fp_fallback,
                                 std::vector<std::string>& uni_gru_output_names) const;
};

Ort::Status StatefulGruOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
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
    RETURN_IF(std::any_of(sequence_lens.begin(), sequence_lens.end(),
                          [&sequence_lens](int32_t i) { return i != sequence_lens[0]; }),
              "QNN EP: Only support StatefulGru with same sequence length.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const float clip = node_helper.Get("clip", 0.0f);
  RETURN_IF(clip != 0,
            "QNN EP: StatefulGru 'clip' (gate-input activation clamp) has no equivalent QNN GRU "
            "parameter and cannot be expressed in the QNN graph.");

  // HtpOpDefSupplement restricts native INT8/INT16 Gru to forward direction. A QDQ group can be
  // fp-degraded, but a non-QDQ quantized StatefulGru must be rejected.
  const std::string direction = node_helper.Get("direction", "forward");
  if (direction == "bidirectional") {
    TensorInfo input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
    RETURN_IF(input_info.quant_param.IsQuantized() && node_unit.UnitType() != OrtNodeUnit::Type::QDQGroup,
              "QNN EP: bidirectional StatefulGru is only supported for float (FP16/FP32); "
              "quantized (INT8/INT16) StatefulGru is forward-only (per HtpOpDefSupplement).");
  }

  if (node_unit.Inputs().size() > kOnnxResetInputIndex && node_unit.Inputs()[kOnnxResetInputIndex].Exists()) {
    TensorInfo reset_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[kOnnxResetInputIndex], reset_info));
    RETURN_IF_NOT(reset_info.qnn_data_type == QNN_DATATYPE_BOOL_8,
                  "QNN EP: StatefulGru reset input must have QNN BOOL_8 data type.");
  }

  return Ort::Status();
}

Ort::Status StatefulGruOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const Ort::Logger& logger,
                                                std::vector<std::string>& input_names,
                                                bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& onnx_inputs = node_unit.Inputs();
  // Iterate every ONNX input including the trailing reset input (ONNX index 6). Non-existent optional
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

Ort::Status StatefulGruOpBuilder::AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                                                     const OrtNodeUnit& node_unit,
                                                     const std::string& direction,
                                                     const std::vector<std::string>& input_names,
                                                     const Ort::Logger& logger,
                                                     const bool& do_op_validation,
                                                     const bool& is_bidirection,
                                                     const bool& use_fp_fallback,
                                                     std::vector<std::string>& uni_gru_output_names) const {
  const auto& onnx_inputs = node_unit.Inputs();
  const std::string reset_name = (onnx_inputs.size() > kOnnxResetInputIndex &&
                                  onnx_inputs[kOnnxResetInputIndex].Exists())
                                     ? input_names[kOnnxResetInputIndex]
                                     : "";
  return rnn_details::AddUnidirectionGRU(qnn_model_wrapper, node_unit, direction, input_names,
                                         logger, do_op_validation, is_bidirection,
                                         use_fp_fallback,
                                         reset_name.empty() ? kQnnGruInputCount - 1 : kQnnGruInputCount,
                                         reset_name.empty() ? rnn_details::kNoReset()
                                                            : rnn_details::ResetInput{reset_name, kQnnGruResetInputIndex},
                                         uni_gru_output_names);
}

Ort::Status StatefulGruOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                              const OrtNodeUnit& node_unit,
                                                              std::vector<std::string>&& input_names,
                                                              const Ort::Logger& logger,
                                                              bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  OrtNodeAttrHelper node_helper(node_unit);
  std::string direction = node_helper.Get("direction", "forward");
  // Standard GRU inputs (3..6) plus the trailing stateful reset at ONNX index 6 -> up to 7 inputs.
  RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= 7, "StatefulGru should receive inputs ranging from 3 to 7!");

  const bool is_qdq = node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup;
  const int64_t linear_before_reset = node_helper.Get("linear_before_reset", static_cast<int64_t>(0));
  std::vector<TensorInfo> input_infos(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].Exists()) {
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[i], input_infos[i]));
    }
  }
  // The selector enforces structural QDQ well-formedness and folds the DQ -> Stateful GRU -> Q group;
  // ShouldFpDegradeQdqGru() makes the op-semantic decision of whether this group runs as a genuine
  // quantized StatefulGru or is fp-degraded (explicit Dequantize -> fp32 Stateful GRU -> Quantize, all on QNN).
  const bool use_fp_fallback =
      is_qdq && rnn_details::ShouldFpDegradeQdqGru(input_infos, inputs, outputs, direction, linear_before_reset);

  if (use_fp_fallback) {
    for (size_t i = 0; i < input_names.size(); i++) {
      // sequence_lens and the BOOL_8 reset signal have no quantization parameters, so the
      // IsQuantized() guard leaves them unchanged.
      if (!inputs[i].Exists() || input_names[i].empty() || !input_infos[i].quant_param.IsQuantized()) {
        continue;
      }
      const std::string dq_name = utils::UniqueNameGenerator().New(input_names[i], "_stateful_gru_to_f32");
      RETURN_IF_ERROR(qnn_model_wrapper.AddDequantizeNode(input_names[i], dq_name, QNN_DATATYPE_FLOAT_32,
                                                          input_infos[i].shape, do_op_validation));
      input_names[i] = dq_name;
    }
  }

  if (direction == "bidirectional") {
    // Unroll each direction independently (is_bidirection=true makes AddUnidirectionGRU emit
    // per-direction intermediate outputs), then Concat the forward and reverse results along the
    // num_directions axis. StatefulGru has up to two outputs: Y and Y_h.
    std::vector<std::string> fwd_out, rev_out;
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, "forward", input_names, logger,
                                       do_op_validation, true, use_fp_fallback, fwd_out));
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, "reverse", input_names, logger,
                                       do_op_validation, true, use_fp_fallback, rev_out));

    for (size_t i = 0; i < 2; i++) {
      TensorInfo output_info = {};
      if (outputs.size() > i && outputs[i].Exists()) {
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[i], output_info));
        const std::string& name = outputs[i].name;

        // Concat axis = the num_directions dimension (rank - 3 for Y [seq, num_dir, batch, hidden]
        // and Y_h [num_dir, batch, hidden]).
        std::vector<std::string> cp;
        uint32_t concat_axis = 0;
        RETURN_IF_ERROR(rnn_details::DeriveNumDirectionsConcatAxis(output_info.shape, concat_axis));
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
                                                        {fwd_out[i], rev_out[i]},
                                                        {concat_fp},
                                                        std::move(cp),
                                                        do_op_validation),
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
                                                        {fwd_out[i], rev_out[i]},
                                                        {name},
                                                        std::move(cp),
                                                        do_op_validation),
                        "Failed to create Concat node.");
        }
      }
    }
  } else {
    std::vector<std::string> uni_out;
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, direction, input_names, logger,
                                       do_op_validation, false, use_fp_fallback, uni_out));
    if (use_fp_fallback) {
      for (size_t i = 0; i < 2; i++) {
        if (outputs.size() > i && outputs[i].Exists()) {
          TensorInfo output_info = {};
          RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[i], output_info));
          const Qnn_TensorType_t output_type = qnn_model_wrapper.IsGraphOutput(outputs[i].name)
                                                   ? QNN_TENSOR_TYPE_APP_READ
                                                   : QNN_TENSOR_TYPE_NATIVE;
          RETURN_IF_ERROR(qnn_model_wrapper.AddQuantizeNode(uni_out[i], outputs[i].name, output_type,
                                                            output_info.qnn_data_type, output_info.quant_param.Copy(),
                                                            output_info.shape, do_op_validation));
        }
      }
    }
  }
  return Ort::Status();
}

void CreateStatefulGruOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<StatefulGruOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
