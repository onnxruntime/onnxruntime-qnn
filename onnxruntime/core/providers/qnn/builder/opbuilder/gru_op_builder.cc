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

// NOTE: The ONNX->QNN decomposition (gate slicing, time-step unroll, bidirectional Concat)
// is shared with the qti_aisw "StatefulGru" builder via rnn_details::AddUnidirectionGRU in
// rnn_op_utils.cc. Both builders delegate to that function — changes there affect both.
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
                                 const bool& use_fp_fallback,
                                 std::vector<std::string>& uni_gru_output_names) const;
};

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
    // Copy into an aligned int32_t buffer; sequence_lens_bytes.data() is only 1-byte aligned, so a
    // reinterpret_cast<const int32_t*> + deref would be undefined behavior.
    std::vector<int32_t> sequence_lens(num_elems);
    if (num_elems > 0) {
      std::memcpy(sequence_lens.data(), sequence_lens_bytes.data(), num_elems * sizeof(int32_t));
    }
    RETURN_IF(std::any_of(sequence_lens.begin(), sequence_lens.end(),
                          [&sequence_lens](int32_t i) { return i != sequence_lens[0]; }),
              "QNN EP: Only support GRU with same sequence length.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  RETURN_IF(node_helper.Get("layout", static_cast<int64_t>(0)) != 0,
            "QNN EP doesn't support layout=1 for GRU (ORT CPU EP cannot provide a reference for accuracy validation).");
  const float clip = node_helper.Get("clip", 0.0f);
  RETURN_IF(clip != 0,
            "QNN EP: GRU 'clip' (gate-input activation clamp) has no equivalent QNN GRU "
            "parameter and cannot be expressed in the QNN graph.");
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

Ort::Status GRUOpBuilder::AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const std::string& direction,
                                             const std::vector<std::string>& input_names,
                                             const Ort::Logger& logger,
                                             const bool& do_op_validation,
                                             const bool& is_bidirection,
                                             const bool& use_fp_fallback,
                                             std::vector<std::string>& uni_gru_output_names) const {
  return rnn_details::AddUnidirectionGRU(qnn_model_wrapper, node_unit, direction, input_names,
                                         logger, do_op_validation, is_bidirection,
                                         use_fp_fallback,
                                         14, rnn_details::kNoReset(),
                                         uni_gru_output_names);
}

namespace {

// Decide whether a QDQ GRU group runs as a genuine quantized Gru or must be fp-degraded. HTP has two
// native quantized Gru configs (HtpOpDefSupplement): an INT8 combo (X/W/R/initial_h u8) and an INT16
// combo (X/initial_h u16, W/R u16-or-u8), both forward-only and both with an int32 (SFIXED_POINT_32)
// bias -- no config takes a quantized-integer bias. Everything else -- a non-forward direction, a
// missing optional output, or a non-supported input dtype (e.g. a non-int32 bias) -- is fp-degraded
// (explicit Dequantize -> fp32 GRU -> Quantize, all on QNN) so the numeric result is still produced on
// QNN. LBR=0 additionally fp-degrades the u8 combo (its u8 cell widens to a mixed-width QUint16Crouton
// that fails HTP finalize, 1002); the u16 combo is already 16-bit with no such widening, so LBR=0 stays
// native there.
bool ShouldFpDegradeQdqGru(gsl::span<const TensorInfo> input_infos,
                           gsl::span<const OrtNodeUnitIODef> inputs,
                           gsl::span<const OrtNodeUnitIODef> outputs,
                           const std::string& direction,
                           int64_t linear_before_reset) {
  const bool missing_output = !(outputs.size() >= 2 && outputs[0].Exists() && outputs[1].Exists());
  const bool is_forward = direction == "forward";
  auto dtype_at = [&](size_t i) { return input_infos[i].qnn_data_type; };
  auto has_input = [&](size_t i) { return inputs.size() > i && inputs[i].Exists(); };
  const bool genuine_u8_combo =
      dtype_at(0) == QNN_DATATYPE_UFIXED_POINT_8 &&
      dtype_at(1) == QNN_DATATYPE_UFIXED_POINT_8 &&
      dtype_at(2) == QNN_DATATYPE_UFIXED_POINT_8 &&
      (!has_input(3) || dtype_at(3) == QNN_DATATYPE_SFIXED_POINT_32) &&
      (!has_input(5) || dtype_at(5) == QNN_DATATYPE_UFIXED_POINT_8);
  const bool genuine_u16_combo =
      dtype_at(0) == QNN_DATATYPE_UFIXED_POINT_16 &&
      (dtype_at(1) == QNN_DATATYPE_UFIXED_POINT_16 || dtype_at(1) == QNN_DATATYPE_UFIXED_POINT_8) &&
      (dtype_at(2) == QNN_DATATYPE_UFIXED_POINT_16 || dtype_at(2) == QNN_DATATYPE_UFIXED_POINT_8) &&
      has_input(3) && dtype_at(3) == QNN_DATATYPE_SFIXED_POINT_32 &&
      (!has_input(5) || dtype_at(5) == QNN_DATATYPE_UFIXED_POINT_16);
  return ((linear_before_reset == 0) && !genuine_u16_combo) || !is_forward ||
         missing_output || !(genuine_u8_combo || genuine_u16_combo);
}

}  // namespace

Ort::Status GRUOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      std::vector<std::string>&& input_names,
                                                      const Ort::Logger& logger,
                                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  OrtNodeAttrHelper node_helper(node_unit);
  std::string direction = node_helper.Get("direction", "forward");
  RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= 6, "GRU should receive inputs ranging from 3 to 6!");

  const bool is_qdq = node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup;
  const int64_t linear_before_reset = node_helper.Get("linear_before_reset", static_cast<int64_t>(0));

  std::vector<TensorInfo> input_infos(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].Exists()) {
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[i], input_infos[i]));
    }
  }

  // The selector enforces structural QDQ well-formedness and folds the DQ -> GRU -> Q group;
  // ShouldFpDegradeQdqGru() makes the op-semantic decision of whether this group runs as a genuine
  // quantized Gru or is fp-degraded (explicit Dequantize -> fp32 GRU -> Quantize, all on QNN).
  const bool use_fp_fallback =
      is_qdq && ShouldFpDegradeQdqGru(input_infos, inputs, outputs, direction, linear_before_reset);

  // fp-degrade input side: Dequantize each present quantized input to fp32 once (shared by both
  // directions in the bidirectional case) and rewrite input_names in place. seq_lens (idx 4) is
  // never quantized, so the IsQuantized() guard leaves it untouched.
  if (use_fp_fallback) {
    for (size_t i = 0; i < input_names.size(); i++) {
      if (!inputs[i].Exists() || input_names[i].empty() || !input_infos[i].quant_param.IsQuantized()) {
        continue;
      }
      const std::string dq_name = utils::UniqueNameGenerator().New(input_names[i], "_gru_to_f32");
      RETURN_IF_ERROR(qnn_model_wrapper.AddDequantizeNode(input_names[i], dq_name, QNN_DATATYPE_FLOAT_32,
                                                          input_infos[i].shape, do_op_validation));
      input_names[i] = dq_name;
    }
  }

  if (direction == "bidirectional") {
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
        std::vector<std::string> cp;
        uint32_t concat_axis = 0;
        RETURN_IF_ERROR(rnn_details::DeriveNumDirectionsConcatAxis(output_info.shape, concat_axis));
        const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(name);
        if (use_fp_fallback) {
          // Concat the two fp32 direction temps into an fp32 temp, then Quantize to the u8 ONNX output.
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
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, direction, input_names, logger,
                                       do_op_validation, false, use_fp_fallback, uni_out));
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

void CreateGRUOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GRUOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
