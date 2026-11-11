// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/rnn_op_builder_base.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/custom_op/qnn_qti_aisw_custom_op.h"

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
// time_major handling) is shared with the standard GRU builder via GruBaseOpBuilder and
// rnn_utils::AddUnidirectionGRU. This builder always passes qnn_input_count=15. When reset is
// omitted, rnn_utils materializes the BlockOp default false at QNN input 14 instead of using QNN's
// incompatible omitted-input default true.
//
// Bidirectional is supported for float inputs (forward + reverse unroll joined by Concat). Native
// quantized (INT8/INT16) Gru is forward-only per HtpOpDefSupplement, so a bidirectional QDQ group
// is fp-degraded. A non-QDQ quantized StatefulGru remains rejected (the float "FP16" configuration
// covers both FLOAT_16 and FLOAT_32 and has no direction restriction).
class StatefulGruOpBuilder : public GruBaseOpBuilder {
 public:
  StatefulGruOpBuilder() : GruBaseOpBuilder("StatefulGruOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StatefulGruOpBuilder);

 protected:
  const char* OpDisplayName() const override { return "StatefulGru"; }
  size_t MaxOnnxInputCount() const override { return 7; }
  const char* FpFallbackInputSuffix() const override { return "_stateful_gru_to_f32"; }

  Ort::Status ValidateAdditionalSupport(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  rnn_utils::ResetInput GetResetInput(const OrtNodeUnit& node_unit,
                                      const std::vector<std::string>& input_names) const override;

  size_t GetQnnGruInputCount(const rnn_utils::ResetInput& reset) const override;

 private:
  // QNN GRU input slot for the reset signal (per QAIRT MasterOpDef Gru in[14]). in[13] = initial_h.
  static constexpr size_t kQnnGruResetInputIndex = 14;
  // QNN GRU input vector size including the reset slot (14 base slots 0..13 + reset at 14).
  static constexpr size_t kQnnGruInputCount = 15;
};

Ort::Status StatefulGruOpBuilder::ValidateAdditionalSupport(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);

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

  const QnnBackendType backend_type = qnn_model_wrapper.GetQnnBackendType();
  RETURN_IF_NOT(IsNpuBackend(backend_type) || IsIrBackend(backend_type),
                "QNN EP: StatefulGru requires the multi-time-step GRU path to preserve state across inferences; "
                "the selected backend does not support the 3D reset input.");

  if (node_unit.Inputs().size() > kQtiAiswStatefulGruResetInputIndex &&
      node_unit.Inputs()[kQtiAiswStatefulGruResetInputIndex].Exists()) {
    TensorInfo reset_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[kQtiAiswStatefulGruResetInputIndex], reset_info));
    RETURN_IF_NOT(reset_info.qnn_data_type == QNN_DATATYPE_BOOL_8,
                  "QNN EP: StatefulGru reset input must have QNN BOOL_8 data type.");
    RETURN_IF_NOT(reset_info.shape == std::vector<uint32_t>{1},
                  "QNN EP: StatefulGru reset input must be a scalar.");
  }

  return Ort::Status();
}

rnn_utils::ResetInput StatefulGruOpBuilder::GetResetInput(const OrtNodeUnit& node_unit,
                                                          const std::vector<std::string>& input_names) const {
  const auto& onnx_inputs = node_unit.Inputs();
  const std::string reset_name = (onnx_inputs.size() > kQtiAiswStatefulGruResetInputIndex &&
                                  onnx_inputs[kQtiAiswStatefulGruResetInputIndex].Exists())
                                     ? input_names[kQtiAiswStatefulGruResetInputIndex]
                                     : "";
  return reset_name.empty() ? rnn_utils::QnnResetInputWithSynthesizedFalse(kQnnGruResetInputIndex)
                            : rnn_utils::ResetInput{reset_name, kQnnGruResetInputIndex};
}

size_t StatefulGruOpBuilder::GetQnnGruInputCount(const rnn_utils::ResetInput& reset) const {
  ORT_UNUSED_PARAMETER(reset);
  return kQnnGruInputCount;
}

void CreateStatefulGruOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<StatefulGruOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
