// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/rnn_op_builder_base.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/custom_op/qnn_qti_aisw_custom_op.h"

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
// vector of size 25) is shared with the standard LSTM builder via LstmBaseOpBuilder and
// rnn_utils::AddUnidirectionLSTM. This builder passes a reset ResetInput and materializes the
// BlockOp false default at QNN input 24 when ONNX reset is omitted; the standard LSTM passes
// NoQnnResetInput().
//
// Bidirectional is supported for float Lstm via a forward + reverse unroll joined by Concat,
// matching HtpOpDefSupplement, which places no direction constraint on the Lstm op. Quantized
// StatefulLstm models are not selected as native QDQ groups today; they run as DQ -> fp Lstm -> Q.
class StatefulLstmOpBuilder : public LstmBaseOpBuilder {
 public:
  StatefulLstmOpBuilder() : LstmBaseOpBuilder("StatefulLstmOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StatefulLstmOpBuilder);

 protected:
  const char* OpDisplayName() const override { return "StatefulLstm"; }
  size_t MaxOnnxInputCount() const override { return 9; }

  Ort::Status ValidateAdditionalSupport(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  rnn_utils::ResetInput GetResetInput(const OrtNodeUnit& node_unit,
                                      const std::vector<std::string>& input_names) const override;

 private:
  // QNN LSTM input slot for the reset signal (per QAIRT MasterOpDef Lstm in[24]).
  static constexpr size_t kQnnLstmResetInputIndex = 24;
};

Ort::Status StatefulLstmOpBuilder::ValidateAdditionalSupport(QnnModelWrapper& qnn_model_wrapper,
                                                             const OrtNodeUnit& node_unit,
                                                             const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t input_forget = node_helper.Get("input_forget", static_cast<int64_t>(0));
  RETURN_IF(input_forget != 0,
            "QNN EP doesn't support input_forget=1 for StatefulLstm.");

  const QnnBackendType backend_type = qnn_model_wrapper.GetQnnBackendType();
  RETURN_IF_NOT(IsNpuBackend(backend_type) || IsIrBackend(backend_type),
                "QNN EP: StatefulLstm requires the multi-time-step LSTM path to preserve state across inferences; "
                "the selected backend does not support the 3D reset input.");

  if (node_unit.Inputs().size() > kQtiAiswStatefulLstmResetInputIndex &&
      node_unit.Inputs()[kQtiAiswStatefulLstmResetInputIndex].Exists()) {
    TensorInfo reset_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[kQtiAiswStatefulLstmResetInputIndex], reset_info));
    RETURN_IF_NOT(reset_info.qnn_data_type == QNN_DATATYPE_BOOL_8,
                  "QNN EP: StatefulLstm reset input must have QNN BOOL_8 data type.");
    RETURN_IF_NOT(reset_info.shape == std::vector<uint32_t>{1},
                  "QNN EP: StatefulLstm reset input must be a scalar.");
  }

  // No direction restriction for fp Lstm: HtpOpDefSupplement places no direction constraint on
  // the Lstm op, and this builder implements bidirectional codegen (forward + reverse unroll
  // joined by Concat in LstmBaseOpBuilder). Quantized StatefulLstm is not selected as a native QDQ
  // group today.
  return Ort::Status();
}

rnn_utils::ResetInput StatefulLstmOpBuilder::GetResetInput(const OrtNodeUnit& node_unit,
                                                           const std::vector<std::string>& input_names) const {
  const auto& onnx_inputs = node_unit.Inputs();
  const std::string reset_name = (onnx_inputs.size() > kQtiAiswStatefulLstmResetInputIndex &&
                                  onnx_inputs[kQtiAiswStatefulLstmResetInputIndex].Exists())
                                     ? input_names[kQtiAiswStatefulLstmResetInputIndex]
                                     : "";
  return reset_name.empty() ? rnn_utils::QnnResetInputWithSynthesizedFalse(kQnnLstmResetInputIndex)
                            : rnn_utils::ResetInput{reset_name, kQnnLstmResetInputIndex};
}

void CreateStatefulLstmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<StatefulLstmOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
