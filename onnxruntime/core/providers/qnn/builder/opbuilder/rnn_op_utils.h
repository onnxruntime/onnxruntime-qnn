// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "QnnOpDef.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
namespace rnn_utils {

// Inserts a StridedSlice QNN node to extract a slice from input_name into output_name.
// Used by GRU and StatefulGru builders.
Ort::Status AddStridedSlice(QnnModelWrapper& qnn_model_wrapper,
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
                            bool is_for_output) ORT_MUST_USE_RESULT;

// Inserts either a Reshape (when the leading dim is 1 and trailing dims match) or a StridedSlice.
// Used by LSTM and StatefulLstm builders.
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
                                     bool is_for_output) ORT_MUST_USE_RESULT;

// Concat axis for the bidirectional forward/reverse merge along the num_directions dimension.
// ONNX RNN outputs are Y [seq, num_dir, batch, hidden] (rank 4) and Y_h/Y_c [num_dir, batch,
// hidden] (rank 3), so the axis is `rank - 3`. Errors out with a clear message when
// `shape.size() < 3` instead of letting size_t subtraction wrap into a huge invalid axis.
Ort::Status DeriveNumDirectionsConcatAxis(const std::vector<uint32_t>& shape, uint32_t& axis) ORT_MUST_USE_RESULT;

// Returns true when a QDQ GRU/StatefulGru group must be lowered as fp32 on QNN instead of using
// a native quantized QNN Gru. The reset input of StatefulGru is BOOL_8 and is deliberately not
// part of the quantization eligibility check.
bool ShouldFpDegradeQdqGru(gsl::span<const TensorInfo> input_infos,
                           gsl::span<const OrtNodeUnitIODef> inputs,
                           gsl::span<const OrtNodeUnitIODef> outputs,
                           const std::string& direction,
                           int64_t linear_before_reset);

// Parameterizes the stateful reset input for AddUnidirectionGRU/LSTM.
// Standard GRU/LSTM have no QNN reset input. Stateful BlockOps use either their ONNX reset input
// or a synthesized false tensor. This is required because an omitted BlockOp reset means false,
// whereas an omitted QNN GRU/LSTM reset input defaults to true.
struct ResetInput {
  std::string onnx_name;  // Empty when the BlockOp reset input is omitted.
  size_t qnn_slot{0};
  bool materialize_false_reset_input{false};

  bool RequiresQnnResetInput() const { return !onnx_name.empty() || materialize_false_reset_input; }
};
// Standard GRU/LSTM do not expose a reset input, so their QNN node has no reset slot.
inline ResetInput NoQnnResetInput() { return {"", 0, false}; }

// Stateful BlockOps with an omitted reset still require a QNN reset slot carrying false.
inline ResetInput QnnResetInputWithSynthesizedFalse(size_t qnn_slot) { return {"", qnn_slot, true}; }

// Performs the ONNX->QNN unidirectional GRU lowering: standard GRU is decomposed into time-step
// cells, while StatefulGru uses one native multi-time-step QNN GRU so QNN can retain state across
// inference calls. Used by GRUOpBuilder (base) and StatefulGruOpBuilder (adds reset slot).
//
// qnn_input_count: size of the QNN GRU input vector. Pass 14 for standard GRU; StatefulGru always
//   passes 15 so QNN input 14 can carry either the ONNX reset input or the BlockOp false default.
// reset: stateful reset input; pass NoQnnResetInput() for the standard GRU.
Ort::Status AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit,
                               const std::string& direction,
                               const std::vector<std::string>& input_names,
                               const Ort::Logger& logger,
                               bool do_op_validation,
                               bool is_bidirection,
                               bool use_fp_fallback,
                               size_t qnn_input_count,
                               const ResetInput& reset,
                               std::vector<std::string>& uni_gru_output_names) ORT_MUST_USE_RESULT;

// Performs the ONNX->QNN unidirectional LSTM decomposition: StridedSlice gate slicing, ifoc gate
// reordering, bias summation, zero-bias/initial-state stubs, bidirectional Concat. Used by
// LSTMOpBuilder (base) and StatefulLstmOpBuilder (adds reset slot).
//
// reset: stateful reset input; pass NoQnnResetInput() for the standard LSTM. StatefulLstm uses the
// 25-slot QNN vector and occupies slot 24 with its ONNX reset input or a false default tensor.
Ort::Status AddUnidirectionLSTM(QnnModelWrapper& qnn_model_wrapper,
                                const OrtNodeUnit& node_unit,
                                const std::string& direction,
                                const std::vector<std::string>& input_names,
                                const Ort::Logger& logger,
                                bool do_op_validation,
                                bool is_bidirection,
                                const ResetInput& reset,
                                std::vector<std::string>& uni_lstm_output_names) ORT_MUST_USE_RESULT;

}  // namespace rnn_utils
}  // namespace qnn
}  // namespace onnxruntime
