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
namespace rnn_details {

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

// Parameterizes the optional stateful reset input for AddUnidirectionGRU/LSTM.
// Pass kNoReset() for the standard GRU/LSTM (no stateful reset slot).
struct ResetInput {
  std::string onnx_name;  // empty = no reset
  size_t qnn_slot{0};     // QNN input slot index (ignored when onnx_name is empty)
};
inline ResetInput kNoReset() { return {"", 0}; }

// Performs the ONNX->QNN unidirectional GRU decomposition: time-step unroll, W/R/B gate slicing,
// zero-bias/initial-state stubs, per-step hidden-state chaining. Used by GRUOpBuilder (base) and
// StatefulGruOpBuilder (adds reset slot).
//
// qnn_input_count: size of the QNN GRU input vector. Pass 14 for the standard GRU; 15 for
//   StatefulGru (which occupies slot 14 for the reset signal).
// reset: optional stateful reset input; pass kNoReset() for the standard GRU.
Ort::Status AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit,
                               const std::string& direction,
                               const std::vector<std::string>& input_names,
                               const Ort::Logger& logger,
                               bool do_op_validation,
                               bool is_bidirection,
                               size_t qnn_input_count,
                               const ResetInput& reset,
                               std::vector<std::string>& uni_gru_output_names) ORT_MUST_USE_RESULT;

// Performs the ONNX->QNN unidirectional LSTM decomposition: StridedSlice gate slicing, ifoc gate
// reordering, bias summation, zero-bias/initial-state stubs, bidirectional Concat. Used by
// LSTMOpBuilder (base) and StatefulLstmOpBuilder (adds reset slot).
//
// reset: optional stateful reset input; pass kNoReset() for the standard LSTM.
//   The QNN LSTM input vector is always size 25; the reset occupies slot 24 (kQnnLstmResetInputIndex).
Ort::Status AddUnidirectionLSTM(QnnModelWrapper& qnn_model_wrapper,
                                const OrtNodeUnit& node_unit,
                                const std::string& direction,
                                const std::vector<std::string>& input_names,
                                const Ort::Logger& logger,
                                bool do_op_validation,
                                bool is_bidirection,
                                const ResetInput& reset,
                                std::vector<std::string>& uni_lstm_output_names) ORT_MUST_USE_RESULT;

}  // namespace rnn_details
}  // namespace qnn
}  // namespace onnxruntime
