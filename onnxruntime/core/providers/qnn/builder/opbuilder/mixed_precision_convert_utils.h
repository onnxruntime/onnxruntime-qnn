// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Converts across a mismatched fixed-point (4/8/16-bit) precision at a QNN op's declared
// input(s) and output by inserting a QNN Convert, for the ops in IsConvertCompatibleUnaryOp /
// IsConvertCompatibleBinaryOp / SimpleOpBuilder cases.

#pragma once

#include <string>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace utils {

// Overrides offset/scale for ops (Sigmoid/HardSigmoid/Tanh) that require a specific
// HTP-fixed encoding regardless of calibration range. Returns true if overridden.
bool OverrideActivationFixedEncoding(const std::string& op_type, Qnn_DataType_t qnn_data_type,
                                     Qnn_ScaleOffset_t& quant_params);

// Aligns a mismatched fixed-point input pair (see IsConvertCompatibleBinaryOp) by inserting
// a Convert on the narrower input, so the op computes natively at the wider precision.
// No-op (returns OK) when `node_unit`'s op is not Convert-compatible or inputs already match.
Ort::Status AlignBinaryInputPrecision(QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& node_unit,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation);

// Builds `qnn_op_type` at `native_dtype` and inserts a QNN Convert to the node's
// declared output tensor/precision.
Ort::Status InsertOutputPrecisionConvert(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         std::vector<std::string>&& input_names,
                                         std::vector<std::string>&& param_tensor_names,
                                         bool do_op_validation,
                                         const std::string& qnn_op_type,
                                         Qnn_DataType_t native_dtype,
                                         const TensorInfo& declared_output_info);

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
