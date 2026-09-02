// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <gsl/gsl>

#include "HTP/QnnHtpDeviceConfigShared.h"
#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

// Shared helpers for block-quantized (BQ) weight handling in the QNN HTP
// BW_FLOAT_BLOCK op-builder paths (Conv, MatMul, Gemm).
// These were previously duplicated verbatim across the individual op builders.
namespace bq {

// Returns BQ weight bitwidth (2/4/8) from an ONNX element data type, or 0 if unsupported.
uint32_t GetBQBitwidth(ONNXTensorElementDataType onnx_type);

// Returns true for the unsigned BQ weight element types (UINT2/UINT4/UINT8), which must be
// shifted to the signed domain before being registered as SFIXED_POINT_8.
bool IsUnsignedBQType(ONNXTensorElementDataType onnx_type);

// Validates a BQ weight bitwidth and block_size against the HTP constraints above.
// `op_tag` is used in the error messages (e.g. "Conv", "MatMul", "Gemm").
Ort::Status ValidateBQBitwidthAndBlockSize(uint32_t bitwidth, int64_t block_size, std::string_view op_tag);

// Resolves the BQ block_size along the contraction axis. The block_size = contraction_dim /
// num_blocks (the value the QNN BW_FLOAT_BLOCK encoding needs) is always derivable from the
// scale shape. Since PR307 the DQ/Q "block_size" attribute is also surfaced on
// quant_param->block_size; when present it is validated to equal the derived value (a malformed
// model where they disagree is rejected) and otherwise ignored. `op_tag` names the op in the
// error message. Returns the resolved block_size via `block_size`.
Ort::Status ResolveBlockSize(const OrtNodeUnitIODef& weight, int64_t contraction_dim,
                             int64_t num_blocks, std::string_view op_tag,
                             /*out*/ int64_t& block_size);

// Returns true if `scale_shape` describes a block-quantized scale for `weight_shape`
// blocked along `block_axis`: the scale dim is positive, strictly smaller than the
// weight dim, and divides it evenly. Caller is responsible for the op-specific rank and
// leading-dim checks.
bool IsBQScale(gsl::span<const int64_t> scale_shape,
               gsl::span<const uint32_t> weight_shape,
               size_t block_axis);

// Computes per-block float offsets from optional ONNX zero-points, matching the shared
// BQ convention: offset = unsigned_bias - onnx_zp, where unsigned_bias = (1 << (bits-1))
// for unsigned weights (compensating the unsigned→signed shift) and 0 for signed weights.
// `zero_point` may be null (all offsets = unsigned_bias). `count` is the expected element
// count (e.g. OC*num_blocks or num_blocks*N); validated against the unpacked zero-points.
Ort::Status ComputeBQOffsets(const QnnModelWrapper& qnn_model_wrapper,
                             const OrtValueInfo* zero_point,
                             bool is_unsigned_weight,
                             uint32_t bitwidth,
                             int64_t count,
                             /*out*/ std::vector<float>& offsets);

// Inserts a QNN_OP_DEQUANTIZE node in front of the BQ activation input.
// The BW_FLOAT_BLOCK kernels compute in FP16, so the (only expected) INT16 activation must be
// dequantized first. Verifies `act_name`'s QNN dtype is SFIXED/UFIXED_POINT_16, registers the
// FP16 output tensor at `fp16_name`, and adds the Dequantize node.
// The caller derives `fp16_name` (typically by reusing the original DequantizeLinear output name).
// `op_tag` is used in error messages.
Ort::Status AddInt16ToFp16DequantForActivation(QnnModelWrapper& qnn_model_wrapper,
                                               const std::string& act_name,
                                               const std::string& fp16_name,
                                               bool do_op_validation,
                                               std::string_view op_tag);

// Adds the FP16→INT16 output tail for a BW_FLOAT_BLOCK op. Registers the INT16 output tensor
// and adds a QNN_OP_QUANTIZE node (fp16_out_name → int16_out_name).
// IMPORTANT: The FP16 tensor (`fp16_out_name`) must be registered and the main BQ op node that
// produces it must be created BEFORE calling this function. QNN graph composition requires a
// tensor's producer node to exist before any consumer node is added.
Ort::Status AddFp16ToInt16QuantizeOutput(QnnModelWrapper& qnn_model_wrapper,
                                         const std::string& fp16_out_name,
                                         const std::string& int16_out_name,
                                         Qnn_TensorType_t int16_tensor_type,
                                         Qnn_DataType_t int16_qnn_data_type,
                                         QnnQuantParamsWrapper int16_quant_param,
                                         std::vector<uint32_t> output_shape,
                                         bool do_op_validation);

// Determine whether given BQ parameters are natively supported by HTP. If true, activation data type can be kept in
// fixed point; otherwise, insert Dequantize and Quantize nodes around to make activation in FP16.
// Note: HTP native BQ requires HTP arch >= 81.
bool IsHTPSupportedNativeBQ(QnnHtpDevice_Arch_t htp_arch,
                            Qnn_DataType_t act_data_type,
                            uint32_t bitwidth,
                            uint32_t block_size,
                            uint32_t output_channel,
                            gsl::span<const float> offsets);

}  // namespace bq
}  // namespace qnn
}  // namespace onnxruntime
