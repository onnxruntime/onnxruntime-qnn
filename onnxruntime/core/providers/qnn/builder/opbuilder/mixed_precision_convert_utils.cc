// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/opbuilder/mixed_precision_convert_utils.h"

#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace utils {

bool OverrideActivationFixedEncoding(const std::string& op_type, Qnn_DataType_t qnn_data_type,
                                     Qnn_ScaleOffset_t& quant_params) {
  const int32_t orig_offset = quant_params.offset;
  const float orig_scale = quant_params.scale;

  if (op_type == "Sigmoid" || op_type == "HardSigmoid") {
    switch (qnn_data_type) {
      case QNN_DATATYPE_UFIXED_POINT_16:
        quant_params.offset = 0;
        quant_params.scale = 1.0f / 65536.0f;
        break;
      case QNN_DATATYPE_SFIXED_POINT_16:
        quant_params.offset = 0;
        quant_params.scale = 1.0f / 32768.0f;
        break;
      default:
        break;  // Do nothing.
    }
  }

  if (op_type == "Tanh") {
    switch (qnn_data_type) {
      case QNN_DATATYPE_UFIXED_POINT_16:
        quant_params.offset = -32768;
        quant_params.scale = 1.0f / 32768.0f;
        break;
      case QNN_DATATYPE_SFIXED_POINT_16:
        quant_params.offset = 0;
        quant_params.scale = 1.0f / 32768.0f;
        break;
      default:
        break;  // Do nothing.
    }
  }

  return quant_params.offset != orig_offset || quant_params.scale != orig_scale;
}

Ort::Status AlignBinaryInputPrecision(QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& node_unit,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) {
  // 1. Only applies to Convert-compatible binary ops with exactly 2 inputs.
  if (!IsConvertCompatibleBinaryOp(node_unit.OpType()) || input_names.size() != 2) {
    return Ort::Status();
  }

  // 2. Same-precision inputs need no Convert. Same-width but differing signedness (e.g.
  // Mul(s8, u8)) also needs no Convert: it isn't a width mismatch, and QNN HTP validates
  // mixed-signedness input pairs natively for these ops, so leave such inputs untouched and let
  // op validation be the arbiter.
  const Qnn_DataType_t dt0 = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]).GetTensorDataType();
  const Qnn_DataType_t dt1 = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]).GetTensorDataType();
  if (dt0 == dt1 || FixedPointBitWidth(dt0) == FixedPointBitWidth(dt1)) {
    return Ort::Status();
  }
  RETURN_IF_NOT(NeedsPrecisionConvert(dt0, dt1),
                "QNN EP's binary-op precision Convert only supports a mismatched fixed-point input pair.");

  // 3. Convert the narrower input up to the wider input's precision.
  const size_t narrow_idx = FixedPointBitWidth(dt0) < FixedPointBitWidth(dt1) ? 0 : 1;
  const Qnn_DataType_t wide_dtype = narrow_idx == 0 ? dt1 : dt0;
  const auto& narrow_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[narrow_idx]);
  RETURN_IF_NOT(narrow_wrapper.GetQnnQuantParams().IsPerTensor(),
                "QNN EP's binary-op precision Convert only supports per-tensor quantization");
  const Qnn_QuantizeParams_t& narrow_qp = narrow_wrapper.GetQnnQuantParams().Get();
  const std::string converted_name = UniqueNameGenerator().New(input_names[narrow_idx], "_convert");
  RETURN_IF_ERROR(InsertConvertOp(qnn_model_wrapper, input_names[narrow_idx], converted_name,
                                  narrow_wrapper.GetTensorDataType(), wide_dtype,
                                  narrow_qp.scaleOffsetEncoding.offset, narrow_qp.scaleOffsetEncoding.scale,
                                  narrow_wrapper.GetTensorDims(), /*output_symmetric*/ false, do_op_validation));
  input_names[narrow_idx] = converted_name;
  return Ort::Status();
}

Ort::Status InsertOutputPrecisionConvert(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         std::vector<std::string>&& input_names,
                                         std::vector<std::string>&& param_tensor_names,
                                         bool do_op_validation,
                                         const std::string& qnn_op_type,
                                         Qnn_DataType_t native_dtype,
                                         const TensorInfo& declared_output_info) {
  const std::string& op_type = node_unit.OpType();
  const std::string& declared_output_name = node_unit.Outputs()[0].name;

  // 1. Guard the two cases this Convert doesn't handle.
  RETURN_IF_NOT(!qnn_model_wrapper.IsGraphOutput(declared_output_name),
                ("QNN EP's precision Convert does not support a graph-output result for " + op_type).c_str());
  RETURN_IF_NOT(declared_output_info.quant_param.IsPerTensor(),
                ("QNN EP's precision Convert only supports per-tensor quantization for " + op_type).c_str());

  // 2. Derive a native-precision encoding, then let any op-specific override (e.g. Sigmoid/Tanh) apply.
  const Qnn_ScaleOffset_t& declared_so = declared_output_info.quant_param.Get().scaleOffsetEncoding;
  Qnn_ScaleOffset_t native_so{};
  RETURN_IF_ERROR(DeriveScaleOffsetForDtype(declared_output_info.qnn_data_type, declared_so.offset, declared_so.scale,
                                            native_dtype, /*to_symmetric*/ false, native_so.scale, native_so.offset));
  OverrideActivationFixedEncoding(op_type, native_dtype, native_so);

  // 3. Build the op with a native-precision intermediate output tensor.
  const std::string native_output_name = UniqueNameGenerator().New(declared_output_name, "_native");
  QnnTensorWrapper native_output_tensorwrapper(native_output_name, QNN_TENSOR_TYPE_NATIVE, native_dtype,
                                               QnnQuantParamsWrapper::PerTensor(native_so.scale, native_so.offset),
                                               std::vector<uint32_t>(declared_output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(native_output_tensorwrapper)),
                "Failed to add native-precision output tensor.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                qnn_op_type,
                                                std::move(input_names),
                                                {native_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to add node.");

  // 4. Convert the native-precision result to the node's declared output tensor/precision.
  return InsertConvertOp(qnn_model_wrapper, native_output_name, declared_output_name,
                         native_dtype, declared_output_info.qnn_data_type,
                         native_so.offset, native_so.scale,
                         declared_output_info.shape, /*output_symmetric*/ false, do_op_validation);
}

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
