// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Op builders for qti_aisw::QnnQuantize and qti_aisw::QnnDequantize — the custom ops used to
// carry QNN encoding types that cannot be expressed as standard ONNX QuantizeLinear /
// DequantizeLinear nodes (int2, LPBQ two-level, float zero-point).
//
// Encoding type values (encoding_type attribute):
//   0 → BLOCKWISE_EXPANSION (LPBQ, QNN ID 5)      — int4, two-level per-block integer scale
//   1 → BW_AXIS_SCALE_OFFSET_MAPPED (QNN ID 8)    — int2, per-channel, mapping enum
//   2 → BW_BLOCK_MAPPED (QNN ID 9)                — int2, block quantization, mapping enum
//   3 → BW_BLOCKWISE_EXPANSION_MAPPED (QNN ID 10) — int2, two-level, mapping enum
//   4 → SCALE_OFFSET / AXIS_SCALE_OFFSET at int32/uint32 — ONNX Q/DQ has no INT32 output type

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/custom_op/qnn_qti_aisw_custom_op.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Maps the string output_dtype attribute to its QNN data type.
Qnn_DataType_t OutputDtypeToQnnDataType(const std::string& dtype) {
  if (dtype == "int2")   return QNN_DATATYPE_SFIXED_POINT_2;
  if (dtype == "uint2")  return QNN_DATATYPE_UFIXED_POINT_2;
  if (dtype == "int4")   return QNN_DATATYPE_SFIXED_POINT_4;
  if (dtype == "uint4")  return QNN_DATATYPE_UFIXED_POINT_4;
  if (dtype == "int8")   return QNN_DATATYPE_SFIXED_POINT_8;
  if (dtype == "uint8")  return QNN_DATATYPE_UFIXED_POINT_8;
  if (dtype == "int16")  return QNN_DATATYPE_SFIXED_POINT_16;
  if (dtype == "uint16") return QNN_DATATYPE_UFIXED_POINT_16;
  if (dtype == "int32")  return QNN_DATATYPE_SFIXED_POINT_32;
  if (dtype == "uint32") return QNN_DATATYPE_UFIXED_POINT_32;
  return QNN_DATATYPE_UNDEFINED;
}

// Builds the right QnnQuantParamsWrapper for the given encoding_type.
// Reads encoding-specific attributes from node_helper.
Ort::Status BuildQuantParamsWrapper(const OrtNodeAttrHelper& node_helper,
                                    int64_t encoding_type,
                                    /*out*/ QnnQuantParamsWrapper& qp) {
  switch (encoding_type) {
    case 0: {
      // BLOCKWISE_EXPANSION (LPBQ)
      const std::vector<float> per_channel_float_scale =
          node_helper.Get("per_channel_float_scale", std::vector<float>{});
      const std::vector<int64_t> per_block_int_scale_raw =
          node_helper.Get("per_block_int_scale", std::vector<int64_t>{});
      const int64_t axis = node_helper.Get("axis", static_cast<int64_t>(0));

      RETURN_IF_NOT(!per_channel_float_scale.empty(), "QNN EP: QnnQuantize LPBQ missing per_channel_float_scale");
      RETURN_IF_NOT(!per_block_int_scale_raw.empty(), "QNN EP: QnnQuantize LPBQ missing per_block_int_scale");

      std::vector<uint8_t> per_block_int_scale(per_block_int_scale_raw.begin(), per_block_int_scale_raw.end());
      std::vector<int32_t> offsets(per_channel_float_scale.size(), 0);

      qp = QnnQuantParamsWrapper::LowPowerBlockwise(per_channel_float_scale, per_block_int_scale,
                                                    offsets, axis, /*block_scale_bitwidth=*/4);
      break;
    }
    case 1: {
      // BW_AXIS_SCALE_OFFSET_MAPPED (int2 per-channel)
      const std::vector<float> y_scale = node_helper.Get("y_scale", std::vector<float>{});
      const std::vector<int64_t> y_zero_point_raw =
          node_helper.Get("y_zero_point", std::vector<int64_t>{});
      const int64_t axis    = node_helper.Get("axis", static_cast<int64_t>(0));
      const int64_t mapping = node_helper.Get("mapping",
          static_cast<int64_t>(QNN_QUANTIZATION_ENCODING_MAPPING_LINEAR_SYMMETRIC_EXCLUDE_ZERO));

      RETURN_IF_NOT(!y_scale.empty(), "QNN EP: QnnQuantize BW_AXIS_SCALE_OFFSET_MAPPED missing y_scale");

      std::vector<int32_t> zp(y_zero_point_raw.begin(), y_zero_point_raw.end());
      qp = QnnQuantParamsWrapper::BwAxisScaleOffsetMapped(
          y_scale, zp, static_cast<int32_t>(axis), /*bitwidth=*/2,
          static_cast<Qnn_QuantizationEncodingMapping_t>(mapping));
      break;
    }
    case 2: {
      // BW_BLOCK_MAPPED (int2 block quantization)
      const std::vector<float> y_scale = node_helper.Get("y_scale", std::vector<float>{});
      const std::vector<int64_t> y_zero_point_raw =
          node_helper.Get("y_zero_point", std::vector<int64_t>{});
      const std::vector<int64_t> block_sizes_raw =
          node_helper.Get("block_sizes", std::vector<int64_t>{});
      const int64_t mapping = node_helper.Get("mapping",
          static_cast<int64_t>(QNN_QUANTIZATION_ENCODING_MAPPING_LINEAR_SYMMETRIC_EXCLUDE_ZERO));

      RETURN_IF_NOT(!y_scale.empty(), "QNN EP: QnnQuantize BW_BLOCK_MAPPED missing y_scale");
      RETURN_IF_NOT(!block_sizes_raw.empty(), "QNN EP: QnnQuantize BW_BLOCK_MAPPED missing block_sizes");

      std::vector<int32_t> zp(y_zero_point_raw.begin(), y_zero_point_raw.end());
      std::vector<uint32_t> block_sizes(block_sizes_raw.begin(), block_sizes_raw.end());
      qp = QnnQuantParamsWrapper::BwBlockMapped(
          y_scale, zp, block_sizes, /*bitwidth=*/2,
          static_cast<Qnn_QuantizationEncodingMapping_t>(mapping));
      break;
    }
    case 3: {
      // BW_BLOCKWISE_EXPANSION_MAPPED (int2 two-level)
      const std::vector<float> per_channel_float_scale =
          node_helper.Get("per_channel_float_scale", std::vector<float>{});
      const std::vector<int64_t> per_block_int_scale_raw =
          node_helper.Get("per_block_int_scale", std::vector<int64_t>{});
      const int64_t axis    = node_helper.Get("axis", static_cast<int64_t>(0));
      const int64_t mapping = node_helper.Get("mapping",
          static_cast<int64_t>(QNN_QUANTIZATION_ENCODING_MAPPING_LINEAR_SYMMETRIC_EXCLUDE_ZERO));

      RETURN_IF_NOT(!per_channel_float_scale.empty(),
                    "QNN EP: QnnQuantize BW_BLOCKWISE_EXPANSION_MAPPED missing per_channel_float_scale");
      RETURN_IF_NOT(!per_block_int_scale_raw.empty(),
                    "QNN EP: QnnQuantize BW_BLOCKWISE_EXPANSION_MAPPED missing per_block_int_scale");

      std::vector<uint8_t> per_block_int_scale(per_block_int_scale_raw.begin(), per_block_int_scale_raw.end());
      std::vector<int32_t> offsets(per_channel_float_scale.size(), 0);

      qp = QnnQuantParamsWrapper::BwBlockwiseExpansionMapped(
          per_channel_float_scale, per_block_int_scale, offsets,
          static_cast<int32_t>(axis), /*block_scale_bitwidth=*/4, /*bitwidth=*/2,
          static_cast<Qnn_QuantizationEncodingMapping_t>(mapping));
      break;
    }
    case 4: {
      // SCALE_OFFSET / AXIS_SCALE_OFFSET at int32/uint32
      // ONNX Q/DQ has no INT32 output type — carried via custom op.
      const std::vector<float> y_scale =
          node_helper.Get("y_scale", std::vector<float>{});
      const std::vector<int64_t> y_zero_point_raw =
          node_helper.Get("y_zero_point", std::vector<int64_t>{});
      const int64_t axis = node_helper.Get("axis", static_cast<int64_t>(-1));

      RETURN_IF_NOT(!y_scale.empty(), "QNN EP: QnnQuantize INT32 missing y_scale");

      std::vector<int32_t> zp(y_zero_point_raw.begin(), y_zero_point_raw.end());
      if (y_scale.size() > 1 && axis >= 0) {
        // Per-channel (AXIS_SCALE_OFFSET)
        if (zp.empty()) zp.assign(y_scale.size(), 0);
        qp = QnnQuantParamsWrapper::PerChannel(y_scale, zp, static_cast<int32_t>(axis));
      } else {
        // Per-tensor (SCALE_OFFSET)
        const int32_t offset = zp.empty() ? 0 : zp[0];
        qp = QnnQuantParamsWrapper::PerTensor(y_scale[0], offset);
      }
      break;
    }
    default:
      return MAKE_EP_FAIL(("QNN EP: QnnQuantize unsupported encoding_type: " +
                           std::to_string(encoding_type))
                              .c_str());
  }
  return Ort::Status();
}

}  // namespace

// ---------------------------------------------------------------------------
// QnnEncQuantizeOpBuilder
// ---------------------------------------------------------------------------

class QnnEncQuantizeOpBuilder : public BaseOpBuilder {
 public:
  QnnEncQuantizeOpBuilder() : BaseOpBuilder("QnnEncQuantizeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnEncQuantizeOpBuilder);

 protected:
  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status QnnEncQuantizeOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                   const OrtNodeUnit& node_unit,
                                                   const Ort::Logger& /*logger*/) const {
  const auto& inputs  = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  RETURN_IF_NOT(inputs.size() >= 1 && inputs[0].Exists(),
                "QNN EP: QnnQuantize requires one float input.");
  RETURN_IF_NOT(outputs.size() >= 1 && outputs[0].Exists(),
                "QNN EP: QnnQuantize requires one output.");

  OrtNodeAttrHelper node_helper(node_unit);
  RETURN_IF_NOT(node_helper.HasAttr("encoding_type"),
                "QNN EP: QnnQuantize requires 'encoding_type' attribute.");
  RETURN_IF_NOT(node_helper.HasAttr("output_dtype"),
                "QNN EP: QnnQuantize requires 'output_dtype' attribute.");
  const std::string output_dtype = node_helper.Get("output_dtype", std::string("int8"));
  RETURN_IF_NOT(OutputDtypeToQnnDataType(output_dtype) != QNN_DATATYPE_UNDEFINED,
                ("QNN EP: QnnQuantize unrecognized output_dtype: " + output_dtype).c_str());
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  return Ort::Status();
}

Ort::Status QnnEncQuantizeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                  const OrtNodeUnit& node_unit,
                                                                  std::vector<std::string>&& input_names,
                                                                  const Ort::Logger& /*logger*/,
                                                                  bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t encoding_type = node_helper.Get("encoding_type", static_cast<int64_t>(-1));
  const std::string output_dtype = node_helper.Get("output_dtype", std::string("int8"));

  const Qnn_DataType_t qnn_data_type = OutputDtypeToQnnDataType(output_dtype);
  RETURN_IF_NOT(qnn_data_type != QNN_DATATYPE_UNDEFINED,
                ("QNN EP: QnnQuantize unrecognized output_dtype: " + output_dtype).c_str());

  QnnQuantParamsWrapper qp;
  RETURN_IF_ERROR(BuildQuantParamsWrapper(node_helper, encoding_type, qp));

  // Derive output tensor shape from the float input.
  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  const std::string& output_name = node_unit.Outputs()[0].name;
  RETURN_IF_NOT(input_names.size() == 1, "QNN EP: QnnQuantize expects exactly one input name.");
  const std::string& input_name = input_names[0];

  RETURN_IF_ERROR(qnn_model_wrapper.AddQuantizeNode(
      input_name, output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type,
      std::move(qp), std::vector<uint32_t>(input_info.shape), do_op_validation));
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// QnnEncDequantizeOpBuilder
// ---------------------------------------------------------------------------

class QnnEncDequantizeOpBuilder : public BaseOpBuilder {
 public:
  QnnEncDequantizeOpBuilder() : BaseOpBuilder("QnnEncDequantizeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnEncDequantizeOpBuilder);

 protected:
  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status QnnEncDequantizeOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                     const OrtNodeUnit& node_unit,
                                                     const Ort::Logger& /*logger*/) const {
  const auto& inputs  = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  RETURN_IF_NOT(inputs.size() >= 1 && inputs[0].Exists(),
                "QNN EP: QnnDequantize requires one quantized input.");
  RETURN_IF_NOT(outputs.size() >= 1 && outputs[0].Exists(),
                "QNN EP: QnnDequantize requires one float output.");

  // encoding_type and output_dtype are required — same attrs as QnnQuantize.
  OrtNodeAttrHelper node_helper(node_unit);
  RETURN_IF_NOT(node_helper.HasAttr("encoding_type"),
                "QNN EP: QnnDequantize requires 'encoding_type' attribute.");
  RETURN_IF_NOT(node_helper.HasAttr("output_dtype"),
                "QNN EP: QnnDequantize requires 'output_dtype' attribute.");
  const std::string output_dtype = node_helper.Get("output_dtype", std::string("int8"));
  RETURN_IF_NOT(OutputDtypeToQnnDataType(output_dtype) != QNN_DATATYPE_UNDEFINED,
                ("QNN EP: QnnDequantize unrecognized output_dtype: " + output_dtype).c_str());
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  return Ort::Status();
}

Ort::Status QnnEncDequantizeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                    const OrtNodeUnit& node_unit,
                                                                    std::vector<std::string>&& input_names,
                                                                    const Ort::Logger& /*logger*/,
                                                                    bool do_op_validation) const {
  RETURN_IF_NOT(input_names.size() == 1, "QNN EP: QnnDequantize expects exactly one input name.");
  const std::string& input_name = input_names[0];
  const std::string& output_name = node_unit.Outputs()[0].name;

  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t encoding_type = node_helper.Get("encoding_type", static_cast<int64_t>(-1));
  const std::string output_dtype = node_helper.Get("output_dtype", std::string("int8"));
  const Qnn_DataType_t qnn_data_type = OutputDtypeToQnnDataType(output_dtype);

  // Build quant params from encoding attrs so the QNN DQ node's input tensor has an
  // explicit encoding — identical to the encoding set by the upstream QnnQuantize builder.
  QnnQuantParamsWrapper qp;
  RETURN_IF_ERROR(BuildQuantParamsWrapper(node_helper, encoding_type, qp));

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  // Register the quantized input tensor with explicit encoding if it isn't already present
  // (the upstream QnnQuantize builder will have registered it; AddTensorWrapper is a no-op
  // for already-registered tensors, so this is safe to call unconditionally).
  QnnTensorWrapper input_wrapper(input_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type,
                                 std::move(qp), std::vector<uint32_t>(input_info.shape));
  qnn_model_wrapper.AddTensorWrapper(std::move(input_wrapper));

  RETURN_IF_ERROR(qnn_model_wrapper.AddDequantizeNode(
      input_name, output_name, QNN_DATATYPE_FLOAT_32,
      std::vector<uint32_t>(input_info.shape), do_op_validation));
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Registration (custom ops)
// ---------------------------------------------------------------------------

void CreateQnnEncQuantizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QnnEncQuantizeOpBuilder>());
}

void CreateQnnEncDequantizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QnnEncDequantizeOpBuilder>());
}

// ---------------------------------------------------------------------------
// OnnxQuantizeLinearOpBuilder
// Handles standard ONNX QuantizeLinear as a standalone op (disable_qdq_fusion=1).
// QNN_OP_QUANTIZE takes exactly 1 input; scale/zp are encoding metadata on the output tensor.
// ---------------------------------------------------------------------------

class OnnxQuantizeLinearOpBuilder : public BaseOpBuilder {
 public:
  OnnxQuantizeLinearOpBuilder() : BaseOpBuilder("OnnxQuantizeLinearOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxQuantizeLinearOpBuilder);

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
};

Ort::Status OnnxQuantizeLinearOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& /*logger*/) const {
  const auto& inputs  = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  // For SingleNode QuantizeLinear units, OrtNodeUnit::Inputs() returns only 1 element (x).
  // Scale/zp are folded into outputs[0].quant_param, not exposed as separate inputs[1]/inputs[2].
  RETURN_IF_NOT(inputs.size() >= 1 && inputs[0].Exists(),
                "QNN EP: QuantizeLinear requires x input.");
  RETURN_IF_NOT(outputs.size() >= 1 && outputs[0].Exists(),
                "QNN EP: QuantizeLinear requires one output.");
  RETURN_IF_NOT(outputs[0].quant_param.has_value(),
                "QNN EP: QuantizeLinear requires quantization params on output.");
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  return Ort::Status();
}

Ort::Status OnnxQuantizeLinearOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& logger,
                                                       std::vector<std::string>& input_names,
                                                       bool /*do_op_validation*/) const {
  // Only pass x (inputs[0]) to QNN_OP_QUANTIZE; scale/zp become encoding metadata.
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, node_unit.Inputs()[0], logger, input_names));
  return Ort::Status();
}

Ort::Status OnnxQuantizeLinearOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                      const OrtNodeUnit& node_unit,
                                                                      std::vector<std::string>&& input_names,
                                                                      const Ort::Logger& /*logger*/,
                                                                      bool do_op_validation) const {
  const auto& inputs  = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // Scale/zp are folded into outputs[0].quant_param for SingleNode QuantizeLinear units.
  QnnQuantParamsWrapper qp;
  RETURN_IF_ERROR(qp.Init(qnn_model_wrapper, outputs[0]));

  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape),
                "QNN EP: QuantizeLinear cannot get input shape.");

  RETURN_IF_NOT(input_names.size() == 1, "QNN EP: QuantizeLinear expects exactly one input name.");
  RETURN_IF_ERROR(qnn_model_wrapper.AddQuantizeNode(
      input_names[0], outputs[0].name,
      qnn_model_wrapper.GetTensorType(outputs[0].name),
      output_info.qnn_data_type,
      std::move(qp), std::move(input_shape), do_op_validation));
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// OnnxDequantizeLinearOpBuilder
// Handles standard ONNX DequantizeLinear as a standalone op (disable_qdq_fusion=1).
// QNN_OP_DEQUANTIZE takes 1 quantized input (encoding on the input tensor) → float output.
// ProcessInputs is NOT overridden: base class ProcessInput -> MakeTensorWrapper correctly
// registers the quantized input with encoding from inputs[0].quant_param.
// ---------------------------------------------------------------------------

class OnnxDequantizeLinearOpBuilder : public BaseOpBuilder {
 public:
  OnnxDequantizeLinearOpBuilder() : BaseOpBuilder("OnnxDequantizeLinearOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxDequantizeLinearOpBuilder);

 protected:
  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status OnnxDequantizeLinearOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         const Ort::Logger& /*logger*/) const {
  const auto& inputs  = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  // For SingleNode DequantizeLinear units, OrtNodeUnit::Inputs() returns only 1 element (x).
  // Scale/zp are folded into inputs[0].quant_param, not exposed as separate inputs[1]/inputs[2].
  RETURN_IF_NOT(inputs.size() >= 1 && inputs[0].Exists(),
                "QNN EP: DequantizeLinear requires x input.");
  RETURN_IF_NOT(outputs.size() >= 1 && outputs[0].Exists(),
                "QNN EP: DequantizeLinear requires one float output.");
  RETURN_IF_NOT(inputs[0].quant_param.has_value(),
                "QNN EP: DequantizeLinear requires quantization params on input.");
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  return Ort::Status();
}

Ort::Status OnnxDequantizeLinearOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                        const OrtNodeUnit& node_unit,
                                                                        std::vector<std::string>&& input_names,
                                                                        const Ort::Logger& /*logger*/,
                                                                        bool do_op_validation) const {
  const auto& outputs = node_unit.Outputs();
  RETURN_IF_NOT(input_names.size() == 1, "QNN EP: DequantizeLinear expects exactly one input name.");

  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].shape, output_shape),
                "QNN EP: DequantizeLinear cannot get output shape.");

  // Pre-register the output tensor with the correct type (APP_READ for graph outputs).
  // AddDequantizeNode hardcodes QNN_TENSOR_TYPE_NATIVE; pre-registering ensures the right
  // type is used when the output is a model graph output. AddTensorWrapper is a no-op if
  // the tensor is already registered.
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(outputs[0].name)) {
    QnnTensorWrapper out_wrapper(outputs[0].name,
                                 qnn_model_wrapper.GetTensorType(outputs[0].name),
                                 QNN_DATATYPE_FLOAT_32,
                                 QnnQuantParamsWrapper(),
                                 std::vector<uint32_t>(output_shape));
    qnn_model_wrapper.AddTensorWrapper(std::move(out_wrapper));
  }

  RETURN_IF_ERROR(qnn_model_wrapper.AddDequantizeNode(
      input_names[0], outputs[0].name, QNN_DATATYPE_FLOAT_32,
      std::move(output_shape), do_op_validation));
  return Ort::Status();
}

// ---------------------------------------------------------------------------
// Registration (standard ONNX ops)
// ---------------------------------------------------------------------------

void CreateOnnxQuantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<OnnxQuantizeLinearOpBuilder>());
}

void CreateOnnxDequantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<OnnxDequantizeLinearOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
