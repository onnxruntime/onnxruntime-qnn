// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Builder for the qti_aisw custom "Buffer" block op, which maps 1:1 to QNN_OP_BUFFER ("Buffer").
//
// QNN Buffer (QnnOpDef.h):
//   in[0]  activation (rank N)
//   in[1]  reset (BOOL_8, 0D scalar, optional; default 0) -- resets the internal sliding-window state
//   out[0] same shape as in[0] except dim[buffer_dim] == buffer_size
//   params: buffer_size (u32, mandatory), buffer_dim (u32, mandatory), stride (u32, default 1),
//           mode (u32, default 0), buffer_padding (u32, default 0)
//   mode values: BLOCKING=0, NON_BLOCKING_LEFT=1, NON_BLOCKING_RIGHT=2
//
// The ONNX Buffer node attributes are passed straight through to the QNN op, so the ONNX
// attribute names match the QNN param names verbatim.
class BufferOpBuilder : public BaseOpBuilder {
 public:
  BufferOpBuilder() : BaseOpBuilder("BufferOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BufferOpBuilder);

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

 protected:
  // Buffer's output shares the input's quantization encodings (see definition).
  Ort::Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const Ort::Logger& logger,
                                       const std::vector<std::string>& input_names,
                                       size_t output_index,
                                       Qnn_DataType_t qnn_data_type,
                                       QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;
};

Ort::Status BufferOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  OrtNodeAttrHelper node_helper(node_unit);

  // HTP constraint: mode 0 (BLOCKING) is not supported on HTP; only NON_BLOCKING_LEFT (1) and
  // NON_BLOCKING_RIGHT (2) are accepted.
  const int64_t mode = node_helper.Get("mode", static_cast<int64_t>(QNN_OP_BUFFER_MODE_BLOCKING));
  RETURN_IF(mode == QNN_OP_BUFFER_MODE_BLOCKING,
            "QNN EP: Buffer mode 0 (BLOCKING) is not supported on HTP. Use mode 1 or 2.");
  RETURN_IF_NOT(mode == QNN_OP_BUFFER_MODE_NON_BLOCKING_LEFT || mode == QNN_OP_BUFFER_MODE_NON_BLOCKING_RIGHT,
                "QNN EP: Buffer mode must be 1 (NON_BLOCKING_LEFT) or 2 (NON_BLOCKING_RIGHT).");

  RETURN_IF_NOT(node_helper.HasAttr("buffer_size"), "QNN EP: Buffer requires the mandatory 'buffer_size' attribute.");
  RETURN_IF_NOT(node_helper.Get("buffer_size", static_cast<int64_t>(0)) > 0,
                "QNN EP: Buffer 'buffer_size' must be greater than 0.");
  RETURN_IF_NOT(node_helper.HasAttr("buffer_dim"), "QNN EP: Buffer requires the mandatory 'buffer_dim' attribute.");

  // buffer_dim must be a valid axis of the input tensor.
  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "QNN EP: Cannot get Buffer input shape.");
  const int64_t buffer_dim = node_helper.Get("buffer_dim", static_cast<int64_t>(0));
  RETURN_IF_NOT(buffer_dim >= 0 && buffer_dim < static_cast<int64_t>(input_shape.size()),
                "QNN EP: Buffer 'buffer_dim' is out of range for the input tensor rank.");

  return Ort::Status();
}

Ort::Status BufferOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger,
                                           std::vector<std::string>& input_names,
                                           bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  // HTP's floating-point Buffer kernel consumes FP16. Keep the ONNX/QNN graph input FP32
  // for the application, but feed the internal Buffer node through an explicit Cast.
  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
  if (input_info.qnn_data_type == QNN_DATATYPE_FLOAT_32) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));
    const std::string& original_input_name = input_names.back();
    const std::string cast_output_name = utils::UniqueNameGenerator().New(original_input_name, "_buffer_cast_fp16");
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(
        utils::UniqueNameGenerator().New(node_unit, "_buffer_fp32_to_fp16"), original_input_name,
        cast_output_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16,
        QnnQuantParamsWrapper(), std::vector<uint32_t>(input_info.shape), do_op_validation));
    input_names.back() = cast_output_name;
  } else {
    // in[0]: activation (mandatory).
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));
  }

  // in[1]: reset (optional BOOL scalar). Wire it through when present so the QNN Buffer node
  // receives its reset signal; leave it off otherwise (matching the handling of a
  // missing reset input).
  if (inputs.size() > 1 && inputs[1].Exists()) {
    TensorInfo reset_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], reset_info));
    RETURN_IF_NOT(reset_info.qnn_data_type == QNN_DATATYPE_BOOL_8,
                  "QNN EP: Buffer reset input must have QNN BOOL_8 data type.");
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[1], logger, input_names));
  }

  return Ort::Status();
}

Ort::Status BufferOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& logger,
                                                         bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  // buffer_size
  const uint32_t buffer_size = SafeInt<uint32_t>(node_helper.Get("buffer_size", static_cast<int64_t>(0)));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), buffer_size,
                                         QNN_OP_BUFFER_PARAM_BUFFER_SIZE, param_tensor_names));

  // buffer_dim
  const uint32_t buffer_dim = SafeInt<uint32_t>(node_helper.Get("buffer_dim", static_cast<int64_t>(0)));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), buffer_dim,
                                         QNN_OP_BUFFER_PARAM_BUFFER_DIM, param_tensor_names));

  // stride (default 1 per QnnOpDef.h)
  const uint32_t stride = SafeInt<uint32_t>(node_helper.Get("stride", static_cast<int64_t>(1)));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), stride,
                                         QNN_OP_BUFFER_PARAM_STRIDE, param_tensor_names));

  // mode (IsOpSupported already rejected the unsupported default 0)
  const uint32_t mode = SafeInt<uint32_t>(node_helper.Get("mode", static_cast<int64_t>(QNN_OP_BUFFER_MODE_BLOCKING)));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), mode,
                                         QNN_OP_BUFFER_PARAM_MODE, param_tensor_names));

  // buffer_padding (default 0 per QnnOpDef.h)
  const uint32_t buffer_padding = SafeInt<uint32_t>(node_helper.Get("buffer_padding", static_cast<int64_t>(0)));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), buffer_padding,
                                         QNN_OP_BUFFER_PARAM_BUFFER_PADDING, param_tensor_names));

  TensorInfo input_info = {};
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  if (input_info.qnn_data_type == QNN_DATATYPE_FLOAT_32) {
    RETURN_IF_NOT(output_info.qnn_data_type == QNN_DATATYPE_FLOAT_32,
                  "QNN EP: FP32 Buffer input requires an FP32 output.");
  }

  // Match the FP32 ONNX output with an FP16 internal Buffer output, then cast back to the
  // application-visible FP32 output after the QNN Buffer node.
  if (input_info.qnn_data_type == QNN_DATATYPE_FLOAT_32 &&
      output_info.qnn_data_type == QNN_DATATYPE_FLOAT_32) {
    const std::string buffer_output_name = utils::UniqueNameGenerator().New(node_unit, "_buffer_fp16_output");
    QnnTensorWrapper buffer_output_wrapper(buffer_output_name, QNN_TENSOR_TYPE_NATIVE,
                                           QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                           std::vector<uint32_t>(output_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(buffer_output_wrapper)),
                  "Failed to add FP16 Buffer output tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_BUFFER),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_BUFFER,
                                                  std::move(input_names), {buffer_output_name},
                                                  std::move(param_tensor_names), do_op_validation),
                  "Failed to add QNN Buffer node.");

    QnnTensorWrapper output_wrapper(node_unit.Outputs()[0].name,
                                    qnn_model_wrapper.GetTensorType(node_unit.Outputs()[0].name),
                                    QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(output_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_wrapper)),
                  "Failed to add FP32 Buffer output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, "_buffer_fp16_to_fp32"),
                      QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST, {buffer_output_name},
                      {node_unit.Outputs()[0].name}, {}, do_op_validation),
                  "Failed to add FP16-to-FP32 Buffer output Cast node.");
    return Ort::Status();
  }

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, QNN_OP_BUFFER);
}

Ort::Status BufferOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      const Ort::Logger& logger,
                                                      const std::vector<std::string>& input_names,
                                                      size_t output_index,
                                                      Qnn_DataType_t qnn_data_type,
                                                      QnnQuantParamsWrapper& quant_param) const {
  if (!quant_param.IsPerTensor()) {
    return Ort::Status();
  }

  // Buffer copies quantization encodings from input to output. The activation
  // is in[0], so force the output qparams to match it when nearly equal.
  return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                  0 /*input_index*/, output_index, qnn_data_type, quant_param);
}

void CreateBufferOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<BufferOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
