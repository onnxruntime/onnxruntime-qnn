// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/qdq_constant_folding.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

// Operator which only need to handle node inputs & outputs, no attributes or no need to handle attributes
class SimpleOpBuilder : public BaseOpBuilder {
 public:
  SimpleOpBuilder() : BaseOpBuilder("SimpleOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SimpleOpBuilder);

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const;

  static constexpr std::array<std::string_view, 3> gridsample_supported_modes = {"bilinear", "nearest", "linear"};
  static constexpr std::array<std::string_view, 3> gridsample_supported_padding_modes = {"zeros", "border", "reflection"};
};

Ort::Status SimpleOpBuilder::ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const {
  const std::string& op_type = node_unit.OpType();

  if (op_type == "GridSample") {
    OrtNodeAttrHelper node_helper(node_unit);
    std::string mode = node_helper.Get("mode", "linear");
    RETURN_IF_NOT(utils::ArrayHasString(gridsample_supported_modes, mode),
                  ("GridSample does not support mode " + mode).c_str());
    std::string padding_mode = node_helper.Get("padding_mode", "zeros");
    RETURN_IF_NOT(utils::ArrayHasString(gridsample_supported_padding_modes, padding_mode),
                  ("GridSample does not support padding_mode " + padding_mode).c_str());
  }

  const auto qnn_backend_type = qnn_model_wrapper.GetQnnBackendType();

  // TODO: Remove once QNN HTP PRelu bug is fixed
  if (op_type == "PRelu") {
    RETURN_IF(qnn_backend_type == QnnBackendType::CPU,
              "QNN EP does not support PRelu op on CPU backend. Falling back to ORT CPU.");
  }

  // ONNX's Min, Max, and Sum operators accept a variable number of inputs (i.e., variadic).
  // However, QNN's Min, Max, and Add operators must take in exactly two inputs.
  if (op_type == "Min" || op_type == "Max") {
    RETURN_IF_NOT(node_unit.Inputs().size() == 2,
                  ("QNN EP only supports " + op_type + " operator with exactly 2 inputs.").c_str());
  }

  if (op_type == "Sum") {
    size_t inputs_num = node_unit.Inputs().size();
    RETURN_IF_NOT(inputs_num == 2,
                  ("QNN EP supports Sum operator with QNN_OP_ELEMENT_WISE_BINARY, which takes exactly 2 inputs."
                   "Got ONNX's Sum operator with " +
                   std::to_string(inputs_num) + " inputs.")
                      .c_str());
  }

  if (op_type == "DequantizeLinear") {
    bool is_per_chan_quant = false;
    int64_t quant_axis = 0;
    RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(node_unit.Inputs()[0], is_per_chan_quant, quant_axis));
    // Per-channel standalone DQ is allowed only if the input is a compile-time constant;
    const bool is_input_const = qnn_model_wrapper.IsEffectivelyConstantInput(node_unit.Inputs()[0].name);
    RETURN_IF(is_per_chan_quant && !is_input_const,
              "QNN EP does not support a standalone DQ op with per-channel quantization");

    if (qnn_model_wrapper.GetModelSettings().offload_graph_io_quantization &&
        qnn_model_wrapper.IsGraphOutput(node_unit.Outputs()[0].name)) {
      // Only register the override for the first DQ node that consumes this graph output.
      // If another DQ node already maps to the same external name, skip registration so
      // that the second output becomes a separate APP_READ tensor instead of creating
      // two APP_READ tensors with the same external name (which reduces the composed
      // QNN graph's input count and causes a null slot in qnn_tensor_infos at runtime).
      if (!qnn_model_wrapper.IsExternalOverrideTarget(node_unit.Outputs()[0].name)) {
        // The tensor name override is used to align the output name of DLC produced by IRBackend
        // with the output name of original onnx graph for better consistency.
        qnn_model_wrapper.SetTensorNameOverride(/*internal=*/node_unit.Inputs()[0].name,
                                                /*external=*/node_unit.Outputs()[0].name);
      }
      return MAKE_EP_FAIL("QNN EP is configured to not take DQ nodes that generate a graph output.");
    }
  }

  if (op_type == "QuantizeLinear") {
    bool is_per_chan_quant = false;
    int64_t quant_axis = 0;
    RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(node_unit.Outputs()[0], is_per_chan_quant, quant_axis));
    // Per-channel standalone Q is allowed only if the input is a compile-time constant;
    const bool is_input_const = qnn_model_wrapper.IsEffectivelyConstantInput(node_unit.Inputs()[0].name);
    RETURN_IF(is_per_chan_quant && !is_input_const,
              "QNN EP does not support a standalone Q op with per-channel quantization");

    if (qnn_model_wrapper.GetModelSettings().offload_graph_io_quantization &&
        qnn_model_wrapper.IsGraphInput(node_unit.Inputs()[0].name)) {
      // Only register the override for the first Q node that consumes this graph input.
      // If another Q node already maps to the same external name, skip registration so
      // that the second input becomes a separate APP_WRITE tensor instead of creating
      // two APP_WRITE tensors with the same external name (which reduces the composed
      // QNN graph's input count and causes a null slot in qnn_tensor_infos at runtime).
      if (!qnn_model_wrapper.IsExternalOverrideTarget(node_unit.Inputs()[0].name)) {
        // The tensor name override is used to align the input name of DLC produced by IRBackend
        // with the input name of original onnx graph for better consistency.
        qnn_model_wrapper.SetTensorNameOverride(/*internal=*/node_unit.Outputs()[0].name,
                                                /*external=*/node_unit.Inputs()[0].name);
      }
      return MAKE_EP_FAIL("QNN EP is configured to not take Q nodes that consume a graph input.");
    }
  }

  return Ort::Status();
}

Ort::Status ProcessBlockSizeAttribute(QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& node_unit,
                                      std::vector<std::string>& param_tensor_names) {
  OrtNodeAttrHelper node_helper(node_unit);
  uint32_t block_size = node_helper.Get("blocksize", static_cast<uint32_t>(0));
  std::vector<uint32_t> block_size_shape{2};
  std::vector<uint32_t> block_size_data(2, block_size);
  QnnParamWrapper block_size_param(node_unit.Index(), node_unit.Name(), QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE,
                                   std::move(block_size_shape), std::move(block_size_data));
  param_tensor_names.push_back(block_size_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(block_size_param));

  return Ort::Status();
}

Ort::Status ProcessModeAttribute(QnnModelWrapper& qnn_model_wrapper,
                                 const OrtNodeUnit& node_unit,
                                 std::vector<std::string>& param_tensor_names) {
  OrtNodeAttrHelper node_helper(node_unit);
  std::string mode = node_helper.Get("mode", "DCR");
  uint32_t mode_value = QNN_OP_DEPTH_TO_SPACE_MODE_DCR;
  if ("DCR" == mode) {
    mode_value = QNN_OP_DEPTH_TO_SPACE_MODE_DCR;
  } else if ("CRD" == mode) {
    mode_value = QNN_OP_DEPTH_TO_SPACE_MODE_CRD;  // CRD mode
  } else {
    return MAKE_EP_FAIL("DepthToSpace mode only support DCR & CRD.");
  }

  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), mode_value,
                                         QNN_OP_DEPTH_TO_SPACE_PARAM_MODE, param_tensor_names));

  return Ort::Status();
}

// Process alpha attribute as input for Qnn LeakyRelu
Ort::Status ProcessAlphaAttributeAsInput(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const std::string input_name) {
  OrtNodeAttrHelper node_helper(node_unit);
  QnnQuantParamsWrapper quantize_param;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  union {
    float alpha;
    uint16_t alpha_fp16;
    uint8_t unpack[sizeof(float)];
  } tensor_data;
  tensor_data.alpha = node_helper.Get("alpha", 0.01f);
  std::vector<uint8_t> unpacked_data;
  // Check LeakyRelu input 0 to see if it's quantized tensor
  bool is_quantized_tensor = node_unit.Outputs()[0].quant_param.has_value();
  if (is_quantized_tensor) {
    qnn_data_type = QNN_DATATYPE_UFIXED_POINT_8;
    std::array<float, 1> scales = {1.0f};
    std::array<int32_t, 1> offsets = {0};
    std::array<uint32_t, 1> shape = {1};
    auto float_data = gsl::make_span<const float>(&tensor_data.alpha, 1);
    RETURN_IF_ERROR(qnn::utils::GetDataQuantParams(float_data, shape, scales, offsets, qnn_data_type));

    unpacked_data.resize(1);
    RETURN_IF_ERROR(qnn::utils::QuantizeData(float_data, shape, scales, offsets, unpacked_data, qnn_data_type));
    quantize_param = QnnQuantParamsWrapper::PerTensor(scales[0], static_cast<int32_t>(offsets[0]));
  } else {
    const auto& inputs = node_unit.Inputs();
    TensorInfo input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
    // QNN requires alpha is fp16 when input is fp16
    if (input_info.qnn_data_type == QNN_DATATYPE_FLOAT_16) {
      tensor_data.alpha_fp16 = Ort::Float16_t(tensor_data.alpha).val;
      qnn_data_type = QNN_DATATYPE_FLOAT_16;
      unpacked_data.assign(tensor_data.unpack, tensor_data.unpack + sizeof(Ort::Float16_t));
    } else {
      unpacked_data.assign(tensor_data.unpack, tensor_data.unpack + sizeof(float));
    }
  }
  std::vector<uint32_t> input_shape{1};
  Qnn_TensorType_t tensor_type = QNN_TENSOR_TYPE_STATIC;
  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, std::move(quantize_param),
                                       std::move(input_shape), std::move(unpacked_data));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  return Ort::Status();
}

Ort::Status ProcessGridSampleAttributes(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        std::vector<std::string>& param_tensor_names) {
  OrtNodeAttrHelper node_helper(node_unit);
  int64_t align_corners = node_helper.Get("align_corners", static_cast<int64_t>(0));
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), align_corners != 0,
                                     QNN_OP_GRID_SAMPLE_PARAM_ALIGN_CORNERS, param_tensor_names));

  std::string mode = node_helper.Get("mode", "linear");
  uint32_t mode_value = QNN_OP_GRID_SAMPLE_MODE_BILINEAR;
  if ("linear" == mode || "bilinear" == mode) {
    mode_value = QNN_OP_GRID_SAMPLE_MODE_BILINEAR;
  } else if ("nearest" == mode) {
    mode_value = QNN_OP_GRID_SAMPLE_MODE_NEAREST;
  } else {
    return MAKE_EP_FAIL("GridSample mode only support [linear, bilinear, nearest].");
  }
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), mode_value,
                                         QNN_OP_GRID_SAMPLE_PARAM_MODE, param_tensor_names));

  std::string padding_mode = node_helper.Get("padding_mode", "zeros");
  uint32_t padding_mode_value = QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS;
  if ("zeros" == padding_mode) {
    padding_mode_value = QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS;
  } else if ("border" == padding_mode) {
    padding_mode_value = QNN_OP_GRID_SAMPLE_PADDING_MODE_BORDER;
  } else if ("reflection" == padding_mode) {
    padding_mode_value = QNN_OP_GRID_SAMPLE_PADDING_MODE_REFLECTION;
  } else {
    return MAKE_EP_FAIL("GridSample padding_mode only support [zeros, border, reflection].");
  }
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), padding_mode_value,
                                         QNN_OP_GRID_SAMPLE_PARAM_PADDING_MODE, param_tensor_names));

  return Ort::Status();
}

Ort::Status SimpleOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& logger,
                                                         bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Ort::Status();
  }

  const std::string& op_type = node_unit.OpType();

  if (do_op_validation) {
    RETURN_IF_ERROR(ExplicitOpCheck(qnn_model_wrapper, node_unit));
    // Skip the op validation for DepthToSpace & SpaceToDepth if it's not NHWC data layout
    if (node_unit.Domain() != kMSInternalNHWCDomain && (op_type == "DepthToSpace" || op_type == "SpaceToDepth" || op_type == "GridSample")) {
      return Ort::Status();
    }
  }

  // Emit a STATIC tensor instead of an APP_WRITE input for standalone Q/DQ on constant inputs.
  if (CanFoldConstantQdq(qnn_model_wrapper, node_unit)) {
    Ort::Status fold_status = TryFoldConstantQDQ(qnn_model_wrapper, node_unit);
    if (fold_status.IsOK()) {
      return Ort::Status();
    }
  }

  std::vector<std::string> param_tensor_names;
  // Add attribute
  if (op_type == "LpNormalization") {
    int32_t axis = 0;
    RETURN_IF_ERROR(GetCanonicalizedAxisAttribute(qnn_model_wrapper, node_unit, "axis", -1, axis));
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                           static_cast<uint32_t>(axis), QNN_OP_L2_NORM_PARAM_AXIS, param_tensor_names));

    OrtNodeAttrHelper node_helper(node_unit);
    int64_t norm_p_order = node_helper.Get("p", static_cast<int64_t>(2));
    RETURN_IF(norm_p_order != 2, "QNN EP only supports LpNormalization with 'p' attribute equal to 2.");
  }

  if (op_type == "LeakyRelu") {
    std::string input_name = utils::UniqueNameGenerator().New(node_unit.Name(), "_alpha");
    RETURN_IF_ERROR(ProcessAlphaAttributeAsInput(qnn_model_wrapper, node_unit, input_name));
    input_names.push_back(input_name);
  }

  if (op_type == "DepthToSpace") {
    RETURN_IF_ERROR(ProcessBlockSizeAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
    RETURN_IF_ERROR(ProcessModeAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
  }

  if (op_type == "SpaceToDepth") {
    RETURN_IF_ERROR(ProcessBlockSizeAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                           static_cast<uint32_t>(QNN_OP_SPACE_TO_DEPTH_MODE_DCR),
                                           QNN_OP_SPACE_TO_DEPTH_PARAM_MODE, param_tensor_names));
  }

  if (op_type == "GridSample") {
    RETURN_IF_ERROR(ProcessGridSampleAttributes(qnn_model_wrapper, node_unit, param_tensor_names));
  }

  static const std::unordered_map<std::string, uint32_t> binary_op_to_operation = {
      {"Add", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD},
      {"Sub", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT},
      {"Mul", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY},
      {"Div", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE},
      {"Max", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM},
      {"Min", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM},
      {"Sum", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD},
      {"Expand", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY},
      {"And", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND},
      {"Or", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR},
      {"Xor", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_XOR},
      {"Equal", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL},
      {"NotEqual", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL},
      {"Greater", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER},
      {"GreaterOrEqual", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL},
      {"Less", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS},
      {"LessOrEqual", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL},
      {"Pow", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER},
  };
  auto binary_it = binary_op_to_operation.find(op_type);
  if (binary_it != binary_op_to_operation.end()) {
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                           static_cast<uint32_t>(binary_it->second),
                                           QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, param_tensor_names));
  }

  static const std::unordered_map<std::string, uint32_t> unary_op_to_operation = {
      {"Abs", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS},
      {"Asin", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN},
      {"Atan", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN},
      {"Ceil", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL},
      {"Cos", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS},
      {"Exp", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP},
      {"Floor", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR},
      {"Log", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG},
      {"Neg", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG},
      {"Not", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT},
      {"Round", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND},
      {"Sign", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN},
      {"Sin", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN},
      {"Sqrt", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT},
  };
  auto unary_it = unary_op_to_operation.find(op_type);
  if (unary_it != unary_op_to_operation.end()) {
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                           static_cast<uint32_t>(unary_it->second),
                                           QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION, param_tensor_names));
  }

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(op_type));
}

void CreateSimpleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SimpleOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
