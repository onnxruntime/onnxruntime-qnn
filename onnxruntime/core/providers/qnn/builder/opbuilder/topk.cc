// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

const int TOPK_MIN_INPUT = 2;
const int TOPK_MAX_INPUT = 2;

class TopKOpBuilder : public BaseOpBuilder {
 public:
  TopKOpBuilder() : BaseOpBuilder("TopKOpBuilder") {}

 protected:
  Qnn_DataType_t GetSupportedOutputDataType(size_t index, Qnn_DataType_t qnn_data_type) const override {
    if (index == 1) {
      if (qnn_data_type == QNN_DATATYPE_INT_64) {
        return QNN_DATATYPE_INT_32;
      } else if (qnn_data_type == QNN_DATATYPE_UINT_64) {
        return QNN_DATATYPE_UINT_32;
      }
    }
    return qnn_data_type;
  }

  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;

  // Helper to check if data type is unsupported SFIXED_POINT (16 or 32 bit)
  // TopK supports SFIXED_POINT_8, UFIXED_POINT_8, UFIXED_POINT_16, but NOT SFIXED_POINT_16/32
  static bool IsUnsupportedQuantizedType(Qnn_DataType_t qnn_data_type) {
    return qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16 ||
           qnn_data_type == QNN_DATATYPE_SFIXED_POINT_32;
  }

  // Store original quantization info for later use in ProcessAttributesAndOutputs
  mutable std::optional<TensorInfo> original_input_info_;
};

Status TopKOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  size_t input_count = node_unit.Inputs().size();
  size_t output_count = node_unit.Outputs().size();
  ORT_RETURN_IF_NOT(input_count >= TOPK_MIN_INPUT && input_count <= TOPK_MAX_INPUT,
                    "For ONNX TopK operation the expected number of inputs is 2.");
  ORT_RETURN_IF_NOT(output_count == 2, "QNN TopK expects exactly 2 outputs.");

  // Skip the first input. The second input needs to be an initializer.
  const auto& input_1 = node_unit.Inputs()[1].node_arg.Name();
  if (!qnn_model_wrapper.IsConstantInput(input_1)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The number of top elements to retrieve must be specified as constant input.");
  }

  return Status::OK();
}

Status TopKOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }

  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  // Get input tensor info to check if it's an unsupported quantized type
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  // Check if input is SFIXED_POINT_16 or SFIXED_POINT_32 (unsupported by TopK)
  // If so, insert a DequantizeLinear node to convert to float
  if (IsUnsupportedQuantizedType(input_info.qnn_data_type)) {
    // Store original input info for later use in ProcessAttributesAndOutputs
    original_input_info_ = input_info;

    LOGS(logger, VERBOSE) << "TopK input has unsupported quantized type. Inserting DequantizeLinear node to convert to float.";

    // Create DequantizeLinear output tensor (float)
    const std::string dq_output_name = utils::GetUniqueName(input_names[0], "_dq_float");
    Qnn_DataType_t float_data_type = QNN_DATATYPE_FLOAT_32;

    QnnTensorWrapper dq_output_tensor(dq_output_name,
                                      QNN_TENSOR_TYPE_NATIVE,
                                      float_data_type,
                                      QnnQuantParamsWrapper(),  // No quantization for float
                                      std::vector<uint32_t>(input_info.shape));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(dq_output_tensor)),
                      "Failed to add DequantizeLinear output tensor.");

    // Create DequantizeLinear node
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(node_unit, "_dequantize"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_DEQUANTIZE,
                          {input_names[0]},
                          {dq_output_name},
                          {},
                          do_op_validation),
                      "Failed to create DequantizeLinear node.");

    // Update input to use the dequantized float tensor
    input_names[0] = dq_output_name;

    // Update input_info to reflect the new float data type
    input_info.qnn_data_type = float_data_type;
    input_info.quant_param = QnnQuantParamsWrapper();
  }

  // HTP only supports TopK at the last axis, and thus check whether extra Transpose is required.
  size_t input_rank = input_info.shape.size();
  int32_t axis = NodeAttrHelper(node_unit).Get("axis", -1);
  if (axis == -1 || axis == static_cast<int32_t>(input_rank - 1)) {
    return Status::OK();
  }

  // Add Transpose to permute axis to the last.
  const std::string transpose_output_name = utils::GetUniqueName(input_names[0], "_transpose");
  std::vector<uint32_t> transpose_perm;
  ORT_RETURN_IF_ERROR(utils::GetPermToLastAxis(static_cast<uint32_t>(axis),
                                               static_cast<uint32_t>(input_rank),
                                               transpose_perm));

  std::vector<uint32_t> transpose_output_shape = input_info.shape;
  transpose_output_shape[input_rank - 1] = input_info.shape[axis];
  transpose_output_shape[axis] = input_info.shape[input_rank - 1];

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                         input_names[0],
                                                         transpose_output_name,
                                                         input_info.shape,
                                                         transpose_perm,
                                                         transpose_output_shape,
                                                         input_info.qnn_data_type,
                                                         input_info.quant_param,
                                                         do_op_validation,
                                                         false,
                                                         false));
  input_names[0] = transpose_output_name;

  return Status::OK();
}

Status TopKOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  auto& input_name = node_unit.Inputs()[1].node_arg.Name();
  uint32_t k = 0;  // The number of elements to extract from the input tensor at each position.
  bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
  if (is_constant_input) {
    std::vector<uint8_t> unpacked_tensor;
    const auto& input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
    const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
    k = static_cast<uint32_t>(*tensor_data);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN TopK operator requires constant input parameter k.");
  }
  Qnn_Scalar_t qnn_scalar_k = QNN_SCALAR_INIT;
  qnn_scalar_k.dataType = QNN_DATATYPE_UINT_32;
  qnn_scalar_k.uint32Value = k;
  QnnParamWrapper k_param(node_unit.Index(), node_unit.Name(), QNN_OP_TOP_K_PARAM_K, qnn_scalar_k);
  std::string k_param_name = k_param.GetParamTensorName();
  qnn_model_wrapper.AddParamWrapper(std::move(k_param));
  std::vector<std::string> param_tensor_names{k_param_name};

  // Add largest to TopK attr
  uint8_t largest = static_cast<uint8_t>(NodeAttrHelper(node_unit).Get("largest", 1));
  Qnn_Scalar_t qnn_largest_k = QNN_SCALAR_INIT;
  qnn_largest_k.dataType = QNN_DATATYPE_BOOL_8;
  qnn_largest_k.bool8Value = largest;
  QnnParamWrapper k_largest(node_unit.Index(), node_unit.Name(), QNN_OP_TOP_K_PARAM_LARGEST, qnn_largest_k);
  std::string k_largest_name = k_largest.GetParamTensorName();
  qnn_model_wrapper.AddParamWrapper(std::move(k_largest));
  param_tensor_names.push_back(k_largest_name);

  // HTP only supports TopK at the last axis, and thus check whether extra Transpose is required.
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  size_t input_rank = input_info.shape.size();
  int32_t axis = NodeAttrHelper(node_unit).Get("axis", -1);

  // Check if we need to handle quantization (i.e., if we inserted a DQ node earlier)
  bool need_quantize_output = original_input_info_.has_value();

  if (axis == -1 || axis == static_cast<int32_t>(input_rank - 1)) {
    // Simple case: no transpose needed
    if (need_quantize_output) {
      // We need to insert Q node after TopK's first output
      const auto& outputs = node_unit.Outputs();

      // Create intermediate float output for TopK
      const std::string topk_float_output_0 = utils::GetUniqueName(outputs[0].node_arg.Name(), "_topk_float");

      // Get output info
      TensorInfo output_info_0 = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info_0));

      // Create float tensor for TopK output[0]
      QnnTensorWrapper topk_float_tensor_0(topk_float_output_0,
                                           QNN_TENSOR_TYPE_NATIVE,
                                           QNN_DATATYPE_FLOAT_32,
                                           QnnQuantParamsWrapper(),
                                           std::vector<uint32_t>(output_info_0.shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(topk_float_tensor_0)),
                        "Failed to add TopK float output tensor.");

      // Create TopK node with float output for values
      std::vector<std::string> topk_output_names = {topk_float_output_0, outputs[1].node_arg.Name()};

      // Add output[1] (indices) tensor - this remains INT32
      TensorInfo output_info_1 = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[1], output_info_1));
      bool is_graph_output_1 = qnn_model_wrapper.IsGraphOutput(outputs[1].node_arg.Name());
      Qnn_TensorType_t tensor_type_1 = is_graph_output_1 ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

      QnnTensorWrapper output_tensor_1(outputs[1].node_arg.Name(),
                                       tensor_type_1,
                                       output_info_1.qnn_data_type,
                                       output_info_1.quant_param.Copy(),
                                       std::vector<uint32_t>(output_info_1.shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_1)),
                        "Failed to add TopK indices output tensor.");

      // Create TopK node
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                            utils::GetUniqueName(node_unit),
                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                            GetQnnOpType(node_unit.OpType()),
                            std::move(input_names),
                            std::move(topk_output_names),
                            std::move(param_tensor_names),
                            do_op_validation),
                        "Failed to create TopK node.");

      // Add QuantizeLinear node to convert output[0] back to original quantized type
      bool is_graph_output_0 = qnn_model_wrapper.IsGraphOutput(outputs[0].node_arg.Name());
      Qnn_TensorType_t tensor_type_0 = is_graph_output_0 ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

      QnnTensorWrapper q_output_tensor(outputs[0].node_arg.Name(),
                                       tensor_type_0,
                                       original_input_info_->qnn_data_type,
                                       original_input_info_->quant_param.Copy(),
                                       std::vector<uint32_t>(output_info_0.shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(q_output_tensor)),
                        "Failed to add QuantizeLinear output tensor.");

      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                            utils::GetUniqueName(node_unit, "_quantize"),
                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                            QNN_OP_QUANTIZE,
                            {topk_float_output_0},
                            {outputs[0].node_arg.Name()},
                            {},
                            do_op_validation),
                        "Failed to create QuantizeLinear node.");
    } else {
      // No quantization handling needed, use standard processing
      ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper,
                                         node_unit,
                                         std::move(input_names),
                                         std::move(param_tensor_names),
                                         logger,
                                         do_op_validation,
                                         GetQnnOpType(node_unit.OpType())));
    }
    return Status::OK();
  }

  const auto& outputs = node_unit.Outputs();
  std::vector<std::string> transpose_input_names;
  std::vector<std::vector<std::uint32_t>> transpose_input_shapes;

  // Add TopK outputs.
  for (size_t output_idx = 0; output_idx < 2; ++output_idx) {
    const auto& output = outputs[output_idx];

    // Since user may not be aware of the additional Transpose, the original output name of TopK node must be used by
    // the additional Transpose node which has the same output as original TopK node.
    const std::string& output_name = output.node_arg.Name();
    const std::string transpose_input_name = utils::GetUniqueName(output_name, "_transpose");
    transpose_input_names.push_back(std::move(transpose_input_name));

    // Since the input of TopK node is permuted, its output shape must be manually calculated.
    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, output_info));
    size_t output_rank = output_info.shape.size();

    std::vector<uint32_t> transpose_input_shape = output_info.shape;
    transpose_input_shape[output_rank - 1] = output_info.shape[axis];
    transpose_input_shape[axis] = output_info.shape[output_rank - 1];
    transpose_input_shapes.push_back(std::move(transpose_input_shape));

    // If we inserted DQ node, TopK output[0] should be float, not quantized
    Qnn_DataType_t topk_output_dtype = output_info.qnn_data_type;
    QnnQuantParamsWrapper topk_output_qparams = output_info.quant_param.Copy();

    if (need_quantize_output && output_idx == 0) {
      topk_output_dtype = QNN_DATATYPE_FLOAT_32;
      topk_output_qparams = QnnQuantParamsWrapper();
    }

    QnnTensorWrapper output_tensorwrapper(transpose_input_names[output_idx],
                                          QNN_TENSOR_TYPE_NATIVE,
                                          topk_output_dtype,
                                          std::move(topk_output_qparams),
                                          std::vector<uint32_t>(transpose_input_shapes[output_idx]));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  }

  // Add TopK node.
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    GetQnnOpType(node_unit.OpType()),
                                                    std::move(input_names),
                                                    std::vector<std::string>(transpose_input_names),
                                                    std::move(param_tensor_names)),
                    "Failed to add node.");

  // Add Transpose nodes for each output to permute back.
  for (size_t output_idx = 0; output_idx < 2; ++output_idx) {
    const auto& output = outputs[output_idx];
    const std::string& output_name = output.node_arg.Name();

    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, output_info));
    size_t output_rank = output_info.shape.size();

    std::vector<uint32_t> transpose_perm;
    ORT_RETURN_IF_ERROR(utils::GetPermToLastAxis(static_cast<uint32_t>(axis),
                                                 static_cast<uint32_t>(output_rank),
                                                 transpose_perm));

    std::string transpose_output_name = output_name;
    bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

    // Check if we need to add QuantizeLinear after Transpose for output[0]
    bool need_quantize_after_transpose = need_quantize_output && output_idx == 0;
    if (need_quantize_after_transpose) {
      // Transpose outputs float, then we need to quantize it
      transpose_output_name = utils::GetUniqueName(output_name, "_transpose_float");
      is_graph_output = false;  // The Q node will be the graph output
    }

    // TopK's second output is indices which could be INT64 dtype, and QnnTensorWrapper directly changes the dtype to
    // INT32 during the wrapper construction. Nevertheless, if this output happens to be graph output, an additional
    // Cast must be added to cast dtype from INT32 back to INT64.
    bool is_cast_required = output_idx == 1 && output_info.qnn_data_type == QNN_DATATYPE_INT_64 && is_graph_output;
    std::string cast_input_name = "";
    if (is_cast_required) {
      cast_input_name = utils::GetUniqueName(transpose_output_name, "_cast");
      // For the same reason described above, the original output name is now used by this Cast.
      transpose_output_name = cast_input_name;
      // Since additional Cast is added, below Transpose is no longer graph output.
      is_graph_output = false;
    }

    // Determine the data type and quant params for the transpose output
    Qnn_DataType_t transpose_out_dtype = output_info.qnn_data_type;
    QnnQuantParamsWrapper transpose_out_qparams = output_info.quant_param.Copy();
    if (need_quantize_after_transpose) {
      transpose_out_dtype = QNN_DATATYPE_FLOAT_32;
      transpose_out_qparams = QnnQuantParamsWrapper();
    }

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                           transpose_input_names[output_idx],
                                                           transpose_output_name,
                                                           transpose_input_shapes[output_idx],
                                                           transpose_perm,
                                                           output_info.shape,
                                                           transpose_out_dtype,
                                                           transpose_out_qparams,
                                                           do_op_validation,
                                                           false,
                                                           is_graph_output));

    if (need_quantize_after_transpose) {
      // Add QuantizeLinear node to convert float back to original quantized type
      bool is_final_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
      Qnn_TensorType_t tensor_type = is_final_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

      QnnTensorWrapper q_output_tensor(output_name,
                                       tensor_type,
                                       original_input_info_->qnn_data_type,
                                       original_input_info_->quant_param.Copy(),
                                       std::vector<uint32_t>(output_info.shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(q_output_tensor)),
                        "Failed to add QuantizeLinear output tensor.");

      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                            utils::GetUniqueName(node_unit, "_quantize"),
                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                            QNN_OP_QUANTIZE,
                            {transpose_output_name},
                            {output_name},
                            {},
                            do_op_validation),
                        "Failed to create QuantizeLinear node.");
    }

    if (is_cast_required) {
      QnnTensorWrapper cast_output_tensorwrapper(output_name,
                                                 QNN_TENSOR_TYPE_APP_READ,
                                                 output_info.qnn_data_type,
                                                 output_info.quant_param.Copy(),
                                                 std::vector<uint32_t>(output_info.shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_output_tensorwrapper)),
                        "Failed to add tensor.");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit, QNN_OP_CAST),
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CAST,
                                                        {cast_input_name},
                                                        {output_name},
                                                        {}),
                        "Failed to add node");
    }
  }

  return Status::OK();
}

void CreateTopKOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<TopKOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
