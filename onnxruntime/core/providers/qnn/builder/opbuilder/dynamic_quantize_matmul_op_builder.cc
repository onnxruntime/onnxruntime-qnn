// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

/**
 * An ONNX DynamicQuantizeMatMul can be translated to QNN's MatMul
 */
class DynamicQuantizeMatMulOpBuilder : public BaseOpBuilder {
 public:
  DynamicQuantizeMatMulOpBuilder() : BaseOpBuilder("DynamicQuantizeMatMulOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DynamicQuantizeMatMulOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit, const logging::Logger& logger,
                       std::vector<std::string>& input_names, bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names, const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ProcessInputsForQnnMatMul(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   const TensorInfo& input_info_0,
                                   const TensorInfo& input_info_1,
                                   const logging::Logger& logger,
                                   std::vector<std::string>& input_names,
                                   bool do_op_validation) const ORT_MUST_USE_RESULT;
  Status ProcessInputsForQnnFullyConnected(QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnit& node_unit,
                                           const TensorInfo& input_info_0,
                                           const TensorInfo& input_info_1,
                                           const logging::Logger& logger,
                                           std::vector<std::string>& input_names,
                                           bool do_op_validation) const ORT_MUST_USE_RESULT;
};

namespace {

// Process input[0] for ONNX DynamicQuantizeMatMul that can be translated to QNN MatMul.
Status ProcessInput0(QnnModelWrapper& qnn_model_wrapper,
                     const TensorInfo& input_0_info,
                     const std::string& original_input_0_name,
                     std::vector<std::string>& input_names,
                     const logging::Logger& logger,
                     bool do_op_validation) {
  bool reshape_input_0 = input_0_info.shape.size() == 1;
  std::string actual_input_0_name = original_input_0_name;

  if (reshape_input_0) {
    actual_input_0_name = original_input_0_name + "_ort_qnn_ep_reshape";
    std::vector<uint32_t> shape_2d{1, input_0_info.shape[0]};
    QnnQuantParamsWrapper quant_param_2d = input_0_info.quant_param.Copy();
    ORT_RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_0_info.shape, shape_2d));

    // If input_0 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_0_info.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_0_info.initializer_tensor, unpacked_tensor));
      QnnTensorWrapper input_tensorwrapper(actual_input_0_name, QNN_TENSOR_TYPE_STATIC, input_0_info.qnn_data_type,
                                           std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(original_input_0_name, actual_input_0_name,
                                                           input_0_info.shape, shape_2d,
                                                           input_0_info.qnn_data_type, input_0_info.quant_param,
                                                           quant_param_2d, do_op_validation,
                                                           qnn_model_wrapper.IsGraphInput(original_input_0_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(actual_input_0_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << actual_input_0_name;
    } else {
      QnnTensorWrapper input_0_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_0_info, actual_input_0_name, input_0_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_0_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(actual_input_0_name);

  return Status::OK();
}
}  // namespace

// Process operator inputs.
Status DynamicQuantizeMatMulOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                                     const logging::Logger& logger, std::vector<std::string>& input_names,
                                                     bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_NOT(inputs.size() > 3);

  // Process input 0.
  TensorInfo input_info_0{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info_0));
  const std::string& org_input_0_name = inputs[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, org_input_0_name, input_names,
                                    logger, do_op_validation));

  // Process input 1, 2, and 3 into input1 of MatMul
  TensorInfo input_info_1{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input_info_1));
  ORT_RETURN_IF_NOT(input_info_1.is_initializer);
  const std::string& org_input_1_name = inputs[1].node_arg.Name();
  std::string input_1_name = org_input_1_name;

  std::vector<uint8_t> unpacked_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info_1.initializer_tensor, unpacked_tensor));

  QnnQuantParamsWrapper weight_qparams;

  // get weight_scale value from input[2]
  TensorInfo w_scale_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], w_scale_info));
  ORT_RETURN_IF_NOT(w_scale_info.is_initializer);
  std::vector<uint8_t> w_scale_initializer_data;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*w_scale_info.initializer_tensor, w_scale_initializer_data));
  // const float32_t* scale_data_ptr = reinterpret_cast<const float32_t*>(w_scale_initializer_data.data());
  // size_t w_scale_count = w_scale_initializer_data.size() / sizeof(float32_t);
  // gsl::span<const float> weight_scales(scale_data_ptr, w_scale_count);
  gsl::span<const float> weight_scales = ReinterpretAsSpan<const float32_t>(gsl::make_span(w_scale_initializer_data));

  TensorInfo w_offset_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[3], w_offset_info));
  ORT_RETURN_IF_NOT(w_offset_info.is_initializer);
  std::vector<uint8_t> w_offset_initializer_data;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*w_offset_info.initializer_tensor, w_offset_initializer_data));
  // const uint8_t* offset_data_ptr = reinterpret_cast<const uint8_t*>(w_offset_initializer_data.data());
  // size_t w_scale_count = w_scale_initializer_data.size() / sizeof(uint8_t);
  // gsl::span<const float> weight_scales(scale_data_ptr, w_scale_count);
  gsl::span<const uint8_t> tensor_elems = gsl::make_span(w_offset_initializer_data);
  gsl::span<const int32_t> weight_offsets = gsl::span<const int32_t>(reinterpret_cast<const int32_t*>(tensor_elems.data()),
                                                                     tensor_elems.size_bytes() / sizeof(int32_t));

  weight_qparams = QnnQuantParamsWrapper(weight_scales, weight_offsets, 1, false);

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_1_name);
  QnnTensorWrapper input_tensorwrapper(input_1_name, tensor_type, QNN_DATATYPE_SFIXED_POINT_8,
                                       std::move(weight_qparams), std::vector<uint32_t>(input_info_1.shape), std::move(unpacked_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  input_names.emplace_back(input_1_name);

  return Status::OK();
}

Status DynamicQuantizeMatMulOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                                                   std::vector<std::string>&& input_names,
                                                                   const logging::Logger& /*logger*/, bool do_op_validation) const {
  // For QNN MatMul: set the input transpose parameters to their default values of 0. These parameters should be
  // optional, but older versions of QNN SDK failed validation if not explicitly provided.
  std::vector<std::string> param_tensor_names;
  Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
  scalar_param.dataType = QNN_DATATYPE_BOOL_8;
  scalar_param.bool8Value = 0;
  QnnParamWrapper transpose_in0_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,
                                      scalar_param);
  param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(transpose_in0_param));

  QnnParamWrapper transpose_in1_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                                      scalar_param);
  param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(transpose_in1_param));

  const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
  std::string op_output_name = org_output_name;
  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  std::vector<uint32_t> op_output_shape = output_info.shape;
  QnnQuantParamsWrapper op_output_quant_param = output_info.quant_param.Copy();

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
  Qnn_TensorType_t op_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper op_output_tensor_wrapper(op_output_name, op_output_tensor_type, output_info.qnn_data_type,
                                            op_output_quant_param.Copy(), std::vector<uint32_t>(op_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(op_output_tensor_wrapper)),
                    "Failed to add output tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit), QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_MAT_MUL,
                                                    std::move(input_names), {op_output_name},
                                                    std::move(param_tensor_names), do_op_validation),
                    "Failed to add fused Matmul node.");

  return Status::OK();
}

void CreateDynamicQuantizeMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<DynamicQuantizeMatMulOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
