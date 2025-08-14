// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"

namespace onnxruntime {
namespace qnn {
class ThresholdedReluOpBuilder : public BaseOpBuilder {
 public:
 ThresholdedReluOpBuilder() : BaseOpBuilder("ThresholdedReluOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ThresholdedReluOpBuilder);

 protected:
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
};

Status ThresholdedReluOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  // Greater op supporting input dtypes
  static const std::unordered_set<int> greater_op_support_dtypes = {
    QNN_DATATYPE_FLOAT_16,
    QNN_DATATYPE_FLOAT_32,
    QNN_DATATYPE_UFIXED_POINT_16,
    QNN_DATATYPE_SFIXED_POINT_16,
    QNN_DATATYPE_UFIXED_POINT_8,
    QNN_DATATYPE_SFIXED_POINT_8,
    QNN_DATATYPE_INT_32};

  if (greater_op_support_dtypes.count(input_info.qnn_data_type) == 0){ 
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ThresholdRelu input data type not supported.");
  }

  return Status::OK();
}

static Status SetQnnScalarValue(Qnn_DataType_t qnn_data_type,
                                std::vector<uint8_t>& zero_bytes,
                                float scalar_value) {
  float one = 1.0f;
  zero_bytes.resize(sizeof(float));
  std::memcpy(zero_bytes.data(), &scalar_value, sizeof(float));

  // switch (qnn_data_type) {
  //   case QNN_DATATYPE_FLOAT_16: {
  //     MLFloat16 zero_fp16 = static_cast<MLFloat16>(scalar_value);
  //     qnn_scalar.uint16Value = *reinterpret_cast<uint16_t*>(&zero_fp16);
  //     break;
  //   }
  //   case QNN_DATATYPE_FLOAT_32: {
  //     qnn_scalar.floatValue = static_cast<float>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_UFIXED_POINT_16: {
  //     qnn_scalar.uint16Value = static_cast<uint16_t>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_SFIXED_POINT_16: {
  //     qnn_scalar.int16Value = static_cast<int16_t>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_UFIXED_POINT_8: {
  //     qnn_scalar.uint8Value = static_cast<uint8_t>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_SFIXED_POINT_8: {
  //     qnn_scalar.int8Value = static_cast<int8_t>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_INT_32: {
  //     qnn_scalar.int32Value = static_cast<int32_t>(scalar_value);
  //     break;
  //   }
  // }

  // switch (qnn_data_type) {
  //   case QNN_DATATYPE_FLOAT_16: {
  //     MLFloat16 zero_fp16 = static_cast<MLFloat16>(scalar_value);
  //     qnn_scalar.uint16Value = *reinterpret_cast<uint16_t*>(&zero_fp16);
  //     break;
  //   }
  //   case QNN_DATATYPE_FLOAT_32: {
  //     qnn_scalar.floatValue = static_cast<float>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_UFIXED_POINT_16: {
  //     qnn_scalar.uint16Value = static_cast<uint16_t>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_SFIXED_POINT_16: {
  //     qnn_scalar.int16Value = static_cast<int16_t>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_UFIXED_POINT_8: {
  //     qnn_scalar.uint8Value = static_cast<uint8_t>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_SFIXED_POINT_8: {
  //     qnn_scalar.int8Value = static_cast<int8_t>(scalar_value);
  //     break;
  //   }
  //   case QNN_DATATYPE_INT_32: {
  //     qnn_scalar.int32Value = static_cast<int32_t>(scalar_value);
  //     break;
  //   }
  // }

  return Status::OK();
}


Status ThresholdedReluOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   const logging::Logger& logger,
                                   std::vector<std::string>& input_names,
                                   bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }
  NodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }

  return Status::OK();
}

Status ThresholdedReluOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 std::vector<std::string>&& input_names,
                                                 const logging::Logger& logger,
                                                 bool do_op_validation) const {
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  NodeAttrHelper node_helper(node_unit);
  std::string& input_name = input_names[0];
  const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);

  std::vector<uint32_t> output_shape = output_info.shape;
  Qnn_TensorType_t op_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;


  // Create alpha tensor.
  float alpha = node_helper.Get("alpha", static_cast<float>(0));
  std::vector<uint32_t> alpha_shape = {3, 5};
  size_t num_elements = alpha_shape[0] * alpha_shape[1];

  std::vector<uint8_t> alpha_bytes;
  alpha_bytes.resize(num_elements * sizeof(float));
  for (size_t i = 0; i < num_elements; ++i) {
      std::memcpy(alpha_bytes.data() + i * sizeof(float), &alpha, sizeof(float));
  }

  std::string alpha_tensor_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_alpha";
  QnnTensorWrapper alpha_tensorwrapper(alpha_tensor_name,
                                       QNN_TENSOR_TYPE_STATIC,
                                       input_info.qnn_data_type,
                                       QnnQuantParamsWrapper(),
                                       std::move(alpha_shape),
                                       std::move(alpha_bytes));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(alpha_tensorwrapper)), "Failed to add tensor.");

  bool path_1 = false;
  if (path_1){
    // input --> greater(alpha) -> [0] elementwise select -> output
    //        \--------------------[1]-/     0 ---[2]-/
    // 1. Greater
    std::string greater_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Greater";
    std::string greater_output_name = greater_name + "_output";
    QnnTensorWrapper greater_output(greater_output_name,
                                    QNN_TENSOR_TYPE_NATIVE,
                                    QNN_DATATYPE_BOOL_8,
                                    QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(greater_output)),
                      "Failed to add ThresholdRelu - Greater output tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(greater_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_GREATER,
                                                      {input_name, alpha_tensor_name},
                                                      {greater_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add ThresholdRelu - Greater node.");

    // 2. Select - out[0] = in[0] ? in[1] : in[2]
    // zero scalar value
    float zero = 0.0f;
    std::vector<uint32_t> zero_shape = {3, 5};
    size_t num_elements_2 = zero_shape[0] * zero_shape[1];

    std::vector<uint8_t> zero_bytes;
    zero_bytes.resize(num_elements * sizeof(float));

    for (size_t i = 0; i < num_elements; ++i) {
        std::memcpy(zero_bytes.data() + i * sizeof(float), &zero, sizeof(float));
    }


    std::string zero_tensor_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_zero";
    QnnTensorWrapper zero_tensorwrapper(zero_tensor_name,
                                        QNN_TENSOR_TYPE_STATIC,
                                        input_info.qnn_data_type,
                                        QnnQuantParamsWrapper(),
                                        std::move(zero_shape),
                                        std::move(zero_bytes));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_tensorwrapper)), "Failed to add tensor.");

    
    std::string select_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Select";
    QnnTensorWrapper select_output(org_output_name,
                                  op_output_tensor_type,
                                  output_info.qnn_data_type,
                                  output_info.quant_param.Copy(),
                                  std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(select_output)),
                      "Failed to add ThresholdRelu - ElementWiseSelect output tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(select_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_SELECT,
                                                      {greater_output_name, input_name, input_name},
                                                      {org_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add ThresholdRelu - ElementWiseSelect node.");
    
  }else{ // Do path 2: sub - relu - sign - mul ->
    // 1. sub
    std::string sub_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Sub";
    std::string sub_output_name = sub_name + "_output";
    // std::string sub_output_name = org_output_name;
    QnnTensorWrapper sub_output(sub_output_name,
                                QNN_TENSOR_TYPE_NATIVE, 
                                input_info.qnn_data_type,
                                QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sub_output)),
                      "Failed to add ThresholdRelu - Sub output tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(sub_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      // QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                      QNN_OP_ELEMENT_WISE_SUBTRACT,
                                                      {input_name, alpha_tensor_name},
                                                      {sub_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add ThresholdRelu - Sub node.");
    
    // 2. Relu
    std::string relu_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Relu";
    std::string relu_output_name = relu_name + "_output";
    // std::string relu_output_name = org_output_name;

    QnnTensorWrapper relu_output(relu_output_name,
                                QNN_TENSOR_TYPE_NATIVE, 
                                input_info.qnn_data_type,
                                QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(relu_output)),
                      "Failed to add ThresholdRelu - Relu output tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(relu_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_RELU,
                                                      {sub_output_name},
                                                      {relu_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add ThresholdRelu - Relu node.");

    // 3. Sign
    std::string sign_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Sign";
    std::string sign_output_name = sub_name + "_output";
    // std::string sign_output_name = org_output_name;
    QnnTensorWrapper sign_output(sign_output_name,
                                op_output_tensor_type,
                                input_info.qnn_data_type,
                                QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sign_output)),
                      "Failed to add ThresholdRelu - Sign output tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(sign_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_SIGN,
                                                      {relu_output_name},
                                                      {sign_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add ThresholdRelu - Sign node.");

    // 4. Mul
    std::string mul_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Mul";

    QnnTensorWrapper mul_output(org_output_name,
                                op_output_tensor_type,
                                output_info.qnn_data_type,
                                output_info.quant_param.Copy(),
                                std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mul_output)),
                      "Failed to add ThresholdRelu - Mul output tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(mul_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                      {input_name, sign_output_name},
                                                      {org_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add ThresholdRelu - Mul node.");
  }
  

  return Status::OK();
}

void CreateThresholdedReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ThresholdedReluOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
