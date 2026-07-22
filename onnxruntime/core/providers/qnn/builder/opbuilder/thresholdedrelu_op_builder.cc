// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
class ThresholdedReluOpBuilder : public BaseOpBuilder {
 public:
  ThresholdedReluOpBuilder() : BaseOpBuilder("ThresholdedReluOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ThresholdedReluOpBuilder);

 protected:
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

 private:
  Ort::Status ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const;
};

Ort::Status ThresholdedReluOpBuilder::ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit) const {
  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  // Greater op supporting input dtypes
  static const std::unordered_set<int> greater_op_support_dtypes = {
      QNN_DATATYPE_FLOAT_16,
      QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_UFIXED_POINT_16,
      QNN_DATATYPE_SFIXED_POINT_16,
      QNN_DATATYPE_UFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_INT_32};

  RETURN_IF(greater_op_support_dtypes.count(input_info.qnn_data_type) == 0,
            "ThresholdRelu input data type not supported.");

  return Ort::Status();
}

static Ort::Status SetAlphaByte(Qnn_DataType_t qnn_data_type,
                                std::vector<uint8_t>& alpha_bytes,
                                float alpha_value) {
  switch (qnn_data_type) {
    case QNN_DATATYPE_FLOAT_16: {
      Ort::Float16_t zero_fp16 = static_cast<Ort::Float16_t>(alpha_value);
      uint16_t cast_value = *reinterpret_cast<uint16_t*>(&zero_fp16);
      alpha_bytes.resize(sizeof(uint16_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(uint16_t));
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      float cast_value = static_cast<float>(alpha_value);
      alpha_bytes.resize(sizeof(float));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(float));
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      uint16_t cast_value = static_cast<uint16_t>(alpha_value);
      alpha_bytes.resize(sizeof(uint16_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(uint16_t));
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      int16_t cast_value = static_cast<int16_t>(alpha_value);
      alpha_bytes.resize(sizeof(int16_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(int16_t));
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_8: {
      uint8_t cast_value = static_cast<uint8_t>(alpha_value);
      alpha_bytes.resize(sizeof(uint8_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(uint8_t));
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      int8_t cast_value = static_cast<int8_t>(alpha_value);
      alpha_bytes.resize(sizeof(int8_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(int8_t));
      break;
    }
    case QNN_DATATYPE_INT_32: {
      int32_t cast_value = static_cast<int32_t>(alpha_value);
      alpha_bytes.resize(sizeof(int32_t));
      std::memcpy(alpha_bytes.data(), &cast_value, sizeof(int32_t));
      break;
    }
    default: {
      return MAKE_EP_FAIL("Unsupported QNN Data type for thresholdedrelu.");
    }
  }

  return Ort::Status();
}

Ort::Status ThresholdedReluOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const OrtNodeUnit& node_unit,
                                                    const Ort::Logger& logger,
                                                    std::vector<std::string>& input_names,
                                                    bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_ERROR(ExplicitOpCheck(qnn_model_wrapper, node_unit));
  }
  OrtNodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }

  return Ort::Status();
}

Ort::Status ThresholdedReluOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                  const OrtNodeUnit& node_unit,
                                                                  std::vector<std::string>&& input_names,
                                                                  const Ort::Logger& logger,
                                                                  bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  OrtNodeAttrHelper node_helper(node_unit);
  std::string& input_name = input_names[0];
  const std::string& org_output_name = node_unit.Outputs()[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);

  std::vector<uint32_t> output_shape = output_info.shape;
  Qnn_TensorType_t op_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  // input --+--> greater(alpha) --> select --> output
  //    \______________________________/

  // 1. Greater
  // Create alpha tensor.
  float alpha = node_helper.Get("alpha", static_cast<float>(0));
  std::vector<uint8_t> alpha_bytes;
  RETURN_IF_ERROR(SetAlphaByte(input_info.qnn_data_type, alpha_bytes, alpha));

  std::string alpha_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_alpha");
  QnnTensorWrapper alpha_tensorwrapper(alpha_tensor_name,
                                       QNN_TENSOR_TYPE_STATIC,
                                       input_info.qnn_data_type,
                                       QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>({1}),
                                       std::move(alpha_bytes));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(alpha_tensorwrapper)), "Failed to add alpha tensor.");

  // Create Greater Node.
  std::string greater_name = utils::UniqueNameGenerator().New(node_unit, "_Greater");
  std::string greater_output_name = utils::UniqueNameGenerator().New(node_unit, "_Greater_output");
  QnnTensorWrapper greater_output(greater_output_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  QNN_DATATYPE_BOOL_8,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(greater_output)),
                "Failed to add ThresholdRelu - Greater output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(greater_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_GREATER,
                                                {input_name, alpha_tensor_name},
                                                {greater_output_name},
                                                {},
                                                do_op_validation),
                "Failed to add ThresholdRelu - Greater node.");

  // 2. Select
  // Create zero tensor.
  float zero = 0.0f;
  std::vector<uint8_t> zero_bytes;
  RETURN_IF_ERROR(SetAlphaByte(input_info.qnn_data_type, zero_bytes, zero));

  std::string zero_tensor_name = utils::UniqueNameGenerator().New(node_unit, "_zero");
  QnnTensorWrapper zero_tensorwrapper(zero_tensor_name,
                                      QNN_TENSOR_TYPE_STATIC,
                                      input_info.qnn_data_type,
                                      QnnQuantParamsWrapper(),
                                      std::vector<uint32_t>({1}),
                                      std::move(zero_bytes));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_tensorwrapper)), "Failed to add zero tensor.");

  // Create Select Node.
  std::string select_name = utils::UniqueNameGenerator().New(node_unit, "_Select");
  QnnTensorWrapper select_output(org_output_name,
                                 op_output_tensor_type,
                                 output_info.qnn_data_type,
                                 output_info.quant_param.Copy(),
                                 std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(select_output)),
                "Failed to add ThresholdRelu - Select output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(select_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_SELECT,
                                                {greater_output_name, input_name, zero_tensor_name},
                                                {org_output_name},
                                                {},
                                                do_op_validation),
                "Failed to add ThresholdRelu - Select node.");

  return Ort::Status();
}

void CreateThresholdedReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ThresholdedReluOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
