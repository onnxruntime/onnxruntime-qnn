// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class RandomUniformLikeOpBuilder : public BaseOpBuilder {
 public:
  RandomUniformLikeOpBuilder() : BaseOpBuilder("RandomUniformLikeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RandomUniformLikeOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override;
};

Status RandomUniformLikeOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 const logging::Logger& logger,
                                                 std::vector<std::string>& input_names,
                                                 bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();

  const auto& input_tensor = inputs[0];
  const std::string& input_tensor_name = input_tensor.node_arg.Name();
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_tensor_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_tensor_name;
    input_names.push_back(input_tensor_name);
  } else {
    QnnTensorWrapper input_tensorwrapper;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_tensor, input_tensorwrapper));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add input tensor.");
    input_names.push_back(input_tensor_name);
  }

  NodeAttrHelper node_helper(node_unit);
  // Extract 'seed' attribute and create tensor input if provided
  if (node_helper.HasAttr("seed")) {
    float seed_value = node_helper.Get("seed", 0.0f);

    // Create scalar tensor data
    std::vector<uint32_t> scalar_shape = {1};
    std::vector<uint8_t> seed_data(sizeof(float));
    memcpy(seed_data.data(), &seed_value, sizeof(float));

    // Create seed tensor name
    const std::string seed_tensor_name = input_names[0] + "_ort_qnn_ep_seed";

    // Create QnnTensorWrapper for seed
    QnnTensorWrapper seed_tensor(seed_tensor_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32,
                                 QnnQuantParamsWrapper(), std::move(scalar_shape), std::move(seed_data));

    // Add to model wrapper
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(seed_tensor)), "Failed to add seed tensor");

    // Add to input names for QNN node creation
    input_names.push_back(seed_tensor_name);
  }
  return Status::OK();
}

Status RandomUniformLikeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                               const NodeUnit& node_unit,
                                                               std::vector<std::string>&& input_names,
                                                               const logging::Logger& logger,
                                                               bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_names;
  // Extract 'low' attribute
  float low = node_helper.Get("low", 0.0f);
  Qnn_Scalar_t low_param = QNN_SCALAR_INIT;
  low_param.dataType = QNN_DATATYPE_FLOAT_32;
  low_param.floatValue = low;
  QnnParamWrapper low_param_wrapper(node_unit.Index(),
                                    node_unit.Name(),
                                    QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_LOW,
                                    low_param);

  param_names.push_back(low_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(low_param_wrapper));

  // Extract 'high' attribute
  float high = node_helper.Get("high", 1.0f);
  Qnn_Scalar_t high_param = QNN_SCALAR_INIT;
  high_param.dataType = QNN_DATATYPE_FLOAT_32;
  high_param.floatValue = high;
  QnnParamWrapper high_param_wrapper(node_unit.Index(),
                                     node_unit.Name(),
                                     QNN_OP_RANDOM_UNIFORM_LIKE_PARAM_HIGH,
                                     high_param);

  param_names.push_back(high_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(high_param_wrapper));

  const auto& outputs = node_unit.Outputs();
  const std::string& output_name = outputs[0].node_arg.Name();

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  QnnTensorWrapper output_tensorwrapper(output_name, tensor_type, output_info.qnn_data_type,
                                        output_info.quant_param.Copy(), std::move(output_info.shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        utils::GetNodeName(node_unit),
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        QNN_OP_RANDOM_UNIFORM_LIKE,
                        std::move(input_names),
                        {output_name},
                        std::move(param_names),
                        do_op_validation),
                    "Failed to create RandomUniformLike node.");

  return Status::OK();
}

void CreateRandomUniformLikeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RandomUniformLikeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
