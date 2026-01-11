// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class ConcatOpBuilder : public BaseOpBuilder {
 public:
  ConcatOpBuilder() : BaseOpBuilder("ConcatOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConcatOpBuilder);

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
};

Status ConcatOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  std::vector<std::string> input_names_with_null_tensor;

  for (const auto& input : inputs) {
    const auto& input_name = input.node_arg.Name();

    // Check if the input tensor already exists in the model
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;

      // Check if the tensor has a 0 dimension
      std::vector<uint32_t> shape;
      ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.node_arg, shape), "Cannot get shape");

      if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        // Found a 0 dimension, add to list of tensors to exclude
        LOGS(logger, VERBOSE) << "Input tensor " << input_name << " has a 0 dimension, excluding from Concat";
        input_names_with_null_tensor.push_back(input_name);
      } else {
        input_names.push_back(input_name);
      }
      continue;
    }

    // Process constant inputs
    if (qnn_model_wrapper.IsConstantInput(input_name)) {
      const auto& input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));

      // Check if the tensor has a 0 dimension
      const auto& shape = input_tensor->dims();
      if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        // Found a 0 dimension, add to list of tensors to exclude
        LOGS(logger, VERBOSE) << "Constant input tensor " << input_name << " has a 0 dimension, excluding from Concat";
        input_names_with_null_tensor.push_back(input_name);
        continue;
      }

      // Add the constant tensor to the model
      std::vector<uint32_t> input_shape;
      ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.node_arg, input_shape), "Cannot get shape");

      Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
      const auto* type_proto = input.node_arg.TypeAsProto();
      ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false, type_proto, qnn_data_type));

      QnnTensorWrapper input_tensorwrapper(input_name, QNN_TENSOR_TYPE_STATIC, qnn_data_type, QnnQuantParamsWrapper(),
                                           std::move(input_shape), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
      input_names.push_back(input_name);
    } else {
      // Process non-constant inputs
      std::vector<uint32_t> shape;
      ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.node_arg, shape), "Cannot get shape");

      if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        // Found a 0 dimension, add to list of tensors to exclude
        LOGS(logger, VERBOSE) << "Input tensor " << input_name << " has a 0 dimension, excluding from Concat";
        input_names_with_null_tensor.push_back(input_name);
        continue;
      }

      // Process the input normally
      ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, input, logger, input_names));
    }
  }

  // Remove inputs with null tensors from the input_names list
  for (const auto& null_tensor_name : input_names_with_null_tensor) {
    auto it = std::find(input_names.begin(), input_names.end(), null_tensor_name);
    if (it != input_names.end()) {
      input_names.erase(it);
    }
  }

  // If all inputs have 0 dimensions, we need at least one input for the Concat op
  if (input_names.empty()) {
    LOGS(logger, WARNING) << "All inputs to Concat have 0 dimensions. Using the first input.";
    const auto& input = inputs[0];
    // Process the first input even though it has 0 dimensions
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, input, logger, input_names));
  }

  return Status::OK();
}

Status ConcatOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Status::OK();
  }

  std::vector<std::string> param_tensor_names;

  // Process axis attribute
  int32_t default_axis = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_CONCAT_PARAM_AXIS, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  // Process outputs
  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ConcatOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
