// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/utils/op_builder_helper.h"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"

#include "QnnOpDef.h"

std::vector<uint32_t> GetPermFromShape(std::vector<uint32_t> input_shape, std::vector<uint32_t> ouput_shape) {
  std::vector<uint32_t> permutation(input_shape.size(), 0);
  std::unordered_map<uint32_t, uint32_t> input_dims;

  for (size_t i = 0; i < input_shape.size(); ++i) {
    input_dims[input_shape[i]] = static_cast<uint32_t>(i);
  }

  for (size_t i = 0; i < ouput_shape.size(); ++i) {
    auto it = input_dims.find(ouput_shape[i]);
    permutation[i] = it->second;
  }

  return permutation;
}

std::vector<uint32_t> ApplyPermutation(const std::vector<uint32_t>& input_shape, const std::vector<uint32_t>& perm) {
  std::vector<uint32_t> output_shape(input_shape.size(), 0);

  for (size_t i = 0; i < perm.size(); ++i) {
    output_shape[i] = input_shape[perm[i]];
  }

  return output_shape;
}

namespace onnxruntime {
namespace qnn {

OpBuilderHelper::OpBuilderHelper(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit)
    : qnn_model_wrapper_(qnn_model_wrapper),
      node_unit_(node_unit),
      org_output_name(node_unit.Outputs()[0].node_arg.Name()),
      is_graph_output(qnn_model_wrapper_.IsGraphOutput(org_output_name)),
      op_output_tensor_type(is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE) {
  ORT_THROW_IF_ERROR(qnn_model_wrapper_.GetTensorInfo(node_unit_.Inputs()[0], input_info_));
  ORT_THROW_IF_ERROR(qnn_model_wrapper_.GetTensorInfo(node_unit_.Outputs()[0], output_info_));
  std::vector<uint32_t> input_shape = input_info_.shape;
  tensor_to_shape_dict[node_unit_.Inputs()[0].node_arg.Name()] = input_shape;
}

Status OpBuilderHelper::RunQnnNodePreLowerValidation(const std::string& qnn_node_type,
                                                     std::vector<std::string>& input_names,
                                                     std::vector<std::string>& param_names,
                                                     std::vector<uint32_t>& output_shape,
                                                     std::vector<std::string>& output_name) {
  // Run Validation
  ORT_UNUSED_PARAMETER(param_names);

  const bool last_output = (output_name[0] == org_output_name);
  if (QNN_OP_RESHAPE == qnn_node_type && last_output) {
    if (output_info_.quant_param.IsPerChannel()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Do not support inserted Reshape nodes with per-channel quantization");
    }
  }

  if (QNN_OP_TRANSPOSE == qnn_node_type) {
    if (output_shape.size() != tensor_to_shape_dict[input_names[0]].size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Transpose expects input and output tensor has same rank.");
    }
  }

  return Status::OK();
}

Status OpBuilderHelper::AddSingleNode(const std::string& qnn_node_type,
                                      std::vector<std::string>&& input_names,
                                      std::vector<std::string>&& param_names,
                                      std::vector<uint32_t>&& output_shape,
                                      std::vector<std::string>&& output_name,
                                      bool do_op_validation) {
  // Add a node to qnn_model_wrapper, and add output tensor if not a graph output node.
  // Nodes with more than 1 output e.g. TopK requries user to create tensors first.

  ORT_RETURN_IF_ERROR(RunQnnNodePreLowerValidation(qnn_node_type, input_names, param_names, output_shape, output_name));

  // Add tensor(s)
  for (size_t j = 0; j < output_name.size(); ++j) {
    const bool last_output = (output_name[j] == org_output_name);

    QnnTensorWrapper node_output_tensor(output_name[j],
                                        last_output ? op_output_tensor_type : QNN_TENSOR_TYPE_NATIVE,
                                        output_info_.qnn_data_type,
                                        last_output ? output_info_.quant_param.Copy() : QnnQuantParamsWrapper(),
                                        std::vector<uint32_t>(output_shape));
    const std::string& error_msg = "Failed to add " + node_unit_.OpType() + " - " + qnn_node_type + " output tensor.";
    ORT_RETURN_IF_NOT(qnn_model_wrapper_.AddTensorWrapper(std::move(node_output_tensor)), error_msg);
  }

  // Append tensor and shape to dict.
  for (size_t i = 0; i < output_name.size(); ++i) {
    tensor_to_shape_dict[output_name[i]] = output_shape;
  }

  const std::string node_name = utils::GetUniqueName(node_unit_, qnn_node_type);
  const std::string& error_msg = "Failed to add " + node_unit_.OpType() + " - " + qnn_node_type + " node.";
  // Create perm param for transpose node.
  if (QNN_OP_TRANSPOSE == qnn_node_type) {
    // Add perm param
    std::vector<uint32_t> transpose_perm = GetPermFromShape(output_shape, tensor_to_shape_dict[input_names[0]]);
    uint32_t perm_size = static_cast<uint32_t>(transpose_perm.size());
    QnnParamWrapper transpose_param(node_unit_.Index(),
                                    std::string{output_name[0]},
                                    QNN_OP_TRANSPOSE_PARAM_PERM,
                                    std::vector<uint32_t>{perm_size},
                                    std::move(transpose_perm));
    param_names = {transpose_param.GetParamTensorName()};
    ORT_RETURN_IF_NOT(qnn_model_wrapper_.AddParamWrapper(std::move(transpose_param)), "Failed to add tensor.");
  }

  ORT_RETURN_IF_NOT(qnn_model_wrapper_.CreateQnnNode(
                        node_name,
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        qnn_node_type,
                        std::move(input_names),
                        std::move(output_name),
                        std::move(param_names),
                        do_op_validation),
                    error_msg);

  return Status::OK();
}

Status OpBuilderHelper::AddSequentialNode(const std::vector<std::string>& qnn_node_type,
                                          std::vector<std::string>&& input_names,
                                          std::vector<std::string>&& param_names,
                                          std::vector<uint32_t>&& output_shape,
                                          std::vector<std::string>&& output_name,
                                          bool do_op_validation) {
  // Add nodes sequentially.
  // Requirements:
  // 1. Each node has same output shape
  // 2. Except 1st node, each node has to be unary.
  std::vector<std::string>& input_tensor_name = input_names;

  for (size_t i = 0; i < qnn_node_type.size(); ++i) {
    const std::string& qnn_node_type_ = qnn_node_type[i];

    std::vector<std::string> output_tensor_name = (i == qnn_node_type.size() - 1) ? output_name : std::vector<std::string>{utils::GetUniqueName(node_unit_, qnn_node_type_)};

    ORT_RETURN_IF_ERROR(AddSingleNode(
        qnn_node_type_,
        std::vector<std::string>(input_names),
        (0 == i) ? std::vector<std::string>(param_names) : std::vector<std::string>({}),
        std::vector<uint32_t>(output_shape),
        std::vector<std::string>(output_tensor_name),
        do_op_validation));

    // Update input with output
    input_tensor_name = output_tensor_name;
  }

  return Status::OK();
}

Status OpBuilderHelper::AddSequentialNode(const std::vector<std::string>& qnn_node_type,
                                          std::vector<std::string>&& input_names,
                                          std::vector<std::string>&& param_names,
                                          std::vector<std::vector<uint32_t>>&& output_shape_and_perm,
                                          std::vector<std::string>&& output_name,
                                          bool do_op_validation) {
  // Add nodes sequentially.
  // Requirements:
  // 1. Each node has same output shape
  // 2. Except 1st node, each node has to be unary.
  std::vector<std::string>& input_tensor_name = input_names;

  for (size_t i = 0; i < qnn_node_type.size(); ++i) {
    const std::string& qnn_node_type_ = qnn_node_type[i];

    std::vector<std::string> output_tensor_name = (i == qnn_node_type.size() - 1) ? output_name : std::vector<std::string>{utils::GetUniqueName(node_unit_, qnn_node_type_)};

    // Update permutation to output_shape for Transpose.
    if (QNN_OP_TRANSPOSE == qnn_node_type_) {
      const std::vector<uint32_t>& perm = output_shape_and_perm[0];
      const std::vector<uint32_t>& input_shape = tensor_to_shape_dict[input_names[0]];
      std::vector<uint32_t> permute_shape = ApplyPermutation(input_shape, perm);
      output_shape_and_perm[0] = permute_shape;  // Update perm to shape.
    }

    // Pop first output_shape when seeing Reshape and when Reshape is not the first Op in the list of Ops.
    if (QNN_OP_RESHAPE == qnn_node_type_ && 0 != i) {
      output_shape_and_perm.erase(output_shape_and_perm.begin());
    }

    ORT_RETURN_IF_ERROR(AddSingleNode(
        qnn_node_type_,
        std::vector<std::string>(input_names),
        (0 == i) ? std::vector<std::string>(param_names) : std::vector<std::string>({}),  // Only unary node are allowed for 1+ nodes.
        std::vector<uint32_t>(output_shape_and_perm[0]),
        std::vector<std::string>(output_tensor_name),
        do_op_validation));

    // Update input with output
    input_tensor_name = output_tensor_name;
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
