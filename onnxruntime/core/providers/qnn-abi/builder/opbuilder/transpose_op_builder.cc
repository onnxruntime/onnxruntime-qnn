// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

class TransposeOpBuilder : public BaseOpBuilder {
 public:
  TransposeOpBuilder() : BaseOpBuilder("TransposeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TransposeOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                                   const OrtNodeUnit& node_unit,
                                   std::vector<std::string>& param_tensor_names) const;
};

Ort::Status TransposeOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnit& node_unit,
                                              const Ort::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  if (inputs.empty() || outputs.empty()) {
    return Ort::Status();
  }

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape),
                "Cannot get input shape");

  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].shape, output_shape),
                "Cannot get output shape");

  const size_t input_rank = input_shape.size();
  const size_t output_rank = output_shape.size();

  if (input_rank == 6 && output_rank == 6 && input_shape[0] == 1 && output_shape[0] == 1) {
    OrtNodeAttrHelper node_helper(node_unit);
    std::vector<int64_t> perm = node_helper.Get("perm", std::vector<int64_t>{});
    if (!perm.empty() && perm[0] != 0) {
      return Ort::Status("Transpose rank 6: first dimension must remain at index 0", ORT_FAIL);
    }
    return Ort::Status();
  }

  if (input_rank > 5) {
    return Ort::Status("QNN Transpose does not support input rank > 5", ORT_FAIL);
  }

  return Ort::Status();
}

Ort::Status TransposeOpBuilder::ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                                                     const OrtNodeUnit& node_unit,
                                                     std::vector<std::string>& param_tensor_names) const {
  auto inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape");
  // set default perm
  uint32_t rank = static_cast<uint32_t>(input_shape.size());
  std::vector<int64_t> transpose_perm(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    transpose_perm[i] = rank - 1 - i;
  }

  OrtNodeAttrHelper node_helper(node_unit);
  transpose_perm = node_helper.Get("perm", transpose_perm);
  auto perm_size = static_cast<uint32_t>(transpose_perm.size());
  std::vector<uint32_t> perm_shape{perm_size};
  std::vector<uint32_t> perm_data;
  perm_data.resize(perm_size);
  std::transform(transpose_perm.begin(), transpose_perm.end(), perm_data.begin(),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });

  QnnParamWrapper transpose_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                                  std::move(perm_shape), std::move(perm_data));
  param_tensor_names.push_back(transpose_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(transpose_param));

  return Ort::Status();
}

Ort::Status TransposeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            std::vector<std::string>&& input_names,
                                                            const Ort::Logger& logger,
                                                            bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Ort::Status();
  }

  // Check if input is rank 6 with first dimension = 1
  const auto& input0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input0_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input0.shape, input0_shape),
                ("Cannot get input shape for " + node_unit.OpType() + " node " + node_unit.Name()).c_str());

  const size_t input0_rank = input0_shape.size();
  bool is_rank6_with_unit_dim0 = (input0_rank == 6 && input0_shape[0] == 1);

  // Get original input name before any potential modifications
  std::string original_input_name = input_names[0];

  std::vector<std::string> param_tensor_names;

  // Handle rank 6 tensors with first dimension = 1
  if (is_rank6_with_unit_dim0) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
                "Converting rank 6 tensor to rank 5 for Transpose Op");
    
    // Create a rank 5 shape by removing the first dimension
    std::vector<uint32_t> rank5_shape;
    rank5_shape.reserve(5);
    for (size_t i = 1; i < input0_shape.size(); ++i) {
      rank5_shape.push_back(input0_shape[i]);
    }

    std::string reshaped_input_name = utils::GetUniqueName(input0.name, "_rank5");

    // Get quantization parameters for the input
    QnnQuantParamsWrapper quant_param;
    RETURN_IF_ERROR(quant_param.Init(qnn_model_wrapper, input0));

    // Get data type
    Qnn_DataType_t data_type;
    RETURN_IF_ERROR(utils::GetQnnDataType(quant_param.IsQuantized(), input0.type, data_type));

    // Add reshape node to convert from rank 6 to rank 5
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        input0.name,
        reshaped_input_name,
        input0_shape,
        rank5_shape,
        data_type,
        quant_param,
        do_op_validation,
        false,  // is_for_input
        false   // is_for_output
        ));

    // Update the input name for the transpose operation
    input_names[0] = reshaped_input_name;

    // Adjust the permutation indices for rank 5
    OrtNodeAttrHelper node_helper(node_unit);
    std::vector<int64_t> perm = node_helper.Get("perm", std::vector<int64_t>{});

    if (!perm.empty()) {
      // Create a new rank 5 permutation by removing the first index (which should be 0)
      std::vector<int64_t> rank5_perm;
      rank5_perm.reserve(5);

      // Skip the first element (index 0) and adjust the remaining indices
      for (size_t i = 1; i < perm.size(); ++i) {
        // Subtract 1 from each permutation index since we're removing the first dimension
        rank5_perm.push_back(perm[i] - 1);
      }

      // Override the permutation attribute
      auto perm_size = static_cast<uint32_t>(rank5_perm.size());
      std::vector<uint32_t> perm_shape{perm_size};
      std::vector<uint32_t> perm_data;
      perm_data.resize(perm_size);
      std::transform(rank5_perm.begin(), rank5_perm.end(), perm_data.begin(),
                     [](int64_t item) { return SafeInt<uint32_t>(item); });

      QnnParamWrapper transpose_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                                      std::move(perm_shape), std::move(perm_data));
      param_tensor_names.push_back(transpose_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(transpose_param));
    } else {
      // Use default permutation for rank 5
      RETURN_IF_ERROR(ProcessPermAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
    }
  } else {
    // Process permutation attribute normally
    RETURN_IF_ERROR(ProcessPermAttribute(qnn_model_wrapper, node_unit, param_tensor_names));
  }

  const auto& output_name = node_unit.Outputs()[0].name;
  std::vector<std::string> output_names;

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  struct CastNodeInfo {
    std::string node_name;
    std::string input_name;
    std::string output_name;
  };
  std::vector<CastNodeInfo> cast_node_info_vec;

  // Check if we need to add a cast node for int64
  bool needs_int64_cast = false;
  if (is_graph_output) {
    for (const auto& input_name : input_names) {
      if (input_name.find("_cast_int32") != std::string::npos) {
        needs_int64_cast = true;
        break;
      }
    }
  }

  const auto& transpose_output = node_unit.Outputs()[0];
  // Get the output info for the transpose output tensor
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(transpose_output, output_info));
  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Outputs()[0].shape, output_shape), "Cannot get shape");

  // Check if the target output shape has rank 6 with unit dimension at index 0
  bool is_rank6_with_unit_dim0_output = (output_shape.size() == 6 && output_shape[0] == 1);

  // Get the input tensor wrapper to access its properties
  const QnnTensorWrapper& input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);

  // Check if we need to reshape the output to rank 6 with unit dimension at index 0
  bool reshape_required = is_rank6_with_unit_dim0_output && input_tensor_wrapper.GetTensorRank() == 5;

  // If a cast to int64 is needed, add the cast node
  if (needs_int64_cast) {
    std::string cast_node_name = utils::GetUniqueName(node_unit, "_cast_int64");
    std::string cast_input_name = utils::GetUniqueName(output_name, "_cast_int64");
    std::string cast_output_name = output_name;

    // Create the cast input tensor wrapper
    QnnTensorWrapper cast_input_tensorwrapper(cast_input_name,
                                              QNN_TENSOR_TYPE_NATIVE,
                                              output_info.qnn_data_type,
                                              output_info.quant_param.Copy(),
                                              std::move(output_shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_input_tensorwrapper)), "Failed to add tensor.");
    cast_node_info_vec.emplace_back(CastNodeInfo{cast_node_name, cast_input_name, cast_output_name});
  }

  // Transpose output uses same data type and quantization parameter with input
  // 1. In QDQ model, the optimization may create scenario like Q -> Transpose -> DQ, Transpose is single node
  // Input tensor is created by previous node which is quantized tensor,
  // so output just copy the same data type and quantization parameters
  // 2. In QDQ model, Transpose also support non-quantized data like int32.
  // If we need to reshape to rank 6, create an intermediate tensor for the transpose output
  std::string transpose_output_name = output_name;
  if (reshape_required) {
    transpose_output_name = utils::GetUniqueName(output_name, "_rank5");

    // Create a rank 5 shape for the transpose output
    std::vector<uint32_t> rank5_shape;
    rank5_shape.reserve(5);
    for (size_t i = 1; i < output_shape.size(); ++i) {
      rank5_shape.push_back(output_shape[i]);
    }

    // Create the transpose output tensor with rank 5
    QnnTensorWrapper transpose_output_wrapper(transpose_output_name,
                                              QNN_TENSOR_TYPE_NATIVE,
                                              input_tensor_wrapper.GetTensorDataType(),
                                              input_tensor_wrapper.GetQnnQuantParams().Copy(),
                                              std::move(rank5_shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(transpose_output_wrapper)), "Failed to add tensor.");
  } else {
    // Create the normal output tensor
    QnnTensorWrapper output_tensorwrapper(output_name,
                                          tensor_type,
                                          input_tensor_wrapper.GetTensorDataType(),
                                          input_tensor_wrapper.GetQnnQuantParams().Copy(),
                                          std::move(output_shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  }

  output_names.push_back(transpose_output_name);
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_TRANSPOSE,
                                                std::move(input_names),
                                                std::move(output_names),
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to add node.");

  if (needs_int64_cast) {
    for (const auto& cast_node_info : cast_node_info_vec) {
      // Insert cast node.
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_info.node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CAST,
                                                    {cast_node_info.input_name},
                                                    {cast_node_info.output_name},
                                                    {}),
                    " Failed to add Cast node");
    }
  }

  // If we need to reshape from rank 5 to rank 6, add a reshape node
  if (reshape_required) {
    // Get data type and quantization parameters
    Qnn_DataType_t data_type = input_tensor_wrapper.GetTensorDataType();
    QnnQuantParamsWrapper quant_param = input_tensor_wrapper.GetQnnQuantParams().Copy();

    // Create the final output tensor with rank 6
    QnnTensorWrapper reshape_output(output_name,
                                    tensor_type,
                                    data_type,
                                    std::move(quant_param),
                                    std::move(output_shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output)), "Failed to add tensor.");

    // Add reshape node to convert from rank 5 to rank 6
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit, QNN_OP_RESHAPE),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_RESHAPE,
                                                  {transpose_output_name},
                                                  {output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add reshape node for rank 6 output.");
  }

  return Ort::Status();
}

void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<TransposeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
