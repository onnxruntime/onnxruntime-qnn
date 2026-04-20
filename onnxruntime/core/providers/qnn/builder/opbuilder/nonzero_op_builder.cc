// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <unordered_set>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

// NonZero op builder.
// Output shape is [rank_of_input, num_elements], where num_elements is the total number of
// elements in the input (the maximum possible number of non-zero elements).
class NonZeroOpBuilder : public BaseOpBuilder {
 public:
  NonZeroOpBuilder() : BaseOpBuilder("NonZeroOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NonZeroOpBuilder);

  // Override IsOpSupported to check backend type, consumer ops, and supported input types.
  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

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
};

Ort::Status NonZeroOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit,
                                            const Ort::Logger& logger) const {
  // NonZero is only supported on HTP backend.
  RETURN_IF_NOT(qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::HTP,
                "NonZero is only supported on HTP backend.");

  // NonZero output must have a static shape (no dynamic dims).
  // The ONNX model should set the output shape to [rank, num_elements] where num_elements
  // is the total number of elements in the input.
  const auto& output = node_unit.Outputs()[0];
  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(output.shape, output_shape),
                "NonZero output shape must be static. Set shape to [rank, num_elements].");

  const std::string& output_name = output.name;
  if (qnn_model_wrapper.IsGraphOutput(output_name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                "NonZero output is a graph output. QNN HTP pads unused elements with -1.");
  }

  // QNN HTP NonZero pads unused output elements with -1. Only these consumer ops handle -1 indices correctly.
  static const std::unordered_set<std::string> allowed_consumers = {
      "Gather", "GatherElements", "GatherND", "ScatterElements", "ScatterND", "Reshape", "Transpose"};

  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  for (const OrtNode* consumer : node_unit.GetOutputNodes(ort_api)) {
    std::string op = Ort::ConstNode(consumer).GetOperatorType();
    RETURN_IF_NOT(allowed_consumers.count(op) == 1,
                  ("NonZero consumer op '" + op + "' does not support -1 padded indices.").c_str());
  }

  // QNN HTP NonZero supported input types:
  //   FLOAT_16, UFIXED_POINT_16, UFIXED_POINT_8, BOOL_8
  // FLOAT_32 is also accepted here because ProcessInputs inserts a Cast(fp32 -> fp16).
  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Ort::Status NonZeroOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit,
                                            const Ort::Logger& logger,
                                            std::vector<std::string>& input_names,
                                            bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));

  // HTP NonZero only supports FLOAT_16 (not FLOAT_32) for float inputs.
  // Insert a Cast(fp32 -> fp16) node if needed.
  if (input_info.qnn_data_type == QNN_DATATYPE_FLOAT_32) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

    const std::string& original_input_name = input_names.back();
    const std::string cast_output_name = utils::UniqueNameGenerator().New(original_input_name, "_cast_fp16");

    // Create fp16 intermediate tensor.
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(cast_output_name + "_cast_node", original_input_name,
                                                  cast_output_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16,
                                                  QnnQuantParamsWrapper(), std::vector<uint32_t>(input_info.shape), do_op_validation));

    // Replace the input name with the casted fp16 tensor.
    input_names.back() = cast_output_name;
  } else {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));
  }

  return Ort::Status();
}

Ort::Status NonZeroOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                          const OrtNodeUnit& node_unit,
                                                          std::vector<std::string>&& input_names,
                                                          const Ort::Logger& logger,
                                                          bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  // Get input shape to compute maximum output size.
  const auto& inputs = node_unit.Inputs();
  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));

  const auto& input_shape = input_info.shape;
  uint32_t input_rank = static_cast<uint32_t>(input_shape.size());
  SafeInt<uint32_t> num_elements = 1;
  for (uint32_t dim : input_shape) {
    num_elements *= dim;
  }

  // QNN NonZero output shape: [num_nonzero_elements, rank_of_input] (transposed vs ONNX).
  // ONNX NonZero output shape: [rank_of_input, num_nonzero_elements].
  // Set num_nonzero_elements to max possible (total elements in input).
  std::vector<uint32_t> qnn_nonzero_shape = {num_elements, input_rank};
  std::vector<uint32_t> onnx_output_shape = {input_rank, num_elements};

  const auto& outputs = node_unit.Outputs();
  const std::string& output_name = outputs[0].name;
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  Qnn_DataType_t output_data_type = QNN_DATATYPE_INT_32;

  // Chain: NonZero(int32, [N, rank]) -> Transpose(int32, [rank, N]) -> Cast(int64) if graph output.
  const std::string nonzero_out_name = output_name + "_nonzero_out";
  const std::string transpose_out_name = output_name + "_transposed";

  // 1. NonZero output tensor: [num_elements, input_rank], int32, native.
  QnnTensorWrapper nonzero_out_tensor(nonzero_out_name,
                                      QNN_TENSOR_TYPE_NATIVE,
                                      output_data_type,
                                      QnnQuantParamsWrapper(),
                                      std::vector<uint32_t>(qnn_nonzero_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(nonzero_out_tensor)),
                "Failed to add NonZero output tensor.");

  // Create the NonZero QNN node.
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_NON_ZERO,
                                                std::move(input_names),
                                                {nonzero_out_name},
                                                {},
                                                do_op_validation),
                "Failed to add NonZero node.");

  // 2. Transpose: [num_elements, rank] -> [rank, num_elements] to match ONNX spec.
  std::vector<uint32_t> perm = {1, 0};
  QnnParamWrapper transpose_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                                  {static_cast<uint32_t>(perm.size())}, std::vector<uint32_t>(perm));
  std::vector<std::string> transpose_params;
  transpose_params.push_back(transpose_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(transpose_param));

  if (is_graph_output) {
    // Transpose output is native int32, then Cast to int64 for graph output.
    QnnTensorWrapper transpose_out_tensor(transpose_out_name,
                                          QNN_TENSOR_TYPE_NATIVE,
                                          output_data_type,
                                          QnnQuantParamsWrapper(),
                                          std::vector<uint32_t>(onnx_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(transpose_out_tensor)),
                  "Failed to add Transpose output tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(transpose_out_name + "_transpose_node",
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_TRANSPOSE,
                                                  {nonzero_out_name},
                                                  {transpose_out_name},
                                                  std::move(transpose_params),
                                                  do_op_validation),
                  "Failed to add Transpose node.");

    // Cast: int32 -> int64 for graph output.
    QnnTensorWrapper graph_output_tensor(output_name,
                                         QNN_TENSOR_TYPE_APP_READ,
                                         QNN_DATATYPE_INT_64,
                                         QnnQuantParamsWrapper(),
                                         std::vector<uint32_t>(onnx_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(graph_output_tensor)),
                  "Failed to add graph output tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(output_name + "_cast_node",
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_CAST,
                                                  {transpose_out_name},
                                                  {output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add Cast node.");
  } else {
    // Transpose output is the final native tensor.
    QnnTensorWrapper transpose_out_tensor(output_name,
                                          QNN_TENSOR_TYPE_NATIVE,
                                          output_data_type,
                                          QnnQuantParamsWrapper(),
                                          std::vector<uint32_t>(onnx_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(transpose_out_tensor)),
                  "Failed to add Transpose output tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(output_name + "_transpose_node",
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_TRANSPOSE,
                                                  {nonzero_out_name},
                                                  {output_name},
                                                  std::move(transpose_params),
                                                  do_op_validation),
                  "Failed to add Transpose node.");
  }

  return Ort::Status();
}

void CreateNonZeroOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<NonZeroOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
