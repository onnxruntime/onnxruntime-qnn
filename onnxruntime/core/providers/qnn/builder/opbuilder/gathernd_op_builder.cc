// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/normalize_indices_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Op builder for ONNX GatherND (https://onnx.ai/onnx/operators/onnx__GatherND.html).
// ONNX allows negative and/or INT_64 indices; QNN accepts only non-negative INT_32.
// Static indices are normalized at compile time; dynamic INT_64 indices get a Cast.
class GatherNDOpBuilder : public BaseOpBuilder {
 public:
  GatherNDOpBuilder() : BaseOpBuilder("GatherNDOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherNDOpBuilder);

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

namespace {

Ort::Status ProcessGatherNDIndices(QnnModelWrapper& qnn_model_wrapper,
                                   const OrtNodeUnitIODef& indices_input,
                                   const std::vector<uint32_t>& data_shape,
                                   int64_t batch_dims,
                                   const Ort::Logger& logger,
                                   std::vector<std::string>& input_names,
                                   bool do_op_validation) {
  std::string indices_tensor_name = indices_input.name;

  TensorInfo indices_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_input, indices_info));

  const uint32_t index_tuple_size = indices_info.shape.back();
  const auto num_batch_dims = static_cast<size_t>(batch_dims);

  // Column `col` of an index tuple addresses data dim `num_batch_dims + col`.
  const auto axis_dim_for_element =
      [index_tuple_size, num_batch_dims, &data_shape](size_t element_index) -> int64_t {
    const size_t col = element_index % static_cast<size_t>(index_tuple_size);
    return static_cast<int64_t>(data_shape[num_batch_dims + col]);
  };

  std::vector<uint8_t> qnn_indices_bytes;
  bool has_negative_indices = false;

  if (indices_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(indices_info.initializer_tensor,
                                                            onnx_indices_bytes));

    RETURN_IF_NOT(utils::NormalizeIndicesBytes<int64_t>(onnx_indices_bytes, axis_dim_for_element,
                                                        qnn_indices_bytes, has_negative_indices),
                  "QNN does not support out-of-range index values for GatherND.");
    indices_info.qnn_data_type = QNN_DATATYPE_INT_32;

    if (has_negative_indices) {
      indices_tensor_name = utils::UniqueNameGenerator().New(indices_tensor_name, "_qnn_idx");
    }
  }

  return utils::AddNormalizedIndicesTensor(qnn_model_wrapper, std::move(indices_info),
                                           indices_tensor_name, std::move(qnn_indices_bytes),
                                           logger, input_names, do_op_validation);
}

}  // namespace

Ort::Status GatherNDOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const Ort::Logger& logger,
                                             std::vector<std::string>& input_names,
                                             bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  TensorInfo data_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], data_info));

  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t batch_dims = node_helper.Get("batch_dims", static_cast<int64_t>(0));

  return ProcessGatherNDIndices(qnn_model_wrapper, inputs[1], data_info.shape, batch_dims,
                                logger, input_names, do_op_validation);
}

Ort::Status GatherNDOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                           const OrtNodeUnit& node_unit,
                                                           std::vector<std::string>&& input_names,
                                                           const Ort::Logger& logger,
                                                           bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  const auto& output = node_unit.Outputs()[0];
  const std::string& output_name = output.name;

  QnnQuantParamsWrapper quant_params;
  RETURN_IF_ERROR(quant_params.Init(qnn_model_wrapper, output));

  ONNXTensorElementDataType output_type = output.type;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(quant_params.IsQuantized(), output_type, qnn_data_type));

  if (quant_params.IsPerTensor()) {
    // Make sure the output quantization parameters are equal to the input.
    RETURN_IF_ERROR(SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                             0 /*input_index*/, 0 /*output_index*/, qnn_data_type,
                                                             quant_params));
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t batch_dims = node_helper.Get("batch_dims", static_cast<int64_t>(0));

  std::vector<std::string> param_tensor_names;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(batch_dims),
                                         QNN_OP_GATHER_ND_PARAM_BATCH_DIMS, param_tensor_names));

  // Get tensor wrappers for shape calculation
  const auto& data_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);
  const auto& indices_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]);

  const auto& data_dims = data_tensor_wrapper.GetTensorDims();
  const auto& indices_dims = indices_tensor_wrapper.GetTensorDims();

  // ONNX GatherND output shape:
  //   data[:num_batch_dims] ++ indices[:-1] ++ data[num_batch_dims + indices.back():]
  const auto num_batch_dims = static_cast<size_t>(batch_dims);
  const size_t index_tuple_size = indices_dims.back();
  const size_t first_trailing_data_dim = num_batch_dims + index_tuple_size;

  std::vector<uint32_t> qnn_output_shape;

  // Batch dims come from data.
  for (size_t i = 0; i < num_batch_dims && i < data_dims.size(); ++i) {
    qnn_output_shape.push_back(data_dims[i]);
  }
  // All indices dims except the innermost index-tuple dim.
  for (size_t i = 0; i < indices_dims.size() - 1; ++i) {
    qnn_output_shape.push_back(indices_dims[i]);
  }
  // Trailing (un-indexed) data dims.
  for (size_t i = first_trailing_data_dim; i < data_dims.size(); ++i) {
    qnn_output_shape.push_back(data_dims[i]);
  }

  std::vector<uint32_t> target_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.shape, target_output_shape),
                "Cannot get target output shape");

  bool reshape_required = (qnn_output_shape.size() != target_output_shape.size());
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

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
  struct CastNodeInfo {
    std::string node_name;
    std::string input_name;
    std::string output_name;
  };
  std::vector<CastNodeInfo> cast_node_info_vec;

  // Get the output info for the gather output tensor
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, output_info));

  // If a cast to int64 is needed, add the cast node
  if (needs_int64_cast) {
    std::string cast_node_name = utils::UniqueNameGenerator().New(node_unit, "_cast_int64");
    std::string cast_input_name = utils::UniqueNameGenerator().New(output_name, "_cast_int64");
    std::string cast_output_name = output_name;

    // Create the cast input tensor wrapper - use qnn_output_shape for the intermediate tensor
    QnnTensorWrapper cast_input_tensorwrapper(cast_input_name,
                                              QNN_TENSOR_TYPE_NATIVE,
                                              output_info.qnn_data_type,
                                              output_info.quant_param.Copy(),
                                              std::vector<uint32_t>(qnn_output_shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_input_tensorwrapper)), "Failed to add tensor.");
    cast_node_info_vec.push_back({cast_node_name, cast_input_name, cast_output_name});
    Qnn_TensorType_t cast_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper cast_output(output_name, cast_tensor_type, qnn_data_type, quant_params.Copy(),
                                 std::vector<uint32_t>(target_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_output)), "Failed to add tensor.");
  }

  std::string gather_output_name = output_name;
  if (reshape_required) {
    gather_output_name = utils::UniqueNameGenerator().New(output_name, "_reshape");
  } else if (needs_int64_cast) {
    gather_output_name = utils::UniqueNameGenerator().New(output_name, "_cast_int64");
  }

  Qnn_TensorType_t tensor_type = (!reshape_required && is_graph_output)
                                     ? QNN_TENSOR_TYPE_APP_READ
                                     : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper gather_output_tensor(gather_output_name, tensor_type, qnn_data_type,
                                        quant_params.Copy(), std::move(qnn_output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gather_output_tensor)),
                "Failed to add GatherND output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_GATHER_ND,
                                                std::move(input_names),
                                                {gather_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to create GatherND node.");

  if (reshape_required) {
    Qnn_TensorType_t reshape_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper reshape_output(output_name, reshape_tensor_type, qnn_data_type,
                                    std::move(quant_params), std::move(target_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output)), "Failed to add reshape output.");

    std::string node_output_name = output_name;
    if (needs_int64_cast) {
      // If needs_int64 is true, the output name should be the input name of the cast node
      node_output_name = utils::UniqueNameGenerator().New(output_name, "_cast_int64");
    }

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_RESHAPE),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_RESHAPE,
                                                  {gather_output_name},
                                                  {node_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add Reshape node.");
  }

  if (needs_int64_cast) {
    for (const auto& cast_node_info : cast_node_info_vec) {
      // Insert cast node.
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_info.node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CAST,
                                                    {cast_node_info.input_name},
                                                    {cast_node_info.output_name},
                                                    {}),
                    "Failed to add Cast node");
    }
  }

  return Ort::Status();
}

void CreateGatherNDOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GatherNDOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
