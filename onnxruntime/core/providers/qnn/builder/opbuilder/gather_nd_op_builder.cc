// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Handles Gather and GatherElements
class GatherNDOpBuilder : public BaseOpBuilder {
 public:
  GatherNDOpBuilder() : BaseOpBuilder("GatherNDOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherNDOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

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

// Fixes negative indices and converts int64 to int32 for GatherND
template <typename SrcType, typename DstType>
static bool FixStaticIndicesForGatherND(const std::vector<uint8_t>& onnx_bytes,
                                        const std::vector<int64_t>& input_shape,
                                        int64_t index_tuple_size,
                                        /*out*/ std::vector<uint8_t>& qnn_bytes) {
  const size_t num_indices = onnx_bytes.size() / sizeof(SrcType);
  gsl::span<const SrcType> onnx_indices{reinterpret_cast<const SrcType*>(onnx_bytes.data()), num_indices};

  qnn_bytes.resize(num_indices * sizeof(DstType));
  gsl::span<DstType> qnn_indices{reinterpret_cast<DstType*>(qnn_bytes.data()), num_indices};

  const size_t num_tuples = num_indices / index_tuple_size;

  for (size_t t = 0; t < num_tuples; ++t) {
    for (int64_t d = 0; d < index_tuple_size; ++d) {
      size_t idx = t * index_tuple_size + d;
      SrcType index = onnx_indices[idx];

      // Fix negative index
      if (index < 0) {
        index += static_cast<SrcType>(input_shape[d]);
      }

      // Bounds check
      if (index < 0 || static_cast<int64_t>(index) >= input_shape[d]) {
        return false;  // Out-of-bounds index
      }

      qnn_indices[idx] = static_cast<DstType>(index);
    }
  }

  return true;
}

Status GatherNDOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                        const NodeUnit& node_unit,
                                        const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(logger);
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF(inputs.size() != 2, "GatherND must have exactly 2 inputs.");
  return Status::OK();
}

Status GatherNDOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                        const NodeUnit& node_unit,
                                        const logging::Logger& logger,
                                        std::vector<std::string>& input_names,
                                        bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF(inputs.size() != 2, "GatherND must have exactly 2 inputs.");

  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  const auto& data_input = inputs[0];
  const auto& indices_input = inputs[1];
  const auto& input_name = indices_input.node_arg.Name();
  TensorInfo indices_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_input, indices_info));

  std::vector<uint8_t> qnn_indices_bytes;

  if (indices_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*indices_info.initializer_tensor, onnx_indices_bytes));

    std::vector<uint32_t> data_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(data_input.node_arg, data_shape),
                      "Failed to get data shape for GatherND.");

    std::vector<uint32_t> indices_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(indices_input.node_arg, indices_shape),
                      "Failed to get indices shape for GatherND.");

    if (indices_shape.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Indices shape is empty for GatherND.");
    }

    // Check input rank constraints
    if (indices_info.qnn_data_type == QNN_DATATYPE_FLOAT_16 && data_shape.size() > 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "FP16 input rank must be 4 or less for GatherND.");
    }
    if ((indices_info.qnn_data_type == QNN_DATATYPE_INT_16 ||
         indices_info.qnn_data_type == QNN_DATATYPE_INT_8 ||
         indices_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_8 ||
         indices_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_8) &&
        data_shape.size() > 5) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "INT16, INT8, SFIXED_POINT_8, UFIXED_POINT_8 input rank must be 5 or less for GatherND.");
    }

    // Check indices rank constraints
    if (indices_shape.size() > 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Indices rank must be 4 or less for GatherND.");
    }
    int64_t index_tuple_size = static_cast<int64_t>(indices_shape.back());

    if (indices_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      ORT_RETURN_IF_NOT((
          FixStaticIndicesForGatherND<int64_t, int32_t>(
              onnx_indices_bytes,
              std::vector<int64_t>(data_shape.begin(), data_shape.begin() + index_tuple_size),
              index_tuple_size,
              qnn_indices_bytes),
          "QNN does not support negative indices for GatherND."));
      indices_info.qnn_data_type = QNN_DATATYPE_INT_32;
    }
  }

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(indices_input.node_arg.Name());
  std::vector<uint32_t> cast_output_shape(indices_info.shape);

  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(indices_input.node_arg.Name())) {
    QnnTensorWrapper input_tensorwrapper(indices_input.node_arg.Name(), tensor_type, indices_info.qnn_data_type,
                                         QnnQuantParamsWrapper(), std::move(indices_info.shape),
                                         std::move(qnn_indices_bytes));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  std::string indices_input_name = indices_input.node_arg.Name();
  if (indices_info.qnn_data_type == QNN_DATATYPE_INT_64) {
    assert(!indices_info.is_initializer);
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddInt64CastNode(input_name, indices_input_name,
                                                           std::move(cast_output_shape),
                                                           do_op_validation));
  }

  input_names.push_back(indices_input_name);

  return Status::OK();
}

Status GatherNDOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const NodeUnit& node_unit,
                                                      std::vector<std::string>&& input_names,
                                                      const logging::Logger& logger,
                                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  const auto& output = node_unit.Outputs()[0];
  const std::string& output_name = output.node_arg.Name();

  QnnQuantParamsWrapper quant_params;
  ORT_RETURN_IF_ERROR(quant_params.Init(qnn_model_wrapper, output));

  const auto* type_proto = output.node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(quant_params.IsQuantized(), type_proto, qnn_data_type));

  if (quant_params.IsPerTensor()) {
    // Make sure the output quantization parameters are equal to the input.
    ORT_RETURN_IF_ERROR(SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                                 0 /*input_index*/, 0 /*output_index*/, qnn_data_type,
                                                                 quant_params));
  }

  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, output_shape),
                    "Failed to get output shape for GatherND.");

  NodeAttrHelper node_helper(node_unit);
  int32_t batch_dims = static_cast<int32_t>(node_helper.Get("batch_dims", 0));

  Qnn_Scalar_t batch_dims_scalar = QNN_SCALAR_INIT;
  batch_dims_scalar.dataType = QNN_DATATYPE_INT_32;
  batch_dims_scalar.int32Value = batch_dims;

  QnnParamWrapper batch_dims_param(node_unit.Index(), node_unit.Name(),
                                   QNN_OP_GATHER_ND_PARAM_BATCH_DIMS, batch_dims_scalar);
  std::vector<std::string> param_tensor_names = {batch_dims_param.GetParamTensorName()};
  qnn_model_wrapper.AddParamWrapper(std::move(batch_dims_param));

  std::vector<uint32_t> target_output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, target_output_shape),
                    "Cannot get target output shape");

  bool reshape_required = (output_shape != target_output_shape);
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  std::string gather_output_name = output_name;
  if (reshape_required) {
    gather_output_name += "_ort_qnn_ep_reshape";
  }

  Qnn_TensorType_t tensor_type = (!reshape_required && is_graph_output)
                                     ? QNN_TENSOR_TYPE_APP_READ
                                     : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper gather_output_tensor(gather_output_name, tensor_type, qnn_data_type,
                                        quant_params.Copy(), std::move(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gather_output_tensor)),
                    "Failed to add GatherNd output tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_GATHER_ND,
                                                    std::move(input_names),
                                                    {gather_output_name},
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to create GatherNd node.");

  if (reshape_required) {
    Qnn_TensorType_t reshape_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper reshape_output(output_name, reshape_tensor_type, qnn_data_type,
                                    std::move(quant_params), std::move(target_output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output)), "Failed to add reshape output.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(output_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      "Reshape",
                                                      {gather_output_name},
                                                      {output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add Reshape node.");
  }

  return Status::OK();
}

void CreateGatherNDOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GatherNDOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime