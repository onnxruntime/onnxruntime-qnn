// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include <numeric>

namespace onnxruntime {
namespace qnn {

/*
  [COMMENT BLOCK UNCHANGED – omitted here for brevity]
*/

class GatherBlockQuantizedOpBuilder : public BaseOpBuilder {
 public:
  GatherBlockQuantizedOpBuilder()
      : BaseOpBuilder("GatherBlockQuantizedOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherBlockQuantizedOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override;

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override;

 private:
  void ToSignedFixedPoint4(std::vector<uint8_t>& quant_data,
                           int64_t num_blocks,
                           int64_t block_size) const;
};

// ================================================================
// uint4 → signed int4
// ================================================================
void GatherBlockQuantizedOpBuilder::ToSignedFixedPoint4(
    std::vector<uint8_t>& quant_data,
    int64_t num_blocks,
    int64_t block_size) const {
  constexpr uint8_t zero_point = 8;

  for (int64_t b = 0; b < num_blocks; ++b) {
    for (int64_t i = 0; i < block_size / 2; ++i) {
      size_t idx = static_cast<size_t>(b * (block_size / 2) + i);
      uint8_t v = quant_data[idx];

      int8_t hi = ((v >> 4) & 0xF) - zero_point;
      int8_t lo = (v & 0xF) - zero_point;

      quant_data[idx] =
          static_cast<uint8_t>(((hi & 0xF) << 4) | (lo & 0xF));
    }
  }
}

// ================================================================
// IsOpSupported
// ================================================================
Ort::Status GatherBlockQuantizedOpBuilder::IsOpSupported(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    const Ort::Logger&) const {
  RETURN_IF_NOT(IsGpuBackend(qnn_model_wrapper.GetQnnBackendType()),
                "GatherBlockQuantized: GPU backend only");

  OrtNodeAttrHelper helper(node_unit);
  const int64_t bits = helper.Get("bits", static_cast<int64_t>(4));
  const int64_t block_size = helper.Get("block_size", static_cast<int64_t>(32));

  RETURN_IF_NOT(bits == 4,
                "GatherBlockQuantized: only INT4 (bits == 4) supported");
  RETURN_IF_NOT(block_size >= 16 && ((block_size & (block_size - 1)) == 0),
                "GatherBlockQuantized: block_size must be power of 2 and >= 16");

  // Validate scales datatype (must be float32)
  const auto& inputs = node_unit.Inputs();
  {
    Qnn_DataType_t weight_datatype;
    const OrtNodeUnitIODef& weight_tensor = inputs[0];
    TensorInfo weights_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(weight_tensor, weights_info, true));
    RETURN_IF_ERROR(utils::GetQnnDataType(
        weight_tensor.quant_param.has_value(),
        weight_tensor.type,
        weight_datatype,
        true));
    RETURN_IF((weight_datatype != QNN_DATATYPE_UINT_8) && (weight_datatype != QNN_DATATYPE_SFIXED_POINT_4),
              "GatherBlockQuantized: weights must be UINT_8 or SFIXED_POINT_4");
  }

  {
    Qnn_DataType_t indices_datatype;
    const OrtNodeUnitIODef& indices_tensor = inputs[1];
    TensorInfo indices_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_tensor, indices_info));
    RETURN_IF_ERROR(utils::GetQnnDataType(
        indices_tensor.quant_param.has_value(),
        indices_tensor.type,
        indices_datatype,
        true));
    RETURN_IF(indices_datatype != QNN_DATATYPE_INT_64,
              "GatherBlockQuantized: indices must be INT_64");
  }

  {
    Qnn_DataType_t scale_datatype;
    const OrtNodeUnitIODef& scales_tensor = inputs[2];
    TensorInfo scales_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(scales_tensor, scales_info));
    RETURN_IF_ERROR(utils::GetQnnDataType(
        scales_tensor.quant_param.has_value(),
        scales_tensor.type,
        scale_datatype,
        true));
    RETURN_IF(scale_datatype != QNN_DATATYPE_FLOAT_32,
              "GatherBlockQuantized: scales must be FLOAT32");
  }
  return Ort::Status();
}

// ================================================================
// ProcessInputs
// ================================================================
Ort::Status GatherBlockQuantizedOpBuilder::ProcessInputs(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    const Ort::Logger& logger,
    std::vector<std::string>& input_names,
    bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_NOT(IsGpuBackend(qnn_model_wrapper.GetQnnBackendType()),
                  "GatherBlockQuantized: GPU backend only");
  }

  // ------------------------------------------------------------
  // 1. Weights and scales
  // ------------------------------------------------------------
  const auto& inputs = node_unit.Inputs();

  // Get weight info
  const auto& weight_tensor = inputs[0];
  TensorInfo weight_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(weight_tensor, weight_info, true));
  Qnn_DataType_t weight_type = weight_info.qnn_data_type;
  std::vector<uint32_t> weight_shape = weight_info.shape;

  // Get scale info
  const auto& scales_tensor = inputs[2];
  TensorInfo scale_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(scales_tensor, scale_info));
  std::vector<uint32_t> scale_shape = scale_info.shape;

  // Required params
  OrtNodeAttrHelper helper(node_unit);
  const int64_t block_size = helper.Get("block_size", static_cast<int64_t>(32));
  const int64_t num_blocks = std::accumulate(scale_shape.begin(),
                                             scale_shape.end(),
                                             int64_t{1},
                                             std::multiplies<int64_t>());
  const std::vector<uint32_t> block_sizes = {1, gsl::narrow_cast<uint32_t>(block_size)};

  // Creating weight+scale wrapper
  const std::string& weight_tensor_name = weight_tensor.name;
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(weight_tensor_name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + weight_tensor_name).c_str());
  } else {
    // Unpack weights
    std::vector<uint8_t> quant_data;
    Qnn_TensorType_t weight_tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
    const OrtValueInfo* weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(weight_tensor_proto, quant_data, false));

    // Transform quantized weights to signed fixed point 4.
    bool needs_uint4_to_int4 = (weight_type == QNN_DATATYPE_UINT_8);
    if (needs_uint4_to_int4) {
      ToSignedFixedPoint4(quant_data, num_blocks, block_size);
    }

    // Unpack scales
    std::vector<uint8_t> uint8_scale;
    const OrtValueInfo* scale_tensor_proto = qnn_model_wrapper.GetConstantTensor(scales_tensor.name);
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(scale_tensor_proto, uint8_scale, false));

    float* float_scale_ptr = reinterpret_cast<float*>(uint8_scale.data());
    const std::vector<float> float_scale(float_scale_ptr, float_scale_ptr + num_blocks);

    // Quantization Offsets : QNN Support only symmetric quantization with default value of 0
    std::vector<int32_t> int32_offset(num_blocks, 0);

    // Create Quantization Parameter and create Weight Tensor
    QnnQuantParamsWrapper quantize_param = QnnQuantParamsWrapper(float_scale,
                                                                 int32_offset,
                                                                 block_sizes,
                                                                 QNN_DATATYPE_SFIXED_POINT_4);
    weight_shape[1] = static_cast<uint32_t>(scale_shape[1] * block_size);
    std::vector<uint32_t> weight_shape_ = {static_cast<uint32_t>(weight_shape[0]), static_cast<uint32_t>(scale_shape[1] * block_size)};
    QnnTensorWrapper weight_tensor_wrapper(weight_tensor_name,
                                           weight_tensor_type,
                                           QNN_DATATYPE_SFIXED_POINT_4,
                                           std::move(quantize_param),
                                           std::move(weight_shape_),
                                           std::move(quant_data));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor_wrapper)), "Failed to add tensor.");
  }
  input_names.push_back(weight_tensor_name);

  // ------------------------------------------------------------
  // 2. Indices
  // ------------------------------------------------------------
  // Creating indices wrapper
  const OrtNodeUnitIODef& indices_tensor = inputs[1];
  const std::string& name = indices_tensor.name;

  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(name)) {
    TensorInfo info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_tensor, info));
    QnnTensorWrapper wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(info, name, wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(wrapper)),
                  "Failed to add indices tensor");
  }
  input_names.push_back(name);
  return Ort::Status();
}

// ================================================================
// ProcessAttributesAndOutputs
// ================================================================
Ort::Status GatherBlockQuantizedOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                       const OrtNodeUnit& node_unit,
                                                                       std::vector<std::string>&& input_names,
                                                                       const Ort::Logger& logger,
                                                                       bool do_op_validation) const {
  if (do_op_validation) {
    bool is_gpu_backend = IsGpuBackend(qnn_model_wrapper.GetQnnBackendType());
    RETURN_IF_NOT(is_gpu_backend, "MatMulNBits Op Supported Only for Qnn Gpu Backend");
  }
  OrtNodeAttrHelper helper(node_unit);
  const int64_t axis_attr = helper.Get("gather_axis", 0);

  // Output info
  const OrtNodeUnitIODef& output_tensor = node_unit.Outputs()[0];
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_tensor, output_info));

  // Creating output wrapper
  const std::string& output_tensor_name = output_tensor.name;
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(output_tensor_name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + output_tensor_name).c_str());
  } else {
    QnnTensorWrapper output_tensor_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_tensor, output_tensor_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)), "Failed to add output");
  }

  // Creating axis param wrapper
  std::vector<std::string> param_tensor_names;
  int32_t axis_value = static_cast<int32_t>(axis_attr);
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  RETURN_IF_ERROR(
      ProcessAxisAttribute(qnn_model_wrapper,
                           node_unit,
                           axis_qnn_scalar,
                           axis_value));

  QnnParamWrapper axis_param(
      node_unit.Index(),
      node_unit.Name(),
      QNN_OP_GATHER_PARAM_AXIS,
      axis_qnn_scalar);

  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  // Creating Qnn node
  RETURN_IF_NOT(
      qnn_model_wrapper.CreateQnnNode(
          output_tensor_name,
          QNN_OP_PACKAGE_NAME_QTI_AISW,
          QNN_OP_GATHER,
          std::move(input_names),
          {output_tensor_name},
          std::move(param_tensor_names),
          do_op_validation),
      "Failed to create Gather node");
  return Ort::Status();
}

// ================================================================
// Registration
// ================================================================
void CreateGatherBlockQuantizedOpBuilder(
    const std::string& op_type,
    OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(
      op_type,
      std::make_unique<GatherBlockQuantizedOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
