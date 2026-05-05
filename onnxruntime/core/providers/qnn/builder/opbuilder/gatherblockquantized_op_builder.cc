// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class GatherBlockQuantizedOpBuilder : public BaseOpBuilder {
 public:
  GatherBlockQuantizedOpBuilder() : BaseOpBuilder("GatherBlockQuantizedOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherBlockQuantizedOpBuilder);

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

Ort::Status GatherBlockQuantizedOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         const Ort::Logger& logger) const {
  RETURN_IF_NOT(IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()),
                "GatherBlockQuantized is supported only for QNN HTP backend.");

  // Extract Parameters
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t bits = node_helper.Get("bits", static_cast<int64_t>(4));
  const int64_t block_size = node_helper.Get("block_size", static_cast<int64_t>(32));
  const int64_t gather_axis = node_helper.Get("gather_axis", static_cast<int64_t>(0));
  const int64_t quantize_axis = node_helper.Get("quantize_axis", static_cast<int64_t>(1));

  RETURN_IF_NOT(bits == 4, "QNN HTP only support GatherBlockQuantized with bits=4.");
  RETURN_IF_NOT(block_size == 32, "QNN HTP only support GatherBlockQuantized with block_size=32.");
  RETURN_IF_NOT(gather_axis == 0, "QNN HTP only support GatherBlockQuantized with gather_axis=0.");
  RETURN_IF_NOT(quantize_axis == 1, "QNN HTP only support GatherBlockQuantized with quantize_axis=1.");

  const auto& inputs = node_unit.Inputs();

  // 1. data: Input dtype must be UINT4 and shape must be 2D.
  {
    const OrtNodeUnitIODef& input_tensor = inputs[0];
    RETURN_IF(input_tensor.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
              "Unsupported input data dtype, expeceting UINT4.");

    TensorInfo input_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_tensor, input_info));
    RETURN_IF(input_info.shape.size() != 2, "Unsupported input data rank, expecting 2D.");
  }

  // 2. indices: Input shape must be 2D.
  {
    const OrtNodeUnitIODef& input_tensor = inputs[1];
    TensorInfo input_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_tensor, input_info));
    RETURN_IF(input_info.shape.size() != 2, "Unsupported input indices rank, expecting 2D.");
  }

  // 3. scales: Input dtype must be float32.
  {
    RETURN_IF(inputs[2].type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
              "Unsupported input scales dtype, expeceting FLOAT.");
  }

  // 3. zero_points: Input must exist and dtype must be float32.
  {
    RETURN_IF(inputs.size() < 4 || !inputs[3].Exists(), "Unsupported optional input zero_points.");
    RETURN_IF(inputs[3].type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
              "Unsupported input zero_points dtype, expeceting UINT4.");
  }

  // Validate Process
  std::vector<std::string> input_names;
  RETURN_IF_ERROR(ProcessInputs(qnn_model_wrapper, node_unit, logger, input_names, true));
  RETURN_IF_ERROR(ProcessAttributesAndOutputs(qnn_model_wrapper, node_unit, std::move(input_names), logger, true));

  return Ort::Status();
}

Ort::Status GatherBlockQuantizedOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         const Ort::Logger& logger,
                                                         std::vector<std::string>& input_names,
                                                         bool do_op_validation) const {
  // Extract Parameters
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t block_size = node_helper.Get("block_size", static_cast<int64_t>(128));

  const auto& inputs = node_unit.Inputs();

  // 1. Add input indices.
  {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[1], logger, input_names));

    TensorInfo input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input_info));

    // 1-1. Add graph input Cast to cast int64 to int32 to pass HTP validation.
    const std::string cast_int32_output_name = utils::UniqueNameGenerator().New(input_names[0], "_cast_int32");
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::UniqueNameGenerator().New(node_unit, "_cast_int32"),
                                                  input_names[0],
                                                  cast_int32_output_name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  QNN_DATATYPE_INT_32,
                                                  input_info.quant_param.Copy(),
                                                  std::vector<uint32_t>(input_info.shape),
                                                  do_op_validation));

    // 1-2. Add OneHot to prepare for the first input of Conv2d.
    const std::string onehot_name = utils::UniqueNameGenerator().New(node_unit, "_onehot");

    // Add OneHot parameters.
    std::vector<std::string> param_tensor_names;
    TensorInfo data_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], data_info));
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper,
                                           node_unit.Index(),
                                           onehot_name,
                                           data_info.shape[0],
                                           QNN_OP_ONE_HOT_PARAM_DEPTH,
                                           param_tensor_names));
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper,
                                           node_unit.Index(),
                                           onehot_name,
                                           static_cast<uint32_t>(data_info.shape.size()),
                                           QNN_OP_ONE_HOT_PARAM_AXIS,
                                           param_tensor_names));

    // Add OneHot output tensor.
    // Input indices having 2D shape is guaranteed in the above check.
    std::vector<uint32_t> onehot_output_shape = {input_info.shape[0], input_info.shape[1], data_info.shape[0]};

    const std::string onehot_output_name = utils::UniqueNameGenerator().New(input_names[0], "_onehot");
    QnnTensorWrapper onehot_output_tensor_wrapper(onehot_output_name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  QNN_DATATYPE_FLOAT_32,
                                                  input_info.quant_param.Copy(),
                                                  std::vector<uint32_t>(onehot_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(onehot_output_tensor_wrapper)),
                  "Failed to add OneHot output tensor.");

    // Add OneHot node.
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(onehot_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ONE_HOT,
                                                  {cast_int32_output_name},
                                                  {onehot_output_name},
                                                  std::move(param_tensor_names),
                                                  do_op_validation),
                  "Failed to add OneHot node.");

    // 1-2. Add pre-Reshape to unsqueeze shape to 4D for Conv2d.
    std::vector<uint32_t> reshape_output_shape = {input_info.shape[0], 1, input_info.shape[1], data_info.shape[0]};

    const std::string reshape_output_name = utils::UniqueNameGenerator().New(input_names[0], "_reshape_4d");
    QnnTensorWrapper reshape_output_tensor_wrapper(reshape_output_name,
                                                   QNN_TENSOR_TYPE_NATIVE,
                                                   QNN_DATATYPE_FLOAT_32,
                                                   input_info.quant_param.Copy(),
                                                   std::vector<uint32_t>(reshape_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output_tensor_wrapper)),
                  "Failed to add pre-Reshape output tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_reshape_4d"),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_RESHAPE,
                                                  {onehot_output_name},
                                                  {reshape_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add pre-Reshape node.");

    // 1-3. Add pre-Cast to cast float32 to float16 to pass HTP validation.
    const std::string cast_fp16_output_name = utils::UniqueNameGenerator().New(input_names[0], "_cast_fp16");
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::UniqueNameGenerator().New(node_unit, "_cast_fp16"),
                                                  reshape_output_name,
                                                  cast_fp16_output_name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  QNN_DATATYPE_FLOAT_16,
                                                  input_info.quant_param.Copy(),
                                                  std::vector<uint32_t>(reshape_output_shape),
                                                  do_op_validation));

    // Reroute to pre-Cast output.
    input_names[0] = cast_fp16_output_name;
  }

  // 2. Add input data and corresponding quantization parameters.
  {
    const auto& weight_tensor = inputs[0];
    const auto& scales_tensor = inputs[2];
    const auto& zero_points_tensor = inputs[3];

    const auto& weight_tensor_name = weight_tensor.name;
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(weight_tensor_name)) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + weight_tensor_name).c_str());
    } else {
      // 2.1 Block-quantized data.
      std::vector<uint8_t> quant_data;
      const OrtValueInfo* weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(weight_tensor_proto, quant_data, true));

      // Transform data from unsigned to signed fixed point.
      {
        const int32_t zero_point = 8;
        const uint32_t mask = 15;

        for (size_t idx = 0; idx < quant_data.size(); ++idx) {
          uint32_t unsigned_value = static_cast<uint32_t>(quant_data[idx]) & mask;
          int32_t signed_value = static_cast<int32_t>(unsigned_value) - zero_point;
          quant_data[idx] = static_cast<uint8_t>(static_cast<uint32_t>(signed_value) & mask);
        }
      }

      // 2.2 Block-quantized scales.
      std::vector<uint8_t> per_block_uint8_scale;
      const OrtValueInfo* scale_tensor_proto = qnn_model_wrapper.GetConstantTensor(scales_tensor.name);
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(scale_tensor_proto, per_block_uint8_scale));

      TensorInfo scales_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], scales_info));
      const int64_t total_blocks = scales_info.shape[0] * scales_info.shape[1];

      float* per_block_float_scale_ptr = reinterpret_cast<float*>(per_block_uint8_scale.data());
      const std::vector<float> per_block_float_scale(per_block_float_scale_ptr,
                                                     per_block_float_scale_ptr + total_blocks);

      // 2.3 Block-quantized offsets.
      std::vector<uint8_t> per_block_uint8_zp;
      const OrtValueInfo* zp_tensor_proto = qnn_model_wrapper.GetConstantTensor(zero_points_tensor.name);
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(zp_tensor_proto, per_block_uint8_zp, true));

      std::vector<float> per_block_float_zp(per_block_uint8_zp.size(), 0.0);
      for (const uint8_t& uint8_zp : per_block_uint8_zp) {
        per_block_float_zp.push_back(static_cast<float>(uint8_zp));
      }

      // 2.4 Create QNN wrappers.
      const std::vector<uint32_t> block_sizes = {1, 1, gsl::narrow_cast<uint32_t>(block_size), 1};
      QnnQuantParamsWrapper quantize_param = QnnQuantParamsWrapper(per_block_float_scale,
                                                                    per_block_float_zp,
                                                                    gsl::narrow_cast<uint32_t>(4),
                                                                    block_sizes);

      // Shape is for Conv2d, expecting in HWIO.
      TensorInfo weight_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], weight_info));
      QnnTensorWrapper weight_tensor_wrapper(weight_tensor_name,
                                              QNN_TENSOR_TYPE_STATIC,
                                              // HTP will derive the actual data type from quant param.
                                              QNN_DATATYPE_SFIXED_POINT_8,
                                              std::move(quantize_param),
                                              {1, 1, weight_info.shape[0], weight_info.shape[1]},
                                              std::move(quant_data));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor_wrapper)),
                    "Failed to add weight tensor.");

      input_names.push_back(weight_tensor_name);
    }
  }

  return Ort::Status();
}


Ort::Status GatherBlockQuantizedOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                       const OrtNodeUnit& node_unit,
                                                                       std::vector<std::string>&& input_names,
                                                                       const Ort::Logger& /*logger*/,
                                                                       bool do_op_validation) const {
  const OrtNodeUnitIODef& output_tensor = node_unit.Outputs()[0];
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_tensor, output_info));

  // 1. Add Conv2d with default stride/pad amount.
  std::vector<std::string> param_tensor_names;

  std::vector<uint32_t> stride = {1, 1};
  QnnParamWrapper stride_param_wrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        QNN_OP_CONV_2D_PARAM_STRIDE,
                                        {2},
                                        std::move(stride));
  param_tensor_names.push_back(stride_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(stride_param_wrapper));

  std::vector<uint32_t> pad_amount = {0, 0, 0, 0};
  QnnParamWrapper pad_amount_param_wrapper(node_unit.Index(),
                                            node_unit.Name(),
                                            QNN_OP_CONV_2D_PARAM_PAD_AMOUNT,
                                            {2, 2},
                                            std::move(pad_amount));
  param_tensor_names.push_back(pad_amount_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_param_wrapper));

  const std::string conv2d_output_name = utils::UniqueNameGenerator().New(output_tensor.name, "_conv2d");
  std::vector<uint32_t> conv2d_output_shape = {output_info.shape[0], 1, output_info.shape[1], output_info.shape[2]};
  QnnTensorWrapper conv2d_output_tensor_wrapper(conv2d_output_name,
                                                QNN_TENSOR_TYPE_NATIVE,
                                                QNN_DATATYPE_FLOAT_16 /*output_info.qnn_data_type*/,
                                                output_info.quant_param.Copy(),
                                                std::vector<uint32_t>(conv2d_output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(conv2d_output_tensor_wrapper)),
                "Failed to add Conv2d output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_CONV_2D,
                                                std::move(input_names),
                                                {conv2d_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to add Conv2d node.");

  // 2. Add post-Cast to cast float16 back to float32 to pass HTP validation.
  const std::string cast_output_name = utils::UniqueNameGenerator().New(conv2d_output_name, "_cast_fp32");
  RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::UniqueNameGenerator().New(node_unit, "_cast_fp32"),
                                                conv2d_output_name,
                                                cast_output_name,
                                                QNN_TENSOR_TYPE_NATIVE,
                                                QNN_DATATYPE_FLOAT_32,
                                                output_info.quant_param.Copy(),
                                                std::vector<uint32_t>(conv2d_output_shape),
                                                do_op_validation));

  // 3. Add post-Reshape to squeeze shape back.
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_tensor.name);
  QnnTensorWrapper reshape_output_tensor_wrapper(output_tensor.name,
                                                  is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                                  output_info.qnn_data_type,
                                                  output_info.quant_param.Copy(),
                                                  std::vector<uint32_t>(output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output_tensor_wrapper)),
                "Failed to add post-Reshape output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_reshape_3d"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {cast_output_name},
                                                {output_tensor.name},
                                                {},
                                                do_op_validation),
                "Failed to add post-Reshape node.");

  return Ort::Status();
}

void CreateGatherBlockQuantizedOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GatherBlockQuantizedOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
