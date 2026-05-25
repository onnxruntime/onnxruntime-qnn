// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

class LpPoolOpBuilder : public BaseOpBuilder {
 public:
  LpPoolOpBuilder() : BaseOpBuilder("LpPoolOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LpPoolOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status LpPoolOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].type, ""));

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape");

  const size_t rank = input_shape.size();
  RETURN_IF_NOT(rank == 3 || rank == 4, "QNN LpPool only supports rank 3 or 4!");

  RETURN_IF(node_unit.Outputs().size() > 1, "QNN LpPool only supports 1 output!");

  OrtNodeAttrHelper node_helper(node_unit);

  const auto p = node_helper.Get("p", static_cast<int64_t>(2));
  RETURN_IF(p != 2, "QNN LpPool only supports p=2 (L2 norm)!");

  const auto ceil_mode = node_helper.Get("ceil_mode", static_cast<int64_t>(0));
  RETURN_IF(ceil_mode != 0, "QNN LpPool does not support ceil_mode=1!");

  const auto dilations = node_helper.Get("dilations", std::vector<uint32_t>(rank - 2, 1));
  RETURN_IF_NOT(dilations == std::vector<uint32_t>(rank - 2, 1), "QNN LpPool does not support dilations > 1!");

  const auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER" && auto_pad != "VALID",
            ("QNN LpPool does not support 'auto_pad' value: " + auto_pad).c_str());

  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  return Ort::Status();
}

Ort::Status LpPoolOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& logger,
                                                         bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();

  std::vector<uint32_t> onnx_input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, onnx_input_shape), "Cannot get shape");

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));

  const bool requires_rank3_reshape = (onnx_input_shape.size() == 3);
  std::vector<uint32_t> qnn_input_shape = onnx_input_shape;

  std::vector<uint32_t> onnx_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Outputs()[0].shape, onnx_output_shape),
                "Cannot get shape");
  std::vector<uint32_t> qnn_output_shape = onnx_output_shape;
  std::vector<uint32_t> intermediate_output_shape_4d;

  if (requires_rank3_reshape) {
    qnn_input_shape = {onnx_input_shape[0], 1, onnx_input_shape[1], onnx_input_shape[2]};
    qnn_output_shape = {onnx_output_shape[0], 1, onnx_output_shape[1], onnx_output_shape[2]};
    intermediate_output_shape_4d = qnn_output_shape;

    const std::string reshaped_input_name = utils::UniqueNameGenerator().New(input_names[0], "_reshape");
    QnnTensorWrapper reshaped_input_tensor(reshaped_input_name,
                                           QNN_TENSOR_TYPE_NATIVE,
                                           input_info.qnn_data_type,
                                           input_info.quant_param.Copy(),
                                           std::vector<uint32_t>(qnn_input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshaped_input_tensor)),
                  "Failed to add reshape prior tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_RESHAPE),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
                                                  {input_names[0]}, {reshaped_input_name}, {}, do_op_validation),
                  "Failed to create reshape prior node for LpPool.");
    input_names[0] = reshaped_input_name;
  }

  const size_t rank = qnn_input_shape.size();

  // kernel_shape is required by ONNX LpPool spec.
  std::vector<uint32_t> filter_size;
  {
    auto raw = node_helper.Get("kernel_shape", std::vector<uint32_t>(rank - 2, 1));
    filter_size = (raw.size() == 1) ? std::vector<uint32_t>{1, raw[0]} : raw;
  }

  std::vector<uint32_t> stride;
  {
    auto raw = node_helper.Get("strides", std::vector<uint32_t>(rank - 2, 1));
    stride = (raw.size() == 1) ? std::vector<uint32_t>{1, raw[0]} : raw;
  }

  // Dilations are validated as all-1 in IsOpSupported; needed only for auto_pad pad calculation.
  std::vector<uint32_t> dilations;
  {
    auto raw = node_helper.Get("dilations", std::vector<uint32_t>(rank - 2, 1));
    dilations = (raw.size() == 1) ? std::vector<uint32_t>{1, raw[0]} : raw;
  }

  // pads in ONNX format: [h_begin, w_begin, h_end, w_end]
  std::vector<uint32_t> pad_amount;
  {
    auto raw = node_helper.Get("pads", std::vector<uint32_t>((rank - 2) * 2, 0));
    pad_amount = (raw.size() == 2) ? std::vector<uint32_t>{0, raw[0], 0, raw[1]} : raw;
  }

  // Derive explicit pads from auto_pad when set.
  const auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  if (auto_pad != "NOTSET") {
    for (size_t axis = 0; axis < rank - 2; ++axis) {
      // VALID leaves pad_amount as zero; only SAME_UPPER / SAME_LOWER require computation.
      if (auto_pad == "SAME_LOWER" || auto_pad == "SAME_UPPER") {
        uint32_t total_pads = (qnn_output_shape[axis + 1] - 1) * stride[axis] +
                              (filter_size[axis] - 1) * dilations[axis] + 1 - qnn_input_shape[axis + 1];
        if (auto_pad == "SAME_LOWER") {
          pad_amount[axis + rank - 2] = total_pads / 2;
          pad_amount[axis] = total_pads - pad_amount[axis + rank - 2];
        } else {
          pad_amount[axis] = total_pads / 2;
          pad_amount[axis + rank - 2] = total_pads - pad_amount[axis];
        }
      }
    }
  }

  // Convert from ONNX format [h_begin, w_begin, h_end, w_end]
  // to QNN format [h_begin, h_end, w_begin, w_end].
  ReArrangePads(pad_amount);

  // QNN L2Pool2d computes RMS (normalized): sqrt(sum(x^2) / N).
  // ONNX LpPool (p=2) computes L2 norm: sqrt(sum(x^2)).
  // Correction: multiply QNN output by sqrt(filter_H * filter_W) to match ONNX semantics.
  const float scale_val = std::sqrt(static_cast<float>(filter_size[0] * filter_size[1]));

  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  // pool_out_shape is always rank-4 at this point (rank-3 inputs were reshaped above).
  const std::vector<uint32_t> pool_out_shape =
      requires_rank3_reshape ? intermediate_output_shape_4d
                             : std::vector<uint32_t>(output_info.shape);

  std::vector<std::string> param_tensor_names;

  {
    QnnParamWrapper filter_size_param(node_unit.Index(), node_unit.Name(),
                                      QNN_OP_L2_POOL_2D_PARAM_FILTER_SIZE,
                                      {static_cast<uint32_t>(filter_size.size())},
                                      std::move(filter_size));
    param_tensor_names.push_back(filter_size_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(filter_size_param)),
                  "Failed to add param filter_size.");
  }

  {
    QnnParamWrapper stride_param(node_unit.Index(), node_unit.Name(),
                                 QNN_OP_L2_POOL_2D_PARAM_STRIDE,
                                 {static_cast<uint32_t>(stride.size())},
                                 std::move(stride));
    param_tensor_names.push_back(stride_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(stride_param)),
                  "Failed to add param stride.");
  }

  {
    QnnParamWrapper pad_amount_param(node_unit.Index(), node_unit.Name(),
                                     QNN_OP_L2_POOL_2D_PARAM_PAD_AMOUNT,
                                     {static_cast<uint32_t>(pad_amount.size() / 2), 2},
                                     std::move(pad_amount));
    param_tensor_names.push_back(pad_amount_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_param)),
                  "Failed to add param pad_amount.");
  }

  // L2Pool2d always writes to an intermediate tensor; the scale correction node follows.
  const std::string pool_out_name =
      utils::UniqueNameGenerator().New(node_unit.Outputs()[0].name, "_l2pool");
  {
    QnnTensorWrapper pool_out_tensor(pool_out_name, QNN_TENSOR_TYPE_NATIVE,
                                     input_info.qnn_data_type,
                                     output_info.quant_param.Copy(),
                                     std::vector<uint32_t>(pool_out_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(pool_out_tensor)),
                  "Failed to add L2Pool2d intermediate output tensor.");
  }
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_L2_POOL_2D),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_L2_POOL_2D,
                                                {input_names[0]},
                                                {pool_out_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to create L2Pool2d node.");

  // Build a static scale tensor matching the pool output data type.
  const std::string scale_name =
      utils::UniqueNameGenerator().New(node_unit.Name(), "_l2pool_scale");
  {
    std::vector<uint8_t> scale_data;
    const Qnn_DataType_t dtype = input_info.qnn_data_type;
    if (dtype == QNN_DATATYPE_FLOAT_16) {
      scale_data.resize(sizeof(uint16_t));
      Ort::Float16_t scale_fp16(scale_val);
      memcpy(scale_data.data(), &scale_fp16.val, sizeof(uint16_t));
    } else if (dtype == QNN_DATATYPE_BFLOAT_16) {
      scale_data.resize(sizeof(uint16_t));
      Ort::BFloat16_t scale_bf16(scale_val);
      memcpy(scale_data.data(), &scale_bf16.val, sizeof(uint16_t));
    } else {
      scale_data.resize(sizeof(float));
      memcpy(scale_data.data(), &scale_val, sizeof(float));
    }
    QnnTensorWrapper scale_tensor(scale_name, QNN_TENSOR_TYPE_STATIC, dtype,
                                  QnnQuantParamsWrapper(), {1}, std::move(scale_data));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_tensor)),
                  "Failed to add scale tensor.");
  }

  if (!requires_rank3_reshape) {
    // Rank-4 path: ProcessOutputs creates the final output tensor and the multiply node.
    return ProcessOutputs(qnn_model_wrapper, node_unit,
                          {pool_out_name, scale_name}, {},
                          logger, do_op_validation, QNN_OP_ELEMENT_WISE_MULTIPLY);
  }

  // Rank-3 path: scale correction (still rank-4), then Reshape back to rank-3.
  const std::string& final_output_name = node_unit.Outputs()[0].name;
  const std::string scaled_out_name =
      utils::UniqueNameGenerator().New(pool_out_name, "_scaled");
  {
    QnnTensorWrapper scaled_out_tensor(scaled_out_name, QNN_TENSOR_TYPE_NATIVE,
                                       input_info.qnn_data_type,
                                       output_info.quant_param.Copy(),
                                       std::vector<uint32_t>(pool_out_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scaled_out_tensor)),
                  "Failed to add scaled intermediate tensor for LpPool rank-3 path.");
  }
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_MULTIPLY),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                    {pool_out_name, scale_name},
                    {scaled_out_name},
                    {},
                    do_op_validation),
                "Failed to create scale correction node for LpPool rank-3 path.");

  const bool final_output_is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  const Qnn_TensorType_t final_output_tensor_type =
      final_output_is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper final_output_tensor(final_output_name,
                                       final_output_tensor_type,
                                       output_info.qnn_data_type,
                                       output_info.quant_param.Copy(),
                                       std::vector<uint32_t>(output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)),
                "Failed to add final output tensor for LpPool rank-3 path.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_RESHAPE),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {scaled_out_name},
                                                {final_output_name},
                                                {},
                                                do_op_validation),
                "Failed to create reshape-after node for LpPool rank-3 path.");

  return Ort::Status();
}

void CreateLpPoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<LpPoolOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
