// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

struct QnnPoolConfig {
  const char* op;
  const char* param_filter_size;
  const char* param_stride;
  const char* param_pad_amount;
  const char* param_rounding_mode;
  const char* param_count_pad_for_edges;
};

struct QnnPoolParams {
  std::vector<uint32_t> filter_size;
  std::vector<uint32_t> stride;
  std::vector<uint32_t> pad_amount;
  int32_t rounding_mode{0};
  bool count_pad_for_edges{false};
};

class PoolOpBuilder : public BaseOpBuilder {
 public:
  PoolOpBuilder() : BaseOpBuilder("PoolOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PoolOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
  Ort::Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const Ort::Logger& logger,
                                       const std::vector<std::string>& input_names,
                                       size_t output_index,
                                       Qnn_DataType_t qnn_data_type,
                                       QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status AddQnnPoolParamWrappers(const OrtNodeUnit& node_unit,
                                      const QnnPoolConfig& qnn_pool_config,
                                      QnnPoolParams&& qnn_pool_params,
                                      std::vector<std::string>& param_tensor_names,
                                      QnnModelWrapper& qnn_model_wrapper) const;
  Ort::Status AddQnnPoolNode(QnnModelWrapper& qnn_model_wrapper,
                             const OrtNodeUnit& node_unit,
                             std::vector<std::string>&& input_names,
                             std::vector<std::string>&& param_tensor_names,
                             const QnnPoolConfig& qnn_pool_config,
                             const TensorInfo& input_info,
                             bool requires_rank3_reshape,
                             std::vector<uint32_t>&& intermediate_output_shape_4d,
                             const Ort::Logger& logger,
                             bool do_op_validation) const;
};

namespace {

// Rank 4 (NHWC) -> QNN Pool2D, Rank 5 (NDHWC) -> QNN Pool3D.
// Rank 3 inputs are reshaped to rank 4 before reaching here.
Ort::Status GetQnnPoolConfig(bool is_max_pool, size_t rank, QnnPoolConfig& config) {
  RETURN_IF_NOT(rank == 4 || rank == 5, "GetQnnPoolConfig expects rank 4 or 5.");

  if (rank == 4 && is_max_pool) {
    config = {QNN_OP_POOL_MAX_2D,
              QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE,
              QNN_OP_POOL_MAX_2D_PARAM_STRIDE,
              QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT,
              QNN_OP_POOL_MAX_2D_PARAM_ROUNDING_MODE,
              nullptr};
  } else if (rank == 4) {
    config = {QNN_OP_POOL_AVG_2D,
              QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE,
              QNN_OP_POOL_AVG_2D_PARAM_STRIDE,
              QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT,
              QNN_OP_POOL_AVG_2D_PARAM_ROUNDING_MODE,
              QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES};
  } else if (is_max_pool) {
    config = {QNN_OP_POOL_MAX_3D,
              QNN_OP_POOL_MAX_3D_PARAM_FILTER_SIZE,
              QNN_OP_POOL_MAX_3D_PARAM_STRIDE,
              QNN_OP_POOL_MAX_3D_PARAM_PAD_AMOUNT,
              QNN_OP_POOL_MAX_3D_PARAM_ROUNDING_MODE,
              nullptr};
  } else {
    config = {QNN_OP_POOL_AVG_3D,
              QNN_OP_POOL_AVG_3D_PARAM_FILTER_SIZE,
              QNN_OP_POOL_AVG_3D_PARAM_STRIDE,
              QNN_OP_POOL_AVG_3D_PARAM_PAD_AMOUNT,
              QNN_OP_POOL_AVG_3D_PARAM_ROUNDING_MODE,
              QNN_OP_POOL_AVG_3D_PARAM_COUNT_PAD_FOR_EDGES};
  }

  return Ort::Status();
}

}  // namespace

// Pool ops are sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
// Need to do op validation in 1st call of GetCapability
// TODO: Check if node domain == kMSInternalNHWCDomain to determine if the layout has been transformed.
Ort::Status PoolOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].type, ""));

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape");

  size_t rank = input_shape.size();
  RETURN_IF_NOT(rank == 3 || rank == 4 || rank == 5, "QNN Pool only supports rank 3, 4, or 5!");

  RETURN_IF(node_unit.Outputs().size() > 1, "QNN Pool only supports 1 output!");

  OrtNodeAttrHelper node_helper(node_unit);
  auto dilations = node_helper.Get("dilations", std::vector<uint32_t>(rank - 2, 1));
  RETURN_IF_NOT(dilations == std::vector<uint32_t>(rank - 2, 1), "QNN Pool only supports dilations 1!");

  const std::string& op_type = node_unit.OpType();
  const bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());

  if (rank == 5 && is_npu_backend) {
    RETURN_IF(op_type == "MaxPool" || op_type == "GlobalMaxPool", "QNN NPU does not support PoolMax3d!");
  }

  if (op_type == "MaxPool" || op_type == "AveragePool") {
    auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
    RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER" && auto_pad != "VALID",
              ("QNN Pool operators do not support 'auto_pad' value: " + auto_pad).c_str());
  }

  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  return Ort::Status();
}

Ort::Status PoolOpBuilder::AddQnnPoolParamWrappers(const OrtNodeUnit& node_unit,
                                                   const QnnPoolConfig& qnn_pool_config,
                                                   QnnPoolParams&& qnn_pool_params,
                                                   std::vector<std::string>& param_tensor_names,
                                                   QnnModelWrapper& qnn_model_wrapper) const {
  {
    QnnParamWrapper filter_size_param(node_unit.Index(), node_unit.Name(),
                                      qnn_pool_config.param_filter_size,
                                      {static_cast<uint32_t>(qnn_pool_params.filter_size.size())},
                                      std::move(qnn_pool_params.filter_size));
    param_tensor_names.push_back(filter_size_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(filter_size_param)),
                  "Failed to add param filter_size.");
  }

  {
    QnnParamWrapper stride_param(node_unit.Index(), node_unit.Name(),
                                 qnn_pool_config.param_stride,
                                 {static_cast<uint32_t>(qnn_pool_params.stride.size())},
                                 std::move(qnn_pool_params.stride));
    param_tensor_names.push_back(stride_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(stride_param)),
                  "Failed to add param stride.");
  }

  {
    ReArrangePads(qnn_pool_params.pad_amount);
    QnnParamWrapper pad_amount_param(node_unit.Index(), node_unit.Name(),
                                     qnn_pool_config.param_pad_amount,
                                     {static_cast<uint32_t>(qnn_pool_params.pad_amount.size() / 2), 2},
                                     std::move(qnn_pool_params.pad_amount));
    param_tensor_names.push_back(pad_amount_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_param)),
                  "Failed to add param pad_amount.");
  }

  if (qnn_pool_params.rounding_mode != 0) {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_UINT_32;
    scalar_param.int32Value = qnn_pool_params.rounding_mode;
    QnnParamWrapper rounding_mode_param(node_unit.Index(),
                                        node_unit.Name(),
                                        qnn_pool_config.param_rounding_mode,
                                        scalar_param);
    param_tensor_names.push_back(rounding_mode_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(rounding_mode_param)),
                  "Failed to add param rounding_mode.");
  }

  if (qnn_pool_config.param_count_pad_for_edges != nullptr) {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = static_cast<uint8_t>(qnn_pool_params.count_pad_for_edges);
    QnnParamWrapper count_pad_param(node_unit.Index(),
                                    node_unit.Name(),
                                    qnn_pool_config.param_count_pad_for_edges,
                                    scalar_param);
    param_tensor_names.push_back(count_pad_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(count_pad_param)),
                  "Failed to add param count_pad_for_edges.");
  }

  return Ort::Status();
}

Ort::Status PoolOpBuilder::AddQnnPoolNode(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          std::vector<std::string>&& param_tensor_names,
                                          const QnnPoolConfig& qnn_pool_config,
                                          const TensorInfo& input_info,
                                          bool requires_rank3_reshape,
                                          std::vector<uint32_t>&& intermediate_output_shape_4d,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const {
  if (!requires_rank3_reshape) {
    return ProcessOutputs(qnn_model_wrapper,
                          node_unit,
                          std::move(input_names),
                          std::move(param_tensor_names),
                          logger,
                          do_op_validation,
                          qnn_pool_config.op);
  }

  const std::string& final_output_name = node_unit.Outputs()[0].name;
  const std::string intermediate_output_name = utils::UniqueNameGenerator().New(final_output_name, "_reshape_after");
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  QnnTensorWrapper intermediate_output_tensor(intermediate_output_name,
                                              QNN_TENSOR_TYPE_NATIVE,
                                              input_info.qnn_data_type,
                                              output_info.quant_param.Copy(),
                                              std::move(intermediate_output_shape_4d));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(intermediate_output_tensor)),
                "Failed to add tensor for pool_out.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, qnn_pool_config.op),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                qnn_pool_config.op,
                                                {input_names[0]},
                                                {intermediate_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to create pool node for rank-3 input.");

  const bool final_output_is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  const Qnn_TensorType_t final_output_tensor_type = final_output_is_graph_output ? QNN_TENSOR_TYPE_APP_READ
                                                                                 : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper final_output_tensor(final_output_name,
                                       final_output_tensor_type,
                                       output_info.qnn_data_type,
                                       output_info.quant_param.Copy(),
                                       std::vector<uint32_t>(output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)),
                "Failed to add reshape after tensor.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_RESHAPE),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {intermediate_output_name},
                                                {final_output_name},
                                                {},
                                                do_op_validation),
                "Failed to create reshape after node for pool op.");

  return Ort::Status();
}

Ort::Status PoolOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const std::string& op_type = node_unit.OpType();
  const bool is_global_pool = (op_type == "GlobalMaxPool" || op_type == "GlobalAveragePool");
  const bool is_avg_pool = (op_type == "AveragePool" || op_type == "GlobalAveragePool");
  const bool is_max_pool = (op_type == "MaxPool" || op_type == "GlobalMaxPool");

  std::vector<uint32_t> onnx_input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, onnx_input_shape), "Cannot get shape");

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));

  const bool requires_rank3_reshape = (onnx_input_shape.size() == 3);
  std::vector<uint32_t> qnn_input_shape = onnx_input_shape;
  std::vector<uint32_t> onnx_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Outputs()[0].shape, onnx_output_shape), "Cannot get shape");

  std::vector<uint32_t> qnn_output_shape = onnx_output_shape;
  std::vector<uint32_t> intermediate_output_shape_4d;
  if (requires_rank3_reshape) {
    qnn_input_shape = {onnx_input_shape[0], 1, onnx_input_shape[1], onnx_input_shape[2]};
    qnn_output_shape = {onnx_output_shape[0], 1, onnx_output_shape[1], onnx_output_shape[2]};
    intermediate_output_shape_4d = qnn_output_shape;
  }

  if (requires_rank3_reshape) {
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
                  "Failed to create reshape prior node for pool op.");
    input_names[0] = reshaped_input_name;
  }

  const size_t rank = qnn_input_shape.size();
  QnnPoolConfig qnn_pool_config;
  RETURN_IF_ERROR(GetQnnPoolConfig(is_max_pool, rank, qnn_pool_config));

  QnnPoolParams qnn_pool_params;
  qnn_pool_params.filter_size.assign(qnn_input_shape.begin() + 1, qnn_input_shape.begin() + rank - 1);
  qnn_pool_params.stride = qnn_pool_params.filter_size;
  qnn_pool_params.pad_amount.assign((rank - 2) * 2, 0);

  if (!is_global_pool) {
    std::vector<uint32_t> dilations;  // unused after the call but required by the helper signature
    RETURN_IF_ERROR(ResolvePoolAttributes(node_helper,
                                          qnn_input_shape,
                                          qnn_output_shape,
                                          qnn_pool_params.filter_size,
                                          qnn_pool_params.stride,
                                          dilations,
                                          qnn_pool_params.pad_amount,
                                          qnn_pool_params.rounding_mode));
  }

  qnn_pool_params.count_pad_for_edges =
      is_avg_pool && node_helper.Get("count_include_pad", static_cast<int64_t>(0)) != 0;

  std::vector<std::string> param_tensor_names;
  RETURN_IF_ERROR(AddQnnPoolParamWrappers(node_unit,
                                          qnn_pool_config,
                                          std::move(qnn_pool_params),
                                          param_tensor_names,
                                          qnn_model_wrapper));

  return AddQnnPoolNode(qnn_model_wrapper,
                        node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        qnn_pool_config,
                        input_info,
                        requires_rank3_reshape,
                        std::move(intermediate_output_shape_4d),
                        logger,
                        do_op_validation);
}

Ort::Status PoolOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                    const OrtNodeUnit& node_unit,
                                                    const Ort::Logger& logger,
                                                    const std::vector<std::string>& input_names,
                                                    size_t output_index,
                                                    Qnn_DataType_t qnn_data_type,
                                                    QnnQuantParamsWrapper& quant_param) const {
  // Force MaxPool outputs to use the same quantization parameters as the input if they are nearly equal.
  // This helps the HTP backend employ certain optimizations.
  if (node_unit.OpType() == "MaxPool" && quant_param.IsPerTensor()) {
    return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                    0 /*input_index*/, output_index, qnn_data_type, quant_param);
  }

  return Ort::Status();
}

void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<PoolOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
