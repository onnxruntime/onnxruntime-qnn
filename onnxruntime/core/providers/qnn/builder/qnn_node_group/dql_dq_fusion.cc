// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/dql_dq_fusion.h"

#include <gsl/gsl>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/dq_integer_op_fusion_utils.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// ---------------------------------------------------------------------------
// TryFusion
// ---------------------------------------------------------------------------
std::unique_ptr<IQnnNodeGroup> DqlDqFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& dql_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  auto reject = [&logger](std::string_view reason) -> std::unique_ptr<IQnnNodeGroup> {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                (std::string("DqlDqFusion rejected: ").append(reason)).c_str());
    return nullptr;
  };

  if (dql_node_unit.OpType() != kOpDynamicQuantizeLinear ||
      dql_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return reject("not a DynamicQuantizeLinear SingleNode");
  }

  const auto& dql_inputs = dql_node_unit.Inputs();
  const auto& dql_outputs = dql_node_unit.Outputs();
  if (dql_inputs.size() != 1 || dql_outputs.size() != 3) {
    return reject("DQL input/output count mismatch (expected 1 input, 3 outputs)");
  }

  // DQL input must be float32.
  TensorInfo input_info{};
  if (!qnn_model_wrapper.GetTensorInfo(dql_inputs[0], input_info).IsOK() ||
      input_info.qnn_data_type != QNN_DATATYPE_FLOAT_32) {
    return reject("DQL input is not float32");
  }

  // All 3 DQL outputs must be consumed exclusively by DequantizeLinear nodes.
  // ConsumersAreAllOfType returns true vacuously for 0 consumers; GetOnlyChildOfOutput below
  // handles the uniqueness check.
  const std::vector<Ort::ConstValueInfo> dql_outs =
      Ort::ConstNode(&dql_node_unit.GetNode()).GetOutputs();
  if (dql_outs.size() != 3) {
    return reject("DQL node does not have 3 output value_infos");
  }
  if (!ConsumersAreAllOfType(dql_outs[0], DEQUANTIZE_LINEAR, node_to_node_unit) ||
      !ConsumersAreAllOfType(dql_outs[1], DEQUANTIZE_LINEAR, node_to_node_unit) ||
      !ConsumersAreAllOfType(dql_outs[2], DEQUANTIZE_LINEAR, node_to_node_unit)) {
    return reject("not all DQL outputs are consumed exclusively by DequantizeLinear");
  }

  // Find the single DQ consuming DQL.output[0]; it must not already belong to another group.
  const OrtNodeUnit* dq = GetOnlyChildOfOutput(
      qnn_model_wrapper, dql_node_unit, dql_outputs[0],
      node_to_node_unit, node_unit_to_qnn_node_group);
  if (dq == nullptr || dq->OpType() != DEQUANTIZE_LINEAR ||
      dq->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return reject("DQL.output[0] is not exclusively consumed by a standalone DequantizeLinear");
  }

  // y_scale and y_zero_point must each have exactly one consumer and it must be the fused DQ node.
  // y_zero_point must not be absent: DQL's y_zp is dynamic and typically non-zero, so allowing DQ
  // to omit it (defaulting to zero_point=0) would break the identity round-trip.
  // These checks also prevent fusion when a second DQ node shares y_scale or y_zp: in that case the
  // second DQ would lose its producer after fusion, corrupting the graph.
  {
    auto has_single_dq_consumer = [&](const Ort::ConstValueInfo& vi) -> bool {
      std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = vi.GetConsumers();
      if (consumers.size() != 1 || consumers[0].node == nullptr) return false;
      const auto it = node_to_node_unit.find(consumers[0].node);
      return it != node_to_node_unit.end() && it->second == dq;
    };

    if (!has_single_dq_consumer(dql_outs[1])) {
      return reject("DQL.y_scale is not exclusively consumed by the fused DQ");
    }

    // y_zero_point: must be consumed exclusively by the same DQ.
    // DQL computes a dynamic y_zp (typically non-zero for float32 inputs; e.g. ~128 for
    // inputs in [-1, 1]).  If DQ omits zero_point it defaults to 0, making
    // DQ(y, y_scale, 0) != x and breaking the identity round-trip.
    std::vector<Ort::ValueInfoConsumerProducerInfo> zp_consumers = dql_outs[2].GetConsumers();
    if (zp_consumers.empty()) {
      return reject("DQL.y_zero_point has no consumers; DQ would use zero_point=0 which breaks the identity round-trip");
    }
    if (zp_consumers.size() != 1 || zp_consumers[0].node == nullptr) {
      return reject("DQL.y_zero_point has unexpected multiple consumers");
    }
    {
      const auto it = node_to_node_unit.find(zp_consumers[0].node);
      if (it == node_to_node_unit.end() || it->second != dq) {
        return reject("DQL.y_zero_point consumer is not the fused DQ");
      }
    }
  }

  // OrtNodeUnit::Inputs() wraps scale/zero_point as quant_param, so DQ has 1 logical input.
  const auto& dq_inputs = dq->Inputs();
  const auto& dq_outputs = dq->Outputs();
  if (dq_inputs.size() != 1 || dq_outputs.size() != 1) {
    return reject(("DQ has unexpected input/output count: " + std::to_string(dq_inputs.size()) +
                   " inputs, " + std::to_string(dq_outputs.size()) + " outputs")
                      .c_str());
  }

  // DQ's x must come from DQL's y (output[0]).
  if (dq_inputs[0].name != dql_outputs[0].name) {
    return reject("DQ.x does not match DQL.y");
  }

  // DQ's scale must be present and come from DQL's y_scale (output[1]).
  if (!dq_inputs[0].quant_param.has_value() || dq_inputs[0].quant_param->scale == nullptr) {
    return reject("DQ is missing scale quant_param");
  }
  if (Ort::ConstValueInfo(dq_inputs[0].quant_param->scale).GetName() != dql_outputs[1].name) {
    return reject("DQ.scale does not match DQL.y_scale");
  }

  // DQ's zero_point must be present and come from DQL's y_zero_point (output[2]).
  // DQL's y_zp is dynamic and typically non-zero; allowing DQ to omit it (defaulting
  // to 0) would break the identity round-trip.
  if (dq_inputs[0].quant_param->zero_point == nullptr) {
    return reject("DQ is missing zero_point quant_param");
  }
  if (Ort::ConstValueInfo(dq_inputs[0].quant_param->zero_point).GetName() != dql_outputs[2].name) {
    return reject("DQ.zero_point does not match DQL.y_zero_point");
  }

  // DQ output must be float32.
  TensorInfo dq_output_info{};
  if (!qnn_model_wrapper.GetTensorInfo(dq_outputs[0], dq_output_info).IsOK() ||
      dq_output_info.qnn_data_type != QNN_DATATYPE_FLOAT_32) {
    return reject("DQ output is not float32");
  }

  auto fusion = std::unique_ptr<DqlDqFusion>(
      new DqlDqFusion(&dql_node_unit, dq, dql_inputs[0].name, dq_outputs[0].name));

  if (Ort::Status status = fusion->CreateOrValidateOnQnn(qnn_model_wrapper, /*validate=*/true);
      !status.IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("DqlDqFusion rejected by QNN validate: " + status.GetErrorMessage()).c_str());
    return nullptr;
  }

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "DqlDqFusion matched and validated");
  return fusion;
}

// ---------------------------------------------------------------------------
// Constructor / IQnnNodeGroup plumbing
// ---------------------------------------------------------------------------
DqlDqFusion::DqlDqFusion(const OrtNodeUnit* dql, const OrtNodeUnit* dq,
                         std::string float_input_name, std::string float_output_name)
    : dql_(dql),
      node_units_({dql, dq}),
      float_input_name_(std::move(float_input_name)),
      float_output_name_(std::move(float_output_name)) {}

Ort::Status DqlDqFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, /*validate=*/true);
}

Ort::Status DqlDqFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, /*validate=*/false);
}

gsl::span<const OrtNodeUnit* const> DqlDqFusion::GetNodeUnits() const {
  return gsl::make_span(node_units_);
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------
Ort::Status DqlDqFusion::CreateOrValidateOnQnn(QnnModelWrapper& qmw, bool validate) const {
  // Get the shape of the float input (DQL's single input tensor).
  TensorInfo input_info{};
  RETURN_IF_ERROR(qmw.GetTensorInfo(dql_->Inputs()[0], input_info));
  const std::vector<uint32_t> shape(input_info.shape.begin(), input_info.shape.end());
  const uint32_t rank = static_cast<uint32_t>(shape.size());
  RETURN_IF_NOT(rank >= 1, "DQL input must have rank >= 1");

  // Identity permutation: [0, 1, ..., rank-1].
  std::vector<uint32_t> perm(rank);
  for (uint32_t i = 0; i < rank; ++i) perm[i] = i;

  const std::string node_name = utils::UniqueNameGenerator().New(*dql_);

  const Qnn_TensorType_t in_type =
      qmw.IsGraphInput(float_input_name_) ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_NATIVE;
  const Qnn_TensorType_t out_type =
      qmw.IsGraphOutput(float_output_name_) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper in_tw(float_input_name_, in_type, QNN_DATATYPE_FLOAT_32,
                         QnnQuantParamsWrapper(), std::vector<uint32_t>(shape));
  QnnTensorWrapper out_tw(float_output_name_, out_type, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>(shape));
  QnnParamWrapper perm_param(dql_->Index(), node_name, QNN_OP_TRANSPOSE_PARAM_PERM,
                             std::vector<uint32_t>{rank}, std::move(perm));

  if (validate) {
    RETURN_IF_ERROR(qmw.ValidateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                        QNN_OP_TRANSPOSE,
                                        {in_tw.GetQnnTensor()},
                                        {out_tw.GetQnnTensor()},
                                        {perm_param.GetQnnParam()}));
  } else {
    if (!qmw.IsQnnTensorWrapperExist(float_input_name_)) {
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(in_tw)),
                    "DqlDqFusion: failed to add float input tensor");
    }
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(out_tw)),
                  "DqlDqFusion: failed to add float output tensor");

    std::vector<std::string> param_names = {perm_param.GetParamTensorName()};
    qmw.AddParamWrapper(std::move(perm_param));

    RETURN_IF_NOT(qmw.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                    QNN_OP_TRANSPOSE,
                                    {float_input_name_}, {float_output_name_},
                                    std::move(param_names), /*do_op_validation=*/false),
                  "DqlDqFusion: failed to create identity Transpose node");
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
