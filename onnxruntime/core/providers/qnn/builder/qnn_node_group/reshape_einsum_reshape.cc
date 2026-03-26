// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/reshape_einsum_reshape.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

constexpr size_t kEinsumRank6 = 6;
constexpr std::array<uint32_t, 6> kEinsumPerm6{0, 5, 1, 3, 2, 4};
constexpr auto RearrangeShape6 = [](const std::vector<uint32_t>& shape) {
  return std::vector<uint32_t>{shape[0], shape[5], shape[1] * shape[3], shape[2] * shape[4]};
};

namespace {

bool IsEquationEqualToTranspose(std::string equation, std::vector<uint32_t>& perm) {
  equation.erase(std::remove(equation.begin(), equation.end(), ' '), equation.end());
  if (equation.empty()) {
    // Empty equation.
    return false;
  }

  size_t idx_arrow = equation.find("->");
  if (idx_arrow == std::string::npos) {
    // Unexpected equation format without "->".
    return false;
  }

  const std::string lhs = equation.substr(0, idx_arrow);
  const std::string rhs = equation.substr(idx_arrow + 2);
  if (lhs.empty() || rhs.empty()) {
    // Unexpected equation format without at least one input term and one output term.
    return false;
  }

  size_t idx_comma = lhs.find(",");
  if (idx_comma != std::string::npos) {
    // Transpose-equivalent equation should have only one input term.
    return false;
  }

  size_t expected_rank = perm.size();
  if (lhs.size() != expected_rank || rhs.size() != expected_rank) {
    // Unexpected rank.
    return false;
  }

  if (std::unordered_set<char>(rhs.begin(), rhs.end()).size() != expected_rank) {
    // Output term should only contain unique characters.
    return false;
  }

  for (size_t dim = 0; dim < expected_rank; ++dim) {
    size_t lhs_dim = lhs.find(rhs[dim]);
    if (lhs_dim == std::string::npos) {
      // Each character of output term should as well present in input term.
      return false;
    }

    perm[dim] = static_cast<uint32_t>(lhs_dim);
  }

  return true;
}

Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& pre_reshape_node_unit,
                                  const OrtNodeUnit& einsum_node_unit,
                                  const OrtNodeUnit& post_reshape_node_unit,
                                  const Ort::Logger& logger,
                                  bool do_op_validation) {
  // Reshape.
  const OrtNodeUnitIODef& pre_reshape_input = pre_reshape_node_unit.Inputs()[0];
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(pre_reshape_input.name)) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Tensor already added, skip it: " + pre_reshape_input.name).c_str());
  } else {
    QnnTensorWrapper pre_reshape_input_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(pre_reshape_input, pre_reshape_input_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(pre_reshape_input_wrapper)), "Failed to add tensor.");
  }

  const OrtNodeUnitIODef& pre_reshape_output = pre_reshape_node_unit.Outputs()[0];
  TensorInfo pre_reshape_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(pre_reshape_output, pre_reshape_output_info));
  // If any QDQ node exists, the previous matching will fail and thus it is guaranteed not quantized.
  assert(!pre_reshape_output_info.quant_param.IsQuantized());

  // Combine the last 3 dimensions to serve as the input depth for the following DepthToSpace.
  std::vector<uint32_t> pre_reshape_output_shape = {
      pre_reshape_output_info.shape[0],
      pre_reshape_output_info.shape[1],
      pre_reshape_output_info.shape[2],
      pre_reshape_output_info.shape[3] * pre_reshape_output_info.shape[4] * pre_reshape_output_info.shape[5]};
  QnnTensorWrapper pre_reshape_output_wrapper(pre_reshape_output.name,
                                              QNN_TENSOR_TYPE_NATIVE,
                                              pre_reshape_output_info.qnn_data_type,
                                              QnnQuantParamsWrapper(),
                                              std::move(pre_reshape_output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(pre_reshape_output_wrapper)), "Failed to add tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(pre_reshape_node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {pre_reshape_input.name},
                                                {pre_reshape_output.name},
                                                {},
                                                do_op_validation),
                "Failed to add node.");

  // DepthToSpace.
  // Add attribute block_size derived from pre-Reshape's output shape.
  QnnParamWrapper block_size_param(einsum_node_unit.Index(),
                                   einsum_node_unit.Name(),
                                   QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE,
                                   {2},
                                   {pre_reshape_output_info.shape[3], pre_reshape_output_info.shape[4]});
  const std::string block_size_param_name = block_size_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(block_size_param)), "Failed to add param.");

  // Add attribute mode fixed to DCR which is guaranteed previously.
  Qnn_Scalar_t mode_qnn_scalar = QNN_SCALAR_INIT;
  mode_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
  mode_qnn_scalar.uint32Value = QNN_OP_DEPTH_TO_SPACE_MODE_DCR;

  QnnParamWrapper mode_param(einsum_node_unit.Index(),
                             einsum_node_unit.Name(),
                             QNN_OP_DEPTH_TO_SPACE_PARAM_MODE,
                             mode_qnn_scalar);
  const std::string mode_param_name = mode_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(mode_param)), "Failed to add param.");

  const OrtNodeUnitIODef& einsum_output = einsum_node_unit.Outputs()[0];
  TensorInfo einsum_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(einsum_output, einsum_output_info));
  // If any QDQ node exists, the previous matching will fail and thus it is guaranteed not quantized.
  assert(!einsum_output_info.quant_param.IsQuantized());

  // Calculate D2S output shape.
  std::vector<uint32_t> d2s_output_shape = {
      pre_reshape_output_info.shape[0],
      pre_reshape_output_info.shape[1] * pre_reshape_output_info.shape[3],
      pre_reshape_output_info.shape[2] * pre_reshape_output_info.shape[4],
      pre_reshape_output_info.shape[5]};
  QnnTensorWrapper d2s_output_wrapper(einsum_node_unit.Outputs()[0].name,
                                      QNN_TENSOR_TYPE_NATIVE,
                                      einsum_output_info.qnn_data_type,
                                      QnnQuantParamsWrapper(),
                                      std::move(d2s_output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(d2s_output_wrapper)), "Failed to add tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(einsum_node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_DEPTH_TO_SPACE,
                                                {pre_reshape_output.name},
                                                {einsum_output.name},
                                                {block_size_param_name, mode_param_name},
                                                do_op_validation),
                "Failed to add node.");

  // Transpose.
  // Add attribute perm fixed to [0,3,1,2] which is guaranteed previously.
  QnnParamWrapper perm_param(post_reshape_node_unit.Index(),
                             post_reshape_node_unit.Name(),
                             QNN_OP_TRANSPOSE_PARAM_PERM,
                             {4},
                             std::vector<uint32_t>{0, 3, 1, 2});
  const std::string perm_param_name = perm_param.GetParamTensorName();
  qnn_model_wrapper.AddParamWrapper(std::move(perm_param));

  const OrtNodeUnitIODef& post_reshape_output = post_reshape_node_unit.Outputs()[0];
  TensorInfo post_reshape_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(post_reshape_output, post_reshape_output_info));
  // If any QDQ node exists, the previous matching will fail and thus it is guaranteed not quantized.
  assert(!post_reshape_output_info.quant_param.IsQuantized());

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(post_reshape_output.name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper post_reshape_output_wrapper(post_reshape_output.name,
                                               tensor_type,
                                               post_reshape_output_info.qnn_data_type,
                                               QnnQuantParamsWrapper(),
                                               std::move(post_reshape_output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(post_reshape_output_wrapper)), "Failed to add tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(post_reshape_node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_TRANSPOSE,
                                                {einsum_output.name},
                                                {post_reshape_output.name},
                                                {perm_param_name},
                                                do_op_validation),
                "Failed to add node.");

  return Ort::Status();
}

}  // namespace

Ort::Status ReshapeEinsumReshapeNodeGroup::IsSupported(QnnModelWrapper& qnn_model_wrapper,
                                                       const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, *node_units_[0], *node_units_[1], *node_units_[2], logger, true);
}

Ort::Status ReshapeEinsumReshapeNodeGroup::AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                                             const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, *node_units_[0], *node_units_[1], *node_units_[2], logger, false);
}

std::unique_ptr<IQnnNodeGroup> ReshapeEinsumReshapeNodeGroup::TryFusion(
    QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& einsum_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& /*logger*/) {
  if (einsum_node_unit.OpType() != "Einsum" || einsum_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Expect Transpose-equivalent Einsum.
  if (einsum_node_unit.Inputs().size() != 1) {
    return nullptr;
  }

  const std::string equation = OrtNodeAttrHelper(einsum_node_unit).Get("equation", std::string(""));
  std::vector<uint32_t> perm(kEinsumRank6, kEinsumRank6);
  if (!IsEquationEqualToTranspose(equation, perm)) {
    return nullptr;
  }

  // Expect Reshape before Einsum.
  const OrtNodeUnit* pre_reshape_node_unit = GetParentOfInputByName(qnn_model_wrapper,
                                                                    einsum_node_unit,
                                                                    einsum_node_unit.Inputs()[0].name,
                                                                    node_to_node_unit,
                                                                    node_unit_to_qnn_node_group);
  if (pre_reshape_node_unit == nullptr || pre_reshape_node_unit->OpType() != "Reshape") {
    return nullptr;
  }

  std::vector<uint32_t> pre_reshape_output_shape;
  if (!qnn_model_wrapper.GetOnnxShape(pre_reshape_node_unit->Outputs()[0].shape, pre_reshape_output_shape)) {
    return nullptr;
  }

  // Expect Reshape after Einsum.
  const OrtNodeUnit* post_reshape_node_unit = GetOnlyChildOfOutput(qnn_model_wrapper,
                                                                   einsum_node_unit,
                                                                   einsum_node_unit.Outputs()[0],
                                                                   node_to_node_unit,
                                                                   node_unit_to_qnn_node_group);
  if (post_reshape_node_unit == nullptr || post_reshape_node_unit->OpType() != "Reshape") {
    return nullptr;
  }

  std::vector<uint32_t> post_reshape_output_shape;
  if (!qnn_model_wrapper.GetOnnxShape(post_reshape_node_unit->Outputs()[0].shape, post_reshape_output_shape)) {
    return nullptr;
  }

  // Check perm and shape as expected.
  if (perm != std::vector<uint32_t>(kEinsumPerm6.begin(), kEinsumPerm6.end()) ||
      post_reshape_output_shape != RearrangeShape6(pre_reshape_output_shape)) {
    return nullptr;
  }

  return std::make_unique<ReshapeEinsumReshapeNodeGroup>(pre_reshape_node_unit,
                                                         &einsum_node_unit,
                                                         post_reshape_node_unit);
}

}  // namespace qnn
}  // namespace onnxruntime
