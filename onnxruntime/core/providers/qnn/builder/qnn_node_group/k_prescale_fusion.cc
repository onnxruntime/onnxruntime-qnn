// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/k_prescale_fusion.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <utility>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
namespace {

Ort::Status GetEncoding(const TensorInfo& info, float& scale, int32_t& offset) {
  RETURN_IF_NOT(info.quant_param.IsPerTensor(), "Expected per-tensor encoding.");
  RETURN_IF_ERROR(info.quant_param.GetPerTensorScaleOffset(scale, offset));
  RETURN_IF_NOT(std::isfinite(scale) && scale > 0, "Invalid scale.");
  return Ort::Status();
}

bool SameEncoding(const TensorInfo& a, const TensorInfo& b) {
  float a_scale = 0, b_scale = 0;
  int32_t a_offset = 0, b_offset = 0;
  return a.qnn_data_type == b.qnn_data_type &&
         GetEncoding(a, a_scale, a_offset).IsOK() && GetEncoding(b, b_scale, b_offset).IsOK() &&
         a_scale == b_scale && a_offset == b_offset;
}

Ort::Status GetDivisor(QnnModelWrapper& qmw, const OrtNodeUnitIODef& input, float& divisor) {
  TensorInfo info{};
  RETURN_IF_ERROR(qmw.GetTensorInfo(input, info));
  RETURN_IF_NOT(info.is_initializer &&
                    std::all_of(info.shape.begin(), info.shape.end(), [](uint32_t d) { return d == 1; }),
                "Divisor must be a scalar initializer.");
  if (!input.quant_param.has_value()) {
    const auto value = GetScalarConstantValue(qmw, input.name);
    RETURN_IF_NOT(value.has_value(), "Unsupported divisor type.");
    divisor = *value;
  } else {
    float scale = 0;
    int32_t offset = 0;
    RETURN_IF_ERROR(GetEncoding(info, scale, offset));
    std::vector<uint8_t> data;
    RETURN_IF_ERROR(qmw.UnpackInitializerData(info.initializer_tensor, data));
    if (info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_8 && data.size() == sizeof(uint8_t)) {
      divisor = (static_cast<float>(data[0]) + offset) * scale;
    } else if (info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16 && data.size() == sizeof(uint16_t)) {
      uint16_t code = 0;
      std::memcpy(&code, data.data(), sizeof(code));
      divisor = (static_cast<float>(code) + offset) * scale;
    } else {
      return MAKE_EP_FAIL("Unsupported quantized divisor type.");
    }
  }
  RETURN_IF_NOT(std::isfinite(divisor) && divisor > 1.0f, "Divisor must attenuate K.");
  return Ort::Status();
}

Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                  const std::array<const OrtNodeUnit*, 5>& nodes,
                                  const Ort::Logger& logger, bool validate) {
  const auto& projection = *nodes[0];
  const auto& reshape = *nodes[1];
  const auto& transpose = *nodes[2];
  const auto& qk = *nodes[3];
  const auto& div = *nodes[4];

  TensorInfo x{}, weight{}, key{}, reshaped_key{}, transposed_key{}, query{}, score{}, output{};
  RETURN_IF_ERROR(qmw.GetTensorInfo(projection.Inputs()[0], x));
  RETURN_IF_ERROR(qmw.GetTensorInfo(projection.Inputs()[1], weight));
  RETURN_IF_ERROR(qmw.GetTensorInfo(projection.Outputs()[0], key));
  RETURN_IF_ERROR(qmw.GetTensorInfo(reshape.Outputs()[0], reshaped_key));
  RETURN_IF_ERROR(qmw.GetTensorInfo(transpose.Outputs()[0], transposed_key));
  RETURN_IF_ERROR(qmw.GetTensorInfo(qk.Inputs()[0], query));
  RETURN_IF_ERROR(qmw.GetTensorInfo(qk.Outputs()[0], score));
  RETURN_IF_ERROR(qmw.GetTensorInfo(div.Outputs()[0], output));

  // Start with a bias-free, static U8 projection and per-tensor U16 activations.
  // Other projection formats keep their existing MatMul/FC lowering.
  RETURN_IF_NOT(weight.is_initializer && weight.shape.size() == 2 &&
                    weight.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_8 &&
                    !x.is_initializer && !query.is_initializer,
                "Unsupported K projection.");
  for (const TensorInfo* info : {&x, &key, &reshaped_key, &transposed_key, &query, &score, &output}) {
    float scale = 0;
    int32_t offset = 0;
    RETURN_IF_NOT(info->qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16, "Expected U16 activation.");
    RETURN_IF_ERROR(GetEncoding(*info, scale, offset));
    RETURN_IF_NOT(!info->shape.empty() &&
                      std::all_of(info->shape.begin(), info->shape.end(), [](uint32_t d) { return d > 0; }),
                  "Expected fixed nonempty shape.");
  }

  // Layout Q/DQ boundaries must be no-ops before their scales can move together.
  RETURN_IF_NOT(SameEncoding(key, reshaped_key) && SameEncoding(key, transposed_key),
                "K layout encodings differ.");
  for (const auto& entry : {std::pair{&reshape.Inputs()[0], &key},
                            std::pair{&transpose.Inputs()[0], &key},
                            std::pair{&qk.Inputs()[1], &key},
                            std::pair{&div.Inputs()[0], &score}}) {
    TensorInfo input_info{};
    RETURN_IF_ERROR(qmw.GetTensorInfo(*entry.first, input_info));
    RETURN_IF_NOT(SameEncoding(input_info, *entry.second), "Inter-node encodings differ.");
  }

  // [B,S,E] * [E,H*D] -> [B,S,H,D] -> [B,H,D,S].
  OrtNodeAttrHelper attr(transpose);
  RETURN_IF_NOT(attr.Get("perm", std::vector<int64_t>{}) == std::vector<int64_t>({0, 2, 3, 1}),
                "Expected K transpose.");
  RETURN_IF_NOT(x.shape.size() == 3 && key.shape.size() == 3 && reshaped_key.shape.size() == 4 &&
                    transposed_key.shape.size() == 4 && query.shape.size() == 4 && score.shape.size() == 4,
                "Unsupported attention ranks.");
  const uint32_t batch = x.shape[0], sequence = x.shape[1], channels = x.shape[2];
  const uint32_t heads = reshaped_key.shape[2], depth = reshaped_key.shape[3];
  RETURN_IF_NOT(batch <= UINT32_MAX / sequence && heads <= UINT32_MAX / depth,
                "Merged dimension overflows.");
  RETURN_IF_NOT(weight.shape == std::vector<uint32_t>({channels, heads * depth}) &&
                    key.shape == std::vector<uint32_t>({batch, sequence, heads * depth}) &&
                    reshaped_key.shape == std::vector<uint32_t>({batch, sequence, heads, depth}) &&
                    transposed_key.shape == std::vector<uint32_t>({batch, heads, depth, sequence}) &&
                    query.shape[0] == batch && query.shape[1] == heads && query.shape[3] == depth &&
                    score.shape == std::vector<uint32_t>({batch, heads, query.shape[2], sequence}) &&
                    output.shape == score.shape,
                "Attention shapes do not match.");

  float divisor = 0, weight_scale = 0, key_scale = 0, score_scale = 0, output_scale = 0;
  int32_t weight_offset = 0, key_offset = 0, score_offset = 0, output_offset = 0;
  RETURN_IF_ERROR(GetDivisor(qmw, div.Inputs()[1], divisor));
  RETURN_IF_ERROR(GetEncoding(weight, weight_scale, weight_offset));
  RETURN_IF_ERROR(GetEncoding(key, key_scale, key_offset));
  RETURN_IF_ERROR(GetEncoding(score, score_scale, score_offset));
  RETURN_IF_ERROR(GetEncoding(output, output_scale, output_offset));
  // Relative tolerance only: an absolute epsilon would hide mismatches in small scales.
  RETURN_IF_NOT(score_offset == output_offset &&
                    std::abs(score_scale / divisor - output_scale) <= output_scale * 1e-6f,
                "Div output encoding does not match the scaled QK encoding.");
  weight_scale /= divisor;
  key_scale /= divisor;
  RETURN_IF_NOT(std::isfinite(weight_scale) && weight_scale > 0 && std::isfinite(key_scale) && key_scale > 0,
                "Scaled encoding is not representable.");

  // (Q*K)/c = Q*(K/c). Scaling W and each K encoding preserves their integer codes
  // and zero points. The QK output takes Div's encoding, eliminating Div and its Q/DQ.
  // This deliberately changes the QNN quantized boundary, not the source ONNX graph.
  const std::string prefix = utils::UniqueNameGenerator().New(qk) + "_k_prescale";
  enum TensorIndex : size_t {
    kX,
    kQuery,
    kWeight,
    kFlattenedInput,
    kProjection,
    kReshapedKey,
    kTransposedKey,
    kConvertedKey,
    kOutput,
    kTensorCount
  };
  std::array<QnnTensorWrapper, kTensorCount> tensors;
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(projection.Inputs()[0], tensors[kX]));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(qk.Inputs()[0], tensors[kQuery]));
  std::vector<uint32_t> weight_shape = weight.shape;
  std::vector<uint8_t> weight_data;
  RETURN_IF_ERROR(utils::TwoDimensionTranspose(qmw, weight_shape, weight.initializer_tensor,
                                               weight_data, logger, validate));
  tensors[kWeight] = QnnTensorWrapper(prefix + "_weight", QNN_TENSOR_TYPE_STATIC, weight.qnn_data_type,
                                      QnnQuantParamsWrapper::PerTensor(weight_scale, weight_offset),
                                      std::move(weight_shape), std::move(weight_data));
  tensors[kFlattenedInput] = QnnTensorWrapper(prefix + "_input", QNN_TENSOR_TYPE_NATIVE, x.qnn_data_type,
                                              x.quant_param.Copy(), {batch * sequence, channels});
  tensors[kProjection] = QnnTensorWrapper(prefix + "_projection", QNN_TENSOR_TYPE_NATIVE, key.qnn_data_type,
                                          QnnQuantParamsWrapper::PerTensor(key_scale, key_offset),
                                          {batch * sequence, heads * depth});
  tensors[kReshapedKey] = QnnTensorWrapper(prefix + "_reshape", QNN_TENSOR_TYPE_NATIVE, key.qnn_data_type,
                                           QnnQuantParamsWrapper::PerTensor(key_scale, key_offset),
                                           std::vector<uint32_t>(reshaped_key.shape));
  tensors[kTransposedKey] = QnnTensorWrapper(prefix + "_transpose", QNN_TENSOR_TYPE_NATIVE, key.qnn_data_type,
                                             QnnQuantParamsWrapper::PerTensor(key_scale, key_offset),
                                             std::vector<uint32_t>(transposed_key.shape));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(div.Outputs()[0], tensors[kOutput]));

  // Use the regular MatMul conversion policy, evaluated on K/c, not on the old K.
  const bool convert = key_offset != -32768;
  if (convert) {
    const bool symmetric = utils::ShouldUseSymmetricU16ForMatMul(key_scale);
    const Qnn_DataType_t dtype = symmetric ? QNN_DATATYPE_UFIXED_POINT_16 : QNN_DATATYPE_UFIXED_POINT_8;
    float scale = 0;
    int32_t offset = 0;
    RETURN_IF_ERROR(utils::DeriveScaleOffsetForDtype(key.qnn_data_type, key_offset, key_scale,
                                                     dtype, symmetric, scale, offset));
    tensors[kConvertedKey] = QnnTensorWrapper(prefix + "_convert", QNN_TENSOR_TYPE_NATIVE, dtype,
                                              QnnQuantParamsWrapper::PerTensor(scale, offset),
                                              std::vector<uint32_t>(transposed_key.shape));
  }

  QnnParamWrapper perm(transpose.Index(), prefix, QNN_OP_TRANSPOSE_PARAM_PERM,
                       std::vector<uint32_t>{4}, std::vector<uint32_t>{0, 2, 3, 1});
  Qnn_Scalar_t no_transpose = QNN_SCALAR_INIT;
  no_transpose.dataType = QNN_DATATYPE_BOOL_8;
  no_transpose.bool8Value = false;
  QnnParamWrapper transpose0(qk.Index(), prefix, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, no_transpose);
  QnnParamWrapper transpose1(qk.Index(), prefix, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, no_transpose);

  // Keep validation local. A declined fusion must not leave scaled tensors in the wrapper.
  std::array<std::string, kTensorCount> names;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (i == kConvertedKey && !convert) continue;
    names[i] = tensors[i].GetName();
  }
  const std::string perm_name = perm.GetParamTensorName();
  const std::vector<std::string> matmul_params = {transpose0.GetParamTensorName(), transpose1.GetParamTensorName()};
  if (!validate) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (i == kConvertedKey && !convert) continue;
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tensors[i])), "Failed to add K pre-scale tensor.");
    }
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(perm)) && qmw.AddParamWrapper(std::move(transpose0)) &&
                      qmw.AddParamWrapper(std::move(transpose1)),
                  "Failed to add K pre-scale parameter.");
  }
  auto emit = [&](const std::string& suffix, const std::string& op, std::initializer_list<size_t> inputs,
                  size_t out, std::vector<Qnn_Param_t> params = {},
                  std::vector<std::string> param_names = {}) -> Ort::Status {
    if (validate) {
      std::vector<Qnn_Tensor_t> input_tensors;
      for (size_t i : inputs) input_tensors.push_back(tensors[i].GetQnnTensor());
      return qmw.ValidateQnnNode(prefix + suffix, QNN_OP_PACKAGE_NAME_QTI_AISW, op,
                                 std::move(input_tensors), {tensors[out].GetQnnTensor()}, std::move(params));
    }
    std::vector<std::string> input_names;
    for (size_t i : inputs) input_names.push_back(names[i]);
    RETURN_IF_NOT(qmw.CreateQnnNode(prefix + suffix, QNN_OP_PACKAGE_NAME_QTI_AISW, op,
                                    std::move(input_names), {names[out]}, std::move(param_names), false),
                  "Failed to add K pre-scale node.");
    return Ort::Status();
  };
  RETURN_IF_ERROR(emit("_flatten", QNN_OP_RESHAPE, {kX}, kFlattenedInput));
  RETURN_IF_ERROR(emit("_fc", QNN_OP_FULLY_CONNECTED, {kFlattenedInput, kWeight}, kProjection));
  RETURN_IF_ERROR(emit("_layout", QNN_OP_RESHAPE, {kProjection}, kReshapedKey));
  RETURN_IF_ERROR(emit("_transpose", QNN_OP_TRANSPOSE, {kReshapedKey}, kTransposedKey,
                       validate ? std::vector<Qnn_Param_t>{perm.GetQnnParam()} : std::vector<Qnn_Param_t>{}, {perm_name}));
  if (convert) RETURN_IF_ERROR(emit("_convert", QNN_OP_CONVERT, {kTransposedKey}, kConvertedKey));
  RETURN_IF_ERROR(emit("_matmul", QNN_OP_MAT_MUL, {kQuery, convert ? kConvertedKey : kTransposedKey}, kOutput,
                       validate ? std::vector<Qnn_Param_t>{transpose0.GetQnnParam(), transpose1.GetQnnParam()}
                                : std::vector<Qnn_Param_t>{},
                       matmul_params));
  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> KPrescaleFusion::TryFusion(
    QnnModelWrapper& qmw, const OrtNodeUnit& node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  if (!IsNpuBackend(qmw.GetQnnBackendType()) || node_unit.OpType() != "MatMul" ||
      node_unit.UnitType() != OrtNodeUnit::Type::QDQGroup || node_unit.Inputs().size() != 2 ||
      !qmw.IsConstantInput(node_unit.Inputs()[1].name)) return nullptr;

  std::array<const OrtNodeUnit*, 5> nodes = {&node_unit};
  const std::array<std::string, 4> child_types = {"Reshape", "Transpose", "MatMul", "Div"};
  for (size_t i = 1; i < nodes.size(); ++i) {
    nodes[i] = GetChildNodeUnitAllowQdq(qmw, *nodes[i - 1], child_types[i - 1],
                                        node_to_node_unit, node_unit_to_qnn_node_group);
    if (!nodes[i] || nodes[i]->UnitType() != OrtNodeUnit::Type::QDQGroup ||
        nodes[i]->Inputs().empty() || nodes[i]->Outputs().size() != 1) return nullptr;
  }
  if (nodes[3]->Inputs().size() != 2 || nodes[4]->Inputs().size() != 2 ||
      GetParentOfInput(qmw, *nodes[3], nodes[3]->Inputs()[1], node_to_node_unit,
                       node_unit_to_qnn_node_group) != nodes[2]) return nullptr;
  // The child walker checks quantized outputs; also guard raw math outputs against extra consumers.
  for (size_t i = 0; i < 4; ++i) {
    const auto outputs = Ort::ConstNode(&nodes[i]->GetNode()).GetOutputs();
    if (outputs.size() != 1 || outputs[0].IsGraphOutput() || outputs[0].GetConsumers().size() != 1) return nullptr;
  }
  if (!CreateOrValidateOnQnn(qmw, nodes, logger, true).IsOK()) return nullptr;
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("KPrescaleFusion: folding K scale for " + nodes[3]->Name()).c_str());
  return std::make_unique<KPrescaleFusion>(nodes);
}

Ort::Status KPrescaleFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, node_units_, logger, true);
}

Ort::Status KPrescaleFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, node_units_, logger, false);
}

}  // namespace qnn
}  // namespace onnxruntime
