// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/dq_matmul_integer_fusion.h"

#include <cstring>
#include <gsl/gsl>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kOpMatMulInteger[] = "MatMulInteger";
constexpr char kOpDynamicQuantizeLinear[] = "DynamicQuantizeLinear";
constexpr char kOpCast[] = "Cast";
constexpr char kOpMul[] = "Mul";
constexpr char kOpAdd[] = "Add";

constexpr std::string_view kFusionType = "DQMatMulIntegerFusion";

struct DqlLookupResult {
  const OrtNodeUnit* dql = nullptr;
  bool already_claimed_by_sibling = false;
};

// Walks up `matmul_integer`'s a_q input to find the producer DynamicQuantizeLinear NodeUnit.
// Tolerates DQL being claimed by a sibling DQMatMulIntegerFusion (multi-MatMulInteger-shared-DQL
// case): only the first sibling actually claims DQL; later siblings detect the existing claim
// and skip the double-claim. Returns dql=nullptr if DQL is claimed by a non-DQMatMulInteger
// fusion (the fusion cannot share DQL across heterogeneous fusion types).
DqlLookupResult FindParentDqlForMatMulInteger(
    const OrtNodeUnit& matmul_integer,
    const OrtNodeUnitIODef& a_q_input,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  DqlLookupResult result;

  const Ort::ConstNode mm_node(&matmul_integer.GetNode());
  const OrtNode* dql_node_raw = nullptr;
  for (const Ort::ConstValueInfo& input_info : mm_node.GetInputs()) {
    if (input_info.GetName() != a_q_input.name) {
      continue;
    }
    const Ort::ConstNode parent = input_info.GetProducerNode().node;
    dql_node_raw = static_cast<const OrtNode*>(parent);
    break;
  }
  if (dql_node_raw == nullptr) {
    return result;
  }

  const auto dql_it = node_to_node_unit.find(dql_node_raw);
  if (dql_it == node_to_node_unit.end()) {
    return result;
  }

  const auto claim_it = qnn_node_group_map.find(dql_it->second);
  if (claim_it != qnn_node_group_map.end()) {
    if (claim_it->second->Type() != kFusionType) {
      return result;
    }
    result.already_claimed_by_sibling = true;
  }

  result.dql = dql_it->second;
  return result;
}

// True if every consumer of `value_info` is a MatMulInteger SingleNode and `value_info` is not
// itself a graph output. Used on DQL's a_q / a_zp outputs.
bool ConsumersAreAllMatMulIntegers(
    const Ort::ConstValueInfo& value_info,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit) {
  if (value_info.IsGraphOutput()) {
    return false;
  }
  for (const auto& c : value_info.GetConsumers()) {
    if (c.node == nullptr) return false;
    const auto it = node_to_node_unit.find(c.node);
    if (it == node_to_node_unit.end()) return false;
    const OrtNodeUnit* nu = it->second;
    if (nu->OpType() != kOpMatMulInteger || nu->UnitType() != OrtNodeUnit::Type::SingleNode) {
      return false;
    }
  }
  return true;
}

// True if every consumer of `value_info` looks like a parallel_Mul: 2-input/1-output Mul
// SingleNode whose other input is a constant initializer. Used on DQL's a_scale output.
bool ConsumersAreAllParallelMuls(
    const Ort::ConstValueInfo& value_info,
    const QnnModelWrapper& qmw,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit) {
  if (value_info.IsGraphOutput()) {
    return false;
  }
  const std::string a_scale_name(value_info.GetName());
  for (const auto& c : value_info.GetConsumers()) {
    if (c.node == nullptr) return false;
    const auto it = node_to_node_unit.find(c.node);
    if (it == node_to_node_unit.end()) return false;
    const OrtNodeUnit* nu = it->second;
    if (nu->OpType() != kOpMul || nu->UnitType() != OrtNodeUnit::Type::SingleNode) return false;
    if (nu->Inputs().size() != 2 || nu->Outputs().size() != 1) return false;
    const auto& mul_inputs = nu->Inputs();
    const std::string& other_name = (mul_inputs[0].name == a_scale_name) ? mul_inputs[1].name
                                                                         : mul_inputs[0].name;
    if (!qmw.IsConstantInput(other_name)) return false;
  }
  return true;
}

Ort::Status ReadFloatInitializer(const QnnModelWrapper& qmw,
                                 const OrtNodeUnitIODef& iodef,
                                 std::vector<float>& out) {
  const std::string& name = iodef.name;
  const OrtValueInfo* info = qmw.GetConstantTensor(name);
  RETURN_IF_NOT(info != nullptr, ("Constant tensor not found: " + name).c_str());

  RETURN_IF_NOT(iodef.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                ("Expected FLOAT initializer for " + name).c_str());

  std::vector<uint8_t> bytes;
  RETURN_IF_ERROR(qmw.UnpackInitializerData(info, bytes));
  RETURN_IF_NOT(bytes.size() % sizeof(float) == 0, "Unexpected byte count for float initializer");

  out.resize(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return Ort::Status();
}

// Reads a zero-point initializer (INT8 or UINT8) as int32 values.
Ort::Status ReadZeroPointAsInt32(const QnnModelWrapper& qmw,
                                 const OrtNodeUnitIODef* zp_iodef,
                                 std::vector<int32_t>& out) {
  out.clear();
  if (zp_iodef == nullptr || !zp_iodef->Exists()) {
    return Ort::Status();
  }
  const std::string& name = zp_iodef->name;
  const OrtValueInfo* info = qmw.GetConstantTensor(name);
  RETURN_IF_NOT(info != nullptr, ("Constant tensor not found: " + name).c_str());

  const ONNXTensorElementDataType elem_type = zp_iodef->type;
  RETURN_IF_NOT(elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
                    elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                ("Expected INT8 or UINT8 zero-point for " + name).c_str());

  std::vector<uint8_t> bytes;
  RETURN_IF_ERROR(qmw.UnpackInitializerData(info, bytes));

  out.resize(bytes.size());
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
    const int8_t* src = reinterpret_cast<const int8_t*>(bytes.data());
    for (size_t i = 0; i < bytes.size(); ++i) {
      out[i] = static_cast<int32_t>(src[i]);
    }
  } else {
    for (size_t i = 0; i < bytes.size(); ++i) {
      out[i] = static_cast<int32_t>(bytes[i]);
    }
  }
  return Ort::Status();
}

// Builds the weight quant params (int8 or uint8) for a [K, N] tensor. ONNX zero-point is
// negated to match QNN's offset convention (QNN: x = scale * (q - offset); offset = -ONNX_zp).
// For per-channel B_scale the result is only used to detect per-channel via IsPerChannel();
// the per-channel emission path pre-dequantizes to float offline and does not consume these
// quant params, so the axis value carried here is informational only.
Ort::Status BuildWeightQuantParams(const QnnModelWrapper& qmw,
                                   const OrtNodeUnitIODef& b_scale_iodef,
                                   const OrtNodeUnitIODef* b_zp_iodef,
                                   uint32_t out_channels,
                                   QnnQuantParamsWrapper& out_params) {
  std::vector<float> scales;
  RETURN_IF_ERROR(ReadFloatInitializer(qmw, b_scale_iodef, scales));
  RETURN_IF_NOT(!scales.empty(), "B_scale has zero elements");

  std::vector<int32_t> offsets;
  RETURN_IF_ERROR(ReadZeroPointAsInt32(qmw, b_zp_iodef, offsets));
  for (int32_t& v : offsets) v = -v;

  if (scales.size() == 1) {
    const int32_t offset = offsets.empty() ? 0 : offsets[0];
    out_params = QnnQuantParamsWrapper(scales[0], offset);
    return Ort::Status();
  }

  RETURN_IF_NOT(static_cast<uint32_t>(scales.size()) == out_channels,
                "Per-channel B_scale length must equal weight out_channels");
  if (offsets.empty()) {
    offsets.assign(scales.size(), 0);
  } else if (offsets.size() == 1) {
    offsets.assign(scales.size(), offsets[0]);
  } else {
    RETURN_IF_NOT(offsets.size() == scales.size(),
                  "B_zp length must equal B_scale length for per-channel");
  }

  out_params = QnnQuantParamsWrapper(gsl::span<const float>(scales),
                                     gsl::span<const int32_t>(offsets),
                                     /*axis=*/1,
                                     /*is_int4=*/false);
  return Ort::Status();
}

// Pre-dequantizes per-channel int8 / uint8 [K, N] weight bytes to float32 bytes (per-channel
// scales / zps are along the output dimension N, which is the last axis).
Ort::Status PreDequantizePerChannelWeight(const QnnModelWrapper& qmw,
                                          const OrtNodeUnitIODef& b_scale_iodef,
                                          const OrtNodeUnitIODef* b_zp_iodef,
                                          bool is_signed_weight,
                                          uint32_t out_channels,
                                          const std::vector<uint8_t>& quant_bytes,
                                          std::vector<uint8_t>& out_float_bytes) {
  std::vector<float> scales;
  RETURN_IF_ERROR(ReadFloatInitializer(qmw, b_scale_iodef, scales));
  RETURN_IF_NOT(scales.size() == static_cast<size_t>(out_channels),
                "Per-channel B_scale length mismatch");

  std::vector<int32_t> zps_onnx;
  RETURN_IF_ERROR(ReadZeroPointAsInt32(qmw, b_zp_iodef, zps_onnx));
  if (zps_onnx.empty()) {
    zps_onnx.assign(scales.size(), 0);
  } else if (zps_onnx.size() == 1) {
    zps_onnx.assign(scales.size(), zps_onnx[0]);
  } else {
    RETURN_IF_NOT(zps_onnx.size() == scales.size(), "Per-channel B_zp length mismatch");
  }

  const size_t num_elems = quant_bytes.size();
  const size_t n = static_cast<size_t>(out_channels);
  RETURN_IF_NOT(n > 0 && num_elems % n == 0, "Weight byte count not divisible by N");

  // Dequantize into a typed float buffer first to avoid uint8_t-to-float aliasing issues,
  // then memcpy out to the byte buffer that QnnTensorWrapper expects.
  std::vector<float> floats(num_elems);
  if (is_signed_weight) {
    const int8_t* src = reinterpret_cast<const int8_t*>(quant_bytes.data());
    for (size_t i = 0; i < num_elems; ++i) {
      const size_t c = i % n;  // last-axis = output channel
      floats[i] = scales[c] * static_cast<float>(static_cast<int32_t>(src[i]) - zps_onnx[c]);
    }
  } else {
    const uint8_t* src = quant_bytes.data();
    for (size_t i = 0; i < num_elems; ++i) {
      const size_t c = i % n;
      floats[i] = scales[c] * static_cast<float>(static_cast<int32_t>(src[i]) - zps_onnx[c]);
    }
  }

  out_float_bytes.resize(num_elems * sizeof(float));
  std::memcpy(out_float_bytes.data(), floats.data(), out_float_bytes.size());
  return Ort::Status();
}

}  // namespace

// ---------------------------------------------------------------------------
// TryFusion
// ---------------------------------------------------------------------------
std::unique_ptr<IQnnNodeGroup> DQMatMulIntegerFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& matmul_integer_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  auto reject = [&logger](std::string_view reason) -> std::unique_ptr<IQnnNodeGroup> {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                (std::string("DQMatMulIntegerFusion rejected: ").append(reason)).c_str());
    return nullptr;
  };

  if (matmul_integer_node_unit.OpType() != kOpMatMulInteger ||
      matmul_integer_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return reject("not a MatMulInteger SingleNode");
  }

  const auto& mm_inputs = matmul_integer_node_unit.Inputs();
  const auto& mm_outputs = matmul_integer_node_unit.Outputs();
  if (mm_inputs.size() < 2 || mm_inputs.size() > 4 || mm_outputs.size() != 1) {
    return reject("MatMulInteger input/output count mismatch");
  }

  TensorInfo a_info{};
  TensorInfo b_info{};
  if (!qnn_model_wrapper.GetTensorInfo(mm_inputs[0], a_info).IsOK() ||
      !qnn_model_wrapper.GetTensorInfo(mm_inputs[1], b_info).IsOK()) {
    return reject("failed to get TensorInfo for MatMulInteger inputs");
  }
  // QNN MatMul requires last 2 dims of A be [M, K], so A must be rank >= 2. B is restricted to
  // rank-2 [K, N] for this fusion - higher-rank batched B can be added later if needed.
  if (a_info.shape.size() < 2 || b_info.shape.size() != 2) {
    return reject("MatMulInteger A must be rank>=2 and B must be rank-2");
  }
  if (a_info.shape.back() != b_info.shape[0]) {
    return reject("MatMulInteger A.last_dim != B.first_dim");
  }

  if (!qnn_model_wrapper.IsConstantInput(mm_inputs[1].name) || !b_info.is_initializer) {
    return reject("weight B is not a constant initializer");
  }
  if (b_info.qnn_data_type != QNN_DATATYPE_SFIXED_POINT_8 &&
      b_info.qnn_data_type != QNN_DATATYPE_INT_8 &&
      b_info.qnn_data_type != QNN_DATATYPE_UFIXED_POINT_8 &&
      b_info.qnn_data_type != QNN_DATATYPE_UINT_8) {
    return reject("weight B is not int8 or uint8");
  }

  const bool has_a_zp = mm_inputs.size() >= 3 && mm_inputs[2].Exists();
  const bool has_b_zp = mm_inputs.size() >= 4 && mm_inputs[3].Exists();
  if (has_b_zp && !qnn_model_wrapper.IsConstantInput(mm_inputs[3].name)) {
    return reject("B_zp is not a constant initializer");
  }

  // The fused float MatMul uses the *pre-DQL* float input as activation; that input still
  // carries the offset DQL would otherwise factor out. MatMulInteger must therefore consume
  // A_zp from DQL for the rewrite to be mathematically equivalent. Without A_zp the output
  // diverges by a_scale * sum_K(a_zp * (B - B_zp)).
  if (!has_a_zp) {
    return reject("MatMulInteger has no A_zp input; fusion would change semantics");
  }

  // Walk up to DQL. Custom lookup tolerates DQL being claimed by a sibling DQMatMulIntegerFusion.
  const DqlLookupResult dql_lookup = FindParentDqlForMatMulInteger(
      matmul_integer_node_unit, mm_inputs[0], node_to_node_unit, node_unit_to_qnn_node_group);
  if (dql_lookup.dql == nullptr ||
      dql_lookup.dql->OpType() != kOpDynamicQuantizeLinear ||
      dql_lookup.dql->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return reject("a_q is not produced by a standalone DynamicQuantizeLinear");
  }

  const OrtNodeUnit& dql = *dql_lookup.dql;
  const auto& dql_inputs = dql.Inputs();
  const auto& dql_outputs = dql.Outputs();
  if (dql_inputs.size() != 1 || dql_outputs.size() != 3) {
    return reject("DQL input/output count mismatch");
  }

  const std::string& a_q_name = dql_outputs[0].name;
  const std::string& a_scale_name = dql_outputs[1].name;
  const std::string& a_zp_name = dql_outputs[2].name;

  if (mm_inputs[0].name != a_q_name) {
    return reject("MatMulInteger input[0] is not DQL.output[0]");
  }
  if (mm_inputs[2].name != a_zp_name) {
    return reject("MatMulInteger input[2] is not DQL.output[2]");
  }

  // Walk down MatMulInteger -> Cast -> requant_Mul.
  const OrtNodeUnit* cast = GetOnlyChildOfOutput(
      qnn_model_wrapper, matmul_integer_node_unit, mm_outputs[0],
      node_to_node_unit, node_unit_to_qnn_node_group);
  if (cast == nullptr || cast->OpType() != kOpCast ||
      cast->UnitType() != OrtNodeUnit::Type::SingleNode || cast->Outputs().size() != 1) {
    return reject("MatMulInteger output is not consumed by a single Cast");
  }
  {
    OrtNodeAttrHelper cast_attrs(*cast);
    if (cast_attrs.Get("to", static_cast<int64_t>(0)) !=
        static_cast<int64_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
      return reject("Cast target type is not FLOAT");
    }
  }

  const OrtNodeUnit* requant_mul = GetOnlyChildOfOutput(
      qnn_model_wrapper, *cast, cast->Outputs()[0],
      node_to_node_unit, node_unit_to_qnn_node_group);
  if (requant_mul == nullptr || requant_mul->OpType() != kOpMul ||
      requant_mul->UnitType() != OrtNodeUnit::Type::SingleNode ||
      requant_mul->Inputs().size() != 2 || requant_mul->Outputs().size() != 1) {
    return reject("Cast output is not consumed by a single Mul");
  }

  const std::string& cast_out_name = cast->Outputs()[0].name;
  const auto& rm_inputs = requant_mul->Inputs();
  const bool cast_is_input0 = (rm_inputs[0].name == cast_out_name);
  const bool cast_is_input1 = (rm_inputs[1].name == cast_out_name);
  if (cast_is_input0 == cast_is_input1) {
    return reject("requant_Mul does not have Cast.out as exactly one input");
  }
  const std::string& parallel_mul_out_name = cast_is_input0 ? rm_inputs[1].name : rm_inputs[0].name;

  const OrtNodeUnit* parallel_mul = GetParentOfInputByName(
      qnn_model_wrapper, *requant_mul, parallel_mul_out_name,
      node_to_node_unit, node_unit_to_qnn_node_group);
  if (parallel_mul == nullptr || parallel_mul->OpType() != kOpMul ||
      parallel_mul->UnitType() != OrtNodeUnit::Type::SingleNode ||
      parallel_mul->Inputs().size() != 2 || parallel_mul->Outputs().size() != 1) {
    return reject("parallel_Mul not found or has wrong shape");
  }

  if (GetOnlyChildOfOutput(qnn_model_wrapper, *parallel_mul, parallel_mul->Outputs()[0],
                           node_to_node_unit, node_unit_to_qnn_node_group) != requant_mul) {
    return reject("parallel_Mul has consumers other than requant_Mul");
  }

  const auto& pm_inputs = parallel_mul->Inputs();
  const bool a_scale_is_pm_input0 = (pm_inputs[0].name == a_scale_name);
  const bool a_scale_is_pm_input1 = (pm_inputs[1].name == a_scale_name);
  if (a_scale_is_pm_input0 == a_scale_is_pm_input1) {
    return reject("parallel_Mul does not have a_scale as exactly one input");
  }
  const OrtNodeUnitIODef& b_scale_def = a_scale_is_pm_input0 ? pm_inputs[1] : pm_inputs[0];
  if (!qnn_model_wrapper.IsConstantInput(b_scale_def.name)) {
    return reject("B_scale is not a constant initializer");
  }
  TensorInfo b_scale_info{};
  if (!qnn_model_wrapper.GetTensorInfo(b_scale_def, b_scale_info).IsOK()) {
    return reject("failed to get TensorInfo for B_scale");
  }
  if (b_scale_info.qnn_data_type != QNN_DATATYPE_FLOAT_32) {
    return reject("B_scale is not float32");
  }
  // Accepted shapes: scalar, [1], [N], or [1, N]. N is the output channel of [K, N].
  const uint32_t out_channels = b_info.shape[1];
  const auto& bs_shape = b_scale_info.shape;
  const bool b_scale_ok =
      bs_shape.empty() ||
      (bs_shape.size() == 1 && (bs_shape[0] == 1 || bs_shape[0] == out_channels)) ||
      (bs_shape.size() == 2 && bs_shape[0] == 1 && bs_shape[1] == out_channels);
  if (!b_scale_ok) {
    return reject("B_scale shape is not scalar/[N]/[1,N]");
  }

  // Optional trailing Add(requant_Mul.out, Bias_init).
  const OrtNodeUnit* add_bias = nullptr;
  std::string bias_name;
  std::string terminator_output_name = requant_mul->Outputs()[0].name;
  if (const OrtNodeUnit* maybe_add = GetOnlyChildOfOutput(
          qnn_model_wrapper, *requant_mul, requant_mul->Outputs()[0],
          node_to_node_unit, node_unit_to_qnn_node_group);
      maybe_add != nullptr && maybe_add->OpType() == kOpAdd) {
    if (maybe_add->UnitType() != OrtNodeUnit::Type::SingleNode ||
        maybe_add->Inputs().size() != 2 || maybe_add->Outputs().size() != 1) {
      return reject("Add node has wrong shape");
    }
    const auto& add_inputs = maybe_add->Inputs();
    const std::string& rm_out = requant_mul->Outputs()[0].name;
    const int bias_idx = (add_inputs[0].name == rm_out) ? 1 : 0;
    if (add_inputs[1 - bias_idx].name != rm_out) {
      return reject("Add inputs do not contain requant_Mul output");
    }
    const OrtNodeUnitIODef& bias_def = add_inputs[bias_idx];
    if (!qnn_model_wrapper.IsConstantInput(bias_def.name)) {
      return reject("Bias is not a constant initializer");
    }
    TensorInfo bias_info{};
    if (!qnn_model_wrapper.GetTensorInfo(bias_def, bias_info).IsOK()) {
      return reject("failed to get TensorInfo for Bias");
    }
    if (bias_info.qnn_data_type != QNN_DATATYPE_FLOAT_32) {
      return reject("Bias is not float32");
    }
    // Accepted bias shapes: [N] or [1, N] (broadcast over [..., M, N] outputs).
    const auto& bsh = bias_info.shape;
    const bool bias_ok =
        (bsh.size() == 1 && bsh[0] == out_channels) ||
        (bsh.size() == 2 && bsh[0] == 1 && bsh[1] == out_channels);
    if (!bias_ok) {
      return reject("Bias shape is not [N] or [1,N]");
    }
    add_bias = maybe_add;
    bias_name = bias_def.name;
    terminator_output_name = maybe_add->Outputs()[0].name;
  }

  // DQL outputs may only feed sanctioned consumers. Any other consumer means we can't bypass
  // DQL safely (claiming DQL would leave an external reader with a missing source).
  {
    const std::vector<Ort::ConstValueInfo> dql_outs = Ort::ConstNode(&dql.GetNode()).GetOutputs();
    if (dql_outs.size() != 3) {
      return reject("DQL does not have 3 outputs");
    }
    if (!ConsumersAreAllMatMulIntegers(dql_outs[0], node_to_node_unit)) {
      return reject("a_q has a consumer that is not a MatMulInteger");
    }
    if (!ConsumersAreAllParallelMuls(dql_outs[1], qnn_model_wrapper, node_to_node_unit)) {
      return reject("a_scale has a consumer that is not a parallel_Mul");
    }
    if (!ConsumersAreAllMatMulIntegers(dql_outs[2], node_to_node_unit)) {
      return reject("a_zp has a consumer that is not a MatMulInteger");
    }
  }

  Pattern pattern{
      /*dql=*/dql_lookup.already_claimed_by_sibling ? nullptr : &dql,
      /*matmul_integer=*/&matmul_integer_node_unit,
      /*cast=*/cast,
      /*parallel_mul=*/parallel_mul,
      /*requant_mul=*/requant_mul,
      /*add_bias=*/add_bias,
      /*float_input_name=*/dql_inputs[0].name,
      /*b_scale_iodef=*/&b_scale_def,
      /*terminator_output_name=*/std::move(terminator_output_name),
      /*bias_name=*/std::move(bias_name),
      /*has_b_zp=*/has_b_zp,
  };

  auto fused = std::unique_ptr<DQMatMulIntegerFusion>(new DQMatMulIntegerFusion(std::move(pattern)));
  if (Ort::Status status = fused->CreateOrValidateOnQnn(qnn_model_wrapper, /*validate=*/true);
      !status.IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("DQMatMulIntegerFusion rejected by QNN validate: " + status.GetErrorMessage()).c_str());
    return nullptr;
  }
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "DQMatMulIntegerFusion matched and validated");
  return fused;
}

// ---------------------------------------------------------------------------
// Constructor / IQnnNodeGroup plumbing
// ---------------------------------------------------------------------------
DQMatMulIntegerFusion::DQMatMulIntegerFusion(Pattern pattern)
    : matmul_integer_(pattern.matmul_integer),
      requant_mul_(pattern.requant_mul),
      add_bias_(pattern.add_bias),
      float_input_name_(std::move(pattern.float_input_name)),
      b_scale_iodef_(pattern.b_scale_iodef),
      terminator_output_name_(std::move(pattern.terminator_output_name)),
      bias_name_(std::move(pattern.bias_name)),
      has_b_zp_(pattern.has_b_zp) {
  if (pattern.dql != nullptr) node_units_.push_back(pattern.dql);
  node_units_.push_back(pattern.matmul_integer);
  node_units_.push_back(pattern.cast);
  node_units_.push_back(pattern.parallel_mul);
  node_units_.push_back(pattern.requant_mul);
  if (pattern.add_bias != nullptr) node_units_.push_back(pattern.add_bias);
}

Ort::Status DQMatMulIntegerFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, /*validate=*/true);
}

Ort::Status DQMatMulIntegerFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, /*validate=*/false);
}

gsl::span<const OrtNodeUnit* const> DQMatMulIntegerFusion::GetNodeUnits() const {
  return gsl::make_span(node_units_);
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------
Ort::Status DQMatMulIntegerFusion::CreateOrValidateOnQnn(QnnModelWrapper& qmw, bool validate) const {
  const OrtNodeUnit& matmul_integer = *matmul_integer_;
  const auto& mm_inputs = matmul_integer.Inputs();
  const bool has_bias = (add_bias_ != nullptr);

  TensorInfo a_info{};
  RETURN_IF_ERROR(qmw.GetTensorInfo(mm_inputs[0], a_info));
  RETURN_IF_NOT(a_info.shape.size() >= 2, "Expected rank>=2 activation");

  TensorInfo b_info{};
  RETURN_IF_ERROR(qmw.GetTensorInfo(mm_inputs[1], b_info));
  RETURN_IF_NOT(b_info.shape.size() == 2 && b_info.is_initializer,
                "Expected rank-2 constant weight");

  const std::vector<uint32_t> a_shape(a_info.shape.begin(), a_info.shape.end());
  const std::vector<uint32_t> b_shape(b_info.shape.begin(), b_info.shape.end());
  const uint32_t n = b_shape[1];

  std::vector<uint8_t> b_quant_bytes;
  RETURN_IF_ERROR(qmw.UnpackInitializerData(b_info.initializer_tensor, b_quant_bytes));

  const OrtNodeUnitIODef* b_zp_iodef = has_b_zp_ ? &mm_inputs[3] : nullptr;
  QnnQuantParamsWrapper weight_qparams;
  RETURN_IF_ERROR(BuildWeightQuantParams(qmw, *b_scale_iodef_, b_zp_iodef, n, weight_qparams));
  const bool is_per_channel = weight_qparams.IsPerChannel();
  const bool is_signed_weight = (b_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_8 ||
                                 b_info.qnn_data_type == QNN_DATATYPE_INT_8);
  const Qnn_DataType_t weight_quant_qnn_type = is_signed_weight ? QNN_DATATYPE_SFIXED_POINT_8
                                                                : QNN_DATATYPE_UFIXED_POINT_8;

  const std::string node_base = utils::UniqueNameGenerator().New(matmul_integer);
  const std::string w_quant_name = node_base + "_w_quant";
  const std::string w_float_name = node_base + "_w_f32";
  const std::string deq_node_name = node_base + "_weight_dq";
  const std::string matmul_node_name = node_base;

  // Step 1: register the float activation tensor (no Transpose - MatMul is layout-agnostic).
  if (!qmw.IsQnnTensorWrapperExist(float_input_name_)) {
    const Qnn_TensorType_t input_type = qmw.IsGraphInput(float_input_name_)
                                            ? QNN_TENSOR_TYPE_APP_WRITE
                                            : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper input_tensor(float_input_name_, input_type, QNN_DATATYPE_FLOAT_32,
                                  QnnQuantParamsWrapper(), std::vector<uint32_t>(a_shape));
    if (!validate) {
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(input_tensor)),
                    "Failed to add float input tensor");
    }
  }

  // Step 2: weight handed to MatMul as a float [K, N] tensor.
  //   Per-tensor:  static int8/uint8 weight + Dequantize op produces a NATIVE float weight.
  //   Per-channel: QNN's Dequantize op does not accept per-channel quantized inputs;
  //                pre-dequantize int8/uint8 -> float offline and emit a STATIC float weight directly.
  std::vector<uint8_t> per_channel_float_bytes;
  if (!is_per_channel) {
    QnnTensorWrapper w_quant(w_quant_name, QNN_TENSOR_TYPE_STATIC,
                             weight_quant_qnn_type, weight_qparams.Copy(),
                             std::vector<uint32_t>(b_shape), std::move(b_quant_bytes));
    QnnTensorWrapper w_float_native(w_float_name, QNN_TENSOR_TYPE_NATIVE,
                                    QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(b_shape));
    if (validate) {
      RETURN_IF_ERROR(qmw.ValidateQnnNode(deq_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                          QNN_OP_DEQUANTIZE,
                                          {w_quant.GetQnnTensor()},
                                          {w_float_native.GetQnnTensor()}, {}));
    } else {
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(w_quant)),
                    "Failed to add quantized weight tensor");
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(w_float_native)),
                    "Failed to add float weight tensor");
      RETURN_IF_NOT(qmw.CreateQnnNode(deq_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                      QNN_OP_DEQUANTIZE,
                                      {w_quant_name}, {w_float_name},
                                      {}, /*do_op_validation=*/false),
                    "Failed to create weight Dequantize node");
    }
  } else {
    std::vector<uint8_t> float_bytes;
    RETURN_IF_ERROR(PreDequantizePerChannelWeight(qmw, *b_scale_iodef_, b_zp_iodef,
                                                  is_signed_weight,
                                                  n, b_quant_bytes, float_bytes));
    if (validate) {
      per_channel_float_bytes = std::move(float_bytes);
    } else {
      QnnTensorWrapper w_float_static(w_float_name, QNN_TENSOR_TYPE_STATIC,
                                      QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                      std::vector<uint32_t>(b_shape), std::move(float_bytes));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(w_float_static)),
                    "Failed to add pre-dequantized float weight tensor");
    }
  }

  // Step 3: MatMul. Output shape comes from the terminator's ONNX shape (downstream consumers
  // expect that shape; the optional bias Add preserves shape, so terminator shape == MatMul
  // output shape).
  const OrtNodeUnitIODef& terminator_def = has_bias ? add_bias_->Outputs()[0]
                                                    : requant_mul_->Outputs()[0];
  std::vector<uint32_t> matmul_out_shape;
  RETURN_IF_NOT(qmw.GetOnnxShape(terminator_def.shape, matmul_out_shape),
                "Failed to get MatMul output shape from ONNX graph");
  RETURN_IF_NOT(matmul_out_shape.size() >= 2, "MatMul output must be rank>=2");

  // QNN MatMul: explicit transpose_in0 / transpose_in1 = false. Older QNN SDKs validate-fail
  // without these even though they default to false.
  Qnn_Scalar_t transpose_scalar = QNN_SCALAR_INIT;
  transpose_scalar.dataType = QNN_DATATYPE_BOOL_8;
  transpose_scalar.bool8Value = 0;
  QnnParamWrapper transpose_in0_param(matmul_integer.Index(), matmul_node_name,
                                      QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, transpose_scalar);
  QnnParamWrapper transpose_in1_param(matmul_integer.Index(), matmul_node_name,
                                      QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, transpose_scalar);

  // Activation handle - validate path uses GetQnnTensor() to assemble the validation call;
  // emit references the input by name (already added in step 1).
  QnnTensorWrapper matmul_in_a(float_input_name_,
                               qmw.IsGraphInput(float_input_name_) ? QNN_TENSOR_TYPE_APP_WRITE
                                                                   : QNN_TENSOR_TYPE_NATIVE,
                               QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                               std::vector<uint32_t>(a_shape));

  // Weight handle. Per-tensor: NATIVE (Dequantize produces it). Per-channel: STATIC pre-baked
  // float; bytes are attached for the validate handle so QNN's type-check sees a fully-formed
  // STATIC tensor.
  const Qnn_TensorType_t weight_type = is_per_channel ? QNN_TENSOR_TYPE_STATIC
                                                      : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper matmul_in_b(w_float_name, weight_type, QNN_DATATYPE_FLOAT_32,
                               QnnQuantParamsWrapper(),
                               std::vector<uint32_t>(b_shape),
                               std::move(per_channel_float_bytes));

  // MatMul output - if no trailing Add and the original ONNX output is a graph output, we
  // emit MatMul directly into the graph output tensor.
  const bool matmul_out_is_graph_output = !has_bias && qmw.IsGraphOutput(terminator_output_name_);
  const std::string matmul_out_name = has_bias ? (node_base + "_matmul_out")
                                               : terminator_output_name_;
  const Qnn_TensorType_t matmul_out_type = matmul_out_is_graph_output ? QNN_TENSOR_TYPE_APP_READ
                                                                      : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper matmul_out_tensor(matmul_out_name, matmul_out_type, QNN_DATATYPE_FLOAT_32,
                                     QnnQuantParamsWrapper(),
                                     std::vector<uint32_t>(matmul_out_shape));

  if (validate) {
    RETURN_IF_ERROR(qmw.ValidateQnnNode(matmul_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                        QNN_OP_MAT_MUL,
                                        {matmul_in_a.GetQnnTensor(), matmul_in_b.GetQnnTensor()},
                                        {matmul_out_tensor.GetQnnTensor()},
                                        {transpose_in0_param.GetQnnParam(),
                                         transpose_in1_param.GetQnnParam()}));
  } else {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(matmul_out_tensor)),
                  "Failed to add MatMul output tensor");

    std::vector<std::string> param_names = {transpose_in0_param.GetParamTensorName(),
                                            transpose_in1_param.GetParamTensorName()};
    qmw.AddParamWrapper(std::move(transpose_in0_param));
    qmw.AddParamWrapper(std::move(transpose_in1_param));

    RETURN_IF_NOT(qmw.CreateQnnNode(matmul_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                    QNN_OP_MAT_MUL,
                                    {float_input_name_, w_float_name},
                                    {matmul_out_name},
                                    std::move(param_names),
                                    /*do_op_validation=*/false),
                  "Failed to create MatMul node");
  }

  // Step 4: optional bias Add. Mirrors the original ONNX Add(requant_out, bias). Bias keeps
  // its original ONNX shape so QNN broadcasting matches ONNX semantics.
  if (has_bias) {
    const auto& add_inputs = add_bias_->Inputs();
    const std::string& rm_out_name = requant_mul_->Outputs()[0].name;
    const OrtNodeUnitIODef& bias_def =
        (add_inputs[0].name == rm_out_name) ? add_inputs[1] : add_inputs[0];

    QnnTensorWrapper bias_tensor;
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(bias_def, bias_tensor));

    const Qnn_TensorType_t add_out_type = qmw.IsGraphOutput(terminator_output_name_)
                                              ? QNN_TENSOR_TYPE_APP_READ
                                              : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper add_out_tensor(terminator_output_name_, add_out_type,
                                    QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(matmul_out_shape));
    QnnTensorWrapper add_lhs_handle(matmul_out_name, QNN_TENSOR_TYPE_NATIVE,
                                    QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(matmul_out_shape));

    const std::string add_node_name = node_base + "_bias_add";
    if (validate) {
      RETURN_IF_ERROR(qmw.ValidateQnnNode(add_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                          QNN_OP_ELEMENT_WISE_ADD,
                                          {add_lhs_handle.GetQnnTensor(), bias_tensor.GetQnnTensor()},
                                          {add_out_tensor.GetQnnTensor()}, {}));
    } else {
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias tensor");
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(add_out_tensor)),
                    "Failed to add bias-Add output tensor");
      RETURN_IF_NOT(qmw.CreateQnnNode(add_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                      QNN_OP_ELEMENT_WISE_ADD,
                                      {matmul_out_name, bias_def.name},
                                      {terminator_output_name_},
                                      {}, /*do_op_validation=*/false),
                    "Failed to create bias-Add node");
    }
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
