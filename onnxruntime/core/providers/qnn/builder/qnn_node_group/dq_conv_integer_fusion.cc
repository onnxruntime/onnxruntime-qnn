// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/dq_conv_integer_fusion.h"

#include <array>
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

constexpr std::array<uint32_t, 4> kPermNchwToNhwc = {0, 2, 3, 1};
constexpr std::array<uint32_t, 4> kPermNhwcToNchw = {0, 3, 1, 2};

constexpr char kOpConvInteger[] = "ConvInteger";
constexpr char kOpDynamicQuantizeLinear[] = "DynamicQuantizeLinear";
constexpr char kOpCast[] = "Cast";
constexpr char kOpMul[] = "Mul";
constexpr char kOpAdd[] = "Add";

constexpr std::string_view kFusionType = "DQConvIntegerFusion";

struct DqlLookupResult {
  const OrtNodeUnit* dql = nullptr;         // matched DQL NodeUnit, nullptr if not found
  bool already_claimed_by_sibling = false;  // true iff DQL is already claimed by another DQConvIntegerFusion
};

// Walks up `conv_integer`'s a_q input to find the producer DynamicQuantizeLinear NodeUnit.
// Tolerates DQL being claimed by a sibling DQConvIntegerFusion (multi-ConvInteger-shared-DQL
// case): only the first sibling actually claims DQL; later siblings detect the existing claim
// and skip the double-claim. Returns dql=nullptr if DQL is claimed by a non-DQConvIntegerFusion.
DqlLookupResult FindParentDqlForConvInteger(
    const OrtNodeUnit& conv_integer,
    const OrtNodeUnitIODef& a_q_input,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  DqlLookupResult result;

  const Ort::ConstNode conv_int_node(&conv_integer.GetNode());
  const OrtNode* dql_node_raw = nullptr;
  for (const Ort::ConstValueInfo& input_info : conv_int_node.GetInputs()) {
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
      return result;  // claimed by a non-DQConvIntegerFusion: cannot share
    }
    result.already_claimed_by_sibling = true;
  }

  result.dql = dql_it->second;
  return result;
}

// True if every consumer of `value_info` is a ConvInteger SingleNode and `value_info` is not
// itself a graph output. Used on DQL's a_q / a_zp outputs.
bool ConsumersAreAllConvIntegers(
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
    if (nu->OpType() != kOpConvInteger || nu->UnitType() != OrtNodeUnit::Type::SingleNode) {
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
                                 const std::string& name,
                                 std::vector<float>& out) {
  const OrtValueInfo* info = qmw.GetConstantTensor(name);
  RETURN_IF_NOT(info != nullptr, ("Constant tensor not found: " + name).c_str());

  const OrtApi& ort_api = qmw.GetOrtApi();
  const OrtTypeInfo* type_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(info, &type_info));
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info));

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(tensor_info, &elem_type));
  RETURN_IF_NOT(elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                ("Expected FLOAT initializer for " + name).c_str());

  std::vector<uint8_t> bytes;
  RETURN_IF_ERROR(qmw.UnpackInitializerData(info, bytes));
  RETURN_IF_NOT(bytes.size() % sizeof(float) == 0, "Unexpected byte count for float initializer");

  out.resize(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return Ort::Status();
}

// Reads a zero-point initializer (INT8 or UINT8) as int32 values. Empty `name` returns an
// empty vector. Per ONNX spec the zero-point dtype matches its corresponding tensor's dtype;
// callers should already have validated the weight dtype before calling this.
Ort::Status ReadZeroPointAsInt32(const QnnModelWrapper& qmw,
                                 const std::string& name,
                                 std::vector<int32_t>& out) {
  out.clear();
  if (name.empty()) {
    return Ort::Status();
  }
  const OrtValueInfo* info = qmw.GetConstantTensor(name);
  RETURN_IF_NOT(info != nullptr, ("Constant tensor not found: " + name).c_str());

  const OrtApi& ort_api = qmw.GetOrtApi();
  const OrtTypeInfo* type_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(info, &type_info));
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info));

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(tensor_info, &elem_type));
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

// Builds the weight quant params (int8 or uint8). ONNX zero-point is negated to match QNN's
// offset convention (QNN: x = scale * (q - offset); offset = -ONNX_zp). For per-channel B_scale
// the result is only used to detect per-channel via IsPerChannel(); the per-channel emission
// path pre-dequantizes to float offline and does not consume these quant params, so the axis
// value carried here is informational only.
Ort::Status BuildWeightQuantParams(const QnnModelWrapper& qmw,
                                   const std::string& b_scale_name,
                                   const std::string& b_zp_name_or_empty,
                                   uint32_t out_channels,
                                   QnnQuantParamsWrapper& out_params) {
  std::vector<float> scales;
  RETURN_IF_ERROR(ReadFloatInitializer(qmw, b_scale_name, scales));
  RETURN_IF_NOT(!scales.empty(), "B_scale has zero elements");

  std::vector<int32_t> offsets;
  RETURN_IF_ERROR(ReadZeroPointAsInt32(qmw, b_zp_name_or_empty, offsets));
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
                                     /*axis=*/0,
                                     /*is_int4=*/false);
  return Ort::Status();
}

// Pre-dequantizes per-channel int8 / uint8 HWCN weight bytes to float32 bytes (out_channels
// axis is HWCN's last axis).
Ort::Status PreDequantizePerChannelWeight(const QnnModelWrapper& qmw,
                                          const std::string& b_scale_name,
                                          const std::string& b_zp_name_or_empty,
                                          bool has_b_zp,
                                          bool is_signed_weight,
                                          uint32_t out_channels,
                                          const std::vector<uint8_t>& hwcn_quant_bytes,
                                          std::vector<uint8_t>& out_float_bytes) {
  std::vector<float> scales;
  RETURN_IF_ERROR(ReadFloatInitializer(qmw, b_scale_name, scales));
  RETURN_IF_NOT(scales.size() == static_cast<size_t>(out_channels),
                "Per-channel B_scale length mismatch");

  std::vector<int32_t> zps_onnx;
  if (has_b_zp) {
    RETURN_IF_ERROR(ReadZeroPointAsInt32(qmw, b_zp_name_or_empty, zps_onnx));
  }
  if (zps_onnx.empty()) {
    zps_onnx.assign(scales.size(), 0);
  } else if (zps_onnx.size() == 1) {
    zps_onnx.assign(scales.size(), zps_onnx[0]);
  } else {
    RETURN_IF_NOT(zps_onnx.size() == scales.size(), "Per-channel B_zp length mismatch");
  }

  const size_t num_elems = hwcn_quant_bytes.size();
  const size_t c_out = static_cast<size_t>(out_channels);
  RETURN_IF_NOT(c_out > 0 && num_elems % c_out == 0,
                "Weight byte count not divisible by C_out");

  // Dequantize into a typed float buffer first to avoid uint8_t-to-float aliasing issues,
  // then memcpy out to the byte buffer that QnnTensorWrapper expects.
  std::vector<float> floats(num_elems);
  if (is_signed_weight) {
    const int8_t* src = reinterpret_cast<const int8_t*>(hwcn_quant_bytes.data());
    for (size_t i = 0; i < num_elems; ++i) {
      const size_t c = i % c_out;
      floats[i] = scales[c] * static_cast<float>(static_cast<int32_t>(src[i]) - zps_onnx[c]);
    }
  } else {
    const uint8_t* src = hwcn_quant_bytes.data();
    for (size_t i = 0; i < num_elems; ++i) {
      const size_t c = i % c_out;
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
std::unique_ptr<IQnnNodeGroup> DQConvIntegerFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& conv_integer_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  auto reject = [&logger](std::string_view reason) -> std::unique_ptr<IQnnNodeGroup> {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                (std::string("DQConvIntegerFusion rejected: ").append(reason)).c_str());
    return nullptr;
  };

  // The 2nd GetCapability pass sees ConvInteger in kMSInternalNHWCDomain only if ORT's layout
  // transformer rewrote it; the EP suppresses that via ShouldConvertDataLayoutForOp, but as a
  // defense skip if it ever does fire here.
  if (conv_integer_node_unit.Domain() == kMSInternalNHWCDomain) {
    return nullptr;
  }

  if (conv_integer_node_unit.OpType() != kOpConvInteger ||
      conv_integer_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return reject("not a ConvInteger SingleNode");
  }

  const auto& ci_inputs = conv_integer_node_unit.Inputs();
  const auto& ci_outputs = conv_integer_node_unit.Outputs();
  if (ci_inputs.size() < 2 || ci_inputs.size() > 4 || ci_outputs.size() != 1) {
    return reject("ConvInteger input/output count mismatch");
  }

  TensorInfo a_info{};
  TensorInfo b_info{};
  if (!qnn_model_wrapper.GetTensorInfo(ci_inputs[0], a_info).IsOK() ||
      !qnn_model_wrapper.GetTensorInfo(ci_inputs[1], b_info).IsOK()) {
    return reject("failed to get TensorInfo for ConvInteger inputs");
  }
  if (a_info.shape.size() != 4 || b_info.shape.size() != 4) {
    return reject("ConvInteger rank != 4");
  }

  if (!qnn_model_wrapper.IsConstantInput(ci_inputs[1].name) || !b_info.is_initializer) {
    return reject("weight B is not a constant initializer");
  }
  if (b_info.qnn_data_type != QNN_DATATYPE_SFIXED_POINT_8 &&
      b_info.qnn_data_type != QNN_DATATYPE_INT_8 &&
      b_info.qnn_data_type != QNN_DATATYPE_UFIXED_POINT_8 &&
      b_info.qnn_data_type != QNN_DATATYPE_UINT_8) {
    return reject("weight B is not int8 or uint8");
  }

  const bool has_a_zp = ci_inputs.size() >= 3 && ci_inputs[2].Exists();
  const bool has_b_zp = ci_inputs.size() >= 4 && ci_inputs[3].Exists();
  if (has_b_zp && !qnn_model_wrapper.IsConstantInput(ci_inputs[3].name)) {
    return reject("B_zp is not a constant initializer");
  }

  // The fused float Conv uses the *pre-DQL* float input as activation; that input still carries
  // the offset DQL would otherwise factor out. ConvInteger must therefore consume A_zp from DQL
  // for the rewrite to be mathematically equivalent. Without A_zp the output diverges by
  // a_scale * sum(a_zp * (B - B_zp)).
  if (!has_a_zp) {
    return reject("ConvInteger has no A_zp input; fusion would change semantics");
  }

  {
    OrtNodeAttrHelper attrs(conv_integer_node_unit);
    if (attrs.Get("auto_pad", std::string("NOTSET")) != "NOTSET") {
      return reject("auto_pad != NOTSET");
    }
  }

  // Walk up to DQL. Custom lookup tolerates DQL being claimed by a sibling DQConvIntegerFusion.
  const DqlLookupResult dql_lookup = FindParentDqlForConvInteger(
      conv_integer_node_unit, ci_inputs[0], node_to_node_unit, node_unit_to_qnn_node_group);
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

  if (ci_inputs[0].name != a_q_name) {
    return reject("ConvInteger input[0] is not DQL.output[0]");
  }
  if (ci_inputs[2].name != a_zp_name) {
    return reject("ConvInteger input[2] is not DQL.output[2]");
  }

  // Walk down ConvInteger -> Cast -> requant_Mul.
  const OrtNodeUnit* cast = GetOnlyChildOfOutput(
      qnn_model_wrapper, conv_integer_node_unit, ci_outputs[0],
      node_to_node_unit, node_unit_to_qnn_node_group);
  if (cast == nullptr || cast->OpType() != kOpCast ||
      cast->UnitType() != OrtNodeUnit::Type::SingleNode || cast->Outputs().size() != 1) {
    return reject("ConvInteger output is not consumed by a single Cast");
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

  // Identify requant_Mul's inputs: one is Cast.out, the other is parallel_Mul.out.
  const std::string& cast_out_name = cast->Outputs()[0].name;
  const auto& rm_inputs = requant_mul->Inputs();
  const bool cast_is_input0 = (rm_inputs[0].name == cast_out_name);
  const bool cast_is_input1 = (rm_inputs[1].name == cast_out_name);
  if (cast_is_input0 == cast_is_input1) {
    return reject("requant_Mul does not have Cast.out as exactly one input");
  }
  const std::string& parallel_mul_out_name = cast_is_input0 ? rm_inputs[1].name : rm_inputs[0].name;

  // Walk up to parallel_Mul (the sibling Mul that produces parallel_Mul.out).
  const OrtNodeUnit* parallel_mul = GetParentOfInputByName(
      qnn_model_wrapper, *requant_mul, parallel_mul_out_name,
      node_to_node_unit, node_unit_to_qnn_node_group);
  if (parallel_mul == nullptr || parallel_mul->OpType() != kOpMul ||
      parallel_mul->UnitType() != OrtNodeUnit::Type::SingleNode ||
      parallel_mul->Inputs().size() != 2 || parallel_mul->Outputs().size() != 1) {
    return reject("parallel_Mul not found or has wrong shape");
  }

  // parallel_Mul's only consumer must be requant_Mul.
  if (GetOnlyChildOfOutput(qnn_model_wrapper, *parallel_mul, parallel_mul->Outputs()[0],
                           node_to_node_unit, node_unit_to_qnn_node_group) != requant_mul) {
    return reject("parallel_Mul has consumers other than requant_Mul");
  }

  // parallel_Mul takes (a_scale, B_scale_init); identify which input is which.
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
  // Accepted shapes: scalar, [1], [C_out], or [1, C_out, 1, 1].
  const uint32_t out_channels = b_info.shape[0];
  const auto& bs_shape = b_scale_info.shape;
  const bool b_scale_ok =
      bs_shape.empty() ||
      (bs_shape.size() == 1 && (bs_shape[0] == 1 || bs_shape[0] == out_channels)) ||
      (bs_shape.size() == 4 && bs_shape[0] == 1 && bs_shape[1] == out_channels &&
       bs_shape[2] == 1 && bs_shape[3] == 1);
  if (!b_scale_ok) {
    return reject("B_scale shape is not scalar/[C_out]/[1,C_out,1,1]");
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
    const auto& bsh = bias_info.shape;
    const bool bias_ok =
        (bsh.size() == 1 && bsh[0] == out_channels) ||
        (bsh.size() == 4 && bsh[0] == 1 && bsh[1] == out_channels && bsh[2] == 1 && bsh[3] == 1);
    if (!bias_ok) {
      return reject("Bias shape is not [C_out] or [1,C_out,1,1]");
    }
    add_bias = maybe_add;
    bias_name = bias_def.name;
    terminator_output_name = maybe_add->Outputs()[0].name;
  }

  // DQL outputs may only feed sanctioned consumers (this fusion's nodes plus, optionally,
  // sibling DQConvIntegerFusion candidates that share the same DQL). Any other consumer means
  // we can't bypass DQL safely, so reject.
  {
    const std::vector<Ort::ConstValueInfo> dql_outs = Ort::ConstNode(&dql.GetNode()).GetOutputs();
    if (dql_outs.size() != 3) {
      return reject("DQL does not have 3 outputs");
    }
    if (!ConsumersAreAllConvIntegers(dql_outs[0], node_to_node_unit)) {
      return reject("a_q has a consumer that is not a ConvInteger");
    }
    if (!ConsumersAreAllParallelMuls(dql_outs[1], qnn_model_wrapper, node_to_node_unit)) {
      return reject("a_scale has a consumer that is not a parallel_Mul");
    }
    if (!ConsumersAreAllConvIntegers(dql_outs[2], node_to_node_unit)) {
      return reject("a_zp has a consumer that is not a ConvInteger");
    }
  }

  // Only the first sibling fusion claims DQL; subsequent siblings ride on that claim.
  Pattern pattern{
      /*dql=*/dql_lookup.already_claimed_by_sibling ? nullptr : &dql,
      /*conv_integer=*/&conv_integer_node_unit,
      /*cast=*/cast,
      /*parallel_mul=*/parallel_mul,
      /*requant_mul=*/requant_mul,
      /*add_bias=*/add_bias,
      /*float_input_name=*/dql_inputs[0].name,
      /*b_scale_name=*/b_scale_def.name,
      /*terminator_output_name=*/std::move(terminator_output_name),
      /*bias_name=*/std::move(bias_name),
      /*has_b_zp=*/has_b_zp,
  };

  auto fused = std::unique_ptr<DQConvIntegerFusion>(new DQConvIntegerFusion(std::move(pattern)));
  if (Ort::Status status = fused->CreateOrValidateOnQnn(qnn_model_wrapper, /*validate=*/true);
      !status.IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("DQConvIntegerFusion rejected by QNN validate: " + status.GetErrorMessage()).c_str());
    return nullptr;
  }
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "DQConvIntegerFusion matched and validated");
  return fused;
}

// ---------------------------------------------------------------------------
// Constructor / IQnnNodeGroup plumbing
// ---------------------------------------------------------------------------
DQConvIntegerFusion::DQConvIntegerFusion(Pattern pattern)
    : conv_integer_(pattern.conv_integer),
      requant_mul_(pattern.requant_mul),
      add_bias_(pattern.add_bias),
      float_input_name_(std::move(pattern.float_input_name)),
      b_scale_name_(std::move(pattern.b_scale_name)),
      terminator_output_name_(std::move(pattern.terminator_output_name)),
      bias_name_(std::move(pattern.bias_name)),
      has_b_zp_(pattern.has_b_zp) {
  // node_units_ records every NodeUnit this fusion claims, for ORT bookkeeping.
  // Order is not required to be topological by the framework.
  if (pattern.dql != nullptr) node_units_.push_back(pattern.dql);
  node_units_.push_back(pattern.conv_integer);
  node_units_.push_back(pattern.cast);
  node_units_.push_back(pattern.parallel_mul);
  node_units_.push_back(pattern.requant_mul);
  if (pattern.add_bias != nullptr) node_units_.push_back(pattern.add_bias);
}

Ort::Status DQConvIntegerFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, /*validate=*/true);
}

Ort::Status DQConvIntegerFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, /*validate=*/false);
}

gsl::span<const OrtNodeUnit* const> DQConvIntegerFusion::GetNodeUnits() const {
  return gsl::make_span(node_units_);
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------
Ort::Status DQConvIntegerFusion::CreateOrValidateOnQnn(QnnModelWrapper& qmw, bool validate) const {
  const OrtNodeUnit& conv_integer = *conv_integer_;
  const auto& ci_inputs = conv_integer.Inputs();
  const bool has_bias = (add_bias_ != nullptr);

  TensorInfo a_info{};
  RETURN_IF_ERROR(qmw.GetTensorInfo(ci_inputs[0], a_info));
  RETURN_IF_NOT(a_info.shape.size() == 4, "Expected rank-4 activation");

  TensorInfo b_info{};
  RETURN_IF_ERROR(qmw.GetTensorInfo(ci_inputs[1], b_info));
  RETURN_IF_NOT(b_info.shape.size() == 4 && b_info.is_initializer,
                "Expected rank-4 constant weight");

  const std::vector<uint32_t> nchw_in_shape(a_info.shape.begin(), a_info.shape.end());
  std::vector<uint32_t> nhwc_in_shape(4);
  RETURN_IF_ERROR(utils::NchwShapeToNhwc<uint32_t>(nchw_in_shape, nhwc_in_shape));

  std::vector<uint32_t> hwcn_weight_shape(4);
  RETURN_IF_ERROR(utils::NchwShapeToHwcn<uint32_t>(b_info.shape, hwcn_weight_shape));

  std::vector<uint8_t> hwcn_weight_bytes;
  RETURN_IF_ERROR(utils::TransposeFromNchwToHwcn(qmw, b_info.initializer_tensor,
                                                 hwcn_weight_bytes, /*is_3d=*/false));

  // Build weight quant params from B_scale / B_zp. Used only on the per-tensor path (the
  // per-channel path pre-dequantizes the weight bytes offline and does not consume these).
  const std::string b_zp_name = has_b_zp_ ? ci_inputs[3].name : std::string{};
  QnnQuantParamsWrapper weight_qparams;
  RETURN_IF_ERROR(BuildWeightQuantParams(qmw, b_scale_name_, b_zp_name,
                                         b_info.shape[0], weight_qparams));
  const bool is_per_channel = weight_qparams.IsPerChannel();
  const bool is_signed_weight = (b_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_8 ||
                                 b_info.qnn_data_type == QNN_DATATYPE_INT_8);
  const Qnn_DataType_t weight_quant_qnn_type = is_signed_weight ? QNN_DATATYPE_SFIXED_POINT_8
                                                                : QNN_DATATYPE_UFIXED_POINT_8;

  const std::string node_base = utils::UniqueNameGenerator().New(conv_integer);
  const std::string t_in_name = node_base + "_input_nhwc";
  const std::string w_hwcn_quant_name = node_base + "_w_hwcn_quant";
  const std::string w_hwcn_f32_name = node_base + "_w_hwcn_f32";
  const std::string conv_out_nhwc_name = node_base + "_conv_nhwc";
  const std::string deq_node_name = node_base + "_weight_dq";
  const std::string conv_node_name = node_base;

  // Step 1: input Transpose NCHW -> NHWC.
  {
    std::vector<uint32_t> perm(kPermNchwToNhwc.begin(), kPermNchwToNhwc.end());
    RETURN_IF_ERROR(qmw.AddTransposeNode(conv_integer.Index(),
                                         float_input_name_, t_in_name,
                                         nchw_in_shape, perm, nhwc_in_shape,
                                         QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                         validate,
                                         /*is_for_input=*/qmw.IsGraphInput(float_input_name_),
                                         /*is_for_output=*/false));
  }

  // Step 2: weight handed to Conv2d as a float HWCN tensor.
  //   Per-tensor:  static int8/uint8 weight + Dequantize op produces a NATIVE float weight.
  //   Per-channel: QNN's Dequantize op does not accept per-channel quantized inputs;
  //                pre-dequantize int8/uint8 -> float offline and emit a STATIC float weight directly.
  std::vector<uint8_t> per_channel_float_bytes;  // populated only on per-channel + validate
  if (!is_per_channel) {
    QnnTensorWrapper w_quant(w_hwcn_quant_name, QNN_TENSOR_TYPE_STATIC,
                             weight_quant_qnn_type, weight_qparams.Copy(),
                             std::vector<uint32_t>(hwcn_weight_shape),
                             std::move(hwcn_weight_bytes));
    QnnTensorWrapper w_float_native(w_hwcn_f32_name, QNN_TENSOR_TYPE_NATIVE,
                                    QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(hwcn_weight_shape));
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
                                      {w_hwcn_quant_name}, {w_hwcn_f32_name},
                                      {}, /*do_op_validation=*/false),
                    "Failed to create weight Dequantize node");
    }
  } else {
    std::vector<uint8_t> float_bytes;
    RETURN_IF_ERROR(PreDequantizePerChannelWeight(qmw, b_scale_name_, b_zp_name, has_b_zp_,
                                                  is_signed_weight,
                                                  b_info.shape[0], hwcn_weight_bytes,
                                                  float_bytes));
    if (validate) {
      // Bytes are needed below to construct a STATIC handle for QNN's type check.
      per_channel_float_bytes = std::move(float_bytes);
    } else {
      QnnTensorWrapper w_float_static(w_hwcn_f32_name, QNN_TENSOR_TYPE_STATIC,
                                      QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                      std::vector<uint32_t>(hwcn_weight_shape),
                                      std::move(float_bytes));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(w_float_static)),
                    "Failed to add pre-dequantized float weight tensor");
    }
  }

  // Step 3: Conv2d / DepthWiseConv2d.
  OrtNodeAttrHelper ci_attrs(conv_integer);
  std::vector<uint32_t> strides = ci_attrs.Get("strides", std::vector<uint32_t>{1u, 1u});
  std::vector<uint32_t> dilations = ci_attrs.Get("dilations", std::vector<uint32_t>{1u, 1u});
  std::vector<uint32_t> pads = ci_attrs.Get("pads", std::vector<uint32_t>{0u, 0u, 0u, 0u});
  const uint32_t group = static_cast<uint32_t>(ci_attrs.Get("group", static_cast<int64_t>(1)));
  RETURN_IF_NOT(strides.size() == 2 && dilations.size() == 2 && pads.size() == 4,
                "Conv2D attributes must be 2D");

  // ONNX `pads` is [h_begin, w_begin, h_end, w_end]; QNN pad_amount is [[h_b,h_e],[w_b,w_e]].
  std::vector<uint32_t> pad_amount = {pads[0], pads[2], pads[1], pads[3]};

  QnnParamWrapper stride_param(conv_integer.Index(), conv_node_name, QNN_OP_CONV_2D_PARAM_STRIDE,
                               {static_cast<uint32_t>(strides.size())}, std::move(strides));
  QnnParamWrapper dilation_param(conv_integer.Index(), conv_node_name, QNN_OP_CONV_2D_PARAM_DILATION,
                                 {static_cast<uint32_t>(dilations.size())}, std::move(dilations));
  QnnParamWrapper pad_param(conv_integer.Index(), conv_node_name, QNN_OP_CONV_2D_PARAM_PAD_AMOUNT,
                            {2u, 2u}, std::move(pad_amount));

  Qnn_Scalar_t group_scalar = QNN_SCALAR_INIT;
  group_scalar.dataType = QNN_DATATYPE_UINT_32;
  group_scalar.uint32Value = group;
  QnnParamWrapper group_param(conv_integer.Index(), conv_node_name, QNN_OP_CONV_2D_PARAM_GROUP,
                              group_scalar);

  const uint32_t in_channels = nchw_in_shape[1];
  const uint32_t out_channels = b_info.shape[0];
  const bool is_depthwise = (group == in_channels) && (group == out_channels);
  const char* conv_qnn_op = is_depthwise ? QNN_OP_DEPTH_WISE_CONV_2D : QNN_OP_CONV_2D;

  // Conv output NHWC shape derived from the terminator's NCHW shape (downstream consumers
  // expect NCHW; the Add/Mul preserve shape, so terminator NCHW == Conv output NCHW).
  const OrtNodeUnitIODef& terminator_def = has_bias ? add_bias_->Outputs()[0]
                                                    : requant_mul_->Outputs()[0];
  std::vector<uint32_t> terminator_nchw_shape;
  RETURN_IF_NOT(qmw.GetOnnxShape(terminator_def.shape, terminator_nchw_shape),
                "Failed to get Conv output shape from ONNX graph");
  RETURN_IF_NOT(terminator_nchw_shape.size() == 4, "Conv output must be rank-4");

  std::vector<uint32_t> conv_out_nhwc_shape(4);
  RETURN_IF_ERROR(utils::NchwShapeToNhwc<uint32_t>(terminator_nchw_shape, conv_out_nhwc_shape));

  QnnTensorWrapper conv_out_tensor(conv_out_nhwc_name, QNN_TENSOR_TYPE_NATIVE,
                                   QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                   std::vector<uint32_t>(conv_out_nhwc_shape));

  // Activation handle (NHWC) - validate-time only; emit references it by name.
  QnnTensorWrapper conv_in_activation(t_in_name, QNN_TENSOR_TYPE_NATIVE,
                                      QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                      std::vector<uint32_t>(nhwc_in_shape));

  // Weight handle for Conv2d.
  //   Per-tensor: NATIVE (Dequantize produces it).
  //   Per-channel: STATIC pre-baked float; bytes are attached for the validate handle so QNN's
  //                type check sees a fully-formed STATIC tensor.
  const Qnn_TensorType_t conv_weight_type = is_per_channel ? QNN_TENSOR_TYPE_STATIC
                                                           : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper conv_in_weight(w_hwcn_f32_name, conv_weight_type, QNN_DATATYPE_FLOAT_32,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(hwcn_weight_shape),
                                  std::move(per_channel_float_bytes));

  std::vector<Qnn_Param_t> conv_qparams = {stride_param.GetQnnParam(),
                                           pad_param.GetQnnParam(),
                                           dilation_param.GetQnnParam()};
  if (!is_depthwise) {
    conv_qparams.push_back(group_param.GetQnnParam());
  }

  if (validate) {
    RETURN_IF_ERROR(qmw.ValidateQnnNode(conv_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, conv_qnn_op,
                                        {conv_in_activation.GetQnnTensor(),
                                         conv_in_weight.GetQnnTensor()},
                                        {conv_out_tensor.GetQnnTensor()},
                                        std::move(conv_qparams)));
  } else {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(conv_out_tensor)),
                  "Failed to add Conv output tensor");
    std::vector<std::string> param_names = {stride_param.GetParamTensorName(),
                                            pad_param.GetParamTensorName(),
                                            dilation_param.GetParamTensorName()};
    qmw.AddParamWrapper(std::move(stride_param));
    qmw.AddParamWrapper(std::move(pad_param));
    qmw.AddParamWrapper(std::move(dilation_param));
    if (!is_depthwise) {
      param_names.push_back(group_param.GetParamTensorName());
      qmw.AddParamWrapper(std::move(group_param));
    }
    RETURN_IF_NOT(qmw.CreateQnnNode(conv_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, conv_qnn_op,
                                    {t_in_name, w_hwcn_f32_name},
                                    {conv_out_nhwc_name},
                                    std::move(param_names),
                                    /*do_op_validation=*/false),
                  "Failed to create Conv2D node");
  }

  // Step 4: output Transpose NHWC -> NCHW. Writes the terminator output directly when there is
  // no trailing Add; otherwise writes an intermediate tensor that the Add then consumes.
  const std::string conv_out_nchw_name = has_bias ? (node_base + "_conv_nchw")
                                                  : terminator_output_name_;
  {
    std::vector<uint32_t> perm(kPermNhwcToNchw.begin(), kPermNhwcToNchw.end());
    const bool is_graph_output = !has_bias && qmw.IsGraphOutput(terminator_output_name_);
    RETURN_IF_ERROR(qmw.AddTransposeNode(conv_integer.Index(),
                                         conv_out_nhwc_name, conv_out_nchw_name,
                                         conv_out_nhwc_shape, perm, terminator_nchw_shape,
                                         QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                         validate,
                                         /*is_for_input=*/false,
                                         /*is_for_output=*/is_graph_output));
  }

  // Step 5: optional bias Add. Mirrors the original ONNX Add(requant_out, bias). Bias keeps its
  // original ONNX shape so QNN broadcasting matches ONNX semantics.
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
                                    std::vector<uint32_t>(terminator_nchw_shape));
    QnnTensorWrapper add_lhs_handle(conv_out_nchw_name, QNN_TENSOR_TYPE_NATIVE,
                                    QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(terminator_nchw_shape));

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
                                      {conv_out_nchw_name, bias_def.name},
                                      {terminator_output_name_},
                                      {}, /*do_op_validation=*/false),
                    "Failed to create bias-Add node");
    }
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
