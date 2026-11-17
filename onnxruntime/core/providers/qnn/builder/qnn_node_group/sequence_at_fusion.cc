// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/sequence_at_fusion.h"

#include <gsl/gsl>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

// ---------------------------------------------------------------------------
// CreateOrValidateOnQnn – forward declaration and convenience macros
// ---------------------------------------------------------------------------

#define ValidateOnQnn(qmw, seq_construct_unit, seq_at_unit) \
  CreateOrValidateOnQnn((qmw), (seq_construct_unit), (seq_at_unit), true)
#define CreateOnQnn(qmw, seq_construct_unit, seq_at_unit) \
  CreateOrValidateOnQnn((qmw), (seq_construct_unit), (seq_at_unit), false)

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         const OrtNodeUnit& seq_construct_unit,
                                         const OrtNodeUnit& seq_at_unit,
                                         bool validate);

// ---------------------------------------------------------------------------
// Helper: check whether an ONNX element data type is within the validated
// set for this fusion.
//
// Only types with passing HTP unit tests are listed here.  Other types that
// are theoretically supported by the underlying QNN ops (int8, uint8, int16,
// uint16, int32, int64, bool; GPU backend) should be added once test coverage
// is in place.  Types absent from the allowlist fall back to CPU EP.
// ---------------------------------------------------------------------------

static bool IsElementTypeSupported(ONNXTensorElementDataType etype) {
  switch (etype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:    // float32 — validated on HTP
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:  // float16 — validated on HTP
      return true;
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// TryFusion
// ---------------------------------------------------------------------------

std::unique_ptr<IQnnNodeGroup> SequenceAtFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& seq_construct_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Step 1: confirm we are starting from a SequenceConstruct SingleNode.
  if (seq_construct_unit.OpType() != "SequenceConstruct" ||
      seq_construct_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Step 2: Find the unique SequenceAt consumer of SequenceConstruct's sequence output.
  //
  // GetConsumers() on a sequence-typed ValueInfo is unreliable — the EP plugin API
  // only enumerates consumers of tensor-typed edges. The sequence output of
  // SequenceConstruct is seq(tensor(T)), so GetConsumers() silently returns an empty
  // list. Instead, get the sequence output name from the raw Node API and scan
  // node_to_node_unit for every node whose first raw input matches the sequence output.
  //
  // We require:
  //   (a) The sequence output is NOT a graph output — if it were, the graph or another
  //       EP segment is observing it as a sequence value that we would silently drop.
  //   (b) Exactly ONE consumer of the sequence output exists across the entire graph.
  //       Other sequence ops (SequenceLength, ConcatFromSequence, SequenceInsert, etc.)
  //       also take seq(tensor(T)) as input; if any of them co-consume the output, fusing
  //       SequenceConstruct away would leave them with a dangling edge QNN cannot produce.
  //   (c) That single consumer has op type "SequenceAt" and is a SingleNode unit.
  const Ort::ConstNode sc_node(&seq_construct_unit.GetNode());
  const auto sc_outputs = sc_node.GetOutputs();
  if (sc_outputs.size() != 1) {
    return nullptr;
  }
  const std::string seq_output_name = sc_outputs[0].GetName();

  // (a) Reject if the sequence tensor is itself a graph output.
  if (qnn_model_wrapper.IsGraphOutput(seq_output_name)) {
    return nullptr;
  }

  // (b)+(c) Exactly one consumer, must be SequenceAt.
  // FindUniqueSequenceConsumer uses raw GetInputs() to bypass GetConsumers()
  // which silently returns empty for seq(tensor(T)) edges.
  const OrtNodeUnit* seq_at_unit_ptr =
      FindUniqueSequenceConsumer(seq_output_name, "SequenceAt", node_to_node_unit);
  if (seq_at_unit_ptr == nullptr) {
    return nullptr;
  }

  // Step 3: make sure neither node unit is already claimed by another group.
  if (node_unit_to_qnn_node_group.count(&seq_construct_unit) != 0 ||
      node_unit_to_qnn_node_group.count(seq_at_unit_ptr) != 0) {
    return nullptr;
  }

  // Step 4: validate all N element-tensor inputs of SequenceConstruct.
  const auto& sc_inputs = seq_construct_unit.Inputs();
  const size_t N = sc_inputs.size();
  if (N == 0) {
    return nullptr;
  }

  // All inputs must have an element type supported by QNN.
  for (size_t i = 0; i < N; ++i) {
    if (!IsElementTypeSupported(sc_inputs[i].type)) {
      return nullptr;
    }
  }

  // All inputs must share the same static shape.
  std::vector<uint32_t> elem_shape;
  if (!qnn_model_wrapper.GetOnnxShape(sc_inputs[0].shape, elem_shape)) {
    return nullptr;
  }
  for (size_t i = 1; i < N; ++i) {
    std::vector<uint32_t> shape_i;
    if (!qnn_model_wrapper.GetOnnxShape(sc_inputs[i].shape, shape_i)) {
      return nullptr;
    }
    if (shape_i != elem_shape) {
      return nullptr;
    }
  }

  // Step 5: validate position input of SequenceAt.
  // SequenceAt inputs: [input_sequence (seq), position (scalar int32/int64)].
  // The first input is the sequence-typed edge; the second is the position.
  const auto& sa_inputs = seq_at_unit_ptr->Inputs();
  if (sa_inputs.size() < 2) {
    return nullptr;
  }
  const auto& pos_input = sa_inputs[1];
  if (pos_input.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      pos_input.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return nullptr;
  }
  // Position must be a scalar (rank-0 tensor): shape must be statically known and empty.
  // Reject nullopt — an unknown shape would cause GetTensorInfo → GetOnnxShape to fail
  // inside CreateOrValidateOnQnn, producing a hard AddToModelBuilder error instead of a
  // graceful CPU fallback.  Reject non-empty shape — a non-scalar position is invalid
  // per ONNX spec.
  // Do NOT use GetOnnxShape here — it converts {} to {1} for QNN, which would
  // incorrectly flag a valid scalar as a non-scalar 1D tensor.
  if (!pos_input.shape.has_value() || !pos_input.shape->empty()) {
    return nullptr;
  }

  // If the position is a static initializer, validate its value now so that an
  // out-of-range pos causes TryFusion to return nullptr (graceful CPU fallback)
  // rather than having AddToModelBuilder fail with a hard error at session creation.
  // All range checks are performed in int64 before any narrowing to int32, so that
  // values like 2^32+1 (which would silently truncate to 1 after static_cast<int32_t>)
  // are correctly rejected.
  {
    TensorInfo pos_info_check{};
    if (qnn_model_wrapper.GetTensorInfo(pos_input, pos_info_check).IsOK() &&
        pos_info_check.is_initializer) {
      std::vector<uint8_t> raw_bytes;
      if (!qnn_model_wrapper.UnpackInitializerData(pos_info_check.initializer_tensor,
                                                   raw_bytes)
               .IsOK()) {
        return nullptr;
      }

      int64_t pos_val64 = 0;
      if (pos_input.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        if (raw_bytes.size() < sizeof(int64_t)) return nullptr;  // short / corrupt initializer
        std::memcpy(&pos_val64, raw_bytes.data(), sizeof(int64_t));
      } else {  // int32
        if (raw_bytes.size() < sizeof(int32_t)) return nullptr;
        int32_t val32 = 0;
        std::memcpy(&val32, raw_bytes.data(), sizeof(int32_t));
        pos_val64 = static_cast<int64_t>(val32);
      }

      // Valid range (before normalization): [-N, N-1].
      // Also guard against values that exceed INT32_MAX — after normalization
      // pos += N the result must still fit in int32 (N itself is a small count
      // of graph inputs, always well within INT32_MAX, so this only catches
      // wildly out-of-range int64 literals).
      const int64_t n64 = static_cast<int64_t>(N);
      if (pos_val64 < -n64 || pos_val64 >= n64 ||
          pos_val64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return nullptr;
      }
    }
  }

  // Step 6: QNN op validation (CreateOrValidateOnQnn in validate=true mode).
  auto group = std::make_unique<SequenceAtFusion>(seq_construct_unit, *seq_at_unit_ptr);
  if (!group->IsSupported(qnn_model_wrapper, logger).IsOK()) {
    return nullptr;
  }
  return group;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SequenceAtFusion::SequenceAtFusion(const OrtNodeUnit& seq_construct_unit,
                                   const OrtNodeUnit& seq_at_unit)
    : node_units_{&seq_construct_unit, &seq_at_unit} {}

// ---------------------------------------------------------------------------
// IQnnNodeGroup interface
// ---------------------------------------------------------------------------

gsl::span<const OrtNodeUnit* const> SequenceAtFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>(node_units_.data(), node_units_.size());
}

const OrtNodeUnit* SequenceAtFusion::GetTargetNodeUnit() const {
  // SequenceAt is the convergence point of all input paths: the N element tensors arrive
  // via SequenceConstruct and the position arrives directly. Using SequenceAt as the target
  // ensures GetQnnNodeGroupsImpl emits this group only after both SequenceConstruct and any
  // in-partition node that produces `pos` have already been emitted, regardless of the order
  // those producer nodes appear in the model.
  return node_units_[1];
}

Ort::Status SequenceAtFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

Ort::Status SequenceAtFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

// ---------------------------------------------------------------------------
// CreateOrValidateOnQnn
//
// Emits:
//   For each input ti (i = 0..N-1):
//     Reshape(ti) -> [1, d0..dk]       named: _seq_reshape_<base>_<i>
//   Concat([reshape_0..reshape_N-1], axis=0) -> [N, d0..dk]
//                                        named: _seq_stacked_<base>
//   [Cast(int64 -> int32) if needed]
//   [For static negative pos: normalize at build time]
//   [For dynamic pos: Add + Less + Select subgraph]
//   Gather(stacked, pos, axis=0) -> [1, d0..dk]
//                                        named: _seq_gather_out_<base>
//   Reshape([1, d0..dk]) -> [d0..dk]   = the registered output of SequenceAt
//
// Note: QNN Gather requires rank-1 indices ({1} shape). ONNX Gather allows
// rank-0 (scalar) indices, but QNN rejects them with error 3110. The extra
// leading-1 dimension is removed by the trailing Reshape.
// ---------------------------------------------------------------------------

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         const OrtNodeUnit& seq_construct_unit,
                                         const OrtNodeUnit& seq_at_unit,
                                         bool validate) {
  const auto& sc_inputs = seq_construct_unit.Inputs();
  const size_t N = sc_inputs.size();
  if (N < 1) {
    return MAKE_EP_FAIL("SequenceAtFusion: SequenceConstruct has no inputs.");
  }

  // Retrieve element shape from the first input.
  std::vector<uint32_t> elem_shape;
  if (!qmw.GetOnnxShape(sc_inputs[0].shape, elem_shape)) {
    return MAKE_EP_FAIL("SequenceAtFusion: cannot get element shape.");
  }

  // SequenceAt inputs: [sequence (skipped), position].
  const auto& sa_inputs = seq_at_unit.Inputs();
  if (sa_inputs.size() < 2) {
    return MAKE_EP_FAIL("SequenceAtFusion: SequenceAt has fewer than 2 inputs.");
  }
  const OrtNodeUnitIODef& pos_input = sa_inputs[1];
  const auto& sa_outputs = seq_at_unit.Outputs();
  if (sa_outputs.empty()) {
    return MAKE_EP_FAIL("SequenceAtFusion: SequenceAt has no outputs.");
  }
  const OrtNodeUnitIODef& final_output = sa_outputs[0];

  // Retrieve element data type info from first SC input.
  TensorInfo elem_info{};
  auto elem_info_status = qmw.GetTensorInfo(sc_inputs[0], elem_info);
  if (!elem_info_status.IsOK()) return elem_info_status;

  // Unique name base derived from SequenceConstruct.
  const std::string base = utils::UniqueNameGenerator().New(seq_construct_unit);

  // ------------------------------------------------------------------
  // Validate mode: probe QNN for every dtype-sensitive op this fusion emits.
  //
  // Reshape and Concat are in the float32/float16 allowlist and are
  // universally supported on HTP/GPU for those dtypes; probing them adds
  // false-rejection risk with no practical benefit.
  //
  // Gather IS dtype-sensitive — probe it with actual element dtype and
  // shapes from the model.  For dynamic positions, also probe Cast (int64
  // only), ElementWiseAdd, ElementWiseLess, ElementWiseSelect so a backend
  // that rejects those int32/bool ops is caught here rather than at graph
  // finalization.
  // ------------------------------------------------------------------
  if (validate) {
    // Probe 1: Gather — always emitted, element-dtype-sensitive.
    {
      std::vector<uint32_t> stacked_shape;
      stacked_shape.reserve(1 + elem_shape.size());
      stacked_shape.push_back(static_cast<uint32_t>(N));
      stacked_shape.insert(stacked_shape.end(), elem_shape.begin(), elem_shape.end());

      std::vector<uint32_t> gather_out_shape;
      gather_out_shape.push_back(1u);
      gather_out_shape.insert(gather_out_shape.end(), elem_shape.begin(), elem_shape.end());

      QnnTensorWrapper data_tw(base + "_validate_stacked",
                               QNN_TENSOR_TYPE_NATIVE,
                               elem_info.qnn_data_type,
                               elem_info.quant_param.Copy(),
                               std::vector<uint32_t>(stacked_shape));
      QnnTensorWrapper idx_tw(base + "_validate_index",
                              QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_INT_32,
                              QnnQuantParamsWrapper(),
                              std::vector<uint32_t>{1u});
      const bool is_graph_output = qmw.IsGraphOutput(final_output.name);
      const Qnn_TensorType_t out_type =
          is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
      QnnTensorWrapper out_tw(base + "_validate_out",
                              out_type,
                              elem_info.qnn_data_type,
                              elem_info.quant_param.Copy(),
                              std::vector<uint32_t>(gather_out_shape));

      Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
      axis_scalar.dataType = QNN_DATATYPE_INT_32;
      axis_scalar.int32Value = 0;
      QnnParamWrapper axis_param(seq_construct_unit.Index(), seq_construct_unit.Name(),
                                 QNN_OP_GATHER_PARAM_AXIS, axis_scalar);
      auto s = qmw.ValidateQnnNode(base + "_validate_gather",
                                   QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GATHER,
                                   {data_tw.GetQnnTensor(), idx_tw.GetQnnTensor()},
                                   {out_tw.GetQnnTensor()},
                                   {axis_param.GetQnnParam()});
      if (!s.IsOK()) return s;
    }

    // Probes 2-5 (conditional): dynamic position path only.
    TensorInfo pos_info_v{};
    const bool pos_is_static = qmw.GetTensorInfo(pos_input, pos_info_v).IsOK() &&
                               pos_info_v.is_initializer;
    if (!pos_is_static) {
      // Helper: minimal rank-1 NATIVE tensor.
      auto probe_tw = [&](const std::string& suffix, Qnn_DataType_t dt) -> QnnTensorWrapper {
        return QnnTensorWrapper(base + suffix, QNN_TENSOR_TYPE_NATIVE, dt,
                                QnnQuantParamsWrapper(), std::vector<uint32_t>{1u});
      };

      // Probe 2 (int64 dynamic only): Cast int64 → int32.
      if (pos_info_v.qnn_data_type == QNN_DATATYPE_INT_64) {
        auto in = probe_tw("_vcast_in", QNN_DATATYPE_INT_64);
        auto out = probe_tw("_vcast_out", QNN_DATATYPE_INT_32);
        auto s = qmw.ValidateQnnNode(base + "_vcast", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {in.GetQnnTensor()}, {out.GetQnnTensor()}, {});
        if (!s.IsOK()) return s;
      }

      // Probe 3: ElementWiseAdd (int32 + int32 → int32).
      {
        auto a = probe_tw("_vadd_a", QNN_DATATYPE_INT_32);
        auto b = probe_tw("_vadd_b", QNN_DATATYPE_INT_32);
        auto out = probe_tw("_vadd_out", QNN_DATATYPE_INT_32);
        auto s = qmw.ValidateQnnNode(base + "_vadd", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                     QNN_OP_ELEMENT_WISE_ADD,
                                     {a.GetQnnTensor(), b.GetQnnTensor()},
                                     {out.GetQnnTensor()}, {});
        if (!s.IsOK()) return s;
      }

      // Probe 4: ElementWiseLess (int32 < int32 → bool).
      {
        auto a = probe_tw("_vless_a", QNN_DATATYPE_INT_32);
        auto b = probe_tw("_vless_b", QNN_DATATYPE_INT_32);
        auto out = probe_tw("_vless_out", QNN_DATATYPE_BOOL_8);
        auto s = qmw.ValidateQnnNode(base + "_vless", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                     QNN_OP_ELEMENT_WISE_LESS,
                                     {a.GetQnnTensor(), b.GetQnnTensor()},
                                     {out.GetQnnTensor()}, {});
        if (!s.IsOK()) return s;
      }

      // Probe 5: ElementWiseSelect (bool, int32, int32 → int32).
      {
        auto cond = probe_tw("_vsel_cond", QNN_DATATYPE_BOOL_8);
        auto t = probe_tw("_vsel_t", QNN_DATATYPE_INT_32);
        auto f = probe_tw("_vsel_f", QNN_DATATYPE_INT_32);
        auto out = probe_tw("_vsel_out", QNN_DATATYPE_INT_32);
        auto s = qmw.ValidateQnnNode(base + "_vsel", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                     QNN_OP_ELEMENT_WISE_SELECT,
                                     {cond.GetQnnTensor(), t.GetQnnTensor(), f.GetQnnTensor()},
                                     {out.GetQnnTensor()}, {});
        if (!s.IsOK()) return s;
      }
    }

    return Ort::Status();
  }

  // ------------------------------------------------------------------
  // Full AddToModelBuilder path
  // ------------------------------------------------------------------

  // Step 1: Register each element-tensor input as a QNN tensor.
  for (size_t i = 0; i < N; ++i) {
    if (!qmw.IsQnnTensorWrapperExist(sc_inputs[i].name)) {
      QnnTensorWrapper elem_tensor;
      auto s = qmw.MakeTensorWrapper(sc_inputs[i], elem_tensor);
      if (!s.IsOK()) return s;
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(elem_tensor)),
                    "SequenceAtFusion: failed to add element tensor.");
    }
  }

  // Step 2: Handle position dtype and normalize.
  TensorInfo pos_info{};
  auto pos_info_status = qmw.GetTensorInfo(pos_input, pos_info);
  if (!pos_info_status.IsOK()) return pos_info_status;

  std::string pos32_name;

  // Helper: register a rank-1 [1] int32 static constant.
  auto AddInt32Const = [&](const std::string& name, int32_t value) -> Ort::Status {
    if (!qmw.IsQnnTensorWrapperExist(name)) {
      std::vector<uint8_t> data(sizeof(int32_t));
      std::memcpy(data.data(), &value, sizeof(int32_t));
      QnnTensorWrapper tw(name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1u}, std::move(data));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add int32 constant.");
    }
    return Ort::Status();
  };

  const bool is_static_pos = pos_info.is_initializer;

  if (is_static_pos) {
    // Static position: normalize at build time (handle negative indices).
    std::vector<uint8_t> raw_bytes;
    auto unpack_s = qmw.UnpackInitializerData(pos_info.initializer_tensor, raw_bytes);
    if (!unpack_s.IsOK()) return unpack_s;

    // Perform range and truncation checks in int64 before narrowing to int32.
    // TryFusion already validated this path; these checks are a defensive last line.
    int64_t pos_val64 = 0;
    if (pos_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      if (raw_bytes.size() < sizeof(int64_t)) {
        return MAKE_EP_FAIL("SequenceAtFusion: int64 position initializer has fewer bytes than expected.");
      }
      std::memcpy(&pos_val64, raw_bytes.data(), sizeof(int64_t));
    } else {
      if (raw_bytes.size() < sizeof(int32_t)) {
        return MAKE_EP_FAIL("SequenceAtFusion: int32 position initializer has fewer bytes than expected.");
      }
      int32_t val32 = 0;
      std::memcpy(&val32, raw_bytes.data(), sizeof(int32_t));
      pos_val64 = static_cast<int64_t>(val32);
    }

    const int64_t n64 = static_cast<int64_t>(N);
    if (pos_val64 < -n64 || pos_val64 >= n64 ||
        pos_val64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      return MAKE_EP_FAIL("SequenceAtFusion: static position is out of range.");
    }

    // Static position was already validated to [-N, N-1] by TryFusion.
    // Negative values in that range are normalized to their positive equivalent here.
    int32_t pos_val = static_cast<int32_t>(pos_val64);
    if (pos_val < 0) pos_val += static_cast<int32_t>(N);

    pos32_name = base + "_pos_normalized";
    auto add_s = AddInt32Const(pos32_name, pos_val);
    if (!add_s.IsOK()) return add_s;

  } else {
    // Dynamic position.
    if (pos_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      if (!qmw.IsQnnTensorWrapperExist(pos_input.name)) {
        QnnTensorWrapper pos_tensor;
        auto s = qmw.MakeTensorWrapper(pos_input, pos_tensor);
        if (!s.IsOK()) return s;
        RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(pos_tensor)),
                      "SequenceAtFusion: failed to add int64 position tensor.");
      }
      const std::string pos32_cast_name = base + "_pos_int32";
      RETURN_IF_ERROR(qmw.AddCastNode(
          utils::UniqueNameGenerator().New(pos_input.name, QNN_OP_CAST),
          pos_input.name, pos32_cast_name, QNN_TENSOR_TYPE_NATIVE,
          QNN_DATATYPE_INT_32, QnnQuantParamsWrapper(), std::vector<uint32_t>{1u},
          /*do_op_validation=*/false));
      pos32_name = pos32_cast_name;
    } else {
      if (!qmw.IsQnnTensorWrapperExist(pos_input.name)) {
        QnnTensorWrapper pos_tensor;
        auto s = qmw.MakeTensorWrapper(pos_input, pos_tensor);
        if (!s.IsOK()) return s;
        RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(pos_tensor)),
                      "SequenceAtFusion: failed to add int32 position tensor.");
      }
      pos32_name = pos_input.name;
    }

    // Runtime normalization subgraph for valid negative indices in [-N, -1]:
    //   pos_plus_N     = ElementWiseAdd(pos32, N)
    //   is_negative    = ElementWiseLess(pos32, 0)
    //   normalized_pos = ElementWiseSelect(is_negative, pos_plus_N, pos32)
    // Note: ONNX SequenceAt requires the caller to supply a position in
    // [-N, N-1]. Values outside that range (e.g. pos < -N or pos >= N) are
    // undefined behavior per the ONNX spec; the normalization above does not
    // clamp them, and QNN Gather behavior for such indices is unspecified.
    const std::string n_const_name = base + "_seq_len_N";
    const std::string zero_const_name = base + "_pos_zero";
    RETURN_IF_ERROR(AddInt32Const(n_const_name, static_cast<int32_t>(N)));
    RETURN_IF_ERROR(AddInt32Const(zero_const_name, 0));

    const std::string pos_plus_n_name = base + "_pos_plus_N";
    if (!qmw.IsQnnTensorWrapperExist(pos_plus_n_name)) {
      QnnTensorWrapper tw(pos_plus_n_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_INT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1u});
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add pos_plus_N tensor.");
    }
    RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base, QNN_OP_ELEMENT_WISE_ADD),
                                    QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD,
                                    {pos32_name, n_const_name}, {pos_plus_n_name}, {}),
                  "SequenceAtFusion: failed to create ElementWiseAdd node.");

    const std::string is_neg_name = base + "_is_negative";
    if (!qmw.IsQnnTensorWrapperExist(is_neg_name)) {
      QnnTensorWrapper tw(is_neg_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_BOOL_8,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1u});
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add is_negative tensor.");
    }
    RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base, QNN_OP_ELEMENT_WISE_LESS),
                                    QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_LESS,
                                    {pos32_name, zero_const_name}, {is_neg_name}, {}),
                  "SequenceAtFusion: failed to create ElementWiseLess node.");

    const std::string norm_pos_name = base + "_pos_normalized";
    if (!qmw.IsQnnTensorWrapperExist(norm_pos_name)) {
      QnnTensorWrapper tw(norm_pos_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_INT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1u});
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add normalized_pos tensor.");
    }
    RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base, QNN_OP_ELEMENT_WISE_SELECT),
                                    QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_SELECT,
                                    {is_neg_name, pos_plus_n_name, pos32_name}, {norm_pos_name}, {}),
                  "SequenceAtFusion: failed to create ElementWiseSelect node.");
    pos32_name = norm_pos_name;
  }

  // Step 3: Reshape each input to [1, d0..dk].
  std::vector<uint32_t> reshaped_shape;
  reshaped_shape.reserve(1 + elem_shape.size());
  reshaped_shape.push_back(1u);
  reshaped_shape.insert(reshaped_shape.end(), elem_shape.begin(), elem_shape.end());

  std::vector<std::string> reshape_names;
  reshape_names.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    const std::string reshape_out_name = base + "_seq_reshape_" + std::to_string(i);
    if (!qmw.IsQnnTensorWrapperExist(reshape_out_name)) {
      TensorInfo elem_info_i{};
      RETURN_IF_ERROR(qmw.GetTensorInfo(sc_inputs[i], elem_info_i));
      QnnTensorWrapper tw(reshape_out_name, QNN_TENSOR_TYPE_NATIVE,
                          elem_info_i.qnn_data_type, elem_info_i.quant_param.Copy(),
                          std::vector<uint32_t>(reshaped_shape));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "SequenceAtFusion: failed to add reshape output tensor.");
    }
    RETURN_IF_NOT(
        qmw.CreateQnnNode(
            utils::UniqueNameGenerator().New(base + "_reshape_" + std::to_string(i), QNN_OP_RESHAPE),
            QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
            {sc_inputs[i].name}, {reshape_out_name}, {}),
        ("SequenceAtFusion: failed to create Reshape node for input " + std::to_string(i)).c_str());
    reshape_names.push_back(reshape_out_name);
  }

  // Step 4: Concat over all reshaped tensors -> [N, d0..dk].
  std::vector<uint32_t> stacked_shape;
  stacked_shape.reserve(1 + elem_shape.size());
  stacked_shape.push_back(static_cast<uint32_t>(N));
  stacked_shape.insert(stacked_shape.end(), elem_shape.begin(), elem_shape.end());
  const std::string stacked_name = base + "_seq_stacked";

  if (!qmw.IsQnnTensorWrapperExist(stacked_name)) {
    TensorInfo elem_info_0{};
    RETURN_IF_ERROR(qmw.GetTensorInfo(sc_inputs[0], elem_info_0));
    QnnTensorWrapper tw(stacked_name, QNN_TENSOR_TYPE_NATIVE,
                        elem_info_0.qnn_data_type, elem_info_0.quant_param.Copy(),
                        std::vector<uint32_t>(stacked_shape));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                  "SequenceAtFusion: failed to add stacked tensor.");
  }
  std::vector<std::string> concat_param_names;
  {
    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_UINT_32;
    axis_scalar.uint32Value = 0u;
    QnnParamWrapper axis_param(seq_construct_unit.Index(),
                               seq_construct_unit.Name() + "_concat",
                               QNN_OP_CONCAT_PARAM_AXIS, axis_scalar);
    const std::string param_name = axis_param.GetParamTensorName();
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axis_param)),
                  "SequenceAtFusion: failed to add Concat axis param.");
    concat_param_names.push_back(param_name);
  }
  RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base, QNN_OP_CONCAT),
                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                  std::vector<std::string>(reshape_names),
                                  {stacked_name}, std::move(concat_param_names)),
                "SequenceAtFusion: failed to create Concat node.");

  // Step 5: Gather(stacked, pos32, axis=0) -> [1, d0..dk], then Reshape -> [d0..dk].
  //
  // QNN Gather requires rank-1 indices, so the output has an extra leading-1 dim
  // compared to what ONNX Gather with scalar index would produce. The trailing
  // Reshape squeezes it back to elem_shape.
  std::vector<uint32_t> gather_out_shape;
  gather_out_shape.push_back(1u);
  gather_out_shape.insert(gather_out_shape.end(), elem_shape.begin(), elem_shape.end());
  const std::string gather_out_name = base + "_gather_out";

  if (!qmw.IsQnnTensorWrapperExist(gather_out_name)) {
    TensorInfo gather_out_info{};
    auto s = qmw.GetTensorInfo(sc_inputs[0], gather_out_info);
    if (!s.IsOK()) return s;
    QnnTensorWrapper gather_tw(gather_out_name, QNN_TENSOR_TYPE_NATIVE,
                               gather_out_info.qnn_data_type, gather_out_info.quant_param.Copy(),
                               std::vector<uint32_t>(gather_out_shape));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(gather_tw)),
                  "SequenceAtFusion: failed to add gather_out tensor.");
  }
  std::vector<std::string> gather_param_names;
  {
    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_INT_32;
    axis_scalar.int32Value = 0;
    QnnParamWrapper axis_param(seq_at_unit.Index(), seq_at_unit.Name(),
                               QNN_OP_GATHER_PARAM_AXIS, axis_scalar);
    const std::string param_name = axis_param.GetParamTensorName();
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axis_param)),
                  "SequenceAtFusion: failed to add Gather axis param.");
    gather_param_names.push_back(param_name);
  }
  RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(seq_at_unit),
                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GATHER,
                                  {stacked_name, pos32_name},
                                  {gather_out_name}, std::move(gather_param_names)),
                "SequenceAtFusion: failed to create Gather node.");

  // Reshape [1, d0..dk] -> [d0..dk] to produce the registered SequenceAt output.
  // Use elem_info/elem_shape
  // from the SequenceConstruct inputs instead (validated homogeneous above).
  const bool is_graph_output = qmw.IsGraphOutput(final_output.name);
  Qnn_TensorType_t out_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  if (!qmw.IsQnnTensorWrapperExist(final_output.name)) {
    QnnTensorWrapper out_tw(final_output.name, out_type,
                            elem_info.qnn_data_type, elem_info.quant_param.Copy(),
                            std::vector<uint32_t>(elem_shape));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(out_tw)),
                  "SequenceAtFusion: failed to add final output tensor.");
  }
  RETURN_IF_NOT(
      qmw.CreateQnnNode(utils::UniqueNameGenerator().New(base + "_reshape_out", QNN_OP_RESHAPE),
                        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
                        {gather_out_name}, {final_output.name}, {}),
      "SequenceAtFusion: failed to create final Reshape node.");

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
