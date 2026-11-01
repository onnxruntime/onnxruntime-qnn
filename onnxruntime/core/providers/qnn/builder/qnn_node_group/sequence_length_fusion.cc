// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/sequence_length_fusion.h"

#include <cstdint>
#include <cstring>
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

#define ValidateOnQnn(qmw, seq_construct_unit, seq_length_unit) \
  CreateOrValidateOnQnn((qmw), (seq_construct_unit), (seq_length_unit), true)
#define CreateOnQnn(qmw, seq_construct_unit, seq_length_unit) \
  CreateOrValidateOnQnn((qmw), (seq_construct_unit), (seq_length_unit), false)

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         const OrtNodeUnit& seq_construct_unit,
                                         const OrtNodeUnit& seq_length_unit,
                                         bool validate);

// ---------------------------------------------------------------------------
// TryFusion
// ---------------------------------------------------------------------------

std::unique_ptr<IQnnNodeGroup> SequenceLengthFusion::TryFusion(
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

  // Step 2: Find the unique SequenceLength consumer of SequenceConstruct's sequence output.
  //
  // GetConsumers() on a sequence-typed ValueInfo is unreliable — the EP plugin API
  // only enumerates consumers of tensor-typed edges. Use FindUniqueSequenceConsumer
  // (utils.h) which scans node_to_node_unit via the raw Node API instead.
  //
  // We require:
  //   (a) The sequence output is NOT a graph output.
  //   (b) Exactly ONE consumer of the sequence output exists across the entire graph.
  //   (c) That single consumer has op type "SequenceLength" and is a SingleNode unit.
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

  // (b)+(c) Exactly one consumer, must be SequenceLength.
  const OrtNodeUnit* seq_len_unit_ptr =
      FindUniqueSequenceConsumer(seq_output_name, "SequenceLength", node_to_node_unit);
  if (seq_len_unit_ptr == nullptr) {
    return nullptr;
  }

  // Step 3: make sure neither node unit is already claimed by another group.
  if (node_unit_to_qnn_node_group.count(&seq_construct_unit) != 0 ||
      node_unit_to_qnn_node_group.count(seq_len_unit_ptr) != 0) {
    return nullptr;
  }

  // Step 4: N >= 1 (SequenceConstruct must have at least one input).
  const size_t N = seq_construct_unit.Inputs().size();
  if (N == 0) {
    return nullptr;
  }

  // Step 5: Validate element shapes.
  // The per-element ReduceSum accumulation requires all elements to share a fully-static,
  // homogeneous shape with rank ≥ 1.  Require:
  //   - Rank must be ≥ 1 — scalar {} elements have no axes to reduce over.
  //   - All shapes must be identical and fully static (heterogeneous or symbolic → CPU).
  //   - All dimensions must be positive (zero-dim → empty tensor; ReduceSum undefined).
  //   - Element count must fit in uint32_t (overflow guard).
  //
  // The scalar check MUST happen before calling GetOnnxShape because GetOnnxShape
  // silently converts {} to {1}.  If we called it first, first_shape would be {1}
  // (never empty) and the scalar guard below would never fire.
  {
    const auto& raw_shape_0 = seq_construct_unit.Inputs()[0].shape;
    if (raw_shape_0.has_value() && raw_shape_0->empty()) {
      return nullptr;  // scalar element (rank-0): no axes to reduce over
    }
    std::vector<uint32_t> first_shape;
    if (!qnn_model_wrapper.GetOnnxShape(raw_shape_0, first_shape)) {
      return nullptr;  // dynamic/unknown shape (nullopt or negative dims)
    }
    // first_shape cannot be empty here: scalar was rejected above.
    for (uint32_t d : first_shape) {
      if (d == 0) return nullptr;  // zero-dimension element
    }
    for (size_t i = 1; i < N; ++i) {
      std::vector<uint32_t> shape_i;
      if (!qnn_model_wrapper.GetOnnxShape(seq_construct_unit.Inputs()[i].shape, shape_i) ||
          shape_i != first_shape) {
        return nullptr;  // heterogeneous or dynamic
      }
    }
    // Overflow guard: element count must fit in uint32_t.
    uint64_t num_elems = 1;
    for (uint32_t d : first_shape) {
      num_elems *= d;
      if (num_elems > UINT32_MAX) return nullptr;
    }
  }

  // Step 6: Validate element dtype — only FP32 and FP16 are supported.
  //   - The zero-tensor byte-size calculation assumes 4 bytes (FP32) or 2 bytes (FP16);
  //     other dtypes would produce incorrectly-sized static buffers.
  //   - Quantized types (UINT8, INT8, etc.) encode zero via the zero-point, not as raw
  //     zero bytes; filling with zero bytes would encode a nonzero real value.
  {
    TensorInfo elem_info{};
    auto dtype_s = qnn_model_wrapper.GetTensorInfo(seq_construct_unit.Inputs()[0], elem_info);
    if (!dtype_s.IsOK() ||
        (elem_info.qnn_data_type != QNN_DATATYPE_FLOAT_32 &&
         elem_info.qnn_data_type != QNN_DATATYPE_FLOAT_16)) {
      return nullptr;
    }
  }

  // Step 7: QNN op validation — probe the emitted graph on the target backend.
  auto group = std::make_unique<SequenceLengthFusion>(seq_construct_unit, *seq_len_unit_ptr);
  if (!group->IsSupported(qnn_model_wrapper, logger).IsOK()) {
    return nullptr;
  }
  return group;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SequenceLengthFusion::SequenceLengthFusion(const OrtNodeUnit& seq_construct_unit,
                                           const OrtNodeUnit& seq_length_unit)
    : node_units_{&seq_construct_unit, &seq_length_unit} {}

// ---------------------------------------------------------------------------
// IQnnNodeGroup interface
// ---------------------------------------------------------------------------

gsl::span<const OrtNodeUnit* const> SequenceLengthFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>(node_units_.data(), node_units_.size());
}

const OrtNodeUnit* SequenceLengthFusion::GetTargetNodeUnit() const {
  // SequenceLength is the convergence point: both element tensors (via SequenceConstruct)
  // and any in-partition producer of the sequence edge arrive here. Using SequenceLength as
  // the target ensures GetQnnNodeGroupsImpl emits the group after all producers are scheduled.
  return node_units_[1];
}

Ort::Status SequenceLengthFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

Ort::Status SequenceLengthFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

// ---------------------------------------------------------------------------
// CreateOrValidateOnQnn
//
// Routes every APP_WRITE element tensor to the final output via a per-element
// ReduceSum + scalar EWAdd chain, satisfying HTP's requirement that every
// registered tensor flows unbroken to a graph output:
//
//   ReduceSum(elem_i, all axes, keepdims=true) → sum_kd_i  (NATIVE, {1,...,1})
//     [N ops, one per element]
//   EWAdd(sum_kd_{i-1}, sum_kd_i) → chain_i  (NATIVE, {1,...,1})
//     [N-1 ops; absent when N=1]
//   Reshape(accum, {1})                        → float_sum  (NATIVE, {1})
//   EWMul(float_sum, STATIC_ZERO_SCALAR)       → zero_float (NATIVE, {1}, ≈ 0.0f)
//   Cast(zero_float, float→int64)              → zero_int64 (NATIVE, {1}, = 0)
//   EWAdd(zero_int64, STATIC_N_const)          → length     (APP_READ or NATIVE, {1}, = N)
//
// This replaces the previous Concat-based chain which created N full-size STATIC
// zero tensors and a potentially large stacked intermediate.  The new chain uses
// only {1,...,1} intermediates after ReduceSum — all very small regardless of N.
// Node count grows as 2N+3 (vs. N+5 for the old design) but memory is O(1).
//
// Output is always registered as shape {1}. ONNX declares the output as scalar
// {} but QNN HTP rejects rank-0 output tensors (error 3110); the byte count is
// identical (8 bytes for int64) so ORT's buffer copy is correct either way.
// ---------------------------------------------------------------------------

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         const OrtNodeUnit& seq_construct_unit,
                                         const OrtNodeUnit& seq_length_unit,
                                         bool validate) {
  const size_t N = seq_construct_unit.Inputs().size();
  if (N < 1) {
    return MAKE_EP_FAIL("SequenceLengthFusion: SequenceConstruct has no inputs.");
  }

  const auto& sl_outputs = seq_length_unit.Outputs();
  if (sl_outputs.empty()) {
    return MAKE_EP_FAIL("SequenceLengthFusion: SequenceLength has no outputs.");
  }
  const OrtNodeUnitIODef& final_output = sl_outputs[0];

  const std::string base = utils::UniqueNameGenerator().New(seq_construct_unit);

  // ------------------------------------------------------------------
  // Validate mode: probe every distinct op category in the emitted chain
  // with the actual element dtype and shapes.
  // TryFusion Steps 5-6 guarantee: static homogeneous non-scalar shapes,
  // FP32 or FP16 dtype, and no overflow.
  // ------------------------------------------------------------------
  if (validate) {
    TensorInfo elem_info{};
    {
      auto s = qmw.GetTensorInfo(seq_construct_unit.Inputs()[0], elem_info);
      if (!s.IsOK()) return s;
    }

    std::vector<uint32_t> elem_shape;
    if (!qmw.GetOnnxShape(seq_construct_unit.Inputs()[0].shape, elem_shape)) {
      return MAKE_EP_FAIL("SequenceLengthFusion: cannot get element shape in validate.");
    }
    const uint32_t rank = static_cast<uint32_t>(elem_shape.size());
    const std::vector<uint32_t> keepdims_shape(rank, 1u);

    // Probe 1: ReduceSum (elem_dtype, elem_shape → {1,...,1}).
    // Pass axes ([0..rank-1]) and keep_dims (true) so the HTP backend can validate
    // correctly — QNN_OP_REDUCE_SUM requires these params and rejects the op without them.
    {
      QnnTensorWrapper rsum_in(base + "_vrsum_in", QNN_TENSOR_TYPE_NATIVE,
                               elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                               std::vector<uint32_t>(elem_shape));
      QnnTensorWrapper rsum_out(base + "_vrsum_out", QNN_TENSOR_TYPE_NATIVE,
                                elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(keepdims_shape));

      // axes: [0, 1, ..., rank-1]
      std::vector<uint32_t> axes_shape_v{rank};
      std::vector<uint32_t> axes_val;
      axes_val.reserve(rank);
      for (uint32_t j = 0; j < rank; ++j) axes_val.push_back(j);
      QnnParamWrapper axes_p(seq_construct_unit.Index(), seq_construct_unit.Name(),
                             QNN_OP_REDUCE_SUM_PARAM_AXES,
                             std::move(axes_shape_v), std::move(axes_val));

      // keep_dims: true
      Qnn_Scalar_t kd_scalar = QNN_SCALAR_INIT;
      kd_scalar.dataType = QNN_DATATYPE_BOOL_8;
      kd_scalar.bool8Value = 1;
      QnnParamWrapper kd_p(seq_construct_unit.Index(), seq_construct_unit.Name(),
                           QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, kd_scalar);

      auto s = qmw.ValidateQnnNode(base + "_vrsum",
                                   QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_REDUCE_SUM,
                                   {rsum_in.GetQnnTensor()}, {rsum_out.GetQnnTensor()},
                                   {axes_p.GetQnnParam(), kd_p.GetQnnParam()});
      if (!s.IsOK()) return s;
    }

    // Probe 2: ElementWiseAdd (elem_dtype, {1,...,1} + {1,...,1} → {1,...,1})
    // for the scalar accumulation chain.
    {
      QnnTensorWrapper eadd_a(base + "_veadd_a", QNN_TENSOR_TYPE_NATIVE,
                              elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                              std::vector<uint32_t>(keepdims_shape));
      QnnTensorWrapper eadd_b(base + "_veadd_b", QNN_TENSOR_TYPE_NATIVE,
                              elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                              std::vector<uint32_t>(keepdims_shape));
      QnnTensorWrapper eadd_out(base + "_veadd_out", QNN_TENSOR_TYPE_NATIVE,
                                elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(keepdims_shape));
      auto s = qmw.ValidateQnnNode(base + "_veadd",
                                   QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD,
                                   {eadd_a.GetQnnTensor(), eadd_b.GetQnnTensor()},
                                   {eadd_out.GetQnnTensor()}, {});
      if (!s.IsOK()) return s;
    }

    // Probe 3: ElementWiseMultiply (elem_dtype, {1} × {1} → {1})
    // for the scalar zero-mask step.
    {
      QnnTensorWrapper mul_a(base + "_vmul_a", QNN_TENSOR_TYPE_NATIVE,
                             elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                             std::vector<uint32_t>{1u});
      QnnTensorWrapper mul_b(base + "_vmul_b", QNN_TENSOR_TYPE_NATIVE,
                             elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                             std::vector<uint32_t>{1u});
      QnnTensorWrapper mul_out(base + "_vmul_out", QNN_TENSOR_TYPE_NATIVE,
                               elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                               std::vector<uint32_t>{1u});
      auto s = qmw.ValidateQnnNode(base + "_vmul",
                                   QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_MULTIPLY,
                                   {mul_a.GetQnnTensor(), mul_b.GetQnnTensor()},
                                   {mul_out.GetQnnTensor()}, {});
      if (!s.IsOK()) return s;
    }

    // Probe 4: Cast (elem_dtype → int64, {1}).
    {
      QnnTensorWrapper cast_in(base + "_vcast_in", QNN_TENSOR_TYPE_NATIVE,
                               elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                               std::vector<uint32_t>{1u});
      QnnTensorWrapper cast_out(base + "_vcast_out", QNN_TENSOR_TYPE_NATIVE,
                                QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
                                std::vector<uint32_t>{1u});
      auto s = qmw.ValidateQnnNode(base + "_vcast",
                                   QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                   {cast_in.GetQnnTensor()}, {cast_out.GetQnnTensor()}, {});
      if (!s.IsOK()) return s;
    }

    // Probe 5: ElementWiseAdd (int64 + int64 → int64) for the final 0+N step.
    {
      QnnTensorWrapper add_a(base + "_vadd_a", QNN_TENSOR_TYPE_NATIVE,
                             QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
                             std::vector<uint32_t>{1u});
      QnnTensorWrapper add_b(base + "_vadd_b", QNN_TENSOR_TYPE_NATIVE,
                             QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
                             std::vector<uint32_t>{1u});
      QnnTensorWrapper add_out(base + "_vadd_out", QNN_TENSOR_TYPE_NATIVE,
                               QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
                               std::vector<uint32_t>{1u});
      return qmw.ValidateQnnNode(base + "_vadd",
                                 QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD,
                                 {add_a.GetQnnTensor(), add_b.GetQnnTensor()},
                                 {add_out.GetQnnTensor()}, {});
    }
  }

  // ------------------------------------------------------------------
  // Full AddToModelBuilder path
  //
  // Per-element ReduceSum + scalar EWAdd chain (memory-efficient variant):
  //   ReduceSum(elem_i, all axes, keepdims=true) → sum_kd_i  (NATIVE, {1,...,1})
  //   EWAdd chain: accum_i = EWAdd(accum_{i-1}, sum_kd_i)     (N-1 ops, absent if N=1)
  //   Reshape(accum, {1})                        → float_sum   (NATIVE, {1})
  //   EWMul(float_sum, STATIC_ZERO_SCALAR)       → zero_float  (NATIVE, {1})
  //   Cast(zero_float, float→int64)              → zero_int64  (NATIVE, {1})
  //   EWAdd(zero_int64, STATIC_N_const)          → length
  //
  // Total: 2N+3 QNN nodes.  All intermediates from ReduceSum onward are shape
  // {1,...,1} or {1} — O(1) memory per intermediate regardless of element size.
  // ------------------------------------------------------------------
  const auto& sc_inputs = seq_construct_unit.Inputs();

  // Obtain element shape and dtype from the first input.
  // TryFusion Step 5 guarantees all elements share the same static non-scalar shape.
  std::vector<uint32_t> elem_shape;
  if (!qmw.GetOnnxShape(sc_inputs[0].shape, elem_shape)) {
    return MAKE_EP_FAIL("SequenceLengthFusion: cannot get element shape.");
  }
  TensorInfo elem_info{};
  {
    auto s = qmw.GetTensorInfo(sc_inputs[0], elem_info);
    if (!s.IsOK()) return s;
  }

  const uint32_t rank = static_cast<uint32_t>(elem_shape.size());
  const std::vector<uint32_t> keepdims_shape(rank, 1u);  // {1,...,1}, same rank as elem

  // Step 1: Register each element tensor as APP_WRITE (idempotent).
  // Required so GetGraphInputOutputTensorWrapper finds them in model_tensors_map_
  // and HTP's input-tensor setup does not abort with "zero graph inputs".
  for (size_t i = 0; i < N; ++i) {
    if (!qmw.IsQnnTensorWrapperExist(sc_inputs[i].name)) {
      QnnTensorWrapper elem_tw;
      auto s = qmw.MakeTensorWrapper(sc_inputs[i], elem_tw);
      if (!s.IsOK()) return s;
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(elem_tw)),
                    "SequenceLengthFusion: failed to add element tensor.");
    }
  }

  // Step 2: ReduceSum(elem_i, all axes, keepdims=true) → sum_kd_i ({1,...,1}) per element.
  // keepdims=true preserves rank so downstream ops don't need to know the element rank.
  auto emit_reduce_sum = [&](const std::string& input_name,
                             const std::string& output_name) -> Ort::Status {
    if (!qmw.IsQnnTensorWrapperExist(output_name)) {
      QnnTensorWrapper out_tw(output_name, QNN_TENSOR_TYPE_NATIVE,
                              elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                              std::vector<uint32_t>(keepdims_shape));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(out_tw)),
                    "SequenceLengthFusion: failed to add ReduceSum output tensor.");
    }
    std::vector<uint32_t> axes_shape{rank};
    std::vector<uint32_t> axes_val;
    axes_val.reserve(rank);
    for (uint32_t j = 0; j < rank; ++j) axes_val.push_back(j);
    QnnParamWrapper axes_p(seq_construct_unit.Index(), seq_construct_unit.Name(),
                           QNN_OP_REDUCE_SUM_PARAM_AXES,
                           std::move(axes_shape), std::move(axes_val));
    const std::string axes_pname = axes_p.GetParamTensorName();
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axes_p)),
                  "SequenceLengthFusion: failed to add ReduceSum axes param.");

    Qnn_Scalar_t kd = QNN_SCALAR_INIT;
    kd.dataType = QNN_DATATYPE_BOOL_8;
    kd.bool8Value = 1;
    QnnParamWrapper kd_p(seq_construct_unit.Index(), seq_construct_unit.Name(),
                         QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, kd);
    const std::string kd_pname = kd_p.GetParamTensorName();
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(kd_p)),
                  "SequenceLengthFusion: failed to add ReduceSum keepdims param.");

    RETURN_IF_NOT(
        qmw.CreateQnnNode(
            utils::UniqueNameGenerator().New(output_name, QNN_OP_REDUCE_SUM),
            QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_REDUCE_SUM,
            {input_name}, {output_name}, {axes_pname, kd_pname}),
        "SequenceLengthFusion: failed to create ReduceSum.");
    return Ort::Status();
  };

  // Emit N ReduceSum ops, one per element.
  std::string accum_name = base + "_sum_kd_0";
  {
    auto s = emit_reduce_sum(sc_inputs[0].name, accum_name);
    if (!s.IsOK()) return s;
  }
  for (size_t i = 1; i < N; ++i) {
    const std::string sum_name = base + "_sum_kd_" + std::to_string(i);
    {
      auto s = emit_reduce_sum(sc_inputs[i].name, sum_name);
      if (!s.IsOK()) return s;
    }
    // Step 3: Accumulate sum_kd_i into the running total via EWAdd.
    const std::string chain_name = base + "_chain_" + std::to_string(i);
    if (!qmw.IsQnnTensorWrapperExist(chain_name)) {
      QnnTensorWrapper chain_tw(chain_name, QNN_TENSOR_TYPE_NATIVE,
                                elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(keepdims_shape));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(chain_tw)),
                    "SequenceLengthFusion: failed to add chain tensor.");
    }
    RETURN_IF_NOT(
        qmw.CreateQnnNode(
            utils::UniqueNameGenerator().New(chain_name, QNN_OP_ELEMENT_WISE_ADD),
            QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD,
            {accum_name, sum_name}, {chain_name}, {}),
        "SequenceLengthFusion: failed to create EWAdd chain step.");
    accum_name = chain_name;
  }

  // Step 4: Reshape(accum, {1}).  Collapses {1,...,1} (variable rank) → {1}.
  const std::string float_sum_name = base + "_float_sum";
  {
    if (!qmw.IsQnnTensorWrapperExist(float_sum_name)) {
      QnnTensorWrapper fs_tw(float_sum_name, QNN_TENSOR_TYPE_NATIVE,
                             elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                             std::vector<uint32_t>{1u});
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(fs_tw)),
                    "SequenceLengthFusion: failed to add float_sum tensor.");
    }
    RETURN_IF_NOT(
        qmw.CreateQnnNode(
            utils::UniqueNameGenerator().New(base + "_reshape_sum", QNN_OP_RESHAPE),
            QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
            {accum_name}, {float_sum_name}, {}),
        "SequenceLengthFusion: failed to create Reshape.");
  }

  // Step 5: STATIC zero scalar ({1}, same dtype as elements) + EWMul → zero_float.
  // A single shared scalar zero replaces N full-size STATIC zero tensors.
  const std::string zero_scalar_name = base + "_zero_scalar";
  if (!qmw.IsQnnTensorWrapperExist(zero_scalar_name)) {
    const size_t bpe = (elem_info.qnn_data_type == QNN_DATATYPE_FLOAT_16) ? 2u : 4u;
    std::vector<uint8_t> zero_bytes(bpe, 0u);
    QnnTensorWrapper zs_tw(zero_scalar_name, QNN_TENSOR_TYPE_STATIC,
                           elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                           std::vector<uint32_t>{1u}, std::move(zero_bytes));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(zs_tw)),
                  "SequenceLengthFusion: failed to add zero scalar tensor.");
  }
  const std::string zero_float_name = base + "_zero_float";
  if (!qmw.IsQnnTensorWrapperExist(zero_float_name)) {
    QnnTensorWrapper zf_tw(zero_float_name, QNN_TENSOR_TYPE_NATIVE,
                           elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                           std::vector<uint32_t>{1u});
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(zf_tw)),
                  "SequenceLengthFusion: failed to add zero_float tensor.");
  }
  RETURN_IF_NOT(
      qmw.CreateQnnNode(
          utils::UniqueNameGenerator().New(base + "_zero_mul", QNN_OP_ELEMENT_WISE_MULTIPLY),
          QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_MULTIPLY,
          {float_sum_name, zero_scalar_name}, {zero_float_name}, {}),
      "SequenceLengthFusion: failed to create zero-multiply.");

  // Step 6: Cast(zero_float, float→int64) → zero_int64 ({1}, = 0).
  const std::string zero_int_name = base + "_zero_int64";
  {
    if (!qmw.IsQnnTensorWrapperExist(zero_int_name)) {
      QnnTensorWrapper zi_tw(zero_int_name, QNN_TENSOR_TYPE_NATIVE,
                             QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
                             std::vector<uint32_t>{1u});
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(zi_tw)),
                    "SequenceLengthFusion: failed to add zero_int64 tensor.");
    }
    RETURN_IF_NOT(
        qmw.CreateQnnNode(
            utils::UniqueNameGenerator().New(base + "_cast_int64", QNN_OP_CAST),
            QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
            {zero_float_name}, {zero_int_name}, {}),
        "SequenceLengthFusion: failed to create Cast.");
  }

  // Step 7: STATIC N_const (int64, {1}, value = N).
  const std::string const_name = base + "_seq_len_const";
  if (!qmw.IsQnnTensorWrapperExist(const_name)) {
    int64_t n_val = static_cast<int64_t>(N);
    std::vector<uint8_t> data(sizeof(int64_t));
    std::memcpy(data.data(), &n_val, sizeof(int64_t));
    QnnTensorWrapper const_tw(const_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_64,
                              QnnQuantParamsWrapper(), std::vector<uint32_t>{1u}, std::move(data));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(const_tw)),
                  "SequenceLengthFusion: failed to add N constant tensor.");
  }

  // Step 8: EWAdd(zero_int64, N_const) → length (= 0 + N = N).
  // Every element tensor has flowed through Steps 2–7; no orphaned inputs.
  const bool is_graph_output = qmw.IsGraphOutput(final_output.name);
  const Qnn_TensorType_t out_type =
      is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  if (!qmw.IsQnnTensorWrapperExist(final_output.name)) {
    QnnTensorWrapper out_tw(final_output.name, out_type, QNN_DATATYPE_INT_64,
                            QnnQuantParamsWrapper(), std::vector<uint32_t>{1u});
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(out_tw)),
                  "SequenceLengthFusion: failed to add output tensor.");
  }
  RETURN_IF_NOT(
      qmw.CreateQnnNode(utils::UniqueNameGenerator().New(seq_length_unit),
                        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD,
                        {zero_int_name, const_name}, {final_output.name}, {}),
      "SequenceLengthFusion: failed to create EWAdd.");

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
