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

  // No element-type or shape validation: element tensors are not used in the emitted
  // QNN graph (the output is purely a compile-time constant N).

  // Step 5: QNN op validation.
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
// Emits:
//   _seq_len_const  : STATIC int64 {1}  value = N
//   QNN_OP_RESHAPE(_seq_len_const) → length_output  int64 {1}
//
// Emits a chain that routes each element all the way to the final output:
//   elem_i (APP_WRITE) × STATIC_ZEROS_i → zero_i            (NATIVE, per elem)
//   Concat(zero_0..N-1, axis=0)          → zero_stacked      (NATIVE)
//   ReduceSum(zero_stacked, all axes, keepdims=true) → sum_kd (NATIVE, {1,...,1})
//   Reshape(sum_kd, {1})                 → zero_scalar        (NATIVE, {1}, 0.0f)
//   Cast(zero_scalar, float→int64)       → zero_int64         (NATIVE, {1}, 0)
//   EWAdd(zero_int64, N_const)           → length             (APP_READ, {1}, N)
// HTP requires every tensor — including NATIVE intermediates — to be consumed
// by some op all the way to a graph output; orphaned NATIVE tensors fail at
// graphExecute with QNN_COMMON_ERROR_GENERAL (Code 1100).
//
// Output shape: {} (rank-0 scalar) per ONNX spec. If QNN rejects rank-0 Reshape
// output tensors (error 3110), fall back to {1} — both shapes hold 8 bytes of
// int64, so ORT's byte-size-based buffer copy succeeds either way.
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
  // Validate mode: probe the two dtype-sensitive ops in the emitted chain:
  //   Cast (element dtype → int64) and ElementWiseAdd (int64 + int64).
  // Universally-supported ops (EWMul, Concat, ReduceSum, Reshape on float32/16)
  // are not probed — per the "probe only dtype-sensitive ops" rule.
  // ------------------------------------------------------------------
  if (validate) {
    TensorInfo elem_info{};
    auto elem_s = qmw.GetTensorInfo(seq_construct_unit.Inputs()[0], elem_info);
    if (!elem_s.IsOK()) return elem_s;

    // Probe Cast (element dtype → int64).
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

    // Probe ElementWiseAdd (int64 + int64 → int64).
    {
      QnnTensorWrapper a(base + "_vadd_a", QNN_TENSOR_TYPE_NATIVE,
                         QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
                         std::vector<uint32_t>{1u});
      QnnTensorWrapper b(base + "_vadd_b", QNN_TENSOR_TYPE_NATIVE,
                         QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
                         std::vector<uint32_t>{1u});
      QnnTensorWrapper out(base + "_vadd_out", QNN_TENSOR_TYPE_NATIVE,
                           QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
                           std::vector<uint32_t>{1u});
      return qmw.ValidateQnnNode(base + "_vadd",
                                 QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD,
                                 {a.GetQnnTensor(), b.GetQnnTensor()}, {out.GetQnnTensor()}, {});
    }
  }

  // ------------------------------------------------------------------
  // Full AddToModelBuilder path
  //
  // Emits a chain that routes every element tensor all the way to the final
  // APP_READ output so HTP never sees an orphaned tensor at any level:
  //
  //   elem_i (APP_WRITE) × STATIC_ZEROS_i → zero_i  (NATIVE)      ⎫ per element
  //   Concat(zero_0..N-1, axis=0) → zero_stacked  (NATIVE)         ⎭
  //   ReduceSum(zero_stacked, all axes, keepdims=true) → sum_kd     (NATIVE, {1,...,1})
  //   Reshape(sum_kd, {1}) → zero_scalar  (NATIVE, {1}, = 0.0f)
  //   Cast(zero_scalar, float→int64) → zero_int  (NATIVE, {1}, = 0)
  //   EWAdd(zero_int, N_const) → length  (APP_READ, {1}, = N)
  // ------------------------------------------------------------------
  const auto& sc_inputs = seq_construct_unit.Inputs();

  // Obtain element shape and dtype from the first input (all elements are
  // assumed to share the same shape, as is standard for SequenceConstruct).
  std::vector<uint32_t> elem_shape;
  if (!qmw.GetOnnxShape(sc_inputs[0].shape, elem_shape)) {
    return MAKE_EP_FAIL("SequenceLengthFusion: cannot get element shape.");
  }
  TensorInfo elem_info{};
  {
    auto s = qmw.GetTensorInfo(sc_inputs[0], elem_info);
    if (!s.IsOK()) return s;
  }

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

  // Step 2: EWMul(elem_i, STATIC_ZEROS_i) → NATIVE zero_i for each element.
  // Multiplying by zero eliminates element values while keeping them live in
  // the graph.  Each zero_i feeds into the Concat in Step 3.
  {
    // Bytes per element (float32=4, float16=2).
    const size_t bpe = (elem_info.qnn_data_type == QNN_DATATYPE_FLOAT_16) ? 2u : 4u;
    uint32_t num_elems = 1u;
    for (uint32_t d : elem_shape) num_elems *= d;
    const size_t total_bytes = static_cast<size_t>(num_elems) * bpe;

    for (size_t i = 0; i < N; ++i) {
      // STATIC zeros tensor (all zero bytes, same shape/dtype as elem_i).
      const std::string zeros_name = base + "_elem_zeros_" + std::to_string(i);
      if (!qmw.IsQnnTensorWrapperExist(zeros_name)) {
        std::vector<uint8_t> zero_bytes(total_bytes, 0u);
        QnnTensorWrapper zeros_tw(zeros_name, QNN_TENSOR_TYPE_STATIC,
                                  elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(elem_shape), std::move(zero_bytes));
        RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(zeros_tw)),
                      "SequenceLengthFusion: failed to add element zeros tensor.");
      }

      // NATIVE zero output tensor.
      const std::string zero_name = base + "_elem_zero_" + std::to_string(i);
      if (!qmw.IsQnnTensorWrapperExist(zero_name)) {
        QnnTensorWrapper zero_tw(zero_name, QNN_TENSOR_TYPE_NATIVE,
                                 elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                                 std::vector<uint32_t>(elem_shape));
        RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(zero_tw)),
                      "SequenceLengthFusion: failed to add element zero tensor.");
      }

      RETURN_IF_NOT(
          qmw.CreateQnnNode(
              utils::UniqueNameGenerator().New(base + "_elem_mul_" + std::to_string(i),
                                               QNN_OP_ELEMENT_WISE_MULTIPLY),
              QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_MULTIPLY,
              {sc_inputs[i].name, zeros_name}, {zero_name}, {}),
          "SequenceLengthFusion: failed to create element zero-multiply.");
    }
  }

  // Step 3: Concat(zero_0..N-1, axis=0) → NATIVE zero_stacked.
  // Shape: {N*elem_shape[0], elem_shape[1], ...}.
  const std::string stacked_name = base + "_zero_stacked";
  {
    std::vector<uint32_t> stacked_shape = elem_shape;
    stacked_shape[0] *= static_cast<uint32_t>(N);

    if (!qmw.IsQnnTensorWrapperExist(stacked_name)) {
      QnnTensorWrapper stacked_tw(stacked_name, QNN_TENSOR_TYPE_NATIVE,
                                  elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(stacked_shape));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(stacked_tw)),
                    "SequenceLengthFusion: failed to add stacked tensor.");
    }

    Qnn_Scalar_t ax = QNN_SCALAR_INIT;
    ax.dataType = QNN_DATATYPE_INT_32;
    ax.int32Value = 0;
    QnnParamWrapper axis_p(seq_construct_unit.Index(), seq_construct_unit.Name(),
                           QNN_OP_CONCAT_PARAM_AXIS, ax);
    const std::string axis_pname = axis_p.GetParamTensorName();
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axis_p)),
                  "SequenceLengthFusion: failed to add Concat axis param.");

    std::vector<std::string> zero_names;
    zero_names.reserve(N);
    for (size_t i = 0; i < N; ++i)
      zero_names.push_back(base + "_elem_zero_" + std::to_string(i));

    RETURN_IF_NOT(
        qmw.CreateQnnNode(
            utils::UniqueNameGenerator().New(base + "_zero_concat", QNN_OP_CONCAT),
            QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
            std::move(zero_names), {stacked_name}, {axis_pname}),
        "SequenceLengthFusion: failed to create zero-concat.");
  }

  // Step 4: ReduceSum(zero_stacked, all axes, keepdims=true) → {1,...,1}.
  // With keepdims=true the output rank matches the input rank but all dims = 1.
  const std::string sum_kd_name = base + "_sum_keepdims";
  {
    const uint32_t rank = static_cast<uint32_t>(elem_shape.size());
    std::vector<uint32_t> sum_kd_shape(rank, 1u);  // {1, 1, ..., 1}

    if (!qmw.IsQnnTensorWrapperExist(sum_kd_name)) {
      QnnTensorWrapper sum_kd_tw(sum_kd_name, QNN_TENSOR_TYPE_NATIVE,
                                 elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                                 std::vector<uint32_t>(sum_kd_shape));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(sum_kd_tw)),
                    "SequenceLengthFusion: failed to add sum_keepdims tensor.");
    }

    // axes param: [0, 1, ..., rank-1] as a uint32 tensor of shape {rank}.
    {
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

      // keepdims=true scalar param.
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
              utils::UniqueNameGenerator().New(base + "_reduce_sum", QNN_OP_REDUCE_SUM),
              QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_REDUCE_SUM,
              {stacked_name}, {sum_kd_name}, {axes_pname, kd_pname}),
          "SequenceLengthFusion: failed to create ReduceSum.");
    }
  }

  // Step 5: Reshape({1,...,1}) → {1}.  Normalises variable-rank keepdims output.
  const std::string zero_scalar_name = base + "_zero_scalar";
  {
    if (!qmw.IsQnnTensorWrapperExist(zero_scalar_name)) {
      QnnTensorWrapper zs_tw(zero_scalar_name, QNN_TENSOR_TYPE_NATIVE,
                             elem_info.qnn_data_type, QnnQuantParamsWrapper(),
                             std::vector<uint32_t>{1u});
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(zs_tw)),
                    "SequenceLengthFusion: failed to add zero_scalar tensor.");
    }
    RETURN_IF_NOT(
        qmw.CreateQnnNode(
            utils::UniqueNameGenerator().New(base + "_reshape_scalar", QNN_OP_RESHAPE),
            QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
            {sum_kd_name}, {zero_scalar_name}, {}),
        "SequenceLengthFusion: failed to create scalar Reshape.");
  }

  // Step 6: Cast(zero_scalar, float→int64) → zero_int64 ({1}, = 0).
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
            {zero_scalar_name}, {zero_int_name}, {}),
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
  // This is the final output; every element tensor has flowed through Steps 2–7.
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
