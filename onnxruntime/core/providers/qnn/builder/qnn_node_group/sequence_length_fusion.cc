// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/sequence_length_fusion.h"

#include <cassert>
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
  // only enumerates consumers of tensor-typed edges. The sequence output is
  // seq(tensor(T)), so GetConsumers() silently returns an empty list.
  // Instead, scan node_to_node_unit using raw Node API input names.
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
  const Ort::ConstValue seq_output_value = sc_outputs[0];
  const std::string seq_output_name = seq_output_value.GetName();

  // (a) Reject if the sequence tensor is itself a graph output.
  if (qnn_model_wrapper.IsGraphOutput(seq_output_name)) {
    return nullptr;
  }

  // (b)+(c) Count ALL consumers of seq_output_name; require exactly one SequenceLength.
  const OrtNodeUnit* seq_len_unit_ptr = nullptr;
  for (const auto& [node_ptr, unit_ptr] : node_to_node_unit) {
    if (unit_ptr == nullptr) continue;
    if (unit_ptr->UnitType() != OrtNodeUnit::Type::SingleNode) continue;
    const Ort::ConstNode candidate_node(node_ptr);
    const auto candidate_inputs = candidate_node.GetInputs();
    bool consumes_sequence = false;
    for (const auto& inp : candidate_inputs) {
      if (inp.GetName() == seq_output_name) {
        consumes_sequence = true;
        break;
      }
    }
    if (!consumes_sequence) continue;

    // This node consumes the sequence output — it must be the one and only SequenceLength.
    if (unit_ptr->OpType() != "SequenceLength" || seq_len_unit_ptr != nullptr) {
      return nullptr;  // non-SequenceLength consumer, or more than one total consumer
    }
    seq_len_unit_ptr = unit_ptr;
  }
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
//   QNN_OP_RESHAPE(_seq_len_const) → length_output  int64 {} (scalar)
//
// The element tensors of SequenceConstruct are NOT registered in the QNN model.
// ORT provides their runtime values to the EP at inference time, but the QNN EP's
// input-copy loop only copies tensors declared as APP_WRITE in the QNN model; since
// we declare none, those element values are silently ignored.
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
  assert(N >= 1);

  const auto& sl_outputs = seq_length_unit.Outputs();
  assert(!sl_outputs.empty());
  const OrtNodeUnitIODef& final_output = sl_outputs[0];

  const std::string base = utils::UniqueNameGenerator().New(seq_construct_unit);

  // ------------------------------------------------------------------
  // Validate mode: probe QNN with a minimal Reshape(int64) node.
  // ------------------------------------------------------------------
  if (validate) {
    QnnTensorWrapper input_tw(
        base + "_validate_in", QNN_TENSOR_TYPE_NATIVE,
        QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
        std::vector<uint32_t>{1u});

    QnnTensorWrapper output_tw(
        base + "_validate_out", QNN_TENSOR_TYPE_NATIVE,
        QNN_DATATYPE_INT_64, QnnQuantParamsWrapper(),
        std::vector<uint32_t>{1u});

    auto validate_status = qmw.ValidateQnnNode(
        base + "_validate_reshape",
        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
        {input_tw.GetQnnTensor()},
        {output_tw.GetQnnTensor()},
        {});
    return std::move(validate_status);
  }

  // ------------------------------------------------------------------
  // Full AddToModelBuilder path
  // ------------------------------------------------------------------

  // Step 1: Register the static int64 constant with value N.
  const std::string const_name = base + "_seq_len_const";
  if (!qmw.IsQnnTensorWrapperExist(const_name)) {
    int64_t n_val = static_cast<int64_t>(N);
    std::vector<uint8_t> data(sizeof(int64_t));
    std::memcpy(data.data(), &n_val, sizeof(int64_t));
    QnnTensorWrapper const_tw(const_name, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_64,
                              QnnQuantParamsWrapper(), std::vector<uint32_t>{1u}, std::move(data));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(const_tw)),
                  "SequenceLengthFusion: failed to add constant tensor.");
  }

  // Step 2: Register the output tensor.
  // ONNX spec: SequenceLength output is a scalar (tensor of empty shape, {}).
  const bool is_graph_output = qmw.IsGraphOutput(final_output.name);
  const Qnn_TensorType_t out_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ
                                                     : QNN_TENSOR_TYPE_NATIVE;
  if (!qmw.IsQnnTensorWrapperExist(final_output.name)) {
    QnnTensorWrapper out_tw(final_output.name, out_type, QNN_DATATYPE_INT_64,
                            QnnQuantParamsWrapper(), std::vector<uint32_t>{});  // {} = scalar
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(out_tw)),
                  "SequenceLengthFusion: failed to add output tensor.");
  }

  // Step 3: Reshape constant {1} → output {}.
  RETURN_IF_NOT(
      qmw.CreateQnnNode(utils::UniqueNameGenerator().New(seq_length_unit),
                        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
                        {const_name}, {final_output.name}, {}),
      "SequenceLengthFusion: failed to create Reshape node.");

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
