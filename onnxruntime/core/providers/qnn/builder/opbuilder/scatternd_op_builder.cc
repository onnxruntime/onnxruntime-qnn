// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstring>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/normalize_indices_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

constexpr std::array<std::string_view, 3> kSupportedReductions = {"none", "add", "mul"};

// Per-axis range [start, end) on the data tensor produced by a rectangular ScatterND.
struct AxisRange {
  uint32_t axis;
  int64_t start;
  int64_t end;
};

// Detects whether the int64 `indices` tensor encodes a contiguous, row-major
// Cartesian product of per-axis ranges over `data_shape`. Returns the list of
// "partial" axes (axes whose [start, end) is a strict sub-range of the data
// dimension) when so; std::nullopt otherwise.
//
// indices has shape [..., K] where K is the index-tuple width (== number of
// indexed leading axes of data). The (..) prefix carries M = prod(prefix)
// tuples in row-major order. For a rectangular slice-assignment, the prefix
// dims correspond 1:1 to the K indexed axes (one prefix dim per indexed
// axis), and indices[i0, i1, ..., iK-1, k] = lo_k + i_k.
std::optional<std::vector<AxisRange>> DetectRectangleScatter(
    const std::vector<int64_t>& idx_values,
    const std::vector<uint32_t>& indices_shape,
    const std::vector<uint32_t>& data_shape) {
  if (indices_shape.size() < 2) return std::nullopt;
  if (data_shape.empty()) return std::nullopt;

  const size_t K = indices_shape.back();
  if (K == 0 || K > data_shape.size()) return std::nullopt;

  // Need exactly K prefix dims (one per indexed axis) for a Cartesian
  // product layout. indices.shape == [d_0, d_1, ..., d_{K-1}, K].
  const size_t prefix_rank = indices_shape.size() - 1;
  if (prefix_rank != K) return std::nullopt;

  // Per-axis lengths from the indices prefix.
  std::vector<uint32_t> lens(K);
  uint64_t total_tuples = 1;
  for (size_t k = 0; k < K; ++k) {
    lens[k] = indices_shape[k];
    if (lens[k] == 0) return std::nullopt;
    total_tuples *= lens[k];
  }
  if (idx_values.size() != total_tuples * K) return std::nullopt;

  // Determine each axis's [start, end) from the first/last tuples along that
  // axis. Compute the row-major stride of the prefix (in units of K).
  std::vector<uint64_t> stride(K);
  stride[K - 1] = 1;
  for (size_t k = K - 1; k > 0; --k) stride[k - 1] = stride[k] * lens[k];

  std::vector<int64_t> lo(K), hi(K);
  for (size_t k = 0; k < K; ++k) {
    // Tuple at i_k = 0 (others 0): linear position 0 -> column k.
    lo[k] = idx_values[k];
    // Tuple at i_k = lens[k] - 1 (others 0): linear position
    // (lens[k] - 1) * stride[k] tuples; column k.
    const uint64_t last_pos = static_cast<uint64_t>(lens[k] - 1) * stride[k];
    hi[k] = idx_values[last_pos * K + k];
    // Each step along axis k must increment the tuple by exactly 1, and
    // start must be the smallest.
    if (hi[k] - lo[k] != static_cast<int64_t>(lens[k]) - 1) return std::nullopt;
    if (lo[k] < 0) return std::nullopt;
    if (hi[k] >= static_cast<int64_t>(data_shape[k])) return std::nullopt;
  }

  // Verify every tuple matches the Cartesian product (lo + i_k along axis k).
  // O(M * K), but M is small (TSM: 8*8 = 64 tuples).
  for (uint64_t t = 0; t < total_tuples; ++t) {
    for (size_t k = 0; k < K; ++k) {
      const uint64_t i_k = (t / stride[k]) % lens[k];
      const int64_t expected = lo[k] + static_cast<int64_t>(i_k);
      if (idx_values[t * K + k] != expected) return std::nullopt;
    }
  }

  // Build the list of "partial" axes — drop axes that span the full
  // dimension (those become trivial slices and would just emit a no-op
  // Concat with one input).
  std::vector<AxisRange> partials;
  for (size_t k = 0; k < K; ++k) {
    const int64_t end = hi[k] + 1;
    if (lo[k] == 0 && end == static_cast<int64_t>(data_shape[k])) continue;
    partials.push_back({static_cast<uint32_t>(k), lo[k], end});
  }
  // If every axis is a full overwrite, this is the degenerate case — the
  // caller's `updates` is the entire `data` shape. Skip the rewrite and let
  // the standard ScatterND path handle it.
  if (partials.empty()) return std::nullopt;
  return partials;
}

// Emits a QNN_OP_STRIDED_SLICE node that takes the [start, end) range along
// `axis` of `input_name` (whose full shape is `input_shape`) and writes it to
// a freshly-allocated NATIVE tensor. Returns the slice output's name.
Ort::Status EmitStridedSlice(QnnModelWrapper& qnn_model_wrapper,
                             const OrtNodeUnit& node_unit,
                             const std::string& input_name,
                             const std::vector<uint32_t>& input_shape,
                             Qnn_DataType_t data_type,
                             const QnnQuantParamsWrapper& quant_param,
                             uint32_t axis,
                             int64_t start,
                             int64_t end,
                             bool do_op_validation,
                             std::string& slice_output_name) {
  const uint32_t rank = static_cast<uint32_t>(input_shape.size());

  // ranges param: [rank, 3] = (start, end, step) per dimension.
  std::vector<uint32_t> ranges_dims{rank, 3};
  std::vector<uint32_t> ranges_data;
  ranges_data.reserve(rank * 3);
  std::vector<uint32_t> output_shape(input_shape);
  for (uint32_t i = 0; i < rank; ++i) {
    if (i == axis) {
      ranges_data.push_back(static_cast<uint32_t>(start));
      ranges_data.push_back(static_cast<uint32_t>(end));
      ranges_data.push_back(1U);
      output_shape[i] = static_cast<uint32_t>(end - start);
    } else {
      ranges_data.push_back(0U);
      ranges_data.push_back(input_shape[i]);
      ranges_data.push_back(1U);
    }
  }

  const std::string slice_name = utils::UniqueNameGenerator().New(node_unit, "_rect_slice");
  slice_output_name = utils::UniqueNameGenerator().New(node_unit, "_rect_slice_out");

  QnnParamWrapper ranges_paramwrapper(node_unit.Index(),
                                      slice_name,
                                      QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                      std::move(ranges_dims),
                                      std::move(ranges_data),
                                      true);
  std::vector<std::string> param_tensor_names{ranges_paramwrapper.GetParamTensorName()};
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(ranges_paramwrapper)),
                "Failed to add StridedSlice ranges param.");

  // begin_mask / end_mask = 0 — every dim is constrained explicitly.
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(),
                                         slice_name, 0U,
                                         QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK,
                                         param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(),
                                         slice_name, 0U,
                                         QNN_OP_STRIDED_SLICE_PARAM_END_MASK,
                                         param_tensor_names));

  QnnTensorWrapper slice_output(slice_output_name,
                                QNN_TENSOR_TYPE_NATIVE,
                                data_type,
                                quant_param.Copy(),
                                std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(slice_output)),
                "Failed to add StridedSlice output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(slice_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_STRIDED_SLICE,
                                                {input_name},
                                                {slice_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to create StridedSlice node.");
  return Ort::Status();
}

// Emits a QNN_OP_CONCAT along `axis`. If `final_output_name` is non-empty,
// the output is wired as the final ScatterND output (NATIVE/APP_READ per
// graph-output check); otherwise a fresh NATIVE intermediate is allocated.
Ort::Status EmitConcat(QnnModelWrapper& qnn_model_wrapper,
                       const OrtNodeUnit& node_unit,
                       const std::vector<std::string>& part_names,
                       const std::vector<uint32_t>& output_shape,
                       Qnn_DataType_t data_type,
                       const QnnQuantParamsWrapper& quant_param,
                       uint32_t axis,
                       const std::string& final_output_name,
                       Qnn_TensorType_t final_tensor_type,
                       bool do_op_validation,
                       std::string& concat_output_name) {
  const std::string concat_name = utils::UniqueNameGenerator().New(node_unit, "_rect_concat");
  if (!final_output_name.empty()) {
    concat_output_name = final_output_name;
  } else {
    concat_output_name = utils::UniqueNameGenerator().New(node_unit, "_rect_concat_out");
  }

  std::vector<std::string> param_tensor_names;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(),
                                         concat_name, axis,
                                         QNN_OP_CONCAT_PARAM_AXIS,
                                         param_tensor_names));

  const Qnn_TensorType_t out_type = !final_output_name.empty() ? final_tensor_type : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper concat_output(concat_output_name,
                                 out_type,
                                 data_type,
                                 quant_param.Copy(),
                                 std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(concat_output)),
                "Failed to add Concat output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(concat_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_CONCAT,
                                                std::vector<std::string>(part_names),
                                                {concat_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to create Concat node.");
  return Ort::Status();
}

// Recursively emits the Slice+Concat chain that replaces a ScatterND over
// rectangular indices.
//
// At each level, peels off the first partial axis (a, s, e):
//   left  = StridedSlice(data, axis=a, [0, s))      if s > 0
//   right = StridedSlice(data, axis=a, [e, dim))    if e < dim
//   inner = (recurse on data[..., s:e on a], partials_tail) — base case = updates_name
//   out   = Concat(axis=a, [left?, inner, right?])
//
// The OUTERMOST call wires its Concat to `final_output_name`. All inner
// concats produce fresh NATIVE intermediates.
Ort::Status EmitRectangleDecomposition(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const std::string& data_name,
                                       const std::vector<uint32_t>& data_shape,
                                       const std::string& updates_name,
                                       Qnn_DataType_t data_type,
                                       const QnnQuantParamsWrapper& quant_param,
                                       const std::vector<AxisRange>& partials,
                                       size_t partial_idx,
                                       const std::string& final_output_name,
                                       Qnn_TensorType_t final_tensor_type,
                                       bool do_op_validation,
                                       std::string& out_name) {
  if (partial_idx == partials.size()) {
    // Base case: the caller has narrowed `data` down to exactly the slab
    // covered by `updates`; just hand back the updates tensor.
    out_name = updates_name;
    return Ort::Status();
  }

  const AxisRange& cur = partials[partial_idx];
  const int64_t dim = static_cast<int64_t>(data_shape[cur.axis]);

  // 1. Slice the inner slab along this axis ([s, e)) for the recursion.
  std::string inner_data_name = data_name;
  std::vector<uint32_t> inner_data_shape = data_shape;
  inner_data_shape[cur.axis] = static_cast<uint32_t>(cur.end - cur.start);
  if (!(cur.start == 0 && cur.end == dim)) {
    RETURN_IF_ERROR(EmitStridedSlice(qnn_model_wrapper, node_unit,
                                     data_name, data_shape, data_type, quant_param,
                                     cur.axis, cur.start, cur.end, do_op_validation,
                                     inner_data_name));
  }

  // 2. Recurse on the inner slab with the remaining partial axes.
  std::string inner_out_name;
  RETURN_IF_ERROR(EmitRectangleDecomposition(qnn_model_wrapper, node_unit,
                                             inner_data_name, inner_data_shape,
                                             updates_name, data_type, quant_param,
                                             partials, partial_idx + 1,
                                             /*final_output_name=*/std::string{},
                                             final_tensor_type, do_op_validation,
                                             inner_out_name));

  // 3. Slice the left/right slabs (along this axis) of the original `data`.
  std::vector<std::string> concat_parts;
  if (cur.start > 0) {
    std::string left_name;
    RETURN_IF_ERROR(EmitStridedSlice(qnn_model_wrapper, node_unit,
                                     data_name, data_shape, data_type, quant_param,
                                     cur.axis, 0, cur.start, do_op_validation,
                                     left_name));
    concat_parts.push_back(std::move(left_name));
  }
  concat_parts.push_back(inner_out_name);
  if (cur.end < dim) {
    std::string right_name;
    RETURN_IF_ERROR(EmitStridedSlice(qnn_model_wrapper, node_unit,
                                     data_name, data_shape, data_type, quant_param,
                                     cur.axis, cur.end, dim, do_op_validation,
                                     right_name));
    concat_parts.push_back(std::move(right_name));
  }

  // 4. Concat along this axis. The OUTERMOST level (partial_idx == 0)
  //    wires its concat to the final ScatterND output.
  const bool is_outermost = (partial_idx == 0);
  RETURN_IF_ERROR(EmitConcat(qnn_model_wrapper, node_unit,
                             concat_parts, data_shape, data_type, quant_param,
                             cur.axis,
                             is_outermost ? final_output_name : std::string{},
                             final_tensor_type, do_op_validation, out_name));
  return Ort::Status();
}

// Bounds each tuple column by the matching `data_shape[c]`.
Ort::Status ProcessScatterNDIndices(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnitIODef& indices_input,
                                    const std::vector<uint32_t>& data_shape,
                                    const Ort::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) {
  std::string indices_tensor_name = indices_input.name;

  TensorInfo indices_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_input, indices_info));

  // ONNX ScatterND rank>=1 is not enforced by shape inference; rely on a well-formed graph.
  const uint32_t index_tuple_size = indices_info.shape.back();

  const auto axis_dim_for_element = [index_tuple_size, &data_shape](size_t element_index) -> int64_t {
    const size_t col = element_index % static_cast<size_t>(index_tuple_size);
    return static_cast<int64_t>(data_shape[col]);
  };

  std::vector<uint8_t> qnn_indices_bytes;
  bool has_negative_indices = false;

  if (indices_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(indices_info.initializer_tensor,
                                                            onnx_indices_bytes));

    // ONNX ScatterND `indices` is hard-typed tensor(int64) (unlike ScatterElements).
    RETURN_IF_NOT(utils::NormalizeIndicesBytes<int64_t>(onnx_indices_bytes, axis_dim_for_element,
                                                        qnn_indices_bytes, has_negative_indices),
                  "QNN does not support out-of-range index values for ScatterND.");
    indices_info.qnn_data_type = QNN_DATATYPE_INT_32;

    // Rename so a sibling op reusing the same ONNX initializer under a different
    // axis bound cannot alias our rewritten copy.
    indices_tensor_name = utils::UniqueNameGenerator().New(indices_tensor_name, "_qnn_idx");
  }

  return utils::AddNormalizedIndicesTensor(qnn_model_wrapper, std::move(indices_info),
                                           indices_tensor_name, std::move(qnn_indices_bytes),
                                           logger, input_names, do_op_validation);
}

}  // namespace

class ScatterNDOpBuilder : public BaseOpBuilder {
 public:
  ScatterNDOpBuilder() : BaseOpBuilder("ScatterNDOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScatterNDOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status ScatterNDOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnit& node_unit,
                                              const Ort::Logger& logger,
                                              std::vector<std::string>& input_names,
                                              bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  RETURN_IF(inputs.size() != 3, "QNN EP: ScatterND operator must have three inputs.");

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  // QNN rejects negative/INT_64 indices; rewrite statics to keep the node on QNN.
  TensorInfo data_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], data_info));
  RETURN_IF_ERROR(ProcessScatterNDIndices(qnn_model_wrapper, inputs[1], data_info.shape,
                                          logger, input_names, do_op_validation));

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, input_names));
  return Ort::Status();
}

Ort::Status ScatterNDOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            std::vector<std::string>&& input_names,
                                                            const Ort::Logger& logger,
                                                            bool do_op_validation) const {
  if (input_names.empty()) {
    return Ort::Status();
  }

  if (do_op_validation) {
    // TODO: Remove once QNN CPU supports ScatterND.
    RETURN_IF(qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::CPU,
              "QNN EP does not support ScatterND op on CPU backend. Falling back to ORT CPU.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const std::string reduction = node_helper.Get("reduction", "none");
  RETURN_IF_NOT(utils::ArrayHasString(kSupportedReductions, reduction),
                ("ScatterND does not support reduction " + reduction).c_str());

  // Conditional decomposition: a contiguous rectangular slice-assignment
  // (PyTorch `out[..., s:e, ...] = src` lowering chain: `aten::copy_` ->
  // `aten::index_put_` -> ONNX ScatterND with Cartesian-product indices)
  // hits the disabled `q::ScatterNd.tcm` HTP kernel for FP16 rank-5+
  // tensors during host graph_prepare. Since slice-assignment is identical
  // to Concat-of-Slices, lower it to that on HTP only — CPU/GPU back-ends
  // have working ScatterND kernels and the extra Slice/Concat ops would
  // be a regression there.
  if (reduction == "none" && IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    const auto& inputs = node_unit.Inputs();
    TensorInfo data_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], data_info));
    TensorInfo indices_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], indices_info));
    TensorInfo updates_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], updates_info));

    if (indices_info.is_initializer && !data_info.shape.empty()) {
      std::vector<uint8_t> indices_bytes;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(indices_info.initializer_tensor,
                                                              indices_bytes));
      const size_t num_idx = indices_bytes.size() / sizeof(int64_t);
      std::vector<int64_t> idx_values(num_idx);
      std::memcpy(idx_values.data(), indices_bytes.data(), indices_bytes.size());

      auto partials_opt = DetectRectangleScatter(idx_values, indices_info.shape, data_info.shape);
      if (partials_opt.has_value()) {
        // input_names[0] = data, [1] = indices (already QNN-side), [2] = updates.
        // The decomposition ignores the rewritten indices entirely.
        const std::string& data_name = input_names[0];
        const std::string& updates_name = input_names[2];
        const std::string& final_output_name = node_unit.Outputs()[0].name;
        const Qnn_TensorType_t final_tensor_type =
            qnn_model_wrapper.IsGraphOutput(final_output_name) ? QNN_TENSOR_TYPE_APP_READ
                                                               : QNN_TENSOR_TYPE_NATIVE;

        std::string final_name;
        RETURN_IF_ERROR(EmitRectangleDecomposition(qnn_model_wrapper, node_unit,
                                                   data_name, data_info.shape,
                                                   updates_name, data_info.qnn_data_type,
                                                   data_info.quant_param,
                                                   *partials_opt, /*partial_idx=*/0,
                                                   final_output_name, final_tensor_type,
                                                   do_op_validation, final_name));
        return Ort::Status();
      }
    }
  }

  Qnn_Scalar_t reduction_scalar = QNN_SCALAR_INIT;
  reduction_scalar.dataType = QNN_DATATYPE_UINT_32;
  if (reduction == "none") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_NONE;
  } else if (reduction == "add") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_ADD;
  } else if (reduction == "mul") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_MUL;
  } else {
    return MAKE_EP_FAIL(("Unexpected ScatterND reduction: " + reduction).c_str());
  }

  QnnParamWrapper reduction_param(node_unit.Index(), node_unit.Name(),
                                  QNN_OP_SCATTER_ND_PARAM_REDUCTION, reduction_scalar);
  std::vector<std::string> param_tensor_names = {reduction_param.GetParamTensorName()};
  qnn_model_wrapper.AddParamWrapper(std::move(reduction_param));

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateScatterNDOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ScatterNDOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
