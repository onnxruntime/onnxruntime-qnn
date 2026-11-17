// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// QNN-EP COPY START
// Below implementations are directly copied from core/providers/cpu/tensor/slice_helper.h.
struct PrepareForComputeMetadata {
  explicit PrepareForComputeMetadata(gsl::span<const int64_t> input_dimensions)
      : input_dimensions_(input_dimensions),
        ends_(input_dimensions.begin(), input_dimensions.end()),
        output_dims_(input_dimensions.begin(), input_dimensions.end()) {
    size_t dimension_count = input_dimensions.size();
    starts_.resize(dimension_count, 0);
    steps_.resize(dimension_count, 1);
  }

  gsl::span<const int64_t> input_dimensions_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> steps_;
  std::vector<int64_t> output_dims_;
  std::vector<int64_t> flattened_input_dims_;
  std::vector<int64_t>* p_flattened_input_dims_ = &flattened_input_dims_;
  std::vector<int64_t> flattened_output_dims_;
  std::vector<int64_t>* p_flattened_output_dims_ = &flattened_output_dims_;
};

inline Ort::Status PrepareForComputeHelper(const gsl::span<const int64_t>& raw_starts,
                                           const gsl::span<const int64_t>& raw_ends,
                                           const gsl::span<const int64_t>& raw_axes,
                                           const gsl::span<const int64_t>& raw_steps,
                                           PrepareForComputeMetadata& compute_metadata) {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes;
  if (raw_axes.empty()) {
    // axes are omitted, they are set to[0, ..., ndim - 1]
    axes.reserve(raw_starts.size());
    for (int64_t i = 0, limit = raw_starts.size(); i < limit; ++i) {
      axes.push_back(i);
    }
  } else {
    axes.assign(raw_axes.begin(), raw_axes.end());
  }

  // Iterate through the provided axes and override the start/end/steps ranges
  using AxesSet = InlinedHashSet<int64_t>;
  const auto axes_count = axes.size();
  if (raw_starts.size() < axes_count || raw_ends.size() < axes_count) {
    return MAKE_EP_FAIL("'starts' and 'ends' must have at least as many elements as 'axes'");
  }
  AxesSet unique_axes;
  unique_axes.reserve(axes_count);

  const auto dimension_count = compute_metadata.input_dimensions_.size();
  for (size_t axis_index = 0; axis_index < axes_count; ++axis_index) {
    const auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(dimension_count) : axes[axis_index];
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return MAKE_EP_FAIL("'axes' has an axis outside of the tensor dimension count");
    auto p = unique_axes.insert(axis);
    if (!p.second)
      return MAKE_EP_FAIL("'axes' has duplicates");
    const auto dim_value = compute_metadata.input_dimensions_[gsl::narrow<size_t>(axis)];

    // process step
    auto step = axis_index < raw_steps.size() ? raw_steps[axis_index] : 1;
    if (step == 0)
      return MAKE_EP_FAIL("'step' value cannot be 0");

    if (dim_value == 0) {
      // shape with empty dim. only output_dims_ matters but set everything for completeness
      compute_metadata.steps_[gsl::narrow<size_t>(axis)] = step;
      compute_metadata.starts_[gsl::narrow<size_t>(axis)] = 0;
      compute_metadata.ends_[gsl::narrow<size_t>(axis)] = 0;
      compute_metadata.output_dims_[gsl::narrow<size_t>(axis)] = 0;
      continue;
    }

    // clamp step to avoid overflow if there's a stupidly large value (which will be multiplied in SliceImpl)
    // as long as the clamped value is >= the size of the dimension a single step will push us past the end
    step = std::clamp(step, -dim_value, dim_value);

    compute_metadata.steps_[gsl::narrow<size_t>(axis)] = step;

    // process start
    auto start = raw_starts[axis_index];
    if (start < 0)
      start += dim_value;
    if (step < 0)
      compute_metadata.starts_[gsl::narrow<size_t>(axis)] = std::clamp(start, int64_t{0}, dim_value - 1);
    else
      compute_metadata.starts_[gsl::narrow<size_t>(axis)] = std::clamp(start, int64_t{0}, dim_value);

    // process end
    auto end = raw_ends[axis_index];
    // INT_MAX has a special meaning for end according to spec
    // equivalent to 'None' in numpy
    // it represent slicing to the end of the dimension
    if (end == std::numeric_limits<int32_t>::max() ||
        end == std::numeric_limits<int64_t>::max()) {
      end = step < 0 ? -1 : dim_value;
    } else {
      if (end < 0)
        end += dim_value;
      if (step < 0)
        end = std::clamp(end, int64_t{-1}, dim_value);
      else
        end = std::clamp(end, int64_t{0}, dim_value);
    }

    compute_metadata.ends_[gsl::narrow<size_t>(axis)] = end;

    // find output dim value for this axis
    const auto temp = static_cast<int64_t>(ceil(1.0 * (compute_metadata.ends_[gsl::narrow<size_t>(axis)] - compute_metadata.starts_[gsl::narrow<size_t>(axis)]) / step));
    if (temp < 0)
      compute_metadata.output_dims_[gsl::narrow<size_t>(axis)] = 0;
    else
      compute_metadata.output_dims_[gsl::narrow<size_t>(axis)] = temp;
  }

  return Ort::Status();
}
// QNN-EP COPY END

// Gets the data from initializer inputs (e.g., starts, ends, axes, or steps) as a std::vector<int64_t>.
Ort::Status GetInitializerInputData(const OrtNodeUnitIODef& input, const QnnModelWrapper& qnn_model_wrapper,
                                    std::vector<int64_t>& output) {
  const auto& input_name = input.name;
  const bool is_constant = qnn_model_wrapper.IsConstantInput(input_name);
  RETURN_IF_NOT(is_constant, ("Expected input " + input_name + " to be an initializer.").c_str());
  const OrtValueInfo* initializer_valueinfo = nullptr;
  RETURN_IF_ERROR(qnn_model_wrapper.FindInitializer(input_name, &initializer_valueinfo));

  std::vector<uint8_t> initializer_bytes;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(initializer_valueinfo, initializer_bytes));

  ONNXTensorElementDataType onnx_type = input.type;
  if (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    gsl::span<const int64_t> tensor_elems = ReinterpretAsSpan<int64_t, uint8_t>(initializer_bytes);
    output.insert(output.end(), tensor_elems.begin(), tensor_elems.end());
  } else if (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    gsl::span<const int32_t> tensor_elems = ReinterpretAsSpan<int32_t, uint8_t>(initializer_bytes);
    output.insert(output.end(), tensor_elems.begin(), tensor_elems.end());
  } else {
    return MAKE_EP_FAIL(("Data type " + std::to_string(onnx_type) +
                         " is not supported for Slice initializer input " + input.name)
                            .c_str());
  }

  return Ort::Status();
}

// Extracts starts/ends/axes/steps data from attributes (opset < 10) or initializer inputs
// (opset >= 10), plus the data input's ONNX shape. Shared by ProcessInputs (elision pre-check) and
// ProcessAttributesAndOutputs (building the QNN node), so both always agree on the computed shape.
Ort::Status ExtractSliceParams(const QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit,
                               std::vector<int64_t>& raw_starts,
                               std::vector<int64_t>& raw_ends,
                               std::vector<int64_t>& raw_axes,
                               std::vector<int64_t>& raw_steps,
                               std::vector<int64_t>& input_dimensions) {
  const auto& inputs = node_unit.Inputs();
  const size_t input_count = inputs.size();

  // Opset 9 only has 1 input. The starts, ends, axes values are attributes.
  if (node_unit.SinceVersion() < 10) {
    OrtNodeAttrHelper node_helper(node_unit);
    auto starts = node_helper.Get("starts", std::vector<int64_t>{0});
    raw_starts.assign(starts.begin(), starts.end());
    auto ends = node_helper.Get("ends", std::vector<int64_t>{0});
    raw_ends.assign(ends.begin(), ends.end());
    if (node_helper.HasAttr("axes")) {
      auto axes = node_helper.Get("axes", std::vector<int64_t>{0});
      raw_axes.assign(axes.begin(), axes.end());
    }
  } else {
    constexpr size_t starts_index = 1;
    constexpr size_t ends_index = 2;
    constexpr size_t axes_index = 3;
    constexpr size_t steps_index = 4;

    // Starts input (required).
    RETURN_IF_ERROR(GetInitializerInputData(inputs[starts_index], qnn_model_wrapper, raw_starts));

    // Ends input (required).
    RETURN_IF_ERROR(GetInitializerInputData(inputs[ends_index], qnn_model_wrapper, raw_ends));

    // Axes input (optional).
    if (input_count > axes_index && inputs[axes_index].Exists()) {
      RETURN_IF_ERROR(GetInitializerInputData(inputs[axes_index], qnn_model_wrapper, raw_axes));
    }

    // Steps input (optional).
    if (input_count > steps_index && inputs[steps_index].Exists()) {
      RETURN_IF_ERROR(GetInitializerInputData(inputs[steps_index], qnn_model_wrapper, raw_steps));
    }
  }

  std::vector<uint32_t> input0_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input0_shape),
                "Cannot get shape for Slice input 0.");
  input_dimensions.assign(input0_shape.cbegin(), input0_shape.cend());

  return Ort::Status();
}

// Returns true only if `node_unit`'s first output has exactly one consumer, that consumer is
// Concat, and the output is not itself a graph output. This is the one topology where it's safe
// to build no QNN tensor for a zero-dim Slice output: ConcatOpBuilder::ProcessInputs independently
// excludes zero-dim inputs by re-checking their ONNX-declared shape, so it never references a
// tensor this builder chose not to create. Any other consumer (or a graph-output boundary) would
// reference the missing tensor directly, so those cases must not be elided.
bool CanElideZeroDimOutput(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) {
  const auto& output_name = node_unit.Outputs()[0].name;
  if (qnn_model_wrapper.IsGraphOutput(output_name)) {
    return false;
  }

  const Ort::ConstNode node(&node_unit.GetNode());
  std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.size() != 1) {
    return false;
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = outputs[0].GetConsumers();
  if (consumers.size() != 1 || consumers[0].node == nullptr) {
    return false;
  }

  return Ort::ConstNode(consumers[0].node).GetOperatorType() == "Concat";
}

// Whether a zero-dim Slice output can and should be elided (no QNN tensor/node built for it), on
// the QNN HTP backend only -- QNN HTP's StridedSlice rejects any config whose computed output has
// a zero-sized dimension (backendValidateOpConfig fails with QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE).
// This shows up in partial-rotary (RoPE) exports where the "pass-through" (non-rotary) slice is
// empty because rotary_dim == head_dim. Elision is only safe when CanElideZeroDimOutput holds;
// otherwise the node must decline support so ORT can fall back to CPU EP for it, exactly like
// ReshapeOpBuilder/ShapeOpBuilder do for the same class of "QNN can't represent a zero-sized dim"
// problem.
Ort::Status ComputeSliceZeroDimDecision(const QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const PrepareForComputeMetadata& compute_metadata,
                                        bool& should_elide) {
  should_elide = false;

  if (qnn_model_wrapper.GetQnnBackendType() != QnnBackendType::HTP) {
    return Ort::Status();
  }

  bool has_zero_dim_output = false;
  for (const int64_t dim : compute_metadata.output_dims_) {
    if (dim == 0) {
      has_zero_dim_output = true;
      break;
    }
  }
  if (!has_zero_dim_output) {
    return Ort::Status();
  }

  if (!CanElideZeroDimOutput(qnn_model_wrapper, node_unit)) {
    return MAKE_EP_FAIL(
        "QNN HTP's StridedSlice rejects a zero-sized output dimension, and this Slice's output "
        "cannot be safely elided (its only consumer must be Concat, and it must not be a graph "
        "output).");
  }

  should_elide = true;
  return Ort::Status();
}

// Re-derives the Slice's compute metadata for `node_unit` and calls ComputeSliceZeroDimDecision.
// Used from ProcessInputs to decide whether to skip registering input 0 (and any BOOL->UINT8
// cast) before ProcessAttributesAndOutputs independently reaches the same decision. Recomputing
// this is cheap (O(rank), pure, no QNN API calls) and keeps the decision logic in one place
// without adding a Slice-specific side channel to the shared IOpBuilder/BaseOpBuilder interface.
Ort::Status ShouldElideZeroDimSliceOutput(const QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          bool& should_elide) {
  std::vector<int64_t> raw_starts;
  std::vector<int64_t> raw_ends;
  std::vector<int64_t> raw_axes;
  std::vector<int64_t> raw_steps;
  std::vector<int64_t> input_dimensions;
  RETURN_IF_ERROR(ExtractSliceParams(qnn_model_wrapper, node_unit, raw_starts, raw_ends, raw_axes,
                                     raw_steps, input_dimensions));

  PrepareForComputeMetadata compute_metadata(input_dimensions);
  RETURN_IF_ERROR(PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, raw_steps, compute_metadata));

  return ComputeSliceZeroDimDecision(qnn_model_wrapper, node_unit, compute_metadata, should_elide);
}

}  // namespace

class SliceOpBuilder : public BaseOpBuilder {
 public:
  SliceOpBuilder() : BaseOpBuilder("SliceOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SliceOpBuilder);

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

 private:
  Ort::Status ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const;
};

Ort::Status SliceOpBuilder::ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const {
  size_t input_count = node_unit.Inputs().size();

  // Opset < 10: Only has 1 data input. The starts, ends, and axes values are attributes.
  // Opset >= 10: Everything is an input. The data, starts, and ends inputs are required.
  if (input_count > 1) {
    // Skip the first input. All other input need to be initializer
    for (size_t i = 1; i < input_count; i++) {
      const auto& next_input = node_unit.Inputs()[i];
      if (next_input.Exists() && !qnn_model_wrapper.IsConstantInput(next_input.name)) {
        return MAKE_EP_FAIL("QNN doesn't support dynamic slice.");
      }
    }
  }

  return Ort::Status();
}

// Note: For ONNX Slice operation the expected number of inputs is between 3 and 5
Ort::Status SliceOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const Ort::Logger& logger,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_ERROR(ExplicitOpCheck(qnn_model_wrapper, node_unit));
  }

  bool should_elide = false;
  RETURN_IF_ERROR(ShouldElideZeroDimSliceOutput(qnn_model_wrapper, node_unit, should_elide));
  if (should_elide) {
    return Ort::Status();
  }

  // Only need to add input 0. The other inputs (if any) contain static data that is passed to QNN APIs
  // as static parameters.
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, node_unit.Inputs()[0], logger, input_names));

  // StridedSlice on HTP BE doesn't support BOOL input. Add Cast node to convert BOOL to UINT8.
  const bool needs_bool_cast = (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::HTP &&
                                node_unit.Inputs()[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
  if (needs_bool_cast) {
    const std::string& input0_name = input_names[0];
    TensorInfo input0_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input0_info));
    const std::string cast_output_name = utils::UniqueNameGenerator().New(input0_name, "_bool_to_u8");
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(cast_output_name)) {
      RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::UniqueNameGenerator().New(input0_name, QNN_OP_CAST),
                                                    input0_name,
                                                    cast_output_name,
                                                    QNN_TENSOR_TYPE_NATIVE,
                                                    QNN_DATATYPE_UFIXED_POINT_8,
                                                    QnnQuantParamsWrapper::PerTensor(1.0f, 0),
                                                    std::vector<uint32_t>(input0_info.shape),
                                                    do_op_validation));
    }
    input_names[0] = cast_output_name;
  }

  return Ort::Status();
}

Ort::Status SliceOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        std::vector<std::string>&& input_names,
                                                        const Ort::Logger& logger,
                                                        bool do_op_validation) const {
  // Extract starts, ends, axes, and steps data from attributes (opset < 10) or initializer inputs (opset >= 10).
  std::vector<int64_t> raw_starts;
  std::vector<int64_t> raw_ends;
  std::vector<int64_t> raw_axes;
  std::vector<int64_t> raw_steps;
  std::vector<int64_t> input_dimensions;
  RETURN_IF_ERROR(ExtractSliceParams(qnn_model_wrapper, node_unit, raw_starts, raw_ends, raw_axes, raw_steps,
                                     input_dimensions));

  PrepareForComputeMetadata compute_metadata(input_dimensions);
  RETURN_IF_ERROR(PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, raw_steps, compute_metadata));

  bool should_elide = false;
  RETURN_IF_ERROR(ComputeSliceZeroDimDecision(qnn_model_wrapper, node_unit, compute_metadata, should_elide));
  if (should_elide) {
    return Ort::Status();
  }

  const size_t input_rank = input_dimensions.size();
  std::vector<uint32_t> ranges_dims{static_cast<uint32_t>(input_rank), 3};
  std::vector<uint32_t> ranges_data;
  ranges_data.reserve(input_rank);

  for (size_t i = 0; i < input_rank; i++) {
    ranges_data.push_back(static_cast<uint32_t>(compute_metadata.starts_[i]));
    ranges_data.push_back(static_cast<uint32_t>(compute_metadata.ends_[i]));
    ranges_data.push_back(static_cast<uint32_t>(compute_metadata.steps_[i]));
  }

  const auto& inputs = node_unit.Inputs();

  QnnParamWrapper ranges_paramwrapper(node_unit.Index(),
                                      node_unit.Name(),
                                      QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                      std::move(ranges_dims),
                                      std::move(ranges_data),
                                      true);
  std::string param_tensor_name(ranges_paramwrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(ranges_paramwrapper));

  const bool needs_bool_cast = (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::HTP &&
                                inputs[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
  if (!needs_bool_cast) {
    RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper,
                                   node_unit,
                                   std::move(input_names),
                                   {param_tensor_name},
                                   logger,
                                   do_op_validation, GetQnnOpType(node_unit.OpType())));
    return Ort::Status();
  }

  // StridedSlice on HTP BE doesn't support BOOL output either. Run the op in UINT8 (done via the
  // BOOL -> UINT8 input cast in ProcessInputs) and Cast the output back to BOOL.
  const auto& slice_output = node_unit.Outputs()[0];
  const auto& output_name = slice_output.name;
  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(slice_output.shape, output_shape), "Cannot get shape");
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  const std::string slice_output_name = utils::UniqueNameGenerator().New(output_name, "_u8_to_bool_in");
  QnnQuantParamsWrapper slice_quant_params = QnnQuantParamsWrapper::PerTensor(1.0f, 0);
  QnnTensorWrapper slice_output_wrapper(slice_output_name,
                                        QNN_TENSOR_TYPE_NATIVE,
                                        QNN_DATATYPE_UFIXED_POINT_8,
                                        std::move(slice_quant_params),
                                        std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(slice_output_wrapper)), "Failed to add tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                GetQnnOpType(node_unit.OpType()),
                                                std::move(input_names),
                                                {slice_output_name},
                                                {param_tensor_name},
                                                do_op_validation),
                "Failed to add node.");

  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(slice_output, output_info));
  Qnn_TensorType_t output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::UniqueNameGenerator().New(output_name, "_u8_to_bool"),
                                                slice_output_name,
                                                output_name,
                                                output_tensor_type,
                                                output_info.qnn_data_type,
                                                output_info.quant_param.Copy(),
                                                std::vector<uint32_t>(output_shape),
                                                do_op_validation));
  return Ort::Status();
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SliceOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
