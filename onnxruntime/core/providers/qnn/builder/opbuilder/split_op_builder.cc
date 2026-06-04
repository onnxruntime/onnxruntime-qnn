// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
class SplitOpBuilder : public BaseOpBuilder {
 public:
  SplitOpBuilder() : BaseOpBuilder("SplitOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SplitOpBuilder);

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

  Ort::Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const Ort::Logger& logger,
                                       const std::vector<std::string>& input_names,
                                       size_t output_index,
                                       Qnn_DataType_t qnn_data_type,
                                       QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;

 private:
  // Lowers an ONNX Split into one QNN StridedSlice node per output. Used on backends where QNN's
  // native Split mis-executes in-graph on HTP (any axis) and corrupts model outputs. See issue
  // #18939. Used for HTP and the serializer (whose DLC is consumed by HTP).
  Ort::Status ProcessSplitAsStridedSlices(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          int32_t axis_value,
                                          const std::vector<uint32_t>& split_index,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const ORT_MUST_USE_RESULT;
};

Ort::Status SplitOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const Ort::Logger& logger,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  // Only support 1 input, Onnx Opset version < 11, or input 2 is initializer
  // doesn't support input 2 (split data) from dynamic input
  const auto& inputs = node_unit.Inputs();
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  return Ort::Status();
}

// Converts an ONNX list of split lengths to a QNN list of split indices.
// Note that the first split index at 0 is implicit (QNN SDK >= 2.19 will raise a validation error if included).
static void ConvertSplitLengthsToSplitIndices(gsl::span<const int64_t> split_lengths,
                                              std::vector<uint32_t>& split_indices) {
  uint32_t split_it = 0;
  for (size_t i = 0; i < split_lengths.size(); ++i) {
    if (i > 0) {  // Do not include the 0th split index.
      split_indices.push_back(split_it);
    }
    split_it += SafeInt<uint32_t>(split_lengths[i]);
  }
}

Ort::Status SplitOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        std::vector<std::string>&& input_names,
                                                        const Ort::Logger& logger,
                                                        bool do_op_validation) const {
  int32_t axis_value = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis_value));

  std::vector<uint32_t> split_index;
  if (node_unit.Inputs().size() > 1) {
    auto& input_name = node_unit.Inputs()[1].name;
    bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
    if (is_constant_input) {
      std::vector<uint8_t> unpacked_tensor;
      const auto* input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_tensor, unpacked_tensor));
      const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
      size_t tensor_byte_size = unpacked_tensor.size();
      size_t size = tensor_byte_size / sizeof(int64_t);
      ConvertSplitLengthsToSplitIndices({tensor_data, size}, split_index);
    } else {
      return MAKE_EP_FAIL("QNN doesn't support dynamic split");
    }
  } else {
    OrtNodeAttrHelper node_helper(node_unit);
    if (node_helper.HasAttr("split")) {
      auto split_lengths = node_helper.Get("split", std::vector<int64_t>{0});
      ConvertSplitLengthsToSplitIndices(split_lengths, split_index);
    }
  }

  if (split_index.size() == 0) {
    if (node_unit.Outputs().size() == 1) {
      // This Split is essentially a no-op.
      RETURN_IF_ERROR(qnn_model_wrapper.AddNoopReshapeNode(node_unit.Name(),
                                                           input_names[0],
                                                           node_unit.Outputs()[0],
                                                           do_op_validation));
      return Ort::Status();
    }

    // Get the length according to axis and split it equally
    std::vector<uint32_t> input_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape), "Cannot get shape");
    RETURN_IF_NOT(static_cast<int32_t>(input_shape.size()) > axis_value, "axis not valid!");
    RETURN_IF_NOT(input_shape.at(axis_value) > 0, "Shape value not valid!");

    // ONNX spec states that if not evenly divisible by `num_outputs`, the last chunk is smaller.
    // Therefore, we have to use ceil() when computing shape[axis] / num_outputs.
    // See: core/providers/cpu/tensor/split.cc::PrepareForCompute()
    const float num_outputs = static_cast<float>(node_unit.Outputs().size());
    const float split_dim_size = static_cast<float>(input_shape[axis_value]);
    const uint32_t step = SafeInt<uint32_t>(std::ceil(split_dim_size / num_outputs));
    uint32_t split_it = 0;

    for (size_t i = 0; i < num_outputs; ++i) {
      if (i > 0) {  // 0th split index is implicit (QNN >= 2.19 raises validation error if included)
        split_index.push_back(split_it);
      }
      split_it += step;
    }
  }

  // QNN's native Split mis-executes when fused with surrounding ops on the HTP backend, silently
  // corrupting model outputs (e.g. Swin accuracy regressions, issue #18939). This affects splits
  // on any axis (including the channel axis) and only manifests at model scale -- an isolated Split
  // runs correctly on-device -- so it cannot be reproduced by a single-op numerical test. Lower the
  // Split to an equivalent set of StridedSlice ops, which HTP executes correctly. The serializer
  // backend is included because the DLC it emits is consumed by HTP.
  //
  // DSP is intentionally NOT included even though IsNpuBackend(DSP) is true: the defect is confirmed
  // only on HTP (V73/V79), which uses a tiled crouton/D32 activation layout that DSP does not share.
  // Revisit if a DSP regression is ever confirmed. Other backends keep native Split.
  const QnnBackendType backend_type = qnn_model_wrapper.GetQnnBackendType();
  const bool lower_to_strided_slice =
      (backend_type == QnnBackendType::HTP || backend_type == QnnBackendType::SERIALIZER);
  if (lower_to_strided_slice) {
    return ProcessSplitAsStridedSlices(qnn_model_wrapper, node_unit, std::move(input_names),
                                       axis_value, split_index, logger, do_op_validation);
  }

  std::vector<std::string> param_tensor_names;
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SPLIT_PARAM_AXIS, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  uint32_t split_size = static_cast<uint32_t>(split_index.size());
  std::vector<uint32_t> split_dim{split_size};
  QnnParamWrapper split_param(node_unit.Index(), node_unit.Name(), QNN_OP_SPLIT_PARAM_SPLIT_INDEX, std::move(split_dim),
                              std::move(split_index));
  param_tensor_names.push_back(split_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(split_param));

  RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                 std::move(input_names),
                                 std::move(param_tensor_names),
                                 logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Ort::Status();
}

Ort::Status SplitOpBuilder::ProcessSplitAsStridedSlices(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        std::vector<std::string>&& input_names,
                                                        int32_t axis_value,
                                                        const std::vector<uint32_t>& split_index,
                                                        const Ort::Logger& logger,
                                                        bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape), "Cannot get shape");
  const size_t input_rank = input_shape.size();
  RETURN_IF_NOT(axis_value >= 0 && static_cast<size_t>(axis_value) < input_rank, "axis not valid!");
  RETURN_IF_NOT(input_shape.at(axis_value) > 0, "Shape value not valid!");
  const uint32_t axis_dim = input_shape[axis_value];

  // `split_index` holds the cumulative split boundaries with the implicit leading 0 omitted, so the
  // full boundary list is {0, split_index..., axis_dim} and there is one output per interval.
  // The equality below is guaranteed by how split_index was built (one boundary per output, minus
  // the implicit 0); it is kept as a cheap guard for the bounds[] index math in the loop.
  const size_t output_count = split_index.size() + 1;
  RETURN_IF_NOT(output_count == GetOutputCountQnnRequired(node_unit),
                "Split boundary count does not match output count.");

  std::vector<uint32_t> bounds;
  bounds.reserve(output_count + 1);
  bounds.push_back(0);
  for (uint32_t boundary : split_index) {
    bounds.push_back(boundary);
  }
  bounds.push_back(axis_dim);

  auto mem_type = QNN_TENSORMEMTYPE_RAW;
  if (qnn_model_wrapper.GetModelSettings().htp_shared_memory) {
    mem_type = QNN_TENSORMEMTYPE_MEMHANDLE;
  }

  for (size_t i = 0; i < output_count; ++i) {
    // Clamp boundaries to the axis size. The equal-split fallback uses ceil(), which can push an
    // interior boundary past the axis size; clamping makes the final slice the (smaller) remainder,
    // matching the ONNX spec / core/providers/cpu/tensor/split.cc.
    const uint32_t start = std::min(bounds[i], axis_dim);
    const uint32_t end = std::min(bounds[i + 1], axis_dim);

    // Build the StridedSlice "ranges" param: one [start, end, step] row per input dim. Only the
    // split axis is sliced; every other dim spans its full extent. Mirrors slice_op_builder.cc.
    std::vector<uint32_t> ranges_dims{static_cast<uint32_t>(input_rank), 3};
    std::vector<uint32_t> ranges_data;
    ranges_data.reserve(input_rank * 3);
    for (size_t dim = 0; dim < input_rank; ++dim) {
      if (static_cast<int32_t>(dim) == axis_value) {
        ranges_data.push_back(start);
        ranges_data.push_back(end);
      } else {
        ranges_data.push_back(0);
        ranges_data.push_back(input_shape[dim]);
      }
      ranges_data.push_back(1);  // step
    }

    const std::string slice_node_name =
        utils::UniqueNameGenerator().New(node_unit, "_slice_" + std::to_string(i));

    QnnParamWrapper ranges_param(node_unit.Index(), slice_node_name, QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                 std::move(ranges_dims), std::move(ranges_data), true);
    std::string ranges_param_name = ranges_param.GetParamTensorName();
    qnn_model_wrapper.AddParamWrapper(std::move(ranges_param));

    // Reproduce the per-output handling of BaseOpBuilder::ProcessOutputs for this single slice
    // output: quant-param override (keeps each output's qparams equal to the Split input, which is
    // required on HTP), supported-data-type selection, graph-output typing, and int64 graph-output
    // casts. Each slice is a self-contained node, so any cast is emitted right after it.
    //
    // This block mirrors BaseOpBuilder::ProcessOutputs (the int64/dtype-cast + output-tensor logic).
    // It cannot share that code directly: ProcessOutputs defers all cast nodes to after a single
    // multi-output node, whereas here each StridedSlice is its own node. Keep the int64 cast logic
    // below in sync with base_op_builder.cc if that handling changes.
    const std::string& output_name = node_unit.Outputs()[i].name;
    TensorInfo output_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[i], output_info));

    if (output_info.quant_param.IsQuantized()) {
      RETURN_IF_ERROR(OverrideOutputQuantParam(qnn_model_wrapper, node_unit, logger, input_names,
                                               i, output_info.qnn_data_type, output_info.quant_param));
    }

    Qnn_DataType_t supported_qnn_data_type = GetSupportedOutputDataType(i, output_info.qnn_data_type);
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

    bool needs_int64_cast = false;
    if (is_graph_output && supported_qnn_data_type == output_info.qnn_data_type &&
        (output_info.qnn_data_type == QNN_DATATYPE_INT_64 || output_info.qnn_data_type == QNN_DATATYPE_UINT_64)) {
      supported_qnn_data_type =
          supported_qnn_data_type == QNN_DATATYPE_INT_64 ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_UINT_32;
      needs_int64_cast = true;
    }

    // The name the StridedSlice writes to. When a cast is inserted, the slice writes to an
    // intermediate tensor and a Cast node maps it to the real graph output.
    std::string slice_output_name = output_name;
    bool insert_cast = false;
    std::string cast_node_name;
    std::string cast_input_name;
    if (needs_int64_cast) {
      cast_node_name = utils::UniqueNameGenerator().New(node_unit, "_cast_int64");
      cast_input_name = utils::UniqueNameGenerator().New(output_name, "_cast_int64");
      QnnTensorWrapper cast_input_tensorwrapper(cast_input_name, QNN_TENSOR_TYPE_NATIVE, supported_qnn_data_type,
                                                output_info.quant_param.Copy(),
                                                std::vector<uint32_t>(output_info.shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_input_tensorwrapper)), "Failed to add tensor.");
      slice_output_name = cast_input_name;
      insert_cast = true;
    } else if (supported_qnn_data_type != output_info.qnn_data_type && is_graph_output && !do_op_validation) {
      cast_node_name = utils::UniqueNameGenerator().New(node_unit, "_cast");
      cast_input_name = utils::UniqueNameGenerator().New(output_name, "_cast");
      QnnTensorWrapper cast_input_tensorwrapper(cast_input_name, QNN_TENSOR_TYPE_NATIVE, supported_qnn_data_type,
                                                output_info.quant_param.Copy(),
                                                std::vector<uint32_t>(output_info.shape), {}, mem_type);
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_input_tensorwrapper)), "Failed to add tensor.");
      slice_output_name = cast_input_name;
      insert_cast = true;
    } else {
      output_info.qnn_data_type = supported_qnn_data_type;
    }

    const Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper output_tensorwrapper(output_name, tensor_type, output_info.qnn_data_type,
                                          output_info.quant_param.Copy(), std::vector<uint32_t>(output_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");

    // All slices read the same Split input, so copy input_names into each node rather than move.
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(slice_node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_STRIDED_SLICE,
                                                  std::vector<std::string>(input_names),
                                                  {slice_output_name},
                                                  {ranges_param_name},
                                                  do_op_validation),
                  "Failed to add StridedSlice node for Split lowering.");

    if (insert_cast) {
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CAST,
                                                    {cast_input_name},
                                                    {output_name},
                                                    {}),
                    "Failed to add Cast node for Split lowering.");
    }
  }

  return Ort::Status();
}

Ort::Status SplitOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                     const OrtNodeUnit& node_unit,
                                                     const Ort::Logger& logger,
                                                     const std::vector<std::string>& input_names,
                                                     size_t output_index,
                                                     Qnn_DataType_t qnn_data_type,
                                                     QnnQuantParamsWrapper& quant_param) const {
  if (!quant_param.IsPerTensor()) {
    return Ort::Status();
  }

  // Force Split outputs to use the same quantization parameters as the input if nearly equal.
  // This helps the HTP backend employ certain optimizations.
  //
  // The quantization tool assigns equal qparams to the input and outputs.
  // However, Sigmoid/Tanh may override their output qparams,
  // which requires us to explicitly handle this in case a Split is consumer of a Sigmoid/Tanh node.
  return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                  0 /*input_index*/, output_index, qnn_data_type, quant_param);
}

void CreateSplitOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SplitOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
