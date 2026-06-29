// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class SoftmaxOpBuilder : public BaseOpBuilder {
 public:
  SoftmaxOpBuilder() : BaseOpBuilder("SoftmaxOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxOpBuilder);

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

constexpr int32_t GetDefaultAxisAttribute(int opset_version) {
  // Default axis changed from 1 to -1 in opset 13.
  return opset_version < 13 ? 1 : -1;
}

// Returns true if the Softmax output uses a non-natural quantized encoding that should be
// decoupled from the consumer's encoding via an explicit Convert.
//
// Softmax output is in [0, 1] (strictly non-negative), so its natural quantized encoding is
// asymmetric with offset (zero-point) 0. However, when the output feeds an op that requires a
// symmetric encoding on that input (e.g. an HTP MatMul RHS on older HTP archs), aimet assigns the softmax
// output a symmetric encoding (zero-point != 0). At 8-bit HTP softmax kernel mishandles the symmetric
// output encoding producing garbage and near-zero accuracy
//
// The fix emits Softmax with its natural (zero-point 0) encoding and then a QNN_OP_CONVERT to the
// original (symmetric) encoding the consumer expects, satisfying both the constraints

static bool NeedsBoundedOutputSplit(const std::string& op_type, const TensorInfo& output_info,
                                    QnnBackendType backend_type) {
  // Limiting to Softmax for now, will scale to other ops next
  if (op_type != "Softmax") {
    return false;
  }
  if (!IsNpuBackend(backend_type)) {
    return false;
  }
  if (!output_info.quant_param.IsPerTensor(/*include_bw*/ false)) {
    return false;
  }

  // Only split when the encoding is symmetric (zero-point at the midpoint of the unsigned integer
  // range), which is what aimet assigns when the [0, 1] output must feed a symmetric-input
  // consumer (e.g. an HTP MatMul RHS)
  const bool is_unsigned = output_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_8 ||
                           output_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16;
  if (!is_unsigned) {
    return false;
  }
  const size_t bitwidth = utils::GetElementSizeByType(output_info.qnn_data_type) * 8;
  const int32_t symmetric_offset = -(static_cast<int32_t>(uint64_t{1} << (bitwidth - 1)));
  return output_info.quant_param.Get().scaleOffsetEncoding.offset == symmetric_offset;
}

// Builds the natural (zero-point 0) quantized encoding for an output bounded to the unit interval
// [0, 1] -- e.g. Softmax, Sigmoid, HardSigmoid. Uses a power-of-two scale so the requant on HTP is
// an exact bit-shift: scale = 1 / 2^bitwidth maps the unsigned range [0, 2^bitwidth - 1] to [0, ~1).
static QnnQuantParamsWrapper BuildUnitRangeQuantParams(Qnn_DataType_t qnn_data_type) {
  const size_t bitwidth = utils::GetElementSizeByType(qnn_data_type) * 8;
  const float scale = 1.0f / static_cast<float>(uint64_t{1} << bitwidth);
  return QnnQuantParamsWrapper(scale, /*offset*/ 0);
}

// Creates the QNN Softmax node and wires its output to `output_name`
//
// Normally this is a single Softmax node writing directly to `output_name` with the output's
// (consumer-demanded) quantization params. When NeedsBoundedOutputSplit() is true, it instead
// emits:  Softmax (natural offset-0 encoding) -> QNN_OP_CONVERT -> output_name (original encoding).
// The Convert bridges the natural softmax encoding to the encoding the consumer expects
//
static Ort::Status CreateSoftmaxOutputNodes(QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit,
                                            std::vector<std::string>&& input_names,
                                            std::vector<std::string>&& param_tensor_names,
                                            const std::string& output_name,
                                            Qnn_DataType_t qnn_data_type,
                                            const QnnQuantParamsWrapper& output_quant_param,
                                            std::vector<uint32_t> output_shape,
                                            bool is_graph_output,
                                            const std::string& qnn_op_type,
                                            bool do_op_validation) {
  TensorInfo output_info = {};
  output_info.shape = output_shape;
  output_info.qnn_data_type = qnn_data_type;
  output_info.quant_param = output_quant_param.Copy();

  const Qnn_TensorType_t output_tensor_type =
      is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  if (!NeedsBoundedOutputSplit(node_unit.OpType(), output_info, qnn_model_wrapper.GetQnnBackendType())) {
    // No split: single Softmax node writing directly to output_name. Reached only for the
    // reshape/transpose-path intermediate tensors (the direct path's no-split case goes through
    // ProcessOutputs in the caller)
    QnnTensorWrapper output_tensorwrapper(output_name, output_tensor_type, qnn_data_type,
                                          output_quant_param.Copy(), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                  "Failed to add (Log)Softmax output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  qnn_op_type,
                                                  std::move(input_names),
                                                  {output_name},
                                                  std::move(param_tensor_names),
                                                  do_op_validation),
                  "Failed to add (Log)Softmax node.");
    return Ort::Status();
  }

  // Split path: Softmax (natural encoding) -> Convert -> output_name (original encoding)
  QnnQuantParamsWrapper natural_quant_param = BuildUnitRangeQuantParams(qnn_data_type);
  const std::string natural_name = utils::UniqueNameGenerator().New(output_name, "_natural");

  // 1) Intermediate softmax output with the natural zero-point=0 encoding.
  QnnTensorWrapper natural_tensor_wrapper(natural_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type,
                                          natural_quant_param.Copy(), std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(natural_tensor_wrapper)),
                "Failed to add natural-encoding (Log)Softmax output tensor.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                qnn_op_type,
                                                std::move(input_names),
                                                {natural_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to add (Log)Softmax node (split path).");

  // 2) Output tensor with the original encoding
  QnnTensorWrapper output_tensor_wrapper(output_name, output_tensor_type, qnn_data_type,
                                         output_quant_param.Copy(), std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                "Failed to add (Log)Softmax converted output tensor.");

  // 3) Convert: natural encoding -> original encoding. Single in/out, no params
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_CONVERT),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_CONVERT,
                                                {natural_name},
                                                {output_name},
                                                {},
                                                do_op_validation),
                "Failed to add Convert node for (Log)Softmax bounded-output split.");

  return Ort::Status();
}

std::vector<uint32_t> FlattenShapeFromAxis(const std::vector<uint32_t>& input_shape, int32_t axis) {
  /*
  Return the shape with all dimensions multiplied onward from the specified axis. If axis is 0, the returned shape
  will include an additional batch of size 1 as the first dimension.
  */
  assert(axis >= 0 && static_cast<size_t>(axis) < input_shape.size());
  std::vector<uint32_t> output_shape(input_shape.begin(), input_shape.begin() + axis);

  if (axis == 0) {
    output_shape.push_back(1);  // Additional batch included
  }
  output_shape.push_back(
      std::accumulate(input_shape.begin() + axis, input_shape.end(), 1, std::multiplies<uint32_t>()));

  return output_shape;
}

Ort::Status SoftmaxOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit,
                                            const Ort::Logger& logger,
                                            std::vector<std::string>& input_names,
                                            bool do_op_validation) const {
  const bool is_qpu_backend = IsQpuBackend(qnn_model_wrapper.GetQnnBackendType());
  const auto& inputs = node_unit.Inputs();
  const std::string& input_name = inputs[0].name;
  assert(inputs.size() == 1);

  const int opset_version = node_unit.SinceVersion();
  int32_t axis = GetDefaultAxisAttribute(opset_version);
  RETURN_IF_ERROR(GetCanonicalizedAxisAttribute(qnn_model_wrapper, node_unit, "axis", axis, axis));

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
  size_t input_rank = input_info.shape.size();
  RETURN_IF(input_info.is_initializer,
            "QNN EP does not support (Log)Softmax with an initializer input, "
            "which should be optimized away by the ORT optimizer");

  if (opset_version < 13) {
    /*
    For Onnx Softmax with opset < 13, its behavior is to flatten the input starting from the axis, and perform
    softmax operation along the axis dimension, then reshape back to the original input shape.
    QNN EP is able to support arbitrary axis attribute by wrapping reshapes around the operator.

    Here provides an example:
    Given an input with shape=(3, 4, 5) and axis=1. Its behavior is to reshape the input to (3, 20), perform softmax,
    and then reshape back to (3, 4, 5).

    When axis equals 0, the reshape output shape includes an additional batch of size 1 as the first dimension.
    Here provides an example:
    Given an input with shape=(3, 4, 5) and axis=0. Its behavior is to reshape the input to (1, 60), perform softmax,
    and then reshape back to (3, 4, 5).
    */
    std::string reshape_output_name = utils::UniqueNameGenerator().New(input_name, "_reshape");
    std::vector<uint32_t> reshape_output_shape = FlattenShapeFromAxis(input_info.shape, axis);

    // Input is dynamic, so add reshape node before input.
    const bool is_graph_input = qnn_model_wrapper.IsGraphInput(input_name);

    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input_name,
                                                     reshape_output_name,
                                                     input_info.shape,
                                                     reshape_output_shape,
                                                     input_info.qnn_data_type,
                                                     input_info.quant_param,
                                                     do_op_validation,
                                                     is_graph_input,
                                                     false));
    input_names.push_back(reshape_output_name);
  } else if (is_qpu_backend && axis != static_cast<int32_t>(input_rank) - 1) {
    /*
    For Onnx Softmax with opset >= 13, the QNN HTP and GPU backends only supports the axis attribute that refers to the last
    input dimension.
    QNN EP is able to support arbitrary axis attribute by wrapping transposes around the operator.
    */
    std::string transpose_output_name = utils::UniqueNameGenerator().New(input_name, "_transpose");
    std::vector<uint32_t> transpose_perm;
    RETURN_IF_ERROR(utils::GetPermToLastAxis(static_cast<uint32_t>(axis),
                                             static_cast<uint32_t>(input_rank),
                                             transpose_perm));

    std::vector<uint32_t> transpose_output_shape = input_info.shape;
    transpose_output_shape[input_rank - 1] = input_info.shape[axis];
    transpose_output_shape[axis] = input_info.shape[input_rank - 1];

    // Input is dynamic, so add transpose node before input.
    const bool is_graph_input = qnn_model_wrapper.IsGraphInput(input_name);

    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                       input_name,
                                                       transpose_output_name,
                                                       input_info.shape,
                                                       transpose_perm,
                                                       transpose_output_shape,
                                                       input_info.qnn_data_type,
                                                       input_info.quant_param,
                                                       do_op_validation,
                                                       is_graph_input,
                                                       false));
    input_names.push_back(transpose_output_name);
  } else {
    // Process the input as normal.
    return ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names);
  }

  return Ort::Status();
}

Ort::Status SoftmaxOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                          const OrtNodeUnit& node_unit,
                                                          std::vector<std::string>&& input_names,
                                                          const Ort::Logger& logger,
                                                          bool do_op_validation) const {
  const bool is_qpu_backend = IsQpuBackend(qnn_model_wrapper.GetQnnBackendType());
  const std::string& op_type = node_unit.OpType();
  const auto& outputs = node_unit.Outputs();
  const std::string& orig_output_name = outputs[0].name;
  assert(outputs.size() == 1);

  const int opset_version = node_unit.SinceVersion();
  int32_t axis = GetDefaultAxisAttribute(opset_version);
  RETURN_IF_ERROR(GetCanonicalizedAxisAttribute(qnn_model_wrapper, node_unit, "axis", axis, axis));

  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));
  size_t output_rank = output_info.shape.size();

  if (opset_version < 13) {
    std::string reshape_input_name = utils::UniqueNameGenerator().New(orig_output_name, "_reshape");

    std::vector<uint32_t> reshape_input_shape = FlattenShapeFromAxis(output_info.shape, axis);
    // Override axis due to the inserted batch=1 to the first dimension
    uint32_t qnn_axis = (axis == 0) ? 1u : static_cast<uint32_t>(axis);

    std::vector<std::string> param_tensor_names;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                           qnn_axis, QNN_OP_SOFTMAX_PARAM_AXIS, param_tensor_names));

    // Softmax writes the (flattened) result to reshape_input_name; the subsequent Reshape restores
    // the original shape. The bounded-output split (if needed) is applied here on reshape_input_name,
    // which is never a graph output (the Reshape produces the graph output).
    RETURN_IF_ERROR(CreateSoftmaxOutputNodes(qnn_model_wrapper, node_unit,
                                             std::move(input_names),
                                             std::move(param_tensor_names),
                                             reshape_input_name,
                                             output_info.qnn_data_type,
                                             output_info.quant_param,
                                             reshape_input_shape,
                                             /*is_graph_output*/ false,
                                             GetQnnOpType(node_unit.OpType()),
                                             do_op_validation));

    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(orig_output_name);
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(reshape_input_name,
                                                     orig_output_name,
                                                     reshape_input_shape,
                                                     output_info.shape,
                                                     output_info.qnn_data_type,
                                                     output_info.quant_param,
                                                     do_op_validation,
                                                     false,
                                                     is_graph_output));
  } else if (is_qpu_backend && axis != static_cast<int32_t>(output_rank) - 1) {
    std::string transpose_input_name = utils::UniqueNameGenerator().New(orig_output_name, "_transpose");

    std::vector<uint32_t> transpose_input_shape = output_info.shape;
    transpose_input_shape[output_rank - 1] = output_info.shape[axis];
    transpose_input_shape[axis] = output_info.shape[output_rank - 1];

    // Override axis due to the actual shape after the inserted transpose node
    const uint32_t qnn_axis = static_cast<uint32_t>(output_rank) - 1;

    std::vector<std::string> param_tensor_names;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                           qnn_axis, QNN_OP_SOFTMAX_PARAM_AXIS, param_tensor_names));

    // Softmax writes the result (axis transposed to last) to transpose_input_name; the subsequent
    // Transpose restores the original layout. The bounded-output split (if needed) is applied here
    // on transpose_input_name, which is never a graph output (the Transpose produces the output).
    RETURN_IF_ERROR(CreateSoftmaxOutputNodes(qnn_model_wrapper, node_unit,
                                             std::move(input_names),
                                             std::move(param_tensor_names),
                                             transpose_input_name,
                                             output_info.qnn_data_type,
                                             output_info.quant_param,
                                             transpose_input_shape,
                                             /*is_graph_output*/ false,
                                             GetQnnOpType(node_unit.OpType()),
                                             do_op_validation));

    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(orig_output_name);
    std::vector<uint32_t> transpose_perm;
    RETURN_IF_ERROR(utils::GetPermToLastAxis(static_cast<uint32_t>(axis),
                                             static_cast<uint32_t>(output_rank),
                                             transpose_perm));

    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                       transpose_input_name,
                                                       orig_output_name,
                                                       transpose_input_shape,
                                                       transpose_perm,
                                                       output_info.shape,
                                                       output_info.qnn_data_type,
                                                       output_info.quant_param,
                                                       do_op_validation,
                                                       false,
                                                       is_graph_output));
  } else {
    std::vector<std::string> param_tensor_names;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                           static_cast<uint32_t>(axis),
                                           QNN_OP_SOFTMAX_PARAM_AXIS, param_tensor_names));

    // Direct path: Softmax output is the node's ONNX output. When the bounded-output split is not
    // needed, defer to ProcessOutputs() which handles graph-output wiring and other generic logic.
    // When the split is needed, emit Softmax(natural) -> Convert -> output directly.
    if (!NeedsBoundedOutputSplit(op_type, output_info, qnn_model_wrapper.GetQnnBackendType())) {
      return ProcessOutputs(qnn_model_wrapper, node_unit,
                            std::move(input_names),
                            std::move(param_tensor_names),
                            logger, do_op_validation, GetQnnOpType(op_type));
    }

    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(orig_output_name);
    RETURN_IF_ERROR(CreateSoftmaxOutputNodes(qnn_model_wrapper, node_unit,
                                             std::move(input_names),
                                             std::move(param_tensor_names),
                                             orig_output_name,
                                             output_info.qnn_data_type,
                                             output_info.quant_param,
                                             output_info.shape,
                                             is_graph_output,
                                             GetQnnOpType(op_type),
                                             do_op_validation));
  }

  return Ort::Status();
}

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SoftmaxOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
