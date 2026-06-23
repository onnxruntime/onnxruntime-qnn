// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <algorithm>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/shape_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

// Shape op builder.
// Maps ONNX Shape -> QNN Shape. ONNX Shape produces an int64 1-D tensor; QNN generates output
// in int32. If the output is not an intermediate tensor in the graph, then a Cast op is
// inserted to convert to int64.
// The ONNX `start`/`end` attributes are mapped to the QNN `start`/`end` scalar params (uint32).
class ShapeOpBuilder : public BaseOpBuilder {
 public:
  ShapeOpBuilder() : BaseOpBuilder("ShapeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ShapeOpBuilder);

 protected:
  Qnn_DataType_t GetSupportedOutputDataType(size_t index,
                                            Qnn_DataType_t qnn_data_type) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Qnn_DataType_t ShapeOpBuilder::GetSupportedOutputDataType(size_t index, Qnn_DataType_t qnn_data_type) const {
  // ONNX Shape always produces an int64 output (no unsigned variant per the ONNX spec), but QNN
  // requires int32. If this node produces a graph output, BaseOpBuilder::ProcessOutputs() adds a
  // Cast node after the Shape op. Otherwise, it just sets the output type to int32.
  ORT_UNUSED_PARAMETER(index);
  if (qnn_data_type == QNN_DATATYPE_INT_64) {
    return QNN_DATATYPE_INT_32;
  }

  return qnn_data_type;
}

Ort::Status ShapeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        std::vector<std::string>&& input_names,
                                                        const Ort::Logger& logger,
                                                        bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape),
                "QNN EP: Cannot get input shape for Shape op.");
  const int64_t rank = static_cast<int64_t>(input_shape.size());
  RETURN_IF(rank < 1, "QNN Shape requires an input of rank >= 1.");

  // Step 1: Resolve `start`/`end` per the ONNX Shape spec (opset >= 15).
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t start_attr = node_helper.Get("start", static_cast<int64_t>(0));
  const int64_t end_attr = node_helper.Get("end", rank);
  const auto [start, end] = ResolveShapeBounds(rank, start_attr, end_attr);
  const int64_t output_length = std::max<int64_t>(0, end - start);

  // Step 2: Postprocess to match the QNN op definition. Per QnnOpDef (MasterOpDef "Shape"), the
  // output is a 1-D tensor of shape [M] with M = end - start, and QNN constrains `end` to
  // [start + 1, N] -- i.e. M >= 1. QNN cannot represent a zero-length output, so the ONNX-valid
  // empty-slice case (output_length == 0) has no QNN equivalent. Clamping `end` up to `start + 1`
  // would silently produce a length-1 output and corrupt results, so instead we reject the op here.
  // Returning an error from IsOpSupported() leaves the node unassigned, and it falls back to CPU EP.
  RETURN_IF(output_length < 1,
            "QNN Shape produces a 1-D output of length (end - start) and requires end >= start + 1; "
            "the ONNX empty-slice case (start >= end) is not supported by QNN and falls back to CPU.");
  // With output_length >= 1 we have end >= start + 1 and end <= rank, which also implies
  // start in [0, rank - 1] (QNN's `start` constraint), so no separate start-range check is needed.

  std::vector<std::string> param_tensor_names;

  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(start), QNN_OP_SHAPE_PARAM_START,
                                         param_tensor_names));

  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(end), QNN_OP_SHAPE_PARAM_END,
                                         param_tensor_names));

  RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                 std::move(input_names),
                                 std::move(param_tensor_names),
                                 logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Ort::Status();
}

void CreateShapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ShapeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
