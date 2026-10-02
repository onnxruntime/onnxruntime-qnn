// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// ArrayFeatureExtractor (ai.onnx.ml domain) → QNN_OP_GATHER decomposition.
//
// ONNX spec: ArrayFeatureExtractor(X, Y) selects elements along the last axis of X
// using indices Y. It is exactly equivalent to Gather(X, Y, axis=rank(X)-1).
//
// Supported X dtypes: float32, int32, int64.
// Y dtype: int64 per ONNX spec. int32 is handled defensively. int64 → Cast to int32.
// int64 X: Cast X to int32 before Gather, cast output back to int64 if graph output.
// Scalar Y (rank 0): not supported. ORT does not infer the output shape for
//   ai.onnx.ml ops with scalar indices, so QnnModel::SetGraphInputOutputInfo would
//   throw at compile time. Falls back to CPU.

#include <cassert>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class ArrayFeatureExtractorOpBuilder : public BaseOpBuilder {
 public:
  ArrayFeatureExtractorOpBuilder() : BaseOpBuilder("ArrayFeatureExtractorOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ArrayFeatureExtractorOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

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

// ---------------------------------------------------------------------------
// IsOpSupported
// ---------------------------------------------------------------------------
Ort::Status ArrayFeatureExtractorOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                          const OrtNodeUnit& node_unit,
                                                          const Ort::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();
  RETURN_IF(inputs.size() != 2,
            "ArrayFeatureExtractor must have exactly two inputs (X, Y).");

  // Validate X dtype: float32, int32, int64 are accepted; double and string are not.
  const ONNXTensorElementDataType x_type = inputs[0].type;
  RETURN_IF(x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
                x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
            "ArrayFeatureExtractor does not support double or string X input.");

  // Validate X rank >= 1.
  std::vector<uint32_t> x_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, x_shape),
                "Cannot get shape for ArrayFeatureExtractor input X.");
  RETURN_IF(x_shape.empty(),
            "ArrayFeatureExtractor requires X rank >= 1.");

  // Reject scalar Y (rank-0 indices). ORT does not infer the output shape for
  // ai.onnx.ml.ArrayFeatureExtractor when Y is scalar, so QnnModel::SetGraphInputOutputInfo
  // would throw at compile time. Falls back to CPU for this edge case.
  const auto& y_input = inputs[1];
  if (y_input.shape.has_value() && y_input.shape->empty()) {
    return MAKE_EP_FAIL("ArrayFeatureExtractor does not support scalar (rank-0) Y indices.");
  }

  // Skip BaseOpBuilder::IsOpSupported — it calls ProcessDataTypes, which calls GetTensorInfo
  // on every output. Shape inference for ai.onnx.ml ops is incomplete in ORT, so GetTensorInfo
  // on the output is not reliable. Go directly to AddToModelBuilder for op validation.
  return BaseOpBuilder::AddToModelBuilder(qnn_model_wrapper, node_unit, logger, /*do_op_validation=*/true);
}

// ---------------------------------------------------------------------------
// ProcessInputs
// ---------------------------------------------------------------------------
Ort::Status ArrayFeatureExtractorOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                          const OrtNodeUnit& node_unit,
                                                          const Ort::Logger& logger,
                                                          std::vector<std::string>& input_names,
                                                          bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  // --- Input 0: X ---
  // If X is int64, cast to int32 (QNN does not natively operate on int64 data).
  const auto& x_input = inputs[0];
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, x_input, logger, input_names));

  const std::string& x_name = input_names[0];
  const auto& x_tensorwrapper = qnn_model_wrapper.GetQnnTensorWrapper(x_name);
  const bool x_is_int64 = (x_tensorwrapper.GetTensorDataType() == QNN_DATATYPE_INT_64);
  if (x_is_int64) {
    const std::string x_cast_name = utils::UniqueNameGenerator().New(x_name, "_int64_to_int32");
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(x_cast_name)) {
      TensorInfo x_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(x_input, x_info));
      RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(
          utils::UniqueNameGenerator().New(x_name, QNN_OP_CAST),
          x_name,
          x_cast_name,
          QNN_TENSOR_TYPE_NATIVE,
          QNN_DATATYPE_INT_32,
          QnnQuantParamsWrapper(),
          std::move(x_info.shape),
          do_op_validation));
    }
    input_names[0] = x_cast_name;
  }

  // --- Input 1: Y (indices) ---
  // QNN supports int32/uint32 indices. Cast int64 → int32 when needed.
  const auto& y_input = inputs[1];
  TensorInfo y_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(y_input, y_info));

  std::string y_name = y_input.name;

  // Obtain the axis dimension of X (axis = rank(X) - 1) for bounds validation of static indices.
  const auto& x_tw_final = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);
  const uint32_t x_rank = x_tw_final.GetTensorRank();
  const int64_t axis_dim = (x_rank > 0) ? static_cast<int64_t>(x_tw_final.GetTensorDims()[x_rank - 1]) : 1;

  // Save Y shape before it may be moved into the tensor wrapper below; needed
  // later for the Cast node output shape if Y is a dynamic int64 input.
  const std::vector<uint32_t> y_shape_saved = y_info.shape;

  std::vector<uint8_t> qnn_indices_bytes;
  if (y_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(y_info.initializer_tensor, onnx_indices_bytes));

    if (y_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      // Static int64 → int32 conversion with negative-index normalization and bounds check.
      const size_t num_elems = onnx_indices_bytes.size() / sizeof(int64_t);
      gsl::span<const int64_t> src{reinterpret_cast<const int64_t*>(onnx_indices_bytes.data()), num_elems};
      qnn_indices_bytes.resize(num_elems * sizeof(int32_t));
      gsl::span<int32_t> dst{reinterpret_cast<int32_t*>(qnn_indices_bytes.data()), num_elems};
      for (size_t i = 0; i < num_elems; ++i) {
        int64_t idx = src[i];
        if (idx < 0) {
          idx += axis_dim;
        }
        RETURN_IF_NOT(idx >= 0 && idx < axis_dim,
                      "ArrayFeatureExtractor static indices contain out-of-bounds values.");
        dst[i] = static_cast<int32_t>(idx);
      }
      y_info.qnn_data_type = QNN_DATATYPE_INT_32;
    } else if (y_info.qnn_data_type == QNN_DATATYPE_INT_32) {
      const size_t num_elems = onnx_indices_bytes.size() / sizeof(int32_t);
      gsl::span<const int32_t> src{reinterpret_cast<const int32_t*>(onnx_indices_bytes.data()), num_elems};
      qnn_indices_bytes.resize(num_elems * sizeof(int32_t));
      gsl::span<int32_t> dst{reinterpret_cast<int32_t*>(qnn_indices_bytes.data()), num_elems};
      for (size_t i = 0; i < num_elems; ++i) {
        int32_t idx = src[i];
        if (idx < 0) {
          idx += static_cast<int32_t>(axis_dim);
        }
        RETURN_IF_NOT(idx >= 0 && static_cast<int64_t>(idx) < axis_dim,
                      "ArrayFeatureExtractor static indices contain out-of-bounds values.");
        dst[i] = idx;
      }
    } else {
      qnn_indices_bytes = std::move(onnx_indices_bytes);
    }
  }

  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(y_name)) {
    const Qnn_TensorType_t tensor_type = y_info.is_initializer
                                             ? QNN_TENSOR_TYPE_STATIC
                                             : qnn_model_wrapper.GetTensorType(y_name);
    QnnTensorWrapper y_tensorwrapper(y_name,
                                     tensor_type,
                                     y_info.qnn_data_type,
                                     QnnQuantParamsWrapper(),
                                     std::move(y_info.shape),
                                     std::move(qnn_indices_bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(y_tensorwrapper)),
                  "Failed to add Y tensor for ArrayFeatureExtractor.");
  }

  // Insert Cast node for dynamic int64 indices → int32.
  const auto& y_tw = qnn_model_wrapper.GetQnnTensorWrapper(y_name);
  std::string y_final_name = y_name;
  if (y_tw.GetTensorDataType() == QNN_DATATYPE_INT_64) {
    assert(!y_info.is_initializer);
    y_final_name = y_name + "_int32";
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(
        utils::UniqueNameGenerator().New(y_name, QNN_OP_CAST),
        y_name,
        y_final_name,
        QNN_TENSOR_TYPE_NATIVE,
        QNN_DATATYPE_INT_32,
        QnnQuantParamsWrapper(),
        std::vector<uint32_t>(y_shape_saved),
        do_op_validation));
  }
  input_names.push_back(y_final_name);

  return Ort::Status();
}

// ---------------------------------------------------------------------------
// ProcessAttributesAndOutputs
// ---------------------------------------------------------------------------
Ort::Status ArrayFeatureExtractorOpBuilder::ProcessAttributesAndOutputs(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    std::vector<std::string>&& input_names,
    const Ort::Logger& logger,
    bool do_op_validation) const {
  // axis = rank(X) - 1 (fixed by ONNX spec for ArrayFeatureExtractor).
  const auto& x_tw = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);
  const uint32_t x_rank = x_tw.GetTensorRank();
  RETURN_IF(x_rank == 0, "ArrayFeatureExtractor requires X rank >= 1.");
  const int32_t axis = static_cast<int32_t>(x_rank - 1);

  std::vector<std::string> param_tensor_names;
  RETURN_IF_ERROR(AddQnnScalar<int32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                        axis, QNN_OP_GATHER_PARAM_AXIS, param_tensor_names));

  // Compute output shape: X.shape[:axis] + Y.shape
  // (X.shape[axis+1:] is omitted because axis is the last axis and is always empty.)
  const auto& y_tw = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]);
  const auto& x_dims = x_tw.GetTensorDims();
  const auto& y_dims = y_tw.GetTensorDims();
  const uint32_t axis_u = static_cast<uint32_t>(axis);

  std::vector<uint32_t> output_shape;
  output_shape.reserve(x_rank - 1 + y_dims.size());
  std::copy(x_dims.begin(), x_dims.begin() + axis_u, std::back_inserter(output_shape));
  std::copy(y_dims.begin(), y_dims.end(), std::back_inserter(output_shape));

  const auto& output_def = node_unit.Outputs()[0];
  const auto& output_name = output_def.name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  // Determine output data type.
  QnnQuantParamsWrapper quant_param;
  RETURN_IF_ERROR(quant_param.Init(qnn_model_wrapper, output_def));
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(quant_param.IsQuantized(), output_def.type, qnn_data_type));

  // If X was int64, it was cast to int32 in ProcessInputs. The Gather output is therefore
  // int32; cast it back to int64 at graph output if needed.
  // output_def.type equals X's type for this op, so checking it tells us if X was int64.
  const bool x_was_int64 = (output_def.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  const bool needs_int64_cast = x_was_int64 && is_graph_output;

  // If a Cast back to int64 is needed, use an intermediate tensor name; otherwise write
  // the Gather output directly to the final output.
  const std::string gather_out_name =
      needs_int64_cast ? utils::UniqueNameGenerator().New(output_name, "_gather_out") : output_name;

  const Qnn_DataType_t gather_qnn_dtype = x_was_int64 ? QNN_DATATYPE_INT_32 : qnn_data_type;
  const Qnn_TensorType_t gather_tensor_type =
      (!needs_int64_cast && is_graph_output) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper gather_out_wrapper(gather_out_name, gather_tensor_type, gather_qnn_dtype,
                                      x_was_int64 ? QnnQuantParamsWrapper() : quant_param.Copy(),
                                      std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gather_out_wrapper)),
                "Failed to add Gather output tensor for ArrayFeatureExtractor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_GATHER,
                                                std::move(input_names),
                                                {gather_out_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to create Gather node for ArrayFeatureExtractor.");

  // Cast int32 Gather output back to int64 if X was originally int64.
  if (needs_int64_cast) {
    // needs_int64_cast implies is_graph_output, so the output tensor type is APP_READ.
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(
        utils::UniqueNameGenerator().New(output_name, "_int32_to_int64"),
        gather_out_name,
        output_name,
        QNN_TENSOR_TYPE_APP_READ,
        qnn_data_type,
        quant_param.Copy(),
        std::vector<uint32_t>(output_shape),
        do_op_validation));
  }

  ORT_UNUSED_PARAMETER(logger);
  return Ort::Status();
}

void CreateArrayFeatureExtractorOpBuilder(const std::string& op_type,
                                          OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ArrayFeatureExtractorOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
