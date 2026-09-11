// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Maps ONNX Range to a pre-computed static tensor in the QNN graph.
// QNN has no native Range op, so all three inputs must be graph initializers.
// Values are computed host-side and emitted as a STATIC tensor. A single bridge
// node wires it to the graph output: Transpose(perm={0}) for float32/int32, or
// Cast(INT_32→INT_64) for int64 (QNN Transpose does not accept INT_64 on NPU).
class RangeOpBuilder : public BaseOpBuilder {
 public:
  RangeOpBuilder() : BaseOpBuilder("RangeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RangeOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override {
    ORT_UNUSED_PARAMETER(qnn_model_wrapper);
    ORT_UNUSED_PARAMETER(node_unit);
    ORT_UNUSED_PARAMETER(logger);
    ORT_UNUSED_PARAMETER(input_names);
    ORT_UNUSED_PARAMETER(do_op_validation);
    return Ort::Status();
  }

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ComputeRangeValues(QnnModelWrapper& qnn_model_wrapper,
                                 const OrtNodeUnit& node_unit,
                                 Qnn_DataType_t& onnx_dtype_out,
                                 Qnn_DataType_t& static_dtype_out,
                                 std::vector<uint8_t>& static_bytes_out,
                                 uint32_t& count_out) const ORT_MUST_USE_RESULT;
};

namespace {

template <typename T>
Ort::Status ComputeRangeTyped(T start, T limit, T delta, std::vector<uint8_t>& out_bytes, uint32_t& count) {
  int64_t n = 0;
  if constexpr (std::is_floating_point_v<T>) {
    const double v = std::ceil((static_cast<double>(limit) - static_cast<double>(start)) /
                               static_cast<double>(delta));
    if (!std::isfinite(v) || v > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
      return MAKE_EP_FAIL("Range: float32 element count is NaN, infinite, or exceeds uint32 range.");
    }
    n = (v > 0.0) ? static_cast<int64_t>(v) : 0;
  } else {
    const long long diff = static_cast<long long>(limit) - static_cast<long long>(start);
    const long long d = static_cast<long long>(delta);
    if ((diff > 0 && d > 0) || (diff < 0 && d < 0)) {
      const long long abs_diff = diff < 0 ? -diff : diff;
      const long long abs_delta = d < 0 ? -d : d;
      n = static_cast<int64_t>((abs_diff + abs_delta - 1) / abs_delta);
    }
  }
  count = static_cast<uint32_t>(n);
  out_bytes.resize(static_cast<size_t>(n) * sizeof(T));
  T* p = reinterpret_cast<T*>(out_bytes.data());
  for (int64_t i = 0; i < n; ++i) {
    p[i] = static_cast<T>(start + static_cast<T>(i) * delta);
  }
  return Ort::Status();
}

}  // namespace

Ort::Status RangeOpBuilder::ComputeRangeValues(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
                                               Qnn_DataType_t& onnx_dtype_out,
                                               Qnn_DataType_t& static_dtype_out,
                                               std::vector<uint8_t>& static_bytes_out,
                                               uint32_t& count_out) const {
  const auto& inputs = node_unit.Inputs();
  RETURN_IF_NOT(inputs.size() == 3, "Range expects exactly 3 inputs (start, limit, delta).");

  TensorInfo infos[3] = {};
  std::vector<uint8_t> bytes[3];
  for (size_t i = 0; i < 3; ++i) {
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[i], infos[i]));
    RETURN_IF_NOT(infos[i].is_initializer,
                  "Range: all inputs (start, limit, delta) must be constant initializers.");
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(infos[i].initializer_tensor, bytes[i]));
  }

  const Qnn_DataType_t dtype = infos[0].qnn_data_type;
  RETURN_IF_NOT(dtype == infos[1].qnn_data_type && dtype == infos[2].qnn_data_type,
                "Range: start, limit, and delta must share the same data type.");
  onnx_dtype_out = dtype;

  switch (dtype) {
    case QNN_DATATYPE_FLOAT_32: {
      const float start = *reinterpret_cast<const float*>(bytes[0].data());
      const float limit = *reinterpret_cast<const float*>(bytes[1].data());
      const float delta = *reinterpret_cast<const float*>(bytes[2].data());
      RETURN_IF_NOT(delta != 0.0f, "Range: delta must be non-zero.");
      RETURN_IF_ERROR(ComputeRangeTyped<float>(start, limit, delta, static_bytes_out, count_out));
      static_dtype_out = QNN_DATATYPE_FLOAT_32;
      break;
    }
    case QNN_DATATYPE_INT_32: {
      const int32_t start = *reinterpret_cast<const int32_t*>(bytes[0].data());
      const int32_t limit = *reinterpret_cast<const int32_t*>(bytes[1].data());
      const int32_t delta = *reinterpret_cast<const int32_t*>(bytes[2].data());
      RETURN_IF_NOT(delta != 0, "Range: delta must be non-zero.");
      RETURN_IF_ERROR(ComputeRangeTyped<int32_t>(start, limit, delta, static_bytes_out, count_out));
      static_dtype_out = QNN_DATATYPE_INT_32;
      break;
    }
    case QNN_DATATYPE_INT_64: {
      const int64_t start = *reinterpret_cast<const int64_t*>(bytes[0].data());
      const int64_t limit = *reinterpret_cast<const int64_t*>(bytes[1].data());
      const int64_t delta = *reinterpret_cast<const int64_t*>(bytes[2].data());
      RETURN_IF_NOT(delta != 0, "Range: delta must be non-zero.");
      constexpr int64_t kI32Min = std::numeric_limits<int32_t>::min();
      constexpr int64_t kI32Max = std::numeric_limits<int32_t>::max();
      RETURN_IF_NOT(start >= kI32Min && start <= kI32Max &&
                        limit >= kI32Min && limit <= kI32Max &&
                        delta >= kI32Min && delta <= kI32Max,
                    "Range: int64 start/limit/delta exceed int32 range; not supported by QNN EP.");
      // Compute as INT_32; a Cast node inserted downstream restores the INT_64
      // contract (QNN Transpose does not accept INT_64 on NPU).
      RETURN_IF_ERROR(ComputeRangeTyped<int32_t>(static_cast<int32_t>(start),
                                                 static_cast<int32_t>(limit),
                                                 static_cast<int32_t>(delta),
                                                 static_bytes_out, count_out));
      static_dtype_out = QNN_DATATYPE_INT_32;
      break;
    }
    default:
      return MAKE_EP_FAIL("Range: QNN EP currently supports only float32, int32, and int64.");
  }
  return Ort::Status();
}

Ort::Status RangeOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const auto& inputs = node_unit.Inputs();
  RETURN_IF_NOT(inputs.size() == 3, "Range: expected 3 inputs (start, limit, delta).");
  for (size_t i = 0; i < 3; ++i) {
    RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(inputs[i].name),
                  "Range: all inputs (start, limit, delta) must be constant initializers. "
                  "Dynamic Range inputs are not supported by QNN EP.");
  }
  // Validate dtype, delta != 0, int64 range, and float NaN/Inf/overflow.
  Qnn_DataType_t onnx_dtype = QNN_DATATYPE_UNDEFINED;
  Qnn_DataType_t static_dtype = QNN_DATATYPE_UNDEFINED;
  std::vector<uint8_t> bytes;
  uint32_t count = 0;
  return ComputeRangeValues(qnn_model_wrapper, node_unit, onnx_dtype, static_dtype, bytes, count);
}

Ort::Status RangeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        std::vector<std::string>&& input_names,
                                                        const Ort::Logger& logger,
                                                        bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(input_names);

  Qnn_DataType_t onnx_dtype = QNN_DATATYPE_UNDEFINED;
  Qnn_DataType_t static_dtype = QNN_DATATYPE_UNDEFINED;
  std::vector<uint8_t> static_bytes;
  uint32_t count = 0;
  RETURN_IF_ERROR(ComputeRangeValues(qnn_model_wrapper, node_unit,
                                     onnx_dtype, static_dtype, static_bytes, count));

  const auto& outputs = node_unit.Outputs();
  RETURN_IF_NOT(outputs.size() == 1, "Range produces exactly one output.");
  const std::string& onnx_output_name = outputs[0].name;
  const std::vector<uint32_t> out_shape = {count};
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(onnx_output_name);

  const std::string static_name = utils::UniqueNameGenerator().New(node_unit, "_range_values");
  QnnTensorWrapper static_tensor(static_name,
                                 QNN_TENSOR_TYPE_STATIC,
                                 static_dtype,
                                 QnnQuantParamsWrapper(),
                                 std::vector<uint32_t>(out_shape),
                                 std::move(static_bytes));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(static_tensor)),
                "Failed to add Range static-values tensor.");

  if (onnx_dtype == QNN_DATATYPE_INT_64) {
    // INT_64 path: STATIC(INT_32) → Cast(INT_32→INT_64) → output.
    // One node; no intermediate Transpose needed since Cast bridges STATIC→output directly.
    const Qnn_TensorType_t out_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper out_tensor(onnx_output_name, out_type, QNN_DATATYPE_INT_64,
                                QnnQuantParamsWrapper(), std::vector<uint32_t>(out_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_tensor)),
                  "Failed to add Range int64 output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_CAST,
                      {static_name},
                      {onnx_output_name},
                      {},
                      do_op_validation),
                  "Failed to add Range int64 Cast node.");
  } else {
    // float32 / int32 path: STATIC → Transpose(perm={0}) → output.
    // Identity-permutation Transpose is the standard zero-cost bridge from STATIC to output.
    const Qnn_TensorType_t out_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper out_tensor(onnx_output_name, out_type, static_dtype,
                                QnnQuantParamsWrapper(), std::vector<uint32_t>(out_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_tensor)),
                  "Failed to add Range Transpose output tensor.");

    std::vector<uint32_t> perm_data = {0};
    QnnParamWrapper perm_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                               std::vector<uint32_t>{1}, std::move(perm_data));
    const std::string perm_param_name = perm_param.GetParamTensorName();
    qnn_model_wrapper.AddParamWrapper(std::move(perm_param));

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_TRANSPOSE,
                      {static_name},
                      {onnx_output_name},
                      {perm_param_name},
                      do_op_validation),
                  "Failed to add Range QNN Transpose node.");
  }

  return Ort::Status();
}

void CreateRangeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RangeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
