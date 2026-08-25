// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <functional>
#include <limits>
#include <numeric>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Input indices for QLinearMatMul (opset 10 and 21)
// 0: a            - quantized input matrix A
// 1: a_scale      - scale for A (scalar initializer, float32/fp16/bf16)
// 2: a_zero_point - zero point for A (scalar initializer, int8/uint8, optional)
// 3: b            - quantized input matrix B
// 4: b_scale      - scale for B (scalar initializer, float32/fp16/bf16)
// 5: b_zero_point - zero point for B (scalar initializer, int8/uint8, optional)
// 6: y_scale      - scale for output Y (scalar initializer, float32/fp16/bf16)
// 7: y_zero_point - zero point for output Y (scalar initializer, int8/uint8, optional)
static constexpr size_t kIdxA = 0;
static constexpr size_t kIdxAScale = 1;
static constexpr size_t kIdxAZeroPoint = 2;
static constexpr size_t kIdxB = 3;
static constexpr size_t kIdxBScale = 4;
static constexpr size_t kIdxBZeroPoint = 5;
static constexpr size_t kIdxYScale = 6;
static constexpr size_t kIdxYZeroPoint = 7;

/**
 * Translates ONNX QLinearMatMul into a QNN MatMul or FullyConnected node.
 *
 * QLinearMatMul carries quantization parameters as explicit inputs (scale + zero_point).
 * Since QNN encodes quant params in tensor metadata rather than as separate op inputs,
 * we read the scale/zp initializers, build QnnQuantParamsWrapper objects from them,
 * and attach those to the QNN tensor wrappers for A, B, and Y.  The rest of the shape
 * handling (rank-1 reshapes, FullyConnected dispatch) is identical to MatMulOpBuilder.
 */
class QLinearMatMulOpBuilder : public BaseOpBuilder {
 public:
  QLinearMatMulOpBuilder() : BaseOpBuilder("QLinearMatMulOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QLinearMatMulOpBuilder);

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
  // Reads a scalar float32 scale from an initializer, upcasting from fp16/bf16 if necessary.
  static Ort::Status ReadScaleAsFloat32(const QnnModelWrapper& qnn_model_wrapper,
                                        const OrtValueInfo* scale_tensor,
                                        float& out_scale);

  // Reads a scalar int32 zero-point from an initializer (int8 or uint8).
  // Returns 0 if zp_tensor is null (absent optional input).
  static Ort::Status ReadZeroPointAsInt32(const QnnModelWrapper& qnn_model_wrapper,
                                          const OrtValueInfo* zp_tensor,
                                          int32_t& out_zp);

  // Builds a per-tensor QnnQuantParamsWrapper from a scale and zero-point initializer pair.
  // scale_input and zp_input are entries from node_unit.Inputs().
  // zp_input may be absent (Exists() == false or empty name) — defaults to zero-point = 0.
  static Ort::Status BuildQuantParam(const QnnModelWrapper& qnn_model_wrapper,
                                     const OrtNodeUnitIODef& scale_input,
                                     const OrtNodeUnitIODef& zp_input,
                                     QnnQuantParamsWrapper& out_quant_param);

  // Validates that all scale/zp inputs are scalar initializers (required for static graph compilation).
  static Ort::Status ValidateQuantInputs(const QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit);

  // Decides whether to translate to QNN FullyConnected (vs MatMul). Mirrors MatMulOpBuilder::CheckInputs
  // so ProcessInputs and ProcessAttributesAndOutputs make the identical decision. quant_a is A's
  // quantization params (used to disable FC for per-channel A with rank > 2).
  static bool DecideUseFullyConnected(const QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& node_unit,
                                      const std::vector<uint32_t>& shape_a,
                                      const std::vector<uint32_t>& shape_b,
                                      Qnn_DataType_t qnn_dtype_a,
                                      Qnn_DataType_t qnn_dtype_b,
                                      const QnnQuantParamsWrapper& quant_a);
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Ort::Status QLinearMatMulOpBuilder::ReadScaleAsFloat32(const QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtValueInfo* scale_tensor,
                                                       float& out_scale) {
  std::vector<float> scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(scale_tensor, scales));
  RETURN_IF(scales.empty(), "QLinearMatMul: scale initializer unpacked to empty vector.");
  out_scale = scales[0];
  return Ort::Status();
}

Ort::Status QLinearMatMulOpBuilder::ReadZeroPointAsInt32(const QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtValueInfo* zp_tensor,
                                                         int32_t& out_zp) {
  if (zp_tensor == nullptr) {
    out_zp = 0;
    return Ort::Status();
  }

  std::vector<int32_t> zero_points;
  ONNXTensorElementDataType onnx_dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(zp_tensor, zero_points, onnx_dt));
  RETURN_IF(zero_points.empty(), "QLinearMatMul: zero_point initializer unpacked to empty vector.");
  out_zp = zero_points[0];
  return Ort::Status();
}

Ort::Status QLinearMatMulOpBuilder::BuildQuantParam(const QnnModelWrapper& qnn_model_wrapper,
                                                    const OrtNodeUnitIODef& scale_input,
                                                    const OrtNodeUnitIODef& zp_input,
                                                    QnnQuantParamsWrapper& out_quant_param) {
  RETURN_IF(!scale_input.Exists(), "QLinearMatMul: scale input does not exist.");
  RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(scale_input.name),
            "QLinearMatMul: scale must be a compile-time constant (initializer).");

  const OrtValueInfo* scale_tensor = qnn_model_wrapper.GetConstantTensor(scale_input.name);
  RETURN_IF(scale_tensor == nullptr, "QLinearMatMul: could not retrieve scale initializer.");

  float scale = 0.0f;
  RETURN_IF_ERROR(ReadScaleAsFloat32(qnn_model_wrapper, scale_tensor, scale));

  int32_t zero_point = 0;
  const OrtValueInfo* zp_tensor = nullptr;
  if (zp_input.Exists() && !zp_input.name.empty()) {
    RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(zp_input.name),
              "QLinearMatMul: zero_point must be a compile-time constant (initializer).");
    zp_tensor = qnn_model_wrapper.GetConstantTensor(zp_input.name);
  }
  RETURN_IF_ERROR(ReadZeroPointAsInt32(qnn_model_wrapper, zp_tensor, zero_point));

  // UnpackZeroPoints already returns -zp (QNN offset convention); pass through directly.
  out_quant_param = QnnQuantParamsWrapper::PerTensor(scale, zero_point);
  return Ort::Status();
}

Ort::Status QLinearMatMulOpBuilder::ValidateQuantInputs(const QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit) {
  const auto& inputs = node_unit.Inputs();

  // Validate each scale input: must be an initializer and scalar (1 element = per-tensor).
  const std::array<size_t, 3> scale_indices = {kIdxAScale, kIdxBScale, kIdxYScale};
  for (size_t idx : scale_indices) {
    if (idx >= inputs.size() || !inputs[idx].Exists()) {
      return MAKE_EP_FAIL("QLinearMatMul: required scale input is missing.");
    }
    RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(inputs[idx].name),
              "QLinearMatMul: scale inputs must be compile-time constants.");

    // Reject per-row/per-column scales: shape must be scalar or {1}.
    if (inputs[idx].shape.has_value()) {
      const auto& shape = inputs[idx].shape.value();
      if (!shape.empty()) {
        const int64_t num_elems = std::accumulate(shape.begin(), shape.end(),
                                                  static_cast<int64_t>(1), std::multiplies<int64_t>());
        RETURN_IF(num_elems != 1, "QLinearMatMul: only per-tensor (scalar) quantization is supported.");
      }
    }
  }

  // Validate optional zero-point inputs: if present, must be initializers.
  const std::array<size_t, 3> zp_indices = {kIdxAZeroPoint, kIdxBZeroPoint, kIdxYZeroPoint};
  for (size_t idx : zp_indices) {
    if (idx < inputs.size() && inputs[idx].Exists() && !inputs[idx].name.empty()) {
      RETURN_IF(!qnn_model_wrapper.IsEffectivelyConstantInput(inputs[idx].name),
                "QLinearMatMul: zero_point inputs must be compile-time constants.");
    }
  }

  return Ort::Status();
}

bool QLinearMatMulOpBuilder::DecideUseFullyConnected(const QnnModelWrapper& qnn_model_wrapper,
                                                     const OrtNodeUnit& node_unit,
                                                     const std::vector<uint32_t>& shape_a,
                                                     const std::vector<uint32_t>& shape_b,
                                                     Qnn_DataType_t qnn_dtype_a,
                                                     Qnn_DataType_t qnn_dtype_b,
                                                     const QnnQuantParamsWrapper& quant_a) {
#if QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR <= 20
  // Validation crashes if QNN FullyConnected is used in QNN SDK versions 2.26 - 2.27.
  // Just use QNN MatMul for these older QNN SDK versions.
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(shape_a);
  ORT_UNUSED_PARAMETER(shape_b);
  ORT_UNUSED_PARAMETER(qnn_dtype_a);
  ORT_UNUSED_PARAMETER(qnn_dtype_b);
  ORT_UNUSED_PARAMETER(quant_a);
  return false;
#else
  const auto& inputs = node_unit.Inputs();
  const bool b_is_initializer = qnn_model_wrapper.IsEffectivelyConstantInput(inputs[kIdxB].name);
  const bool a_is_initializer = qnn_model_wrapper.IsEffectivelyConstantInput(inputs[kIdxA].name);

  // Use FullyConnected if B is a rank-2 initializer or a rank-1 tensor.
  bool use_fully_connected = (shape_b.size() == 2 && b_is_initializer) || shape_b.size() == 1;
  // FullyConnected cannot set output quant params for a reshaped rank-2 tensor when A is
  // per-channel quantized with rank > 2.
  use_fully_connected = use_fully_connected && !(quant_a.IsPerChannel() && shape_a.size() > 2);
  // Don't use FullyConnected if both inputs are dynamic and 16-bit quantized (QNN validation fails).
  use_fully_connected = use_fully_connected && !(utils::IsQuant16bit(qnn_dtype_a) && !a_is_initializer &&
                                                 utils::IsQuant16bit(qnn_dtype_b) && !b_is_initializer);
  return use_fully_connected;
#endif
}

// ---------------------------------------------------------------------------
// ProcessInputs
// ---------------------------------------------------------------------------

Ort::Status QLinearMatMulOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const OrtNodeUnit& node_unit,
                                                  const Ort::Logger& logger,
                                                  std::vector<std::string>& input_names,
                                                  bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_ERROR(ValidateQuantInputs(qnn_model_wrapper, node_unit));
  }

  const auto& inputs = node_unit.Inputs();

  // Build quant params for A and B from explicit scale/zp inputs.
  QnnQuantParamsWrapper quant_a;
  RETURN_IF_ERROR(BuildQuantParam(qnn_model_wrapper, inputs[kIdxAScale],
                                  inputs.size() > kIdxAZeroPoint ? inputs[kIdxAZeroPoint]
                                                                 : OrtNodeUnitIODef{},
                                  quant_a));

  QnnQuantParamsWrapper quant_b;
  RETURN_IF_ERROR(BuildQuantParam(qnn_model_wrapper, inputs[kIdxBScale],
                                  inputs.size() > kIdxBZeroPoint ? inputs[kIdxBZeroPoint]
                                                                 : OrtNodeUnitIODef{},
                                  quant_b));

  // Determine QNN data types for A and B.
  Qnn_DataType_t qnn_dtype_a = QNN_DATATYPE_UNDEFINED;
  Qnn_DataType_t qnn_dtype_b = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxA].type, qnn_dtype_a));
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxB].type, qnn_dtype_b));

  // Get shapes for A and B.
  std::vector<uint32_t> shape_a, shape_b;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(inputs[kIdxA].shape, shape_a), "QLinearMatMul: cannot get shape of A.");
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(inputs[kIdxB].shape, shape_b), "QLinearMatMul: cannot get shape of B.");

  // Decide MatMul vs FullyConnected (same rule as MatMulOpBuilder).
  const bool b_is_initializer = qnn_model_wrapper.IsEffectivelyConstantInput(inputs[kIdxB].name);
  const bool use_fully_connected =
      DecideUseFullyConnected(qnn_model_wrapper, node_unit, shape_a, shape_b, qnn_dtype_a, qnn_dtype_b, quant_a);

  // ---- Process input A ----
  const std::string& org_a_name = inputs[kIdxA].name;
  std::string actual_a_name = org_a_name;
  const bool a_is_rank1 = shape_a.size() == 1;
  const bool reshape_a = a_is_rank1 || (use_fully_connected && shape_a.size() > 2);

  if (reshape_a) {
    actual_a_name = utils::UniqueNameGenerator().New(org_a_name, "_reshape");
    std::vector<uint32_t> shape_a_2d;
    QnnQuantParamsWrapper quant_a_2d = quant_a.Copy();
    if (a_is_rank1) {
      shape_a_2d = {1, shape_a[0]};
    } else {
      const int64_t batch_i64 = std::accumulate(shape_a.begin(), shape_a.end() - 1,
                                                static_cast<int64_t>(1), std::multiplies<int64_t>());
      RETURN_IF(batch_i64 <= 0 || batch_i64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "QLinearMatMul: A batch dimension overflows uint32_t.");
      shape_a_2d = {static_cast<uint32_t>(batch_i64), shape_a.back()};
    }
    if (a_is_rank1) {
      RETURN_IF_ERROR(quant_a_2d.HandleUnsqueeze<uint32_t>(shape_a, shape_a_2d));
    }

    if (qnn_model_wrapper.IsEffectivelyConstantInput(org_a_name)) {
      std::vector<uint8_t> unpacked;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(
          qnn_model_wrapper.GetConstantTensor(org_a_name), unpacked));
      QnnTensorWrapper tw(actual_a_name, QNN_TENSOR_TYPE_STATIC, qnn_dtype_a,
                          std::move(quant_a_2d), std::move(shape_a_2d), std::move(unpacked));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "QLinearMatMul: failed to add reshaped A.");
    } else {
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
          org_a_name, actual_a_name, shape_a, shape_a_2d, qnn_dtype_a, quant_a, quant_a_2d,
          do_op_validation, qnn_model_wrapper.IsGraphInput(org_a_name), false));
    }
  } else {
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(actual_a_name)) {
      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_a_name);
      std::vector<uint8_t> unpacked;
      if (qnn_model_wrapper.IsEffectivelyConstantInput(org_a_name)) {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(
            qnn_model_wrapper.GetConstantTensor(org_a_name), unpacked));
      }
      QnnTensorWrapper tw(actual_a_name, tensor_type, qnn_dtype_a, quant_a.Copy(),
                          std::vector<uint32_t>(shape_a), std::move(unpacked));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "QLinearMatMul: failed to add A tensor.");
    } else {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip: " + actual_a_name).c_str());
    }
  }
  input_names.emplace_back(actual_a_name);

  // ---- Process input B ----
  const std::string& org_b_name = inputs[kIdxB].name;
  std::string actual_b_name = org_b_name;
  const bool b_is_rank1 = shape_b.size() == 1;

  if (use_fully_connected) {
    // FullyConnected expects B in shape [n, k] (transposed from ONNX [k, n]).
    std::vector<uint32_t> shape_b_fc;
    QnnQuantParamsWrapper quant_b_fc = quant_b.Copy();

    if (b_is_rank1) {
      actual_b_name = utils::UniqueNameGenerator().New(org_b_name, "_reshape");
      shape_b_fc = {1, shape_b[0]};
      RETURN_IF_ERROR(quant_b_fc.HandleUnsqueeze<uint32_t>(shape_b, shape_b_fc));
    } else {
      // Rank-2: transpose to [n, k].
      actual_b_name = utils::UniqueNameGenerator().New(org_b_name, "_transpose");
      shape_b_fc = {shape_b[1], shape_b[0]};
      RETURN_IF_ERROR(quant_b_fc.HandleTranspose<uint32_t>(std::vector<uint32_t>({1, 0})));
    }

    if (b_is_initializer) {
      std::vector<uint8_t> unpacked;
      if (!b_is_rank1) {
        std::vector<uint32_t> shape_b_copy = shape_b;
        RETURN_IF_ERROR(utils::TwoDimensionTranspose(qnn_model_wrapper, shape_b_copy,
                                                     qnn_model_wrapper.GetConstantTensor(org_b_name),
                                                     unpacked, logger));
      } else {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(
            qnn_model_wrapper.GetConstantTensor(org_b_name), unpacked));
      }
      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_b_name);
      QnnTensorWrapper tw(actual_b_name, tensor_type, qnn_dtype_b,
                          std::move(quant_b_fc), std::move(shape_b_fc), std::move(unpacked));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "QLinearMatMul: failed to add FC B tensor.");
    } else {
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
          org_b_name, actual_b_name, shape_b, shape_b_fc, qnn_dtype_b, quant_b, quant_b_fc,
          do_op_validation, qnn_model_wrapper.IsGraphInput(org_b_name), false));
    }
  } else {
    // QNN MatMul path.
    if (b_is_rank1) {
      actual_b_name = utils::UniqueNameGenerator().New(org_b_name, "_reshape");
      std::vector<uint32_t> shape_b_2d = {shape_b[0], 1};
      QnnQuantParamsWrapper quant_b_2d = quant_b.Copy();
      RETURN_IF_ERROR(quant_b_2d.HandleUnsqueeze<uint32_t>(shape_b, shape_b_2d));

      if (b_is_initializer) {
        std::vector<uint8_t> unpacked;
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(
            qnn_model_wrapper.GetConstantTensor(org_b_name), unpacked));
        Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_b_name);
        QnnTensorWrapper tw(actual_b_name, tensor_type, qnn_dtype_b,
                            std::move(quant_b_2d), std::move(shape_b_2d), std::move(unpacked));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "QLinearMatMul: failed to add reshaped B.");
      } else {
        RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
            org_b_name, actual_b_name, shape_b, shape_b_2d, qnn_dtype_b, quant_b, quant_b_2d,
            do_op_validation, qnn_model_wrapper.IsGraphInput(org_b_name), false));
      }
    } else {
      if (!qnn_model_wrapper.IsQnnTensorWrapperExist(actual_b_name)) {
        Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_b_name);
        std::vector<uint8_t> unpacked;
        if (b_is_initializer) {
          RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(
              qnn_model_wrapper.GetConstantTensor(org_b_name), unpacked));
        }
        QnnTensorWrapper tw(actual_b_name, tensor_type, qnn_dtype_b, quant_b.Copy(),
                            std::vector<uint32_t>(shape_b), std::move(unpacked));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "QLinearMatMul: failed to add B tensor.");
      } else {
        ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip: " + actual_b_name).c_str());
      }
    }
  }
  input_names.emplace_back(actual_b_name);

  return Ort::Status();
}

// ---------------------------------------------------------------------------
// ProcessAttributesAndOutputs
// ---------------------------------------------------------------------------

Ort::Status QLinearMatMulOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                const OrtNodeUnit& node_unit,
                                                                std::vector<std::string>&& input_names,
                                                                const Ort::Logger& /*logger*/,
                                                                bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  // Re-derive shapes, quant params, and the FullyConnected decision. ProcessInputs already computed
  // these, but the builder is stateless across the two phases (BaseOpBuilder calls them separately),
  // so we recompute here to keep the QNN op type, output shape, and reshape handling consistent with
  // the tensors emitted in ProcessInputs. DecideUseFullyConnected is the single source of truth for
  // the MatMul-vs-FullyConnected choice.
  std::vector<uint32_t> shape_a, shape_b;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(inputs[kIdxA].shape, shape_a), "QLinearMatMul: cannot get shape of A.");
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(inputs[kIdxB].shape, shape_b), "QLinearMatMul: cannot get shape of B.");

  Qnn_DataType_t qnn_dtype_a = QNN_DATATYPE_UNDEFINED;
  Qnn_DataType_t qnn_dtype_b = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxA].type, qnn_dtype_a));
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxB].type, qnn_dtype_b));

  QnnQuantParamsWrapper quant_a;
  RETURN_IF_ERROR(BuildQuantParam(qnn_model_wrapper, inputs[kIdxAScale],
                                  inputs.size() > kIdxAZeroPoint ? inputs[kIdxAZeroPoint]
                                                                 : OrtNodeUnitIODef{},
                                  quant_a));

  const bool use_fully_connected =
      DecideUseFullyConnected(qnn_model_wrapper, node_unit, shape_a, shape_b, qnn_dtype_a, qnn_dtype_b, quant_a);

  const bool a_is_rank1 = shape_a.size() == 1;
  const bool b_is_rank1 = shape_b.size() == 1;
  const bool reshape_output = a_is_rank1 || b_is_rank1 || (use_fully_connected && shape_a.size() > 2);

  // Build output quant params from y_scale / y_zero_point.
  QnnQuantParamsWrapper quant_y;
  RETURN_IF_ERROR(BuildQuantParam(qnn_model_wrapper, inputs[kIdxYScale],
                                  inputs.size() > kIdxYZeroPoint ? inputs[kIdxYZeroPoint]
                                                                 : OrtNodeUnitIODef{},
                                  quant_y));

  // Determine output QNN data type from the ONNX output type.
  const auto& outputs = node_unit.Outputs();
  Qnn_DataType_t qnn_dtype_y = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, outputs[0].type, qnn_dtype_y));

  // Determine output shape.
  const std::string& org_output_name = outputs[0].name;
  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(outputs[0].shape, output_shape),
                "QLinearMatMul: cannot get output shape.");

  std::string op_output_name = org_output_name;
  std::vector<uint32_t> op_output_shape = output_shape;
  QnnQuantParamsWrapper op_output_quant = quant_y.Copy();

  if (reshape_output) {
    op_output_name = utils::UniqueNameGenerator().New(org_output_name, "_reshape");
    if (use_fully_connected && shape_a.size() > 2) {
      const int64_t batch_i64 = std::accumulate(shape_a.begin(), shape_a.end() - 1,
                                                static_cast<int64_t>(1), std::multiplies<int64_t>());
      RETURN_IF(batch_i64 <= 0 || batch_i64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "QLinearMatMul: output batch dimension overflows uint32_t.");
      op_output_shape = {static_cast<uint32_t>(batch_i64), b_is_rank1 ? 1u : shape_b.back()};
      RETURN_IF(op_output_quant.IsPerChannel(), "QLinearMatMul: output does not support per-channel quant.");
    } else {
      if (b_is_rank1) {
        op_output_shape.emplace_back(1);
      } else if (a_is_rank1) {
        op_output_shape.insert(op_output_shape.end() - 1, 1);
      }
      RETURN_IF_ERROR(op_output_quant.HandleUnsqueeze<uint32_t>(output_shape, op_output_shape));
    }
  }

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
  const bool is_op_output_graph_output = is_graph_output && !reshape_output;
  const Qnn_TensorType_t op_output_type =
      is_op_output_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper output_tw(op_output_name, op_output_type, qnn_dtype_y,
                             op_output_quant.Copy(), std::vector<uint32_t>(op_output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tw)),
                "QLinearMatMul: failed to add output tensor.");

  // Add transpose params (required even at default=false for older QNN SDK versions).
  std::vector<std::string> param_tensor_names;
  if (!use_fully_connected) {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 0;

    QnnParamWrapper transpose_in0_param(node_unit.Index(), node_unit.Name(),
                                        QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, scalar_param);
    param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in0_param));

    QnnParamWrapper transpose_in1_param(node_unit.Index(), node_unit.Name(),
                                        QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, scalar_param);
    param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in1_param));
  }

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::UniqueNameGenerator().New(node_unit),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    use_fully_connected ? QNN_OP_FULLY_CONNECTED : QNN_OP_MAT_MUL,
                    std::move(input_names), {op_output_name},
                    std::move(param_tensor_names), do_op_validation),
                "QLinearMatMul: failed to create QNN node.");

  if (reshape_output) {
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        op_output_name, org_output_name, op_output_shape, output_shape,
        qnn_dtype_y, op_output_quant, quant_y, do_op_validation, false, is_graph_output));
  }

  return Ort::Status();
}

void CreateQLinearMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QLinearMatMulOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
