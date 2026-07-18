// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/qlinear_matmul_node_group.h"

#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// ---------------------------------------------------------------------------
// Input index constants (QLinearMatMul opset 10 / 21)
// ---------------------------------------------------------------------------
static constexpr size_t kIdxA = 0;
static constexpr size_t kIdxAScale = 1;
static constexpr size_t kIdxAZeroPoint = 2;
static constexpr size_t kIdxB = 3;
static constexpr size_t kIdxBScale = 4;
static constexpr size_t kIdxBZeroPoint = 5;
static constexpr size_t kIdxYScale = 6;
static constexpr size_t kIdxYZeroPoint = 7;

namespace {

inline bool IsQuant16bit(Qnn_DataType_t qnn_data_type) {
  return qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16 || qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16;
}

// Reads a scalar float32 scale from an initializer, upcasting from fp16/bf16 if necessary.
Ort::Status ReadScaleAsFloat32(const QnnModelWrapper& qmw,
                               const OrtValueInfo* scale_tensor,
                               float& out_scale) {
  std::vector<float> scales;
  RETURN_IF_ERROR(qmw.UnpackScales(scale_tensor, scales));
  RETURN_IF(scales.empty(), "QLinearMatMul: scale initializer unpacked to empty vector.");
  out_scale = scales[0];
  return Ort::Status();
}

// Reads a scalar int32 zero-point from an initializer (int8 or uint8).
// Returns 0 if zp_tensor is null (absent optional input).
Ort::Status ReadZeroPointAsInt32(const QnnModelWrapper& qmw,
                                 const OrtValueInfo* zp_tensor,
                                 int32_t& out_zp) {
  if (zp_tensor == nullptr) {
    out_zp = 0;
    return Ort::Status();
  }
  std::vector<int32_t> zero_points;
  ONNXTensorElementDataType onnx_dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  RETURN_IF_ERROR(qmw.UnpackZeroPoints(zp_tensor, zero_points, onnx_dt));
  RETURN_IF(zero_points.empty(), "QLinearMatMul: zero_point initializer unpacked to empty vector.");
  out_zp = zero_points[0];
  return Ort::Status();
}

// Builds a per-tensor QnnQuantParamsWrapper from a scale/zp initializer pair.
// zp_input may have Exists()==false or empty name — defaults to zero_point = 0.
Ort::Status BuildQuantParam(const QnnModelWrapper& qmw,
                            const OrtNodeUnitIODef& scale_input,
                            const OrtNodeUnitIODef& zp_input,
                            QnnQuantParamsWrapper& out_quant_param) {
  RETURN_IF(!scale_input.Exists(), "QLinearMatMul: scale input does not exist.");
  RETURN_IF(!qmw.IsEffectivelyConstantInput(scale_input.name),
            "QLinearMatMul: scale must be a compile-time constant (initializer).");

  const OrtValueInfo* scale_tensor = qmw.GetConstantTensor(scale_input.name);
  RETURN_IF(scale_tensor == nullptr, "QLinearMatMul: could not retrieve scale initializer.");

  float scale = 0.0f;
  RETURN_IF_ERROR(ReadScaleAsFloat32(qmw, scale_tensor, scale));

  int32_t zero_point = 0;
  const OrtValueInfo* zp_tensor = nullptr;
  if (zp_input.Exists() && !zp_input.name.empty()) {
    RETURN_IF(!qmw.IsEffectivelyConstantInput(zp_input.name),
              "QLinearMatMul: zero_point must be a compile-time constant (initializer).");
    zp_tensor = qmw.GetConstantTensor(zp_input.name);
  }
  RETURN_IF_ERROR(ReadZeroPointAsInt32(qmw, zp_tensor, zero_point));

  // UnpackZeroPoints already returns -zp (QNN offset convention); pass through directly.
  out_quant_param = QnnQuantParamsWrapper::PerTensor(scale, zero_point);
  return Ort::Status();
}

// Validates that all scale/zp inputs are scalar (per-tensor) initializers.
Ort::Status ValidateQuantInputs(const QnnModelWrapper& qmw, const OrtNodeUnit& node_unit) {
  const auto& inputs = node_unit.Inputs();

  const std::array<size_t, 3> scale_indices = {kIdxAScale, kIdxBScale, kIdxYScale};
  for (size_t idx : scale_indices) {
    if (idx >= inputs.size() || !inputs[idx].Exists()) {
      return MAKE_EP_FAIL("QLinearMatMul: required scale input is missing.");
    }
    RETURN_IF(!qmw.IsEffectivelyConstantInput(inputs[idx].name),
              "QLinearMatMul: scale inputs must be compile-time constants.");

    // Reject per-row/per-column scales — shape must be scalar or {1}.
    if (inputs[idx].shape.has_value()) {
      const auto& shape = inputs[idx].shape.value();
      if (!shape.empty()) {
        const int64_t num_elems = std::accumulate(shape.begin(), shape.end(),
                                                  static_cast<int64_t>(1), std::multiplies<int64_t>());
        RETURN_IF(num_elems != 1, "QLinearMatMul: only per-tensor (scalar) quantization is supported.");
      }
    }
  }

  // Validate optional zero-point inputs: if present must be initializers.
  const std::array<size_t, 3> zp_indices = {kIdxAZeroPoint, kIdxBZeroPoint, kIdxYZeroPoint};
  for (size_t idx : zp_indices) {
    if (idx < inputs.size() && inputs[idx].Exists() && !inputs[idx].name.empty()) {
      RETURN_IF(!qmw.IsEffectivelyConstantInput(inputs[idx].name),
                "QLinearMatMul: zero_point inputs must be compile-time constants.");
    }
  }

  return Ort::Status();
}

// Decides whether to use QNN_OP_FULLY_CONNECTED instead of QNN_OP_MAT_MUL.
bool DecideUseFullyConnected(const QnnModelWrapper& qmw,
                             const OrtNodeUnit& node_unit,
                             const std::vector<uint32_t>& shape_a,
                             const std::vector<uint32_t>& shape_b,
                             Qnn_DataType_t qnn_dtype_a,
                             Qnn_DataType_t qnn_dtype_b,
                             const QnnQuantParamsWrapper& quant_a) {
#if QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR <= 20
  // Validation crashes if QNN FullyConnected is used in QNN SDK versions 2.26 - 2.27.
  ORT_UNUSED_PARAMETER(qmw);
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(shape_a);
  ORT_UNUSED_PARAMETER(shape_b);
  ORT_UNUSED_PARAMETER(qnn_dtype_a);
  ORT_UNUSED_PARAMETER(qnn_dtype_b);
  ORT_UNUSED_PARAMETER(quant_a);
  return false;
#else
  const auto& inputs = node_unit.Inputs();
  const bool b_is_initializer = qmw.IsEffectivelyConstantInput(inputs[kIdxB].name);
  const bool a_is_initializer = qmw.IsEffectivelyConstantInput(inputs[kIdxA].name);

  bool use_fully_connected = (shape_b.size() == 2 && b_is_initializer) || shape_b.size() == 1;
  use_fully_connected = use_fully_connected && !(quant_a.IsPerChannel() && shape_a.size() > 2);
  use_fully_connected = use_fully_connected && !(IsQuant16bit(qnn_dtype_a) && !a_is_initializer &&
                                                 IsQuant16bit(qnn_dtype_b) && !b_is_initializer);
  return use_fully_connected;
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

QLinearMatMulNodeGroup::QLinearMatMulNodeGroup(const OrtNodeUnit& node_unit)
    : node_unit_(&node_unit) {}

// ---------------------------------------------------------------------------
// TryFusion
// ---------------------------------------------------------------------------

std::unique_ptr<IQnnNodeGroup> QLinearMatMulNodeGroup::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& /*node_to_node_unit*/,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& /*node_unit_to_qnn_node_group*/,
    const Ort::Logger& logger) {
  // Only claim standalone SingleNode QLinearMatMul from the default ONNX domain.
  if (node_unit.OpType() != "QLinearMatMul" ||
      node_unit.UnitType() != OrtNodeUnit::Type::SingleNode ||
      node_unit.Domain() != kOnnxDomain) {
    return nullptr;
  }

  auto candidate = std::unique_ptr<QLinearMatMulNodeGroup>(new QLinearMatMulNodeGroup(node_unit));
  if (Ort::Status status = candidate->CreateOrValidateOnQnn(qnn_model_wrapper, /*validate=*/true, logger);
      !status.IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("QLinearMatMulNodeGroup rejected by QNN validate: " + status.GetErrorMessage()).c_str());
    return nullptr;
  }
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "QLinearMatMulNodeGroup matched and validated.");
  return candidate;
}

// ---------------------------------------------------------------------------
// IQnnNodeGroup interface
// ---------------------------------------------------------------------------

Ort::Status QLinearMatMulNodeGroup::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, /*validate=*/true, logger);
}

Ort::Status QLinearMatMulNodeGroup::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, /*validate=*/false, logger);
}

gsl::span<const OrtNodeUnit* const> QLinearMatMulNodeGroup::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{&node_unit_, 1ULL};
}

// ---------------------------------------------------------------------------
// Core implementation — ported from QLinearMatMulOpBuilder
// ---------------------------------------------------------------------------

Ort::Status QLinearMatMulNodeGroup::CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                                          bool validate,
                                                          const Ort::Logger& logger) const {
  const OrtNodeUnit& node_unit = *node_unit_;
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // Validate quant parameters in both passes (cheap, catches dynamic inputs early).
  RETURN_IF_ERROR(ValidateQuantInputs(qmw, node_unit));

  // Build quant params for A and B.
  QnnQuantParamsWrapper quant_a;
  RETURN_IF_ERROR(BuildQuantParam(qmw,
                                  inputs[kIdxAScale],
                                  inputs.size() > kIdxAZeroPoint ? inputs[kIdxAZeroPoint]
                                                                 : OrtNodeUnitIODef{},
                                  quant_a));
  QnnQuantParamsWrapper quant_b;
  RETURN_IF_ERROR(BuildQuantParam(qmw,
                                  inputs[kIdxBScale],
                                  inputs.size() > kIdxBZeroPoint ? inputs[kIdxBZeroPoint]
                                                                 : OrtNodeUnitIODef{},
                                  quant_b));

  // Determine QNN data types.
  Qnn_DataType_t qnn_dtype_a = QNN_DATATYPE_UNDEFINED;
  Qnn_DataType_t qnn_dtype_b = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxA].type, qnn_dtype_a));
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, inputs[kIdxB].type, qnn_dtype_b));

  // Get shapes.
  std::vector<uint32_t> shape_a, shape_b;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(inputs[kIdxA].shape, shape_a),
                "QLinearMatMul: cannot get shape of A.");
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(inputs[kIdxB].shape, shape_b),
                "QLinearMatMul: cannot get shape of B.");

  const bool b_is_initializer = qmw.IsEffectivelyConstantInput(inputs[kIdxB].name);
  const bool use_fully_connected =
      DecideUseFullyConnected(qmw, node_unit, shape_a, shape_b, qnn_dtype_a, qnn_dtype_b, quant_a);

  // ---- Process input A ----
  const std::string& org_a_name = inputs[kIdxA].name;
  std::string actual_a_name = org_a_name;
  const bool a_is_rank1 = shape_a.size() == 1;
  const bool reshape_a = a_is_rank1 || (use_fully_connected && shape_a.size() > 2);

  // Compute the A shape that will be presented to the QNN matmul/FC node.
  std::vector<uint32_t> a_qnn_shape;
  if (reshape_a) {
    if (a_is_rank1) {
      a_qnn_shape = {1, shape_a[0]};
    } else {
      const int64_t batch_i64 = std::accumulate(shape_a.begin(), shape_a.end() - 1,
                                                static_cast<int64_t>(1), std::multiplies<int64_t>());
      RETURN_IF(batch_i64 <= 0 || batch_i64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "QLinearMatMul: A batch dimension overflows uint32_t.");
      a_qnn_shape = {static_cast<uint32_t>(batch_i64), shape_a.back()};
    }
  } else {
    a_qnn_shape = shape_a;
  }

  if (reshape_a) {
    actual_a_name = utils::UniqueNameGenerator().New(org_a_name, "_reshape");
    QnnQuantParamsWrapper quant_a_2d = quant_a.Copy();
    if (a_is_rank1) {
      RETURN_IF_ERROR(quant_a_2d.HandleUnsqueeze<uint32_t>(shape_a, a_qnn_shape));
    }

    if (!validate) {
      if (qmw.IsEffectivelyConstantInput(org_a_name)) {
        std::vector<uint8_t> unpacked;
        RETURN_IF_ERROR(qmw.UnpackInitializerData(qmw.GetConstantTensor(org_a_name), unpacked));
        QnnTensorWrapper tw(actual_a_name, QNN_TENSOR_TYPE_STATIC, qnn_dtype_a,
                            std::move(quant_a_2d), std::vector<uint32_t>(a_qnn_shape), std::move(unpacked));
        RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                      "QLinearMatMul: failed to add reshaped A tensor.");
      } else {
        RETURN_IF_ERROR(qmw.AddReshapeNode(
            org_a_name, actual_a_name, shape_a, a_qnn_shape, qnn_dtype_a, quant_a, quant_a_2d,
            /*do_op_validation=*/false, qmw.IsGraphInput(org_a_name), /*is_graph_output=*/false));
      }
    }
  } else {
    if (!validate && !qmw.IsQnnTensorWrapperExist(actual_a_name)) {
      Qnn_TensorType_t tensor_type = qmw.GetTensorType(org_a_name);
      std::vector<uint8_t> unpacked;
      if (qmw.IsEffectivelyConstantInput(org_a_name)) {
        RETURN_IF_ERROR(qmw.UnpackInitializerData(qmw.GetConstantTensor(org_a_name), unpacked));
      }
      QnnTensorWrapper tw(actual_a_name, tensor_type, qnn_dtype_a, quant_a.Copy(),
                          std::vector<uint32_t>(shape_a), std::move(unpacked));
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                    "QLinearMatMul: failed to add A tensor.");
    }
  }

  // ---- Process input B ----
  const std::string& org_b_name = inputs[kIdxB].name;
  std::string actual_b_name = org_b_name;
  const bool b_is_rank1 = shape_b.size() == 1;

  // Compute the B shape that will be presented to the QNN matmul/FC node.
  std::vector<uint32_t> b_qnn_shape;
  if (use_fully_connected) {
    b_qnn_shape = b_is_rank1 ? std::vector<uint32_t>{1, shape_b[0]}
                             : std::vector<uint32_t>{shape_b[1], shape_b[0]};
  } else {
    b_qnn_shape = b_is_rank1 ? std::vector<uint32_t>{shape_b[0], 1} : shape_b;
  }

  if (use_fully_connected) {
    QnnQuantParamsWrapper quant_b_fc = quant_b.Copy();
    if (b_is_rank1) {
      actual_b_name = utils::UniqueNameGenerator().New(org_b_name, "_reshape");
      RETURN_IF_ERROR(quant_b_fc.HandleUnsqueeze<uint32_t>(shape_b, b_qnn_shape));
    } else {
      actual_b_name = utils::UniqueNameGenerator().New(org_b_name, "_transpose");
      RETURN_IF_ERROR(quant_b_fc.HandleTranspose<uint32_t>(std::vector<uint32_t>({1, 0})));
    }

    if (!validate) {
      if (b_is_initializer) {
        std::vector<uint8_t> unpacked;
        if (!b_is_rank1) {
          std::vector<uint32_t> shape_b_copy = shape_b;
          RETURN_IF_ERROR(utils::TwoDimensionTranspose(qmw, shape_b_copy,
                                                       qmw.GetConstantTensor(org_b_name),
                                                       unpacked, logger));
        } else {
          RETURN_IF_ERROR(qmw.UnpackInitializerData(qmw.GetConstantTensor(org_b_name), unpacked));
        }
        Qnn_TensorType_t tensor_type = qmw.GetTensorType(org_b_name);
        QnnTensorWrapper tw(actual_b_name, tensor_type, qnn_dtype_b,
                            std::move(quant_b_fc), std::vector<uint32_t>(b_qnn_shape), std::move(unpacked));
        RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                      "QLinearMatMul: failed to add FC B tensor.");
      } else {
        RETURN_IF_ERROR(qmw.AddReshapeNode(
            org_b_name, actual_b_name, shape_b, b_qnn_shape, qnn_dtype_b, quant_b, quant_b_fc,
            /*do_op_validation=*/false, qmw.IsGraphInput(org_b_name), /*is_graph_output=*/false));
      }
    }
  } else {
    // QNN MatMul path.
    if (b_is_rank1) {
      actual_b_name = utils::UniqueNameGenerator().New(org_b_name, "_reshape");
      QnnQuantParamsWrapper quant_b_2d = quant_b.Copy();
      RETURN_IF_ERROR(quant_b_2d.HandleUnsqueeze<uint32_t>(shape_b, b_qnn_shape));

      if (!validate) {
        if (b_is_initializer) {
          std::vector<uint8_t> unpacked;
          RETURN_IF_ERROR(qmw.UnpackInitializerData(qmw.GetConstantTensor(org_b_name), unpacked));
          Qnn_TensorType_t tensor_type = qmw.GetTensorType(org_b_name);
          QnnTensorWrapper tw(actual_b_name, tensor_type, qnn_dtype_b,
                              std::move(quant_b_2d), std::vector<uint32_t>(b_qnn_shape), std::move(unpacked));
          RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                        "QLinearMatMul: failed to add reshaped B tensor.");
        } else {
          RETURN_IF_ERROR(qmw.AddReshapeNode(
              org_b_name, actual_b_name, shape_b, b_qnn_shape, qnn_dtype_b, quant_b, quant_b_2d,
              /*do_op_validation=*/false, qmw.IsGraphInput(org_b_name), /*is_graph_output=*/false));
        }
      }
    } else {
      if (!validate && !qmw.IsQnnTensorWrapperExist(actual_b_name)) {
        Qnn_TensorType_t tensor_type = qmw.GetTensorType(org_b_name);
        std::vector<uint8_t> unpacked;
        if (b_is_initializer) {
          RETURN_IF_ERROR(qmw.UnpackInitializerData(qmw.GetConstantTensor(org_b_name), unpacked));
        }
        QnnTensorWrapper tw(actual_b_name, tensor_type, qnn_dtype_b, quant_b.Copy(),
                            std::vector<uint32_t>(shape_b), std::move(unpacked));
        RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tw)),
                      "QLinearMatMul: failed to add B tensor.");
      }
    }
  }

  // ---- Build output quant params and output shape ----
  QnnQuantParamsWrapper quant_y;
  RETURN_IF_ERROR(BuildQuantParam(qmw,
                                  inputs[kIdxYScale],
                                  inputs.size() > kIdxYZeroPoint ? inputs[kIdxYZeroPoint]
                                                                 : OrtNodeUnitIODef{},
                                  quant_y));

  Qnn_DataType_t qnn_dtype_y = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(/*is_quantized=*/true, outputs[0].type, qnn_dtype_y));

  const std::string& org_output_name = outputs[0].name;
  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(outputs[0].shape, output_shape),
                "QLinearMatMul: cannot get output shape.");

  // Determine whether the MatMul/FC output needs a post-reshape back to the original ONNX shape.
  const bool reshape_output = a_is_rank1 || b_is_rank1 || (use_fully_connected && shape_a.size() > 2);

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
      RETURN_IF(op_output_quant.IsPerChannel(),
                "QLinearMatMul: output does not support per-channel quant.");
    } else {
      if (b_is_rank1) {
        op_output_shape.emplace_back(1);
      } else if (a_is_rank1) {
        op_output_shape.insert(op_output_shape.end() - 1, 1);
      }
      RETURN_IF_ERROR(op_output_quant.HandleUnsqueeze<uint32_t>(output_shape, op_output_shape));
    }
  }

  const bool is_graph_output = qmw.IsGraphOutput(org_output_name);
  const bool is_op_output_graph_output = is_graph_output && !reshape_output;
  const Qnn_TensorType_t op_output_type =
      is_op_output_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  // Build QNN_OP_MAT_MUL transpose params (required even at default=false for older QNN SDKs).
  std::vector<std::string> param_tensor_names;
  std::vector<QnnParamWrapper> param_wrappers;
  if (!use_fully_connected) {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 0;

    param_wrappers.emplace_back(node_unit.Index(), node_unit.Name(),
                                QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, scalar_param);
    param_tensor_names.push_back(param_wrappers.back().GetParamTensorName());

    param_wrappers.emplace_back(node_unit.Index(), node_unit.Name(),
                                QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, scalar_param);
    param_tensor_names.push_back(param_wrappers.back().GetParamTensorName());
  }

  // ---- Emit or validate ----
  if (validate) {
    QnnTensorWrapper a_handle(actual_a_name, QNN_TENSOR_TYPE_NATIVE, qnn_dtype_a,
                              quant_a.Copy(), std::vector<uint32_t>(a_qnn_shape));
    QnnTensorWrapper b_handle(actual_b_name, QNN_TENSOR_TYPE_NATIVE, qnn_dtype_b,
                              quant_b.Copy(), std::vector<uint32_t>(b_qnn_shape));
    QnnTensorWrapper output_handle(op_output_name, op_output_type, qnn_dtype_y,
                                   op_output_quant.Copy(), std::vector<uint32_t>(op_output_shape));

    if (!use_fully_connected) {
      std::vector<Qnn_Param_t> qnn_params;
      qnn_params.reserve(param_wrappers.size());
      for (const auto& pw : param_wrappers) {
        qnn_params.push_back(pw.GetQnnParam());
      }
      RETURN_IF_ERROR(qmw.ValidateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                          QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                                          {a_handle.GetQnnTensor(), b_handle.GetQnnTensor()},
                                          {output_handle.GetQnnTensor()},
                                          std::move(qnn_params)));
    } else {
      RETURN_IF_ERROR(qmw.ValidateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                          QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                          {a_handle.GetQnnTensor(), b_handle.GetQnnTensor()},
                                          {output_handle.GetQnnTensor()},
                                          {}));
    }
  } else {
    // Emit mode: tensors for A and B are already added above; add output tensor and create node.
    QnnTensorWrapper output_tw(op_output_name, op_output_type, qnn_dtype_y,
                               op_output_quant.Copy(), std::vector<uint32_t>(op_output_shape));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(output_tw)),
                  "QLinearMatMul: failed to add output tensor.");

    for (auto& pw : param_wrappers) {
      qmw.AddParamWrapper(std::move(pw));
    }

    RETURN_IF_NOT(qmw.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                    use_fully_connected ? QNN_OP_FULLY_CONNECTED : QNN_OP_MAT_MUL,
                                    {actual_a_name, actual_b_name},
                                    {op_output_name},
                                    std::move(param_tensor_names),
                                    /*do_op_validation=*/false),
                  "QLinearMatMul: failed to create QNN node.");

    if (reshape_output) {
      RETURN_IF_ERROR(qmw.AddReshapeNode(
          op_output_name, org_output_name, op_output_shape, output_shape,
          qnn_dtype_y, op_output_quant, quant_y,
          /*do_op_validation=*/false, /*is_graph_input=*/false, is_graph_output));
    }
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
