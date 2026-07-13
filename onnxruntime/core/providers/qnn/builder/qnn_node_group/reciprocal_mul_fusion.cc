// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// ReciprocalMulFusion: Fuses SingleNode Reciprocal->Mul into ElementWiseBinary (DIVIDE).
// QDQGroup pattern avoided to preserve separate quantization of 1/b.

#include "core/providers/qnn/builder/qnn_node_group/reciprocal_mul_fusion.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <gsl/gsl>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Convenience macros for validation and creation paths.
#define ValidateOnQnn(qnn_model_wrapper, reciprocal_node_unit, mul_node_unit, recip_is_mul_input0) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (reciprocal_node_unit), (mul_node_unit), (recip_is_mul_input0), /*validate=*/true)
#define CreateOnQnn(qnn_model_wrapper, reciprocal_node_unit, mul_node_unit, recip_is_mul_input0) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (reciprocal_node_unit), (mul_node_unit), (recip_is_mul_input0), /*validate=*/false)

// Forward declaration.
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& reciprocal_node_unit,
                                         const OrtNodeUnit& mul_node_unit,
                                         bool recip_is_mul_input0,
                                         bool validate);

// TryFusion: Matches Reciprocal->Mul pattern and validates fusion.
std::unique_ptr<IQnnNodeGroup> ReciprocalMulFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reciprocal_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Step 1: Check op-type and unit type.
  // Only accept SingleNode Reciprocal to preserve separate quantization of 1/b.
  if (reciprocal_node_unit.OpType() != "Reciprocal" ||
      reciprocal_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Step 2: Locate single Mul consumer (handles QDQ boundaries).
  const OrtNodeUnit* mul_node_unit =
      GetChildNodeUnitAllowQdq(qnn_model_wrapper, reciprocal_node_unit, "Mul",
                               node_to_node_unit, node_unit_to_qnn_node_group);
  if (mul_node_unit == nullptr) {
    return nullptr;
  }

  // Step 3: Determine which Mul input carries the Reciprocal output.
  const auto& mul_inputs = mul_node_unit->Inputs();
  const std::string& recip_output_name = reciprocal_node_unit.Outputs()[0].name;
  bool recip_is_mul_input0 = (mul_inputs[0].name == recip_output_name);
  bool recip_is_mul_input1 = (mul_inputs[1].name == recip_output_name);

  if (!recip_is_mul_input0 && !recip_is_mul_input1) {
    return nullptr;
  }

  if (recip_is_mul_input0 && recip_is_mul_input1) {
    return nullptr;  // Both inputs same: would change semantics.
  }

  // Step 4: QNN capability validation (dry-run).
  if (Ort::Status status = ValidateOnQnn(qnn_model_wrapper, reciprocal_node_unit, *mul_node_unit, recip_is_mul_input0);
      !status.IsOK()) {
    return nullptr;
  }

  // Step 5: Construct fusion object.
  return std::make_unique<ReciprocalMulFusion>(reciprocal_node_unit, *mul_node_unit, recip_is_mul_input0);
}

ReciprocalMulFusion::ReciprocalMulFusion(const OrtNodeUnit& reciprocal_node_unit,
                                         const OrtNodeUnit& mul_node_unit,
                                         bool recip_is_mul_input0)
    : node_units_{&reciprocal_node_unit, &mul_node_unit},
      recip_is_mul_input0_{recip_is_mul_input0} {
}

Ort::Status ReciprocalMulFusion::IsSupported(QnnModelWrapper& qmw,
                                             const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1], recip_is_mul_input0_);
}

Ort::Status ReciprocalMulFusion::AddToModelBuilder(QnnModelWrapper& qmw,
                                                   const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1], recip_is_mul_input0_);
}

gsl::span<const OrtNodeUnit* const> ReciprocalMulFusion::GetNodeUnits() const {
  return node_units_;
}

const OrtNodeUnit* ReciprocalMulFusion::GetTargetNodeUnit() const {
  return node_units_[1];  // Mul is the convergence point.
}

// CreateOrValidateOnQnn: Shared validate/build path.
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& reciprocal_node_unit,
                                         const OrtNodeUnit& mul_node_unit,
                                         bool recip_is_mul_input0,
                                         bool validate) {
  RETURN_IF_NOT(reciprocal_node_unit.OpType() == "Reciprocal",
                ("ReciprocalMulFusion: expected Reciprocal op, got " + reciprocal_node_unit.OpType()).c_str());
  RETURN_IF_NOT(mul_node_unit.OpType() == "Mul",
                ("ReciprocalMulFusion: expected Mul op, got " + mul_node_unit.OpType()).c_str());

  // Resolve tensor roles.
  const OrtNodeUnitIODef& denominator_def = reciprocal_node_unit.Inputs()[0];
  const auto& mul_inputs = mul_node_unit.Inputs();
  const OrtNodeUnitIODef& numerator_def = recip_is_mul_input0 ? mul_inputs[1] : mul_inputs[0];
  const OrtNodeUnitIODef& output_def = mul_node_unit.Outputs()[0];
  const std::string node_name = utils::UniqueNameGenerator().New(reciprocal_node_unit);

  if (validate) {
    // Dry-run: capability query only.
    QnnTensorWrapper numerator_tensor;
    QnnTensorWrapper denominator_tensor;
    QnnTensorWrapper output_tensor;

    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(numerator_def, numerator_tensor));
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(denominator_def, denominator_tensor));
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

    // Create operation parameter for DIVIDE
    Qnn_Scalar_t div_op_scalar = QNN_SCALAR_INIT;
    div_op_scalar.dataType = QNN_DATATYPE_UINT_32;
    div_op_scalar.uint32Value = QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE;
    QnnParamWrapper div_op_param(reciprocal_node_unit.Index(), node_name,
                                 QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, div_op_scalar);

    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(
        node_name,
        QNN_OP_PACKAGE_NAME_QTI_AISW,
        QNN_OP_ELEMENT_WISE_BINARY,
        /*input_tensors=*/{numerator_tensor.GetQnnTensor(), denominator_tensor.GetQnnTensor()},
        /*output_tensors=*/{output_tensor.GetQnnTensor()},
        /*params=*/{div_op_param.GetQnnParam()}));
  } else {
    // Build path: register tensors and create QNN node.

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(numerator_def.name)) {
      QnnTensorWrapper numerator_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(numerator_def, numerator_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(numerator_tensor)),
                    "ReciprocalMulFusion: failed to add numerator tensor wrapper.");
    }

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(denominator_def.name)) {
      QnnTensorWrapper denominator_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(denominator_def, denominator_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(denominator_tensor)),
                    "ReciprocalMulFusion: failed to add denominator tensor wrapper.");
    }

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(output_def.name)) {
      QnnTensorWrapper output_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)),
                    "ReciprocalMulFusion: failed to add output tensor wrapper.");
    }

    // Add operation parameter for DIVIDE
    std::vector<std::string> div_param_names;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, reciprocal_node_unit.Index(), node_name,
                                           static_cast<uint32_t>(QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE),
                                           QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, div_param_names));

    // Create fused ElementWiseBinary node with DIVIDE operation.
    RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(
            node_name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            QNN_OP_ELEMENT_WISE_BINARY,
            /*input_names=*/{numerator_def.name, denominator_def.name},
            /*output_names=*/{output_def.name},
            /*param_tensor_names=*/std::move(div_param_names),
            /*do_op_validation=*/validate),
        "ReciprocalMulFusion: failed to create fused ElementWiseBinary node.");
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
