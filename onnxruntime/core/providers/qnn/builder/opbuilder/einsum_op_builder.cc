// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"

namespace {

// Represented as a tuple of 3 strings <lhs_term1, lhs_term2, rhs>.
using Equation = std::tuple<std::string, std::string, std::string>;

bool IsLowerAlpha(std::string_view term) {
  for (const char c : term) {
    auto uc = static_cast<unsigned char>(c);
    if (!std::isalpha(uc) || !std::islower(uc)) {
      return false;
    }
  }
  return true;
}

std::optional<Equation> ParseEquation(std::string_view equation_string) {
  std::string equation(equation_string);
  equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                 equation.end());
  if (equation.empty()) {
    return std::nullopt;
  }
  auto index_arrow = equation.find("->");
  if (index_arrow == std::string::npos) {
    return std::nullopt;
  }
  const std::string lhs = equation.substr(0, index_arrow);
  const std::string rhs = equation.substr(index_arrow + 2);
  if (lhs.empty() || rhs.empty()) {
    return std::nullopt;
  }
  auto index_comma = lhs.find(",");
  if (index_comma == std::string::npos) {
    return std::nullopt;
  }
  const std::string lhs_term1 = lhs.substr(0, index_comma);
  const std::string lhs_term2 = lhs.substr(index_comma + 1);
  if (lhs_term1.empty() || lhs_term2.empty()) {
    return std::nullopt;
  }
  if (lhs_term1.size() < 2) {
    return std::nullopt;
  }
  if (lhs_term1.size() != lhs_term2.size()) {
    return std::nullopt;
  }
  if (lhs_term1.size() != rhs.size()) {
    return std::nullopt;
  }
  if (!IsLowerAlpha(lhs_term1)) {
    return std::nullopt;
  }
  if (!IsLowerAlpha(lhs_term2)) {
    return std::nullopt;
  }
  if (!IsLowerAlpha(rhs)) {
    return std::nullopt;
  }
  return std::make_tuple(lhs_term1, lhs_term2, rhs);
}

bool IsEquationMatMul(const Equation& equation) {
  // MatMul: e.g., "ij,jk->ik"
  const auto& [lhs_term1, lhs_term2, rhs] = equation;
  const size_t num_dims = lhs_term1.size();
  for (size_t i = 0; i < num_dims; ++i) {
    if (i >= num_dims - 2) {
      continue;
    }
    if (!(lhs_term1[i] == lhs_term2[i] && lhs_term1[i] == rhs[i])) {
      return false;
    }
  }
  char term1_m = lhs_term1[num_dims - 2];
  char term2_k = lhs_term2[num_dims - 2];
  char result_m = rhs[num_dims - 2];
  char term1_k = lhs_term1[num_dims - 1];
  char term2_n = lhs_term2[num_dims - 1];
  char result_n = rhs[num_dims - 1];
  if (term1_m != result_m) {
    return false;
  }
  if (term1_k != term2_k) {
    return false;
  }
  if (term2_n != result_n) {
    return false;
  }
  return true;
}

bool IsEquationMatMulTransposeY(const Equation& equation) {
  // MatMul with 2nd input transposed: e.g., "id,jd->ij"
  const auto& [lhs_term1, lhs_term2, rhs] = equation;
  const size_t num_dims = lhs_term1.size();
  for (size_t i = 0; i < num_dims; ++i) {
    if (i >= num_dims - 2) {
      continue;
    }
    if (!(lhs_term1[i] == lhs_term2[i] && lhs_term1[i] == rhs[i])) {
      return false;
    }
  }
  char term1_m = lhs_term1[num_dims - 2];
  char term2_k = lhs_term2[num_dims - 2];
  char result_m = rhs[num_dims - 2];
  char term1_k = lhs_term1[num_dims - 1];
  char term2_n = lhs_term2[num_dims - 1];
  char result_n = rhs[num_dims - 1];
  if (term1_m != result_m) {
    return false;
  }
  if (term1_k != term2_n) {
    return false;
  }
  if (term2_k != result_n) {
    return false;
  }
  return true;
}

bool IsEquationMatMulTransposeAll(const Equation& equation) {
  // MatMul transpose both inputs and output, e.g., "bchq,bkhc->bkhq", "bkhq,bchk->bchq"
  const auto& [lhs_term1, lhs_term2, rhs] = equation;
  const size_t num_dims = lhs_term1.size();
  if (num_dims != 4) {
    return false;
  }
  if (lhs_term1[0] != lhs_term2[0] || lhs_term1[0] != rhs[0]) {
    return false;
  }
  char term1_m = lhs_term1[num_dims - 1];
  char term1_k = lhs_term1[num_dims - 3];
  char term2_k = lhs_term2[num_dims - 1];
  char term2_n = lhs_term2[num_dims - 3];
  char result_m = rhs[num_dims - 1];
  char result_n = rhs[num_dims - 3];
  if (term1_m != result_m) {
    return false;
  }
  if (term1_k != term2_k) {
    return false;
  }
  if (term2_n != result_n) {
    return false;
  }
  return true;
}

}  // namespace

namespace onnxruntime {
namespace qnn {

class EinsumOpBuilder : public BaseOpBuilder {
 public:
  EinsumOpBuilder() : BaseOpBuilder("EinsumOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(EinsumOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                  const NodeUnit& node_unit,
                                  const logging::Logger& logger,
                                  const std::vector<std::string>& input_names,
                                  size_t output_index,
                                  Qnn_DataType_t qnn_data_type,
                                  QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;
};

Status EinsumOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger) const {
  if (node_unit.Inputs().size() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_unit.OpType() + " requires at least 2 inputs.");
  }
  NodeAttrHelper node_helper{node_unit};
  const std::string equation = node_helper.Get("equation", std::string(""));
  std::optional<Equation> parsed_equation = ParseEquation(equation);
  if (!parsed_equation.has_value()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_unit.OpType() + " unsupported equation: " + equation);
  }
  if (!IsEquationMatMul(parsed_equation.value()) &&
      !IsEquationMatMulTransposeY(parsed_equation.value()) &&
      !IsEquationMatMulTransposeAll(parsed_equation.value())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_unit.OpType() + " unsupported equation: " + equation);
  }
  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Status EinsumOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[1], logger, input_names));
  return Status::OK();
}

Status EinsumOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  const std::string equation = node_helper.Get("equation", std::string(""));
  std::optional<Equation> parsed_equation = ParseEquation(equation);
  if (IsEquationMatMul(parsed_equation.value())) {
    ORT_RETURN_IF_ERROR(ProcessOutputs(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                       /*node_unit=*/node_unit,
                                       /*input_names=*/std::move(input_names),
                                       /*param_tensor_names=*/{},
                                       /*logger=*/logger,
                                       /*do_op_validation=*/do_op_validation,
                                       /*qnn_op_type=*/QNN_OP_MAT_MUL));
  } else if (IsEquationMatMulTransposeY(parsed_equation.value())) {
    std::vector<std::string> param_tensor_names;
    Qnn_Scalar_t scalar_param0 = QNN_SCALAR_INIT;
    Qnn_Scalar_t scalar_param1 = QNN_SCALAR_INIT;
    scalar_param0.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param1.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param0.bool8Value = 0;
    scalar_param1.bool8Value = 1;
    QnnParamWrapper transpose_in0_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,
                                        scalar_param0);
    QnnParamWrapper transpose_in1_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                                        scalar_param1);
    param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
    param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in0_param));
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in1_param));
    ORT_RETURN_IF_ERROR(ProcessOutputs(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                       /*node_unit=*/node_unit,
                                       /*input_names=*/std::move(input_names),
                                       /*param_tensor_names=*/std::move(param_tensor_names),
                                       /*logger=*/logger,
                                       /*do_op_validation=*/do_op_validation,
                                       /*qnn_op_type=*/QNN_OP_MAT_MUL));
  } else if (IsEquationMatMulTransposeAll(parsed_equation.value())) {
    TensorInfo input_info0{}, input_info1{};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info0));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], input_info1));
    std::vector<uint32_t> input_shape0(input_info0.shape);
    std::vector<uint32_t> input_shape1(input_info1.shape);
    std::swap(input_shape0[1], input_shape0[2]);
    std::swap(input_shape1[1], input_shape1[2]);
    const std::string input_transpos0 = input_names[0] + "_t0";
    const std::string input_transpos1 = input_names[1] + "_t1";
    const std::vector<uint32_t> transpose_perm{0, 2, 1, 3};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        /*node_index=*/node_unit.Index(),
        /*input_name=*/input_names[0],
        /*output_name=*/input_transpos0,
        /*input_shape=*/input_info0.shape,
        /*transpose_perm=*/transpose_perm,
        /*output_shape=*/input_shape0,
        /*qnn_data_type=*/input_info0.qnn_data_type,
        /*quantize_param=*/input_info0.quant_param.Copy(),
        /*do_op_validation=*/do_op_validation,
        /*is_for_input=*/qnn_model_wrapper.IsGraphInput(input_names[0])));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        /*node_index=*/node_unit.Index(),
        /*input_name=*/input_names[1],
        /*output_name=*/input_transpos1,
        /*input_shape=*/input_info1.shape,
        /*transpose_perm=*/transpose_perm,
        /*output_shape=*/input_shape1,
        /*qnn_data_type=*/input_info1.qnn_data_type,
        /*quantize_param=*/input_info1.quant_param.Copy(),
        /*do_op_validation=*/do_op_validation,
        /*is_for_input=*/qnn_model_wrapper.IsGraphInput(input_names[1])));
    TensorInfo matmul_output_info{};
    const auto& output = node_unit.Outputs()[0];
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, matmul_output_info));
    const std::string matmul_output_name = utils::GetNodeName(node_unit) + "_matmul";
    std::vector<uint32_t> matmul_output_shape(matmul_output_info.shape);
    std::swap(matmul_output_shape[1], matmul_output_shape[2]);
    QnnTensorWrapper matmul_output_wrapper(matmul_output_name, QNN_TENSOR_TYPE_NATIVE, matmul_output_info.qnn_data_type,
                                           matmul_output_info.quant_param.Copy(), std::vector<uint32_t>(matmul_output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(matmul_output_wrapper)), node_unit.OpType() + " failed to add tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(/*qnn_node_name=*/utils::GetNodeName(node_unit),
                                                      /*package_name=*/QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      /*qnn_node_type=*/QNN_OP_MAT_MUL,
                                                      /*input_names=*/{input_transpos1, input_transpos0},
                                                      /*output_names=*/{matmul_output_name},
                                                      /*param_tensor_names=*/{},
                                                      /*do_op_validation=*/do_op_validation),
                      node_unit.OpType() + " failed to add node.");
    std::vector<uint32_t> transpose_output_shape(matmul_output_info.shape);
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        /*node_index=*/node_unit.Index(),
        /*input_name=*/matmul_output_name,
        /*output_name=*/output.node_arg.Name(),
        /*input_shape=*/std::move(matmul_output_shape),
        /*transpose_perm=*/transpose_perm,
        /*output_shape=*/matmul_output_info.shape,
        /*tensor_data_type=*/matmul_output_info.qnn_data_type,
        /*quantize_param=*/matmul_output_info.quant_param.Copy(),
        /*do_op_validation=*/do_op_validation,
        /*is_for_input=*/qnn_model_wrapper.IsGraphInput(output.node_arg.Name()),
        /*is_for_output=*/qnn_model_wrapper.IsGraphOutput(output.node_arg.Name())));
  }
  return Status::OK();
}

Status EinsumOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 const logging::Logger& logger,
                                                 const std::vector<std::string>& input_names,
                                                 size_t output_index,
                                                 Qnn_DataType_t qnn_data_type,
                                                 QnnQuantParamsWrapper& quant_param) const {
  if (!quant_param.IsPerTensor()) {
    return Status::OK();
  }

  // Force the Tile operator output to use the same quantization parameters as the input if nearly equal.
  // This helps the HTP backend employ certain optimizations.
  return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                  0 /*input_index*/, output_index, qnn_data_type, quant_param);
}

void CreateEinsumOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<EinsumOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
