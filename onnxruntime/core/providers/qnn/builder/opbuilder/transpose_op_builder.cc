// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/qdq_constant_folding.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class TransposeOpBuilder : public BaseOpBuilder {
 public:
  TransposeOpBuilder() : BaseOpBuilder("TransposeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TransposeOpBuilder);

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                                   const OrtNodeUnit& node_unit,
                                   std::vector<std::string>& param_tensor_names) const;
};

static Ort::Status GetTransposePerm(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    /*out*/ std::vector<uint32_t>& perm) {
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape), "Cannot get shape");
  // set default perm
  uint32_t rank = static_cast<uint32_t>(input_shape.size());
  std::vector<int64_t> transpose_perm(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    transpose_perm[i] = rank - 1 - i;
  }

  OrtNodeAttrHelper node_helper(node_unit);
  transpose_perm = node_helper.Get("perm", transpose_perm);
  perm.resize(transpose_perm.size());
  std::transform(transpose_perm.begin(), transpose_perm.end(), perm.begin(),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });
  return Ort::Status();
}

// Only inputs produced by an earlier Q/DQ fold are handled here; ORT's transpose optimizer already
// folds Transposes of real initializers. Such inputs arise when a constant sits behind a Q/DQ chain
// the optimizer cannot push through, and keeping the Transpose at runtime makes its consumer read
// an activation instead of a static tensor.
static Ort::Status TryFoldConstantTranspose(QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& node_unit,
                                            const std::string& input_name) {
  const std::string& output_name = node_unit.Outputs()[0].name;
  // An earlier fusion may already have registered the output as a NATIVE tensor that expects a producer.
  RETURN_IF(qnn_model_wrapper.IsQnnTensorWrapperExist(output_name), "Transpose output is already registered.");

  const QnnTensorWrapper& input_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_name);
  const Qnn_DataType_t data_type = input_wrapper.GetTensorDataType();
  RETURN_IF(data_type == QNN_DATATYPE_SFIXED_POINT_4 || data_type == QNN_DATATYPE_UFIXED_POINT_4,
            "Unsupported folded Transpose data type.");
  const size_t elem_size = utils::GetElementSizeByType(data_type);

  std::vector<uint32_t> perm;
  RETURN_IF_ERROR(GetTransposePerm(qnn_model_wrapper, node_unit, perm));
  const std::vector<uint32_t>& input_shape = input_wrapper.GetTensorDims();
  const size_t rank = input_shape.size();
  RETURN_IF_NOT(rank > 0 && perm.size() == rank, "Transpose perm rank mismatch.");
  std::vector<uint32_t> perm_inv(rank);
  RETURN_IF_ERROR(utils::InvertPerm<uint32_t>(perm, perm_inv));
  for (size_t d = 0; d < rank; ++d) {
    RETURN_IF(perm[perm_inv[d]] != d, "Transpose perm is not a permutation.");
  }

  std::vector<uint32_t> output_shape(rank);
  RETURN_IF_ERROR((utils::PermuteShape<uint32_t, uint32_t>(input_shape, perm, output_shape)));

  std::vector<size_t> input_strides(rank, 1);
  size_t num_elems = input_shape[rank - 1];
  for (size_t i = rank - 1; i-- > 0;) {
    input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    num_elems *= input_shape[i];
  }

  std::vector<uint8_t> input_bytes;
  RETURN_IF_ERROR(GetEffectivelyConstantTensorBytes(qnn_model_wrapper, input_name, input_bytes));
  RETURN_IF(input_bytes.size() != SafeInt<size_t>(num_elems) * elem_size,
            "Folded Transpose input byte size mismatch with shape.");

  std::vector<uint8_t> output_bytes(input_bytes.size());
  std::vector<uint32_t> index(rank, 0);
  for (size_t out_elem = 0; out_elem < num_elems; ++out_elem) {
    size_t in_elem = 0;
    for (size_t d = 0; d < rank; ++d) {
      in_elem += index[d] * input_strides[perm[d]];
    }
    std::memcpy(output_bytes.data() + out_elem * elem_size, input_bytes.data() + in_elem * elem_size, elem_size);
    for (size_t d = rank; d-- > 0;) {
      if (++index[d] < output_shape[d]) {
        break;
      }
      index[d] = 0;
    }
  }

  QnnQuantParamsWrapper quant_param = input_wrapper.GetQnnQuantParams().Copy();
  RETURN_IF_ERROR(quant_param.HandleTranspose<uint32_t>(perm_inv));

  QnnTensorWrapper output_wrapper(output_name, QNN_TENSOR_TYPE_STATIC, data_type, std::move(quant_param),
                                  std::move(output_shape), std::move(output_bytes));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_wrapper)),
                "Failed to add folded Transpose output tensor.");
  qnn_model_wrapper.MarkTensorAsFoldedConstant(output_name);
  return Ort::Status();
}

Ort::Status TransposeOpBuilder::ProcessPermAttribute(QnnModelWrapper& qnn_model_wrapper,
                                                     const OrtNodeUnit& node_unit,
                                                     std::vector<std::string>& param_tensor_names) const {
  std::vector<uint32_t> perm_data;
  RETURN_IF_ERROR(GetTransposePerm(qnn_model_wrapper, node_unit, perm_data));
  std::vector<uint32_t> perm_shape{static_cast<uint32_t>(perm_data.size())};

  QnnParamWrapper transpose_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                                  std::move(perm_shape), std::move(perm_data));
  param_tensor_names.push_back(transpose_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(transpose_param));

  return Ort::Status();
}

Ort::Status TransposeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            std::vector<std::string>&& input_names,
                                                            const Ort::Logger& logger,
                                                            bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Ort::Status();
  }

  if (qnn_model_wrapper.IsFoldedConstant(input_names[0]) &&
      !qnn_model_wrapper.IsGraphOutput(node_unit.Outputs()[0].name)) {
    Ort::Status fold_status = TryFoldConstantTranspose(qnn_model_wrapper, node_unit, input_names[0]);
    if (fold_status.IsOK()) {
      return Ort::Status();
    }
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("QNN EP declined constant folding for node '" + node_unit.Name() +
                 "': " + fold_status.GetErrorMessage())
                    .c_str());
  }

  std::vector<std::string> param_tensor_names;
  RETURN_IF_ERROR(ProcessPermAttribute(qnn_model_wrapper, node_unit, param_tensor_names));

  const auto& output_name = node_unit.Outputs()[0].name;
  std::vector<std::string> output_names;

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  struct CastNodeInfo {
    std::string node_name;
    std::string input_name;
    std::string output_name;
  };
  std::vector<CastNodeInfo> cast_node_info_vec;

  // Check if we need to add a cast node for int64
  bool needs_int64_cast = false;
  if (is_graph_output) {
    for (const auto& input_name : input_names) {
      if (input_name.find("_cast_int32") != std::string::npos) {
        needs_int64_cast = true;
        break;
      }
    }
  }

  const auto& transpose_output = node_unit.Outputs()[0];
  // Get the output info for the gather output tensor
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(transpose_output, output_info));
  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Outputs()[0].shape, output_shape), "Cannot get shape");

  const QnnTensorWrapper& input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);

  // If a cast to int64 is needed, add the cast node
  if (needs_int64_cast) {
    std::string cast_node_name = utils::UniqueNameGenerator().New(node_unit, "_cast_int64");
    std::string cast_input_name = utils::UniqueNameGenerator().New(output_name, "_cast_int64");
    std::string cast_output_name = output_name;

    // Create the cast input tensor wrapper
    QnnTensorWrapper cast_input_tensorwrapper(cast_input_name,
                                              QNN_TENSOR_TYPE_NATIVE,
                                              output_info.qnn_data_type,
                                              output_info.quant_param.Copy(),
                                              std::move(output_shape));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_input_tensorwrapper)), "Failed to add tensor.");
    cast_node_info_vec.emplace_back(CastNodeInfo{cast_node_name, cast_input_name, cast_output_name});
  }

  // Transpose output uses same data type and quantization parameter with input
  // 1. In QDQ model, the optimization may create scenario like Q -> Transpose -> DQ, Transpose is single node
  // Input tensor is created by previous node which is quantized tensor,
  // so output just copy the same data type and quantization parameters
  // 2. In QDQ model, Transpose also support non-quantized data like int32.
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        input_tensor_wrapper.GetTensorDataType(),
                                        input_tensor_wrapper.GetQnnQuantParams().Copy(),
                                        std::move(output_shape));

  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");

  output_names.push_back(output_name);
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_TRANSPOSE,
                                                std::move(input_names),
                                                std::move(output_names),
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to add node.");

  if (needs_int64_cast) {
    for (const auto& cast_node_info : cast_node_info_vec) {
      // Insert cast node.
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_info.node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CAST,
                                                    {cast_node_info.input_name},
                                                    {cast_node_info.output_name},
                                                    {}),
                    " Failed to add Cast node");
    }
  }
  return Ort::Status();
}

void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<TransposeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
