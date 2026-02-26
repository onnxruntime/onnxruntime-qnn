// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class MultiHeadAttentionOpBuilder : public BaseOpBuilder {
 public:
  MultiHeadAttentionOpBuilder() : BaseOpBuilder("MultiHeadAttentionOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MultiHeadAttentionOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  // Helper to create intermediate tensor
  Ort::Status CreateIntermediateTensor(QnnModelWrapper& qnn_model_wrapper,
                                       const std::string& tensor_name,
                                       Qnn_DataType_t data_type,
                                       const std::vector<uint32_t>& shape,
                                       const QnnQuantParamsWrapper& quant_params) const;

  // Helper to unpack QKV from packed format
  Ort::Status UnpackQKV(QnnModelWrapper& qnn_model_wrapper,
                        const OrtNodeUnit& node_unit,
                        const std::string& packed_input,
                        const std::vector<uint32_t>& packed_shape,
                        const TensorInfo& input_info,
                        std::string& q_output,
                        std::string& k_output,
                        std::string& v_output,
                        bool do_op_validation) const;

  // Helper to unpack KV from packed format
  Ort::Status UnpackKV(QnnModelWrapper& qnn_model_wrapper,
                       const OrtNodeUnit& node_unit,
                       const std::string& packed_input,
                       const std::vector<uint32_t>& packed_shape,
                       const TensorInfo& input_info,
                       std::string& k_output,
                       std::string& v_output,
                       bool do_op_validation) const;
};

Ort::Status MultiHeadAttentionOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // MultiHeadAttention requires at least query input
  RETURN_IF(inputs.size() < 1, "MultiHeadAttention requires at least query input");

  // Check that we have at least one output
  RETURN_IF(outputs.size() < 1, "MultiHeadAttention requires at least one output");

  // Get num_heads attribute
  OrtNodeAttrHelper node_helper(node_unit);
  int64_t num_heads = node_helper.Get("num_heads", static_cast<int64_t>(0));
  RETURN_IF(num_heads <= 0, "num_heads must be positive");

  // Check input shapes
  std::vector<uint32_t> query_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, query_shape),
                "Cannot get shape of query input");

  RETURN_IF(query_shape.size() != 3 && query_shape.size() != 5,
            "MultiHeadAttention query input must be 3D or 5D tensor");

  // Determine input format is packed_qkv, packed_kv, or separated_qkv
  bool is_packed_qkv = (query_shape.size() == 5);
  bool is_packed_kv = false;
  bool is_seperated_qkv = false;

  if (!is_packed_qkv && inputs.size() >= 2 && inputs[1].Exists()) {
    std::vector<uint32_t> key_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].shape, key_shape),
                  "Cannot get shape of key input");
    is_packed_kv = (key_shape.size() == 5);
    is_seperated_qkv = !is_packed_kv;
  }

  // Validate input size accroding to imput format
  RETURN_IF(is_packed_qkv && inputs.size() > 1 && inputs[1].Exists(),
            "MultiHeadAttention only supports Packed QKV format with one input");
  RETURN_IF(is_packed_kv && inputs.size() > 2 && inputs[2].Exists(),
            "MultiHeadAttention only supports Packed KV format with two inputs");
  RETURN_IF(is_seperated_qkv && inputs.size() > 3 && inputs[3].Exists(),
            "MultiHeadAttention only supports Seperated QKV format with three inputs");

  // Don't support additional inputs for now
  RETURN_IF(inputs.size() > 3 && inputs[3].Exists(),
            "MultiHeadAttention bias input is not supported in QNN provider yet");

  // Don't support multiple outputs
  RETURN_IF(outputs.size() > 1,
            "MultiHeadAttention with multiple outputs is not supported in QNN provider yet");

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Ort::Status MultiHeadAttentionOpBuilder::CreateIntermediateTensor(
    QnnModelWrapper& qnn_model_wrapper,
    const std::string& tensor_name,
    Qnn_DataType_t data_type,
    const std::vector<uint32_t>& shape,
    const QnnQuantParamsWrapper& quant_params) const {
  QnnTensorWrapper tensor_wrapper(tensor_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  data_type,
                                  quant_params.Copy(),
                                  std::vector<uint32_t>(shape));

  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper)),
                "Failed to add intermediate tensor");

  return Ort::Status();
}

Ort::Status MultiHeadAttentionOpBuilder::UnpackQKV(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    const std::string& packed_input,
    const std::vector<uint32_t>& packed_shape,
    const TensorInfo& input_info,
    std::string& q_output,
    std::string& k_output,
    std::string& v_output,
    bool do_op_validation) const {
  // Packed QKV shape: (batch_size, sequence_length, num_heads, 3, head_size)
  // We need to split along dimension 3 to get Q, K, V

  const uint32_t batch_size = packed_shape[0];
  const uint32_t seq_len = packed_shape[1];
  const uint32_t num_heads = packed_shape[2];
  const uint32_t head_size = packed_shape[4];

  std::vector<uint32_t> qkv_shape = {batch_size, seq_len, num_heads, head_size};

  // Create output tensors for Q, K, V
  q_output = utils::GetUniqueName(node_unit, "_q_unpacked");
  k_output = utils::GetUniqueName(node_unit, "_k_unpacked");
  v_output = utils::GetUniqueName(node_unit, "_v_unpacked");

  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, q_output,
                                           input_info.qnn_data_type, qkv_shape,
                                           input_info.quant_param));
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, k_output,
                                           input_info.qnn_data_type, qkv_shape,
                                           input_info.quant_param));
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, v_output,
                                           input_info.qnn_data_type, qkv_shape,
                                           input_info.quant_param));

  // Use StridedSlice to extract Q, K, V
  // QNN uses RANGES parameter: 2D tensor [rank, 3] where each row is [start, end, step]
  // For Q: slice [:, :, :, 0:1, :]
  std::vector<uint32_t> q_ranges_data;
  q_ranges_data.reserve(15);  // 5 dimensions * 3 values

  // Dimension 0: batch_size (0:batch_size:1)
  q_ranges_data.push_back(0);
  q_ranges_data.push_back(batch_size);
  q_ranges_data.push_back(1);

  // Dimension 1: seq_len (0:seq_len:1)
  q_ranges_data.push_back(0);
  q_ranges_data.push_back(seq_len);
  q_ranges_data.push_back(1);

  // Dimension 2: num_heads (0:num_heads:1)
  q_ranges_data.push_back(0);
  q_ranges_data.push_back(num_heads);
  q_ranges_data.push_back(1);

  // Dimension 3: slice index 0 (0:1:1) - extract Q
  q_ranges_data.push_back(0);
  q_ranges_data.push_back(1);
  q_ranges_data.push_back(1);

  // Dimension 4: head_size (0:head_size:1)
  q_ranges_data.push_back(0);
  q_ranges_data.push_back(head_size);
  q_ranges_data.push_back(1);

  // Create Q slice
  std::vector<std::string> q_param_names;
  std::vector<uint32_t> ranges_dims{5, 3};  // [rank, 3]

  QnnParamWrapper q_ranges_param(node_unit.Index(), node_unit.Name() + "_q_ranges",
                                 QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                 std::move(ranges_dims),
                                 std::move(q_ranges_data),
                                 true);
  q_param_names.push_back(q_ranges_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(q_ranges_param));

  std::string q_slice_output = utils::GetUniqueName(node_unit, "_q_slice");
  std::vector<uint32_t> q_slice_shape = {batch_size, seq_len, num_heads, 1, head_size};
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, q_slice_output,
                                           input_info.qnn_data_type, q_slice_shape,
                                           input_info.quant_param));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::GetUniqueName(node_unit, "_q_slice"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_STRIDED_SLICE,
                    {packed_input},
                    {q_slice_output},
                    std::move(q_param_names),
                    do_op_validation),
                "Failed to create Q slice node");

  // Reshape Q to remove dimension 3
  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(q_slice_output, q_output,
                                                   q_slice_shape, qkv_shape,
                                                   input_info.qnn_data_type,
                                                   input_info.quant_param,
                                                   do_op_validation,
                                                   false, false));

  // Similar for K and V...
  // For K: slice [:, :, :, 1:2, :]
  std::vector<uint32_t> k_ranges_data;
  k_ranges_data.reserve(15);

  k_ranges_data.push_back(0);
  k_ranges_data.push_back(batch_size);
  k_ranges_data.push_back(1);

  k_ranges_data.push_back(0);
  k_ranges_data.push_back(seq_len);
  k_ranges_data.push_back(1);

  k_ranges_data.push_back(0);
  k_ranges_data.push_back(num_heads);
  k_ranges_data.push_back(1);

  // Dimension 3: slice index 1 (1:2:1) - extract K
  k_ranges_data.push_back(1);
  k_ranges_data.push_back(2);
  k_ranges_data.push_back(1);

  k_ranges_data.push_back(0);
  k_ranges_data.push_back(head_size);
  k_ranges_data.push_back(1);

  std::vector<std::string> k_param_names;
  std::vector<uint32_t> k_ranges_dims{5, 3};

  QnnParamWrapper k_ranges_param(node_unit.Index(), node_unit.Name() + "_k_ranges",
                                 QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                 std::move(k_ranges_dims),
                                 std::move(k_ranges_data),
                                 true);
  k_param_names.push_back(k_ranges_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(k_ranges_param));

  std::string k_slice_output = utils::GetUniqueName(node_unit, "_k_slice");
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, k_slice_output,
                                           input_info.qnn_data_type, q_slice_shape,
                                           input_info.quant_param));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::GetUniqueName(node_unit, "_k_slice"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_STRIDED_SLICE,
                    {packed_input},
                    {k_slice_output},
                    std::move(k_param_names),
                    do_op_validation),
                "Failed to create K slice node");

  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(k_slice_output, k_output,
                                                   q_slice_shape, qkv_shape,
                                                   input_info.qnn_data_type,
                                                   input_info.quant_param,
                                                   do_op_validation,
                                                   false, false));

  // For V: slice [:, :, :, 2:3, :]
  std::vector<uint32_t> v_ranges_data;
  v_ranges_data.reserve(15);

  v_ranges_data.push_back(0);
  v_ranges_data.push_back(batch_size);
  v_ranges_data.push_back(1);

  v_ranges_data.push_back(0);
  v_ranges_data.push_back(seq_len);
  v_ranges_data.push_back(1);

  v_ranges_data.push_back(0);
  v_ranges_data.push_back(num_heads);
  v_ranges_data.push_back(1);

  // Dimension 3: slice index 2 (2:3:1) - extract V
  v_ranges_data.push_back(2);
  v_ranges_data.push_back(3);
  v_ranges_data.push_back(1);

  v_ranges_data.push_back(0);
  v_ranges_data.push_back(head_size);
  v_ranges_data.push_back(1);

  std::vector<std::string> v_param_names;
  std::vector<uint32_t> v_ranges_dims{5, 3};

  QnnParamWrapper v_ranges_param(node_unit.Index(), node_unit.Name() + "_v_ranges",
                                 QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                 std::move(v_ranges_dims),
                                 std::move(v_ranges_data),
                                 true);
  v_param_names.push_back(v_ranges_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(v_ranges_param));

  std::string v_slice_output = utils::GetUniqueName(node_unit, "_v_slice");
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, v_slice_output,
                                           input_info.qnn_data_type, q_slice_shape,
                                           input_info.quant_param));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::GetUniqueName(node_unit, "_v_slice"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_STRIDED_SLICE,
                    {packed_input},
                    {v_slice_output},
                    std::move(v_param_names),
                    do_op_validation),
                "Failed to create V slice node");

  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(v_slice_output, v_output,
                                                   q_slice_shape, qkv_shape,
                                                   input_info.qnn_data_type,
                                                   input_info.quant_param,
                                                   do_op_validation,
                                                   false, false));

  return Ort::Status();
}

Ort::Status MultiHeadAttentionOpBuilder::UnpackKV(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    const std::string& packed_input,
    const std::vector<uint32_t>& packed_shape,
    const TensorInfo& input_info,
    std::string& k_output,
    std::string& v_output,
    bool do_op_validation) const {
  // Packed KV shape: (batch_size, kv_sequence_length, num_heads, 2, head_size)
  // Similar to UnpackQKV but only for K and V

  const uint32_t batch_size = packed_shape[0];
  const uint32_t seq_len = packed_shape[1];
  const uint32_t num_heads = packed_shape[2];
  const uint32_t head_size = packed_shape[4];

  std::vector<uint32_t> kv_shape = {batch_size, seq_len, num_heads, head_size};

  k_output = utils::GetUniqueName(node_unit, "_k_unpacked");
  v_output = utils::GetUniqueName(node_unit, "_v_unpacked");

  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, k_output,
                                           input_info.qnn_data_type, kv_shape,
                                           input_info.quant_param));
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, v_output,
                                           input_info.qnn_data_type, kv_shape,
                                           input_info.quant_param));

  // Extract K: slice [:, :, :, 0:1, :]
  std::vector<uint32_t> k_ranges_data;
  k_ranges_data.reserve(15);

  k_ranges_data.push_back(0);
  k_ranges_data.push_back(batch_size);
  k_ranges_data.push_back(1);

  k_ranges_data.push_back(0);
  k_ranges_data.push_back(seq_len);
  k_ranges_data.push_back(1);

  k_ranges_data.push_back(0);
  k_ranges_data.push_back(num_heads);
  k_ranges_data.push_back(1);

  // Dimension 3: slice index 0 (0:1:1) - extract K
  k_ranges_data.push_back(0);
  k_ranges_data.push_back(1);
  k_ranges_data.push_back(1);

  k_ranges_data.push_back(0);
  k_ranges_data.push_back(head_size);
  k_ranges_data.push_back(1);

  std::vector<std::string> k_param_names;
  std::vector<uint32_t> k_ranges_dims{5, 3};

  QnnParamWrapper k_ranges_param(node_unit.Index(), node_unit.Name() + "_k_ranges",
                                 QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                 std::move(k_ranges_dims),
                                 std::move(k_ranges_data),
                                 true);
  k_param_names.push_back(k_ranges_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(k_ranges_param));

  std::string k_slice_output = utils::GetUniqueName(node_unit, "_k_slice");
  std::vector<uint32_t> slice_shape = {batch_size, seq_len, num_heads, 1, head_size};
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, k_slice_output,
                                           input_info.qnn_data_type, slice_shape,
                                           input_info.quant_param));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::GetUniqueName(node_unit, "_k_slice"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_STRIDED_SLICE,
                    {packed_input},
                    {k_slice_output},
                    std::move(k_param_names),
                    do_op_validation),
                "Failed to create K slice node");

  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(k_slice_output, k_output,
                                                   slice_shape, kv_shape,
                                                   input_info.qnn_data_type,
                                                   input_info.quant_param,
                                                   do_op_validation,
                                                   false, false));

  // Extract V: slice [:, :, :, 1:2, :]
  std::vector<uint32_t> v_ranges_data;
  v_ranges_data.reserve(15);

  v_ranges_data.push_back(0);
  v_ranges_data.push_back(batch_size);
  v_ranges_data.push_back(1);

  v_ranges_data.push_back(0);
  v_ranges_data.push_back(seq_len);
  v_ranges_data.push_back(1);

  v_ranges_data.push_back(0);
  v_ranges_data.push_back(num_heads);
  v_ranges_data.push_back(1);

  // Dimension 3: slice index 1 (1:2:1) - extract V
  v_ranges_data.push_back(1);
  v_ranges_data.push_back(2);
  v_ranges_data.push_back(1);

  v_ranges_data.push_back(0);
  v_ranges_data.push_back(head_size);
  v_ranges_data.push_back(1);

  std::vector<std::string> v_param_names;
  std::vector<uint32_t> v_ranges_dims{5, 3};

  QnnParamWrapper v_ranges_param(node_unit.Index(), node_unit.Name() + "_v_ranges",
                                 QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                 std::move(v_ranges_dims),
                                 std::move(v_ranges_data),
                                 true);
  v_param_names.push_back(v_ranges_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(v_ranges_param));

  std::string v_slice_output = utils::GetUniqueName(node_unit, "_v_slice");
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, v_slice_output,
                                           input_info.qnn_data_type, slice_shape,
                                           input_info.quant_param));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::GetUniqueName(node_unit, "_v_slice"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_STRIDED_SLICE,
                    {packed_input},
                    {v_slice_output},
                    std::move(v_param_names),
                    do_op_validation),
                "Failed to create V slice node");

  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(v_slice_output, v_output,
                                                   slice_shape, kv_shape,
                                                   input_info.qnn_data_type,
                                                   input_info.quant_param,
                                                   do_op_validation,
                                                   false, false));

  return Ort::Status();
}

Ort::Status MultiHeadAttentionOpBuilder::ProcessAttributesAndOutputs(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    std::vector<std::string>&& input_names,
    const Ort::Logger& logger,
    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // Get attributes
  OrtNodeAttrHelper node_helper(node_unit);
  int64_t num_heads = node_helper.Get("num_heads", static_cast<int64_t>(0));
  float scale = node_helper.Get("scale", 0.0f);

  // Get input tensor info
  TensorInfo query_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], query_info));

  std::vector<uint32_t> query_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, query_shape),
                "Cannot get shape of query input");

  // Determine input format
  bool is_packed_qkv = (query_shape.size() == 5);
  bool is_packed_kv = false;

  if (!is_packed_qkv && inputs.size() >= 2 && inputs[1].Exists()) {
    std::vector<uint32_t> key_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].shape, key_shape),
                  "Cannot get shape of key input");
    is_packed_kv = (key_shape.size() == 5);
  }

  // Get Q, K, V tensors (unpacking if necessary)
  std::string q_tensor, k_tensor, v_tensor;

  if (is_packed_qkv) {
    // Unpack QKV from packed format
    RETURN_IF_ERROR(UnpackQKV(qnn_model_wrapper, node_unit, input_names[0],
                              query_shape, query_info, q_tensor, k_tensor, v_tensor,
                              do_op_validation));
  } else if (is_packed_kv) {
    // Q is separate, unpack K and V
    q_tensor = input_names[0];
    TensorInfo key_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], key_info));
    std::vector<uint32_t> key_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].shape, key_shape),
                  "Cannot get shape of key input");
    RETURN_IF_ERROR(UnpackKV(qnn_model_wrapper, node_unit, input_names[1],
                             key_shape, key_info, k_tensor, v_tensor,
                             do_op_validation));
  } else {
    // Separate Q, K, V
    q_tensor = input_names[0];
    k_tensor = input_names[1];
    v_tensor = input_names[2];
  }

  // Now we have Q, K, V tensors in shape (batch_size, seq_len, hidden_size) or
  // (batch_size, seq_len, num_heads, head_size) depending on format

  // For separate QKV format, we need to reshape to (batch, seq, num_heads, head_size)
  uint32_t batch_size, q_seq_len, kv_seq_len, hidden_size, head_size;

  if (is_packed_qkv) {
    // query_shape is in (batch, qkv_seq, num_heads, 3, head_size) format
    batch_size = query_shape[0];
    q_seq_len = query_shape[1];
    kv_seq_len = q_seq_len;
    head_size = query_shape[4];
    hidden_size = static_cast<uint32_t>(num_heads) * head_size;
  } else if (is_packed_kv) {
    // query_shape is in (batch, q_seq, hidden_size) format
    // key_shape is in (batch, kv_seq, num_heads, 2, head_size) format
    std::vector<uint32_t> key_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].shape, key_shape),
                  "Cannot get shape of key input");
    batch_size = query_shape[0];
    q_seq_len = query_shape[1];
    kv_seq_len = key_shape[1];
    head_size = key_shape[4];
    hidden_size = query_shape[2];  // Q is 3D: (batch, seq, hidden_size)

    // Reshape Q from 3D to 4D: (batch, seq, hidden_size) -> (batch, seq, num_heads, head_size)
    std::vector<uint32_t> q_4d_shape = {batch_size, q_seq_len, static_cast<uint32_t>(num_heads), head_size};
    std::vector<uint32_t> q_3d_shape = {batch_size, q_seq_len, hidden_size};

    std::string q_reshaped = utils::GetUniqueName(node_unit, "_q_reshaped");
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(q_tensor, q_reshaped,
                                                     q_3d_shape, q_4d_shape,
                                                     query_info.qnn_data_type,
                                                     query_info.quant_param,
                                                     do_op_validation,
                                                     false, false));
    q_tensor = q_reshaped;
  } else {
    // Separate format: (batch, seq, hidden_size)
    batch_size = query_shape[0];
    q_seq_len = query_shape[1];
    hidden_size = query_shape[2];
    head_size = hidden_size / static_cast<uint32_t>(num_heads);

    std::vector<uint32_t> key_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].shape, key_shape),
                  "Cannot get shape of key input");
    kv_seq_len = key_shape[1];

    // Reshape Q, K, V to (batch, seq, num_heads, head_size)
    std::vector<uint32_t> qkv_4d_shape = {batch_size, q_seq_len, static_cast<uint32_t>(num_heads), head_size};
    std::vector<uint32_t> kv_4d_shape = {batch_size, kv_seq_len, static_cast<uint32_t>(num_heads), head_size};
    std::vector<uint32_t> q_3d_shape = {batch_size, q_seq_len, hidden_size};
    std::vector<uint32_t> kv_3d_shape = {batch_size, kv_seq_len, hidden_size};

    std::string q_reshaped = utils::GetUniqueName(node_unit, "_q_reshaped");
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(q_tensor, q_reshaped,
                                                     q_3d_shape, qkv_4d_shape,
                                                     query_info.qnn_data_type,
                                                     query_info.quant_param,
                                                     do_op_validation,
                                                     false, false));
    q_tensor = q_reshaped;

    // Reshape K
    TensorInfo key_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], key_info));

    std::string k_reshaped = utils::GetUniqueName(node_unit, "_k_reshaped");
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(k_tensor, k_reshaped,
                                                     kv_3d_shape, kv_4d_shape,
                                                     key_info.qnn_data_type,
                                                     key_info.quant_param,
                                                     do_op_validation,
                                                     false, false));
    k_tensor = k_reshaped;

    // Reshape V
    TensorInfo value_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], value_info));

    std::string v_reshaped = utils::GetUniqueName(node_unit, "_v_reshaped");
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(v_tensor, v_reshaped,
                                                     kv_3d_shape, kv_4d_shape,
                                                     value_info.qnn_data_type,
                                                     value_info.quant_param,
                                                     do_op_validation,
                                                     false, false));
    v_tensor = v_reshaped;
  }

  // Now Q, K, V are all in shape (batch, seq, num_heads, head_size)
  // Transpose K for attention: (batch, num_heads, head_size, kv_seq)
  std::string k_transposed = utils::GetUniqueName(node_unit, "_k_transposed");
  std::vector<uint32_t> k_input_shape = {batch_size, kv_seq_len, static_cast<uint32_t>(num_heads), head_size};
  std::vector<uint32_t> k_transposed_shape = {batch_size, static_cast<uint32_t>(num_heads), head_size, kv_seq_len};
  std::vector<uint32_t> k_perm = {0, 2, 3, 1};  // (batch, seq, num_heads, head_size) -> (batch, num_heads, head_size, seq)

  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                     k_tensor,
                                                     k_transposed,
                                                     k_input_shape,
                                                     k_perm,
                                                     k_transposed_shape,
                                                     query_info.qnn_data_type,
                                                     query_info.quant_param,
                                                     do_op_validation,
                                                     false, false));

  // Transpose Q to (batch, num_heads, q_seq, head_size)
  std::string q_transposed = utils::GetUniqueName(node_unit, "_q_transposed");
  std::vector<uint32_t> q_input_shape = {batch_size, q_seq_len, static_cast<uint32_t>(num_heads), head_size};
  std::vector<uint32_t> q_transposed_shape = {batch_size, static_cast<uint32_t>(num_heads), q_seq_len, head_size};
  std::vector<uint32_t> q_perm = {0, 2, 1, 3};  // (batch, seq, num_heads, head_size) -> (batch, num_heads, seq, head_size)

  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                     q_tensor,
                                                     q_transposed,
                                                     q_input_shape,
                                                     q_perm,
                                                     q_transposed_shape,
                                                     query_info.qnn_data_type,
                                                     query_info.quant_param,
                                                     do_op_validation,
                                                     false, false));

  // Compute Q * K^T: (batch, num_heads, q_seq, head_size) x (batch, num_heads, head_size, kv_seq)
  //                = (batch, num_heads, q_seq, kv_seq)
  std::string qk_scores = utils::GetUniqueName(node_unit, "_qk_scores");
  std::vector<uint32_t> qk_shape = {batch_size, static_cast<uint32_t>(num_heads), q_seq_len, kv_seq_len};
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, qk_scores,
                                           query_info.qnn_data_type, qk_shape,
                                           query_info.quant_param));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::GetUniqueName(node_unit, "_qk_matmul"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_MAT_MUL,
                    {q_transposed, k_transposed},
                    {qk_scores},
                    {},
                    do_op_validation),
                "Failed to create QK matmul node");

  // Scale the attention scores
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  }

  std::string qk_scaled = utils::GetUniqueName(node_unit, "_qk_scaled");
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, qk_scaled,
                                           query_info.qnn_data_type, qk_shape,
                                           query_info.quant_param));

  // Create scale tensor with the same data type as input
  std::string scale_tensor_name = utils::GetUniqueName(node_unit, "_scale");
  std::vector<uint8_t> scale_data;

  // Convert scale to the appropriate data type
  if (query_info.qnn_data_type == QNN_DATATYPE_FLOAT_16) {
    scale_data.resize(sizeof(MLFloat16));
    MLFloat16 scale_fp16(scale);
    memcpy(scale_data.data(), &scale_fp16.val, sizeof(MLFloat16));
  } else {
    // Default to FLOAT_32
    scale_data.resize(sizeof(float));
    memcpy(scale_data.data(), &scale, sizeof(float));
  }

  std::vector<uint32_t> scale_shape = {1};

  QnnTensorWrapper scale_tensor(scale_tensor_name,
                                QNN_TENSOR_TYPE_STATIC,
                                query_info.qnn_data_type,
                                QnnQuantParamsWrapper(),
                                std::move(scale_shape),
                                std::move(scale_data));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_tensor)),
                "Failed to add scale tensor");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::GetUniqueName(node_unit, "_scale_mul"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                    {qk_scores, scale_tensor_name},
                    {qk_scaled},
                    {},
                    do_op_validation),
                "Failed to create scale multiply node");

  // Apply softmax over the last dimension
  std::string attention_weights = utils::GetUniqueName(node_unit, "_attention_weights");
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, attention_weights,
                                           query_info.qnn_data_type, qk_shape,
                                           query_info.quant_param));

  std::vector<std::string> softmax_params;
  uint32_t softmax_axis = 3;  // Last dimension of qk_shape (batch, num_heads, q_seq, kv_seq)
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  axis_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
  axis_qnn_scalar.uint32Value = softmax_axis;

  QnnParamWrapper axis_param_wrapper(node_unit.Index(), node_unit.Name() + "_softmax_axis",
                                     QNN_OP_SOFTMAX_PARAM_AXIS,
                                     axis_qnn_scalar);
  softmax_params.push_back(axis_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param_wrapper));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::GetUniqueName(node_unit, "_softmax"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_SOFTMAX,
                    {qk_scaled},
                    {attention_weights},
                    std::move(softmax_params),
                    do_op_validation),
                "Failed to create softmax node");

  // Transpose V to (batch, num_heads, kv_seq, head_size)
  std::string v_transposed = utils::GetUniqueName(node_unit, "_v_transposed");
  std::vector<uint32_t> v_input_shape = {batch_size, kv_seq_len, static_cast<uint32_t>(num_heads), head_size};
  std::vector<uint32_t> v_transposed_shape = {batch_size, static_cast<uint32_t>(num_heads), kv_seq_len, head_size};
  std::vector<uint32_t> v_perm = {0, 2, 1, 3};  // (batch, seq, num_heads, head_size) -> (batch, num_heads, seq, head_size)

  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                     v_tensor,
                                                     v_transposed,
                                                     v_input_shape,
                                                     v_perm,
                                                     v_transposed_shape,
                                                     query_info.qnn_data_type,
                                                     query_info.quant_param,
                                                     do_op_validation,
                                                     false, false));

  // Compute attention_weights * V: (batch, num_heads, q_seq, kv_seq) x (batch, num_heads, kv_seq, head_size)
  //                               = (batch, num_heads, q_seq, head_size)
  std::string context = utils::GetUniqueName(node_unit, "_context");
  std::vector<uint32_t> context_shape = {batch_size, static_cast<uint32_t>(num_heads), q_seq_len, head_size};
  RETURN_IF_ERROR(CreateIntermediateTensor(qnn_model_wrapper, context,
                                           query_info.qnn_data_type, context_shape,
                                           query_info.quant_param));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::GetUniqueName(node_unit, "_context_matmul"),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_MAT_MUL,
                    {attention_weights, v_transposed},
                    {context},
                    {},
                    do_op_validation),
                "Failed to create context matmul node");

  // Transpose context back to (batch, q_seq, num_heads, head_size)
  std::string context_transposed = utils::GetUniqueName(node_unit, "_context_transposed");
  std::vector<uint32_t> context_transposed_shape = {batch_size, q_seq_len, static_cast<uint32_t>(num_heads), head_size};
  std::vector<uint32_t> context_perm = {0, 2, 1, 3};  // (batch, num_heads, seq, head_size) -> (batch, seq, num_heads, head_size)

  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(),
                                                     context,
                                                     context_transposed,
                                                     context_shape,
                                                     context_perm,
                                                     context_transposed_shape,
                                                     query_info.qnn_data_type,
                                                     query_info.quant_param,
                                                     do_op_validation,
                                                     false, false));

  // Reshape output to (batch, q_seq, hidden_size)
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  const std::string& output_name = outputs[0].name;
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  std::vector<uint32_t> output_shape = {batch_size, q_seq_len, hidden_size};
  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(context_transposed, output_name,
                                                   context_transposed_shape, output_shape,
                                                   output_info.qnn_data_type,
                                                   output_info.quant_param,
                                                   do_op_validation,
                                                   false, is_graph_output));

  return Ort::Status();
}

void CreateMultiHeadAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MultiHeadAttentionOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
