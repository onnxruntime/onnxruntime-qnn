// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include <numeric>

namespace onnxruntime {
namespace qnn {

/* Op Resolution
  --> Incoming ONNX Node
    1. com.microsoft.GatherBlockQuantized
      Attributes : INT64
        - bits            : 4 (GPU)
        - block_size      : power of 2, >= 16 (GPU)
        - gather_axis     : axis of `data` to gather along (default 0)
        - quantize_axis   : axis of `data` that is block-quantized (only 1 supported on GPU)
      Inputs
        - data        : Init            : uint8 (UInt4x2 packed) : [vocab, hidden * bits / 8]
        - indices     :                 : int32/int64            : [batch_size, seq_len]
        - scales      : Init            : fp16/fp32             : [vocab, hidden / block_size]
        - zero_points : (optional)Init  : uint8                 : [vocab, hidden / block_size * bits / 8]
                                                                  (not supported on GPU: symmetric quant only)
      Outputs
        - Y           :                 : fp16/fp32             : [batch_size, seq_len, hidden]
  <-- Outgoing QNN Node (GPU)
    1. Gather
      Attributes : UINT32
        - axis            : gather_axis
      Inputs
        - Weight      : Static : qint4(BlockEncoding) : [vocab, hidden]
          - Scales    :        : fp32                 : [vocab * (hidden / block_size)]
          - Offsets   :        : int32_t (all zero)   : [vocab * (hidden / block_size)]
        - Indices     :        : int32                : [batch_size, seq_len]
      Outputs
        - Output      :        : fp16/fp32            : [batch_size, seq_len, hidden]

  Shape mapping / INT4-packed reinterpretation:
    The packed UInt4x2 weights (shape [vocab, hidden * bits / 8]) are re-biased
    by -8 (zero_point = 2^(bits-1)) so each nibble is reinterpretable as
    QNN_DATATYPE_SFIXED_POINT_4 in place. The QNN weight tensor is then declared
    with the unpacked element shape [vocab, scales_shape[1] * block_size] (== hidden).
    Per-block float scales pair with zero offsets, since the QNN GPU backend
    currently only supports symmetric quantization. This op builder is restricted
    to the QNN GPU backend.
*/

class GatherBlockQuantizedOpBuilder : public BaseOpBuilder {
 public:
  GatherBlockQuantizedOpBuilder()
      : BaseOpBuilder("GatherBlockQuantizedOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherBlockQuantizedOpBuilder);

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

// ================================================================
// IsOpSupported
// ================================================================
Ort::Status GatherBlockQuantizedOpBuilder::IsOpSupported(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    const Ort::Logger& logger) const {
  RETURN_IF_NOT(IsGpuBackend(qnn_model_wrapper.GetQnnBackendType()),
                "GatherBlockQuantized: GPU backend only");

  OrtNodeAttrHelper helper(node_unit);
  const int64_t bits = helper.Get("bits", static_cast<int64_t>(4));
  const int64_t block_size = helper.Get("block_size", static_cast<int64_t>(32));
  const int64_t quantize_axis = helper.Get("quantize_axis", static_cast<int64_t>(1));

  RETURN_IF_NOT(bits == 4,
                "GatherBlockQuantized: only INT4 (bits == 4) supported");
  RETURN_IF_NOT(block_size >= 16 && ((block_size & (block_size - 1)) == 0),
                "GatherBlockQuantized: block_size must be power of 2 and >= 16");
  RETURN_IF_NOT(quantize_axis == 1,
                "GatherBlockQuantized: only quantize_axis == 1 supported on QNN GPU");

  const auto& inputs = node_unit.Inputs();

  // Optional zero_points (4th input) implies asymmetric quantization. The QNN
  // GPU backend only supports symmetric quantization (offsets forced to 0), so
  // reject models that supply zero_points rather than produce silent wrong output.
  RETURN_IF(inputs.size() > 3 && inputs[3].Exists(),
            "GatherBlockQuantized: zero_points input not supported (symmetric quant only)");
  {
    Qnn_DataType_t weight_datatype;
    const OrtNodeUnitIODef& weight_tensor = inputs[0];
    TensorInfo weights_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(weight_tensor, weights_info));
    RETURN_IF_ERROR(utils::GetQnnDataType(
        weight_tensor.quant_param.has_value(),
        weight_tensor.type,
        weight_datatype,
        qnn_model_wrapper.GetQnnBackendType()));
    RETURN_IF((weight_datatype != QNN_DATATYPE_UINT_8) && (weight_datatype != QNN_DATATYPE_SFIXED_POINT_4),
              "GatherBlockQuantized: weights must be UINT_8 or SFIXED_POINT_4");
    RETURN_IF_NOT(weights_info.shape.size() == 2,
                  "GatherBlockQuantized: only rank-2 weights supported");
  }

  // Indices datatype is constrained by the ONNX OpDef (int32/int64), so it is
  // not re-validated here. int64 indices are converted to int32 in ProcessInputs
  // (static cast for initializers, Cast node for dynamic inputs).

  // Validate scales datatype (float32 or float16).
  {
    Qnn_DataType_t scale_datatype;
    const OrtNodeUnitIODef& scales_tensor = inputs[2];
    TensorInfo scales_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(scales_tensor, scales_info));
    RETURN_IF_ERROR(utils::GetQnnDataType(
        scales_tensor.quant_param.has_value(),
        scales_tensor.type,
        scale_datatype,
        qnn_model_wrapper.GetQnnBackendType()));
    RETURN_IF(scale_datatype != QNN_DATATYPE_FLOAT_32 && scale_datatype != QNN_DATATYPE_FLOAT_16,
              "GatherBlockQuantized: scales must be FLOAT32 or FLOAT16");
    RETURN_IF_NOT(scales_info.shape.size() == 2,
                  "GatherBlockQuantized: only rank-2 scales supported");
  }

  // Run the input/output processing with do_op_validation=true so the QNN
  // backend validates the op at partition time. Backend rejections then surface
  // as partition-time fallbacks instead of compile-time fatals.
  std::vector<std::string> input_names;
  RETURN_IF_ERROR(ProcessInputs(qnn_model_wrapper, node_unit, logger, input_names, true));
  RETURN_IF_ERROR(ProcessAttributesAndOutputs(qnn_model_wrapper, node_unit, std::move(input_names), logger, true));

  return Ort::Status();
}

// ================================================================
// ProcessInputs
// ================================================================
Ort::Status GatherBlockQuantizedOpBuilder::ProcessInputs(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    const Ort::Logger& logger,
    std::vector<std::string>& input_names,
    bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_NOT(IsGpuBackend(qnn_model_wrapper.GetQnnBackendType()),
                  "GatherBlockQuantized: GPU backend only");
  }

  // ------------------------------------------------------------
  // 1. Weights and scales
  // ------------------------------------------------------------
  const auto& inputs = node_unit.Inputs();

  // Get weight info
  const auto& weight_tensor = inputs[0];
  TensorInfo weight_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(weight_tensor, weight_info));
  Qnn_DataType_t weight_type = weight_info.qnn_data_type;
  std::vector<uint32_t> weight_shape = weight_info.shape;

  // Get scale info
  const auto& scales_tensor = inputs[2];
  TensorInfo scale_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(scales_tensor, scale_info));
  std::vector<uint32_t> scale_shape = scale_info.shape;

  // Required params
  OrtNodeAttrHelper helper(node_unit);
  const int64_t block_size = helper.Get("block_size", static_cast<int64_t>(32));
  const int64_t num_blocks = std::accumulate(scale_shape.begin(),
                                             scale_shape.end(),
                                             int64_t{1},
                                             std::multiplies<int64_t>());
  const std::vector<uint32_t> block_sizes = {1, gsl::narrow_cast<uint32_t>(block_size)};

  // Creating weight+scale wrapper
  const std::string& weight_tensor_name = weight_tensor.name;
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(weight_tensor_name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + weight_tensor_name).c_str());
  } else {
    // Unpack weights
    std::vector<uint8_t> quant_data;
    Qnn_TensorType_t weight_tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
    const OrtValueInfo* weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
    RETURN_IF_NOT(weight_tensor_proto != nullptr,
                  "GatherBlockQuantized: weights must be a constant initializer");
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(weight_tensor_proto, quant_data, false));

    // Transform quantized weights to signed fixed point 4.
    bool needs_uint4_to_int4 = (weight_type == QNN_DATATYPE_UINT_8);
    if (needs_uint4_to_int4) {
      RETURN_IF_ERROR(utils::TransformUnsignedToSignedFixedPoint(quant_data, 4));
    }

    // Unpack scales
    std::vector<uint8_t> uint8_scale;
    const OrtValueInfo* scale_tensor_proto = qnn_model_wrapper.GetConstantTensor(scales_tensor.name);
    RETURN_IF_NOT(scale_tensor_proto != nullptr,
                  "GatherBlockQuantized: scales must be a constant initializer");
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(scale_tensor_proto, uint8_scale, false));

    const OrtTypeInfo* type_info = nullptr;
    const auto& ort_api = qnn_model_wrapper.GetOrtApi();
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(scale_tensor_proto, &type_info));
    const OrtTensorTypeAndShapeInfo* tensor_type_and_shape_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_type_and_shape_info));
    ONNXTensorElementDataType onnx_data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(tensor_type_and_shape_info, &onnx_data_type));

    RETURN_IF(onnx_data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                  onnx_data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
              "Unsupported scales datatype");

    const size_t elem_byte_size = qnn::utils::GetElementSizeByType(onnx_data_type);
    RETURN_IF_NOT(uint8_scale.size() == (static_cast<size_t>(num_blocks) * elem_byte_size),
                  "Scale Initializer Invalid Size");

    std::vector<float> float_scale;
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      float* float_scale_ptr = reinterpret_cast<float*>(uint8_scale.data());
      float_scale = std::vector<float>(float_scale_ptr, float_scale_ptr + num_blocks);
    } else {
      Ort::Float16_t* fp16_scale_ptr = reinterpret_cast<Ort::Float16_t*>(uint8_scale.data());
      float_scale.reserve(num_blocks);
      for (int64_t i = 0; i < num_blocks; i++) {
        float_scale.emplace_back(static_cast<float>(fp16_scale_ptr[i]));
      }
    }

    // Quantization Offsets : QNN GPU backend currently supports only symmetric
    // quantization, so offsets are forced to 0.
    std::vector<int32_t> int32_offset(num_blocks, 0);

    // Create Quantization Parameter and create Weight Tensor.
    // The packed UInt4x2 bytes (weight_shape[1] bytes/row, 2 int4 nibbles per
    // byte) must expand to the unpacked element count implied by scales:
    // scale_shape[1] * block_size.
    RETURN_IF_NOT(static_cast<int64_t>(weight_shape[1]) * 2 ==
                      static_cast<int64_t>(scale_shape[1]) * block_size,
                  "GatherBlockQuantized: weight packed bytes mismatch with scales * block_size");
    QnnQuantParamsWrapper quantize_param = QnnQuantParamsWrapper(float_scale,
                                                                 int32_offset,
                                                                 block_sizes,
                                                                 QNN_DATATYPE_SFIXED_POINT_4);
    std::vector<uint32_t> weight_shape_ = {static_cast<uint32_t>(weight_shape[0]),
                                           static_cast<uint32_t>(scale_shape[1] * block_size)};
    QnnTensorWrapper weight_tensor_wrapper(weight_tensor_name,
                                           weight_tensor_type,
                                           QNN_DATATYPE_SFIXED_POINT_4,
                                           std::move(quantize_param),
                                           std::move(weight_shape_),
                                           std::move(quant_data));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor_wrapper)), "Failed to add tensor.");
  }
  input_names.push_back(weight_tensor_name);

  // ------------------------------------------------------------
  // 2. Indices
  // ------------------------------------------------------------
  // QNN Gather only supports int32 / uint32 indices.
  //  - Static int64 indices: statically reinterpret int64 -> int32.
  //  - Dynamic int64 indices: add an explicit QNN Cast node int64 -> int32.
  const OrtNodeUnitIODef& indices_tensor = inputs[1];
  const std::string& indices_name = indices_tensor.name;

  TensorInfo indices_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_tensor, indices_info));

  const bool indices_is_int64 = (indices_info.qnn_data_type == QNN_DATATYPE_INT_64);

  if (qnn_model_wrapper.IsQnnTensorWrapperExist(indices_name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + indices_name).c_str());
  } else {
    if (indices_info.is_initializer && indices_is_int64) {
      // Statically convert int64 indices to int32.
      std::vector<uint8_t> onnx_indices_bytes;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(indices_info.initializer_tensor, onnx_indices_bytes));

      const size_t num_elems = onnx_indices_bytes.size() / sizeof(int64_t);
      gsl::span<const int64_t> onnx_indices{reinterpret_cast<const int64_t*>(onnx_indices_bytes.data()), num_elems};

      std::vector<uint8_t> qnn_indices_bytes(num_elems * sizeof(int32_t));
      gsl::span<int32_t> qnn_indices{reinterpret_cast<int32_t*>(qnn_indices_bytes.data()), num_elems};
      for (size_t i = 0; i < num_elems; ++i) {
        qnn_indices[i] = static_cast<int32_t>(onnx_indices[i]);
      }

      QnnTensorWrapper indices_tensor_wrapper(indices_name,
                                              QNN_TENSOR_TYPE_STATIC,
                                              QNN_DATATYPE_INT_32,
                                              QnnQuantParamsWrapper(),
                                              std::vector<uint32_t>(indices_info.shape),
                                              std::move(qnn_indices_bytes));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(indices_tensor_wrapper)),
                    "Failed to add indices tensor");
    } else {
      QnnTensorWrapper indices_tensor_wrapper;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(indices_info, indices_name, indices_tensor_wrapper));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(indices_tensor_wrapper)),
                    "Failed to add indices tensor");
    }
  }

  // Add an explicit Cast node for dynamic int64 indices.
  std::string indices_input_name = indices_name;
  if (indices_is_int64 && !indices_info.is_initializer) {
    const std::string indices_casted_name = indices_name + "_int32";
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::UniqueNameGenerator().New(indices_name, QNN_OP_CAST),
                                                  indices_name,
                                                  indices_casted_name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  QNN_DATATYPE_INT_32,
                                                  QnnQuantParamsWrapper(),
                                                  std::vector<uint32_t>(indices_info.shape),
                                                  do_op_validation));
    indices_input_name = indices_casted_name;
  }
  input_names.push_back(indices_input_name);
  return Ort::Status();
}

// ================================================================
// ProcessAttributesAndOutputs
// ================================================================
Ort::Status GatherBlockQuantizedOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                       const OrtNodeUnit& node_unit,
                                                                       std::vector<std::string>&& input_names,
                                                                       const Ort::Logger& logger,
                                                                       bool do_op_validation) const {
  OrtNodeAttrHelper helper(node_unit);
  const int64_t axis_attr = helper.Get("gather_axis", 0);

  // Output info
  const OrtNodeUnitIODef& output_tensor = node_unit.Outputs()[0];
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_tensor, output_info));

  // Creating output wrapper
  const std::string& output_tensor_name = output_tensor.name;
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(output_tensor_name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + output_tensor_name).c_str());
  } else {
    QnnTensorWrapper output_tensor_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_tensor, output_tensor_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)), "Failed to add output");
  }

  // Creating axis param wrapper
  std::vector<std::string> param_tensor_names;
  int32_t axis_value = static_cast<int32_t>(axis_attr);
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis_value));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(),
                             QNN_OP_GATHER_PARAM_AXIS, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(axis_param)),
                "Failed to add axis param.");

  // Creating Qnn node
  RETURN_IF_NOT(
      qnn_model_wrapper.CreateQnnNode(
          output_tensor_name,
          QNN_OP_PACKAGE_NAME_QTI_AISW,
          QNN_OP_GATHER,
          std::move(input_names),
          {output_tensor_name},
          std::move(param_tensor_names),
          do_op_validation),
      "Failed to create Gather node");
  return Ort::Status();
}

// ================================================================
// Registration
// ================================================================
void CreateGatherBlockQuantizedOpBuilder(
    const std::string& op_type,
    OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(
      op_type,
      std::make_unique<GatherBlockQuantizedOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
