// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>

#include <gsl/gsl>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

/**
 * An ONNX MatMul can be translated to either a QNN MatMul or a QNN FullyConnected.
 * ONNX's MatMul supports inputs of rank 1, but neither QNN's MatMul nor FullyConnected support two rank 1 inputs.
 * So, we need to add Reshape Ops if necessary.
 * In two cases, FullyConnected (input_1's shape is [n, k]) is used instead of MatMul without extra Transpose Op:
 * 1. input_1 is a rank 2 initializer.
 * 2. input_1 is a rank 1 tensor.
 */
class MatMulOpBuilder : public BaseOpBuilder {
 public:
  MatMulOpBuilder() : BaseOpBuilder("MatMulOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulOpBuilder);

 protected:
  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names, const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ProcessInputsForQnnMatMul(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const TensorInfo& input_info_0,
                                        const TensorInfo& input_info_1,
                                        const Ort::Logger& logger,
                                        std::vector<std::string>& input_names,
                                        bool do_op_validation) const ORT_MUST_USE_RESULT;
  Ort::Status ProcessInputsForQnnFullyConnected(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const TensorInfo& input_info_0,
                                                const TensorInfo& input_info_1,
                                                const Ort::Logger& logger,
                                                std::vector<std::string>& input_names,
                                                bool do_op_validation) const ORT_MUST_USE_RESULT;
  // Block-quantized (BW_FLOAT_BLOCK) weight path. Translates to a QNN MatMul whose weight carries a
  // per-block float scale; activation is dequantized to FP16 and the FP16 output is re-quantized to INT16.
  Ort::Status ProcessInputsForBQMatMul(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const TensorInfo& input_info_1,
                                       const Ort::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) const ORT_MUST_USE_RESULT;
};

namespace {
inline bool IsQuant16bit(Qnn_DataType_t qnn_data_type) {
  return qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16 || qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16;
}

// HTP BQ MatMul: supported weight bitwidths and their block_size divisor constraints.
// block_size must be a multiple of the corresponding value (same as Conv BQ / MatMulNBits HTP constraints).
const std::unordered_map<uint32_t, int64_t> kHtpMatMulBQBitsAndBlockSizeMultipliers{
    {2, 16}, {4, 8}, {8, 4}};

// Returns BQ weight bitwidth (2/4/8) from an ONNX element data type, or 0 if unsupported.
uint32_t GetBQBitwidth(ONNXTensorElementDataType onnx_type) {
  switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return 8;
    default:
      return 0;
  }
}

// Detects a block-quantized MatMul weight (ONNX MatMul input[1], shape [K, N]).
// Per ONNX opset 21, the weight scale has the same rank as the weight with the blocked axis
// dimension smaller. MatMul blocks the contraction axis K (axis 0), so the rank-2 scale is
// [K/block_size, N] and scale_shape[0] < weight_shape[0]. Only meaningful on the NPU backend.
// On success, sets num_blocks = scale_shape[0] and block_size = K / num_blocks.
bool IsBQWeight(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnitIODef& weight,
                int64_t& num_blocks, int64_t& block_size) {
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    return false;
  }
  if (!weight.quant_param.has_value() || weight.quant_param->scale == nullptr) {
    return false;
  }
  const auto scale_shape = utils::GetInitializerShape(weight.quant_param->scale, qnn_model_wrapper.GetOrtApi());
  std::vector<uint32_t> weight_shape;
  if (!QnnModelWrapper::GetOnnxShape(weight.shape, weight_shape) || weight_shape.size() != 2) {
    return false;  // BQ only supported for rank-2 MatMul weight [K, N].
  }
  if (scale_shape.size() != weight_shape.size() ||
      scale_shape[0] >= static_cast<int64_t>(weight_shape[0])) {
    return false;
  }
  num_blocks = scale_shape[0];
  if (num_blocks <= 0 || static_cast<int64_t>(weight_shape[0]) % num_blocks != 0) {
    return false;
  }
  block_size = static_cast<int64_t>(weight_shape[0]) / num_blocks;
  return true;
}

// Flattens the leading dims of `shape` (all but the last) into a single uint32_t batch value.
Ort::Status FlattenLeadingDims(const std::vector<uint32_t>& shape, uint32_t& batch) {
  const int64_t batch_i64 = std::accumulate(shape.begin(), shape.end() - 1,
                                            static_cast<int64_t>(1), std::multiplies<int64_t>());
  RETURN_IF(batch_i64 <= 0 ||
                batch_i64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
            "MatMul: flattened batch dimension product overflows uint32_t.");
  batch = static_cast<uint32_t>(batch_i64);
  return Ort::Status();
}

Ort::Status CheckInputs(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnitIODef& input_def_0,
                        const OrtNodeUnitIODef& input_def_1, TensorInfo& input_info_0, TensorInfo& input_info_1,
                        bool& use_fully_connected) {
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def_0, input_info_0));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def_1, input_info_1));

#if QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR <= 20
  // Validation crashes if use QNN FullyConnected in QNN SDK versions 2.26 - 2.27
  // Just use QNN MatMul for these older QNN SDK versions.
  use_fully_connected = false;
#else
  // Use FullyConnected if 2nd input is a rank 2 initializer or a rank 1 tensor.
  // FullyConnected cannot pass the Op validation if keep_dims is true, so if input_0 is per-channel quantized tensor
  // with rank > 2, it's not easy to set the quantization parameters for the output reshaped rank 2 tensor.
  // In this case, we will not use FullyConnected.
  use_fully_connected =
      (input_info_1.shape.size() == 2 && input_info_1.is_initializer) || input_info_1.shape.size() == 1;
  use_fully_connected =
      use_fully_connected && !(input_info_0.quant_param.IsPerChannel() && input_info_0.shape.size() > 2);
  // Don't use FullyConnected if both inputs are dynamic and uint16 (quantized)
  use_fully_connected = use_fully_connected && !(IsQuant16bit(input_info_0.qnn_data_type) &&
                                                 !input_info_0.is_initializer &&
                                                 IsQuant16bit(input_info_1.qnn_data_type) &&
                                                 !input_info_1.is_initializer);
#endif
  return Ort::Status();
}

// Process input[0] for ONNX MatMul that can be translated to either a QNN MatMul or a QNN FullyConnected.
Ort::Status ProcessInput0(QnnModelWrapper& qnn_model_wrapper,
                          const TensorInfo& input_0_info,
                          const std::string& original_input_0_name,
                          std::vector<std::string>& input_names,
                          const Ort::Logger& logger,
                          bool do_op_validation,
                          bool use_fully_connected) {
  const bool is_rank1 = input_0_info.shape.size() == 1;
  const bool reshape_input_0 = is_rank1 || (use_fully_connected && input_0_info.shape.size() > 2);
  std::string actual_input_0_name = original_input_0_name;

  if (reshape_input_0) {
    actual_input_0_name = utils::UniqueNameGenerator().New(original_input_0_name, "_reshape");
    std::vector<uint32_t> shape_2d;
    if (is_rank1) {
      shape_2d = {1, input_0_info.shape[0]};
    } else {
      uint32_t batch = 0;
      RETURN_IF_ERROR(FlattenLeadingDims(input_0_info.shape, batch));
      shape_2d = {batch, input_0_info.shape.back()};
    }
    QnnQuantParamsWrapper quant_param_2d = input_0_info.quant_param.Copy();
    if (is_rank1) {
      RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_0_info.shape, shape_2d));
    }

    // If input_0 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_0_info.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_0_info.initializer_tensor, unpacked_tensor));
      QnnTensorWrapper input_tensorwrapper(actual_input_0_name, QNN_TENSOR_TYPE_STATIC, input_0_info.qnn_data_type,
                                           std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(original_input_0_name, actual_input_0_name,
                                                       input_0_info.shape, shape_2d,
                                                       input_0_info.qnn_data_type, input_0_info.quant_param,
                                                       quant_param_2d, do_op_validation,
                                                       qnn_model_wrapper.IsGraphInput(original_input_0_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(actual_input_0_name)) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + actual_input_0_name).c_str());
    } else {
      QnnTensorWrapper input_0_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_0_info, actual_input_0_name, input_0_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_0_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(actual_input_0_name);

  return Ort::Status();
}
}  // namespace

// Process operator inputs. Dispatches to other processing functions depending on whether we're
// translating an ONNX MatMul to a QNN MatMul or a QNN FullyConnected.
Ort::Status MatMulOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();

  // Block-quantized (BW_FLOAT_BLOCK) weight: validate HTP constraints, then defer to the base
  // implementation, which runs full QNN validation through our BQ ProcessInputs/Outputs path.
  if (inputs.size() >= 2) {
    int64_t num_blocks = 0;
    int64_t block_size = 0;
    if (IsBQWeight(qnn_model_wrapper, inputs[1], num_blocks, block_size)) {
      const uint32_t bitwidth = GetBQBitwidth(inputs[1].type);
      auto bq_it = kHtpMatMulBQBitsAndBlockSizeMultipliers.find(bitwidth);
      RETURN_IF(bq_it == kHtpMatMulBQBitsAndBlockSizeMultipliers.end(),
                ("QNN HTP MatMul BQ: unsupported weight bitwidth=" + std::to_string(bitwidth)).c_str());
      RETURN_IF(block_size % bq_it->second != 0,
                ("QNN HTP MatMul BQ: block_size=" + std::to_string(block_size) +
                 " must be a multiple of " + std::to_string(bq_it->second) +
                 " for " + std::to_string(bitwidth) + "-bit weight")
                    .c_str());
      // BQ requires a constant weight and a dynamic (quantized) activation that we dequantize to FP16.
      TensorInfo weight_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], weight_info));
      RETURN_IF_NOT(weight_info.is_initializer, "QNN EP: BQ MatMul weight must be a constant initializer");
      TensorInfo act_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], act_info));
      RETURN_IF(act_info.is_initializer, "QNN EP: BQ MatMul activation must be a dynamic (non-constant) tensor");
    }
  }

  return BaseOpBuilder::IsOpSupported(qnn_model_wrapper, node_unit, logger);
}

// Process operator inputs. Dispatches to other processing functions depending on whether we're
// translating an ONNX MatMul to a QNN MatMul or a QNN FullyConnected.
Ort::Status MatMulOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger, std::vector<std::string>& input_names,
                                           bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  // Block-quantized weight: translate to a QNN MatMul with a BW_FLOAT_BLOCK weight (weight stays 2-D).
  {
    int64_t num_blocks = 0;
    int64_t block_size = 0;
    if (IsBQWeight(qnn_model_wrapper, inputs[1], num_blocks, block_size)) {
      TensorInfo input_info_1{};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input_info_1));
      return ProcessInputsForBQMatMul(qnn_model_wrapper, node_unit, input_info_1, logger, input_names,
                                      do_op_validation);
    }
  }

  TensorInfo input_info_0{};
  TensorInfo input_info_1{};
  bool use_fully_connected = false;
  RETURN_IF_ERROR(
      CheckInputs(qnn_model_wrapper, inputs[0], inputs[1], input_info_0, input_info_1, use_fully_connected));

  if (use_fully_connected) {
    return ProcessInputsForQnnFullyConnected(qnn_model_wrapper,
                                             node_unit,
                                             input_info_0,
                                             input_info_1,
                                             logger,
                                             input_names,
                                             do_op_validation);
  }
  return ProcessInputsForQnnMatMul(qnn_model_wrapper,
                                   node_unit,
                                   input_info_0,
                                   input_info_1,
                                   logger,
                                   input_names,
                                   do_op_validation);
}

Ort::Status MatMulOpBuilder::ProcessInputsForQnnMatMul(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const TensorInfo& input_info_0,
                                                       const TensorInfo& input_info_1,
                                                       const Ort::Logger& logger,
                                                       std::vector<std::string>& input_names,
                                                       bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const bool reshape_input_1 = input_info_1.shape.size() == 1;

  const std::string& org_input_0_name = inputs[0].name;
  RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, org_input_0_name, input_names,
                                logger, do_op_validation, /*use_fully_connected=*/false));

  // Process input 1.
  const std::string& org_input_1_name = inputs[1].name;
  std::string input_1_name = org_input_1_name;
  if (reshape_input_1) {
    // Input[1] is a rank 1 tensor that needs to be reshaped.
    std::vector<uint32_t> shape_2d;
    QnnQuantParamsWrapper quant_param_2d = input_info_1.quant_param.Copy();
    input_1_name = utils::UniqueNameGenerator().New(org_input_1_name, "_reshape");
    shape_2d = {input_info_1.shape[0], 1};
    RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_info_1.shape, shape_2d));

    // If input_1 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_info_1.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_info_1.initializer_tensor, unpacked_tensor));

      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_1_name);
      QnnTensorWrapper input_tensorwrapper(input_1_name, tensor_type, input_info_1.qnn_data_type,
                                           std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(org_input_1_name, input_1_name, input_info_1.shape, shape_2d,
                                                       input_info_1.qnn_data_type, input_info_1.quant_param,
                                                       quant_param_2d, do_op_validation,
                                                       qnn_model_wrapper.IsGraphInput(org_input_1_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_1_name)) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_1_name).c_str());
    } else {
      QnnTensorWrapper input_1_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[1], input_1_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_1_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(input_1_name);

  // Inserts a QNN Convert op before uint16 input[1] to avoid QNN HTP validation failure.
  //
  // QNN graph that fails validation:
  //     input_0_uint16 ---> MatMul ---> output_uint16
  //                         ^
  //                         |
  //     input_1_uint16 -----+
  //
  // For dynamic weights, QNN graph that passes validation:
  //     input_0_uint16 ---------------------------> MatMul ---> output_uint16
  //                                                   ^
  //                                                   |
  //     input_1_uint16_asym --> Convert(uint16_sym) --+
  //
  // For static weights, QNN graph that passes validation:
  //     input_0_uint16 ---------------------> MatMul ---> output_uint16
  //                                             ^
  //                                             |
  //     input_1_uint16 --> Convert(int16_sym) --+
  if (!input_info_0.is_initializer &&
      input_info_0.qnn_data_type == input_info_1.qnn_data_type &&
      input_info_0.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16) {
    RETURN_IF_NOT(input_info_1.quant_param.IsPerTensor(),
                  "MatMul's activation inputs only support per-tensor quantization");
    const Qnn_QuantizeParams_t& quant_param = input_info_1.quant_param.Get();
    // insert Convert op after input1
    std::string convert_input_name = input_names.back();
    input_names.pop_back();
    const std::string convert_output_name = utils::UniqueNameGenerator().New(convert_input_name, "_convert");
    std::vector<uint32_t> input_1_shape = input_info_1.shape;
    if (reshape_input_1) {
      input_1_shape = {input_info_1.shape[0], 1};
    }
    if (!input_info_1.is_initializer) {
      // Only insert Convert for asymmetric quantization (i.e., offset != 2^(16-1)).
      if (quant_param.scaleOffsetEncoding.offset != 32768) {
        RETURN_IF_ERROR(utils::InsertConvertOp(qnn_model_wrapper,
                                               convert_input_name,
                                               convert_output_name,
                                               input_info_1.qnn_data_type,
                                               QNN_DATATYPE_UFIXED_POINT_16,
                                               quant_param.scaleOffsetEncoding.offset,
                                               quant_param.scaleOffsetEncoding.scale,
                                               input_1_shape,
                                               true,  // symmetric
                                               do_op_validation));
        input_names.push_back(convert_output_name);
      } else {
        input_names.push_back(convert_input_name);
      }
    } else {
      RETURN_IF_ERROR(utils::InsertConvertOp(qnn_model_wrapper,
                                             convert_input_name,
                                             convert_output_name,
                                             input_info_1.qnn_data_type,
                                             QNN_DATATYPE_SFIXED_POINT_16,
                                             quant_param.scaleOffsetEncoding.offset,
                                             quant_param.scaleOffsetEncoding.scale,
                                             input_1_shape,
                                             true,  // symmetric
                                             do_op_validation));
      input_names.push_back(convert_output_name);
    }
  }
  return Ort::Status();
}

Ort::Status MatMulOpBuilder::ProcessInputsForQnnFullyConnected(QnnModelWrapper& qnn_model_wrapper,
                                                               const OrtNodeUnit& node_unit,
                                                               const TensorInfo& input_info_0,
                                                               const TensorInfo& input_info_1,
                                                               const Ort::Logger& logger,
                                                               std::vector<std::string>& input_names,
                                                               bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const bool reshape_input_1 = input_info_1.shape.size() == 1;

  const std::string& org_input_0_name = inputs[0].name;
  RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, org_input_0_name, input_names,
                                logger, do_op_validation, /*use_fully_connected=*/true));

  // Process input 1.
  const std::string& org_input_1_name = inputs[1].name;
  std::string input_1_name = org_input_1_name;
  std::vector<uint32_t> shape_2d;
  QnnQuantParamsWrapper quant_param_2d = input_info_1.quant_param.Copy();
  if (reshape_input_1) {
    // Input[1] is a rank 1 tensor that needs to be reshaped.
    input_1_name = utils::UniqueNameGenerator().New(org_input_1_name, "_reshape");

    // FullyConnected requires input_1's shape to be [n, k].
    shape_2d = {1, input_info_1.shape[0]};
    RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_info_1.shape, shape_2d));
  } else {
    assert(input_info_1.shape.size() == 2);
    input_1_name = utils::UniqueNameGenerator().New(org_input_1_name, "_transpose");
    shape_2d = {input_info_1.shape[1], input_info_1.shape[0]};
    RETURN_IF_ERROR(quant_param_2d.HandleTranspose<uint32_t>(std::vector<uint32_t>({1, 0})));
  }

  // If input_1 is initializer, unpack it and add the tensor with new quantization parameter and shape.
  // Otherwise, add a Reshape node.
  if (input_info_1.is_initializer) {
    std::vector<uint8_t> unpacked_tensor;
    if (!reshape_input_1) {
      // 2D initializer should be transposed to [n, k].
      std::vector<uint32_t> original_shape_copy = input_info_1.shape;
      RETURN_IF_ERROR(utils::TwoDimensionTranspose(qnn_model_wrapper,
                                                   original_shape_copy,  // Will be modified to new shape (unnecessary)
                                                   input_info_1.initializer_tensor,
                                                   unpacked_tensor,
                                                   logger));
    } else {
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_info_1.initializer_tensor, unpacked_tensor));
    }

    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_1_name);
    QnnTensorWrapper input_tensorwrapper(input_1_name, tensor_type, input_info_1.qnn_data_type,
                                         std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  } else {
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(org_input_1_name, input_1_name, input_info_1.shape, shape_2d,
                                                     input_info_1.qnn_data_type, input_info_1.quant_param,
                                                     quant_param_2d, do_op_validation,
                                                     qnn_model_wrapper.IsGraphInput(org_input_1_name), false));
  }
  input_names.emplace_back(input_1_name);

  // Workaround that inserts a QNN Convert op before input[1] (converts from quantized uint16 to signed symmetric int16)
  // to avoid a QNN validation failure.
  //
  // QNN graph WITHOUT workaround (fails validation):
  //     input_0_uint16 ---> FC ---> output_uint16
  //                         ^
  //                         |
  //     input_1_uint16 -----+
  //
  // QNN graph WITH workaround (passes validation):
  //     input_0_uint16 ----------------------> FC ---> output_uint16
  //                                            ^
  //                                            |
  //     input_1_uint16 --> Convert(to int16) --+

  std::string weight_input_name = input_names.back();
  const auto& weight_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(weight_input_name);

  if (weight_tensor_wrapper.GetTensorDataType() == QNN_DATATYPE_UFIXED_POINT_16) {
    const auto& quant_param_wrapper = weight_tensor_wrapper.GetQnnQuantParams();
    const Qnn_QuantizeParams_t& quant_param = quant_param_wrapper.Get();
    const auto& transformed_input1_shape = weight_tensor_wrapper.GetTensorDims();

    RETURN_IF_NOT(quant_param_wrapper.IsPerTensor(),
                  "FC's INT16 weight inputs only support INT16 per-tensor quantization");

    // Pop Conv weight. Insert Convert op after Weight
    input_names.pop_back();
    std::string convert_output_name = utils::UniqueNameGenerator().New(weight_input_name, "_convert");

    RETURN_IF_ERROR(utils::InsertConvertOp(qnn_model_wrapper,
                                           weight_input_name,
                                           convert_output_name,
                                           QNN_DATATYPE_UFIXED_POINT_16,
                                           QNN_DATATYPE_SFIXED_POINT_16,
                                           quant_param.scaleOffsetEncoding.offset,
                                           quant_param.scaleOffsetEncoding.scale,
                                           transformed_input1_shape,
                                           true,  // Symmetric
                                           do_op_validation));
    input_names.push_back(convert_output_name);
  }
  return Ort::Status();
}

Ort::Status MatMulOpBuilder::ProcessInputsForBQMatMul(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      const TensorInfo& input_info_1,
                                                      const Ort::Logger& logger,
                                                      std::vector<std::string>& input_names,
                                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  const auto& inputs = node_unit.Inputs();

  RETURN_IF_NOT(input_info_1.is_initializer, "QNN EP: BQ MatMul weight must be a constant initializer");
  RETURN_IF_NOT(input_info_1.shape.size() == 2, "QNN EP: BQ MatMul weight must be rank-2 [K, N]");
  const int64_t K = static_cast<int64_t>(input_info_1.shape[0]);
  const int64_t N = static_cast<int64_t>(input_info_1.shape[1]);

  //
  // Input 0: activation. BW_FLOAT_BLOCK MatMul computes in FP16, so an INT16 activation must be
  // dequantized to FP16 first (mirrors the Conv BQ activation path). QNN HTP additionally requires the
  // activation to be 4-D, so the [..., M, K] activation is reshaped to [batch, 1, M, K], where batch is
  // the product of all leading dims. Any rank >= 2 reshapes cleanly; the output is reshaped back the
  // same way.
  //
  TensorInfo input_info_0{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info_0));
  RETURN_IF_NOT(input_info_0.shape.size() >= 2,
                "QNN EP: BQ MatMul activation must be rank >= 2 so it can be reshaped to 4-D [batch, 1, M, K]");
  RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, inputs[0].name, input_names, logger,
                                do_op_validation, /*use_fully_connected=*/false));
  {
    const std::string act_name = input_names[0];
    const auto& act_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(act_name);
    const Qnn_DataType_t act_dtype = act_wrapper.GetTensorDataType();
    std::vector<uint32_t> act_shape = act_wrapper.GetTensorDims();

    std::string fp16_name = act_name;
    if (act_dtype == QNN_DATATYPE_SFIXED_POINT_16 || act_dtype == QNN_DATATYPE_UFIXED_POINT_16) {
      // Reuse the original DequantizeLinear node's output name (the target MatMul's input[0]) for the
      // FP16 tensor. That tensor is conceptually the dequantized activation — exactly what this
      // INT16→FP16 Dequantize produces — and QNN EP otherwise skips it, so the name is free and keeps
      // the QNN graph aligned with the ONNX graph naming.
      fp16_name = Ort::ConstNode(&node_unit.GetNode()).GetInputs()[0].GetName();
      QnnTensorWrapper fp16_act_wrapper(fp16_name, QNN_TENSOR_TYPE_NATIVE,
                                        QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                        std::vector<uint32_t>(act_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fp16_act_wrapper)),
                    "Failed to add FP16 activation tensor for BQ MatMul.");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        utils::UniqueNameGenerator().New(act_name, "_int16_dequantize"),
                        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_DEQUANTIZE,
                        {act_name}, {fp16_name}, {}, do_op_validation),
                    "Failed to add INT16→FP16 Dequantize node for BQ MatMul activation.");
    }

    // Reshape the FP16 activation [..., M, K] to 4-D [batch, 1, M, K] for the QNN HTP BQ MatMul.
    const uint32_t k_dim = act_shape.back();
    const uint32_t m_dim = act_shape[act_shape.size() - 2];
    uint32_t batch = 1u;
    for (size_t i = 0; i + 2 < act_shape.size(); ++i) {
      batch *= act_shape[i];
    }
    const std::vector<uint32_t> act_shape_4d = {batch, 1u, m_dim, k_dim};
    const std::string act_4d_name = utils::UniqueNameGenerator().New(fp16_name, "_reshape_4d");
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(fp16_name, act_4d_name, act_shape, act_shape_4d,
                                                     QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                                     do_op_validation,
                                                     /*is_for_input=*/false, /*is_for_output=*/false));
    input_names[0] = act_4d_name;
  }

  //
  // Input 1: weight. Build QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK quant params on the 2-D weight
  // [K, N], blocked along the contraction axis K.
  //
  const std::string& input1_name = inputs[1].name;

  // Determine num_blocks/block_size from the ONNX scale shape [K/block_size, N].
  const auto scale_shape = utils::GetInitializerShape(inputs[1].quant_param->scale, qnn_model_wrapper.GetOrtApi());
  RETURN_IF_NOT(scale_shape.size() == 2, "QNN EP: BQ MatMul scale must be rank-2 [K/block_size, N]");
  const int64_t num_blocks = scale_shape[0];
  RETURN_IF(num_blocks <= 0 || K % num_blocks != 0, "QNN EP: BQ MatMul K must be divisible by num_blocks");
  const int64_t block_size = K / num_blocks;
  const uint32_t bitwidth = GetBQBitwidth(inputs[1].type);

  // Unpack the weight to one byte per element (sub-byte INT2/INT4 expanded to INT8).
  std::vector<uint8_t> unpacked_tensor;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_info_1.initializer_tensor, unpacked_tensor));

  // For unsigned types (UINT2/UINT4/UINT8), shift weight data to the signed domain. QNN BW_FLOAT_BLOCK
  // only supports SFIXED_POINT_8 (signed); unsigned data must be converted (see conv_op_builder.cc).
  const bool is_unsigned_weight = (inputs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2 ||
                                   inputs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 ||
                                   inputs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
  if (is_unsigned_weight) {
    RETURN_IF_ERROR(utils::TransformUnsignedToSignedFixedPoint(unpacked_tensor, static_cast<int64_t>(bitwidth)));
  }

  // QNN HTP requires a BQ MatMul to be expressed with 4-D activation, 4-D weight, and a 4-D blockSize.
  // The weight [K, N] is reshaped to [1, 1, K, N]; with transpose_in1 = 0 the contraction axis K is
  // axis 2, so blockSize is {1, 1, block_size, 1}.
  const std::vector<uint32_t> block_size_arr = {1u, 1u, static_cast<uint32_t>(block_size), 1u};

  // ONNX per-block float scales are laid out [num_blocks, N] (block-major). QNN expects the
  // scale/offset array ordered output-channel-major with the output channel (N) as the last weight
  // axis and the block index inner — i.e. [N, num_blocks]. Transpose from [num_blocks, N] → [N, nb].
  std::vector<float> onnx_scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(inputs[1].quant_param->scale, onnx_scales));
  RETURN_IF_NOT(static_cast<int64_t>(onnx_scales.size()) == num_blocks * N,
                "QNN EP: BQ MatMul scale size mismatch");

  // Float offsets in ONNX [num_blocks, N] order before transpose. Matches the Conv BQ formula:
  //   offsets_qnn[idx] = unsigned_bias - zp_values[idx]
  // where zp_values come from UnpackZeroPoints and unsigned_bias = (1 << (bits-1)) for unsigned weights
  // (compensating for the unsigned→signed shift above), 0 for signed weights.
  const float unsigned_bias = is_unsigned_weight ? static_cast<float>(1u << (bitwidth - 1)) : 0.0f;
  std::vector<float> onnx_offsets(static_cast<size_t>(num_blocks * N), unsigned_bias);
  if (inputs[1].quant_param->zero_point != nullptr) {
    std::vector<int32_t> zp_values;
    ONNXTensorElementDataType zp_onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(inputs[1].quant_param->zero_point, zp_values, zp_onnx_type));
    RETURN_IF_NOT(static_cast<int64_t>(zp_values.size()) == num_blocks * N,
                  "QNN EP: BQ MatMul zero_point size must match num_blocks * N");
    for (size_t idx = 0; idx < zp_values.size(); ++idx) {
      onnx_offsets[idx] = unsigned_bias - static_cast<float>(zp_values[idx]);
    }
  }

  // Transpose scales/offsets [num_blocks, N] → [N, num_blocks].
  std::vector<float> scales_qnn(static_cast<size_t>(N * num_blocks));
  std::vector<float> offsets_qnn(static_cast<size_t>(N * num_blocks));
  for (int64_t b = 0; b < num_blocks; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      const size_t src = static_cast<size_t>(b * N + n);
      const size_t dst = static_cast<size_t>(n * num_blocks + b);
      scales_qnn[dst] = onnx_scales[src];
      offsets_qnn[dst] = onnx_offsets[src];
    }
  }

  QnnQuantParamsWrapper bq_quant_params(gsl::span<const float>(scales_qnn),
                                        gsl::span<const float>(offsets_qnn),
                                        bitwidth,
                                        gsl::span<const uint32_t>(block_size_arr));

  // Always use SFIXED_POINT_8: unsigned types are pre-converted by TransformUnsignedToSignedFixedPoint.
  // Weight is reshaped to 4-D [1, 1, K, N] to satisfy the QNN HTP BQ MatMul requirement.
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input1_name);
  std::vector<uint32_t> weight_shape = {1u, 1u, static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
  QnnTensorWrapper bq_weight_wrapper(input1_name, tensor_type,
                                     QNN_DATATYPE_SFIXED_POINT_8,
                                     std::move(bq_quant_params),
                                     std::move(weight_shape),
                                     std::move(unpacked_tensor));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bq_weight_wrapper)),
                "Failed to add BQ MatMul weight tensor.");
  input_names.push_back(input1_name);

  return Ort::Status();
}

Ort::Status MatMulOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& /*logger*/, bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  TensorInfo input_info_0{};
  TensorInfo input_info_1{};
  bool use_fully_connected = false;
  RETURN_IF_ERROR(
      CheckInputs(qnn_model_wrapper, inputs[0], inputs[1], input_info_0, input_info_1, use_fully_connected));

  // A block-quantized weight is always emitted as a QNN MatMul (see ProcessInputsForBQMatMul), even
  // when CheckInputs would otherwise route a rank-2 initializer weight to FullyConnected. Force the
  // MatMul path here so the output handling matches how the inputs were built.
  int64_t bq_num_blocks = 0;
  int64_t bq_block_size = 0;
  if (IsBQWeight(qnn_model_wrapper, inputs[1], bq_num_blocks, bq_block_size)) {
    use_fully_connected = false;
  }

  bool reshape_input_0 = input_info_0.shape.size() == 1;
  bool reshape_input_1 = input_info_1.shape.size() == 1;
  bool reshape_output = reshape_input_0 || reshape_input_1 || (use_fully_connected && input_info_0.shape.size() > 2);

  // For QNN MatMul: set the input transpose parameters to their default values of 0. These parameters should be
  // optional, but older versions of QNN SDK failed validation if not explicitly provided.
  std::vector<std::string> param_tensor_names;
  if (!use_fully_connected) {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 0;
    QnnParamWrapper transpose_in0_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,
                                        scalar_param);
    param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in0_param));

    QnnParamWrapper transpose_in1_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                                        scalar_param);
    param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in1_param));
  }

  const std::string& org_output_name = node_unit.Outputs()[0].name;
  std::string op_output_name = org_output_name;
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  std::vector<uint32_t> op_output_shape = output_info.shape;
  QnnQuantParamsWrapper op_output_quant_param = output_info.quant_param.Copy();
  if (reshape_output) {
    op_output_name = utils::UniqueNameGenerator().New(org_output_name, "_reshape");
    if (use_fully_connected && input_info_0.shape.size() > 2) {
      uint32_t batch = 0;
      RETURN_IF_ERROR(FlattenLeadingDims(input_info_0.shape, batch));
      op_output_shape = {batch, reshape_input_1 ? 1 : input_info_1.shape.back()};
      RETURN_IF(op_output_quant_param.IsPerChannel(), "QNN MatMul output does not support per-channel quant.");
    } else {
      // If both inputs are 1D tensors, the output shape is [1] instead of scalar. So if both inputs are 1D tensors,
      // we only need to add one "1" to the op_output_shape.
      if (reshape_input_1) {
        op_output_shape.emplace_back(1);
      } else if (reshape_input_0) {
        op_output_shape.insert(op_output_shape.end() - 1, 1);
      }
      RETURN_IF_ERROR(op_output_quant_param.HandleUnsqueeze<uint32_t>(output_info.shape, op_output_shape));
    }
  }

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
  const bool is_op_output_graph_output = is_graph_output && !reshape_output;
  Qnn_TensorType_t op_output_tensor_type =
      is_op_output_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  // Detect a BQ (BW_FLOAT_BLOCK) MatMul from the weight tensor's quant encoding. input_names[1] is
  // the weight (BQ MatMul always has exactly 2 inputs and is never reshaped). A BQ MatMul computes
  // in FP16, so it must output FP16 and then re-quantize to the INT16 the downstream QDQ expects.
  bool is_bq_matmul = false;
  if (!use_fully_connected && input_names.size() > 1 &&
      qnn_model_wrapper.IsQnnTensorWrapperExist(input_names[1])) {
    is_bq_matmul = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]).GetQnnQuantParams().IsBlockQuantized();
  }

  if (is_bq_matmul && output_info.quant_param.IsQuantized()) {
    // The QNN HTP BQ MatMul runs on 4-D tensors and outputs FP16. The ONNX output is INT16-quantized,
    // so the pipeline is: MatMul (4-D FP16 [batch,1,M,N]) → Reshape (to ONNX [...,M,N] FP16) → Quantize
    // (FP16 → INT16). The reshape target reuses the original QuantizeLinear node's input name (the
    // un-quantized MatMul output), keeping the QNN graph aligned with the ONNX graph naming.
    const uint32_t n_dim = op_output_shape.back();
    const uint32_t m_dim = op_output_shape[op_output_shape.size() - 2];
    uint32_t batch = 1u;
    for (size_t i = 0; i + 2 < op_output_shape.size(); ++i) {
      batch *= op_output_shape[i];
    }
    const std::vector<uint32_t> matmul_out_shape_4d = {batch, 1u, m_dim, n_dim};

    const std::string matmul_4d_out = utils::UniqueNameGenerator().New(op_output_name, "_matmul_4d");
    QnnTensorWrapper matmul_4d_wrapper(matmul_4d_out, QNN_TENSOR_TYPE_NATIVE,
                                       QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>(matmul_out_shape_4d));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(matmul_4d_wrapper)),
                  "Failed to add 4-D FP16 BQ MatMul output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                                                  std::move(input_names), {matmul_4d_out},
                                                  std::move(param_tensor_names), do_op_validation),
                  "Failed to add BQ MatMul node.");

    // Reshape 4-D FP16 [batch,1,M,N] back to the ONNX FP16 output shape [...,M,N].
    const std::string matmul_fp16_out = Ort::ConstNode(&node_unit.GetNode()).GetOutputs()[0].GetName();
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(matmul_4d_out, matmul_fp16_out, matmul_out_shape_4d,
                                                     op_output_shape, QNN_DATATYPE_FLOAT_16,
                                                     QnnQuantParamsWrapper(), do_op_validation,
                                                     /*is_for_input=*/false, /*is_for_output=*/false));

    // INT16 quantized output tensor consumed by downstream nodes (or the graph output).
    QnnTensorWrapper int16_out_wrapper(op_output_name, op_output_tensor_type, output_info.qnn_data_type,
                                       op_output_quant_param.Copy(), std::vector<uint32_t>(op_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(int16_out_wrapper)),
                  "Failed to add INT16 BQ MatMul output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(op_output_name, "_fp16_quantize"),
                      QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_QUANTIZE,
                      {matmul_fp16_out}, {op_output_name}, {}, do_op_validation),
                  "Failed to add FP16→INT16 Quantize node for BQ MatMul output.");
  } else {
    QnnTensorWrapper op_output_tensor_wrapper(op_output_name, op_output_tensor_type, output_info.qnn_data_type,
                                              op_output_quant_param.Copy(), std::vector<uint32_t>(op_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(op_output_tensor_wrapper)),
                  "Failed to add output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit), QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  use_fully_connected ? QNN_OP_FULLY_CONNECTED : QNN_OP_MAT_MUL,
                                                  std::move(input_names), {op_output_name},
                                                  std::move(param_tensor_names), do_op_validation),
                  "Failed to add fused Matmul node.");
  }

  if (reshape_output) {
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        op_output_name, org_output_name, op_output_shape, output_info.shape, output_info.qnn_data_type,
        op_output_quant_param, output_info.quant_param, do_op_validation, false, is_graph_output));
  }

  return Ort::Status();
}

void CreateMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MatMulOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
