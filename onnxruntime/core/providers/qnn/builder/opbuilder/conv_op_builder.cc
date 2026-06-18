// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gsl/gsl>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {
// HTP BQ Conv: supported bitwidths and their block_size divisor constraints.
// block_size must be a multiple of the corresponding value (same as MatMulNBits HTP constraints).
const std::unordered_map<uint32_t, int64_t> kHtpConvBQBitsAndBlockSizeMultipliers{
    {2, 16}, {4, 8}, {8, 4}};

// Returns BQ weight bitwidth (2/4/8) from an ONNX element data type, or 0 if unsupported.
static uint32_t GetBQBitwidth(ONNXTensorElementDataType onnx_type) {
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
}  // namespace

// ONNX convolution types supported by this builder.
// We translate node_unit.OpType() into this enum to avoid repeated string comparisons.
enum class OnnxConvType {
  kConv,
  kConvTranspose,
};

static Ort::Status GetOnnxConvType(const std::string& onnx_op_type, OnnxConvType& conv_type) {
  if (onnx_op_type == "Conv") {
    conv_type = OnnxConvType::kConv;
  } else if (onnx_op_type == "ConvTranspose") {
    conv_type = OnnxConvType::kConvTranspose;
  } else {
    return MAKE_EP_FAIL(("QNN EP: Unsupported ONNX convolution op type: " + onnx_op_type).c_str());
  }
  return Ort::Status();
}

class ConvOpBuilder : public BaseOpBuilder {
 public:
  ConvOpBuilder() : BaseOpBuilder("ConvOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;
  Ort::Status ProcessConv1DInputs(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const Ort::Logger& logger,
                                  std::vector<std::string>& input_names,
                                  bool do_op_validation) const ORT_MUST_USE_RESULT;
  Ort::Status ProcessConv2D3DInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const Ort::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const ORT_MUST_USE_RESULT;
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status GetInputChannelNumber(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    uint32_t& input_channel_number) const;
};

// Conv/ConvTranspose ops are sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
// Need to do op validation in 1st call of GetCapability
Ort::Status ConvOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger) const {
  if (node_unit.Domain() == kMSInternalNHWCDomain) {  // Use QNN validation API if layout is NHWC.
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  const auto& inputs = node_unit.Inputs();
  RETURN_IF(inputs.size() < 2, "QNN Conv must have at least 2 inputs.");

  const auto& input_0 = inputs[0];
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.shape, input_shape), "Cannot get shape");
  if (input_shape.size() != 5 && input_shape.size() != 4 && input_shape.size() != 3) {
    return MAKE_EP_FAIL("QNN Conv only supports 3D(rank 5), 2D (rank 4) or 1D (rank 3) inputs.");
  }

  ONNXTensorElementDataType input_data_type = input_0.type;
  std::string error_msg = "QNN EP: Data type " + std::to_string(static_cast<int>(input_data_type)) +
                          " is not supported for Conv operator in CPU backend.";
  RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, input_data_type, error_msg));

  OrtNodeAttrHelper node_helper(node_unit);
  auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER" && auto_pad != "VALID",
            ("QNN Conv operators do not support 'auto_pad' value: " + auto_pad).c_str());

  OnnxConvType conv_type = {};
  RETURN_IF_ERROR(GetOnnxConvType(node_unit.OpType(), conv_type));

  if (conv_type == OnnxConvType::kConvTranspose) {
    // QNN's TransposeConv2d only supports default dilation values of 1.
    constexpr int32_t default_dilation = 1;
    auto dilations = node_helper.Get("dilations", std::vector<int32_t>{default_dilation, default_dilation});

    for (auto dilation : dilations) {
      RETURN_IF(dilation != default_dilation,
                "QNN EP: QNN's TransposeConv2d operator only supports default dilation values of 1.");
    }
  }

  // Validate quantization axis for per-channel quantized weights.
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    // Detect block-quantized weight. Per ONNX opset 21, the scale rank equals the weight
    // rank: for a Conv2D weight [OC, IC, H, W] blocked on axis=1 the scale is
    // [OC, IC/block_size, H, W], so scale_shape[1] == num_blocks_per_oc < weight_shape[1].
    if (inputs[1].quant_param.has_value() && inputs[1].quant_param->scale != nullptr) {
      const auto weight_scale_shape = utils::GetInitializerShape(inputs[1].quant_param->scale,
                                                                 qnn_model_wrapper.GetOrtApi());
      std::vector<uint32_t> weight_shape;
      if (!qnn_model_wrapper.GetOnnxShape(inputs[1].shape, weight_shape) ||
          weight_shape.size() != 4) {
        // BQ only supported for Conv2D (rank-4 weight); skip for Conv1D (rank-3) and Conv3D (rank-5).
      } else {
        const bool is_block_quant = (weight_scale_shape.size() == weight_shape.size() &&
                                     weight_scale_shape.size() > 1 &&
                                     weight_scale_shape[1] < weight_shape[1]);
        if (is_block_quant) {
          const int64_t num_blocks_per_oc = weight_scale_shape[1];
          RETURN_IF(num_blocks_per_oc <= 0, "QNN EP: BQ num_blocks must be positive");
          const int64_t ic = static_cast<int64_t>(weight_shape[1]);
          RETURN_IF(ic % num_blocks_per_oc != 0,
                    "QNN EP: Conv BQ: IC must be divisible by num_blocks_per_oc");

          // Validate bitwidth (from weight element type) and block_size against HTP constraints.
          const uint32_t bitwidth = GetBQBitwidth(inputs[1].type);
          const int64_t block_size = ic / num_blocks_per_oc;
          auto bq_it = kHtpConvBQBitsAndBlockSizeMultipliers.find(bitwidth);
          RETURN_IF(bq_it == kHtpConvBQBitsAndBlockSizeMultipliers.end(),
                    ("QNN HTP Conv BQ: unsupported weight bitwidth=" +
                     std::to_string(bitwidth))
                        .c_str());
          RETURN_IF(block_size % bq_it->second != 0,
                    ("QNN HTP Conv BQ: block_size=" + std::to_string(block_size) +
                     " must be a multiple of " + std::to_string(bq_it->second) +
                     " for " + std::to_string(bitwidth) + "-bit weight")
                        .c_str());

          // Return success; full QNN validation is done in the NHWC IsOpSupported path.
          return Ort::Status();
        }  // end is_block_quant
      }  // end else (weight shape obtainable)
    }  // end quant_param check
    // checking for per-channel quantization
    const auto& input_1 = inputs[1];  // weight
    bool is_per_axis_quant = false;
    int64_t quant_axis = 0;
    RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(input_1, is_per_axis_quant, quant_axis));

    if (is_per_axis_quant) {
      if (conv_type == OnnxConvType::kConvTranspose) {
        RETURN_IF_NOT(quant_axis == 1,
                      "ConvTranspose's input[1] must be use axis == 1 for per-channel quantization");
      } else {
        RETURN_IF_NOT(quant_axis == 0, "Conv's input[1] must be use axis == 0 for per-channel quantization");
      }
    }
  }

  return Ort::Status();
}

Ort::Status ConvOpBuilder::GetInputChannelNumber(QnnModelWrapper& qnn_model_wrapper,
                                                 const OrtNodeUnit& node_unit,
                                                 uint32_t& input_channel_number) const {
  const auto& input_0 = node_unit.Inputs()[0];
  input_channel_number = 0;
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.shape, input_shape), "Cannot get shape");
  // Conv input 0 is NHWC layout now, get the channel data from the last dim.
  input_channel_number = input_shape.back();

  return Ort::Status();
}

Ort::Status ConvOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  assert(inputs.size() >= 2);

  std::vector<uint32_t> input0_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input0_shape),
                "QNN EP: Cannot get shape for first input");

  if (input0_shape.size() == 3) {
    return ProcessConv1DInputs(qnn_model_wrapper, node_unit, logger, input_names, do_op_validation);
  } else if (input0_shape.size() == 4 || input0_shape.size() == 5) {
    return ProcessConv2D3DInputs(qnn_model_wrapper, node_unit, logger, input_names, do_op_validation);
  }

  return MAKE_EP_FAIL("QNN Conv only supports 3D(rank 5), 2D (rank 4) or 1D (rank 3) inputs.");
}

Ort::Status ConvOpBuilder::ProcessConv2D3DInputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const OrtNodeUnit& node_unit,
                                                 const Ort::Logger& logger,
                                                 std::vector<std::string>& input_names,
                                                 bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();
  OnnxConvType conv_type = {};
  RETURN_IF_ERROR(GetOnnxConvType(node_unit.OpType(), conv_type));

  assert(num_inputs >= 2);  // Checked by IsOpSupported.

  //
  // Input 0
  //
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  // Detect block-quantized weight. Per ONNX opset 21, the scale rank equals the weight rank
  // with scale_shape[1] < weight_shape[1] (the blocked IC axis). Weight is always NCHW
  // [OC,IC,H,W] in both NCHW and NHWC domains; IC at index 1, so scale_shape[1] == num_blocks_per_oc.
  bool is_bq_weight = false;
  std::vector<int64_t> bq_scale_shape;
  if (IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) &&
      inputs[1].quant_param.has_value() && inputs[1].quant_param->scale != nullptr) {
    bq_scale_shape = utils::GetInitializerShape(inputs[1].quant_param->scale,
                                                qnn_model_wrapper.GetOrtApi());
    std::vector<uint32_t> bq_weight_shape;
    if (qnn_model_wrapper.GetOnnxShape(inputs[1].shape, bq_weight_shape) &&
        bq_scale_shape.size() > 1 &&
        bq_weight_shape.size() == 4) {  // BQ only for Conv2D (rank-4 weight); not Conv3D (rank-5)
      // Weight is NCHW [OC,IC,H,W]: IC is always at index 1.
      is_bq_weight = (bq_scale_shape.size() == bq_weight_shape.size() &&
                      bq_scale_shape[1] < static_cast<int64_t>(bq_weight_shape[1]));
    }
  }

  if (is_bq_weight) {
    // Determine BQ vs LPBQ by reading weight quant params via GetTensorInfo.
    // When PR307 is active, Init() auto-converts BQ scales → LPBQ (BLOCKWISE_EXPANSION).
    // Before PR307: Init() cannot handle block_size, so IsLPBQ() is false → BQ path.
    const std::string& input1_name = inputs[1].name;
    TensorInfo input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input_info));
    RETURN_IF(!input_info.is_initializer,
              "QNN EP: BQ/LPBQ Conv weight must be a constant initializer");

    const bool is_lpbq = input_info.quant_param.IsLPBQ();

    // Activation handling: BQ kernel (BW_FLOAT_BLOCK) requires FP16, so INT16 → FP16 via Dequantize.
    //                      LPBQ kernel (BLOCKWISE_EXPANSION) accepts INT16 as-is.
    if (!is_lpbq) {
      const std::string act_name = input_names[0];  // activation (input 0), pushed by ProcessInput above
      const auto& act_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(act_name);
      const Qnn_DataType_t act_dtype = act_wrapper.GetTensorDataType();
      if (act_dtype == QNN_DATATYPE_SFIXED_POINT_16 || act_dtype == QNN_DATATYPE_UFIXED_POINT_16) {
        // Reuse the original DequantizeLinear node's output name (the target Conv's input[0]) for the
        // FP16 tensor. That tensor is conceptually the dequantized activation — exactly what this
        // INT16→FP16 Dequantize produces — and QNN EP otherwise skips it, so the name is free and
        // keeps the QNN graph aligned with the ONNX graph naming.
        const std::string fp16_act_name = Ort::ConstNode(&node_unit.GetNode()).GetInputs()[0].GetName();
        std::vector<uint32_t> act_shape = act_wrapper.GetTensorDims();
        QnnTensorWrapper fp16_act_wrapper(fp16_act_name, QNN_TENSOR_TYPE_NATIVE,
                                          QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                          std::vector<uint32_t>(act_shape));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fp16_act_wrapper)),
                      "Failed to add FP16 activation tensor for BQ Conv.");
        RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::UniqueNameGenerator().New(act_name, "_int16_dequantize"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_DEQUANTIZE,
                          {act_name}, {fp16_act_name}, {}, do_op_validation),
                      "Failed to add INT16→FP16 Dequantize node for BQ Conv activation.");
        input_names[0] = fp16_act_name;
      }
    }

    // Common: transpose weight data from OIHW→HWIO (or IOHW→HWIO for ConvTranspose).
    // TransposeFromNchwToHwcn unpacks INT4 to INT8 internally (1 byte per element).
    const bool is_3d = (input_info.shape.size() == 5);
    std::vector<uint8_t> unpacked_tensor;
    if (conv_type == OnnxConvType::kConv) {
      RETURN_IF_ERROR(utils::TransposeFromNchwToHwcn(qnn_model_wrapper,
                                                     input_info.initializer_tensor,
                                                     unpacked_tensor, is_3d));
    } else {
      RETURN_IF_ERROR(utils::TransposeFromCnhwToHwcn(qnn_model_wrapper,
                                                     input_info.initializer_tensor,
                                                     unpacked_tensor, is_3d));
    }
    std::vector<uint32_t> hwcn_shape(input_info.shape.size());
    if (conv_type == OnnxConvType::kConv) {
      RETURN_IF_ERROR(utils::NchwShapeToHwcn<uint32_t>(input_info.shape, hwcn_shape));
    } else {
      RETURN_IF_ERROR(utils::CnhwShapeToHwcn<uint32_t>(input_info.shape, hwcn_shape));
    }
    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input1_name);

    if (is_lpbq) {
      // LPBQ path (PR307 active): Init() already produced BLOCKWISE_EXPANSION quant params.
      // Update the LPBQ axis for the HWCN transposition (PR307 extended HandleTranspose for LPBQ).
      std::vector<size_t> perm;
      if (is_3d) {
        perm = conv_type == OnnxConvType::kConv ? nchw2hwcn_perm_3d : cnhw2hwcn_perm_3d;
      } else {
        perm = conv_type == OnnxConvType::kConv ? nchw2hwcn_perm : cnhw2hwcn_perm;
      }
      std::vector<size_t> perm_inv(perm.size());
      RETURN_IF_ERROR(utils::InvertPerm<size_t>(perm, perm_inv));
      RETURN_IF_ERROR(input_info.quant_param.HandleTranspose<size_t>(perm_inv));

      QnnTensorWrapper lpbq_weight_wrapper(input1_name, tensor_type,
                                           input_info.qnn_data_type,
                                           std::move(input_info.quant_param),
                                           std::move(hwcn_shape),
                                           std::move(unpacked_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(lpbq_weight_wrapper)),
                    "Failed to add LPBQ Conv weight tensor.");
    } else {
      // BQ path: manually build QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK quant params.
      const int64_t OC = static_cast<int64_t>(input_info.shape[0]);
      const int64_t IC = static_cast<int64_t>(input_info.shape[1]);
      const int64_t nb = bq_scale_shape[1];  // num_blocks_per_oc (validated: IC % nb == 0)
      const int64_t block_size = IC / nb;
      const uint32_t bitwidth = GetBQBitwidth(inputs[1].type);

      // For unsigned types (UINT2/UINT4/UINT8), shift weight data to the signed domain.
      // QNN BW_FLOAT_BLOCK only supports SFIXED_POINT_8 (signed); unsigned data must be converted.
      // TransformUnsignedToSignedFixedPoint XORs the MSB of each packed element:
      //   - UINT8 (bitwidth=8, mask=0x80): 1 byte/element — directly converts [0,255] → [-128,127].
      //   - UINT4 (bitwidth=4, mask=0x88): after TransposeFromNchwToHwcn unpack, each byte holds
      //     a UINT4 value in its lower nibble (upper nibble=0). XOR with 0x88 flips bits 3 and 7;
      //     QNN reads only the lower nibble for bitwidth=4, so the result is equivalent to
      //     flipping bit 3, which shifts [0,15] → signed-nibble-interpreted [-8,7] correctly.
      const bool is_unsigned_weight = (inputs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2 ||
                                       inputs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 ||
                                       inputs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      if (is_unsigned_weight) {
        RETURN_IF_ERROR(utils::TransformUnsignedToSignedFixedPoint(unpacked_tensor,
                                                                   static_cast<int64_t>(bitwidth)));
      }

      // QNN BW_FLOAT_BLOCK blockSize for HWCN weight: {1, 1, block_size, 1}.
      // Each block spans block_size consecutive IC elements; H and W dimensions are not blocked.
      const std::vector<uint32_t> block_size_arr = {1u, 1u, static_cast<uint32_t>(block_size), 1u};

      // Read ONNX per-block float scales: flat [OC * nb] in OC-major order.
      std::vector<float> scales_qnn;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(inputs[1].quant_param->scale, scales_qnn));
      RETURN_IF_NOT(static_cast<int64_t>(scales_qnn.size()) == OC * nb,
                    "QNN EP: BQ Conv scale size mismatch");
      // QNN BW_FLOAT_BLOCK expects scales in [OC, nb] order — same as ONNX, no reordering needed.

      // Float offsets in [OC, nb] order.
      // Signed:   offset = -onnx_zp
      // Unsigned: offset = (1 << (bits-1)) - onnx_zp   (compensates for unsigned→signed shift above)
      const float unsigned_bias = is_unsigned_weight ? static_cast<float>(1u << (bitwidth - 1)) : 0.0f;
      std::vector<float> offsets_qnn(static_cast<size_t>(OC * nb), unsigned_bias);
      if (inputs[1].quant_param->zero_point != nullptr) {
        std::vector<int32_t> zp_values;
        ONNXTensorElementDataType zp_onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(inputs[1].quant_param->zero_point,
                                                           zp_values, zp_onnx_type));
        RETURN_IF_NOT(static_cast<int64_t>(zp_values.size()) == OC * nb,
                      "QNN EP: BQ Conv zero_point size must match OC * num_blocks");
        for (size_t idx = 0; idx < static_cast<size_t>(OC * nb); ++idx) {
          offsets_qnn[idx] = unsigned_bias - static_cast<float>(zp_values[idx]);
        }
      }

      QnnQuantParamsWrapper bq_quant_params(gsl::span<const float>(scales_qnn),
                                            gsl::span<const float>(offsets_qnn),
                                            bitwidth,
                                            gsl::span<const uint32_t>(block_size_arr));

      // Always use SFIXED_POINT_8: unsigned types are pre-converted by TransformUnsignedToSignedFixedPoint.
      QnnTensorWrapper bq_weight_wrapper(input1_name, tensor_type,
                                         QNN_DATATYPE_SFIXED_POINT_8,
                                         std::move(bq_quant_params),
                                         std::move(hwcn_shape),
                                         std::move(unpacked_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bq_weight_wrapper)),
                    "Failed to add BQ Conv weight tensor.");
    }
    input_names.push_back(input1_name);
  } else {
    //
    // Input 1: weight. This input must be transposed manually by QNN EP.
    //
    {
      const std::string& input1_name = inputs[1].name;
      TensorInfo input_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input_info));

      std::string actual_name = input_info.is_initializer ? input1_name : utils::UniqueNameGenerator().New(input1_name, "_transpose");
      input_names.push_back(actual_name);

      std::vector<uint32_t> actual_shape;
      actual_shape.resize(input_info.shape.size());

      // Change shape to HWCN, it could be initializer or normal input
      if (conv_type == OnnxConvType::kConv) {
        RETURN_IF_ERROR(utils::NchwShapeToHwcn<uint32_t>(input_info.shape, actual_shape));
      } else if (conv_type == OnnxConvType::kConvTranspose) {
        RETURN_IF_ERROR(utils::CnhwShapeToHwcn<uint32_t>(input_info.shape, actual_shape));
      } else {
        return MAKE_EP_FAIL(("QNN EP: Unexpected convolution op type: " + node_unit.OpType()).c_str());
      }

      bool is_3d = (input_info.shape.size() == 5);

      std::vector<uint8_t> unpacked_tensor;
      if (input_info.is_initializer) {
        // Get transposed initializer bytes.
        if (conv_type == OnnxConvType::kConv) {
          RETURN_IF_ERROR(utils::TransposeFromNchwToHwcn(qnn_model_wrapper, input_info.initializer_tensor, unpacked_tensor, is_3d));
        } else if (conv_type == OnnxConvType::kConvTranspose) {
          RETURN_IF_ERROR(utils::TransposeFromCnhwToHwcn(qnn_model_wrapper, input_info.initializer_tensor, unpacked_tensor, is_3d));
        } else {
          return MAKE_EP_FAIL(("QNN EP: Unexpected convolution op type: " + node_unit.OpType()).c_str());
        }

        // Transpose quantization parameter's axis if this is using per-channel quantization.
        if (input_info.quant_param.IsPerChannel()) {
          std::vector<size_t> perm;
          if (is_3d) {
            perm = conv_type == OnnxConvType::kConv ? nchw2hwcn_perm_3d : cnhw2hwcn_perm_3d;
          } else {
            perm = conv_type == OnnxConvType::kConv ? nchw2hwcn_perm : cnhw2hwcn_perm;
          }
          std::vector<size_t> perm_inv(perm.size());
          RETURN_IF_ERROR(utils::InvertPerm<size_t>(perm, perm_inv));
          RETURN_IF_ERROR(input_info.quant_param.HandleTranspose<size_t>(perm_inv));
        }
      } else {
        // Add transpose node above weight input.
        RETURN_IF(input_info.quant_param.IsPerChannel(),
                  "Non-constant Conv inputs only support per-tensor quantization");
        bool is_graph_input = qnn_model_wrapper.IsGraphInput(input1_name);
        ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Add HWCN Transpose node after input: " + input1_name).c_str());

        if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input1_name)) {
          QnnTensorWrapper weight_tensor_wrapper;
          RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[1], weight_tensor_wrapper));
          RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor_wrapper)), "Failed to add weight tensor.");
        }

        if (conv_type == OnnxConvType::kConv) {
          RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(node_unit.Index(),
                                                                   input1_name,
                                                                   actual_name,
                                                                   input_info.shape,
                                                                   actual_shape,
                                                                   input_info.qnn_data_type,
                                                                   input_info.quant_param,
                                                                   do_op_validation,
                                                                   is_graph_input,
                                                                   false,
                                                                   is_3d));
        } else if (conv_type == OnnxConvType::kConvTranspose) {
          RETURN_IF_ERROR(qnn_model_wrapper.AddCnhwToHwcnTranspose(node_unit.Index(),
                                                                   input1_name,
                                                                   actual_name,
                                                                   input_info.shape,
                                                                   actual_shape,
                                                                   input_info.qnn_data_type,
                                                                   input_info.quant_param,
                                                                   do_op_validation,
                                                                   is_graph_input,
                                                                   false,
                                                                   is_3d));
        } else {
          return MAKE_EP_FAIL(("QNN EP: Unexpected convolution op type: " + node_unit.OpType()).c_str());
        }
      }

      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(actual_name);
      QnnTensorWrapper input_tensorwrapper(actual_name, tensor_type, input_info.qnn_data_type,
                                           std::move(input_info.quant_param),
                                           std::move(actual_shape), std::move(unpacked_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");

      // Workaround that inserts a QNN Convert op before input[1] (converts from quantized uint16 to signed symmetric int16)
      // to avoid a QNN validation failure.
      //
      // QNN graph WITHOUT workaround (fails validation):
      //     input_0_uint16 ---> Conv ---> output_uint16
      //                         ^
      //                         |
      //     input_1_uint16 -----+
      //
      // QNN graph WITH workaround (passes validation):
      //     input_0_uint16 ----------------------> Conv ---> output_uint16
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
                      "Conv's INT16 weight inputs only support INT16 per-tensor quantization");

        // Insert Convert op after Weight, replacing the weight name (last element) in place.
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
        input_names.back() = convert_output_name;
      }
    }
  }  // end else (non-BQ weight path)

  //
  // Input 2: bias
  //
  const bool has_bias_input = num_inputs == 3;
  if (has_bias_input) {
    // For BQ Conv: QNN BW_FLOAT_BLOCK Conv2d requires FP16 bias matching the computation precision.
    // The bias in the QDQ NodeUnit is an INT32 quantized initializer — dequantize it to FP16.
    if (is_bq_weight) {
      TensorInfo bias_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info));
      RETURN_IF(!bias_info.is_initializer, "QNN EP: BQ Conv bias must be a constant initializer");

      // Read the INT32 quantized bias and its scale(s), then dequantize to FP16.
      // QNN BW_FLOAT_BLOCK Conv2d requires FP16 bias matching the computation precision.
      // Bias scale may be per-tensor (1 value) or per-channel (one value per output channel).
      std::vector<uint8_t> raw_bias_bytes;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor,
                                                              raw_bias_bytes));
      std::vector<float> bias_scale_vals;
      if (inputs[2].quant_param.has_value() && inputs[2].quant_param->scale != nullptr) {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(inputs[2].quant_param->scale, bias_scale_vals));
      }

      // Dequantize INT32 → FP16 (stored as uint16 in memory): bias[i] = q[i] * scale[i or 0].
      const size_t num_elems = bias_info.shape[0];
      RETURN_IF_NOT(raw_bias_bytes.size() == num_elems * sizeof(int32_t), "BQ bias size mismatch");
      // Scale is per-tensor (1), per-channel (num_elems == OC), or absent (defaults to 1.0f).
      const bool is_per_channel_bias = bias_scale_vals.size() == num_elems;
      RETURN_IF_NOT(bias_scale_vals.size() <= 1 || is_per_channel_bias,
                    "QNN EP: BQ Conv bias scale count must be 1 (per-tensor) or OC (per-channel)");
      std::vector<uint8_t> fp16_bias_bytes(num_elems * sizeof(uint16_t));
      const auto* i32_ptr = reinterpret_cast<const int32_t*>(raw_bias_bytes.data());
      auto* u16_ptr = reinterpret_cast<uint16_t*>(fp16_bias_bytes.data());
      for (size_t i = 0; i < num_elems; ++i) {
        const float scale = bias_scale_vals.empty() ? 1.0f
                                                    : (is_per_channel_bias ? bias_scale_vals[i]
                                                                           : bias_scale_vals[0]);
        const float f = static_cast<float>(i32_ptr[i]) * scale;
        const Ort::Float16_t fp16_val(f);
        memcpy(&u16_ptr[i], &fp16_val.val, sizeof(uint16_t));
      }

      const std::string fp16_bias_name = utils::UniqueNameGenerator().New(inputs[2].name, "_fp16");
      QnnTensorWrapper fp16_bias_wrapper(fp16_bias_name, QNN_TENSOR_TYPE_STATIC,
                                         QNN_DATATYPE_FLOAT_16,
                                         QnnQuantParamsWrapper(),
                                         std::vector<uint32_t>(bias_info.shape),
                                         std::move(fp16_bias_bytes));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fp16_bias_wrapper)),
                    "Failed to add FP16 bias for BQ Conv.");
      input_names.push_back(fp16_bias_name);
    } else {
      const auto& bias_input = inputs[2];
      TensorInfo bias_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(bias_input, bias_info));

      // For static quantized bias, handle requantization if needed
      if (bias_info.is_initializer && bias_info.quant_param.IsQuantized()) {
        // Get activation and weight quantization parameters
        TensorInfo input0_info = {};
        TensorInfo input1_info = {};
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input0_info));
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input1_info));

        if (input0_info.quant_param.IsQuantized() && input1_info.quant_param.IsQuantized()) {
          // Get activation scale (must be per-tensor for Conv)
          float activation_scale = 1.0f;
          const auto& act_quant_params = input0_info.quant_param.Get();
          if (act_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
            activation_scale = act_quant_params.scaleOffsetEncoding.scale;
          } else if (act_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
            activation_scale = act_quant_params.bwScaleOffsetEncoding.scale;
          }

          // Get weight scales (per-tensor or per-channel)
          std::vector<float> weights_scales;

          if (input1_info.quant_param.IsPerTensor()) {
            // Handle per-tensor quantization (encodings 0 and 2)
            const auto& weight_quant_params = input1_info.quant_param.Get();

            if (weight_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
              weights_scales.push_back(weight_quant_params.scaleOffsetEncoding.scale);
            } else if (weight_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
              weights_scales.push_back(weight_quant_params.bwScaleOffsetEncoding.scale);
            }
          } else {
            // Handle per-channel quantization (encodings 1 and 3)
            const auto& weight_quant_params = input1_info.quant_param.Get();

            if (weight_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
              if (weight_quant_params.axisScaleOffsetEncoding.scaleOffset != nullptr &&
                  weight_quant_params.axisScaleOffsetEncoding.numScaleOffsets > 0) {
                for (size_t i = 0; i < weight_quant_params.axisScaleOffsetEncoding.numScaleOffsets; ++i) {
                  weights_scales.push_back(weight_quant_params.axisScaleOffsetEncoding.scaleOffset[i].scale);
                }
              }
            } else if (weight_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
              if (weight_quant_params.bwAxisScaleOffsetEncoding.scales != nullptr &&
                  weight_quant_params.bwAxisScaleOffsetEncoding.numElements > 0) {
                for (size_t i = 0; i < weight_quant_params.bwAxisScaleOffsetEncoding.numElements; ++i) {
                  weights_scales.push_back(weight_quant_params.bwAxisScaleOffsetEncoding.scales[i]);
                }
              }
            }
          }

          // Safety check to prevent crashes
          RETURN_IF_NOT(!weights_scales.empty(), "No weight scales found for quantized weights");

          // Check bias quantization type
          if (bias_info.quant_param.IsPerTensor()) {
            float bias_scale = 0.0f;
            int32_t bias_offset = 0;
            const auto& bias_quant_params = bias_info.quant_param.Get();
            if (bias_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
              bias_scale = bias_quant_params.scaleOffsetEncoding.scale;
              bias_offset = bias_quant_params.scaleOffsetEncoding.offset;
            } else if (bias_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
              bias_scale = bias_quant_params.bwScaleOffsetEncoding.scale;
              bias_offset = bias_quant_params.bwScaleOffsetEncoding.offset;
            } else {
              return MAKE_EP_FAIL("Unsupported bias quantization encoding for per-tensor quantization.");
            }

            // Check if bias_offset = 0 AND bias_scale = (weights_scale[0] * activation_scale)
            if (bias_offset == 0 && utils::CheckBiasScaleMatch(bias_scale, weights_scales[0], activation_scale, 1e-5f)) {
              // No change needed - scales match and offset is 0
            } else {
              ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Requantizing per-tensor bias " + bias_input.name).c_str());
              // Need to requantize the bias tensor
              std::vector<uint8_t> original_bias_data;
              RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor, original_bias_data));

              std::vector<float> current_scales = {bias_scale};
              std::vector<int32_t> current_offsets = {bias_offset};
              std::vector<uint8_t> requantized_bias_data;
              std::vector<float> new_scales;
              std::vector<int32_t> new_offsets;

              RETURN_IF_ERROR(utils::RequantizeBiasTensor(
                  original_bias_data, bias_info.shape, current_scales, current_offsets,
                  weights_scales, activation_scale, bias_info.qnn_data_type,
                  requantized_bias_data, new_scales, new_offsets));

              // Create new tensor wrapper with requantized data
              std::string bias_name = bias_input.name;
              QnnQuantParamsWrapper new_quant_params(new_scales[0], new_offsets[0]);
              QnnTensorWrapper bias_tensorwrapper(bias_name, QNN_TENSOR_TYPE_STATIC, bias_info.qnn_data_type,
                                                  std::move(new_quant_params), std::vector<uint32_t>(bias_info.shape),
                                                  std::move(requantized_bias_data));
              RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensorwrapper)), "Failed to add requantized bias tensor.");
              input_names.push_back(bias_name);
              return Ort::Status();  // We've handled the bias, return early
            }
          } else {
            // Handle per-channel bias
            const auto& bias_quant_params = bias_info.quant_param.Get();

            if (bias_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET ||
                bias_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
              // Extract scales and offsets based on encoding type
              std::vector<float> current_scales;
              std::vector<int32_t> current_offsets;
              int32_t quant_axis = 0;
              size_t num_channels = 0;

              if (bias_quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
                // Safety checks for AXIS_SCALE_OFFSET encoding
                RETURN_IF_NOT(bias_quant_params.axisScaleOffsetEncoding.scaleOffset != nullptr,
                              "Invalid bias quantization parameters: scaleOffset is null");
                RETURN_IF_NOT(bias_quant_params.axisScaleOffsetEncoding.numScaleOffsets > 0,
                              "Invalid bias quantization parameters: numScaleOffsets is zero");

                num_channels = bias_quant_params.axisScaleOffsetEncoding.numScaleOffsets;
                quant_axis = bias_quant_params.axisScaleOffsetEncoding.axis;
                for (size_t i = 0; i < num_channels; ++i) {
                  current_scales.push_back(bias_quant_params.axisScaleOffsetEncoding.scaleOffset[i].scale);
                  current_offsets.push_back(bias_quant_params.axisScaleOffsetEncoding.scaleOffset[i].offset);
                }
              } else {  // QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET
                // Safety checks for BW_AXIS_SCALE_OFFSET encoding
                RETURN_IF_NOT(bias_quant_params.bwAxisScaleOffsetEncoding.scales != nullptr,
                              "Invalid bias quantization parameters: scales is null");
                RETURN_IF_NOT(bias_quant_params.bwAxisScaleOffsetEncoding.offsets != nullptr,
                              "Invalid bias quantization parameters: offsets is null");
                RETURN_IF_NOT(bias_quant_params.bwAxisScaleOffsetEncoding.numElements > 0,
                              "Invalid bias quantization parameters: numElements is zero");

                num_channels = bias_quant_params.bwAxisScaleOffsetEncoding.numElements;
                quant_axis = bias_quant_params.bwAxisScaleOffsetEncoding.axis;
                for (size_t i = 0; i < num_channels; ++i) {
                  current_scales.push_back(bias_quant_params.bwAxisScaleOffsetEncoding.scales[i]);
                  current_offsets.push_back(bias_quant_params.bwAxisScaleOffsetEncoding.offsets[i]);
                }
              }

              // Check if all offsets are 0 and scales match expected values
              bool all_offsets_zero = true;
              bool all_scales_match = true;

              for (size_t i = 0; i < num_channels; ++i) {
                if (current_offsets[i] != 0) {
                  all_offsets_zero = false;
                }

                // Calculate expected scale for this channel
                // Use the corresponding weight scale if available, otherwise use the first one
                float weight_scale = (i < weights_scales.size()) ? weights_scales[i] : weights_scales[0];

                if (!utils::CheckBiasScaleMatch(current_scales[i], weight_scale, activation_scale, 1e-5f)) {
                  all_scales_match = false;
                }
              }

              if (all_offsets_zero && all_scales_match) {
                // No change needed - scales match and offsets are 0
              } else {
                // Need to requantize per-channel bias
                ORT_CXX_LOG(logger,
                            ORT_LOGGING_LEVEL_VERBOSE,
                            ("Requantizing per-channel bias " + bias_input.name).c_str());

                // Get current bias data and requantize
                std::vector<uint8_t> original_bias_data;
                RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor, original_bias_data));

                std::vector<uint8_t> requantized_bias_data;
                std::vector<float> new_scales;
                std::vector<int32_t> new_offsets;

                RETURN_IF_ERROR(utils::RequantizeBiasTensor(
                    original_bias_data, bias_info.shape, current_scales, current_offsets,
                    weights_scales, activation_scale, bias_info.qnn_data_type,
                    requantized_bias_data, new_scales, new_offsets,
                    quant_axis));

                // Create new tensor wrapper with requantized data
                std::string bias_name = bias_input.name;
                QnnQuantParamsWrapper new_quant_params(new_scales, new_offsets, quant_axis, false);
                QnnTensorWrapper bias_tensorwrapper(bias_name, QNN_TENSOR_TYPE_STATIC, bias_info.qnn_data_type,
                                                    std::move(new_quant_params), std::vector<uint32_t>(bias_info.shape),
                                                    std::move(requantized_bias_data));
                RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensorwrapper)), "Failed to add requantized bias tensor.");
                input_names.push_back(bias_name);
                return Ort::Status();  // We've handled the bias, return early
              }
            }
          }
        }
      }

      // Process bias normally (non-quantized or static non-quantized or scales already match)
      RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, bias_input, logger, input_names));
    }  // end else (non-BQ bias path)
  }

#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 16 && QNN_API_VERSION_MINOR <= 18)
  if (!has_bias_input && IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    // Bias is implicit. QNN SDK 2.23/2.24/2.25 (QNN API version 2.16/2.17/2.18) has a validation bug for
    // implicit bias inputs, so provide an explicit bias of all 0 (quantized int32).
    TensorInfo input0_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input0_info));

    TensorInfo input1_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input1_info));

    if (input0_info.quant_param.IsPerTensor(/*include_bw*/ true) && input1_info.quant_param.IsQuantized()) {
      const std::string bias_name = qnn::utils::UniqueNameGenerator().New(node_unit, "_implicit_bias");
      std::vector<uint32_t> bias_shape = {input1_info.shape[0]};
      RETURN_IF_ERROR(AddZeroBiasInput(qnn_model_wrapper, input0_info.quant_param, input1_info.quant_param,
                                       std::move(bias_shape), bias_name, logger, input_names));
    }
  }
#endif

  return Ort::Status();
}

Ort::Status ConvOpBuilder::ProcessConv1DInputs(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
                                               const Ort::Logger& logger,
                                               std::vector<std::string>& input_names,
                                               bool do_op_validation) const {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();
  OnnxConvType conv_type = {};
  RETURN_IF_ERROR(GetOnnxConvType(node_unit.OpType(), conv_type));

  assert(num_inputs >= 2);  // Checked by IsOpSupported.

  //
  // Input 0
  //

  {
    const std::string& input0_name = inputs[0].name;
    TensorInfo input0_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input0_info));

    const std::string conv_input0_name = input0_info.is_initializer ? input0_name
                                                                    : utils::UniqueNameGenerator().New(input0_name, "_reshape");
    input_names.push_back(conv_input0_name);

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(conv_input0_name)) {
      std::vector<uint8_t> unpacked_tensor;
      if (input0_info.is_initializer) {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input0_info.initializer_tensor, unpacked_tensor));
      }

      std::vector<uint32_t> shape = {
          input0_info.shape[0],  // N
          1,                     // Height == 1
          input0_info.shape[1],  // Width
          input0_info.shape[2]   // Channels
      };

      if (!input0_info.is_initializer) {
        RETURN_IF(input0_info.quant_param.IsPerChannel(),
                  "Non-constant Conv inputs only support per-tensor quantization");

        // Add Reshape node to transform 1D input to 2D (i.e., set height to 1).
        // We don't need to do this for initializers, because the number of elements does not change. We can just
        // modify the shape dimensions.
        bool is_graph_input = qnn_model_wrapper.IsGraphInput(input0_name);
        RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input0_name,
                                                         conv_input0_name,
                                                         input0_info.shape,
                                                         shape,
                                                         input0_info.qnn_data_type,
                                                         input0_info.quant_param,
                                                         do_op_validation,
                                                         is_graph_input));
      } else if (input0_info.quant_param.IsPerChannel()) {
        // The reshape (unsqueeze) may require us to shift the quant parameter's axis.
        RETURN_IF_ERROR(input0_info.quant_param.HandleUnsqueeze<uint32_t>(input0_info.shape, shape));
      }

      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(conv_input0_name);
      QnnTensorWrapper input_tensorwrapper(conv_input0_name, tensor_type, input0_info.qnn_data_type,
                                           std::move(input0_info.quant_param), std::move(shape),
                                           std::move(unpacked_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input0_name).c_str());
    }
  }

  //
  // Input 1: weight
  // We need to first reshape the weight in order to handle 1D convolutions with the Conv2d operator.
  // Next, we have to transpose the weight because ORT layout transformations do not change the weight layout.
  //
  {
    const std::string& input1_name = inputs[1].name;
    TensorInfo input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input_info));

    std::string conv_weight_input_name = input_info.is_initializer ? input1_name : utils::UniqueNameGenerator().New(input1_name, "_transpose");
    input_names.push_back(conv_weight_input_name);

    // Create the shape after reshaping.
    // Set height to 1 to be able to use 2D convolution.
    // Note: Conv shape is [N,C,1,W]. ConvTranspose shape is [C,N,1,W]
    std::vector<uint32_t> shape_2d = {
        input_info.shape[0],  // N
        input_info.shape[1],  // Channels
        1,                    // Height == 1
        input_info.shape[2],  // Width
    };

    std::vector<uint32_t> final_shape;
    final_shape.resize(4);

    // Create the final shape after the weights are transposed to HWCN.
    if (conv_type == OnnxConvType::kConv) {
      RETURN_IF_ERROR(utils::NchwShapeToHwcn<uint32_t>(shape_2d, final_shape));
    } else if (conv_type == OnnxConvType::kConvTranspose) {
      RETURN_IF_ERROR(utils::CnhwShapeToHwcn<uint32_t>(shape_2d, final_shape));
    } else {
      return MAKE_EP_FAIL(("QNN EP: Unexpected convolution op type: " + node_unit.OpType()).c_str());
    }

    const std::string reshape_output = utils::UniqueNameGenerator().New(input1_name, "_reshape");
    std::vector<uint8_t> unpacked_tensor;
    if (input_info.is_initializer) {
      //
      // Create a reshaped "view" of the initializer tensor with [N, C, 1, W] dims for Conv
      // ([C, N, 1, W] for ConvTranspose).
      //
      std::vector<int64_t> shape_2d_int64;
      shape_2d_int64.resize(4);

      std::transform(shape_2d.begin(), shape_2d.end(), shape_2d_int64.begin(), [](uint32_t dim) -> int64_t {
        return static_cast<int64_t>(dim);
      });

      // The reshape (unsqueeze) may require us to shift the quant parameter's axis.
      if (input_info.quant_param.IsPerChannel()) {
        RETURN_IF_ERROR(input_info.quant_param.HandleUnsqueeze<uint32_t>(input_info.shape, shape_2d));
      }

      //
      // Get transposed initializer bytes.
      //
      std::vector<uint8_t> original_tensor_bytes;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_info.initializer_tensor,
                                                              original_tensor_bytes));
      unpacked_tensor.resize(original_tensor_bytes.size());

      const OrtTypeInfo* type_info = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(
          static_cast<const OrtValueInfo*>(input_info.initializer_tensor), &type_info));

      const OrtTensorTypeAndShapeInfo* type_shape = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &type_shape));
      ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(type_shape, &elem_type));
      const size_t elem_byte_size = qnn::utils::GetElementSizeByType(elem_type);
      RETURN_IF(elem_byte_size == 0,
                ("Can't get element byte size from given ONNX type for initializer " + input1_name).c_str());

      if (conv_type == OnnxConvType::kConv) {
        RETURN_IF_ERROR(utils::TransposeFromNchwToHwcn(std::move(shape_2d_int64), elem_byte_size, original_tensor_bytes,
                                                       unpacked_tensor, /*is_3d*/ false));
      } else if (conv_type == OnnxConvType::kConvTranspose) {
        RETURN_IF_ERROR(utils::TransposeFromCnhwToHwcn(std::move(shape_2d_int64), elem_byte_size, original_tensor_bytes,
                                                       unpacked_tensor, /*is_3d*/ false));
      } else {
        return MAKE_EP_FAIL(("QNN EP: Unexpected convolution op type: " + node_unit.OpType()).c_str());
      }

      // Transpose quantization parameter's axis if this is using per-channel quantization.
      if (input_info.quant_param.IsPerChannel()) {
        const std::vector<size_t>& perm = conv_type == OnnxConvType::kConv ? nchw2hwcn_perm : cnhw2hwcn_perm;
        std::vector<size_t> perm_inv(perm.size());
        RETURN_IF_ERROR(utils::InvertPerm<size_t>(perm, perm_inv));
        RETURN_IF_ERROR(input_info.quant_param.HandleTranspose<size_t>(perm_inv));
      }
    } else {
      // Dynamic weight: Add nodes to reshape to 2D, and then transpose.
      RETURN_IF(input_info.quant_param.IsPerChannel(),
                "Non-constant Conv inputs only support per-tensor quantization");

      if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input1_name)) {
        QnnTensorWrapper weight_tensor_wrapper;
        RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[1], weight_tensor_wrapper));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor_wrapper)), "Failed to add weight tensor.");
      }

      bool is_graph_input = qnn_model_wrapper.IsGraphInput(input1_name);
      ORT_CXX_LOG(logger,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("Adding Reshape (to 2D) and HWCN Transpose node after input: " + input1_name).c_str());
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input1_name,
                                                       reshape_output,
                                                       input_info.shape,
                                                       shape_2d,
                                                       input_info.qnn_data_type,
                                                       input_info.quant_param,
                                                       do_op_validation,
                                                       is_graph_input));
      if (conv_type == OnnxConvType::kConv) {
        RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(node_unit.Index(),
                                                                 reshape_output,
                                                                 conv_weight_input_name,
                                                                 shape_2d,
                                                                 final_shape,
                                                                 input_info.qnn_data_type,
                                                                 input_info.quant_param,
                                                                 do_op_validation,
                                                                 false));
      } else if (conv_type == OnnxConvType::kConvTranspose) {
        RETURN_IF_ERROR(qnn_model_wrapper.AddCnhwToHwcnTranspose(node_unit.Index(),
                                                                 reshape_output,
                                                                 conv_weight_input_name,
                                                                 shape_2d,
                                                                 final_shape,
                                                                 input_info.qnn_data_type,
                                                                 input_info.quant_param,
                                                                 do_op_validation,
                                                                 false));
      } else {
        return MAKE_EP_FAIL(("QNN EP: Unexpected convolution op type: " + node_unit.OpType()).c_str());
      }
    }

    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(conv_weight_input_name);
    QnnTensorWrapper input_tensorwrapper(conv_weight_input_name, tensor_type, input_info.qnn_data_type,
                                         std::move(input_info.quant_param), std::move(final_shape),
                                         std::move(unpacked_tensor));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  //
  // Input 2: bias
  //
  if (num_inputs == 3) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, input_names));
  }

  return Ort::Status();
}

Ort::Status ConvOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  const auto& outputs = node_unit.Outputs();

  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].shape, output_shape), "Cannot get shape");
  const bool is_1d_conv = output_shape.size() == 3;
  const bool is_3d_conv = output_shape.size() == 5;

  OnnxConvType conv_type = {};
  RETURN_IF_ERROR(GetOnnxConvType(node_unit.OpType(), conv_type));

  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  const auto& input_0 = node_unit.Inputs()[0];
  const auto& input_1 = node_unit.Inputs()[1];
  std::vector<uint32_t> input_0_shape;  // NHW[D]C
  std::vector<uint32_t> input_1_shape;  // NCHW[D]
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.shape, input_0_shape), "Cannot get shape");
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_1.shape, input_1_shape), "Cannot get shape");

  // Kernel shape
  std::vector<uint32_t> kernel_shape;
  kernel_shape = node_helper.Get("kernel_shape", kernel_shape);
  if (kernel_shape.empty()) {  // infer from weight shape
    kernel_shape.assign(input_1_shape.begin() + 2, input_1_shape.end());
  }
  if (is_1d_conv) {
    // insert Hight = 1 for 1D
    kernel_shape.insert(kernel_shape.begin(), 1);
  }

  // Dilations parameter
  std::vector<uint32_t> dilations;
  dilations.assign(kernel_shape.size(), 1);

  if (conv_type == OnnxConvType::kConv) {
    dilations = node_helper.Get("dilations", dilations);

    // Handle 1D conv by setting height dilation to 1.
    if (dilations.size() == 1) {
      const uint32_t width_dilation = dilations[0];
      dilations.resize(2);
      dilations[0] = 1;  // Height == 1
      dilations[1] = width_dilation;
    }

    QnnParamWrapper dilation_paramwrapper(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_DILATION,
                                          {SafeInt<uint32_t>(dilations.size())}, std::vector<uint32_t>(dilations));
    param_tensor_names.push_back(dilation_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(dilation_paramwrapper));
  }

  // Strides parameter.
  std::vector<uint32_t> strides;
  strides.assign(kernel_shape.size(), 1);
  strides = node_helper.Get("strides", strides);
  {
    // Handle 1D conv by setting the height stride to 1.
    if (strides.size() == 1) {
      const uint32_t width_stride = strides[0];
      strides.resize(2);
      strides[0] = 1;  // Height
      strides[1] = width_stride;
    }

    QnnParamWrapper stride_amount_paramwrapper(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_STRIDE,
                                               {SafeInt<uint32_t>(strides.size())}, std::vector<uint32_t>(strides));
    param_tensor_names.push_back(stride_amount_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(stride_amount_paramwrapper));
  }

  // Output padding parameter. (Only for ConvTranspose)
  std::vector<uint32_t> output_padding;
  output_padding.assign(kernel_shape.size(), 0);
  if (conv_type == OnnxConvType::kConvTranspose) {
    output_padding = node_helper.Get("output_padding", output_padding);

    // Handle 1D conv.
    if (output_padding.size() == 1) {
      const uint32_t width_out_pad = output_padding[0];
      output_padding.resize(2);
      output_padding[0] = 0;  // Height: default output padding of 0
      output_padding[1] = width_out_pad;
    }

    QnnParamWrapper output_padding_paramwrapper(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_PADDING,
                                                {static_cast<uint32_t>(output_padding.size())}, std::vector<uint32_t>(output_padding));
    param_tensor_names.push_back(output_padding_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(output_padding_paramwrapper));
  }

  // Pads attribute
  {
    std::vector<uint32_t> pads;
    pads.assign(kernel_shape.size() * 2, 0);
    pads = node_helper.Get("pads", pads);
    auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
    RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER" && auto_pad != "VALID",
              ("QNN Conv operators do not support 'auto_pad' value: " + auto_pad).c_str());

    std::vector<int64_t> output_shape_attribute_value = node_helper.Get("output_shape", std::vector<int64_t>());
    bool has_output_shape_attr = !output_shape_attribute_value.empty();

    if (conv_type == OnnxConvType::kConvTranspose && has_output_shape_attr) {
      // Pads are auto generated using the formula:
      // total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
      // Then distributed using auto_pad rules.

      ORT_CXX_LOG(logger,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  "ConvTranspose with 'output_shape' attribute. Calculating pads since output_shape is specified, pad values are ignored");

      // input_dims for calculation are (H, W, D...) excluding N, C
      std::vector<uint32_t> input_dims(input_0_shape.begin() + 1, input_0_shape.end() - 1);

      if (is_1d_conv) {  // Adjust input_dims and output_shape_attribute_value for 1D conv logic
        input_dims.insert(input_dims.begin(), 1);
        output_shape_attribute_value.insert(output_shape_attribute_value.begin(), 1);
      }

      pads.assign(kernel_shape.size() * 2, 0);  // Reset pads before filling
      size_t rank = input_dims.size();

      RETURN_IF_NOT(rank == output_shape_attribute_value.size(),
                    "QNN EP: ConvTranspose 'output_shape' attribute rank mismatch "
                    "with input dims for padding calculation.");

      for (size_t dim = 0; dim < rank; ++dim) {
        int64_t pad_head = 0;
        int64_t pad_tail = 0;
        AutoPadType pad_type = StringToAutoPadType(auto_pad);  // Use current auto_pad for distribution

        auto total_pad = ComputeTotalPad(input_dims[dim], strides[dim], output_padding[dim],
                                         kernel_shape[dim], dilations[dim], output_shape_attribute_value[dim]);
        DistributePadding(pad_type, total_pad, pad_head, pad_tail);

        pads[dim] = gsl::narrow<uint32_t>(pad_head);
        pads[rank + dim] = gsl::narrow<uint32_t>(pad_tail);
      }

    } else if (auto_pad != "NOTSET") {  // Case: auto_pad is SAME_UPPER/LOWER/VALID, no output_shape attribute
      auto pad_type = qnn::StringToAutoPadType(auto_pad);
      // skip N, C, input0 shape NHWC
      std::vector<uint32_t> input_dims(input_0_shape.begin() + 1, input_0_shape.end() - 1);
      std::vector<uint32_t> output_dims(output_shape.begin() + 1, output_shape.end() - 1);
      if (is_1d_conv) {
        // insert Height = 1 for 1D
        input_dims.insert(input_dims.begin(), 1);
        output_dims.insert(output_dims.begin(), 1);
      }
      size_t rank = input_dims.size();
      for (size_t dim = 0; dim < rank; ++dim) {
        int64_t pad_head = pads[dim];
        int64_t pad_tail = pads[rank + dim];
        if (conv_type == OnnxConvType::kConv) {
          RETURN_IF_ERROR(qnn::ComputePad(input_dims[dim],
                                          strides[dim],
                                          kernel_shape[dim],
                                          dilations[dim],
                                          pad_type,
                                          pad_head,
                                          pad_tail));
        } else if (conv_type == OnnxConvType::kConvTranspose) {
          auto total_pad = qnn::ComputeTotalPad(input_dims[dim], strides[dim], output_padding[dim],
                                                kernel_shape[dim], dilations[dim], output_dims[dim]);
          qnn::DistributePadding(pad_type, total_pad, pad_head, pad_tail);
        }
        pads[dim] = gsl::narrow<uint32_t>(pad_head);
        pads[rank + dim] = gsl::narrow<uint32_t>(pad_tail);
      }
    } else {
      // Handle 1D conv by setting padding for height to 0.
      if (pads.size() == 2) {
        const uint32_t width_pad_begin = pads[0];
        const uint32_t width_pad_end = pads[1];
        pads.resize(4);
        pads[0] = 0;  // Height pad begin: 0
        pads[1] = width_pad_begin;
        pads[2] = 0;  // Height pad end: 0
        pads[3] = width_pad_end;
      }
    }

    ReArrangePads(pads);
    uint32_t pad_size = gsl::narrow<uint32_t>(pads.size() / 2);
    QnnParamWrapper pad_amount_paramwrapper(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_PAD_AMOUNT,
                                            {pad_size, 2}, std::move(pads));
    param_tensor_names.push_back(pad_amount_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_paramwrapper));
  }

  const uint32_t group = node_helper.Get("group", static_cast<uint32_t>(1));
  const uint32_t num_output_channels = output_shape.back();
  uint32_t num_input_channels = 0;
  RETURN_IF_ERROR(GetInputChannelNumber(qnn_model_wrapper, node_unit, num_input_channels));

  // There's DepthWiseConv2d, but no DepthWiseConv3d
  const bool is_depthwise_conv2d = (!is_3d_conv) && (conv_type == OnnxConvType::kConv) &&
                                   (num_input_channels == num_output_channels) &&
                                   (group == num_output_channels);

  if (!is_depthwise_conv2d) {  // DepthWiseConv2d does not need a group parameter.
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), group,
                                           QNN_OP_CONV_2D_PARAM_GROUP, param_tensor_names));
  } else {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Using DepthWiseConv2d instead of Conv2d for node " + node_unit.Name()).c_str());
  }

  std::string output_node_type;
  if (is_3d_conv) {
    if (conv_type == OnnxConvType::kConv) {
      output_node_type = QNN_OP_CONV_3D;
    } else {
      output_node_type = QNN_OP_TRANSPOSE_CONV_3D;
    }
  } else {
    output_node_type = is_depthwise_conv2d ? QNN_OP_DEPTH_WISE_CONV_2D : GetQnnOpType(node_unit.OpType());
  }

  QnnQuantParamsWrapper output_quantize_param;
  RETURN_IF_ERROR(output_quantize_param.Init(qnn_model_wrapper, outputs[0]));
  bool is_quantized_tensor = outputs[0].quant_param.has_value();

  ONNXTensorElementDataType output_type = outputs[0].type;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_tensor, output_type, qnn_data_type));

  // Detect BQ or LPBQ Conv from the weight tensor's quant encoding.
  // BQ  (BW_FLOAT_BLOCK):       Conv outputs FP16 → need FP16 intermediate + Convert(FP16→INT16).
  // LPBQ (BLOCKWISE_EXPANSION): Conv outputs INT16 → standard quantized output path.
  // input_names[1] is the weight — IsOpSupported guarantees Conv has >= 2 inputs.
  bool is_bq_conv = false;
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_names[1])) {
    is_bq_conv = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]).GetQnnQuantParams().IsBlockQuantized();
  }

  const auto& output_name = outputs[0].name;
  if (is_1d_conv) {
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
    std::vector<uint32_t> output_shape_2d = {
        output_shape[0],  // N
        1,                // H == 1
        output_shape[1],  // W
        output_shape[2],  // C
    };
    const std::string conv_output_name = utils::UniqueNameGenerator().New(output_name, "_conv");
    QnnTensorWrapper output_tensorwrapper(conv_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type,
                                          output_quantize_param.Copy(), std::vector<uint32_t>(output_shape_2d));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  output_node_type,
                                                  std::move(input_names),
                                                  {conv_output_name},
                                                  std::move(param_tensor_names),
                                                  do_op_validation),
                  "Failed to add node.");

    // Add Reshape to convert QNN Conv2d/TransposeConv2d/DepthWiseConv2d output back to 1D.
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(conv_output_name,
                                                     output_name,
                                                     output_shape_2d,
                                                     output_shape,
                                                     qnn_data_type,
                                                     output_quantize_param,
                                                     do_op_validation,
                                                     false,
                                                     is_graph_output));
  } else {
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
    Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

    if (is_bq_conv && is_quantized_tensor) {
      // BQ Conv outputs FP16; downstream QDQ expects INT16.
      // Emit: Conv (FP16 output) → Quantize (FP16 → INT16 quantized output).
      // Reuse the original QuantizeLinear node's input name (the target Conv's output[0]) for the
      // FP16 tensor. That tensor is conceptually the un-quantized Conv output — exactly what this
      // BQ Conv produces — and QNN EP otherwise skips it, so the name is free and keeps the QNN
      // graph aligned with the ONNX graph naming.
      const std::string conv_fp16_out = Ort::ConstNode(&node_unit.GetNode()).GetOutputs()[0].GetName();
      QnnTensorWrapper fp16_out_wrapper(conv_fp16_out, QNN_TENSOR_TYPE_NATIVE,
                                        QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                        std::vector<uint32_t>(output_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fp16_out_wrapper)),
                    "Failed to add FP16 Conv BQ output tensor.");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    output_node_type,
                                                    std::move(input_names),
                                                    {conv_fp16_out},
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to add BQ Conv node.");
      // INT16 quantized output tensor consumed by downstream nodes.
      QnnTensorWrapper int16_out_wrapper(output_name, tensor_type, qnn_data_type,
                                         std::move(output_quantize_param),
                                         std::move(output_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(int16_out_wrapper)),
                    "Failed to add INT16 BQ Conv output tensor.");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        utils::UniqueNameGenerator().New(output_name, "_fp16_quantize"),
                        QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_QUANTIZE,
                        {conv_fp16_out}, {output_name}, {}, do_op_validation),
                    "Failed to add FP16→INT16 Quantize node for BQ Conv output.");
    } else {
      QnnTensorWrapper output_tensorwrapper(output_name, tensor_type, qnn_data_type,
                                            std::move(output_quantize_param), std::move(output_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    output_node_type,
                                                    std::move(input_names),
                                                    {output_name},
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to add node.");
    }
  }

  return Ort::Status();
}

void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ConvOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
