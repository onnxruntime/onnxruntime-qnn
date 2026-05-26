// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

// LpPool is implemented as a primitive decomposition rather than a single QNN op.
//   p = 2:  x -> Multiply(x, x) -> AvgPool -> SquareRoot -> Multiply(scale=sqrt(K))
//   p = 1:  x -> Abs(x)         -> AvgPool ->            Multiply(scale=K)
// where K = product(kernel_shape) and AvgPool uses count_pad_for_edges = true so the
// denominator is always K regardless of how many padded elements fall in the window.
// Rank-3 inputs are bracketed by Reshape (NCL <-> NC1L) and use the 2D pool path.
//
// LpPool is registered as layout-sensitive in QnnEp::ShouldConvertDataLayoutForOpImpl, so the
// op builder receives NHWC tensors after the layout transformer runs.
class LpPoolOpBuilder : public BaseOpBuilder {
 public:
  LpPoolOpBuilder() : BaseOpBuilder("LpPoolOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LpPoolOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status LpPoolOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].type, ""));

  RETURN_IF(node_unit.Outputs().size() != 1, "QNN LpPool only supports 1 output.");

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape");

  const size_t rank = input_shape.size();
  RETURN_IF_NOT(rank >= 3 && rank <= 5, "QNN LpPool only supports input rank 3, 4, or 5.");

  // Rank-5 (3D spatial) requires QNN_OP_POOL_AVG_3D, which is not supported on the QNN GPU backend.
  // CPU and HTP have PoolAvg3d kernels; rank-5 LpPool falls back to the ORT CPU EP on GPU.
  RETURN_IF(rank == 5 && IsGpuBackend(qnn_model_wrapper.GetQnnBackendType()),
            "QNN LpPool: rank-5 (3D pooling) is not supported on the QNN GPU backend.");

  // Rank-5 native-float QNN_OP_POOL_AVG_3D fails NHWC dry-run validation on the HTP backend
  // (mirrors the rank-5 PoolMax3d rejection in pool_op_builder.cc). HTP only validates 3D
  // pooling for QDQ paths in the existing test suite.
  RETURN_IF(rank == 5 && IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()),
            "QNN LpPool: rank-5 (3D pooling) is not supported on the QNN HTP backend.");

  OrtNodeAttrHelper node_helper(node_unit);

  const auto p = node_helper.Get("p", static_cast<int64_t>(2));
  RETURN_IF(p != 1 && p != 2, "QNN LpPool only supports p=1 or p=2.");

  const auto ceil_mode = node_helper.Get("ceil_mode", static_cast<int64_t>(0));
  if (ceil_mode != 0) {
    // QNN's CPU backend silently ignores PoolAvg2d's rounding_mode and produces a floor-shape
    // output, leaving the extra ceil-mode positions filled with garbage / NaN. HTP and GPU honor
    // rounding_mode correctly, so the rejection is CPU-specific.
    RETURN_IF(IsCpuBackend(qnn_model_wrapper.GetQnnBackendType()),
              "QNN LpPool does not support ceil_mode=1 on the CPU backend "
              "(QNN CPU PoolAvg2d ignores rounding_mode).");
  }

  const auto dilations = node_helper.Get("dilations", std::vector<uint32_t>(rank - 2, 1));
  RETURN_IF_NOT(dilations == std::vector<uint32_t>(rank - 2, 1),
                "QNN LpPool does not support dilations > 1.");

  const auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_UPPER" &&
                auto_pad != "SAME_LOWER" && auto_pad != "VALID",
            ("QNN LpPool does not support 'auto_pad' value: " + auto_pad).c_str());

  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  return Ort::Status();
}

namespace {

// Encodes a single floating-point scale value into a tensor blob of the requested QNN dtype.
Ort::Status EncodeScalarScaleData(Qnn_DataType_t dtype, float value, std::vector<uint8_t>& out) {
  if (dtype == QNN_DATATYPE_FLOAT_16) {
    out.resize(sizeof(uint16_t));
    Ort::Float16_t v(value);
    std::memcpy(out.data(), &v.val, sizeof(uint16_t));
  } else if (dtype == QNN_DATATYPE_BFLOAT_16) {
    out.resize(sizeof(uint16_t));
    Ort::BFloat16_t v(value);
    std::memcpy(out.data(), &v.val, sizeof(uint16_t));
  } else if (dtype == QNN_DATATYPE_FLOAT_32) {
    out.resize(sizeof(float));
    std::memcpy(out.data(), &value, sizeof(float));
  } else {
    return Ort::Status("QNN LpPool: unsupported input dtype for scale tensor.", ORT_INVALID_ARGUMENT);
  }
  return Ort::Status();
}

}  // namespace

Ort::Status LpPoolOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& logger,
                                                         bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();

  std::vector<uint32_t> onnx_input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, onnx_input_shape), "Cannot get input shape");

  std::vector<uint32_t> onnx_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Outputs()[0].shape, onnx_output_shape),
                "Cannot get output shape");

  TensorInfo input_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  const int64_t p_value = node_helper.Get("p", static_cast<int64_t>(2));
  const bool requires_rank3_reshape = (onnx_input_shape.size() == 3);

  // Working shapes: rank-3 inputs are reshaped to rank-4 NHWC (N, 1, L, C); other ranks are unchanged.
  // The layout transformer hands us NHWC tensors (kMSInternalNHWCDomain), so for NHWC rank-3
  // [N, L, C] the spatial L lands at W (index 2) with H=1 (index 1) and C stays at index 3.
  std::vector<uint32_t> qnn_input_shape = onnx_input_shape;
  std::vector<uint32_t> qnn_output_shape = onnx_output_shape;
  if (requires_rank3_reshape) {
    qnn_input_shape = {onnx_input_shape[0], 1, onnx_input_shape[1], onnx_input_shape[2]};
    qnn_output_shape = {onnx_output_shape[0], 1, onnx_output_shape[1], onnx_output_shape[2]};

    const std::string reshaped_input_name = utils::UniqueNameGenerator().New(input_names[0], "_reshape");
    QnnTensorWrapper reshaped_input_tensor(reshaped_input_name,
                                           QNN_TENSOR_TYPE_NATIVE,
                                           input_info.qnn_data_type,
                                           input_info.quant_param.Copy(),
                                           std::vector<uint32_t>(qnn_input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshaped_input_tensor)),
                  "Failed to add reshape prior tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_RESHAPE),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
                                                  {input_names[0]}, {reshaped_input_name}, {}, do_op_validation),
                  "Failed to create reshape prior node for LpPool.");
    input_names[0] = reshaped_input_name;
  }

  const size_t rank = qnn_input_shape.size();  // 4 or 5
  const size_t spatial_rank = rank - 2;        // 2 or 3
  const bool is_3d_pool = (spatial_rank == 3);

  // ------------------------------------------------------------------------------------------------
  // 1. Read attributes (expanding 1D values to 2D form for the rank-3 path).
  // ------------------------------------------------------------------------------------------------
  std::vector<uint32_t> filter_size;
  {
    auto raw = node_helper.Get("kernel_shape", std::vector<uint32_t>(spatial_rank, 1));
    filter_size = (raw.size() == 1) ? std::vector<uint32_t>{1, raw[0]} : raw;
  }
  RETURN_IF_NOT(filter_size.size() == spatial_rank,
                "QNN LpPool: kernel_shape rank mismatch with input spatial rank.");

  std::vector<uint32_t> stride;
  {
    auto raw = node_helper.Get("strides", std::vector<uint32_t>(spatial_rank, 1));
    stride = (raw.size() == 1) ? std::vector<uint32_t>{1, raw[0]} : raw;
  }

  std::vector<uint32_t> dilations;
  {
    auto raw = node_helper.Get("dilations", std::vector<uint32_t>(spatial_rank, 1));
    dilations = (raw.size() == 1) ? std::vector<uint32_t>{1, raw[0]} : raw;
  }

  std::vector<uint32_t> pad_amount;
  {
    auto raw = node_helper.Get("pads", std::vector<uint32_t>(spatial_rank * 2, 0));
    pad_amount = (raw.size() == 2) ? std::vector<uint32_t>{0, raw[0], 0, raw[1]} : raw;
  }
  RETURN_IF_NOT(pad_amount.size() == spatial_rank * 2,
                "QNN LpPool: pads rank mismatch with input spatial rank.");

  const auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  if (auto_pad != "NOTSET") {
    for (size_t axis = 0; axis < spatial_rank; ++axis) {
      if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        const uint32_t total_pads = (qnn_output_shape[axis + 1] - 1) * stride[axis] +
                                    (filter_size[axis] - 1) * dilations[axis] + 1 -
                                    qnn_input_shape[axis + 1];
        if (auto_pad == "SAME_LOWER") {
          pad_amount[axis + spatial_rank] = total_pads / 2;
          pad_amount[axis] = total_pads - pad_amount[axis + spatial_rank];
        } else {
          pad_amount[axis] = total_pads / 2;
          pad_amount[axis + spatial_rank] = total_pads - pad_amount[axis];
        }
      }
    }
  }

  // ------------------------------------------------------------------------------------------------
  // 2. Validate kernel does not exceed padded input on any spatial axis.
  // ------------------------------------------------------------------------------------------------
  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    const uint32_t in_dim = qnn_input_shape[axis + 1];
    const uint32_t pad_total = pad_amount[axis] + pad_amount[axis + spatial_rank];
    RETURN_IF(filter_size[axis] > in_dim + pad_total,
              "QNN LpPool: kernel exceeds padded input on a spatial axis.");
  }

  // Convert ONNX pad layout [begins..., ends...] to QNN pair layout [begin0, end0, begin1, end1, ...].
  ReArrangePads(pad_amount);

  // ------------------------------------------------------------------------------------------------
  // 3a. HTP-only workaround: when AvgPool is asked to handle non-zero pad_amount with
  //     count_pad_for_edges=1, the HTP backend silently returns 0 across every output position
  //     on certain SoCs. CPU and GPU honor the param correctly. To sidestep the HTP bug without
  //     changing the math, we pre-pad the input with an explicit constant-zero Pad node and run
  //     AvgPool with pad_amount = 0. The output is identical because Mul(x,x)/Abs(x) of a
  //     zero is zero, and an unpadded AvgPool with K real elements per window has denominator K.
  // ------------------------------------------------------------------------------------------------
  const bool is_htp_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  const bool any_pad = std::any_of(pad_amount.begin(), pad_amount.end(),
                                   [](uint32_t v) { return v != 0; });
  if (is_htp_backend && any_pad) {
    // Padded shape: each spatial axis grows by pad_begin + pad_end.
    std::vector<uint32_t> padded_shape = qnn_input_shape;
    for (size_t a = 0; a < spatial_rank; ++a) {
      padded_shape[a + 1] += pad_amount[2 * a] + pad_amount[2 * a + 1];
    }

    // QNN_OP_PAD pad_amount param is shape [input_rank, 2], one row per tensor axis (N, spatial..., C).
    // Only the spatial axes get non-zero values.
    std::vector<uint32_t> pad_full(rank * 2, 0);
    for (size_t a = 0; a < spatial_rank; ++a) {
      pad_full[(a + 1) * 2 + 0] = pad_amount[2 * a];
      pad_full[(a + 1) * 2 + 1] = pad_amount[2 * a + 1];
    }

    const std::string padded_name = utils::UniqueNameGenerator().New(input_names[0], "_padded");
    QnnTensorWrapper padded_tensor(padded_name, QNN_TENSOR_TYPE_NATIVE,
                                   input_info.qnn_data_type, input_info.quant_param.Copy(),
                                   std::vector<uint32_t>(padded_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(padded_tensor)),
                  "Failed to add padded input tensor for LpPool HTP workaround.");

    std::vector<std::string> pad_param_names;
    {
      Qnn_Scalar_t scheme_scalar = QNN_SCALAR_INIT;
      scheme_scalar.dataType = QNN_DATATYPE_UINT_32;
      scheme_scalar.uint32Value = QNN_OP_PAD_SCHEME_CONSTANT;
      QnnParamWrapper scheme_param(node_unit.Index(), node_unit.Name(),
                                   QNN_OP_PAD_PARAM_SCHEME, scheme_scalar);
      pad_param_names.push_back(scheme_param.GetParamTensorName());
      RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(scheme_param)),
                    "Failed to add Pad scheme param.");
    }
    {
      QnnParamWrapper pa_param(node_unit.Index(), node_unit.Name(),
                               QNN_OP_PAD_PARAM_PAD_AMOUNT,
                               {static_cast<uint32_t>(rank), 2},
                               std::move(pad_full));
      pad_param_names.push_back(pa_param.GetParamTensorName());
      RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(pa_param)),
                    "Failed to add Pad pad_amount param.");
    }

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, QNN_OP_PAD),
                      QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PAD,
                      {input_names[0]}, {padded_name},
                      std::move(pad_param_names), do_op_validation),
                  "Failed to create Pad node for LpPool HTP workaround.");

    // Rewire the chain: Op A (Multiply/Abs) and AvgPool now operate on the padded tensor with
    // zero pad_amount sent to AvgPool itself.
    input_names[0] = padded_name;
    qnn_input_shape = std::move(padded_shape);
    std::fill(pad_amount.begin(), pad_amount.end(), 0u);
  }

  // ------------------------------------------------------------------------------------------------
  // 3. Compute scale constant K (product of kernel dims). For p=2 we apply sqrt(K) at the end;
  //    for p=1 we apply K.
  // ------------------------------------------------------------------------------------------------
  uint64_t k_product = 1;
  for (uint32_t k : filter_size) k_product *= k;
  const float scale_value = (p_value == 2) ? std::sqrt(static_cast<float>(k_product))
                                           : static_cast<float>(k_product);

  // ------------------------------------------------------------------------------------------------
  // 4. Op A: Multiply(x, x) for p=2, or Abs(x) for p=1.
  // ------------------------------------------------------------------------------------------------
  const std::string preprocess_out_name =
      utils::UniqueNameGenerator().New(node_unit.Name(), p_value == 2 ? "_squared" : "_abs");
  {
    QnnTensorWrapper preprocess_tensor(preprocess_out_name,
                                       QNN_TENSOR_TYPE_NATIVE,
                                       input_info.qnn_data_type,
                                       input_info.quant_param.Copy(),
                                       std::vector<uint32_t>(qnn_input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(preprocess_tensor)),
                  "Failed to add preprocess tensor.");
  }

  if (p_value == 2) {
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_MULTIPLY),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_ELEMENT_WISE_MULTIPLY,
                      {input_names[0], input_names[0]},
                      {preprocess_out_name},
                      {},
                      do_op_validation),
                  "Failed to create square (Multiply) node.");
  } else {
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_ABS),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_ELEMENT_WISE_ABS,
                      {input_names[0]},
                      {preprocess_out_name},
                      {},
                      do_op_validation),
                  "Failed to create Abs node.");
  }

  // ------------------------------------------------------------------------------------------------
  // 5. Op B: PoolAvg2d / PoolAvg3d with count_pad_for_edges = true.
  // ------------------------------------------------------------------------------------------------
  const char* pool_op = is_3d_pool ? QNN_OP_POOL_AVG_3D : QNN_OP_POOL_AVG_2D;
  const char* p_filter = is_3d_pool ? QNN_OP_POOL_AVG_3D_PARAM_FILTER_SIZE : QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE;
  const char* p_stride = is_3d_pool ? QNN_OP_POOL_AVG_3D_PARAM_STRIDE : QNN_OP_POOL_AVG_2D_PARAM_STRIDE;
  const char* p_pad = is_3d_pool ? QNN_OP_POOL_AVG_3D_PARAM_PAD_AMOUNT : QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT;
  const char* p_round = is_3d_pool ? QNN_OP_POOL_AVG_3D_PARAM_ROUNDING_MODE : QNN_OP_POOL_AVG_2D_PARAM_ROUNDING_MODE;
  const char* p_count = is_3d_pool ? QNN_OP_POOL_AVG_3D_PARAM_COUNT_PAD_FOR_EDGES
                                   : QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES;

  std::vector<std::string> pool_param_names;
  {
    QnnParamWrapper filter_size_param(node_unit.Index(), node_unit.Name(), p_filter,
                                      {static_cast<uint32_t>(filter_size.size())},
                                      std::vector<uint32_t>(filter_size));
    pool_param_names.push_back(filter_size_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(filter_size_param)),
                  "Failed to add param filter_size.");
  }
  {
    QnnParamWrapper stride_param(node_unit.Index(), node_unit.Name(), p_stride,
                                 {static_cast<uint32_t>(stride.size())},
                                 std::vector<uint32_t>(stride));
    pool_param_names.push_back(stride_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(stride_param)),
                  "Failed to add param stride.");
  }
  {
    QnnParamWrapper pad_amount_param(node_unit.Index(), node_unit.Name(), p_pad,
                                     {static_cast<uint32_t>(pad_amount.size() / 2), 2},
                                     std::vector<uint32_t>(pad_amount));
    pool_param_names.push_back(pad_amount_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_param)),
                  "Failed to add param pad_amount.");
  }
  // ceil_mode = 1 is rejected on CPU backend in IsOpSupported; HTP/GPU honor rounding_mode.
  const int64_t ceil_mode = node_helper.Get("ceil_mode", static_cast<int64_t>(0));
  if (ceil_mode != 0) {
    Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
    scalar.dataType = QNN_DATATYPE_UINT_32;
    scalar.int32Value = static_cast<int32_t>(ceil_mode);
    QnnParamWrapper rounding_mode_param(node_unit.Index(), node_unit.Name(), p_round, scalar);
    pool_param_names.push_back(rounding_mode_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(rounding_mode_param)),
                  "Failed to add param rounding_mode.");
  }
  {
    Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
    scalar.dataType = QNN_DATATYPE_BOOL_8;
    scalar.bool8Value = static_cast<uint8_t>(1);
    QnnParamWrapper count_pad_param(node_unit.Index(), node_unit.Name(), p_count, scalar);
    pool_param_names.push_back(count_pad_param.GetParamTensorName());
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(count_pad_param)),
                  "Failed to add param count_pad_for_edges.");
  }

  const std::string pool_out_name = utils::UniqueNameGenerator().New(node_unit.Name(), "_pool");
  {
    QnnTensorWrapper pool_out_tensor(pool_out_name,
                                     QNN_TENSOR_TYPE_NATIVE,
                                     input_info.qnn_data_type,
                                     output_info.quant_param.Copy(),
                                     std::vector<uint32_t>(qnn_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(pool_out_tensor)),
                  "Failed to add pool output tensor.");
  }
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, pool_op),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW, pool_op,
                                                {preprocess_out_name}, {pool_out_name},
                                                std::move(pool_param_names), do_op_validation),
                "Failed to create AvgPool node for LpPool.");

  // ------------------------------------------------------------------------------------------------
  // 6. (p=2 only) Op C: ElementWiseSquareRoot.
  // ------------------------------------------------------------------------------------------------
  std::string sqrt_or_pool_out_name = pool_out_name;
  if (p_value == 2) {
    const std::string sqrt_out_name = utils::UniqueNameGenerator().New(node_unit.Name(), "_sqrt");
    QnnTensorWrapper sqrt_out_tensor(sqrt_out_name,
                                     QNN_TENSOR_TYPE_NATIVE,
                                     input_info.qnn_data_type,
                                     output_info.quant_param.Copy(),
                                     std::vector<uint32_t>(qnn_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sqrt_out_tensor)),
                  "Failed to add sqrt output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_SQUARE_ROOT),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_ELEMENT_WISE_SQUARE_ROOT,
                      {pool_out_name}, {sqrt_out_name}, {}, do_op_validation),
                  "Failed to create SquareRoot node.");
    sqrt_or_pool_out_name = sqrt_out_name;
  }

  // ------------------------------------------------------------------------------------------------
  // 7. Build static scalar tensor and emit final Multiply.
  // ------------------------------------------------------------------------------------------------
  const std::string scale_name = utils::UniqueNameGenerator().New(node_unit.Name(), "_scale");
  {
    std::vector<uint8_t> scale_data;
    RETURN_IF_ERROR(EncodeScalarScaleData(input_info.qnn_data_type, scale_value, scale_data));
    // Use an explicit broadcast-compatible rank to avoid any rank-mismatch checks on HTP.
    std::vector<uint32_t> scalar_shape(rank, 1);
    QnnTensorWrapper scale_tensor(scale_name,
                                  QNN_TENSOR_TYPE_STATIC,
                                  input_info.qnn_data_type,
                                  QnnQuantParamsWrapper(),
                                  std::move(scalar_shape),
                                  std::move(scale_data));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_tensor)),
                  "Failed to add scale tensor.");
  }

  if (!requires_rank3_reshape) {
    // Final Multiply produces the ONNX output tensor directly (handled by ProcessOutputs).
    return ProcessOutputs(qnn_model_wrapper, node_unit,
                          {sqrt_or_pool_out_name, scale_name}, {},
                          logger, do_op_validation, QNN_OP_ELEMENT_WISE_MULTIPLY);
  }

  // Rank-3 path: scaled output is intermediate, then Reshape back to rank-3 produces the ONNX output.
  const std::string scaled_out_name = utils::UniqueNameGenerator().New(node_unit.Name(), "_scaled");
  {
    QnnTensorWrapper scaled_out_tensor(scaled_out_name,
                                       QNN_TENSOR_TYPE_NATIVE,
                                       input_info.qnn_data_type,
                                       output_info.quant_param.Copy(),
                                       std::vector<uint32_t>(qnn_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scaled_out_tensor)),
                  "Failed to add scaled intermediate tensor.");
  }
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_MULTIPLY),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_ELEMENT_WISE_MULTIPLY,
                    {sqrt_or_pool_out_name, scale_name},
                    {scaled_out_name},
                    {},
                    do_op_validation),
                "Failed to create final scale (Multiply) node.");

  const std::string& final_output_name = node_unit.Outputs()[0].name;
  const bool final_is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  const Qnn_TensorType_t final_output_tensor_type =
      final_is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper final_output_tensor(final_output_name,
                                       final_output_tensor_type,
                                       output_info.qnn_data_type,
                                       output_info.quant_param.Copy(),
                                       std::vector<uint32_t>(output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)),
                "Failed to add final output tensor.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_RESHAPE),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {scaled_out_name},
                                                {final_output_name},
                                                {},
                                                do_op_validation),
                "Failed to create reshape-after node for LpPool rank-3 path.");

  return Ort::Status();
}

void CreateLpPoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<LpPoolOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
