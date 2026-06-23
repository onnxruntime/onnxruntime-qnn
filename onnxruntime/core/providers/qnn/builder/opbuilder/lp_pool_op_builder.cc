// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

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
//   p = 2:  x -> Multiply(x, x) -> AvgPool -> SquareRoot -> Multiply(scale = sqrt(count_real))
//   p = 1:  x -> Abs(x)         -> AvgPool ->            Multiply(scale = count_real)
//
// AvgPool runs with count_pad_for_edges = false (the QNN default), so its denominator is
// count_real (the number of non-padding elements) per output position. The per-position static
// scale tensor compensates so the final result equals sqrt(Σx²) / Σ|x| matching ONNX LpPool
// semantics, including for ceil-mode boundary windows where count_real may be 0.
//
// Rank-3 inputs are bracketed by Reshape (NCL <-> NC1L) and use the 2D pool path. LpPool is
// registered as layout-sensitive in QnnEp::ShouldConvertDataLayoutForOpImpl, so the op builder
// receives NHWC tensors after the layout transformer runs.
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

  // QDQ pattern is not supported in this PR. Reject non-float inputs here so QDQ-wrapped LpPool
  // nodes (which the ORT QDQ-fusion path can hand to the op builder with quantized input dtypes)
  // fall back to the ORT CPU EP at partitioning time, instead of failing later in the
  // scale-tensor encoder with an opaque "unsupported input dtype" message.
  RETURN_IF(inputs[0].type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                inputs[0].type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
                inputs[0].type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
            "QNN LpPool only supports float32, float16, and bfloat16 inputs (QDQ not implemented).");

  RETURN_IF(node_unit.Outputs().size() != 1, "QNN LpPool only supports 1 output.");

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape");

  const size_t rank = input_shape.size();
  RETURN_IF_NOT(rank >= 3 && rank <= 5, "QNN LpPool only supports input rank 3, 4, or 5.");

  // Rank-5 (3D spatial) requires QNN_OP_POOL_AVG_3D, which is not supported on the QNN GPU backend.
  // CPU and HTP have PoolAvg3d kernels; rank-5 LpPool falls back to the ORT CPU EP on GPU.
  RETURN_IF(rank == 5 && IsGpuBackend(qnn_model_wrapper.GetQnnBackendType()),
            "QNN LpPool: rank-5 (3D pooling) is not supported on the QNN GPU backend.");

  // On HTP, PoolAvg3d fails NHWC dry-run validation for native FP32 / FP16 inputs. (cf. the
  // rank-5 PoolMax3d NPU rejection in pool_op_builder.cc::PoolOpBuilder::IsOpSupported, lines
  // 146-148 — same rank==5 && NPU rejection shape, different pool-op family.)
  // BF16 (htp_bf16_enable=1) does have a working PoolAvg3d kernel on V81+ HTP, so we allow rank-5
  // in that mode. QDQ rank-5 support is deferred to the QDQ follow-up PR.
  const bool htp_bf16_mode =
      IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) &&
      qnn_model_wrapper.GetModelSettings().htp_bf16_enable;
  RETURN_IF(rank == 5 && IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) && !htp_bf16_mode,
            "QNN LpPool: rank-5 (3D pooling) is not supported on HTP for FP32/FP16. "
            "Use BF16 (htp_bf16_enable=1) or fall back to the ORT CPU EP.");

  OrtNodeAttrHelper node_helper(node_unit);

  const auto p = node_helper.Get("p", static_cast<int64_t>(2));
  RETURN_IF(p != 1 && p != 2, "QNN LpPool only supports p=1 or p=2.");

  const auto ceil_mode = node_helper.Get("ceil_mode", static_cast<int64_t>(0));
  if (ceil_mode != 0) {
    // This decomposition lowers LpPool onto QNN PoolAvg2d / PoolAvg3d, so ceil_mode is realized through that AvgPool's rounding_mode
    // param (QNN_OP_POOL_AVG_2D_PARAM_ROUNDING_MODE;
    // Two backend-specific problems make ceil_mode=1 unsafe outside GPU:
    //  - QNN CPU PoolAvg2d/3d silently ignores rounding_mode and emits a floor-shape output,
    //    leaving the extra ceil-mode positions filled with garbage / NaN.
    //  - QNN HTP PoolAvg2d/3d reads out-of-bounds memory at ceil-mode boundary windows (positions
    //    whose window extends past the input), producing NaN / garbage regardless of the
    //    count_pad_for_edges setting.
    // Only the QNN GPU backend honors rounding_mode correctly,
    // so reject ceil_mode=1 elsewhere.
    RETURN_IF(IsCpuBackend(qnn_model_wrapper.GetQnnBackendType()) ||
                  IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()),
              "QNN LpPool: ceil_mode=1 is only supported on the QNN GPU backend.");
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
  // 1. Resolve common pooling attributes (kernel_shape, strides, dilations, pads, auto_pad,
  //    ceil_mode) via the shared helper used by pool_op_builder. The helper expands 1D->2D and
  //    converts SAME_UPPER/SAME_LOWER auto_pad to explicit pads.
  // ------------------------------------------------------------------------------------------------
  std::vector<uint32_t> filter_size, stride, dilations, pad_amount;
  int32_t rounding_mode = 0;
  RETURN_IF_ERROR(ResolvePoolAttributes(node_helper,
                                        qnn_input_shape,
                                        qnn_output_shape,
                                        filter_size,
                                        stride,
                                        dilations,
                                        pad_amount,
                                        rounding_mode));
  RETURN_IF_NOT(filter_size.size() == spatial_rank,
                "QNN LpPool: kernel_shape rank mismatch with input spatial rank.");
  RETURN_IF_NOT(pad_amount.size() == spatial_rank * 2,
                "QNN LpPool: pads rank mismatch with input spatial rank.");

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
  // 3. Compute per-output-position count_real (number of non-padding input elements that fall in
  //    each pooling window). AvgPool runs with count_pad_for_edges=0 (default), so its denominator
  //    is count_real per window. The static scale tensor below multiplies by sqrt(count_real)
  //    (p=2) or count_real (p=1) to recover ONNX LpPool semantics: sqrt(Σx²) / Σ|x| over the
  //    real (non-padding) elements in each window.
  //
  //    pad_amount layout after ReArrangePads is [begin0, end0, begin1, end1, ...]; the begin
  //    for spatial axis a is at index 2*a.
  // ------------------------------------------------------------------------------------------------
  std::vector<std::vector<int64_t>> per_axis_counts(spatial_rank);
  for (size_t a = 0; a < spatial_rank; ++a) {
    const int64_t in_dim = static_cast<int64_t>(qnn_input_shape[a + 1]);
    const int64_t out_dim = static_cast<int64_t>(qnn_output_shape[a + 1]);
    const int64_t k = static_cast<int64_t>(filter_size[a]);
    const int64_t s = static_cast<int64_t>(stride[a]);
    const int64_t pb = static_cast<int64_t>(pad_amount[2 * a]);
    per_axis_counts[a].resize(out_dim);
    for (int64_t o = 0; o < out_dim; ++o) {
      const int64_t start = o * s - pb;
      const int64_t end = std::min(start + k, in_dim);
      const int64_t start_clamped = std::max(start, int64_t{0});
      per_axis_counts[a][o] = std::max(end - start_clamped, int64_t{0});
    }
  }

  // ------------------------------------------------------------------------------------------------
  // 4. Op A: Multiply(x, x) for p=2, or Abs(x) for p=1.
  // ------------------------------------------------------------------------------------------------
  const std::string preprocess_out_name =
      utils::UniqueNameGenerator().New(node_unit.Name(), p_value == 2 ? "_squared" : "_abs");
  {
    // TODO(QDQ follow-up): QDQ is rejected in IsOpSupported, so the input is float and
    // quant_param.Copy() is empty here. When adding QDQ support, this intermediate tensor will
    // need its own quant params: x*x has a different value range than x, so reusing input
    // quant params would be incorrect for the p=2 (Multiply) path. (For p=1 / Abs, |x| has a
    // tighter range too.)
    QnnTensorWrapper preprocess_tensor(preprocess_out_name,
                                       QNN_TENSOR_TYPE_NATIVE,
                                       input_info.qnn_data_type,
                                       input_info.quant_param.Copy(),
                                       std::vector<uint32_t>(qnn_input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(preprocess_tensor)),
                  "Failed to add preprocess tensor.");
  }

  if (p_value == 2) {
    std::string square_node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_BINARY);
    Qnn_Scalar_t square_op_scalar = QNN_SCALAR_INIT;
    square_op_scalar.dataType = QNN_DATATYPE_UINT_32;
    square_op_scalar.uint32Value = QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY;
    QnnParamWrapper square_op_param(node_unit.Index(), square_node_name,
                                    QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, square_op_scalar);
    std::string square_op_param_name = square_op_param.GetParamTensorName();
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(square_op_param)),
                  "Failed to add square (Multiply) operation param.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      square_node_name,
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_ELEMENT_WISE_BINARY,
                      {input_names[0], input_names[0]},
                      {preprocess_out_name},
                      {square_op_param_name},
                      do_op_validation),
                  "Failed to create square (Multiply) node.");
  } else {
    std::string abs_node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_UNARY);
    Qnn_Scalar_t abs_op_scalar = QNN_SCALAR_INIT;
    abs_op_scalar.dataType = QNN_DATATYPE_UINT_32;
    abs_op_scalar.uint32Value = QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS;
    QnnParamWrapper abs_op_param(node_unit.Index(), abs_node_name,
                                 QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION, abs_op_scalar);
    std::string abs_op_param_name = abs_op_param.GetParamTensorName();
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(abs_op_param)), "Failed to add Abs operation param.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      abs_node_name,
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_ELEMENT_WISE_UNARY,
                      {input_names[0]},
                      {preprocess_out_name},
                      {abs_op_param_name},
                      do_op_validation),
                  "Failed to create Abs node.");
  }

  // ------------------------------------------------------------------------------------------------
  // 5. Op B: PoolAvg2d / PoolAvg3d. count_pad_for_edges is left at the QNN default (false), so the
  //    AvgPool denominator is count_real per output position; the per-position scale tensor below
  //    compensates back to ONNX LpPool semantics.
  // ------------------------------------------------------------------------------------------------
  const char* pool_op = is_3d_pool ? QNN_OP_POOL_AVG_3D : QNN_OP_POOL_AVG_2D;
  const char* p_filter = is_3d_pool ? QNN_OP_POOL_AVG_3D_PARAM_FILTER_SIZE : QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE;
  const char* p_stride = is_3d_pool ? QNN_OP_POOL_AVG_3D_PARAM_STRIDE : QNN_OP_POOL_AVG_2D_PARAM_STRIDE;
  const char* p_pad = is_3d_pool ? QNN_OP_POOL_AVG_3D_PARAM_PAD_AMOUNT : QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT;
  const char* p_round = is_3d_pool ? QNN_OP_POOL_AVG_3D_PARAM_ROUNDING_MODE : QNN_OP_POOL_AVG_2D_PARAM_ROUNDING_MODE;

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
  // ceil_mode = 1 is rejected on CPU and HTP backends in IsOpSupported; only the QNN GPU backend
  // honors rounding_mode correctly. rounding_mode was populated by ResolvePoolAttributes from the
  // ONNX ceil_mode attribute.
  if (rounding_mode != 0) {
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                           static_cast<uint32_t>(rounding_mode), p_round, pool_param_names));
  }
  // count_pad_for_edges intentionally left at the QNN default (false). The denominator becomes
  // count_real per window; the per-position scale tensor below compensates.

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
    std::string sqrt_node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_UNARY);
    Qnn_Scalar_t sqrt_op_scalar = QNN_SCALAR_INIT;
    sqrt_op_scalar.dataType = QNN_DATATYPE_UINT_32;
    sqrt_op_scalar.uint32Value = QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT;
    QnnParamWrapper sqrt_op_param(node_unit.Index(), sqrt_node_name,
                                  QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION, sqrt_op_scalar);
    std::string sqrt_op_param_name = sqrt_op_param.GetParamTensorName();
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(sqrt_op_param)),
                  "Failed to add SquareRoot operation param.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      sqrt_node_name,
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_ELEMENT_WISE_UNARY,
                      {pool_out_name}, {sqrt_out_name}, {sqrt_op_param_name}, do_op_validation),
                  "Failed to create SquareRoot node.");
    sqrt_or_pool_out_name = sqrt_out_name;
  }

  // ------------------------------------------------------------------------------------------------
  // 7. Build per-position static scale tensor and emit final Multiply.
  //    Shape broadcasts over batch and channel: rank-4 NHWC -> [1, outH, outW, 1];
  //    rank-5 NDHWC -> [1, outD, outH, outW, 1].
  //    Values: sqrt(count_real[oh, ow, ...]) for p=2; count_real[oh, ow, ...] for p=1.
  //    Together with AvgPool's count_real-based denominator this recovers the ONNX LpPool result.
  // ------------------------------------------------------------------------------------------------
  const std::string scale_name = utils::UniqueNameGenerator().New(node_unit.Name(), "_scale");
  {
    // Output spatial extents (NHWC indices [1 .. rank-2]).
    std::vector<uint32_t> spatial_extents(spatial_rank);
    size_t total_positions = 1;
    for (size_t a = 0; a < spatial_rank; ++a) {
      spatial_extents[a] = qnn_output_shape[a + 1];
      total_positions *= spatial_extents[a];
    }

    // Walk the spatial grid in row-major order, compute count_real, encode the scale value.
    std::vector<float> scale_values;
    scale_values.reserve(total_positions);
    std::vector<size_t> idx(spatial_rank, 0);
    for (size_t i = 0; i < total_positions; ++i) {
      int64_t count_real = 1;
      for (size_t a = 0; a < spatial_rank; ++a) count_real *= per_axis_counts[a][idx[a]];
      const float c = static_cast<float>(count_real);
      scale_values.push_back(p_value == 2 ? std::sqrt(c) : c);
      for (size_t a = spatial_rank; a-- > 0;) {
        if (++idx[a] < spatial_extents[a]) break;
        idx[a] = 0;
      }
    }

    // Encode per input dtype (matches the chain dtype so no Cast is inserted).
    const Qnn_DataType_t dtype = input_info.qnn_data_type;
    std::vector<uint8_t> scale_data;
    if (dtype == QNN_DATATYPE_FLOAT_16) {
      scale_data.resize(total_positions * sizeof(uint16_t));
      for (size_t i = 0; i < total_positions; ++i) {
        Ort::Float16_t v(scale_values[i]);
        std::memcpy(scale_data.data() + i * sizeof(uint16_t), &v.val, sizeof(uint16_t));
      }
    } else if (dtype == QNN_DATATYPE_BFLOAT_16) {
      scale_data.resize(total_positions * sizeof(uint16_t));
      for (size_t i = 0; i < total_positions; ++i) {
        Ort::BFloat16_t v(scale_values[i]);
        std::memcpy(scale_data.data() + i * sizeof(uint16_t), &v.val, sizeof(uint16_t));
      }
    } else if (dtype == QNN_DATATYPE_FLOAT_32) {
      scale_data.resize(total_positions * sizeof(float));
      std::memcpy(scale_data.data(), scale_values.data(), scale_data.size());
    } else {
      return Ort::Status("QNN LpPool: unsupported input dtype for scale tensor.", ORT_INVALID_ARGUMENT);
    }

    // Broadcast-compatible shape: 1 on batch and channel axes, spatial extents in between.
    std::vector<uint32_t> scale_shape(rank, 1);
    for (size_t a = 0; a < spatial_rank; ++a) scale_shape[a + 1] = spatial_extents[a];

    QnnTensorWrapper scale_tensor(scale_name,
                                  QNN_TENSOR_TYPE_STATIC,
                                  dtype,
                                  QnnQuantParamsWrapper(),
                                  std::move(scale_shape),
                                  std::move(scale_data));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_tensor)),
                  "Failed to add per-position scale tensor.");
  }

  if (!requires_rank3_reshape) {
    // Final Multiply produces the ONNX output tensor directly (handled by ProcessOutputs).
    Qnn_Scalar_t mul_op_scalar = QNN_SCALAR_INIT;
    mul_op_scalar.dataType = QNN_DATATYPE_UINT_32;
    mul_op_scalar.uint32Value = QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY;
    QnnParamWrapper mul_op_param(node_unit.Index(), node_unit.Name(),
                                 QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, mul_op_scalar);
    std::string mul_op_param_name = mul_op_param.GetParamTensorName();
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(mul_op_param)),
                  "Failed to add final scale (Multiply) operation param.");
    return ProcessOutputs(qnn_model_wrapper, node_unit,
                          {sqrt_or_pool_out_name, scale_name}, {mul_op_param_name},
                          logger, do_op_validation, QNN_OP_ELEMENT_WISE_BINARY);
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
  std::string scale_mul_node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_BINARY);
  Qnn_Scalar_t scale_mul_op_scalar = QNN_SCALAR_INIT;
  scale_mul_op_scalar.dataType = QNN_DATATYPE_UINT_32;
  scale_mul_op_scalar.uint32Value = QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY;
  QnnParamWrapper scale_mul_op_param(node_unit.Index(), scale_mul_node_name,
                                     QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, scale_mul_op_scalar);
  std::string scale_mul_op_param_name = scale_mul_op_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(scale_mul_op_param)),
                "Failed to add final scale (Multiply) operation param.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    scale_mul_node_name,
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_ELEMENT_WISE_BINARY,
                    {sqrt_or_pool_out_name, scale_name},
                    {scaled_out_name},
                    {scale_mul_op_param_name},
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
