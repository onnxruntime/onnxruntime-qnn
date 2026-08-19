// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/window_partition_reverse_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kOpReshape[] = "Reshape";
constexpr char kOpTranspose[] = "Transpose";
constexpr size_t kRank6 = 6;

// The window partition/reverse Transpose only swaps axes 2 and 3 of a rank-6 tensor.
const std::array<int64_t, 6> kWindowPerm{0, 1, 3, 2, 4, 5};

using Params = WindowPartitionReverseFusion::Params;
using Direction = WindowPartitionReverseFusion::Direction;

// Match Reshape -> Transpose(perm=[0,1,3,2,4,5]) -> Reshape with rank-6 intermediates and
// resolve the window geometry (B, Gh, Gw, ws, C). Returns nullopt if not a window chain.
std::optional<std::pair<std::array<const OrtNodeUnit*, 3>, Params>> MatchWindowChain(
    const QnnModelWrapper& qmw,
    const OrtNodeUnit& reshape1,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  if (reshape1.OpType() != kOpReshape || reshape1.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  const std::array<std::string_view, 1> transpose_type{kOpTranspose};
  const OrtNodeUnit* transpose = GetOnlyChildOfType(qmw, reshape1, transpose_type,
                                                    node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose == nullptr) {
    return std::nullopt;
  }
  const std::array<std::string_view, 1> reshape_type{kOpReshape};
  const OrtNodeUnit* reshape2 = GetOnlyChildOfType(qmw, *transpose, reshape_type,
                                                   node_to_node_unit, node_unit_to_qnn_node_group);
  if (reshape2 == nullptr) {
    return std::nullopt;
  }

  // Perm must be exactly the window axis-swap.
  {
    OrtNodeAttrHelper helper(*transpose);
    const std::vector<int64_t> perm = helper.Get("perm", std::vector<int64_t>{});
    if (perm.size() != kRank6 ||
        !std::equal(perm.begin(), perm.end(), kWindowPerm.begin())) {
      return std::nullopt;
    }
  }

  std::vector<uint32_t> in_shape, t1_shape, t2_shape, out_shape;
  if (!qmw.GetOnnxShape(reshape1.Inputs()[0].shape, in_shape) ||
      !qmw.GetOnnxShape(reshape1.Outputs()[0].shape, t1_shape) ||
      !qmw.GetOnnxShape(transpose->Outputs()[0].shape, t2_shape) ||
      !qmw.GetOnnxShape(reshape2->Outputs()[0].shape, out_shape)) {
    return std::nullopt;
  }
  if (t1_shape.size() != kRank6 || t2_shape.size() != kRank6) {
    return std::nullopt;
  }

  Params p;
  // Partition: t1 = [B, Gh, ws, Gw, ws, C]  (in_rank=4 [B,H,W,C], out_rank=3).
  // Reverse:   t1 = [B, Gh, Gw, ws, ws, C]  (in_rank=3, out_rank=4 [B,H,W,C]).
  if (in_shape.size() == 4 && out_shape.size() == 3) {
    p.direction = Direction::kPartition;
    p.batch = t1_shape[0];
    p.gh = t1_shape[1];
    p.ws = t1_shape[2];
    p.gw = t1_shape[3];
    const uint32_t ws2 = t1_shape[4];
    p.channels = t1_shape[5];
    // Consistency: square window, and dims match the [B,H,W,C] input / [nW,ws*ws,C] output.
    if (p.ws != ws2 || p.ws == 0 || p.gh == 0 || p.gw == 0 || p.channels == 0) {
      return std::nullopt;
    }
    if (in_shape[0] != p.batch || in_shape[1] != p.gh * p.ws || in_shape[2] != p.gw * p.ws ||
        in_shape[3] != p.channels) {
      return std::nullopt;
    }
    if (out_shape[0] != p.batch * p.gh * p.gw || out_shape[1] != p.ws * p.ws ||
        out_shape[2] != p.channels) {
      return std::nullopt;
    }
  } else if (in_shape.size() == 3 && out_shape.size() == 4) {
    p.direction = Direction::kReverse;
    p.batch = t1_shape[0];
    p.gh = t1_shape[1];
    p.gw = t1_shape[2];
    p.ws = t1_shape[3];
    const uint32_t ws2 = t1_shape[4];
    p.channels = t1_shape[5];
    if (p.ws != ws2 || p.ws == 0 || p.gh == 0 || p.gw == 0 || p.channels == 0) {
      return std::nullopt;
    }
    if (in_shape[0] != p.batch * p.gh * p.gw || in_shape[1] != p.ws * p.ws ||
        in_shape[2] != p.channels) {
      return std::nullopt;
    }
    if (out_shape[0] != p.batch || out_shape[1] != p.gh * p.ws || out_shape[2] != p.gw * p.ws ||
        out_shape[3] != p.channels) {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }

  // The rewrite folds window/channel axes together (ws*C and ws*ws*C) and merges the leading
  // block axes (B*Gh*Gw). Compute those products in int64_t and reject geometry whose folded
  // dims cannot be represented as the uint32_t QNN shapes require.
  {
    const int64_t ws64 = static_cast<int64_t>(p.ws);
    const int64_t c64 = static_cast<int64_t>(p.channels);
    const int64_t blocks64 =
        static_cast<int64_t>(p.batch) * static_cast<int64_t>(p.gh) * static_cast<int64_t>(p.gw);
    const int64_t max_u32 = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    if (ws64 * c64 > max_u32 || ws64 * ws64 * c64 > max_u32 || blocks64 > max_u32) {
      return std::nullopt;
    }
  }

  // The two rank-6 intermediates are absorbed by the rewrite and never re-emitted, so
  // neither may be a graph output -- otherwise the tensor would vanish from the QNN graph.
  if (qmw.IsGraphOutput(reshape1.Outputs()[0].name) ||
      qmw.IsGraphOutput(transpose->Outputs()[0].name)) {
    return std::nullopt;
  }

  p.input_name = reshape1.Inputs()[0].name;
  p.output_name = reshape2->Outputs()[0].name;
  return std::make_pair(std::array<const OrtNodeUnit*, 3>{&reshape1, transpose, reshape2}, p);
}

// Build the rank-<=4 op sequence. On do_op_validation the ops are validated on QNN; otherwise
// they are appended to the model. The final Reshape writes p.output_name (the original name).
Ort::Status EmitChain(QnnModelWrapper& qmw,
                      const OrtNodeUnit& reshape1,
                      const Params& p,
                      bool do_op_validation) {
  const OrtNodeUnitIODef& in_def = reshape1.Inputs()[0];
  TensorInfo in_info = {};
  RETURN_IF_ERROR(qmw.GetTensorInfo(in_def, in_info));
  const Qnn_DataType_t dt = in_info.qnn_data_type;
  const QnnQuantParamsWrapper& qp = in_info.quant_param;

  const uint32_t B = p.batch, Gh = p.gh, Gw = p.gw, ws = p.ws, C = p.channels;
  const uint32_t H = Gh * ws, W = Gw * ws;
  const size_t idx = reshape1.Index();
  const bool out_is_graph_output = qmw.IsGraphOutput(p.output_name);

  // Ensure the chain input tensor exists in the QNN graph.
  if (!qmw.IsQnnTensorWrapperExist(p.input_name)) {
    QnnTensorWrapper in_wrap;
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(in_def, in_wrap));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(in_wrap)), "Failed to add window-chain input tensor.");
  }

  const std::string& base = p.output_name;
  auto name = [&base](std::string_view suffix) {
    return utils::UniqueNameGenerator().New(base, std::string(suffix));
  };

  const std::vector<uint32_t> perm4{0u, 2u, 1u, 3u};

  if (p.direction == Direction::kPartition) {
    // x[B,H,W,C] -> [B,H,Gw,ws*C]
    const std::string r1 = name("_wp_r1");
    RETURN_IF_ERROR(qmw.AddReshapeNode(p.input_name, r1,
                                       {B, H, W, C}, {B, H, Gw, ws * C},
                                       dt, qp, do_op_validation, false, false));
    // -> transpose [B,Gw,H,ws*C]
    const std::string t1 = name("_wp_t1");
    RETURN_IF_ERROR(qmw.AddTransposeNode(idx, r1, t1,
                                         {B, H, Gw, ws * C}, perm4, {B, Gw, H, ws * C},
                                         dt, qp, do_op_validation, false, false));
    // -> [B,Gw,Gh,ws*ws*C]
    const std::string r2 = name("_wp_r2");
    RETURN_IF_ERROR(qmw.AddReshapeNode(t1, r2,
                                       {B, Gw, H, ws * C}, {B, Gw, Gh, ws * ws * C},
                                       dt, qp, do_op_validation, false, false));
    // -> transpose [B,Gh,Gw,ws*ws*C]
    const std::string t2 = name("_wp_t2");
    RETURN_IF_ERROR(qmw.AddTransposeNode(idx, r2, t2,
                                         {B, Gw, Gh, ws * ws * C}, perm4, {B, Gh, Gw, ws * ws * C},
                                         dt, qp, do_op_validation, false, false));
    // -> [B*Gh*Gw, ws*ws, C]  (final, original output name)
    RETURN_IF_ERROR(qmw.AddReshapeNode(t2, p.output_name,
                                       {B, Gh, Gw, ws * ws * C}, {B * Gh * Gw, ws * ws, C},
                                       dt, qp, do_op_validation, false, out_is_graph_output));
  } else {
    // x[B*Gh*Gw, ws*ws, C] -> [B,Gh,Gw,ws*ws*C]
    const std::string r1 = name("_wr_r1");
    RETURN_IF_ERROR(qmw.AddReshapeNode(p.input_name, r1,
                                       {B * Gh * Gw, ws * ws, C}, {B, Gh, Gw, ws * ws * C},
                                       dt, qp, do_op_validation, false, false));
    // -> transpose [B,Gw,Gh,ws*ws*C]
    const std::string t1 = name("_wr_t1");
    RETURN_IF_ERROR(qmw.AddTransposeNode(idx, r1, t1,
                                         {B, Gh, Gw, ws * ws * C}, perm4, {B, Gw, Gh, ws * ws * C},
                                         dt, qp, do_op_validation, false, false));
    // -> [B,Gw,H,ws*C]
    const std::string r2 = name("_wr_r2");
    RETURN_IF_ERROR(qmw.AddReshapeNode(t1, r2,
                                       {B, Gw, Gh, ws * ws * C}, {B, Gw, H, ws * C},
                                       dt, qp, do_op_validation, false, false));
    // -> transpose [B,H,Gw,ws*C]
    const std::string t2 = name("_wr_t2");
    RETURN_IF_ERROR(qmw.AddTransposeNode(idx, r2, t2,
                                         {B, Gw, H, ws * C}, perm4, {B, H, Gw, ws * C},
                                         dt, qp, do_op_validation, false, false));
    // -> [B,H,W,C]  (final, original output name)
    RETURN_IF_ERROR(qmw.AddReshapeNode(t2, p.output_name,
                                       {B, H, Gw, ws * C}, {B, H, W, C},
                                       dt, qp, do_op_validation, false, out_is_graph_output));
  }
  return Ort::Status();
}

}  // namespace

WindowPartitionReverseFusion::WindowPartitionReverseFusion(
    gsl::span<const OrtNodeUnit* const> node_units, Params params)
    : params_(std::move(params)) {
  if (node_units.size() != node_units_.size()) {
    ORT_CXX_API_THROW("WindowPartitionReverseFusion expects exactly 3 NodeUnits.", ORT_EP_FAIL);
  }
  std::copy(node_units.begin(), node_units.end(), node_units_.begin());
}

std::unique_ptr<IQnnNodeGroup> WindowPartitionReverseFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  // NPU-only: the win is replacing the rank-5/6 window Transpose with rank-4 ops on HTP.
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    return nullptr;
  }

  std::optional<std::pair<std::array<const OrtNodeUnit*, 3>, Params>> matched =
      MatchWindowChain(qnn_model_wrapper, reshape_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!matched.has_value()) {
    return nullptr;
  }

  if (!EmitChain(qnn_model_wrapper, reshape_node_unit, matched->second, /*do_op_validation=*/true).IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("WindowPartitionReverseFusion: match at Reshape '" + reshape_node_unit.Name() +
                 "' failed QNN validation; leaving unfused.")
                    .c_str());
    return nullptr;
  }

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
              (std::string("WindowPartitionReverseFusion: rewriting window ") +
               (matched->second.direction == Direction::kPartition ? "partition" : "reverse") +
               " at Reshape '" + reshape_node_unit.Name() + "' to rank-<=4 ops.")
                  .c_str());
  return std::make_unique<WindowPartitionReverseFusion>(
      gsl::span<const OrtNodeUnit* const>{matched->first.data(), matched->first.size()}, matched->second);
}

gsl::span<const OrtNodeUnit* const> WindowPartitionReverseFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status WindowPartitionReverseFusion::IsSupported(QnnModelWrapper& qnn_model_wrapper,
                                                      const Ort::Logger& /*logger*/) const {
  return EmitChain(qnn_model_wrapper, *node_units_[0], params_, /*do_op_validation=*/true);
}

Ort::Status WindowPartitionReverseFusion::AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                                            const Ort::Logger& /*logger*/) const {
  return EmitChain(qnn_model_wrapper, *node_units_[0], params_, /*do_op_validation=*/false);
}

}  // namespace qnn
}  // namespace onnxruntime
