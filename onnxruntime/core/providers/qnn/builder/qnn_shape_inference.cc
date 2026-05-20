// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/providers/qnn/builder/qnn_shape_inference.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

bool TryPropagateShapeOverrides(QnnModelWrapper& qmw, const OrtNodeUnit& node_unit) {
  // Check whether any output already needs an override (dynamic from ORT + not yet set).
  bool any_needs_override = false;
  for (const auto& output : node_unit.Outputs()) {
    std::vector<uint32_t> dummy;
    if (!QnnModelWrapper::GetOnnxShape(output.shape, dummy) &&
        qmw.GetTensorShapeOverride(output.name) == nullptr) {
      any_needs_override = true;
      break;
    }
  }
  if (!any_needs_override) return false;

  // Get an input's effective shape (ORT shape overridden by the shape-override map if present).
  // Returns nullopt if the shape cannot be determined.
  auto get_input_shape = [&](size_t idx) -> std::optional<std::vector<uint32_t>> {
    const auto& inputs = node_unit.Inputs();
    if (idx >= inputs.size()) return std::nullopt;
    TensorInfo info;
    if (!qmw.GetTensorInfo(inputs[idx], info).IsOK()) return std::nullopt;
    return info.shape;
  };

  // Register a shape override for output[out_idx] only if the shape is still dynamic.
  bool any_registered = false;
  auto try_register = [&](size_t out_idx, const std::vector<uint32_t>& shape) {
    const auto& outputs = node_unit.Outputs();
    if (out_idx >= outputs.size()) return;
    const std::string& name = outputs[out_idx].name;

    // Skip if ORT already has a static shape or we already have an override.
    std::vector<uint32_t> dummy;
    if (QnnModelWrapper::GetOnnxShape(outputs[out_idx].shape, dummy)) return;
    if (qmw.GetTensorShapeOverride(name) != nullptr) return;

    std::vector<int64_t> shape_i64(shape.begin(), shape.end());
    qmw.SetTensorShapeOverride(name, std::move(shape_i64));
    any_registered = true;
  };

  // Read an int64 vector from a constant initializer input (for opset-13+ ops where axes are
  // passed as a tensor rather than an attribute).
  auto read_const_int64_input = [&](size_t idx) -> std::optional<std::vector<int64_t>> {
    const auto& inputs = node_unit.Inputs();
    if (idx >= inputs.size()) return std::nullopt;
    const OrtValueInfo* tensor = qmw.GetConstantTensor(inputs[idx].name);
    if (tensor == nullptr) return std::nullopt;
    std::vector<uint8_t> bytes;
    if (!qmw.UnpackInitializerData(tensor, bytes).IsOK()) return std::nullopt;
    if (bytes.empty() || bytes.size() % sizeof(int64_t) != 0) return std::nullopt;
    const size_t n = bytes.size() / sizeof(int64_t);
    std::vector<int64_t> values(n);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
  };

  const std::string& op_type = node_unit.OpType();
  OrtNodeAttrHelper attrs(node_unit);

  if (op_type == "Cast") {
    auto s = get_input_shape(0);
    if (s) try_register(0, *s);

  } else if (op_type == "Identity") {
    auto s = get_input_shape(0);
    if (s) try_register(0, *s);

  } else if (op_type == "Transpose") {
    auto s = get_input_shape(0);
    if (!s) return false;
    std::vector<int64_t> perm = attrs.Get("perm", std::vector<int64_t>{});
    if (perm.empty()) {
      perm.resize(s->size());
      std::iota(perm.rbegin(), perm.rend(), int64_t{0});
    }
    if (perm.size() != s->size()) return false;
    std::vector<uint32_t> out_shape(s->size());
    for (size_t i = 0; i < perm.size(); ++i) {
      out_shape[i] = (*s)[static_cast<size_t>(perm[i])];
    }
    try_register(0, out_shape);

  } else if (op_type == "Gather") {
    // output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:]
    auto data_shape = get_input_shape(0);
    auto idx_shape = get_input_shape(1);
    if (!data_shape || !idx_shape) return false;

    int64_t rank = static_cast<int64_t>(data_shape->size());
    int64_t axis = attrs.Get("axis", int64_t{0});
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return false;

    std::vector<uint32_t> out_shape;
    for (int64_t i = 0; i < axis; ++i) out_shape.push_back((*data_shape)[static_cast<size_t>(i)]);
    for (auto d : *idx_shape) out_shape.push_back(d);
    for (int64_t i = axis + 1; i < rank; ++i) out_shape.push_back((*data_shape)[static_cast<size_t>(i)]);
    try_register(0, out_shape);

  } else if (op_type == "GatherElements") {
    // Output shape = indices shape (same rank and shape as data, element-wise gather).
    auto idx_shape = get_input_shape(1);
    if (idx_shape) try_register(0, *idx_shape);

  } else if (op_type == "GatherND") {
    // output shape: indices.shape[:-1] + data.shape[batch_dims + indices.shape[-1]:]
    auto data_shape = get_input_shape(0);
    auto idx_shape = get_input_shape(1);
    if (!data_shape || !idx_shape || idx_shape->empty()) return false;

    int64_t batch_dims = attrs.Get("batch_dims", int64_t{0});
    uint32_t slice_depth = idx_shape->back();
    size_t data_start = static_cast<size_t>(batch_dims) + static_cast<size_t>(slice_depth);
    if (data_start > data_shape->size()) return false;

    std::vector<uint32_t> out_shape;
    for (size_t i = 0; i + 1 < idx_shape->size(); ++i) out_shape.push_back((*idx_shape)[i]);
    for (size_t i = data_start; i < data_shape->size(); ++i) out_shape.push_back((*data_shape)[i]);
    try_register(0, out_shape);

  } else if (op_type == "ScatterElements" || op_type == "Scatter") {
    // Output shape = data (input[0]) shape.
    auto s = get_input_shape(0);
    if (s) try_register(0, *s);

  } else if (op_type == "ScatterND") {
    // Output shape = data (input[0]) shape.
    auto s = get_input_shape(0);
    if (s) try_register(0, *s);

  } else if (op_type == "Squeeze") {
    auto s = get_input_shape(0);
    if (!s) return false;

    // In opset < 13, axes is an attribute; in opset >= 13 it is input[1].
    std::optional<std::vector<int64_t>> axes = attrs.GetInt64s("axes");
    if (!axes) axes = read_const_int64_input(1);  // opset 13+ input tensor
    if (!axes) return false;

    int64_t rank = static_cast<int64_t>(s->size());
    std::unordered_set<int64_t> axes_set;
    for (auto ax : *axes) {
      if (ax < 0) ax += rank;
      axes_set.insert(ax);
    }
    std::vector<uint32_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
      if (!axes_set.count(i)) out_shape.push_back((*s)[static_cast<size_t>(i)]);
    }
    try_register(0, out_shape);

  } else if (op_type == "Unsqueeze") {
    auto s = get_input_shape(0);
    if (!s) return false;

    std::optional<std::vector<int64_t>> axes = attrs.GetInt64s("axes");
    if (!axes) axes = read_const_int64_input(1);
    if (!axes) return false;

    int64_t new_rank = static_cast<int64_t>(s->size()) + static_cast<int64_t>(axes->size());
    std::unordered_set<int64_t> axes_set;
    for (auto ax : *axes) {
      if (ax < 0) ax += new_rank;
      axes_set.insert(ax);
    }
    std::vector<uint32_t> out_shape;
    int src_i = 0;
    for (int64_t i = 0; i < new_rank; ++i) {
      if (axes_set.count(i)) {
        out_shape.push_back(1);
      } else {
        out_shape.push_back((*s)[static_cast<size_t>(src_i++)]);
      }
    }
    try_register(0, out_shape);

  } else if (op_type == "Flatten") {
    auto s = get_input_shape(0);
    if (!s) return false;
    int64_t axis = attrs.Get("axis", int64_t{1});
    int64_t rank = static_cast<int64_t>(s->size());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis > rank) return false;
    uint32_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= (*s)[static_cast<size_t>(i)];
    for (int64_t i = axis; i < rank; ++i) inner *= (*s)[static_cast<size_t>(i)];
    try_register(0, {outer, inner});

  } else if (op_type == "Concat") {
    int64_t axis = attrs.Get("axis", int64_t{0});
    auto first_shape = get_input_shape(0);
    if (!first_shape) return false;
    if (axis < 0) axis += static_cast<int64_t>(first_shape->size());
    if (axis < 0 || static_cast<size_t>(axis) >= first_shape->size()) return false;

    std::vector<uint32_t> out_shape(*first_shape);
    const auto& inputs = node_unit.Inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      auto s = get_input_shape(i);
      if (!s || s->size() != first_shape->size()) return false;
      out_shape[static_cast<size_t>(axis)] += (*s)[static_cast<size_t>(axis)];
    }
    try_register(0, out_shape);
  }

  return any_registered;
}

}  // namespace qnn
}  // namespace onnxruntime
