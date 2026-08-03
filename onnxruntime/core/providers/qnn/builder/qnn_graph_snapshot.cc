// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_graph_snapshot.h"

#include <unordered_set>
#include <utility>

#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Ensure a SnapshotValueInfo exists for the given name and return it.
SnapshotValueInfo* EnsureValueInfo(SnapshotGraph& snap, const std::string& name) {
  if (name.empty()) {
    return nullptr;
  }
  auto it = snap.value_info_by_name.find(name);
  if (it != snap.value_info_by_name.end()) {
    return it->second;
  }
  auto vi = std::make_unique<SnapshotValueInfo>();
  vi->name = name;
  SnapshotValueInfo* raw = vi.get();
  snap.value_infos.push_back(std::move(vi));
  snap.value_info_by_name.emplace(name, raw);
  return raw;
}

// Fill elem_type / has_shape / shape from an OrtValueInfo's type info.
void CaptureTypeAndShape(const OrtApi& ort_api, const OrtValueInfo* ort_vi, SnapshotValueInfo& out) {
  const OrtTypeInfo* type_info = nullptr;
  if (ort_api.GetValueInfoTypeInfo(ort_vi, &type_info) != nullptr || type_info == nullptr) {
    return;
  }
  const OrtTensorTypeAndShapeInfo* ts = nullptr;
  if (ort_api.CastTypeInfoToTensorInfo(type_info, &ts) != nullptr || ts == nullptr) {
    return;
  }
  ort_api.GetTensorElementType(ts, &out.elem_type);

  if (ort_api.TensorTypeAndShape_HasShape(ts)) {
    size_t num_dims = 0;
    if (ort_api.GetDimensionsCount(ts, &num_dims) == nullptr) {
      out.shape.resize(num_dims);
      if (num_dims == 0 || ort_api.GetDimensions(ts, out.shape.data(), num_dims) == nullptr) {
        out.has_shape = true;
      } else {
        out.shape.clear();
      }
    }
  }
}

void CaptureAttributes(const OrtNode& node, SnapshotNode& snap_node) {
  Ort::ConstNode cn(&node);
  for (const Ort::ConstOpAttr& attr : cn.GetAttributes()) {
    SnapshotAttr sa;
    sa.name = attr.GetName();
    sa.type = attr.GetType();
    switch (sa.type) {
      case ORT_OP_ATTR_INT: {
        int64_t v = 0;
        if (attr.GetValue<int64_t>(v).IsOK()) sa.i = v;
        break;
      }
      case ORT_OP_ATTR_INTS: {
        std::vector<int64_t> v;
        if (attr.GetValueArray<int64_t>(v).IsOK()) sa.ints = std::move(v);
        break;
      }
      case ORT_OP_ATTR_FLOAT: {
        float v = 0.f;
        if (attr.GetValue<float>(v).IsOK()) sa.f = v;
        break;
      }
      case ORT_OP_ATTR_FLOATS: {
        std::vector<float> v;
        if (attr.GetValueArray<float>(v).IsOK()) sa.floats = std::move(v);
        break;
      }
      case ORT_OP_ATTR_STRING: {
        std::string v;
        if (attr.GetValue<std::string>(v).IsOK()) sa.s = std::move(v);
        break;
      }
      case ORT_OP_ATTR_STRINGS: {
        std::vector<std::string> v;
        if (attr.GetValueArray<std::string>(v).IsOK()) sa.strings = std::move(v);
        break;
      }
      default:
        // GRAPH / TENSOR / UNDEFINED: not needed for the float path (deferred: nested subgraphs).
        break;
    }
    snap_node.attributes.push_back(std::move(sa));
  }
}

}  // namespace

std::unique_ptr<SnapshotGraph> SnapshotGraphFromOrt(const OrtApi& ort_api,
                                                    const OrtGraph* graph,
                                                    const OrtNode* fused_node,
                                                    const Ort::Logger& logger) {
  auto snap = std::make_unique<SnapshotGraph>();

  const ORTCHAR_T* model_path = nullptr;
  if (ort_api.Graph_GetModelPath(graph, &model_path) == nullptr && model_path != nullptr) {
    snap->model_path = std::filesystem::path(model_path);
  }

  const char* graph_name = nullptr;
  if (ort_api.Graph_GetName(graph, &graph_name) == nullptr && graph_name != nullptr) {
    snap->name = graph_name;
  }

  // 1. Nodes, in declaration order.
  size_t num_nodes = 0;
  if (auto* st = ort_api.Graph_GetNumNodes(graph, &num_nodes)) {
    ort_api.ReleaseStatus(st);
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, "SnapshotGraphFromOrt: Graph_GetNumNodes failed.");
    return nullptr;
  }
  std::vector<const OrtNode*> ort_nodes(num_nodes);
  if (auto* st = ort_api.Graph_GetNodes(graph, ort_nodes.data(), ort_nodes.size())) {
    ort_api.ReleaseStatus(st);
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, "SnapshotGraphFromOrt: Graph_GetNodes failed.");
    return nullptr;
  }

  snap->nodes.reserve(num_nodes);
  snap->node_order.reserve(num_nodes);

  const auto capture_ios = [&](const OrtNode* ort_node, size_t num,
                               OrtStatus* (ORT_API_CALL* get_num)(const OrtNode*, size_t*),
                               OrtStatus* (ORT_API_CALL* get)(const OrtNode*, const OrtValueInfo**, size_t),
                               std::vector<SnapshotValueInfo*>& out) {
    (void)num;
    size_t count = 0;
    if (get_num(ort_node, &count) != nullptr) return;
    std::vector<const OrtValueInfo*> vis(count);
    if (count > 0 && get(ort_node, vis.data(), vis.size()) != nullptr) return;
    out.reserve(count);
    for (const OrtValueInfo* vi : vis) {
      if (vi == nullptr) {
        out.push_back(nullptr);
        continue;
      }
      const char* nm = nullptr;
      std::string name = (ort_api.GetValueInfoName(vi, &nm) == nullptr && nm) ? nm : std::string{};
      SnapshotValueInfo* svi = EnsureValueInfo(*snap, name);
      if (svi && svi->elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED && !svi->has_shape) {
        CaptureTypeAndShape(ort_api, vi, *svi);
      }
      out.push_back(svi);
    }
  };

  for (const OrtNode* ort_node : ort_nodes) {
    auto sn = std::make_unique<SnapshotNode>();
    Ort::ConstNode cn(ort_node);
    sn->op_type = cn.GetOperatorType();
    sn->domain = cn.GetDomain();
    sn->name = cn.GetName();
    sn->since_version = cn.GetSinceVersion();
    sn->id = cn.GetId();

    capture_ios(ort_node, 0, ort_api.Node_GetNumInputs, ort_api.Node_GetInputs, sn->inputs);
    capture_ios(ort_node, 0, ort_api.Node_GetNumOutputs, ort_api.Node_GetOutputs, sn->outputs);
    capture_ios(ort_node, 0, ort_api.Node_GetNumImplicitInputs, ort_api.Node_GetImplicitInputs,
                sn->implicit_inputs);

    CaptureAttributes(*ort_node, *sn);

    snap->node_order.push_back(sn.get());
    snap->nodes.push_back(std::move(sn));
  }

  // 2. Precompute producer/consumer edges.
  for (SnapshotNode* sn : snap->node_order) {
    for (size_t out_idx = 0; out_idx < sn->outputs.size(); ++out_idx) {
      if (SnapshotValueInfo* vi = sn->outputs[out_idx]) {
        vi->producer_node = sn;
        vi->producer_output_index = out_idx;
      }
    }
  }
  for (SnapshotNode* sn : snap->node_order) {
    for (size_t in_idx = 0; in_idx < sn->inputs.size(); ++in_idx) {
      if (SnapshotValueInfo* vi = sn->inputs[in_idx]) {
        vi->consumers.emplace_back(sn, static_cast<int64_t>(in_idx));
      }
    }
    for (SnapshotValueInfo* vi : sn->implicit_inputs) {
      if (vi) vi->consumers.emplace_back(sn, static_cast<int64_t>(-1));
    }
  }

  // 3. Graph inputs / outputs / initializers.
  const auto capture_graph_ios = [&](OrtStatus* (ORT_API_CALL* get_num)(const OrtGraph*, size_t*),
                                     OrtStatus* (ORT_API_CALL* get)(const OrtGraph*, const OrtValueInfo**, size_t),
                                     std::vector<SnapshotValueInfo*>& out) -> std::vector<const OrtValueInfo*> {
    size_t count = 0;
    if (get_num(graph, &count) != nullptr) return {};
    std::vector<const OrtValueInfo*> vis(count);
    if (count > 0 && get(graph, vis.data(), vis.size()) != nullptr) return {};
    out.reserve(count);
    for (const OrtValueInfo* vi : vis) {
      const char* nm = nullptr;
      std::string name = (ort_api.GetValueInfoName(vi, &nm) == nullptr && nm) ? nm : std::string{};
      SnapshotValueInfo* svi = EnsureValueInfo(*snap, name);
      if (svi) {
        if (svi->elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED && !svi->has_shape) {
          CaptureTypeAndShape(ort_api, vi, *svi);
        }
        out.push_back(svi);
      }
    }
    return vis;
  };

  capture_graph_ios(ort_api.Graph_GetNumInputs, ort_api.Graph_GetInputs, snap->graph_inputs);
  std::vector<const OrtValueInfo*> ort_outputs =
      capture_graph_ios(ort_api.Graph_GetNumOutputs, ort_api.Graph_GetOutputs, snap->graph_outputs);
  std::vector<const OrtValueInfo*> ort_inits =
      capture_graph_ios(ort_api.Graph_GetNumInitializers, ort_api.Graph_GetInitializers, snap->initializers);

  for (SnapshotValueInfo* vi : snap->graph_outputs) {
    if (vi) vi->is_graph_output = true;
  }

  // Initializer flags + copied bytes.
  for (size_t i = 0; i < snap->initializers.size(); ++i) {
    SnapshotValueInfo* vi = snap->initializers[i];
    if (!vi) continue;
    vi->is_initializer = true;

    const OrtValueInfo* ort_vi = ort_inits[i];
    bool is_const = false;
    if (ort_api.ValueInfo_IsConstantInitializer(ort_vi, &is_const) == nullptr) {
      vi->is_constant_initializer = is_const;
    }

    std::vector<uint8_t> bytes;
    Ort::Status unpack = qnn::utils::UnpackInitializerData(ort_api, ort_vi, snap->model_path, bytes);
    if (unpack.IsOK()) {
      vi->initializer_bytes = std::move(bytes);
    } else {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                  ("SnapshotGraphFromOrt: failed to copy initializer bytes for " + vi->name).c_str());
    }
  }

  // 4. The EP fused node.
  if (fused_node != nullptr) {
    auto fn = std::make_unique<SnapshotNode>();
    Ort::ConstNode cn(fused_node);
    fn->op_type = cn.GetOperatorType();
    fn->domain = cn.GetDomain();
    fn->name = cn.GetName();
    fn->since_version = cn.GetSinceVersion();
    fn->id = cn.GetId();
    capture_ios(fused_node, 0, ort_api.Node_GetNumInputs, ort_api.Node_GetInputs, fn->inputs);
    capture_ios(fused_node, 0, ort_api.Node_GetNumOutputs, ort_api.Node_GetOutputs, fn->outputs);
    snap->fused_node = std::move(fn);
  }

  return snap;
}

std::vector<SegmentDesc> SplitSnapshotIntoSegments(
    const SnapshotGraph& parent,
    const std::unordered_set<std::string>& gpu_only_names) {
  struct RawSegment {
    std::vector<SnapshotNode*> nodes;
    bool is_gpu = false;
  };
  std::vector<RawSegment> raw_segments;

  for (SnapshotNode* node : parent.node_order) {
    bool is_gpu = gpu_only_names.count(node->name) > 0;
    if (raw_segments.empty() || raw_segments.back().is_gpu != is_gpu) {
      raw_segments.push_back({{}, is_gpu});
    }
    raw_segments.back().nodes.push_back(node);
  }

  // Determine graph I/O per segment: a value is a segment input if produced outside, a segment
  // output if consumed outside or is a parent graph output.
  std::unordered_set<std::string> parent_output_names;
  for (SnapshotValueInfo* vi : parent.graph_outputs) {
    parent_output_names.insert(vi->name);
  }

  std::vector<SegmentDesc> result;
  result.reserve(raw_segments.size());

  for (size_t seg_idx = 0; seg_idx < raw_segments.size(); ++seg_idx) {
    auto& raw = raw_segments[seg_idx];
    auto view = std::make_unique<SnapshotSegmentView>();
    view->parent = &parent;
    view->node_order = raw.nodes;
    view->model_path = parent.model_path;
    view->name = parent.name + "_seg" + std::to_string(seg_idx);

    std::unordered_set<SnapshotNode*> segment_node_set(raw.nodes.begin(), raw.nodes.end());

    std::unordered_set<std::string> seen_inputs;
    std::unordered_set<std::string> seen_outputs;

    for (SnapshotNode* node : raw.nodes) {
      for (SnapshotValueInfo* vi : node->inputs) {
        if (!vi) continue;
        if (seen_inputs.count(vi->name)) continue;
        bool producer_outside = (vi->producer_node == nullptr) ||
                                (segment_node_set.count(vi->producer_node) == 0);
        if (producer_outside && !vi->is_initializer) {
          view->graph_inputs.push_back(vi);
          seen_inputs.insert(vi->name);
        }
      }
      for (SnapshotValueInfo* vi : node->outputs) {
        if (!vi) continue;
        if (seen_outputs.count(vi->name)) continue;
        bool consumed_outside = parent_output_names.count(vi->name) > 0;
        if (!consumed_outside) {
          for (auto& [consumer, _] : vi->consumers) {
            if (segment_node_set.count(consumer) == 0) {
              consumed_outside = true;
              break;
            }
          }
        }
        if (consumed_outside) {
          view->graph_outputs.push_back(vi);
          seen_outputs.insert(vi->name);
        }
      }
    }

    for (SnapshotNode* node : raw.nodes) {
      for (SnapshotValueInfo* vi : node->inputs) {
        if (!vi) continue;
        if (vi->is_initializer) {
          view->initializers.push_back(vi);
        }
      }
    }

    for (SnapshotNode* node : raw.nodes) {
      for (SnapshotValueInfo* vi : node->inputs) {
        if (vi) view->value_info_by_name[vi->name] = vi;
      }
      for (SnapshotValueInfo* vi : node->outputs) {
        if (vi) view->value_info_by_name[vi->name] = vi;
      }
    }
    for (SnapshotValueInfo* vi : view->graph_inputs) {
      view->value_info_by_name[vi->name] = vi;
    }

    auto fused = std::make_unique<SnapshotNode>();
    fused->op_type = "QnnPartialMigrationSegment";
    fused->domain = "";
    fused->name = view->name;
    fused->since_version = 1;
    fused->id = seg_idx;
    fused->inputs.reserve(view->graph_inputs.size() + view->initializers.size());
    for (auto* vi : view->graph_inputs) fused->inputs.push_back(vi);
    for (auto* vi : view->initializers) fused->inputs.push_back(vi);
    fused->outputs = view->graph_outputs;
    view->fused_node = std::move(fused);

    result.push_back({std::move(view), raw.is_gpu});
  }

  return result;
}

}  // namespace qnn
}  // namespace onnxruntime
