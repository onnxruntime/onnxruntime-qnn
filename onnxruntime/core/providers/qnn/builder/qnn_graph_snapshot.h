// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Hot Migration (Phase 2 extension) - Background NPU Compose.
//
// GraphSnapshot is an EP-owned copy of the structural information ComposeGraph reads out of the
// live OrtGraph, captured during CompileImpl while the OrtGraph* pointers are still valid. The
// compose pipeline is later driven from this snapshot (off the CompileImpl critical path) via a
// copy of OrtApi whose graph/node/valueinfo read accessors are repointed at readers over these
// structures. See qnn_graph_snapshot_accessors.* for the accessors and shim.
//
// The opaque handles the accessors hand out (OrtGraph*/OrtNode*/OrtValueInfo*/OrtOpAttr*) are
// reinterpret_cast pointers to the SnapshotGraph/SnapshotNode/SnapshotValueInfo/SnapshotAttr
// objects below. Only our accessors dereference them; ORT never does. Addresses must stay stable
// for the snapshot's lifetime, so nodes/value-infos/attrs are held by stable-address containers
// and never reallocated after the walk completes.

struct SnapshotNode;
struct SnapshotGraph;

// One ONNX attribute captured from a node.
struct SnapshotAttr {
  std::string name;
  OrtOpAttrType type;

  std::optional<int64_t> i;
  std::optional<float> f;
  std::optional<std::string> s;
  std::vector<int64_t> ints;
  std::vector<float> floats;
  std::vector<std::string> strings;
};

// A snapshotted value (tensor edge). Producer/consumer edges are precomputed during the walk.
struct SnapshotValueInfo {
  std::string name;

  // Type/shape. has_shape distinguishes "scalar / rank-0" from "unknown shape".
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  bool has_shape = false;
  std::vector<int64_t> shape;

  bool is_constant_initializer = false;
  bool is_graph_output = false;

  SnapshotNode* producer_node = nullptr;
  size_t producer_output_index = 0;

  // (node, input index) pairs. Input index is -1 for implicit inputs.
  std::vector<std::pair<SnapshotNode*, int64_t>> consumers;

  std::vector<uint8_t> initializer_bytes;
  bool is_initializer = false;
};

struct SnapshotNode {
  std::string op_type;
  std::string domain;
  std::string name;
  int since_version = 0;
  size_t id = 0;

  // Ordered inputs/outputs by pointer into SnapshotGraph::value_infos. A missing optional input
  // is represented by a nullptr entry to preserve positional indices.
  std::vector<SnapshotValueInfo*> inputs;
  std::vector<SnapshotValueInfo*> outputs;
  std::vector<SnapshotValueInfo*> implicit_inputs;

  std::vector<SnapshotAttr> attributes;
};

struct SnapshotGraph {
  // Stable-address storage. Populated once by SnapshotGraphFromOrt, never resized afterward.
  std::vector<std::unique_ptr<SnapshotNode>> nodes;
  std::vector<std::unique_ptr<SnapshotValueInfo>> value_infos;

  // The EP fused node (subgraph boundary). Not part of node_order and never a producer/consumer.
  std::unique_ptr<SnapshotNode> fused_node;

  std::vector<SnapshotNode*> node_order;
  std::vector<SnapshotValueInfo*> graph_inputs;
  std::vector<SnapshotValueInfo*> graph_outputs;
  std::vector<SnapshotValueInfo*> initializers;

  std::filesystem::path model_path;
  std::string name;

  // Name -> value-info lookup.
  std::unordered_map<std::string, SnapshotValueInfo*> value_info_by_name;

  SnapshotValueInfo* FindValueInfo(const std::string& value_name) const {
    auto it = value_info_by_name.find(value_name);
    return it == value_info_by_name.end() ? nullptr : it->second;
  }

  // Release heavy payload (initializer bytes, attributes) after HTP compilation completes.
  size_t ReleasePayload() {
    size_t freed = 0;
    for (auto& vi : value_infos) {
      freed += vi->initializer_bytes.capacity();
      std::vector<uint8_t>().swap(vi->initializer_bytes);
    }
    for (auto& node : nodes) {
      freed += node->attributes.capacity() * sizeof(SnapshotAttr);
      std::vector<SnapshotAttr>().swap(node->attributes);
    }
    if (fused_node) {
      freed += fused_node->attributes.capacity() * sizeof(SnapshotAttr);
      std::vector<SnapshotAttr>().swap(fused_node->attributes);
    }
    value_info_by_name.clear();
    return freed;
  }
};

// View over a subset of a SnapshotGraph's nodes for partial migration. The parent SnapshotGraph
// must outlive this view. Represented as a SnapshotGraph so existing SnapshotShim/accessors work
// unchanged. `nodes` and `value_infos` owning vectors are empty; `node_order` etc. hold
// non-owning pointers into the parent's stable storage.
struct SnapshotSegmentView : SnapshotGraph {
  const SnapshotGraph* parent = nullptr;
};

// Build segment views by splitting a snapshot at nodes whose names are in `gpu_only_names`.
struct SegmentDesc {
  std::unique_ptr<SnapshotSegmentView> view;
  bool is_gpu;  // true if this segment contains GPU-only nodes
};
std::vector<SegmentDesc> SplitSnapshotIntoSegments(
    const SnapshotGraph& parent,
    const std::unordered_set<std::string>& gpu_only_names);

// Walk the live OrtGraph and populate a heap-allocated SnapshotGraph. Must be called while the
// OrtGraph* is valid (inside CompileImpl).
std::unique_ptr<SnapshotGraph> SnapshotGraphFromOrt(const OrtApi& ort_api,
                                                    const OrtGraph* graph,
                                                    const OrtNode* fused_node,
                                                    const Ort::Logger& logger);

}  // namespace qnn
}  // namespace onnxruntime
