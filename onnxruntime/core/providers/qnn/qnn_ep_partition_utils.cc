// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/qnn_ep_partition_utils.h"

#include <algorithm>
#include <deque>
#include <string>

#include "core/providers/qnn/common/inlined_containers.h"

namespace onnxruntime {

namespace utils {

// QNN-EP COPY START
// Below implementations are directly copied from "core/common/common.h"
// Returns whether `key` is in `container`.
// Like C++20's map/set contains() member function.
template <typename Key, typename... OtherContainerArgs,
          template <typename...> typename AssociativeContainer,
          typename LookupKey>
inline bool Contains(const AssociativeContainer<Key, OtherContainerArgs...>& container, LookupKey&& key) {
  return container.find(std::forward<LookupKey>(key)) != container.end();
}
// QNN-EP COPY END

std::vector<std::vector<const OrtNode*>> CreateSupportedPartitionNodeGroups(
    const OrtGraph* graph,
    const OrtApi& ort_api,
    const std::vector<const OrtNode*>& supported_nodes,
    const std::string& ep_type,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map) {
  std::vector<std::vector<const OrtNode*>> supported_groups{};

  size_t num_nodes = 0;
  auto status = ort_api.Graph_GetNumNodes(graph, &num_nodes);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return {};
  }
  std::vector<const OrtNode*> graph_nodes(num_nodes);
  status = ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return {};
  }

  // #inputs from unprocessed nodes (in-degree) per node.
  std::unordered_map<size_t, size_t> in_degree{};
  // Nodes that are ready to process.
  std::deque<const OrtNode*> nodes_to_process{};
  // Nodes that will be processed when considering the next partition node group.
  std::deque<const OrtNode*> nodes_to_process_with_next_group{};

  // Initialize in-degrees and find root nodes.
  for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const OrtNode* node = graph_nodes[node_idx];
    const OrtNodeUnit* node_unit = node_unit_map.at(node);

    if (&node_unit->GetNode() != node) {
      // Only process the target node.
      continue;
    }

    size_t degree = node_unit->GetInputEdgesCount(ort_api);
    in_degree.insert({node_unit->Index(), degree});
    if (degree == 0) {
      nodes_to_process.push_back(node);
    }
  }

  std::vector<const OrtNode*> supported_group{};
  // The partition node group's border is the aggregate of its nodes' output nodes.
  InlinedHashSet<const OrtNode*> supported_group_border{};

  auto close_group = [&]() {
    if (!supported_group.empty()) {
      supported_groups.emplace_back(std::move(supported_group));
      supported_group.clear();
      supported_group_border.clear();
    }
  };

  size_t num_nodes_processed = 0;

  while (!nodes_to_process.empty() || !nodes_to_process_with_next_group.empty()) {
    if (nodes_to_process.empty()) {
      // We have processed all the nodes that we can while building this partition node group, start a new one.
      close_group();
      nodes_to_process.swap(nodes_to_process_with_next_group);
      continue;
    }

    const OrtNode* node = nodes_to_process.front();
    nodes_to_process.pop_front();

    const OrtNodeUnit* node_unit = node_unit_map.at(node);
    const bool is_qdq_node_unit = node_unit->UnitType() == OrtNodeUnit::Type::QDQGroup;

    // A node that is already assigned to an EP other than current EP is unsupported.
    const char* node_ep_name;
    ORT_CONTINUE_ON_ERROR(ort_api.Node_GetEpName(node, &node_ep_name), ort_api);
    const bool is_node_supported = ((std::string(node_ep_name).empty() || node_ep_name == ep_type) &&
                                    std::find(supported_nodes.cbegin(), supported_nodes.cend(), node) != supported_nodes.cend());

    if (!is_node_supported && Contains(supported_group_border, node)) {
      // An unsupported node on the border will be processed after the current partition node group.
      nodes_to_process_with_next_group.push_back(node);
      continue;
    }

    if (is_node_supported) {
      if (is_qdq_node_unit) {
        // Add DQ -> node -> Q for the node unit and must be in topological order.
        for (const OrtNode* dq : node_unit->GetDQNodes()) {
          supported_group.push_back(dq);
        }

        supported_group.push_back(node);
        const OrtNode* output_reshape_node = node_unit->GetOutputReshapeNode();
        if (output_reshape_node) {
          supported_group.push_back(output_reshape_node);
          supported_group_border.erase(output_reshape_node);
        }
        const OrtNode* redundent_clip_node = node_unit->GetRedundantClipNode();
        if (redundent_clip_node) {
          supported_group.push_back(redundent_clip_node);
          supported_group_border.erase(redundent_clip_node);
        }

        for (const OrtNode* q : node_unit->GetQNodes()) {
          supported_group.push_back(q);
        }
      } else {
        supported_group.push_back(node);
      }

      // Remove node from the border.
      supported_group_border.erase(node);
    }

    // For each downstream node:
    //   1: Add the downstream node to the border if the current node is supported.
    //   2: Adjust in-degrees of the nodes consuming the current node's outputs, and add any new nodes to process.
    for (const OrtNode* output_node : node_unit->GetOutputNodes(ort_api)) {
      const OrtNodeUnit* downstream_node_unit = node_unit_map.at(output_node);
      const OrtNode* downstream_node = &downstream_node_unit->GetNode();

      if (is_node_supported) {
        supported_group_border.insert(downstream_node);
      }

      auto& downstream_node_in_degree = in_degree[downstream_node_unit->Index()];
      --downstream_node_in_degree;

      if (downstream_node_in_degree == 0) {
        nodes_to_process.push_back(downstream_node);
      }
    }

    ++num_nodes_processed;
  }

  close_group();

  if (num_nodes_processed != in_degree.size()) {
    ORT_CXX_API_THROW("Processed " + std::to_string(num_nodes_processed) +
                          " nodes. Expected to process " + std::to_string(in_degree.size()),
                      ORT_EP_FAIL);
  }

  return supported_groups;
}

}  // namespace utils

}  // namespace onnxruntime
