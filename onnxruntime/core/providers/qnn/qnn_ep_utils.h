// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include <tuple>

#include "core/providers/qnn/ort_api.h"

// Forward declaration of OrtNode
struct OrtNode;

// Forward declaration
namespace onnxruntime {

namespace QDQ {

// Forward declaration
class OrtNodeGroupSelector;

// struct that provides a join between selector and op versions supported
struct OrtOpVersionsAndSelector {
  using OpVersionsMap = std::unordered_map<std::string, std::vector<int>>;

  OrtOpVersionsAndSelector(const OpVersionsMap& ops_and_versions_in,
                           std::unique_ptr<OrtNodeGroupSelector> selector_in)
      : op_versions_map{ops_and_versions_in},
        selector{std::move(selector_in)} {}

  OpVersionsMap op_versions_map;
  std::unique_ptr<OrtNodeGroupSelector> selector;
};

// class that manages a set of node group selectors
class OrtSelectors {
 public:
  OrtSelectors() = default;

  // register a selector for the specified ops.
  void RegisterSelector(const OrtOpVersionsAndSelector::OpVersionsMap& ops_and_versions_in,
                        std::unique_ptr<OrtNodeGroupSelector> selector_in);

  const std::vector<std::unique_ptr<OrtOpVersionsAndSelector>>& SelectorsSet() const {
    return selectors_set_;
  }

 private:
  std::vector<std::unique_ptr<OrtOpVersionsAndSelector>> selectors_set_;
};

// Base class for node group selectors
class OrtNodeGroupSelector {
 public:
  virtual ~OrtNodeGroupSelector() = default;
  // Check if the node group is supported
  bool virtual Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                     const OrtNode* redundant_clip_node,
                     const std::vector<const OrtNode*>& dq_nodes,
                     const std::vector<const OrtNode*>& q_nodes) const = 0;

 protected:
  // Helper function to check if a node has the expected number of DQ inputs and Q outputs
  bool CheckQDQNodes(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                     const OrtNode* redundant_clip_node,
                     const std::vector<const OrtNode*>& dq_nodes,
                     const std::vector<const OrtNode*>& q_nodes,
                     int num_dq_inputs = -1,
                     bool is_empty_q_nodes_allowed = false) const;
};

// Single DQ -> node that does not change data -> Q.
// Zero point and scale are constant scalars and must match
class OrtDropQDQNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtDropQDQNodeGroupSelector(bool allow_nonpositive_scale = true)
      : allow_nonpositive_scale_(allow_nonpositive_scale) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_nonpositive_scale_;
};

// Selector for drop DQ operations
class OrtDropDQNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtDropDQNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// Selector for unary operations
class OrtUnaryNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtUnaryNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

class OrtClipNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtClipNodeGroupSelector() = default;

 private:
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// Selector for binary operations
class OrtBinaryNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtBinaryNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// Selector for variadic operations
class OrtVariadicNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtVariadicNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// DQ node -> Split -> multiple Q nodes with equal quantization types.
// Optionally, the selector can require all input and output quantization parameters to be
// equal and constant.
class OrtSplitNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtSplitNodeGroupSelector(bool req_equal_quant_params = false)
      : req_equal_quant_params_(req_equal_quant_params) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool req_equal_quant_params_;  // If true, only selects a node group if the input and output
                                 // quantization parameters are all equal/constant, which enables the
                                 // optimizer to drop the Q/DQ ops if the group is assigned to the CPU EP.
};

// DQ nodes for X, W and optionally B -> node -> Q
class OrtConvNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtConvNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

class OrtWhereNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtWhereNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

class OrtPadNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtPadNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// one ore more DQ nodes for each input -> node -> Q
class OrtEinsumNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtEinsumNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

class OrtReciprocalNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtReciprocalNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// 2 DQ nodes for input -> node -> Q.
// A group without a trailing Q isn't really a QDQ node group, so it is not selected.
class OrtMatMulNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtMatMulNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// Input: DQ nodes for A, B and optional C
// Output: optional Q node for Y
class OrtGemmNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtGemmNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// Input: DQ nodes for input, scale, and B
// Output: Q node for output
class OrtInstanceAndLayerNormalizationNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// DQ nodes for X, W and optionally B, not used for mean, var -> node -> Q
class OrtBatchNormalizationNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtBatchNormalizationNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// 2 DQ nodes providing input -> node with bool output tensor.
// Example: Equal, Less, Greater.
class OrtLogicalComparisonNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// TopK has 1 DQ input node and 1 Q output node.
// Zero point and scale are constant scalars and must match
class OrtTopKNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// one DQ node for first input -> node -> Q
class OrtCumSumNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// Input: DQ nodes for Data, and Update
// Output: Q node for output
class OrtScatterElementsNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// Input: DQ nodes for input, scale
// Output: Q node for output
class OrtRMSNormalizationNodeGroupSelector : public OrtNodeGroupSelector {
 private:
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// Input 1: DQ node for input A.
// Input 2: no DQ node for input B.
// Input 3: no DQ node for input scales.
// Input 4: no DQ node for input zero_points.
// Output: Q node for output Y.
class OrtMatMulNBitsNodeGroupSelector : public OrtNodeGroupSelector {
 private:
  bool Check(const OrtGraph* graph,
             const OrtApi& ort_api,
             const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// SelectorManager for OrtGraph
class OrtSelectorManager {
 public:
  OrtSelectorManager();

  // Method that finds and returns a vector of QDQ::NodeGroup in a given OrtGraph
  std::vector<OrtNodeGroup> GetOrtQDQSelections(const OrtGraph* graph,
                                                const OrtApi& ort_api,
                                                const Ort::Logger& logger) const;

 private:
  OrtSelectors ort_selectors_;

  std::unordered_map<std::string, const OrtOpVersionsAndSelector*> op_type_to_selectors_map_;

  void InitializeSelectorsMap();

  void CreateSelectors();
};
}  // namespace QDQ

namespace utils {

// Describes an IQnnNodeGroup (multi-NodeUnit fusion or a single NodeUnit wrapped as a trivial 1-member group)
// in terms the partitioner can reason about. One entry exists per IQnnNodeGroup produced by GetQnnNodeGroups
// (supported or not), so every NodeUnit in the graph belongs to exactly one group.
//
// The partitioner treats each group as an atomic BFS scheduling unit: all members are admitted into the same
// partition, or none. This prevents BFS from splitting fusion members across partition boundaries when an
// unsupported op sits topologically between them.
struct QnnNodeGroupInfo {
  size_t group_id = 0;                                // Dense index into the groups vector.
  const OrtNodeUnit* target_node_unit = nullptr;      // From IQnnNodeGroup::GetTargetNodeUnit(); nullptr if defunct.
  std::vector<const OrtNodeUnit*> member_node_units;  // All member NodeUnits (includes the target).
  std::unordered_set<const OrtNodeUnit*> member_set;  // O(1) membership test; mirrors member_node_units.
  bool is_supported = false;                          // Cached IQnnNodeGroup::IsSupported() result.
  bool is_defunct = false;                            // Set when the group is demoted; BFS skips defunct entries.
  size_t external_in_degree = 0;                      // Count of input edges from NodeUnits NOT in member_set.
};

// Walks each member's OrtNode inputs and counts edges whose producer NodeUnit is not in the group's member_set.
// Initializers (null producer) are skipped. For QDQGroup members, walks DQ/target/Q nodes.
size_t ComputeGroupExternalInDegree(
    const QnnNodeGroupInfo& group,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
    const OrtApi& ort_api);

// Refer to CreateSupportedPartitions in partitioning_utils.cc.
//
// Group-aware partitioner. `groups` describes every IQnnNodeGroup (fusion or 1-member wrapper);
// `node_unit_to_group_id` maps each NodeUnit to its owning group. The BFS iterates groups as atomic units,
// guaranteeing that members of any multi-NodeUnit fusion land in the same partition.
std::vector<std::vector<const OrtNode*>> CreateSupportedPartitionNodeGroups(
    const OrtGraph* graph,
    const OrtApi& ort_api,
    const std::vector<const OrtNode*>& supported_nodes,
    const std::string& ep_type,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
    const std::vector<QnnNodeGroupInfo>& groups,
    const std::unordered_map<const OrtNodeUnit*, size_t>& node_unit_to_group_id,
    const Ort::Logger& logger);

}  // namespace utils

// Function to get QDQ node units from OrtGraph
std::pair<std::vector<std::unique_ptr<OrtNodeUnit>>, std::unordered_map<const OrtNode*, const OrtNodeUnit*>>
GetAllOrtNodeUnits(OrtApi ort_api, const OrtGraph* graph, const Ort::Logger& logger);

}  // namespace onnxruntime
