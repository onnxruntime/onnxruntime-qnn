// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {

namespace utils {

// Refer to CreateSupportedPartitions in partitioning_utils.cc.
std::vector<std::vector<const OrtNode*>> CreateSupportedPartitionNodeGroups(
    const OrtGraph* graph,
    const OrtApi& ort_api,
    const std::vector<const OrtNode*>& supported_nodes,
    const std::string& ep_type,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map);

}  // namespace utils

}  // namespace onnxruntime
