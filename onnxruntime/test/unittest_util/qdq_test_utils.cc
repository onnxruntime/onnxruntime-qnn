// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qdq_test_utils.h"
#include <type_traits>
#include <utility>
#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/span_utils.h"

namespace onnxruntime {
namespace test {

GetQDQTestCaseFn BuildQDQReshapeTestCase(const std::vector<int64_t>& input_shape,
                                         const std::vector<int64_t>& reshape_shape) {
  return [input_shape, reshape_shape](ModelTestBuilder& builder) {
    builder.MakeInput<uint8_t>("X", input_shape,
                               std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    builder.MakeOutput("Y");

    // add DQ
    builder.AddDequantizeLinearNode<uint8_t>("dq", "X", .003f, 1, "dq_out");

    // add Reshape
    builder.Make1DInitializer<int64_t>("shape", reshape_shape);
    builder.AddNode("reshape", "Reshape",
                    {"dq_out", "shape"},
                    {"reshape_output"});

    // add Q
    builder.AddQuantizeLinearNode<uint8_t>(
        "q", "reshape_output",
        .003f, 1, "q_out");
  };
}

std::vector<std::string> GetNodeOpTypesInTopologicalOrder(const Graph& graph, bool include_domain) {
  std::vector<std::string> op_types{};
  GraphViewer graph_viewer{graph};
  const auto& ordering = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : ordering) {
    const auto* node = graph.GetNode(node_idx);
    std::string full_op_type;

    if (include_domain) {
      const std::string& domain = node->Domain();
      full_op_type = domain.empty() ? node->OpType() : domain + "." + node->OpType();
    } else {
      full_op_type = node->OpType();
    }

    op_types.push_back(std::move(full_op_type));
  }
  return op_types;
}

}  // namespace test
}  // namespace onnxruntime
