// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include <type_traits>

#include "graph_transform_test_builder.h"

#include "core/framework/int4.h"
#include "core/common/span_utils.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/session/inference_session.h"

#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {

using GetQDQTestCaseFn = std::function<void(ModelTestBuilder& builder)>;

template <typename T>
std::string
AddQDQNodePair(ModelTestBuilder& builder, std::string qdq_name, std::string inp_name, float scale, T zp = T(), bool use_ms_domain = false) {
  builder.AddQuantizeLinearNode<T>(qdq_name + "_q", inp_name.c_str(), scale, zp, (qdq_name + "_q_out").c_str(), use_ms_domain);
  builder.AddDequantizeLinearNode<T>(qdq_name + "_dq", (qdq_name + "_q_out").c_str(), scale, zp, (qdq_name + "_dq_out").c_str(), use_ms_domain);
  return qdq_name + "_dq_out";
}

template <typename T>
std::string
AddQDQNodePairWithOutputAsGraphOutput(ModelTestBuilder& builder, std::string qdq_name, std::string inp_name, float scale, T zp = T(),
                                      bool use_ms_domain = false) {
  builder.AddQuantizeLinearNode<T>(qdq_name + "_q", inp_name.c_str(), scale, zp, (qdq_name + "_q_out").c_str(), use_ms_domain);
  builder.AddDequantizeLinearNode<T>(qdq_name + "_dq", (qdq_name + "_q_out").c_str(), scale, zp, (qdq_name + "_dq_out").c_str(), use_ms_domain);
  builder.MakeOutput((qdq_name + "_dq_out").c_str());
  return qdq_name + "_dq_out";
}

GetQDQTestCaseFn BuildQDQReshapeTestCase(const std::vector<int64_t>& input_shape,
                                         const std::vector<int64_t>& reshape_shape);

std::vector<std::string> GetNodeOpTypesInTopologicalOrder(const Graph& graph, bool include_domain = false);

}  // namespace test
}  // namespace onnxruntime
