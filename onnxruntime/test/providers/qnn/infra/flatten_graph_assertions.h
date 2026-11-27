// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace test {

// Used with both a direct builder and an actual session dump. Do not accept
// tensor existence alone: the sole node must connect the declared graph IO.
inline void AssertFlattenReshapeGraph(const nlohmann::json& json,
                                      const std::string& input_name,
                                      const std::string& output_name,
                                      const std::vector<int64_t>& input_shape,
                                      const std::vector<int64_t>& output_shape,
                                      uint32_t dtype,
                                      bool expect_equal_qparams = true) {
  const auto& graph = json.at("graph");
  const auto& nodes = graph.at("nodes");
  ASSERT_EQ(nodes.size(), 1u);
  const auto node = nodes.begin().value();
  EXPECT_EQ(node.at("type"), "Reshape");
  EXPECT_EQ(node.at("input_names"), nlohmann::json(std::vector<std::string>{input_name}));
  EXPECT_EQ(node.at("output_names"), nlohmann::json(std::vector<std::string>{output_name}));
  const auto& input = graph.at("tensors").at(input_name);
  const auto& output = graph.at("tensors").at(output_name);
  EXPECT_EQ(input.at("dims"), nlohmann::json(input_shape));
  EXPECT_EQ(output.at("dims"), nlohmann::json(output_shape));
  EXPECT_EQ(input.at("data_type"), dtype);
  EXPECT_EQ(output.at("data_type"), dtype);
  if (expect_equal_qparams) {
    EXPECT_EQ(input.at("quant_params"), output.at("quant_params"));
  }
}

}  // namespace test
}  // namespace onnxruntime
