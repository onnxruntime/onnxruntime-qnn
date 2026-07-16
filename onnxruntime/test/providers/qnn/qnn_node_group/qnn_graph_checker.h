// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace onnxruntime {
namespace test {

// Asserts that the given QNN op type appears exactly `count` times in
// the compiled QNN graph JSON (root["graph"]["nodes"][*]["type"]).
// Finds the JSON graph file in `dump_dir`, skipping tensor log files.
void AssertOpInQnnGraph(const std::filesystem::path& dump_dir,
                        const std::string& op,
                        size_t count = 1);

// Asserts that a node with the exact `node_name` does not appear in
// the compiled QNN graph JSON (root["graph"]["nodes"]).
void AssertNodeNotInQnnGraph(const std::filesystem::path& dump_dir,
                             const std::string& node_name);

// Asserts that the tensor `tensor_name` in the compiled QNN graph JSON
// (root["graph"]["tensors"][tensor_name]["dims"]) has shape == `expected_dims`.
// Use to verify post-fusion ranks/shapes — e.g. that a Transpose's input/output
// tensors are rank-4 after a rank-5-to-rank-4 fusion fired.
void AssertTensorShapeInQnnGraph(const std::filesystem::path& dump_dir,
                                 const std::string& tensor_name,
                                 const std::vector<uint32_t>& expected_dims);

}  // namespace test
}  // namespace onnxruntime
