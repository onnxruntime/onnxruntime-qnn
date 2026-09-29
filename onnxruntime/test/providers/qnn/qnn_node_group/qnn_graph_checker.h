// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
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

// Checks the datatype of the tensor produced by the single Convert node.
void AssertConvertOutputDataType(const std::filesystem::path& dump_dir,
                                 uint32_t expected_data_type);

// Asserts that a node with the exact `node_name` does not appear in
// the compiled QNN graph JSON (root["graph"]["nodes"]).
void AssertNodeNotInQnnGraph(const std::filesystem::path& dump_dir,
                             const std::string& node_name);

// Total bytes of FP32 STATIC tensors in the compiled QNN graph JSON, i.e. the DLC cost of
// constant folding. Use Below to assert a large weight stayed compact, Above to assert an
// expected fold actually materialized.
void AssertFp32StaticBytesBelow(const std::filesystem::path& dump_dir, size_t max_bytes);
void AssertFp32StaticBytesAbove(const std::filesystem::path& dump_dir, size_t min_bytes);

// Asserts that the tensor `tensor_name` in the compiled QNN graph JSON
// (root["graph"]["tensors"][tensor_name]["dims"]) has shape == `expected_dims`.
// Use to verify post-fusion ranks/shapes — e.g. that a Transpose's input/output
// tensors are rank-4 after a rank-5-to-rank-4 fusion fired.
void AssertTensorShapeInQnnGraph(const std::filesystem::path& dump_dir,
                                 const std::string& tensor_name,
                                 const std::vector<uint32_t>& expected_dims);

// Asserts that no two nodes of type `op` in the compiled QNN graph read the same tensor at
// `input_index`. Use this where the EP derives a static input per consuming node: each consumer must
// end up with its own tensor instead of all of them collapsing onto one shared name.
void AssertNodeInputsDistinctInQnnGraph(const std::filesystem::path& dump_dir,
                                        const std::string& op,
                                        size_t input_index);

}  // namespace test
}  // namespace onnxruntime
