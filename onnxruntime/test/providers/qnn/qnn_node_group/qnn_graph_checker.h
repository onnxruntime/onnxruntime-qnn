// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace onnxruntime {
namespace test {

// Asserts that the given QNN op type appears exactly `count` times in
// the compiled QNN graph JSON (root["graph"]["nodes"][*]["type"]).
// Finds the JSON graph file in `dump_dir`, skipping tensor log files.
void AssertOpInQnnGraph(const std::filesystem::path& dump_dir,
                        const std::string& op,
                        size_t count = 1);

// Asserts that a node of type `op` has a name containing `name_substring`.
void AssertOpWithNameInQnnGraph(const std::filesystem::path& dump_dir,
                                const std::string& op,
                                const std::string& name_substring);

// Checks the datatype of the tensor produced by the single Convert node.
void AssertConvertOutputDataType(const std::filesystem::path& dump_dir,
                                 uint32_t expected_data_type);

// Checks the output tensor of a single node of type `op`, including per-tensor
// scale/offset when supplied. QNN's offset is the negative ONNX zero point.
void AssertOpOutputInQnnGraph(const std::filesystem::path& dump_dir,
                              const std::string& op,
                              uint32_t expected_data_type,
                              std::optional<std::pair<float, int32_t>> expected_scale_offset = std::nullopt);

// Count APP_WRITE tensors to detect parameters left outside a QDQ group.
void AssertQnnGraphInputCount(const std::filesystem::path& dump_dir, size_t expected_count);

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

// Asserts that the tensor at `input_index` of every node of type `op` has the expected
// QNN quantization encoding.
void AssertOpInputQuantizationEncodingInQnnGraph(const std::filesystem::path& dump_dir,
                                                 const std::string& op,
                                                 size_t input_index,
                                                 uint32_t expected_encoding);

// Asserts that no two nodes of type `op` in the compiled QNN graph read the same tensor at
// `input_index`. Use this where the EP derives a static input per consuming node: each consumer must
// end up with its own tensor instead of all of them collapsing onto one shared name.
void AssertNodeInputsDistinctInQnnGraph(const std::filesystem::path& dump_dir,
                                        const std::string& op,
                                        size_t input_index);

}  // namespace test
}  // namespace onnxruntime
