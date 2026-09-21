// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <cstdint>
#include <string>

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

}  // namespace test
}  // namespace onnxruntime
