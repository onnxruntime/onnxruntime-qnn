// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>

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

// Asserts that no two nodes of type `op` in the compiled QNN graph read the same tensor at
// `input_index`. Use this where the EP derives a static input per consuming node: each consumer must
// end up with its own tensor instead of all of them collapsing onto one shared name.
void AssertNodeInputsDistinctInQnnGraph(const std::filesystem::path& dump_dir,
                                        const std::string& op,
                                        size_t input_index);

}  // namespace test
}  // namespace onnxruntime
