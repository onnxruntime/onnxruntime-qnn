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

// Asserts that a QNN op of the given type has a specific scalar parameter set.
// Searches all nodes in the compiled QNN graph JSON for a node matching `op_type`,
// then checks that `param_name` exists in its "scalar_params".
void AssertOpHasScalarParam(const std::filesystem::path& dump_dir,
                            const std::string& op_type,
                            const std::string& param_name);

}  // namespace test
}  // namespace onnxruntime
