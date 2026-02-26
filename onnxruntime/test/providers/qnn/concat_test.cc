// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// ============================================================
// CPU backend tests
// ============================================================

TEST_F(QnnCPUBackendTests, Concat_EmptyInput) {
  RunF32ModelOnCpu("Concat",
                   {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                    TestInputDef<float>({1, 0, 4, 4}, false, {})},
                   {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                   13,
                   ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ============================================================
// HTP backend tests
// ============================================================

TEST_F(QnnHTPBackendTests, Concat_EmptyInput) {
  RunF32ModelOnHtp("Concat",
                   {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                    TestInputDef<float>({1, 0, 4, 4}, false, {})},
                   {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                   13,
                   ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Concat_EmptyInitializer) {
  RunF32ModelOnHtp("Concat",
                   {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                    TestInputDef<float>({1, 0, 4, 4}, true, {})},  // true makes this an initializer
                   {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                   13,
                   ExpectedEPNodeAssignment::All);
}

// 3 inputs concatenated at the last axis.
TEST_F(QnnHTPBackendTests, VariadicOp_Concat_3Inputs_LastAxis) {
  RunQdqModelOnHtp<uint8_t>("Concat",
                            {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                             TestInputDef<float>({1, 2, 2, 3}, false, -1.0f, 1.0f),
                             TestInputDef<float>({1, 2, 2, 1}, false, -2.0f, 2.0f)},
                            {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                            13,
                            ExpectedEPNodeAssignment::All);
}

// 2 inputs concatenated at the second axis.
TEST_F(QnnHTPBackendTests, VariadicOp_Concat_2Inputs_2ndAxis) {
  RunQdqModelOnHtp<uint8_t>("Concat",
                            {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                             TestInputDef<float>({1, 3, 2, 2}, false, -2.0f, 2.0f)},
                            {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                            13,
                            ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
