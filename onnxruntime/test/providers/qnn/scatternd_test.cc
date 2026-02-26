// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, ScatterND_int64_int64) {
  std::vector<int64_t> data = {0, 1, 2, 3};
  std::vector<int64_t> indices = {1};
  std::vector<int64_t> updates = {10};
  RunF32ModelOnHtp<int64_t>("ScatterND",
                            {TestInputDef<int64_t>({4}, false, std::move(data)),
                             TestInputDef<int64_t>({1, 1}, false, std::move(indices)),
                             TestInputDef<int64_t>({1}, false, std::move(updates))},
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ScatterND_int64_int64_reduction_add) {
  std::vector<int64_t> data = {0, 1, 2, 3};
  std::vector<int64_t> indices = {1};
  std::vector<int64_t> updates = {10};
  RunF32ModelOnHtp<int64_t>("ScatterND",
                            {TestInputDef<int64_t>({4}, false, std::move(data)),
                             TestInputDef<int64_t>({1, 1}, false, std::move(indices)),
                             TestInputDef<int64_t>({1}, false, std::move(updates))},
                            {utils::MakeAttribute("reduction", "add")},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ScatterND_int64_int64_reduction_mul) {
  std::vector<int64_t> data = {0, 1, 2, 3};
  std::vector<int64_t> indices = {1};
  std::vector<int64_t> updates = {10};
  RunF32ModelOnHtp<int64_t>("ScatterND",
                            {TestInputDef<int64_t>({4}, false, std::move(data)),
                             TestInputDef<int64_t>({1, 1}, false, std::move(indices)),
                             TestInputDef<int64_t>({1}, false, std::move(updates))},
                            {utils::MakeAttribute("reduction", "mul")},
                            17,
                            ExpectedEPNodeAssignment::All);
}

// Reduction max/min fall back to CPU.
TEST_F(QnnHTPBackendTests, ScatterND_int64_int64_reduction_max) {
  std::vector<int64_t> data = {0, 1, 2, 3};
  std::vector<int64_t> indices = {1};
  std::vector<int64_t> updates = {10};
  RunF32ModelOnHtp<int64_t>("ScatterND",
                            {TestInputDef<int64_t>({4}, false, std::move(data)),
                             TestInputDef<int64_t>({1, 1}, false, std::move(indices)),
                             TestInputDef<int64_t>({1}, false, std::move(updates))},
                            {utils::MakeAttribute("reduction", "max")},
                            17,
                            ExpectedEPNodeAssignment::None);
}

TEST_F(QnnHTPBackendTests, ScatterND_int64_int64_reduction_min) {
  std::vector<int64_t> data = {0, 1, 2, 3};
  std::vector<int64_t> indices = {1};
  std::vector<int64_t> updates = {10};
  RunF32ModelOnHtp<int64_t>("ScatterND",
                            {TestInputDef<int64_t>({4}, false, std::move(data)),
                             TestInputDef<int64_t>({1, 1}, false, std::move(indices)),
                             TestInputDef<int64_t>({1}, false, std::move(updates))},
                            {utils::MakeAttribute("reduction", "min")},
                            17,
                            ExpectedEPNodeAssignment::None);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
