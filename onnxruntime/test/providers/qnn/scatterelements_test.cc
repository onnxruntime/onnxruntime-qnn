// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/node_attr_utils.h"
#include "core/graph/graph.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// ============================================================
// CPU backend tests
// ============================================================

TEST_F(QnnCPUBackendTests, ScatterElements_Float_Reduction_None) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunF32ModelOnCpu<float, int64_t>(
      "ScatterElements",
      {TestInputDef<float>({4}, false, std::move(data))},
      {TestInputDef<int64_t>({1}, false, std::move(indices))},
      {TestInputDef<float>({1}, false, std::move(updates))},
      {},
      17,
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, ScatterElements_Float_Reduction_Add) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunF32ModelOnCpu<float, int64_t>(
      "ScatterElements",
      {TestInputDef<float>({4}, false, std::move(data))},
      {TestInputDef<int64_t>({1}, false, std::move(indices))},
      {TestInputDef<float>({1}, false, std::move(updates))},
      {utils::MakeAttribute("reduction", "add")},
      17,
      ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ============================================================
// HTP backend tests
// ============================================================

// ScatterElements on HTP needs soc_model override on x86 Linux for HTP emulation.
static void RunScatterElementsOnHtp(
    const std::vector<TestInputDef<float>>& data_defs,
    const std::vector<TestInputDef<int64_t>>& indices_defs,
    const std::vector<TestInputDef<float>>& updates_defs,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(BuildOpTestCase<float, int64_t>("ScatterElements", data_defs, indices_defs, updates_defs,
                                                 attrs, kOnnxDomain),
                  provider_options,
                  17,
                  expected_ep_assignment);
}

TEST_F(QnnHTPBackendTests, ScatterElements_Float_Reduction_None) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunScatterElementsOnHtp(
      {TestInputDef<float>({4}, false, std::move(data))},
      {TestInputDef<int64_t>({1}, false, std::move(indices))},
      {TestInputDef<float>({1}, false, std::move(updates))},
      {},
      ExpectedEPNodeAssignment::All);
}

// Disabled due to an accuracy issue with selected data range.
TEST_F(QnnHTPBackendTests, DISABLED_ScatterElements_Int8_Reduction_None) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunQdqModelOnHtp<uint8_t, int64_t>(
      "ScatterElements",
      {TestInputDef<float>({4}, false, std::move(data))},
      {TestInputDef<int64_t>({1}, false, std::move(indices))},
      {TestInputDef<float>({1}, false, std::move(updates))},
      {},
      17,
      ExpectedEPNodeAssignment::All);
}

// Disabled due to an accuracy issue with selected data range.
TEST_F(QnnHTPBackendTests, DISABLED_ScatterElements_Int8_Reduction_Add) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunQdqModelOnHtp<uint8_t, int64_t>(
      "ScatterElements",
      {TestInputDef<float>({4}, false, std::move(data))},
      {TestInputDef<int64_t>({1}, false, std::move(indices))},
      {TestInputDef<float>({1}, false, std::move(updates))},
      {utils::MakeAttribute("reduction", "add")},
      17,
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ScatterElements_Int8_Reduction_Max) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunQdqModelOnHtp<uint8_t, int64_t>(
      "ScatterElements",
      {TestInputDef<float>({4}, false, std::move(data))},
      {TestInputDef<int64_t>({1}, false, std::move(indices))},
      {TestInputDef<float>({1}, false, std::move(updates))},
      {utils::MakeAttribute("reduction", "max")},
      17,
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ScatterElements_int8_reduction_mul) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunQdqModelOnHtp<uint8_t, int64_t>(
      "ScatterElements",
      {TestInputDef<float>({4}, false, std::move(data))},
      {TestInputDef<int64_t>({1}, false, std::move(indices))},
      {TestInputDef<float>({1}, false, std::move(updates))},
      {utils::MakeAttribute("reduction", "mul")},
      17,
      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
