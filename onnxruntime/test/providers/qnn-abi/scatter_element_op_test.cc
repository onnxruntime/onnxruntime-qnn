// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cassert>
#include <string>

#include "test/providers/qnn-abi/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a non-QDQ model with indices inputs (int64) on HTP and compares output to CPU EP.
template <typename InputType1, typename InputType2 = int64_t>
static void RunOpTest(const std::string& op_type,
                      const std::vector<TestInputDef<InputType1>>& input_defs_1,
                      const std::vector<TestInputDef<InputType2>>& input_defs_2,
                      const std::vector<TestInputDef<InputType1>>& input_defs_3,
                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                      int opset_version,
                      ExpectedEPNodeAssignment expected_ep_assignment,
                      const std::string& op_domain = kOnnxDomain,
                      float fp32_abs_err = 1e-5f,
                      bool enable_htp_fp16_precision = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["soc_model"] = "87";

  if (enable_htp_fp16_precision) {
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTestABI(BuildOpTestCase<InputType1, InputType2>(op_type, input_defs_1, input_defs_2, input_defs_3, attrs, op_domain),
                     provider_options,
                     opset_version,
                     expected_ep_assignment,
                     fp32_abs_err);
}

template <typename InputQType = uint8_t, typename InputType2 = int64_t>
static void RunQDQOpTest(const std::string& op_type,
                         const std::vector<TestInputDef<float>>& input_defs_1,
                         const std::vector<TestInputDef<InputType2>>& input_defs_2,
                         const std::vector<TestInputDef<float>>& input_defs_3,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                         int opset_version,
                         ExpectedEPNodeAssignment expected_ep_assignment,
                         const std::string& op_domain = kOnnxDomain,
                         bool use_contrib_qdq = false,
                         QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracyABI(BuildOpTestCase<float, InputType2>(op_type, input_defs_1, input_defs_2, input_defs_3, attrs, op_domain),
                          BuildQDQOpTestCase<InputQType, InputType2>(op_type, input_defs_1, input_defs_2, input_defs_3, attrs,
                                                                     op_domain, use_contrib_qdq),
                          provider_options,
                          opset_version,
                          expected_ep_assignment,
                          tolerance);
}

template <typename InputType1, typename InputType2 = int64_t>
static void RunOpTestOnCPU(const std::string& op_type,
                           const std::vector<TestInputDef<InputType1>>& input_defs_1,
                           const std::vector<TestInputDef<InputType2>>& input_defs_2,
                           const std::vector<TestInputDef<InputType1>>& input_defs_3,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           const std::string& op_domain = kOnnxDomain) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTestABI(BuildOpTestCase<InputType1, InputType2>(op_type, input_defs_1, input_defs_2, input_defs_3, attrs, op_domain),
                     provider_options,
                     opset_version,
                     expected_ep_assignment);
}

//
// CPU tests:
//

// Test ScatterElements with default attributes on CPU
TEST_F(QnnABICPUBackendTests, ScatterElements_Float_Reduction_None) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunOpTestOnCPU<float, int64_t>("ScatterElements",
                                 {
                                     TestInputDef<float>({4}, false, std::move(data)),
                                 },
                                 {
                                     TestInputDef<int64_t>({1}, false, std::move(indices)),
                                 },
                                 {
                                     TestInputDef<float>({1}, false, std::move(updates)),
                                 },
                                 {},
                                 17,
                                 ExpectedEPNodeAssignment::All);
}

// Test ScatterElements with reduction Add on CPU
TEST_F(QnnABICPUBackendTests, ScatterElements_Float_Reduction_Add) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunOpTestOnCPU<float, int64_t>("ScatterElements",
                                 {
                                     TestInputDef<float>({4}, false, std::move(data)),
                                 },
                                 {
                                     TestInputDef<int64_t>({1}, false, std::move(indices)),
                                 },
                                 {
                                     TestInputDef<float>({1}, false, std::move(updates)),
                                 },
                                 {
                                     utils::MakeAttribute("reduction", "add"),
                                 },
                                 17,
                                 ExpectedEPNodeAssignment::All);
}

//
// HTP tests:
//

// Test ScatterElements with default attributes on HTP
TEST_F(QnnABIHTPBackendTests, ScatterElements_Float_Reduction_None) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunOpTest<float, int64_t>("ScatterElements",
                            {
                                TestInputDef<float>({4}, false, std::move(data)),
                            },
                            {
                                TestInputDef<int64_t>({1}, false, std::move(indices)),
                            },
                            {
                                TestInputDef<float>({1}, false, std::move(updates)),
                            },
                            {},
                            17,
                            ExpectedEPNodeAssignment::All);
}

// Test ScatterElements with default attributes on HTP
// Disable this due to an accuracy issue with selected data range
TEST_F(QnnABIHTPBackendTests, DISABLED_ScatterElements_Int8_Reduction_None) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunQDQOpTest<uint8_t, int64_t>("ScatterElements",
                                 {
                                     TestInputDef<float>({4}, false, std::move(data)),
                                 },
                                 {
                                     TestInputDef<int64_t>({1}, false, std::move(indices)),
                                 },
                                 {
                                     TestInputDef<float>({1}, false, std::move(updates)),
                                 },
                                 {},
                                 17,
                                 ExpectedEPNodeAssignment::All);
}

// Test ScatterElements with reduction ADD on HTP
// Disable this due to an accuracy issue with selected data range
TEST_F(QnnABIHTPBackendTests, DISABLED_ScatterElements_Int8_Reduction_Add) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunQDQOpTest<uint8_t, int64_t>("ScatterElements",
                                 {
                                     TestInputDef<float>({4}, false, std::move(data)),
                                 },
                                 {
                                     TestInputDef<int64_t>({1}, false, std::move(indices)),
                                 },
                                 {
                                     TestInputDef<float>({1}, false, std::move(updates)),
                                 },
                                 {
                                     utils::MakeAttribute("reduction", "add"),
                                 },
                                 17,
                                 ExpectedEPNodeAssignment::All);
}

// Test ScatterElements with reduction Max on HTP
TEST_F(QnnABIHTPBackendTests, ScatterElements_Int8_Reduction_Max) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunQDQOpTest<uint8_t, int64_t>("ScatterElements",
                                 {
                                     TestInputDef<float>({4}, false, std::move(data)),
                                 },
                                 {
                                     TestInputDef<int64_t>({1}, false, std::move(indices)),
                                 },
                                 {
                                     TestInputDef<float>({1}, false, std::move(updates)),
                                 },
                                 {
                                     utils::MakeAttribute("reduction", "max"),
                                 },
                                 17,
                                 ExpectedEPNodeAssignment::All);
}

// Test ScatterElements with reduction Mul on HTP
TEST_F(QnnABIHTPBackendTests, ScatterElements_int8_Reduction_Mul) {
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices = {1};
  std::vector<float> updates = {10.0f};
  RunQDQOpTest<uint8_t, int64_t>("ScatterElements",
                                 {
                                     TestInputDef<float>({4}, false, std::move(data)),
                                 },
                                 {
                                     TestInputDef<int64_t>({1}, false, std::move(indices)),
                                 },
                                 {
                                     TestInputDef<float>({1}, false, std::move(updates)),
                                 },
                                 {
                                     utils::MakeAttribute("reduction", "mul"),
                                 },
                                 17,
                                 ExpectedEPNodeAssignment::All);
}

// Data and updates in int 32 are not valid for HTP.
// Test int 64 Data and updates are correctly casted to float 32. And casting logic keeps indices in int 32.
TEST_F(QnnABIHTPBackendTests, TestScatterElement_Int64) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  // Default soc_model = 30 [SM8350] doesn't support fp32. Use 87 [SM8850] for testing.
  provider_options["soc_model"] = "87";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::vector<TestInputDef<int64_t>>& input_defs = {
      TestInputDef<int64_t>({256, 1024}, false, 0, 100),  // initializer
      TestInputDef<int64_t>({256, 1}, false, 0, 100),     // initializer
      TestInputDef<int64_t>({256, 1}, true, 0, 100)       // tensor
  };

  const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs = {utils::MakeAttribute("axis", static_cast<int64_t>(1))};

  RunQnnModelTestABI(BuildOpTestCase<int64_t>(
                         "ScatterElements",
                         std::move(input_defs),
                         {},
                         std::move(attrs)),
                     provider_options,
                     13,
                     ExpectedEPNodeAssignment::All,
                     1e-5f);
}

// Data and updates in int 32 are not valid for HTP.
// Test int 64 Data and updates are correctly casted to float 32. And casting logic keeps indices in int 32.
TEST_F(QnnABIHTPBackendTests, TestScatterElement_Int32) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  // Default soc_model = 30 [SM8350] doesn't support fp32. Use 87 [SM8850] for testing.
  provider_options["soc_model"] = "87";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::vector<TestInputDef<int32_t>>& input_defs = {
      TestInputDef<int32_t>({256, 1024}, false, 0, 100),  // initializer
      TestInputDef<int32_t>({256, 1}, false, 0, 100),     // initializer
      TestInputDef<int32_t>({256, 1}, true, 0, 100)       // tensor
  };

  const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs = {utils::MakeAttribute("axis", static_cast<int64_t>(1))};

  RunQnnModelTestABI(BuildOpTestCase<int32_t>(
                         "ScatterElements",
                         std::move(input_defs),
                         {},
                         std::move(attrs)),
                     provider_options,
                     13,
                     ExpectedEPNodeAssignment::All,
                     1e-5f);
}

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)