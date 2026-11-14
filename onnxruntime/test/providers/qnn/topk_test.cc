// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

// Returns a function that builds a model with a TopK operator.
template <typename DataType>
inline GetTestModelFn BuildTopKTestCase(const TestInputDef<DataType>& input_def,
                                        const TestInputDef<int64_t>& k_def,
                                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, k_def, attrs](ModelTestBuilder& builder) {
    MakeTestInput<DataType>(builder, "input", input_def);
    MakeTestInput<int64_t>(builder, "k", k_def);

    builder.MakeOutput("values");
    builder.MakeOutput("indices");

    builder.AddNode("TopK",
                    "TopK",
                    {"input", "k"},
                    {"values", "indices"},
                    kOnnxDomain,
                    attrs);
  };
}

// Runs a model with a TopK operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunTopKTestOnCPU(const TestInputDef<DataType>& input_def,
                             const TestInputDef<int64_t>& k_def,
                             const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             int opset = 19,
                             bool verify_outputs = true) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "cpu";

  RunQnnModelTest(BuildTopKTestCase<DataType>(input_def, k_def, attrs),
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(1e-5f)},
                  /*log_severity*/ OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs*/ verify_outputs);
}

//
// CPU tests:
//

// Repeated values at the K boundary must select smaller original indices.
// Compare both values and INT64 graph-output indices with the ORT CPU EP.
TEST_F(QnnCPUBackendTests, TopK_Ties_PreferSmallerIndices) {
  for (int64_t largest : {0, 1}) {
    for (int64_t k : {1, 3, 6}) {
      SCOPED_TRACE("largest=" + std::to_string(largest) + ", k=" + std::to_string(k));
      for (const auto& data : std::vector<std::vector<float>>{
               {0, 0, 0, 0, 0, 0}, {5, 5, 1, 5, 2, 5}, {1, 4, 4, 4, 0, 4}}) {
        RunTopKTestOnCPU<float>(TestInputDef<float>({1, 6}, false, data),
                                TestInputDef<int64_t>({1}, true, {k}),
                                {test::MakeAttribute("largest", largest)}, ExpectedEPNodeAssignment::All);
      }
    }
  }
}

TEST_F(QnnCPUBackendTests, TopK_Ties_NonLastAxis) {
  for (int64_t axis : {1, -2}) {
    for (int64_t largest : {0, 1}) {
      SCOPED_TRACE("axis=" + std::to_string(axis) + ", largest=" + std::to_string(largest));
      RunTopKTestOnCPU<float>(TestInputDef<float>({1, 6, 2}, false,
                                                  {5, 0, 5, 0, 1, 2, 5, 0, 2, 1, 5, 0}),
                              TestInputDef<int64_t>({1}, true, {3}),
                              {test::MakeAttribute("axis", axis), test::MakeAttribute("largest", largest)},
                              ExpectedEPNodeAssignment::All);
    }
  }
}

TEST_F(QnnCPUBackendTests, TopK_Ties_Int32) {
  for (int64_t largest : {0, 1}) {
    RunTopKTestOnCPU<int32_t>(TestInputDef<int32_t>({1, 6}, false, {5, 5, 1, 5, 2, 5}),
                              TestInputDef<int64_t>({1}, true, {3}),
                              {test::MakeAttribute("largest", largest)}, ExpectedEPNodeAssignment::All);
  }
}

// Runtime INT64 inputs need a cast before reversal, and both graph outputs
// need casts back to INT64. Verify assignment so CPU fallback cannot hide a failure.
TEST_F(QnnCPUBackendTests, TopK_Ties_Int64GraphOutputs) {
  for (int64_t largest : {0, 1}) {
    for (int64_t k : {1, 3, 6}) {
      SCOPED_TRACE("largest=" + std::to_string(largest) + ", k=" + std::to_string(k));
      RunTopKTestOnCPU<int64_t>(TestInputDef<int64_t>({1, 6}, false, {5, 5, -1, 5, -2, 5}),
                                TestInputDef<int64_t>({1}, true, {k}),
                                {test::MakeAttribute("largest", largest)}, ExpectedEPNodeAssignment::All);
    }
  }
}

TEST_F(QnnCPUBackendTests, TopK_Ties_Int64NonLastAxis) {
  for (int64_t axis : {1, -2}) {
    for (int64_t largest : {0, 1}) {
      SCOPED_TRACE("axis=" + std::to_string(axis) + ", largest=" + std::to_string(largest));
      RunTopKTestOnCPU<int64_t>(TestInputDef<int64_t>({1, 6, 2}, false,
                                                      {5, 0, 5, 0, -1, 2, 5, 0, -2, 1, 5, 0}),
                                TestInputDef<int64_t>({1}, true, {3}),
                                {test::MakeAttribute("axis", axis), test::MakeAttribute("largest", largest)},
                                ExpectedEPNodeAssignment::All);
    }
  }
}

// INT64 values consumed inside the QNN partition stay native INT32, without
// an unnecessary graph-output cast before the consumer.
TEST_F(QnnCPUBackendTests, TopK_Ties_Int64ValuesConsumedByCast) {
  for (int64_t axis : {-1, 1, -2}) {
    SCOPED_TRACE("axis=" + std::to_string(axis));
    auto build = [axis](ModelTestBuilder& builder) {
      const bool last_axis = axis == -1;
      MakeTestInput<int64_t>(builder, "input",
                             last_axis ? TestInputDef<int64_t>({1, 6}, false, {5, 5, -1, 5, -2, 5})
                                       : TestInputDef<int64_t>({1, 6, 2}, false,
                                                               {5, 0, 5, 0, -1, 2, 5, 0, -2, 1, 5, 0}));
      MakeTestInput<int64_t>(builder, "k", TestInputDef<int64_t>({1}, true, {3}));
      builder.MakeOutput("cast_values");
      builder.MakeOutput("indices");
      builder.AddNode("TopK", "TopK", {"input", "k"}, {"values", "indices"},
                      kOnnxDomain, {test::MakeAttribute("axis", axis)});
      builder.AddNode("Cast", "Cast", {"values"}, {"cast_values"}, kOnnxDomain,
                      {test::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32))});
    };
    RunQnnModelTest(build, {{"backend_type", "cpu"}}, 19,
                    EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.0f)});
  }
}

TEST_F(QnnCPUBackendTests, TopK_Ties_Int64SharedInputAndUnnamedNodes) {
  auto build = [](ModelTestBuilder& builder) {
    MakeTestInput<int64_t>(builder, "input",
                           TestInputDef<int64_t>({1, 6, 2}, false, {5, 0, 5, 0, -1, 2, 5, 0, -2, 1, 5, 0}));
    MakeTestInput<int64_t>(builder, "k", TestInputDef<int64_t>({1}, true, {3}));
    for (const std::string suffix : {"_a", "_b"}) {
      builder.MakeOutput("values" + suffix);
      builder.MakeOutput("indices" + suffix);
      builder.AddNode("", "TopK", {"input", "k"}, {"values" + suffix, "indices" + suffix},
                      kOnnxDomain, {test::MakeAttribute("axis", static_cast<int64_t>(1))});
    }
  };
  RunQnnModelTest(build, {{"backend_type", "cpu"}}, 19,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.0f)});
}

TEST_F(QnnCPUBackendTests, TopK_Ties_LongZeroRun) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 1280}, false, std::vector<float>(1280, 0.0f)),
                          TestInputDef<int64_t>({1}, true, {20}), {}, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, TopK_SingleElementAxis) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({2, 1}, false, {1, 2}),
                          TestInputDef<int64_t>({1}, true, {1}), {}, ExpectedEPNodeAssignment::All);
}

// The corrected indices must also be visible to downstream nodes inside a partition.
TEST_F(QnnCPUBackendTests, TopK_Ties_IndicesConsumedByGather) {
  auto build = [](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", TestInputDef<float>({6}, false, {5, 5, 1, 5, 2, 5}));
    MakeTestInput<int64_t>(builder, "k", TestInputDef<int64_t>({1}, true, {3}));
    MakeTestInput<int32_t>(builder, "table", TestInputDef<int32_t>({6}, true, {10, 20, 30, 40, 50, 60}));
    builder.MakeOutput("values");
    builder.MakeOutput("selected");
    builder.AddNode("TopK", "TopK", {"input", "k"}, {"values", "indices"});
    builder.AddNode("Gather", "Gather", {"table", "indices"}, {"selected"});
  };
  RunQnnModelTest(build, {{"backend_type", "cpu"}}, 19,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.0f)});
}

TEST_F(QnnCPUBackendTests, TopK_Ties_SharedInputAndUnnamedNodes) {
  auto build = [](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", TestInputDef<float>({1, 6, 2}, false, {5, 0, 5, 0, 1, 2, 5, 0, 2, 1, 5, 0}));
    MakeTestInput<int64_t>(builder, "k", TestInputDef<int64_t>({1}, true, {3}));
    for (const std::string suffix : {"_a", "_b"}) {
      builder.MakeOutput("values" + suffix);
      builder.MakeOutput("indices" + suffix);
      builder.AddNode("", "TopK", {"input", "k"}, {"values" + suffix, "indices" + suffix},
                      kOnnxDomain, {test::MakeAttribute("axis", static_cast<int64_t>(1))});
    }
  };
  RunQnnModelTest(build, {{"backend_type", "cpu"}}, 19,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.0f)});
}

// Test that TopK with a dynamic K input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, TopK_DynamicK_Unsupported) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, false /* is_initializer */, {2}),
                          {},                               // Attributes
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test that TopK with an axis attribute that is not the last dimension.
TEST_F(QnnCPUBackendTests, TopK_NonLastAxis) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                          ExpectedEPNodeAssignment::All);
}

// Test that TopK with an axis attribute that is not the last dimension. Set largest to 0.
TEST_F(QnnCPUBackendTests, TopK_NonLastAxis_Largest_0) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {test::MakeAttribute("axis", static_cast<int64_t>(1)),
                           test::MakeAttribute("largest", static_cast<int64_t>(0))},
                          ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest floats from last axis
TEST_F(QnnCPUBackendTests, TopK_LargestFloats_LastAxis) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {},  // Attributes
                          ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest floats from last axis. Largest to 0.
TEST_F(QnnCPUBackendTests, TopK_LargestFloats_LastAxis_Largest_0) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {test::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                          ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest floats from last axis. Set largest to 0. Set sorted to false.
TEST_F(QnnCPUBackendTests, TopK_LargestFloats_LastAxis_Largest_0_Sorted) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {test::MakeAttribute("largest", static_cast<int64_t>(0)),
                           test::MakeAttribute("sorted", static_cast<int64_t>(0))},  // Attributes
                          ExpectedEPNodeAssignment::All,
                          /*opset*/ 19,
                          /*verify_outputs*/ false);  // Disable verify output. Unsorted doesn't guarantee output sequence
}

// Test TopK on CPU backend: top 2 largest int32s from last axis
TEST_F(QnnCPUBackendTests, TopK_LargestInt32s_LastAxis) {
  std::vector<int32_t> input_data = {-6, -5, -4, -3, -2, 0, 1, 2, 3, 4, 5, 6};
  RunTopKTestOnCPU<int32_t>(TestInputDef<int32_t>({1, 2, 2, 3}, false, input_data),
                            TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                            {},  // Attributes
                            ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest int32s from last axis. Set largest to 0.
TEST_F(QnnCPUBackendTests, TopK_LargestInt32s_LastAxis_Largest_0) {
  std::vector<int32_t> input_data = {-6, -5, -4, -3, -2, 0, 1, 2, 3, 4, 5, 6};
  RunTopKTestOnCPU<int32_t>(TestInputDef<int32_t>({1, 2, 2, 3}, false, input_data),
                            TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                            {test::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                            ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest int32s from last axis. Set largest to 0. Set sorted to false.
TEST_F(QnnCPUBackendTests, TopK_LargestInt32s_LastAxis_Largest_0_Sorted) {
  std::vector<int32_t> input_data = {-6, -5, -4, -3, -2, 0, 1, 2, 3, 4, 5, 6};
  RunTopKTestOnCPU<int32_t>(TestInputDef<int32_t>({1, 2, 2, 3}, false, input_data),
                            TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                            {test::MakeAttribute("largest", static_cast<int64_t>(0)),
                             test::MakeAttribute("sorted", static_cast<int64_t>(0))},  // Attributes
                            ExpectedEPNodeAssignment::All,
                            /*opset*/ 19,
                            /*verify_outputs*/ false);  // Disable verify output. Unsorted doesn't guarantee output sequence
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that creates a graph with a QDQ TopK operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQTopKTestCase(const TestInputDef<float>& input_def,
                                                  const TestInputDef<int64_t>& k_def,
                                                  const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                  bool use_contrib_qdq = false) {
  return [input_def, k_def, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                                    std::vector<QuantParams<QuantType>>& output_qparams) {
    // input
    MakeTestInput(builder, "input", input_def);

    // input -> Q -> DQ -> input_qdq
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                  input_qparams.scale, input_qparams.zero_point, use_contrib_qdq);

    // K input
    MakeTestInput(builder, "k", k_def);

    builder.AddNode("TopK",
                    "TopK",
                    {input_qdq, "k"},
                    {"values", "indices"},
                    kOnnxDomain,
                    attrs);

    // NOTE: Create output QDQ nodes before the TopK node so that TopK's 'values' output is the graph's first output.
    output_qparams[0] = input_qparams;  // Input and output qparams must be equal.
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder,
                                                     "qdq_values",
                                                     "values",
                                                     input_qparams.scale,
                                                     input_qparams.zero_point,
                                                     use_contrib_qdq);
    builder.MakeOutput("indices");
  };
}

// Runs a QDQ TopK model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename QType>
static void RunQDQTopKTestOnHTP(const TestInputDef<float>& input_def,
                                const TestInputDef<int64_t>& k_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 19,
                                bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildTopKTestCase<float>(input_def, k_def, attrs);
  auto qdq_model_builder = BuildQDQTopKTestCase<QType>(input_def, k_def, attrs, use_contrib_qdq);
  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Exercise the INT64 input/output casts around the complete HTP decomposition.
TEST_F(QnnHTPBackendTests, TopK_Ties_Int64GraphOutputs) {
  for (int64_t axis : {-1, 1, -2}) {
    for (int64_t largest : {0, 1}) {
      for (int64_t k : {1, 3, 6}) {
        SCOPED_TRACE("axis=" + std::to_string(axis) + ", largest=" + std::to_string(largest) +
                     ", k=" + std::to_string(k));
        const auto input_def = axis == -1
                                   ? TestInputDef<int64_t>({1, 6}, false, {5, 5, -1, 5, -2, 5})
                                   : TestInputDef<int64_t>({1, 6, 2}, false,
                                                           {5, 0, 5, 0, -1, 2, 5, 0, -2, 1, 5, 0});
        RunQnnModelTest(BuildTopKTestCase<int64_t>(input_def, TestInputDef<int64_t>({1}, true, {k}),
                                                   {test::MakeAttribute("axis", axis),
                                                    test::MakeAttribute("largest", largest)}),
                        {{"backend_type", "htp"}}, 19,
                        EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.0f)});
      }
    }
  }
}

// Check the same tie rule after QDQ folding, where duplicate quantized values are common.
TEST_F(QnnHTPBackendTests, TopK_Ties_U8_PreferSmallerIndices) {
  for (int64_t largest : {0, 1}) {
    for (int64_t k : {1, 3, 6}) {
      SCOPED_TRACE("largest=" + std::to_string(largest) + ", k=" + std::to_string(k));
      for (const auto& data : std::vector<std::vector<float>>{
               {0, 0, 0, 0, 0, 0}, {5, 5, 1, 5, 2, 5}, {1, 4, 4, 4, 0, 4}}) {
        RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 6}, false, data),
                                     TestInputDef<int64_t>({1}, true, {k}),
                                     {test::MakeAttribute("largest", largest)}, ExpectedEPNodeAssignment::All);
      }
    }
  }
}

TEST_F(QnnHTPBackendTests, TopK_Ties_U8_NonLastAxis) {
  for (int64_t axis : {1, -2}) {
    for (int64_t largest : {0, 1}) {
      RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 6, 2}, false,
                                                       {5, 0, 5, 0, 1, 2, 5, 0, 2, 1, 5, 0}),
                                   TestInputDef<int64_t>({1}, true, {3}),
                                   {test::MakeAttribute("axis", axis), test::MakeAttribute("largest", largest)},
                                   ExpectedEPNodeAssignment::All);
    }
  }
}

TEST_F(QnnHTPBackendTests, TopK_Ties_U16_PreferSmallerIndices) {
  for (int64_t largest : {0, 1}) {
    RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 6}, false, {5, 5, 1, 5, 2, 5}),
                                  TestInputDef<int64_t>({1}, true, {3}),
                                  {test::MakeAttribute("largest", largest)}, ExpectedEPNodeAssignment::All, 21);
  }
}

TEST_F(QnnHTPBackendTests, TopK_Ties_U8_LongZeroRun) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 1280}, false, std::vector<float>(1280, 0.0f)),
                               TestInputDef<int64_t>({1}, true, {20}), {}, ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ TopK on HTP backend: top 2 largest floats from last axis
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U8_LastAxis) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                               {},  // Attributes
                               ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ TopK on HTP backend: top 2 largest floats from last axis. Set largest to 0.
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U8_LastAxis_Largest_0) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                               {test::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                               ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ TopK on HTP backend: non-last axis
TEST_F(QnnHTPBackendTests, TopK_U8_NonLastAxis) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                               {test::MakeAttribute("axis", static_cast<int64_t>(1))},  // Attributes
                               ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ TopK on HTP backend: non-last axis. Set largest to 0.
TEST_F(QnnHTPBackendTests, TopK_U8_NonLastAxis_Largest_0) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                               {test::MakeAttribute("axis", static_cast<int64_t>(1)),
                                test::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                               ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ TopK on HTP backend: top 2 largest floats from last axis
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U16_LastAxis) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                21);  // opset
}

// Test 16-bit QDQ TopK on HTP backend: top 2 largest floats from last axis. Set largest to 0.
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U16_LastAxis_Largest_0) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {test::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                21);  // opset
}

// Test 16-bit QDQ TopK on HTP backend: non-last axis
TEST_F(QnnHTPBackendTests, TopK_U16_NonLastAxis) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {test::MakeAttribute("axis", static_cast<int64_t>(1))},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                21);  // opset
}

// Test 16-bit QDQ TopK on HTP backend: non-last axis. Set largest to 0.
TEST_F(QnnHTPBackendTests, TopK_U16_NonLastAxis_Largest_0) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {test::MakeAttribute("axis", static_cast<int64_t>(1)),
                                 test::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                21);  // opset
}
// A quantized TopK must fold its surrounding DQ/Q into one quantized QNN TopK node. TopK's second
// output is integer indices and is never Q-wrapped, which used to make the node group
// unrepresentable and left TopK running on dequantized float32 data.
// Checks graph structure, not accuracy: the float island is value-preserving either way.
TEST_F(QnnHTPBackendTests, TopK_QDQ_U8_FoldsQDQIntoQuantizedTopK) {
  const std::filesystem::path json_qnn_graph_dir = "TopK_QDQ_U8_FoldsQDQIntoQuantizedTopK";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  auto input_def = TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48));
  auto k_def = TestInputDef<int64_t>({1}, true /* is_initializer */, {2});

  TestQDQModelAccuracy(BuildTopKTestCase<float>(input_def, k_def, {}),
                       BuildQDQTopKTestCase<uint8_t>(input_def, k_def, {}),
                       provider_options,
                       /*opset_version=*/19,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",
                       {},
                       std::nullopt,
                       nullptr,
                       nullptr,
                       /*skip_v68_qdq=*/false);

  // `qdq_in_dq` is the DequantizeLinear feeding TopK; folding it in is what keeps TopK quantized.
  AssertOpInQnnGraph(json_qnn_graph_dir, "TopK", 1);
  AssertNodeNotInQnnGraph(json_qnn_graph_dir, "qdq_in_dq");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
