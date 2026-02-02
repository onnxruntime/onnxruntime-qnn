// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "core/graph/node_attr_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

template <typename DataType>
GetTestModelFn BuildSplitTestCase(const TestInputDef<DataType>& input_def,
                                  const std::vector<int64_t>& split, bool split_is_input,
                                  int64_t axis, int64_t num_outputs) {
  return [input_def, split, split_is_input, axis, num_outputs](ModelTestBuilder& builder) {
    // Determine the actual number of outputs from the 'split' or 'num_outputs' arguments.
    // In opset 18, the num_outputs attribute or the split input can determine the actual number of outputs.
    // In opset 13, the split input determines the number of actual outputs.
    // In opsets < 13, the split attribute determines the number of actual outputs.
    const size_t actual_num_outputs = (num_outputs > -1) ? static_cast<size_t>(num_outputs) : split.size();

    // input
    MakeTestInput<DataType>(builder, "input", input_def);

    // split input (initializer)
    if (split_is_input && !split.empty()) {
      builder.Make1DInitializer<int64_t>("split", split);
    }

    // outputs
    std::vector<std::string> split_outputs;
    split_outputs.reserve(actual_num_outputs);
    for (size_t i = 0; i < actual_num_outputs; i++) {
      split_outputs.push_back("Y" + std::to_string(i));
      builder.MakeOutput(split_outputs.back());
    }

    // attributes passed during node creation
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.reserve(3);

    if (!split_is_input && !split.empty()) {
      attrs.push_back(test::MakeAttribute("split", split));
    }

    if (num_outputs > -1) {
      attrs.push_back(test::MakeAttribute("num_outputs", num_outputs));
    }

    attrs.push_back(test::MakeAttribute("axis", axis));

    std::vector<std::string> inputs;
    inputs.reserve(2);
    inputs.push_back("input");
    if (split_is_input && !split.empty()) {
      inputs.push_back("split");
    }

    builder.AddNode("Split",
                    "Split",
                    inputs,
                    split_outputs,
                    kOnnxDomain,
                    attrs);
  };
}

template <typename DataType>
static void RunSplitOpTestOnCPU(const TestInputDef<DataType>& input_def,
                                const std::vector<int64_t>& split,
                                int64_t axis,
                                int64_t num_outputs,
                                int opset,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "cpu";

  const bool split_is_input = opset >= 13;
  RunQnnModelTest(BuildSplitTestCase<DataType>(input_def, split, split_is_input, axis, num_outputs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test Split opset 18 on CPU backend: equal split of axis 0 via 'num_outputs' attribute
// and 'split' input.
TEST_F(QnnCPUBackendTests, Split_Equal_Axis0_Opset18) {
  // Use 'split' input (initializer).
  RunSplitOpTestOnCPU<float>(TestInputDef<float>({4, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                             {2, 2},  // split
                             0,       // axis
                             -1,      // num_outputs
                             18,      // opset
                             ExpectedEPNodeAssignment::All);
  RunSplitOpTestOnCPU<int32_t>(TestInputDef<int32_t>({4, 2}, false, {1, 2, 3, 4, 5, 6, 7, 8}),
                               {2, 2},  // split
                               0,       // axis
                               -1,      // num_outputs
                               18,      // opset
                               ExpectedEPNodeAssignment::All);

  // Use 'num_outputs' attribute.
  RunSplitOpTestOnCPU<float>(TestInputDef<float>({4, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                             {},  // split (use num_outputs instead)
                             0,   // axis
                             2,   // num_outputs
                             18,  // opset
                             ExpectedEPNodeAssignment::All);
  RunSplitOpTestOnCPU<int32_t>(TestInputDef<int32_t>({4, 2}, false, {1, 2, 3, 4, 5, 6, 7, 8}),
                               {},  // split (use num_outputs instead)
                               0,   // axis
                               2,   // num_outputs
                               18,  // opset
                               ExpectedEPNodeAssignment::All);
}

// Test Split opset 13 on CPU backend: equal split of axis 0
TEST_F(QnnCPUBackendTests, Split_Equal_Axis0_Opset13) {
  RunSplitOpTestOnCPU<float>(TestInputDef<float>({4, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                             {2, 2},  // split
                             0,       // axis
                             -1,      // num_outputs (not in opset 13)
                             13,      // opset
                             ExpectedEPNodeAssignment::All);
  RunSplitOpTestOnCPU<int32_t>(TestInputDef<int32_t>({4, 2}, false, {1, 2, 3, 4, 5, 6, 7, 8}),
                               {2, 2},  // split
                               0,       // axis
                               -1,      // num_outputs (not in opset 13)
                               13,      // opset
                               ExpectedEPNodeAssignment::All);
}

// Test Split opset 11 on CPU backend: equal split of axis 0
TEST_F(QnnCPUBackendTests, Split_Equal_Axis0_Opset11) {
  RunSplitOpTestOnCPU<float>(TestInputDef<float>({4, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                             {2, 2},  // split
                             0,       // axis
                             -1,      // num_outputs (not in opset 11)
                             11,      // opset
                             ExpectedEPNodeAssignment::All);
  RunSplitOpTestOnCPU<int32_t>(TestInputDef<int32_t>({4, 2}, false, {1, 2, 3, 4, 5, 6, 7, 8}),
                               {2, 2},  // split
                               0,       // axis
                               -1,      // num_outputs (not in opset 11)
                               11,      // opset
                               ExpectedEPNodeAssignment::All);
}

// Test Split opset 13 on CPU backend: unequal split of axis 1
TEST_F(QnnCPUBackendTests, Split_Unequal_Axis1_Opset13) {
  RunSplitOpTestOnCPU<float>(TestInputDef<float>({2, 4}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                             {1, 3},  // split
                             1,       // axis
                             -1,      // num_outputs (not in opset 13)
                             13,      // opset
                             ExpectedEPNodeAssignment::All);
  RunSplitOpTestOnCPU<int32_t>(TestInputDef<int32_t>({2, 4}, false, {1, 2, 3, 4, 5, 6, 7, 8}),
                               {1, 3},  // split
                               1,       // axis
                               -1,      // num_outputs (not in opset 13)
                               13,      // opset
                               ExpectedEPNodeAssignment::All);
}

// Test Split opset 11 on CPU backend: unequal split of axis 1
TEST_F(QnnCPUBackendTests, Split_Unequal_Axis1_Opset11) {
  RunSplitOpTestOnCPU<float>(TestInputDef<float>({2, 4}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                             {1, 3},  // split
                             1,       // axis
                             -1,      // num_outputs (not in opset 11)
                             11,      // opset
                             ExpectedEPNodeAssignment::All);
  RunSplitOpTestOnCPU<int32_t>(TestInputDef<int32_t>({2, 4}, false, {1, 2, 3, 4, 5, 6, 7, 8}),
                               {1, 3},  // split
                               1,       // axis
                               -1,      // num_outputs (not in opset 11)
                               11,      // opset
                               ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Return function that builds a model with a QDQ Split.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQSplitTestCase(const TestInputDef<float>& input_def,
                                                   const std::vector<int64_t>& split,
                                                   bool split_is_input,
                                                   int64_t axis,
                                                   int64_t num_outputs,
                                                   bool use_contrib_qdq = false) {
  return [input_def, split, split_is_input, axis, num_outputs,
          use_contrib_qdq](ModelTestBuilder& builder,
                           std::vector<QuantParams<QuantType>>& output_qparams) {
    // Determine the actual number of outputs from the 'split' or 'num_outputs' arguments.
    // In opset 18, the num_outputs attribute or the split input can determine the actual number of outputs.
    // In opset 13, the split input determines the number of actual outputs.
    // In opsets < 13, the split attribute determines the number of actual outputs.
    const size_t actual_num_outputs = (num_outputs > -1) ? static_cast<size_t>(num_outputs) : split.size();

    // input
    MakeTestInput<float>(builder, "input", input_def);

    // QDQ input: input -> Q -> DQ -> input_after_qdq
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_after_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                  input_qparams.scale, input_qparams.zero_point, use_contrib_qdq);

    // split input (initializer)
    if (split_is_input && !split.empty()) {
      builder.Make1DInitializer<int64_t>("split", split);
    }

    // split outputs (unquantized)
    std::vector<std::string> split_outputs;
    split_outputs.reserve(actual_num_outputs);
    for (size_t i = 0; i < actual_num_outputs; i++) {
      split_outputs.push_back("split_out" + std::to_string(i));
    }

    // attributes passed during node creation
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.reserve(3);

    if (!split_is_input && !split.empty()) {
      attrs.push_back(test::MakeAttribute("split", split));
    }

    if (num_outputs > -1) {
      attrs.push_back(test::MakeAttribute("num_outputs", num_outputs));
    }

    attrs.push_back(test::MakeAttribute("axis", axis));

    std::vector<std::string> inputs;
    inputs.reserve(2);
    inputs.push_back(input_after_qdq);
    if (split_is_input && !split.empty()) {
      inputs.push_back("split");
    }

    builder.AddNode("Split",
                    "Split",
                    inputs,
                    split_outputs,
                    kOnnxDomain,
                    attrs);

    // outputs: each split_out{i} -> Q -> DQ -> graph output
    assert(output_qparams.size() == actual_num_outputs);
    for (size_t i = 0; i < actual_num_outputs; i++) {
      // NOTE: Input and output quantization parameters must be equal for Split.
      output_qparams[i] = input_qparams;

      AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder,
                                                       "qdq_out" + std::to_string(i),
                                                       split_outputs[i],
                                                       output_qparams[i].scale,
                                                       output_qparams[i].zero_point,
                                                       use_contrib_qdq);
    }
  };
}

// Runs a non-QDQ Split operator on the HTP backend.
template <typename DataType>
static void RunSplitOpTestOnHTP(const TestInputDef<DataType>& input_def,
                                const std::vector<int64_t>& split,
                                int64_t axis,
                                int64_t num_outputs,
                                int opset,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";

  const bool split_is_input = opset >= 13;
  RunQnnModelTest(BuildSplitTestCase<DataType>(input_def, split, split_is_input, axis, num_outputs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ Split operator on the HTP backend.
template <typename QuantType>
static void RunQDQSplitOpTestOnHTP(const TestInputDef<float>& input_def,
                                   const std::vector<int64_t>& split,
                                   int64_t axis,
                                   int64_t num_outputs,
                                   int opset,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  const bool split_is_input = opset >= 13;
  auto f32_model_builder = BuildSplitTestCase<float>(input_def, split, split_is_input, axis, num_outputs);
  auto qdq_model_builder = BuildQDQSplitTestCase<QuantType>(input_def, split, split_is_input, axis, num_outputs,
                                                            use_contrib_qdq);
  TestQDQModelAccuracy<QuantType>(f32_model_builder,
                                  qdq_model_builder,
                                  provider_options,
                                  opset,
                                  expected_ep_assignment);
}

// Test that HTP can run non-QDQ Split (int32 input).
TEST_F(QnnHTPBackendTests, Split_Int32_Opset13) {
  // Equal split.
  RunSplitOpTestOnHTP<int32_t>(TestInputDef<int32_t>({4, 2}, false, {1, 2, 3, 4, 5, 6, 7, 8}),
                               {2, 2},  // split
                               0,       // axis
                               -1,      // num_outputs (not in opset 13)
                               13,      // opset
                               ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Split opset 18 on HTP backend: equal split of axis 0 via 'num_outputs' attribute
// and 'split' input.
TEST_F(QnnHTPBackendTests, Split_Equal_Axis0_Opset18) {
  // Split 6 into 3 outputs of lengths [2, 2, 2]
  TestInputDef<float> input_def({6, 2}, false,
                                {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f, 9.0f, 10.0f, 11.0f});

  // Use 'split' input (initializer).
  RunQDQSplitOpTestOnHTP<uint8_t>(input_def,
                                  {2, 2, 2},  // split
                                  0,          // axis
                                  -1,         // num_outputs
                                  18,         // opset
                                  ExpectedEPNodeAssignment::All);

  // Use 'num_outputs' attribute.
  RunQDQSplitOpTestOnHTP<uint8_t>(input_def,
                                  {},  // split (use num_outputs instead)
                                  0,   // axis
                                  3,   // num_outputs
                                  18,  // opset
                                  ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Split opset 18 on HTP backend. Use an uneven split (last chunk should be smaller).
TEST_F(QnnHTPBackendTests, Split_NonEqual_Axis0_Opset18) {
  // Split 7 into 3 outputs of lengths [3, 3, 1]
  TestInputDef<float> input_def({7, 2}, false,
                                {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f});

  // Use a `split` input with uneven split lengths.
  RunQDQSplitOpTestOnHTP<uint8_t>(input_def,
                                  {3, 3, 1},  // split
                                  0,          // axis
                                  -1,         // num_outputs
                                  18,         // opset
                                  ExpectedEPNodeAssignment::All);

  // Use a `num_outputs` attribute that does not evenly divide into shape[axis].
  RunQDQSplitOpTestOnHTP<uint8_t>(input_def,
                                  {},  // split (use num_outputs instead)
                                  0,   // axis
                                  3,   // num_outputs
                                  18,  // opset
                                  ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Split opset 18 on HTP backend: equal split of axis 0 via 'num_outputs' attribute
// and 'split' input.
TEST_F(QnnHTPBackendTests, Split_Equal_Axis0_Opset18_U16) {
  // Use 'split' input (initializer).
  RunQDQSplitOpTestOnHTP<uint16_t>(TestInputDef<float>({4, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                                   {2, 2},  // split
                                   0,       // axis
                                   -1,      // num_outputs
                                   18,      // opset
                                   ExpectedEPNodeAssignment::All,
                                   true);  // Use com.microsoft Q/DQ ops

  // Use 'num_outputs' attribute.
  RunQDQSplitOpTestOnHTP<uint16_t>(TestInputDef<float>({4, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                                   {},  // split (use num_outputs instead)
                                   0,   // axis
                                   2,   // num_outputs
                                   18,  // opset
                                   ExpectedEPNodeAssignment::All,
                                   true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ Split op on HTP backend: equal split on axis 0 with opset 13.
TEST_F(QnnHTPBackendTests, Split_Equal_Axis0_Opset13) {
  RunQDQSplitOpTestOnHTP<uint8_t>(TestInputDef<float>({4, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                                  {2, 2},  // split
                                  0,       // axis
                                  -1,      // num_outputs (not in opset 13)
                                  13,      // opset
                                  ExpectedEPNodeAssignment::All);
}

// Test QDQ Split op on HTP backend: equal split on axis 0 with opset 11.
TEST_F(QnnHTPBackendTests, Split_Equal_Axis0_Opset11) {
  RunQDQSplitOpTestOnHTP<uint8_t>(TestInputDef<float>({4, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                                  {2, 2},  // split
                                  0,       // axis
                                  -1,      // num_outputs (not in opset 11)
                                  11,      // opset
                                  ExpectedEPNodeAssignment::All);
}

// Test Split opset 13 on HTP backend: unequal split of axis 1
TEST_F(QnnHTPBackendTests, Split_Unequal_Axis1_Opset13) {
  RunQDQSplitOpTestOnHTP<uint8_t>(TestInputDef<float>({2, 4}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                                  {1, 3},  // split
                                  1,       // axis
                                  -1,      // num_outputs (not in opset 13)
                                  13,      // opset
                                  ExpectedEPNodeAssignment::All);
}

// Test Split opset 11 on HTP backend: unequal split of axis 1
TEST_F(QnnHTPBackendTests, Split_Unequal_Axis1_Opset11) {
  RunQDQSplitOpTestOnHTP<uint8_t>(TestInputDef<float>({2, 4}, false, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.f, 8.f}),
                                  {1, 3},  // split
                                  1,       // axis
                                  -1,      // num_outputs (not in opset 11)
                                  11,      // opset
                                  ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
