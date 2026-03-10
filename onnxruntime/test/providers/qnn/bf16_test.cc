// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

// Helper function to create a simple Add model for BF16 testing
[[maybe_unused]] static GetTestModelFn BuildBF16AddTestCase(const TestInputDef<float>& input1_def,
                                                            const TestInputDef<float>& input2_def) {
  return [input1_def, input2_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input1", input1_def);
    MakeTestInput<float>(builder, "input2", input2_def);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("add_node", "Add", {"input1", "input2"}, {"output"}, "", attributes);
    builder.MakeOutput("output");
  };
}

// Helper function to create a simple MatMul model for BF16 testing
[[maybe_unused]] static GetTestModelFn BuildBF16MatMulTestCase(const TestInputDef<float>& input1_def,
                                                               const TestInputDef<float>& input2_def) {
  return [input1_def, input2_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input1", input1_def);
    MakeTestInput<float>(builder, "input2", input2_def);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("matmul_node", "MatMul", {"input1", "input2"}, {"output"}, "", attributes);
    builder.MakeOutput("output");
  };
}

// Helper function to create a Conv model for BF16 testing
[[maybe_unused]] static GetTestModelFn BuildBF16ConvTestCase(const TestInputDef<float>& input_def,
                                                             const TestInputDef<float>& weights_def) {
  return [input_def, weights_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    MakeTestInput<float>(builder, "weights", weights_def);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("conv_node", "Conv", {"input", "weights"}, {"output"}, "", attributes);
    builder.MakeOutput("output");
  };
}

// Helper function to run BF16 model test
[[maybe_unused]] static void RunBF16ModelTest(const GetTestModelFn& build_test_case,
                                              const std::vector<int64_t>& input_shape,
                                              ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                                              int opset = 18,
                                              float fp32_abs_err = 1e-2f) {
  ORT_UNUSED_PARAMETER(input_shape);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["htp_bf16_enable"] = "1";  // Enable BF16 mode
  provider_options["soc_model"] = "88";       // TODO: Use QnnTypes.h when it's availible
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(build_test_case, provider_options, opset, expected_ep_assignment, fp32_abs_err);
}

#if defined(__aarch64__) || defined(_M_ARM64)

//
// HTP BF16 tests:
//

// Test BF16 handling with Add operator - both inputs dynamic
TEST_F(QnnHTPBackendTests, DISABLED_BF16_Add_DynamicInputs) {
  std::vector<int64_t> shape = {2, 3, 4};
  RunBF16ModelTest(
      BuildBF16AddTestCase(
          TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.0f, 0.1f)),
          TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.1f, 0.1f))),
      shape);
}

// Test BF16 handling with Add operator - one input static (initializer)
TEST_F(QnnHTPBackendTests, DISABLED_BF16_Add_StaticInput) {
  std::vector<int64_t> shape = {2, 3, 4};
  RunBF16ModelTest(
      BuildBF16AddTestCase(
          TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.0f, 0.1f)),
          TestInputDef<float>(shape, true, GetSequentialFloatData(shape, 0.1f, 0.1f))),
      shape);
}

// Test BF16 handling with Add operator - both inputs static
TEST_F(QnnHTPBackendTests, DISABLED_BF16_Add_BothStatic) {
  std::vector<int64_t> shape = {2, 3, 4};
  RunBF16ModelTest(
      BuildBF16AddTestCase(
          TestInputDef<float>(shape, true, GetSequentialFloatData(shape, 0.0f, 0.1f)),
          TestInputDef<float>(shape, true, GetSequentialFloatData(shape, 0.1f, 0.1f))),
      shape);
}

// Test BF16 handling with MatMul operator - dynamic inputs
TEST_F(QnnHTPBackendTests, DISABLED_BF16_MatMul_DynamicInputs) {
  RunBF16ModelTest(
      BuildBF16MatMulTestCase(
          TestInputDef<float>({2, 3}, false, GetSequentialFloatData({2, 3}, 0.0f, 0.1f)),
          TestInputDef<float>({3, 4}, false, GetSequentialFloatData({3, 4}, 0.1f, 0.1f))),
      {2, 3});
}

// Test BF16 handling with MatMul operator - static weight
TEST_F(QnnHTPBackendTests, DISABLED_BF16_MatMul_StaticWeight) {
  RunBF16ModelTest(
      BuildBF16MatMulTestCase(
          TestInputDef<float>({2, 3}, false, GetSequentialFloatData({2, 3}, 0.0f, 0.1f)),
          TestInputDef<float>({3, 4}, true, GetSequentialFloatData({3, 4}, 0.1f, 0.1f))),
      {2, 3});
}

// Test BF16 handling with MatMul operator - batched inputs
TEST_F(QnnHTPBackendTests, DISABLED_BF16_MatMul_BatchedInputs) {
  RunBF16ModelTest(
      BuildBF16MatMulTestCase(
          TestInputDef<float>({2, 3, 4}, false, GetSequentialFloatData({2, 3, 4}, 0.0f, 0.1f)),
          TestInputDef<float>({4, 5}, false, GetSequentialFloatData({4, 5}, 0.1f, 0.1f))),
      {2, 3, 4});
}

// Test BF16 handling with Conv operator - dynamic input
TEST_F(QnnHTPBackendTests, DISABLED_BF16_Conv_DynamicInput) {
  std::vector<int64_t> input_shape = {1, 3, 8, 8};
  std::vector<int64_t> weights_shape = {16, 3, 3, 3};

  RunBF16ModelTest(
      BuildBF16ConvTestCase(
          TestInputDef<float>(input_shape, false, GetSequentialFloatData(input_shape, 0.0f, 0.01f)),
          TestInputDef<float>(weights_shape, true, GetSequentialFloatData(weights_shape, -0.1f, 0.01f))),
      input_shape);
}

// Test BF16 handling with Conv operator - larger input
TEST_F(QnnHTPBackendTests, DISABLED_BF16_Conv_LargerInput) {
  std::vector<int64_t> input_shape = {1, 64, 32, 32};
  std::vector<int64_t> weights_shape = {128, 64, 3, 3};

  RunBF16ModelTest(
      BuildBF16ConvTestCase(
          TestInputDef<float>(input_shape, false, GetSequentialFloatData(input_shape, 0.0f, 0.001f)),
          TestInputDef<float>(weights_shape, true, GetSequentialFloatData(weights_shape, -0.05f, 0.001f))),
      input_shape,
      ExpectedEPNodeAssignment::All,
      18,
      1e-1f);  // Larger tolerance for bigger models
}

// Test BF16 handling with multiple operations in sequence
static GetTestModelFn BuildBF16MultiOpTestCase() {
  return [](ModelTestBuilder& builder) {
    std::vector<int64_t> shape = {2, 3, 4};

    // Create inputs
    MakeTestInput<float>(builder, "input1", TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.0f, 0.1f)));
    MakeTestInput<float>(builder, "input2", TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.1f, 0.1f)));
    MakeTestInput<float>(builder, "input3", TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.2f, 0.1f)));

    // Add1: input1 + input2
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes1;
    builder.AddNode("add1_node", "Add", {"input1", "input2"}, {"add1_output"}, "", attributes1);

    // Add2: add1_output + input3
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes2;
    builder.AddNode("add2_node", "Add", {"add1_output", "input3"}, {"output"}, "", attributes2);
    builder.MakeOutput("output");
  };
}

TEST_F(QnnHTPBackendTests, DISABLED_BF16_MultipleOps) {
  std::vector<int64_t> shape = {2, 3, 4};
  RunBF16ModelTest(BuildBF16MultiOpTestCase(), shape);
}

// Test BF16 handling with graph that has multiple outputs
static GetTestModelFn BuildBF16MultiOutputTestCase() {
  return [](ModelTestBuilder& builder) {
    std::vector<int64_t> shape = {2, 3, 4};

    // Create inputs
    MakeTestInput<float>(builder, "input1", TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.0f, 0.1f)));
    MakeTestInput<float>(builder, "input2", TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.1f, 0.1f)));

    // Add: input1 + input2 -> output1
    std::vector<ONNX_NAMESPACE::AttributeProto> add_attributes;
    builder.AddNode("add_node", "Add", {"input1", "input2"}, {"output1"}, "", add_attributes);
    builder.MakeOutput("output1");

    // Mul: input1 * input2 -> output2
    std::vector<ONNX_NAMESPACE::AttributeProto> mul_attributes;
    builder.AddNode("mul_node", "Mul", {"input1", "input2"}, {"output2"}, "", mul_attributes);
    builder.MakeOutput("output2");
  };
}

TEST_F(QnnHTPBackendTests, DISABLED_BF16_MultipleOutputs) {
  std::vector<int64_t> shape = {2, 3, 4};
  RunBF16ModelTest(BuildBF16MultiOutputTestCase(), shape);
}

// Test BF16 handling with Relu activation
static GetTestModelFn BuildBF16ReluTestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("relu_node", "Relu", {"input"}, {"output"}, "", attributes);
    builder.MakeOutput("output");
  };
}

TEST_F(QnnHTPBackendTests, DISABLED_BF16_Relu) {
  std::vector<int64_t> shape = {2, 3, 4, 5};
  RunBF16ModelTest(
      BuildBF16ReluTestCase(
          TestInputDef<float>(shape, false, GetSequentialFloatData(shape, -1.0f, 0.1f))),
      shape);
}

// Test BF16 handling with Sigmoid activation
static GetTestModelFn BuildBF16SigmoidTestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("sigmoid_node", "Sigmoid", {"input"}, {"output"}, "", attributes);
    builder.MakeOutput("output");
  };
}

TEST_F(QnnHTPBackendTests, DISABLED_BF16_Sigmoid) {
  std::vector<int64_t> shape = {2, 3, 4};
  RunBF16ModelTest(
      BuildBF16SigmoidTestCase(
          TestInputDef<float>(shape, false, GetSequentialFloatData(shape, -2.0f, 0.2f))),
      shape);
}

// Test BF16 handling with Softmax
static GetTestModelFn BuildBF16SoftmaxTestCase(const TestInputDef<float>& input_def, int64_t axis) {
  return [input_def, axis](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("axis", axis));
    builder.AddNode("softmax_node", "Softmax", {"input"}, {"output"}, "", attributes);
    builder.MakeOutput("output");
  };
}

TEST_F(QnnHTPBackendTests, DISABLED_BF16_Softmax) {
  std::vector<int64_t> shape = {2, 3, 4};
  RunBF16ModelTest(
      BuildBF16SoftmaxTestCase(
          TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.0f, 0.1f)),
          -1),
      shape);
}

// Test BF16 handling with Transpose
static GetTestModelFn BuildBF16TransposeTestCase(const TestInputDef<float>& input_def,
                                                 const std::vector<int64_t>& perm) {
  return [input_def, perm](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeIntsAttribute("perm", perm));
    builder.AddNode("transpose_node", "Transpose", {"input"}, {"output"}, "", attributes);
    builder.MakeOutput("output");
  };
}

TEST_F(QnnHTPBackendTests, DISABLED_BF16_Transpose) {
  std::vector<int64_t> shape = {2, 3, 4, 5};
  std::vector<int64_t> perm = {0, 2, 1, 3};
  RunBF16ModelTest(
      BuildBF16TransposeTestCase(
          TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.0f, 0.1f)),
          perm),
      shape);
}

// Test BF16 handling with Reshape
static GetTestModelFn BuildBF16ReshapeTestCase(const TestInputDef<float>& input_def,
                                               const std::vector<int64_t>& new_shape) {
  return [input_def, new_shape](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    builder.MakeInitializer<int64_t>("shape", {static_cast<int64_t>(new_shape.size())}, new_shape);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("reshape_node", "Reshape", {"input", "shape"}, {"output"}, "", attributes);
    builder.MakeOutput("output");
  };
}

TEST_F(QnnHTPBackendTests, DISABLED_BF16_Reshape) {
  std::vector<int64_t> input_shape = {2, 3, 4};
  std::vector<int64_t> output_shape = {6, 4};
  RunBF16ModelTest(
      BuildBF16ReshapeTestCase(
          TestInputDef<float>(input_shape, false, GetSequentialFloatData(input_shape, 0.0f, 0.1f)),
          output_shape),
      input_shape);
}

// Test BF16 handling with Concat
static GetTestModelFn BuildBF16ConcatTestCase(const std::vector<TestInputDef<float>>& input_defs, int64_t axis) {
  return [input_defs, axis](ModelTestBuilder& builder) {
    std::vector<std::string> input_names;
    for (size_t i = 0; i < input_defs.size(); i++) {
      std::string input_name = "input" + std::to_string(i);
      MakeTestInput<float>(builder, input_name, input_defs[i]);
      input_names.push_back(input_name);
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("axis", axis));
    builder.AddNode("concat_node", "Concat", input_names, {"output"}, "", attributes);
    builder.MakeOutput("output");
  };
}

TEST_F(QnnHTPBackendTests, DISABLED_BF16_Concat) {
  std::vector<int64_t> shape1 = {2, 3, 4};
  std::vector<int64_t> shape2 = {2, 5, 4};
  std::vector<TestInputDef<float>> input_defs = {
      TestInputDef<float>(shape1, false, GetSequentialFloatData(shape1, 0.0f, 0.1f)),
      TestInputDef<float>(shape2, false, GetSequentialFloatData(shape2, 0.5f, 0.1f))};
  RunBF16ModelTest(BuildBF16ConcatTestCase(input_defs, 1), shape1);
}

// Test BF16 handling with Split
static GetTestModelFn BuildBF16SplitTestCase(const TestInputDef<float>& input_def, int64_t axis, int64_t num_outputs) {
  return [input_def, axis, num_outputs](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    std::vector<std::string> output_names;
    for (int64_t i = 0; i < num_outputs; i++) {
      std::string output_name = "output" + std::to_string(i);
      output_names.push_back(output_name);
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("axis", axis));
    builder.AddNode("split_node", "Split", {"input"}, output_names, "", attributes);

    for (const auto& output_name : output_names) {
      builder.MakeOutput(output_name);
    }
  };
}

TEST_F(QnnHTPBackendTests, DISABLED_BF16_Split) {
  std::vector<int64_t> shape = {2, 6, 4};
  RunBF16ModelTest(
      BuildBF16SplitTestCase(
          TestInputDef<float>(shape, false, GetSequentialFloatData(shape, 0.0f, 0.1f)),
          1,
          2),
      shape);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
