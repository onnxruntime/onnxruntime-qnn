// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Test for "index-out-of-bounds" bug that occurred when a Slice operator
// shared one of its initializer inputs with another op that was processed by QNN EP first.
TEST_F(QnnCPUBackendTests, Slice_SharedInitializersBugFix) {
  // Model with an Add that processes a shared initializer before Slice is processed.
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    builder.MakeInput<int32_t>("input0", {2, 2}, {1, 2, 3, 4});

    // Initializers
    builder.Make1DInitializer<int32_t>("starts_input", {1, 0});  // Shared by Add
    builder.Make1DInitializer<int32_t>("ends_input", {2, 2});
    builder.Make1DInitializer<int32_t>("axes_input", {0, 1});
    builder.Make1DInitializer<int32_t>("steps_input", {1, 1});

    // Add input0 with a shared initializer.
    builder.AddNode("Add",
                    "Add",
                    {"input0", "starts_input"},
                    {"add_output"});

    // Cast Add's output to float.
    std::vector<ONNX_NAMESPACE::AttributeProto> cast_attrs;
    cast_attrs.reserve(1);
    cast_attrs.push_back(MakeAttribute(
        "to",
        static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)));

    builder.AddNode("Cast",
                    "Cast",
                    {"add_output"},
                    {"cast_output"},
                    kOnnxDomain,
                    cast_attrs);

    // Slice Cast's output
    builder.MakeOutput("Y");
    builder.AddNode("Slice",
                    "Slice",
                    {"cast_output", "starts_input", "ends_input", "axes_input", "steps_input"},
                    {"Y"});
  };

  ProviderOptions provider_options;

  provider_options["backend_type"] = "cpu";

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

/**
 * Runs an Slice model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param data_def The data input's definition (shape, is_initializer, data).
 * \param starts_def The starts input's definition.
 * \param ends_def The ends input's definition.
 * \param axes_def The axes input's definition.
 * \param steps_def The steps input's definition.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param use_contrib_qdq Force Q/DQ ops to use the com.microsoft domain (enable 16-bit).
 */
template <typename QuantType = uint8_t>
static void RunSliceQDQTest(const TestInputDef<float>& data_def,
                            const TestInputDef<int64_t>& starts_def,
                            const TestInputDef<int64_t>& ends_def,
                            const TestInputDef<int64_t>& axes_def,
                            const TestInputDef<int64_t>& steps_def,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  const std::vector<TestInputDef<float>> f32_inputs = {data_def};
  const std::vector<TestInputDef<int64_t>> int64_inputs = {starts_def, ends_def, axes_def, steps_def};

  TestQDQModelAccuracy(BuildOpTestCase<float, int64_t>("Slice_node", "Slice", f32_inputs, int64_inputs, {}),
                       BuildQDQOpTestCase<QuantType, int64_t>("Slice_node", "Slice", f32_inputs, int64_inputs, {}, kOnnxDomain,
                                                              use_contrib_qdq),
                       provider_options,
                       18,
                       expected_ep_assignment);
}

/**
 * Runs an Slice model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param data_def The data (int32_t) input's definition (shape, is_initializer, data).
 * \param starts_def The starts input's definition.
 * \param ends_def The ends input's definition.
 * \param axes_def The axes input's definition.
 * \param steps_def The steps input's definition.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 */
template <typename DataType>
static void RunSliceNonQDQOnHTP(const TestInputDef<DataType>& data_def,
                                const TestInputDef<int64_t>& starts_def,
                                const TestInputDef<int64_t>& ends_def,
                                const TestInputDef<int64_t>& axes_def,
                                const TestInputDef<int64_t>& steps_def,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  auto f32_model_builder = BuildOpTestCase<DataType, int64_t>("Slice_node", "Slice", {data_def},
                                                              {starts_def, ends_def, axes_def, steps_def}, {});
  RunQnnModelTest(f32_model_builder,
                  provider_options,
                  13,
                  expected_ep_assignment);
}

// Check that QNN compiles DQ -> Slice -> Q as a single unit.
TEST_F(QnnHTPBackendTests, SliceSmallDataQDQU8) {
  RunSliceQDQTest(TestInputDef<float>({8}, false, 0.0f, 1.0f),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {-1}),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {2}),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Slice -> Q as a single unit.
TEST_F(QnnHTPBackendTests, SliceLargePositiveDataQDQU8) {
  RunSliceQDQTest(TestInputDef<float>({5120}, false, 0.0f, 1.0f),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {-1}),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {2}),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Slice -> Q as a single unit.
TEST_F(QnnHTPBackendTests, SliceLargeNegativeDataQDQU8) {
  RunSliceQDQTest(TestInputDef<float>({5120}, false, 0.0f, 1.0f),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {-1}),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {2}),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN supports Slice with int32 data input on HTP
TEST_F(QnnHTPBackendTests, SliceInt32OnHTP) {
  RunSliceNonQDQOnHTP<int32_t>(TestInputDef<int32_t>({5120}, false, -100, 100),
                               TestInputDef<int64_t>({1}, true, {0}),
                               TestInputDef<int64_t>({1}, true, {-1}),
                               TestInputDef<int64_t>({1}, true, {0}),
                               TestInputDef<int64_t>({1}, true, {2}),
                               ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Slice with more than 1 axis.
TEST_F(QnnHTPBackendTests, SliceU8_MultAxes) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunSliceQDQTest<uint8_t>(TestInputDef<float>({2, 4}, false, input_data),
                           TestInputDef<int64_t>({2}, true, {1, 0}),  // starts
                           TestInputDef<int64_t>({2}, true, {2, 3}),  // ends
                           TestInputDef<int64_t>({2}, true, {0, 1}),  // axes
                           TestInputDef<int64_t>({2}, true, {1, 2}),  // steps
                           ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Slice with more than 1 axis.
TEST_F(QnnHTPBackendTests, SliceU16_MultAxes) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunSliceQDQTest<uint16_t>(TestInputDef<float>({2, 4}, false, input_data),
                            TestInputDef<int64_t>({2}, true, {1, 0}),  // starts
                            TestInputDef<int64_t>({2}, true, {2, 3}),  // ends
                            TestInputDef<int64_t>({2}, true, {0, 1}),  // axes
                            TestInputDef<int64_t>({2}, true, {1, 2}),  // steps
                            ExpectedEPNodeAssignment::All,
                            true);  // Use com.microsoft Q/DQ ops for 16-bit
}

// Test 8-bit QDQ Slice with more than 1 axis and an end value that exceeds the associated dimension size.
TEST_F(QnnHTPBackendTests, SliceU8_MultAxes_LargeEnd) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunSliceQDQTest<uint8_t>(TestInputDef<float>({2, 4}, false, input_data),
                           TestInputDef<int64_t>({2}, true, {0, 1}),      // starts
                           TestInputDef<int64_t>({2}, true, {-1, 1000}),  // ends
                           TestInputDef<int64_t>({2}, true, {0, 1}),      // axes
                           TestInputDef<int64_t>({2}, true, {1, 1}),      // steps
                           ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Slice on a partial subset of axes for a 3D tensor.
// Slices only axis 1 of a [2,4,3] tensor, leaving axes 0 and 2 unsliced.
// This exercises the begin_mask/end_mask parameters for unspecified axes.
TEST_F(QnnHTPBackendTests, SliceU8_PartialAxes_3D) {
  RunSliceQDQTest<uint8_t>(TestInputDef<float>({2, 4, 3}, false, 0.0f, 1.0f),
                           TestInputDef<int64_t>({1}, true, {1}),  // starts: axis 1 from index 1
                           TestInputDef<int64_t>({1}, true, {3}),  // ends: axis 1 to index 3
                           TestInputDef<int64_t>({1}, true, {1}),  // axes: only axis 1
                           TestInputDef<int64_t>({1}, true, {1}),  // steps: step 1
                           ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Slice on a single axis of a 4D tensor.
// Slices only axis 2 of a [1,3,4,2] tensor, leaving axes 0, 1, and 3 unsliced.
// This exercises begin_mask/end_mask with multiple unspecified axes.
TEST_F(QnnHTPBackendTests, SliceU8_PartialAxes_4D) {
  RunSliceQDQTest<uint8_t>(TestInputDef<float>({1, 3, 4, 2}, false, 0.0f, 1.0f),
                           TestInputDef<int64_t>({1}, true, {0}),  // starts: axis 2 from index 0
                           TestInputDef<int64_t>({1}, true, {2}),  // ends: axis 2 to index 2
                           TestInputDef<int64_t>({1}, true, {2}),  // axes: only axis 2
                           TestInputDef<int64_t>({1}, true, {1}),  // steps: step 1
                           ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Slice with negative step (backward slicing).
// Slices axis 0 of a [4,3] tensor in reverse.
TEST_F(QnnHTPBackendTests, SliceU8_NegativeStep) {
  RunSliceQDQTest<uint8_t>(TestInputDef<float>({4, 3}, false, 0.0f, 1.0f),
                           TestInputDef<int64_t>({1}, true, {3}),   // starts: axis 0 from index 3
                           TestInputDef<int64_t>({1}, true, {0}),   // ends: axis 0 to index 0 (exclusive)
                           TestInputDef<int64_t>({1}, true, {0}),   // axes: only axis 0
                           TestInputDef<int64_t>({1}, true, {-1}),  // steps: step -1 (reverse)
                           ExpectedEPNodeAssignment::All);
}

// Test int32 Slice on a partial subset of axes for a 3D tensor on HTP.
TEST_F(QnnHTPBackendTests, SliceInt32_PartialAxes_3D) {
  RunSliceNonQDQOnHTP<int32_t>(TestInputDef<int32_t>({2, 4, 3}, false, -50, 50),
                                TestInputDef<int64_t>({1}, true, {1}),  // starts
                                TestInputDef<int64_t>({1}, true, {3}),  // ends
                                TestInputDef<int64_t>({1}, true, {1}),  // axes: only axis 1
                                TestInputDef<int64_t>({1}, true, {1}),  // steps
                                ExpectedEPNodeAssignment::All);
}

// Verify that begin_mask and end_mask scalar params are present on the QNN StridedSlice node
// when slicing a partial subset of axes. Dumps the QNN graph to JSON and inspects it.
// Uses a non-QDQ int32 model to directly test the op builder's parameter generation.
TEST_F(QnnHTPBackendTests, SliceInt32_PartialAxes_VerifyMaskParams) {
  const std::filesystem::path json_qnn_graph_dir = "SliceInt32_PartialAxes_VerifyMaskParams";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  // Slice only axis 1 of a [2,4,3] int32 tensor. Axes 0 and 2 are unspecified,
  // so begin_mask and end_mask should have bits 0 and 2 set.
  auto model_builder = BuildOpTestCase<int32_t, int64_t>(
      "Slice_node", "Slice",
      {TestInputDef<int32_t>({2, 4, 3}, false, -50, 50)},
      {TestInputDef<int64_t>({1}, true, {1}),   // starts
       TestInputDef<int64_t>({1}, true, {3}),   // ends
       TestInputDef<int64_t>({1}, true, {1}),   // axes: only axis 1
       TestInputDef<int64_t>({1}, true, {1})},  // steps
      {});

  RunQnnModelTest(model_builder,
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_qnn_graph_dir, "StridedSlice");
  AssertOpHasScalarParam(json_qnn_graph_dir, "StridedSlice", "begin_mask");
  AssertOpHasScalarParam(json_qnn_graph_dir, "StridedSlice", "end_mask");
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
