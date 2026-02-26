// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cassert>
#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

#include "test/util/include/test_utils.h"

#include <onnx/onnx_pb.h>
#include <fstream>

namespace onnxruntime {
namespace test {

// Returns a function that creates a graph with a single Roialign operator.
static GetTestModelFn BuildRoialignTestCase(const TestInputDef<float>& input_def,
                                            const TestInputDef<float>& roi_def,
                                            const TestInputDef<int64_t>& batch_indices_def,
                                            const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, roi_def, batch_indices_def, attrs](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput(builder, input_def);
    NodeArg* roi = MakeTestInput(builder, roi_def);
    NodeArg* batch_indices = MakeTestInput(builder, batch_indices_def);

    std::vector<NodeArg*> inputs{input, roi, batch_indices};

    NodeArg* output = builder.MakeOutput();
    Node& roialign_node = builder.AddNode("RoiAlign", inputs, {output});

    for (const auto& attr : attrs) {
      roialign_node.AddAttributeProto(attr);
    }
  };
}

// Returns a function that creates a graph with a QDQ Roialign operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildRoialignQDQTestCase(const TestInputDef<float>& input_def,
                                                      const TestInputDef<float>& roi_def,
                                                      const TestInputDef<int64_t>& batch_indices_def,
                                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, roi_def, batch_indices_def, attrs](ModelTestBuilder& builder,
                                                        std::vector<QuantParams<QuantType>>& output_qparams) {
    std::vector<NodeArg*> inputs;
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point);
    inputs.push_back(input_qdq);

    // roi -> Q -> DQ ->
    NodeArg* roi = MakeTestInput(builder, roi_def);
    QuantParams<QuantType> roi_qparams = GetTestInputQuantParams<QuantType>(roi_def);
    NodeArg* roi_qdq = AddQDQNodePair<QuantType>(builder, roi, roi_qparams.scale, roi_qparams.zero_point);
    inputs.push_back(roi_qdq);

    NodeArg* batch_indices = MakeTestInput(builder, batch_indices_def);
    inputs.push_back(batch_indices);

    NodeArg* output = builder.MakeIntermediate();
    Node& roialign_node = builder.AddNode("RoiAlign", inputs, {output});

    for (const auto& attr : attrs) {
      roialign_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, output, output_qparams[0].scale,
                                                     output_qparams[0].zero_point);
  };
}

// Runs an Roialign model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunRoiAlignOpTest(const TestInputDef<float>& input_def,
                              const TestInputDef<float>& roi_def,
                              const TestInputDef<int64_t>& batch_indices_def,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              const std::string& backend_name = "cpu",
                              int opset = 16,
                              float f32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["soc_model"] = "87";

  RunQnnModelTest(BuildRoialignTestCase(input_def, roi_def, batch_indices_def, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment, f32_abs_err);
}

// Runs a QDQ Roialign model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
template <typename QuantType>
static void RunQDQRoiAlignOpTest(const TestInputDef<float>& input_def,
                                 const TestInputDef<float>& roi_def,
                                 const TestInputDef<int64_t>& batch_indices_def,
                                 const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 16) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["soc_model"] = "87";

  TestQDQModelAccuracy(BuildRoialignTestCase(input_def, roi_def, batch_indices_def, attrs),
                       BuildRoialignQDQTestCase<QuantType>(input_def, roi_def, batch_indices_def, attrs),
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

//
// CPU tests:
//

TEST_F(QnnCPUBackendTests, TestRoialign) {
  RunRoiAlignOpTest(TestInputDef<float>({1, 1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                    TestInputDef<float>({1, 4}, true, {0.0f, 0.0f, 1.0f, 1.0f}),
                    TestInputDef<int64_t>({1}, true, {0}),
                    {utils::MakeAttribute("coordinate_transformation_mode", "output_half_pixel"),
                     utils::MakeAttribute("mode", "avg"),
                     utils::MakeAttribute("sampling_ratio", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_height", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_width", static_cast<int64_t>(1)),
                     utils::MakeAttribute("spatial_scale", 1.0f)},
                    ExpectedEPNodeAssignment::All);
}

// QNN doesn't support coordinate_transformation_mode = output_half
TEST_F(QnnCPUBackendTests, TestRoialign_Unsupported_output_half) {
  RunRoiAlignOpTest(TestInputDef<float>({1, 1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                    TestInputDef<float>({1, 4}, true, {0.0f, 0.0f, 1.0f, 1.0f}),
                    TestInputDef<int64_t>({1}, true, {0}),
                    {utils::MakeAttribute("coordinate_transformation_mode", "output_half"),
                     utils::MakeAttribute("mode", "avg"),
                     utils::MakeAttribute("sampling_ratio", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_height", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_width", static_cast<int64_t>(1)),
                     utils::MakeAttribute("spatial_scale", 1.0f)},
                    ExpectedEPNodeAssignment::None);
}

// QNN only supports pooling mode = average
TEST_F(QnnCPUBackendTests, TestRoialign_Unsupported_mode_max) {
  RunRoiAlignOpTest(TestInputDef<float>({1, 1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                    TestInputDef<float>({1, 4}, true, {0.0f, 0.0f, 1.0f, 1.0f}),
                    TestInputDef<int64_t>({1}, true, {0}),
                    {utils::MakeAttribute("coordinate_transformation_mode", "output_half_pixel"),
                     utils::MakeAttribute("mode", "max"),  //  mode = max will Unsupported
                     utils::MakeAttribute("sampling_ratio", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_height", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_width", static_cast<int64_t>(1)),
                     utils::MakeAttribute("spatial_scale", 1.0f)},
                    ExpectedEPNodeAssignment::None);
}

// QNN doesn't support adaptive sampling_ratio (sampling_ratio = 0)
TEST_F(QnnCPUBackendTests, TestRoialign_Unsupported_sampling_ratio) {
  RunRoiAlignOpTest(TestInputDef<float>({1, 1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                    TestInputDef<float>({1, 4}, true, {0.0f, 0.0f, 1.0f, 1.0f}),
                    TestInputDef<int64_t>({1}, true, {0}),
                    {utils::MakeAttribute("coordinate_transformation_mode", "output_half_pixel"),
                     utils::MakeAttribute("mode", "avg"),
                     utils::MakeAttribute("sampling_ratio", static_cast<int64_t>(0)),
                     utils::MakeAttribute("output_height", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_width", static_cast<int64_t>(1)),
                     utils::MakeAttribute("spatial_scale", 1.0f)},
                    ExpectedEPNodeAssignment::None);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// HTP tests:
//
TEST_F(QnnHTPBackendTests, TestRoialign) {
  RunRoiAlignOpTest(TestInputDef<float>({1, 1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                    TestInputDef<float>({1, 4}, true, {0.0f, 0.0f, 1.0f, 1.0f}),
                    TestInputDef<int64_t>({1}, true, {0}),
                    {utils::MakeAttribute("coordinate_transformation_mode", "output_half_pixel"),
                     utils::MakeAttribute("mode", "avg"),
                     utils::MakeAttribute("sampling_ratio", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_height", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_width", static_cast<int64_t>(1)),
                     utils::MakeAttribute("spatial_scale", 1.0f)},
                    ExpectedEPNodeAssignment::All,
                    "htp");
}

// QNN doesn't support coordinate_transformation_mode = output_half
TEST_F(QnnHTPBackendTests, TestRoialign_Unsupported_output_half) {
  RunRoiAlignOpTest(TestInputDef<float>({1, 1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                    TestInputDef<float>({1, 4}, true, {0.0f, 0.0f, 1.0f, 1.0f}),
                    TestInputDef<int64_t>({1}, true, {0}),
                    {utils::MakeAttribute("coordinate_transformation_mode", "output_half"),
                     utils::MakeAttribute("mode", "avg"),
                     utils::MakeAttribute("sampling_ratio", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_height", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_width", static_cast<int64_t>(1)),
                     utils::MakeAttribute("spatial_scale", 1.0f)},
                    ExpectedEPNodeAssignment::None,
                    "htp");
}

// QNN only supports pooling mode = average
TEST_F(QnnHTPBackendTests, TestRoialign_Unsupported_mode_max) {
  RunRoiAlignOpTest(TestInputDef<float>({1, 1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                    TestInputDef<float>({1, 4}, true, {0.0f, 0.0f, 1.0f, 1.0f}),
                    TestInputDef<int64_t>({1}, true, {0}),
                    {utils::MakeAttribute("coordinate_transformation_mode", "output_half_pixel"),
                     utils::MakeAttribute("mode", "max"),  //  mode = max will Unsupported
                     utils::MakeAttribute("sampling_ratio", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_height", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_width", static_cast<int64_t>(1)),
                     utils::MakeAttribute("spatial_scale", 1.0f)},
                    ExpectedEPNodeAssignment::None,
                    "htp");
}

// QNN doesn't support adaptive sampling_ratio (sampling_ratio = 0)
TEST_F(QnnHTPBackendTests, TestRoialign_Unsupported_sampling_ratio) {
  RunRoiAlignOpTest(TestInputDef<float>({1, 1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                    TestInputDef<float>({1, 4}, true, {0.0f, 0.0f, 1.0f, 1.0f}),
                    TestInputDef<int64_t>({1}, true, {0}),
                    {utils::MakeAttribute("coordinate_transformation_mode", "output_half_pixel"),
                     utils::MakeAttribute("mode", "avg"),
                     utils::MakeAttribute("sampling_ratio", static_cast<int64_t>(0)),
                     utils::MakeAttribute("output_height", static_cast<int64_t>(1)),
                     utils::MakeAttribute("output_width", static_cast<int64_t>(1)),
                     utils::MakeAttribute("spatial_scale", 1.0f)},
                    ExpectedEPNodeAssignment::None,
                    "htp");
}

//
// QDQ Roialign
TEST_F(QnnHTPBackendTests, TestRoialignQdq) {
  RunQDQRoiAlignOpTest<uint8_t>(TestInputDef<float>({1, 1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                                TestInputDef<float>({1, 4}, true, {0.0f, 0.0f, 1.0f, 1.0f}),
                                TestInputDef<int64_t>({1}, true, {0}),
                                {utils::MakeAttribute("coordinate_transformation_mode", "output_half_pixel"),
                                 utils::MakeAttribute("mode", "avg"),
                                 utils::MakeAttribute("sampling_ratio", static_cast<int64_t>(1)),
                                 utils::MakeAttribute("output_height", static_cast<int64_t>(1)),
                                 utils::MakeAttribute("output_width", static_cast<int64_t>(1)),
                                 utils::MakeAttribute("spatial_scale", 1.0f)},
                                ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
