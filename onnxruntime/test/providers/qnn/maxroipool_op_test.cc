// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

#include "test/util/include/test_utils.h"

#include <onnx/onnx_pb.h>

namespace onnxruntime {
namespace test {

// Returns a function that creates a graph with a single MaxRoiPool operator.
// ONNX MaxRoiPool has two inputs: X [N, C, H, W] and rois [num_rois, 5] laid out as
// [batch_index, x1, y1, x2, y2].
static GetTestModelFn BuildMaxRoiPoolTestCase(const TestInputDef<float>& input_def,
                                              const TestInputDef<float>& roi_def,
                                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, roi_def, attrs](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", input_def);
    MakeTestInput<float>(builder, "rois", roi_def);

    builder.AddNode("maxroipool_node", "MaxRoiPool", {"X", "rois"}, {"Y"}, "", attrs);

    builder.MakeOutput("Y");
  };
}

// Returns a function that creates a graph with a QDQ MaxRoiPool operator. MaxRoiPool is decomposed
// into StridedSlice/ReduceMax/Concat, all of which run quantized on the HTP backend.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildMaxRoiPoolQDQTestCase(const TestInputDef<float>& input_def,
                                                        const TestInputDef<float>& roi_def,
                                                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, roi_def, attrs](ModelTestBuilder& builder,
                                     std::vector<QuantParams<QuantType>>& output_qparams) {
    // X -> Q -> DQ ->
    MakeTestInput<float>(builder, "X", input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    std::string input_qdq = AddQDQNodePair<QuantType>(builder, "qdq1", "X", input_qparams.scale, input_qparams.zero_point);

    // rois -> Q -> DQ ->
    MakeTestInput<float>(builder, "rois", roi_def);
    QuantParams<QuantType> roi_qparams = GetTestInputQuantParams<QuantType>(roi_def);
    std::string roi_qdq = AddQDQNodePair<QuantType>(builder, "qdq2", "rois", roi_qparams.scale, roi_qparams.zero_point);

    builder.AddNode("maxroipool_node", "MaxRoiPool", {input_qdq, roi_qdq}, {"maxroipool_output"}, "", attrs);

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "qdq_out", "maxroipool_output",
        output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

// Runs a MaxRoiPool model on the QNN CPU/HTP backend. Checks the graph node assignment, and that
// inference outputs for QNN and CPU match.
static void RunMaxRoiPoolOpTest(const TestInputDef<float>& input_def,
                                const TestInputDef<float>& roi_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                const std::string& backend_name = "cpu",
                                int opset = 13,
                                float f32_abs_err = 1e-4f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";
  if (backend_name != "cpu") {
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  }

  RunQnnModelTest(BuildMaxRoiPoolTestCase(input_def, roi_def, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment, f32_abs_err);
}

// Runs a QDQ MaxRoiPool model on the QNN HTP backend. Checks the graph node assignment, and that
// inference outputs for QNN and CPU match.
template <typename QuantType>
static void RunQDQMaxRoiPoolOpTest(const TestInputDef<float>& input_def,
                                   const TestInputDef<float>& roi_def,
                                   const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   int opset = 13) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);

  TestQDQModelAccuracy(BuildMaxRoiPoolTestCase(input_def, roi_def, attrs),
                       BuildMaxRoiPoolQDQTestCase<QuantType>(input_def, roi_def, attrs),
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

//
// CPU tests:
//
TEST_F(QnnCPUBackendTests, TestMaxRoiPool) {
  RunMaxRoiPoolOpTest(TestInputDef<float>({1, 1, 4, 4}, false,
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                           9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}),
                      TestInputDef<float>({1, 5}, true, {0.0f, 0.0f, 0.0f, 3.0f, 3.0f}),
                      {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                       test::MakeAttribute("spatial_scale", 1.0f)},
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, TestMaxRoiPool_spatial_scale) {
  RunMaxRoiPoolOpTest(TestInputDef<float>({1, 1, 4, 4}, false,
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                           9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}),
                      TestInputDef<float>({1, 5}, true, {0.0f, 0.0f, 0.0f, 6.0f, 6.0f}),
                      {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                       test::MakeAttribute("spatial_scale", 0.5f)},
                      ExpectedEPNodeAssignment::All);
}

// MaxRoiPool requires the rois to be a constant initializer (bin geometry is computed at build
// time). A non-constant rois input must not be assigned to QNN.
TEST_F(QnnCPUBackendTests, TestMaxRoiPool_NonConstRois_Unsupported) {
  RunMaxRoiPoolOpTest(TestInputDef<float>({1, 1, 4, 4}, false,
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                           9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}),
                      TestInputDef<float>({1, 5}, false, {0.0f, 0.0f, 0.0f, 3.0f, 3.0f}),
                      {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                       test::MakeAttribute("spatial_scale", 1.0f)},
                      ExpectedEPNodeAssignment::None);
}

// Adaptive binning: a 4x4 ROI pooled into a 3x3 grid produces non-uniform / overlapping bins.
// Exercises the per-bin StridedSlice + ReduceMax decomposition against the ORT CPU reference.
TEST_F(QnnCPUBackendTests, TestMaxRoiPool_AdaptiveBins) {
  RunMaxRoiPoolOpTest(TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(0.0f, 32.0f, 32)),
                      TestInputDef<float>({1, 5}, true, {0.0f, 0.0f, 0.0f, 3.0f, 3.0f}),
                      {test::MakeAttribute("pooled_shape", std::vector<int64_t>{3, 3}),
                       test::MakeAttribute("spatial_scale", 1.0f)},
                      ExpectedEPNodeAssignment::All);
}

// Empty-bin path: a ROI extending past the boundary (y2=6 > H=4) yields empty bins after clamping.
TEST_F(QnnCPUBackendTests, TestMaxRoiPool_EmptyBins) {
  RunMaxRoiPoolOpTest(TestInputDef<float>({1, 1, 4, 4}, false,
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                           9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}),
                      TestInputDef<float>({1, 5}, true, {0.0f, 0.0f, 3.0f, 1.0f, 6.0f}),
                      {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                       test::MakeAttribute("spatial_scale", 1.0f)},
                      ExpectedEPNodeAssignment::All);
}

// Multi-ROI: num_rois > 1 exercises the final Concat branch (single-ROI takes a reshape-only path).
TEST_F(QnnCPUBackendTests, TestMaxRoiPool_MultiRoi) {
  RunMaxRoiPoolOpTest(TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(0.0f, 32.0f, 32)),
                      TestInputDef<float>({3, 5}, true, {0.0f, 0.0f, 0.0f, 3.0f, 3.0f, 0.0f, 1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f}),
                      {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                       test::MakeAttribute("spatial_scale", 1.0f)},
                      ExpectedEPNodeAssignment::All);
}

// Cross-image sampling: N > 1 feature map with ROIs whose batch_index selects different images.
TEST_F(QnnCPUBackendTests, TestMaxRoiPool_MultiImage) {
  RunMaxRoiPoolOpTest(TestInputDef<float>({2, 1, 4, 4}, false, GetFloatDataInRange(0.0f, 32.0f, 32)),
                      TestInputDef<float>({2, 5}, true, {0.0f, 0.0f, 0.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f, 3.0f, 3.0f}),
                      {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                       test::MakeAttribute("spatial_scale", 1.0f)},
                      ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// HTP tests:
//
// MaxRoiPool is decomposed into StridedSlice/ReduceMax/Concat, which the HTP backend supports
// with 8-bit quantization, so the QDQ model fully offloads to QNN.
TEST_F(QnnHTPBackendTests, TestMaxRoiPoolQdq) {
  RunQDQMaxRoiPoolOpTest<uint8_t>(TestInputDef<float>({1, 1, 4, 4}, false,
                                                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}),
                                  TestInputDef<float>({1, 5}, true, {0.0f, 0.0f, 0.0f, 3.0f, 3.0f}),
                                  {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                                   test::MakeAttribute("spatial_scale", 1.0f)},
                                  ExpectedEPNodeAssignment::All);
}

// Adaptive binning on HTP (QDQ).
TEST_F(QnnHTPBackendTests, TestMaxRoiPoolQdq_AdaptiveBins) {
  RunQDQMaxRoiPoolOpTest<uint8_t>(TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(0.0f, 32.0f, 32)),
                                  TestInputDef<float>({1, 5}, true, {0.0f, 0.0f, 0.0f, 3.0f, 3.0f}),
                                  {test::MakeAttribute("pooled_shape", std::vector<int64_t>{3, 3}),
                                   test::MakeAttribute("spatial_scale", 1.0f)},
                                  ExpectedEPNodeAssignment::All);
}

// Multi-ROI on HTP: exercises the final Concat branch.
TEST_F(QnnHTPBackendTests, TestMaxRoiPoolQdq_MultiRoi) {
  RunQDQMaxRoiPoolOpTest<uint8_t>(TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(0.0f, 32.0f, 32)),
                                  TestInputDef<float>({3, 5}, true, {0.0f, 0.0f, 0.0f, 3.0f, 3.0f, 0.0f, 1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f}),
                                  {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                                   test::MakeAttribute("spatial_scale", 1.0f)},
                                  ExpectedEPNodeAssignment::All);
}

// Empty-bin path on HTP. All-negative data so the empty bins (filled with 0.0) pin the output max,
// forcing a non-zero output zero_point.
TEST_F(QnnHTPBackendTests, TestMaxRoiPoolQdq_EmptyBins) {
  RunQDQMaxRoiPoolOpTest<uint8_t>(TestInputDef<float>({1, 1, 4, 4}, false,
                                                      {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f,
                                                       -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f, -16.0f}),
                                  TestInputDef<float>({1, 5}, true, {0.0f, 0.0f, 3.0f, 1.0f, 6.0f}),
                                  {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                                   test::MakeAttribute("spatial_scale", 1.0f)},
                                  ExpectedEPNodeAssignment::All);
}

// Cross-image sampling on HTP: ROIs select different images.
TEST_F(QnnHTPBackendTests, TestMaxRoiPoolQdq_MultiImage) {
  RunQDQMaxRoiPoolOpTest<uint8_t>(TestInputDef<float>({2, 1, 4, 4}, false, GetFloatDataInRange(0.0f, 32.0f, 32)),
                                  TestInputDef<float>({2, 5}, true, {0.0f, 0.0f, 0.0f, 3.0f, 3.0f, 1.0f, 0.0f, 0.0f, 3.0f, 3.0f}),
                                  {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                                   test::MakeAttribute("spatial_scale", 1.0f)},
                                  ExpectedEPNodeAssignment::All);
}

// spatial_scale on HTP (QDQ): rois are scaled before binning.
TEST_F(QnnHTPBackendTests, TestMaxRoiPoolQdq_spatial_scale) {
  RunQDQMaxRoiPoolOpTest<uint8_t>(TestInputDef<float>({1, 1, 4, 4}, false,
                                                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}),
                                  TestInputDef<float>({1, 5}, true, {0.0f, 0.0f, 0.0f, 6.0f, 6.0f}),
                                  {test::MakeAttribute("pooled_shape", std::vector<int64_t>{2, 2}),
                                   test::MakeAttribute("spatial_scale", 0.5f)},
                                  ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
