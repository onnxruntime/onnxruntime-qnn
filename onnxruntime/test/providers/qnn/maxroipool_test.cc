// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

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

    builder.AddNode("maxroipool_node", "MaxRoiPool", {"X", "rois"}, {"Y"}, kOnnxDomain, attrs);

    builder.MakeOutput("Y");
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
                                float f32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";
  if (backend_name != "cpu") {
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  }

  RunQnnModelTest(BuildMaxRoiPoolTestCase(input_def, roi_def, attrs),
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(f32_abs_err)});
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

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
