// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// ─────────────────────────────────────────────────────────────────────────────
// NMS model builder helpers
// ─────────────────────────────────────────────────────────────────────────────

// NonMaxSuppression model configuration.
struct NmsTestConfig {
  std::vector<float> boxes_data;      // Flat row-major [B, S, 4]
  std::vector<int64_t> boxes_shape;   // [B, S, 4]
  std::vector<float> scores_data;     // Flat row-major [B, C, S]
  std::vector<int64_t> scores_shape;  // [B, C, S]
  int64_t max_output_boxes = 0;
  float iou_threshold = 0.0f;
  float score_threshold = 0.0f;
  int64_t center_point_box = 0;
  // Static output shape declared in the model: [B*C*max_output_boxes, 3].
  // Must be set explicitly by the caller so QNN EP can claim the node.
  std::vector<int64_t> output_shape;
};

// Build a NonMaxSuppression float32 test model.
// Optional scalar inputs are constant initializers; output shape is static.
static GetTestModelFn BuildNmsTestCase(const NmsTestConfig& cfg) {
  return [cfg](ModelTestBuilder& builder) {
    builder.MakeInput<float>("boxes", cfg.boxes_shape, cfg.boxes_data);
    builder.MakeInput<float>("scores", cfg.scores_shape, cfg.scores_data);
    builder.MakeInitializer<int64_t>("max_output_boxes", {}, {cfg.max_output_boxes});
    builder.MakeInitializer<float>("iou_threshold", {}, {cfg.iou_threshold});
    builder.MakeInitializer<float>("score_threshold", {}, {cfg.score_threshold});
    builder.AddNode("nms_node", "NonMaxSuppression",
                    {"boxes", "scores", "max_output_boxes", "iou_threshold", "score_threshold"},
                    {"selected_indices"}, kOnnxDomain,
                    {test::MakeAttribute("center_point_box", cfg.center_point_box)});
    builder.MakeOutput<int64_t>("selected_indices", cfg.output_shape);
  };
}

// Build a QDQ-wrapped NMS model.
// Boxes and scores are wrapped in Q/DQ pairs; scalar params are constant initializers.
template <typename QuantType>
static GetTestModelFn BuildQdqNmsTestCase(const NmsTestConfig& cfg, bool use_ms_domain = false) {
  return [cfg, use_ms_domain](ModelTestBuilder& builder) {
    builder.MakeInput<float>("boxes", cfg.boxes_shape, cfg.boxes_data);
    builder.MakeInput<float>("scores", cfg.scores_shape, cfg.scores_data);

    float boxes_scale = 0.01f;
    float scores_scale = 0.005f;
    QuantType zero_pt = QuantType(0);

    std::string dq_boxes_name = AddQDQNodePair<QuantType>(builder, "qdq_boxes", "boxes",
                                                          boxes_scale, zero_pt, use_ms_domain);
    std::string dq_scores_name = AddQDQNodePair<QuantType>(builder, "qdq_scores", "scores",
                                                           scores_scale, zero_pt, use_ms_domain);

    builder.MakeInitializer<int64_t>("max_output_boxes", {}, {cfg.max_output_boxes});
    builder.MakeInitializer<float>("iou_threshold", {}, {cfg.iou_threshold});
    builder.MakeInitializer<float>("score_threshold", {}, {cfg.score_threshold});

    builder.AddNode("nms_node", "NonMaxSuppression",
                    {dq_boxes_name, dq_scores_name,
                     "max_output_boxes", "iou_threshold", "score_threshold"},
                    {"selected_indices"}, kOnnxDomain,
                    {test::MakeAttribute("center_point_box", cfg.center_point_box)});
    builder.MakeOutput<int64_t>("selected_indices", cfg.output_shape);
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider options
// ─────────────────────────────────────────────────────────────────────────────
static ProviderOptions CpuProviderOptions() {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";
  return opts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom NMS test runner
//
// QNN NMS output is padded to [B*C*max_boxes, 3] as int64 (after the INT_32->INT_64 Cast).
// ORT CPU reference output is unpadded [num_selected, 3] as int64.
//
// This runner compares the first exp_rows rows of the QNN output against the CPU reference.
// Both are sorted by (batch, class, box) triple before comparison so row ordering differences
// (allowed by the ONNX spec) do not cause false failures.
// ─────────────────────────────────────────────────────────────────────────────
static void RunNmsTest(const GetTestModelFn& build_test_case,
                       int opset_version,
                       ExpectedEPNodeAssignment expected_ep_assignment,
                       ProviderOptions provider_options = CpuProviderOptions()) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};
  ModelTestBuilder helper;
  build_test_case(helper);

  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  // Fallback tests: only verify QNN EP doesn't claim the node.
  // Skip the CPU reference run — the model may be invalid for CPU EP too.
  if (expected_ep_assignment == ExpectedEPNodeAssignment::None) {
    std::vector<Ort::Value> actual;
    InferenceModel(model_data, "NMS_QNN", provider_options,
                   expected_ep_assignment, helper.feeds_, actual);
    return;
  }

  // Run ORT CPU EP to get unpadded reference output.
  std::vector<Ort::Value> expected;
  InferenceModelCPU(model_data, "NMS_CPU", helper.feeds_, expected);

  // Run QNN EP.
  std::vector<Ort::Value> actual;
  InferenceModel(model_data, "NMS_QNN", provider_options,
                 expected_ep_assignment, helper.feeds_, actual);

  // Validate shapes.
  auto exp_shape = expected[0].GetTensorTypeAndShapeInfo().GetShape();
  auto act_shape = actual[0].GetTensorTypeAndShapeInfo().GetShape();

  ASSERT_EQ(exp_shape.size(), 2u) << "CPU NMS output must be 2D [K, 3]";
  ASSERT_EQ(act_shape.size(), 2u) << "QNN NMS output must be 2D [K, 3]";
  ASSERT_EQ(exp_shape[1], int64_t{3});
  ASSERT_EQ(act_shape[1], int64_t{3});

  int64_t exp_rows = exp_shape[0];
  int64_t act_rows = act_shape[0];
  ASSERT_GE(act_rows, exp_rows)
      << "QNN NMS output must have >= rows compared to CPU reference (padded vs. unpadded)";

  // Sort both row-sets and compare the first exp_rows rows.
  const int64_t* exp_data = expected[0].GetTensorData<int64_t>();
  const int64_t* act_data = actual[0].GetTensorData<int64_t>();

  using Row3 = std::array<int64_t, 3>;
  std::vector<Row3> exp_rows_vec, act_rows_vec;
  exp_rows_vec.reserve(static_cast<size_t>(exp_rows));
  act_rows_vec.reserve(static_cast<size_t>(exp_rows));

  for (int64_t r = 0; r < exp_rows; ++r) {
    exp_rows_vec.push_back({exp_data[r * 3], exp_data[r * 3 + 1], exp_data[r * 3 + 2]});
    act_rows_vec.push_back({act_data[r * 3], act_data[r * 3 + 1], act_data[r * 3 + 2]});
  }

  std::sort(exp_rows_vec.begin(), exp_rows_vec.end());
  std::sort(act_rows_vec.begin(), act_rows_vec.end());

  for (size_t i = 0; i < exp_rows_vec.size(); ++i) {
    EXPECT_EQ(exp_rows_vec[i], act_rows_vec[i])
        << "NMS index mismatch at sorted row " << i
        << ": expected [" << exp_rows_vec[i][0] << "," << exp_rows_vec[i][1] << "," << exp_rows_vec[i][2]
        << "] vs actual [" << act_rows_vec[i][0] << "," << act_rows_vec[i][1] << "," << act_rows_vec[i][2] << "]";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU float tests (QnnCPUBackendTests fixture)
// ─────────────────────────────────────────────────────────────────────────────

// 1-batch, 1-class, 6 boxes; iou=0.5, max_boxes=3 — basic suppression.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_Basic) {
  std::vector<float> boxes = {
      0.0f, 0.0f, 1.0f, 1.0f,      // box 0
      0.0f, 0.1f, 1.0f, 1.1f,      // box 1 (overlaps box 0 at iou>0.5)
      0.0f, -0.1f, 1.0f, 0.9f,     // box 2 (overlaps box 0 at iou>0.5)
      0.0f, 10.0f, 1.0f, 11.0f,    // box 3 (no overlap)
      0.0f, 10.1f, 1.0f, 11.1f,    // box 4 (overlaps box 3)
      0.0f, 100.0f, 1.0f, 101.0f,  // box 5 (no overlap)
  };
  std::vector<float> scores = {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};

  NmsTestConfig cfg;
  cfg.boxes_data = boxes;
  cfg.boxes_shape = {1, 6, 4};
  cfg.scores_data = scores;
  cfg.scores_shape = {1, 1, 6};
  cfg.max_output_boxes = 3;
  cfg.iou_threshold = 0.5f;
  cfg.score_threshold = 0.0f;
  cfg.center_point_box = 0;
  cfg.output_shape = {3, 3};  // B * C * max_output_boxes = 1 * 1 * 3

  RunNmsTest(BuildNmsTestCase(cfg), 11, ExpectedEPNodeAssignment::All);
}

// center_point_box=1: boxes are [x_center, y_center, width, height].
// QNN's NMS op only supports the diagonal-corners format, so center_point_box != 0 must
// fall back to the CPU EP rather than being claimed by QNN EP.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_Fallback_CenterPointBox) {
  std::vector<float> boxes = {
      0.5f, 0.5f, 1.0f, 1.0f,   // box 0
      0.5f, 0.6f, 1.0f, 1.0f,   // box 1 (overlaps box 0)
      0.5f, 10.5f, 1.0f, 1.0f,  // box 2 (no overlap)
  };
  std::vector<float> scores = {0.9f, 0.75f, 0.85f};

  NmsTestConfig cfg;
  cfg.boxes_data = boxes;
  cfg.boxes_shape = {1, 3, 4};
  cfg.scores_data = scores;
  cfg.scores_shape = {1, 1, 3};
  cfg.max_output_boxes = 2;
  cfg.iou_threshold = 0.5f;
  cfg.score_threshold = 0.0f;
  cfg.center_point_box = 1;
  cfg.output_shape = {2, 3};

  RunNmsTest(BuildNmsTestCase(cfg), 11, ExpectedEPNodeAssignment::None);
}

// iou_threshold=0.3: stricter suppression yields fewer selected boxes.
// NOTE: QNN CPU NMS errors (QNN_GRAPH_ERROR_INVALID_ARGUMENT) when
// actual_selected < max_boxes_selected AND max_boxes_selected < num_spatial.
// To avoid this, set max_output_boxes == num_spatial (4) so the allocated
// output capacity matches the total candidate count.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_IouThresholdLow) {
  std::vector<float> boxes = {
      0.0f, 0.0f, 1.0f, 1.0f,  // box 0
      0.0f, 0.4f, 1.0f, 1.4f,  // box 1 (moderate overlap — suppressed at 0.3)
      0.0f, 5.0f, 1.0f, 6.0f,  // box 2 (no overlap)
      0.0f, 5.5f, 1.0f, 6.5f,  // box 3 (overlaps box 2 — suppressed at 0.3)
  };
  std::vector<float> scores = {0.9f, 0.85f, 0.8f, 0.75f};

  NmsTestConfig cfg;
  cfg.boxes_data = boxes;
  cfg.boxes_shape = {1, 4, 4};
  cfg.scores_data = scores;
  cfg.scores_shape = {1, 1, 4};
  cfg.max_output_boxes = 4;  // == num_spatial; avoids QNN INVALID_ARGUMENT when max < num_spatial
  cfg.iou_threshold = 0.3f;
  cfg.score_threshold = 0.0f;
  cfg.center_point_box = 0;
  cfg.output_shape = {4, 3};  // padded; actual selected = 2 (boxes 0 and 2)

  RunNmsTest(BuildNmsTestCase(cfg), 11, ExpectedEPNodeAssignment::All);
}

// iou_threshold=0.7: looser suppression preserves more boxes.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_IouThresholdHigh) {
  std::vector<float> boxes = {
      0.0f, 0.0f, 1.0f, 1.0f,  // box 0
      0.0f, 0.4f, 1.0f, 1.4f,  // box 1 (IOU ~0.43 — not suppressed at 0.7)
      0.0f, 5.0f, 1.0f, 6.0f,  // box 2 (no overlap)
  };
  std::vector<float> scores = {0.9f, 0.85f, 0.8f};

  NmsTestConfig cfg;
  cfg.boxes_data = boxes;
  cfg.boxes_shape = {1, 3, 4};
  cfg.scores_data = scores;
  cfg.scores_shape = {1, 1, 3};
  cfg.max_output_boxes = 3;
  cfg.iou_threshold = 0.7f;
  cfg.score_threshold = 0.0f;
  cfg.center_point_box = 0;
  cfg.output_shape = {3, 3};

  RunNmsTest(BuildNmsTestCase(cfg), 11, ExpectedEPNodeAssignment::All);
}

// score_threshold=0.5: boxes below score 0.5 are discarded before IOU suppression.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_ScoreThreshold) {
  std::vector<float> boxes = {
      0.0f, 0.0f, 1.0f, 1.0f,  // box 0, score=0.9 (kept)
      0.0f, 0.0f, 1.0f, 1.0f,  // box 1, score=0.4 (filtered by score_threshold)
      0.0f, 5.0f, 1.0f, 6.0f,  // box 2, score=0.8 (kept)
      0.0f, 5.0f, 1.0f, 6.0f,  // box 3, score=0.2 (filtered by score_threshold)
  };
  std::vector<float> scores = {0.9f, 0.4f, 0.8f, 0.2f};

  NmsTestConfig cfg;
  cfg.boxes_data = boxes;
  cfg.boxes_shape = {1, 4, 4};
  cfg.scores_data = scores;
  cfg.scores_shape = {1, 1, 4};
  cfg.max_output_boxes = 4;
  cfg.iou_threshold = 0.5f;
  cfg.score_threshold = 0.5f;
  cfg.center_point_box = 0;
  cfg.output_shape = {4, 3};

  RunNmsTest(BuildNmsTestCase(cfg), 11, ExpectedEPNodeAssignment::All);
}

// max_output_boxes_per_class=1: only the top-scoring box per class is returned.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_MaxBoxesLimit) {
  std::vector<float> boxes = {
      0.0f, 0.0f, 1.0f, 1.0f,    // box 0
      0.0f, 5.0f, 1.0f, 6.0f,    // box 1 (no overlap)
      0.0f, 10.0f, 1.0f, 11.0f,  // box 2 (no overlap)
  };
  std::vector<float> scores = {0.9f, 0.85f, 0.8f};

  NmsTestConfig cfg;
  cfg.boxes_data = boxes;
  cfg.boxes_shape = {1, 3, 4};
  cfg.scores_data = scores;
  cfg.scores_shape = {1, 1, 3};
  cfg.max_output_boxes = 1;  // Limit to 1 box per class.
  cfg.iou_threshold = 0.5f;
  cfg.score_threshold = 0.0f;
  cfg.center_point_box = 0;
  cfg.output_shape = {1, 3};  // B * C * max_output_boxes = 1 * 1 * 1

  RunNmsTest(BuildNmsTestCase(cfg), 11, ExpectedEPNodeAssignment::All);
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative / fallback tests
// These configurations must be rejected by QNN EP (fall back to CPU EP).
// ─────────────────────────────────────────────────────────────────────────────

// Dynamic max_output_boxes_per_class (non-constant graph input) must fall back.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_Fallback_DynamicMaxBoxes) {
  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("boxes", {1, 3, 4},
                             {0.0f, 0.0f, 1.0f, 1.0f,
                              0.0f, 5.0f, 1.0f, 6.0f,
                              0.0f, 10.0f, 1.0f, 11.0f});
    builder.MakeInput<float>("scores", {1, 1, 3}, {0.9f, 0.85f, 0.8f});
    // Dynamic (graph input, not constant initializer): QNN EP must reject.
    builder.MakeInput<int64_t>("max_boxes", {}, {3});
    builder.AddNode("nms_node", "NonMaxSuppression",
                    {"boxes", "scores", "max_boxes"}, {"selected_indices"}, kOnnxDomain,
                    {test::MakeAttribute("center_point_box", int64_t{0})});
    builder.MakeOutput<int64_t>("selected_indices", std::vector<int64_t>{3, 3});
  };

  RunNmsTest(build_model, 11, ExpectedEPNodeAssignment::None);
}

// Dynamic iou_threshold (non-constant graph input) must fall back.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_Fallback_DynamicIouThreshold) {
  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("boxes", {1, 3, 4},
                             {0.0f, 0.0f, 1.0f, 1.0f,
                              0.0f, 5.0f, 1.0f, 6.0f,
                              0.0f, 10.0f, 1.0f, 11.0f});
    builder.MakeInput<float>("scores", {1, 1, 3}, {0.9f, 0.85f, 0.8f});
    builder.MakeInitializer<int64_t>("max_boxes", {}, {3});
    // Dynamic iou_threshold: QNN EP must reject.
    builder.MakeInput<float>("iou_threshold", {}, {0.5f});
    builder.AddNode("nms_node", "NonMaxSuppression",
                    {"boxes", "scores", "max_boxes", "iou_threshold"},
                    {"selected_indices"}, kOnnxDomain,
                    {test::MakeAttribute("center_point_box", int64_t{0})});
    builder.MakeOutput<int64_t>("selected_indices", std::vector<int64_t>{3, 3});
  };

  RunNmsTest(build_model, 11, ExpectedEPNodeAssignment::None);
}

// boxes rank 2 (instead of required rank 3) must fall back.
// REMOVED: a rank-2 boxes tensor violates the ONNX spec, so ORT's own
// ONNX validator rejects the model before it reaches any EP. This test
// was checking ORT validation, not a QNN-specific fallback.

// center_point_box=2 (invalid value not in {0,1}) must fall back.
// REMOVED: center_point_box=2 is rejected by ORT's CPU NMS kernel at
// session-init time (before any EP gets a chance). Not a QNN-specific test.

// ─────────────────────────────────────────────────────────────────────────────
// HTP tests
// ─────────────────────────────────────────────────────────────────────────────

#if defined(__aarch64__) || defined(_M_ARM64)

static ProviderOptions HtpNmsProviderOptions() {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  return opts;
}

// HTP float32: basic suppression on HTP with plain float32 inputs.
TEST_F(QnnHTPBackendTests, NonMaxSuppression_HTP_Float) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  std::vector<float> boxes = {
      0.0f, 0.0f, 0.5f, 0.5f,  // box 0
      0.0f, 0.1f, 0.5f, 0.6f,  // box 1 (overlaps box 0, suppressed)
      1.0f, 1.0f, 1.5f, 1.5f,  // box 2 (no overlap)
      1.0f, 1.1f, 1.5f, 1.6f,  // box 3 (overlaps box 2, suppressed)
  };
  std::vector<float> scores = {0.9f, 0.75f, 0.95f, 0.5f};

  NmsTestConfig cfg;
  cfg.boxes_data = boxes;
  cfg.boxes_shape = {1, 4, 4};
  cfg.scores_data = scores;
  cfg.scores_shape = {1, 1, 4};
  cfg.max_output_boxes = 2;
  cfg.iou_threshold = 0.5f;
  cfg.score_threshold = 0.0f;
  cfg.center_point_box = 0;
  cfg.output_shape = {2, 3};  // B * C * max_output_boxes = 1 * 1 * 2

  RunNmsTest(BuildNmsTestCase(cfg), 11,
             ExpectedEPNodeAssignment::All, HtpNmsProviderOptions());
}

// HTP QDQ uint8: boxes and scores wrapped in Q/DQ pairs (uint8).
TEST_F(QnnHTPBackendTests, NonMaxSuppression_HTP_QDQ_Uint8) {
  CONDITIONAL_SKIP_TEST_ON_LINUX_ARM64(HtpNmsProviderOptions(), QNN_HTP_DEVICE_ARCH_V68, "QDQ", uint8_t);

  std::vector<float> boxes = {
      0.0f, 0.0f, 0.5f, 0.5f,  // box 0
      0.0f, 0.1f, 0.5f, 0.6f,  // box 1 (overlaps box 0, suppressed)
      1.0f, 1.0f, 1.5f, 1.5f,  // box 2 (no overlap)
      1.0f, 1.1f, 1.5f, 1.6f,  // box 3 (overlaps box 2, suppressed)
  };
  std::vector<float> scores = {0.9f, 0.75f, 0.95f, 0.5f};

  NmsTestConfig cfg;
  cfg.boxes_data = boxes;
  cfg.boxes_shape = {1, 4, 4};
  cfg.scores_data = scores;
  cfg.scores_shape = {1, 1, 4};
  cfg.max_output_boxes = 2;
  cfg.iou_threshold = 0.5f;
  cfg.score_threshold = 0.0f;
  cfg.center_point_box = 0;
  cfg.output_shape = {2, 3};  // B * C * max_output_boxes = 1 * 1 * 2

  RunNmsTest(BuildQdqNmsTestCase<uint8_t>(cfg), 11,
             ExpectedEPNodeAssignment::All, HtpNmsProviderOptions());
}

TEST_F(QnnHTPBackendTests, NonMaxSuppression_HTP_QDQ_Uint16) {
  CONDITIONAL_SKIP_TEST_ON_LINUX_ARM64(HtpNmsProviderOptions(), QNN_HTP_DEVICE_ARCH_V68, "QDQ", uint16_t);

  std::vector<float> boxes = {
      0.0f, 0.0f, 0.5f, 0.5f,  // box 0
      0.0f, 0.1f, 0.5f, 0.6f,  // box 1 (overlaps box 0, suppressed)
      1.0f, 1.0f, 1.5f, 1.5f,  // box 2 (no overlap)
      1.0f, 1.1f, 1.5f, 1.6f,  // box 3 (overlaps box 2, suppressed)
  };
  std::vector<float> scores = {0.9f, 0.75f, 0.95f, 0.5f};

  NmsTestConfig cfg;
  cfg.boxes_data = boxes;
  cfg.boxes_shape = {1, 4, 4};
  cfg.scores_data = scores;
  cfg.scores_shape = {1, 1, 4};
  cfg.max_output_boxes = 2;
  cfg.iou_threshold = 0.5f;
  cfg.score_threshold = 0.0f;
  cfg.center_point_box = 0;
  cfg.output_shape = {2, 3};

  RunNmsTest(BuildQdqNmsTestCase<uint16_t>(cfg, /*use_ms_domain=*/true), 11,
             ExpectedEPNodeAssignment::All, HtpNmsProviderOptions());
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

// max_output_boxes_per_class == 0 must fall back.
// 0 is legal in ONNX (selects no boxes) but produces a degenerate [0, 3] QNN output
// that QNN rejects at compose. QNN EP must decline the node instead of hard-failing.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_Fallback_ZeroMaxBoxes) {
  CONDITIONAL_SKIP_TEST_ON_LINUX_ARM64(HtpNmsProviderOptions(), QNN_HTP_DEVICE_ARCH_V68, "QDQ", uint8_t);
  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("boxes", {1, 3, 4},
                             {0.0f, 0.0f, 1.0f, 1.0f,
                              0.0f, 5.0f, 1.0f, 6.0f,
                              0.0f, 10.0f, 1.0f, 11.0f});
    builder.MakeInput<float>("scores", {1, 1, 3}, {0.9f, 0.85f, 0.8f});
    builder.MakeInitializer<int64_t>("max_boxes", {}, {0});  // 0 → no boxes selected
    builder.AddNode("nms_node", "NonMaxSuppression",
                    {"boxes", "scores", "max_boxes"}, {"selected_indices"}, kOnnxDomain,
                    {test::MakeAttribute("center_point_box", int64_t{0})});
    builder.MakeOutput<int64_t>("selected_indices", std::vector<int64_t>{0, 3});
  };

  RunNmsTest(build_model, 11, ExpectedEPNodeAssignment::None);
}

// max_output_boxes_per_class absent (only boxes + scores) must fall back.
// Absent defaults to 0 in ONNX, which selects no boxes.
TEST_F(QnnCPUBackendTests, NonMaxSuppression_Fallback_AbsentMaxBoxes) {
  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("boxes", {1, 3, 4},
                             {0.0f, 0.0f, 1.0f, 1.0f,
                              0.0f, 5.0f, 1.0f, 6.0f,
                              0.0f, 10.0f, 1.0f, 11.0f});
    builder.MakeInput<float>("scores", {1, 1, 3}, {0.9f, 0.85f, 0.8f});
    builder.AddNode("nms_node", "NonMaxSuppression",
                    {"boxes", "scores"}, {"selected_indices"}, kOnnxDomain,
                    {test::MakeAttribute("center_point_box", int64_t{0})});
    builder.MakeOutput<int64_t>("selected_indices", std::vector<int64_t>{0, 3});
  };

  RunNmsTest(build_model, 11, ExpectedEPNodeAssignment::None);
}

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
