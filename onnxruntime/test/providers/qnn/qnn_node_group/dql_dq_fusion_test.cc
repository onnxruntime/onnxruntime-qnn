// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// ---------------------------------------------------------------------------
// Helper: JSON graph directory setup / teardown
// ---------------------------------------------------------------------------
void ResetQnnGraphDir(const std::filesystem::path& dir) {
  std::filesystem::remove_all(dir);
  ASSERT_TRUE(std::filesystem::create_directory(dir));
}

// Returns true if at least one QNN JSON graph file exists in `dump_dir`.
// Used to skip graph assertions when the test was not executed (e.g., FP32
// HTP not available and no JSON dump is produced).
bool HasQnnJsonGraph(const std::filesystem::path& dump_dir) {
  if (!std::filesystem::exists(dump_dir)) return false;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      return true;
    }
  }
  return false;
}

ProviderOptions GetHTPProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  // On x86-64 Linux the HTP emulator needs a concrete SoC model to run.
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return provider_options;
}

// ---------------------------------------------------------------------------
// Happy-path model builders
// ---------------------------------------------------------------------------

// Builds:
//   input (float32, shape) --> DynamicQuantizeLinear --> (y, y_scale, y_zp)
//   DequantizeLinear(y, y_scale, y_zp) --> dq_out (float32)
//   Relu(dq_out) --> output (float32)
//
// The fusion should replace DQL + DQ with an identity Transpose so the graph
// visible to QNN contains only Transpose + Relu.
GetTestModelFn BuildDqlDqReluTestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("dql_dq_relu_graph");
    MakeTestInput<float>(builder, "input", input_def);

    // DynamicQuantizeLinear: input -> (y, y_scale, y_zp)
    builder.AddNode("dql", "DynamicQuantizeLinear", {"input"},
                    {"dql_y", "dql_y_scale", "dql_y_zp"});

    // DequantizeLinear: (y, y_scale, y_zp) -> dq_out
    builder.AddNode("dq", "DequantizeLinear",
                    {"dql_y", "dql_y_scale", "dql_y_zp"}, {"dq_out"});

    // Relu: dq_out -> output
    builder.AddNode("relu", "Relu", {"dq_out"}, {"output"});
    builder.MakeOutput("output");
  };
}

// Builds the same DQL -> DQ pair but uses an Add instead of Relu so that we
// can also verify 2D / flat tensor shapes go through the fusion.
//
//   input (float32) --> DQL --> DQ --> Add(const) --> output (float32)
GetTestModelFn BuildDqlDqAddTestCase(const TestInputDef<float>& input_def,
                                     float addend = 0.1f) {
  return [input_def, addend](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("dql_dq_add_graph");
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("dql", "DynamicQuantizeLinear", {"input"},
                    {"dql_y", "dql_y_scale", "dql_y_zp"});

    builder.AddNode("dq", "DequantizeLinear",
                    {"dql_y", "dql_y_scale", "dql_y_zp"}, {"dq_out"});

    // Small constant addend so the downstream Add has a concrete initializer.
    const std::vector<int64_t> shape = input_def.GetShape();
    const size_t num_elems =
        static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                            std::multiplies<int64_t>{}));
    builder.MakeInitializer<float>("addend", shape,
                                   std::vector<float>(num_elems, addend));

    builder.AddNode("add", "Add", {"dq_out", "addend"}, {"output"});
    builder.MakeOutput("output");
  };
}

// Builds:
//   input (float32) --> DynamicQuantizeLinear --> (y, y_scale, y_zp)
//   DequantizeLinear(y, y_scale) --> dq_out (float32)   [y_zp not consumed]
//   Relu(dq_out) --> output (float32)
//
// DQL.y_zp has zero consumers: DQ omits zero_point and defaults to 0. Because
// DQL's y_zp is dynamic and typically non-zero (e.g. ~128 for inputs in [-1,1]),
// DQ(y, y_scale, 0) != x and the round-trip is NOT identity. Fusion must be
// rejected to avoid silently producing wrong results.
GetTestModelFn BuildDqlDqNoZpTestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("dql_dq_no_zp_graph");
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("dql", "DynamicQuantizeLinear", {"input"},
                    {"dql_y", "dql_y_scale", "dql_y_zp"});

    // DQ omits y_zp (zero_point is optional per ONNX spec).
    // DQL.y_zp has zero consumers after this.
    builder.AddNode("dq", "DequantizeLinear",
                    {"dql_y", "dql_y_scale"}, {"dq_out"});

    builder.AddNode("relu", "Relu", {"dq_out"}, {"output"});
    builder.MakeOutput("output");
  };
}

// ---------------------------------------------------------------------------
// No-fusion model builders
// ---------------------------------------------------------------------------

// Builds a model where DQL.y (output[0]) is consumed by BOTH a
// DequantizeLinear AND a direct Cast to float32.  This violates the
// "all DQL outputs exclusively consumed by DQ" requirement, so
// DqlDqFusion::TryFusion should return nullptr and the graph must fall
// through to the regular op builders (DQL is not supported on HTP in
// isolation, so at least some nodes land on CPU EP).
//
//   input --> DQL --> (y, y_scale, y_zp)
//                y ----> DequantizeLinear(y, y_scale, y_zp) --> dq_out
//                y ----> Cast (float32) --> cast_out
//             Relu(dq_out) --> relu_out
//             Add(cast_out, relu_out) --> output
GetTestModelFn BuildDqlDqNoFusionTestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("dql_dq_no_fusion_graph");
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("dql", "DynamicQuantizeLinear", {"input"},
                    {"dql_y", "dql_y_scale", "dql_y_zp"});

    // DequantizeLinear consumes all three DQL outputs.
    builder.AddNode("dq", "DequantizeLinear",
                    {"dql_y", "dql_y_scale", "dql_y_zp"}, {"dq_out"});

    // Extra consumer of dql_y: Cast uint8 -> float32.
    // This makes ConsumersAreAllOfType(dql_outs[0], DEQUANTIZE_LINEAR) return
    // false and the fusion should be rejected.
    builder.AddNode("cast_y", "Cast", {"dql_y"}, {"cast_y_out"}, kOnnxDomain,
                    {builder.MakeScalarAttribute(
                        "to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});

    builder.AddNode("relu", "Relu", {"dq_out"}, {"relu_out"});

    builder.AddNode("add", "Add", {"cast_y_out", "relu_out"}, {"output"});
    builder.MakeOutput("output");
  };
}

// ---------------------------------------------------------------------------
// Common runner: runs the model, verifies outputs, and optionally checks the
// QNN graph JSON.
// ---------------------------------------------------------------------------
struct FusionTestParams {
  std::filesystem::path json_dir;
  GetTestModelFn build_model;
  int opset_version = 13;
  ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All;
  float fp32_abs_err = 1e-2f;
  OrtLoggingLevel log_severity = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
  bool verify_outputs = true;
};

void RunFusionTest(const FusionTestParams& p) {
  ResetQnnGraphDir(p.json_dir);
  auto cleanup = gsl::finally([&p]() { std::filesystem::remove_all(p.json_dir); });

  ProviderOptions provider_options = GetHTPProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = p.json_dir.string();

  RunQnnModelTest(p.build_model,
                  provider_options,
                  p.opset_version,
                  EPVerificationParams{p.expected_ep_assignment, ElementwiseAbsoluteVerifier(p.fp32_abs_err)},
                  p.log_severity,
                  p.verify_outputs);
}

}  // namespace

// ==========================================================================
// Happy-path tests — fusion fires, QNN sees a Transpose instead of DQL+DQ
// ==========================================================================

// Basic 4-D input: DQL + DQ should be replaced by an identity Transpose; Relu
// is the downstream consumer that keeps the graph alive.
TEST_F(QnnHTPBackendTests, DqlDqFusion_4D_WithRelu) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::filesystem::path json_dir = "DqlDqFusion_4D_WithRelu";
  RunFusionTest({json_dir,
                 BuildDqlDqReluTestCase(
                     TestInputDef<float>({1, 4, 4, 8}, /*is_initializer=*/false,
                                         -1.0f, 1.0f))});

  if (!HasQnnJsonGraph(json_dir)) return;

  // The fused graph should contain exactly one Transpose (the identity
  // placeholder for DQL+DQ) and one Relu; DQL and DQ must be absent.
  AssertOpInQnnGraph(json_dir, "Transpose", /*count=*/1);
  AssertOpInQnnGraph(json_dir, "Relu", /*count=*/1);
  AssertOpInQnnGraph(json_dir, "DynamicQuantizeLinear", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "Dequantize", /*count=*/0);
}

// 2-D input — checks that the identity-permutation [0,1] is correctly emitted.
TEST_F(QnnHTPBackendTests, DqlDqFusion_2D_WithRelu) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::filesystem::path json_dir = "DqlDqFusion_2D_WithRelu";
  RunFusionTest({json_dir,
                 BuildDqlDqReluTestCase(
                     TestInputDef<float>({8, 16}, /*is_initializer=*/false,
                                         -1.0f, 1.0f))});

  if (!HasQnnJsonGraph(json_dir)) return;

  AssertOpInQnnGraph(json_dir, "Transpose", /*count=*/1);
  AssertOpInQnnGraph(json_dir, "Relu", /*count=*/1);
  AssertOpInQnnGraph(json_dir, "DynamicQuantizeLinear", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "Dequantize", /*count=*/0);
}

// 3-D input with Add downstream — exercises the case where the fused Transpose
// output feeds an Add with a constant initializer.
TEST_F(QnnHTPBackendTests, DqlDqFusion_3D_WithAdd) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::filesystem::path json_dir = "DqlDqFusion_3D_WithAdd";
  RunFusionTest({json_dir,
                 BuildDqlDqAddTestCase(
                     TestInputDef<float>({1, 4, 8}, /*is_initializer=*/false,
                                         -1.0f, 1.0f),
                     /*addend=*/0.05f),
                 /*opset_version=*/13,
                 ExpectedEPNodeAssignment::All,
                 /*fp32_abs_err=*/1e-2f});

  if (!HasQnnJsonGraph(json_dir)) return;

  AssertOpInQnnGraph(json_dir, "Transpose", /*count=*/1);
  AssertOpInQnnGraph(json_dir, "DynamicQuantizeLinear", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "Dequantize", /*count=*/0);
}

// Larger activation tensor similar to a typical transformer hidden-state shape.
// Primarily a correctness check: output must be close to the CPU EP reference.
// The DQL→DQ round-trip is a fake-quantize, so the approximation error is
// bounded by the quantization step (|x_approx - x| <= scale/2), which for
// inputs in [-1, 1] is at most ~0.004 with uint8.  We use 1e-2 as a safe
// upper bound.
TEST_F(QnnHTPBackendTests, DqlDqFusion_LargeActivation_Correctness) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::filesystem::path json_dir = "DqlDqFusion_LargeActivation";
  RunFusionTest({json_dir,
                 BuildDqlDqReluTestCase(
                     TestInputDef<float>({1, 128, 768}, /*is_initializer=*/false,
                                         -1.5f, 1.5f)),
                 /*opset_version=*/13,
                 ExpectedEPNodeAssignment::All,
                 /*fp32_abs_err=*/1e-2f});

  if (!HasQnnJsonGraph(json_dir)) return;

  AssertOpInQnnGraph(json_dir, "Transpose", /*count=*/1);
  AssertOpInQnnGraph(json_dir, "Relu", /*count=*/1);
  AssertOpInQnnGraph(json_dir, "DynamicQuantizeLinear", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "Dequantize", /*count=*/0);
}

// DQL.y_zp has zero consumers (DQ omits zero_point, which is optional per ONNX
// spec). DQL's y_zp is dynamic and typically non-zero, so DQ(y, y_scale, 0) != x:
// the round-trip is NOT identity and fusion must be rejected.
TEST_F(QnnHTPBackendTests, DqlDqFusion_NoZeroPoint_NoFusion) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::filesystem::path json_dir = "DqlDqFusion_NoZeroPoint_NoFusion";

  // DQL has no standalone HTP op-builder, so at least some nodes must land on
  // CPU EP when fusion does not fire.
  RunFusionTest({json_dir,
                 BuildDqlDqNoZpTestCase(
                     TestInputDef<float>({1, 4, 4, 8}, /*is_initializer=*/false,
                                         -1.0f, 1.0f)),
                 /*opset_version=*/13,
                 /*expected_ep_assignment=*/ExpectedEPNodeAssignment::Some,
                 /*fp32_abs_err=*/1e-2f,
                 /*log_severity=*/OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                 /*verify_outputs=*/false});  // graph is split across CPU/QNN EP

  if (!HasQnnJsonGraph(json_dir)) return;

  // The identity Transpose emitted by DqlDqFusion must be absent.
  AssertOpInQnnGraph(json_dir, "Transpose", /*count=*/0);
}

// ==========================================================================
// No-fusion tests — guard conditions not met; graph must not crash
// ==========================================================================

// DQL.y is consumed by both DequantizeLinear AND a Cast node, which breaks the
// "all DQL outputs exclusively consumed by DQ" guard.
//
// Primary no-fusion guard: ExpectedEPNodeAssignment::Some.
//   If fusion incorrectly fired, DQL+DQ would become a Transpose on QNN EP and
//   the downstream Cast/Relu/Add would also land on QNN EP, yielding an "All"
//   assignment.  Asserting "Some" verifies that at least one node stayed on CPU
//   EP — which can only happen when fusion did NOT fire.
//
// Secondary graph-content guard: Transpose count == 0.
//   DqlDqFusion emits an identity-permutation Transpose as a placeholder.
//   All ops in this graph are elementwise, so the layout optimizer will not
//   insert its own Transpose; asserting count == 0 is therefore safe and
//   directly checks that the fusion product is absent.
TEST_F(QnnHTPBackendTests, DqlDqFusion_ExtraConsumer_NoFusion) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::filesystem::path json_dir = "DqlDqFusion_ExtraConsumer_NoFusion";

  // Primary guard: DQL has no standalone QNN HTP op-builder, so at least some
  // nodes must land on CPU EP when fusion does not fire.
  RunFusionTest({json_dir,
                 BuildDqlDqNoFusionTestCase(
                     TestInputDef<float>({1, 4, 4, 8}, /*is_initializer=*/false, -1.0f, 1.0f)),
                 /*opset_version=*/13,
                 /*expected_ep_assignment=*/ExpectedEPNodeAssignment::Some,
                 /*fp32_abs_err=*/1e-2f,
                 /*log_severity=*/OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                 /*verify_outputs=*/false});  // graph is split across CPU/QNN EP; output comparison is not meaningful

  if (!HasQnnJsonGraph(json_dir)) return;

  // Secondary guard: the identity Transpose emitted by DqlDqFusion must be absent.
  AssertOpInQnnGraph(json_dir, "Transpose", /*count=*/0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
