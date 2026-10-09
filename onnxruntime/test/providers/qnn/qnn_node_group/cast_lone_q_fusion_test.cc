// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
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
// Model builders
// ---------------------------------------------------------------------------

// Builds a graph where a standalone Cast is directly followed by a Q. The Cast
// has a non-DQ parent (a uint8 graph input) and its only child is the Q, so
// CastLoneQFusion replaces (Cast -> Q) with a single QNN Convert.
//
//   cast_input(uint8) -> Cast(->float) -> Q(uint8) -> DQ ----.
//                                                            Add -> Q -> DQ -> output
//   input2(float) ----------------------> Q(uint8) -> DQ ----'
//
// The fused (Cast -> Q) shows up as one Convert; no standalone Cast remains.
GetTestModelFn BuildCastLoneQFusionTestCase(const TestInputDef<uint8_t>& cast_input_def,
                                            const TestInputDef<float>& input2_def) {
  return [cast_input_def, input2_def](ModelTestBuilder& builder) {
    builder.graph_->set_name("cast_lone_q_fusion_graph");
    MakeTestInput<uint8_t>(builder, "cast_input", cast_input_def);
    MakeTestInput<float>(builder, "input2", input2_def);

    // cast_input(uint8) -> Cast(float) -> cast_out
    builder.AddNode("cast", "Cast", {"cast_input"}, {"cast_out"}, kOnnxDomain,
                    {builder.MakeScalarAttribute(
                        "to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});

    // cast_out -> Q(uint8) -> DQ -> cast_after_qdq.
    // The Q's parent is the Cast (not a DQ), so Cast + Q fuse into a Convert.
    const std::string cast_after_qdq =
        AddQDQNodePair<uint8_t>(builder, "cast_qdq", "cast_out", /*scale=*/1.0f, /*zp=*/0);

    // input2(float) -> Q(uint8) -> DQ -> input2_after_qdq.
    const QuantParams<uint8_t> input2_qparams = GetTestInputQuantParams<uint8_t>(input2_def);
    const std::string input2_after_qdq =
        AddQDQNodePair<uint8_t>(builder, "in2_qdq", "input2", input2_qparams.scale, input2_qparams.zero_point);

    // Add -> Q -> DQ -> graph output.
    builder.AddNode("add", "Add", {cast_after_qdq, input2_after_qdq}, {"add_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "out_qdq", "add_out",
                                                   input2_qparams.scale, input2_qparams.zero_point);
  };
}

// Builds the same Cast but routes it straight into an Add (no Q child), so
// CastLoneQFusion is not eligible (quantize_linear == nullptr) and the Cast
// stays a standalone QNN Cast. No Convert must be produced.
//
//   cast_input(uint8) -> Cast(->float) -> Add(input2) -> output
GetTestModelFn BuildCastNoFusionTestCase(const TestInputDef<uint8_t>& cast_input_def,
                                         const TestInputDef<float>& input2_def) {
  return [cast_input_def, input2_def](ModelTestBuilder& builder) {
    builder.graph_->set_name("cast_no_fusion_graph");
    MakeTestInput<uint8_t>(builder, "cast_input", cast_input_def);
    MakeTestInput<float>(builder, "input2", input2_def);

    builder.AddNode("cast", "Cast", {"cast_input"}, {"cast_out"}, kOnnxDomain,
                    {builder.MakeScalarAttribute(
                        "to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});

    builder.AddNode("add", "Add", {"cast_out", "input2"}, {"output"});
    builder.MakeOutput("output");
  };
}

// ---------------------------------------------------------------------------
// Common runner
// ---------------------------------------------------------------------------
struct FusionTestParams {
  std::filesystem::path json_dir;
  GetTestModelFn build_model;
  int opset_version = 21;
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
                  p.expected_ep_assignment,
                  p.fp32_abs_err,
                  p.log_severity,
                  p.verify_outputs);
}

}  // namespace

// ==========================================================================
// Happy-path test — CastLoneQFusion fires, QNN sees a Convert instead of Cast+Q
// ==========================================================================

// uint8 Cast input feeding a Q: (Cast -> Q) must collapse into one Convert and
// the standalone Cast must be absent from the compiled graph.
TEST_F(QnnHTPBackendTests, CastLoneQFusion_U8_Convert) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestInputDef<uint8_t> cast_input_def({1, 2, 2, 2}, false, {0, 1, 2, 3, 4, 5, 6, 7});
  TestInputDef<float> input2_def({1, 2, 2, 2}, false, GetFloatDataInRange(-4.0f, 4.0f, 8));

  const std::filesystem::path json_dir = "CastLoneQFusion_U8_Convert";
  RunFusionTest({json_dir,
                 BuildCastLoneQFusionTestCase(cast_input_def, input2_def),
                 /*opset_version=*/21,
                 ExpectedEPNodeAssignment::All,
                 /*fp32_abs_err=*/1e-2f,
                 /*log_severity=*/OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                 /*verify_outputs=*/false});  // integer Cast input; structure is what we assert

  if (!HasQnnJsonGraph(json_dir)) return;

  // The fused (Cast -> Q) appears as exactly one Convert; no standalone Cast remains.
  AssertOpInQnnGraph(json_dir, "Convert", /*count=*/1);
  AssertOpInQnnGraph(json_dir, "Cast", /*count=*/0);
}

// ==========================================================================
// No-fusion test — Cast has no Q child; it stays a standalone Cast
// ==========================================================================

// The Cast feeds an Add (not a Q), so CastLoneQFusion is not eligible. No
// Convert must be produced and the Cast must survive as a standalone op.
TEST_F(QnnHTPBackendTests, CastLoneQFusion_NoQChild_NoFusion) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestInputDef<uint8_t> cast_input_def({1, 2, 2, 2}, false, {0, 1, 2, 3, 4, 5, 6, 7});
  TestInputDef<float> input2_def({1, 2, 2, 2}, false, GetFloatDataInRange(-4.0f, 4.0f, 8));

  const std::filesystem::path json_dir = "CastLoneQFusion_NoQChild_NoFusion";
  RunFusionTest({json_dir,
                 BuildCastNoFusionTestCase(cast_input_def, input2_def),
                 /*opset_version=*/21,
                 ExpectedEPNodeAssignment::All,
                 /*fp32_abs_err=*/1e-2f,
                 /*log_severity=*/OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                 /*verify_outputs=*/false});  // integer Cast input; structure is what we assert

  if (!HasQnnJsonGraph(json_dir)) return;

  // No Q child -> no fusion. The Cast survives; no Convert is produced.
  AssertOpInQnnGraph(json_dir, "Convert", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "Cast", /*count=*/1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
