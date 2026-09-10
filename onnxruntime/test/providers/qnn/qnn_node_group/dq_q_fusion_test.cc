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

// Builds a graph where a standalone (DQ -> Q) sequence at the graph's output
// is fused into a QNN Convert operator.
//   ONNX Graph: DQ -> Add -> Q -> DQ -> Q -> graph_output
//   QNN  Graph: DQ -> Add -> Q -> Convert -> graph_output
//
// The trailing DQ (OutQuantType-from-input) feeds a lone Q (to OutQuantType).
// DQQFusion requires the DQ to be a standalone SingleNode whose only child is
// a Q with the same scale type; that pair is replaced by a single Convert.
// `add_out_qparams.scale` is nudged so the DQ->Q pair is NOT folded away by
// the ORT QDQ optimizer before the QNN EP sees the graph.
template <typename InQuantType, typename OutQuantType>
GetTestModelFn BuildDQQConvertAtOutputTestCase(const TestInputDef<float>& input0_def,
                                               const TestInputDef<float>& input1_def,
                                               const QuantParams<OutQuantType>& output_qparams) {
  return [input0_def, input1_def, output_qparams](ModelTestBuilder& builder) {
    builder.graph_->set_name("dq_q_convert_at_output_graph");
    MakeTestInput<float>(builder, "input0", input0_def);
    MakeTestInput<float>(builder, "input1", input1_def);

    // Input0 -> Q(InQuantType) -> DQ -> input0_after_qdq
    const QuantParams<InQuantType> input0_qparams = GetTestInputQuantParams<InQuantType>(input0_def);
    const std::string input0_after_qdq =
        AddQDQNodePair<InQuantType>(builder, "qdq0", "input0", input0_qparams.scale, input0_qparams.zero_point);

    // Input1 -> Q(InQuantType) -> DQ -> input1_after_qdq
    const QuantParams<InQuantType> input1_qparams = GetTestInputQuantParams<InQuantType>(input1_def);
    const std::string input1_after_qdq =
        AddQDQNodePair<InQuantType>(builder, "qdq1", "input1", input1_qparams.scale, input1_qparams.zero_point);

    // Add -> add_out
    builder.AddNode("Add", "Add", {input0_after_qdq, input1_after_qdq}, {"add_out"});

    // add_out -> Q(InQuantType) -> DQ -> add_qdq_name
    QuantParams<InQuantType> add_out_qparams = ConvertQuantParams<OutQuantType, InQuantType>(output_qparams);
    add_out_qparams.scale *= 1.01f;  // Make qparams slightly different so DQ->Q are not optimized out.
    const std::string add_qdq_name =
        AddQDQNodePair(builder, "add_qdq", "add_out", add_out_qparams.scale, add_out_qparams.zero_point);

    // add_qdq_name -> Q(OutQuantType) -> graph output.
    // The preceding DQ and this Q are fused into a QNN Convert.
    builder.MakeOutput("Y");
    builder.AddQuantizeLinearNode<OutQuantType>("final_q", add_qdq_name,
                                                output_qparams.scale, output_qparams.zero_point, "Y");
  };
}

// Builds a graph where the DQ feeding the final Q has a SECOND consumer, so it
// is not a lone DQ->Q pair and DQQFusion must be rejected.
//   ONNX Graph:           input0 -> Q -> DQ ---> Add -> Q(out0)
//                                            \--> Q -> graph output (out1)
//   input1 -> Q -> DQ ----------------------/
//
// The shared DQ has two consumers, so GetOnlyChildOfType returns nullptr and
// no Convert is produced.
template <typename QuantType>
GetTestModelFn BuildDQQNoFusionExtraConsumerTestCase(const TestInputDef<float>& input0_def,
                                                     const TestInputDef<float>& input1_def) {
  return [input0_def, input1_def](ModelTestBuilder& builder) {
    builder.graph_->set_name("dq_q_no_fusion_extra_consumer_graph");
    MakeTestInput<float>(builder, "input0", input0_def);
    MakeTestInput<float>(builder, "input1", input1_def);

    const QuantParams<QuantType> input0_qparams = GetTestInputQuantParams<QuantType>(input0_def);
    // input0 -> Q -> DQ -> input0_after_qdq  (this DQ will gain a second consumer)
    const std::string input0_after_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq0", "input0", input0_qparams.scale, input0_qparams.zero_point);

    const QuantParams<QuantType> input1_qparams = GetTestInputQuantParams<QuantType>(input1_def);
    const std::string input1_after_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq1", "input1", input1_qparams.scale, input1_qparams.zero_point);

    // First consumer of input0's DQ: Add.
    builder.AddNode("Add", "Add", {input0_after_qdq, input1_after_qdq}, {"add_out"});
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "add_qdq", "add_out",
                                                     input0_qparams.scale, input0_qparams.zero_point);

    // Second consumer of input0's DQ: a lone Q to the graph output. Because the
    // DQ has two consumers, the DQ->Q pair is not eligible for fusion.
    builder.MakeOutput("Y");
    builder.AddQuantizeLinearNode<QuantType>("extra_q", input0_after_qdq,
                                             input0_qparams.scale, input0_qparams.zero_point, "Y");
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
// Happy-path tests — DQQFusion fires, QNN sees a Convert instead of DQ+Q
// ==========================================================================

// uint8 -> uint8 requantize at the graph output. The lone DQ->Q must collapse
// into a single QNN Convert; no standalone Quantize must remain for that pair.
TEST_F(QnnHTPBackendTests, DQQFusion_U8_Convert) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  std::vector<float> input0_data = GetFloatDataInRange(-8.0f, 8.0f, 8);
  std::vector<float> input1_data = GetFloatDataInRange(-8.0f, 8.0f, 8);
  TestInputDef<float> input0_def({1, 2, 2, 2}, false, input0_data);
  TestInputDef<float> input1_def({1, 2, 2, 2}, false, input1_data);
  QuantParams<uint8_t> out_qparams_u8 = {1.0f, 128};

  const std::filesystem::path json_dir = "DQQFusion_U8_Convert";
  RunFusionTest({json_dir,
                 BuildDQQConvertAtOutputTestCase<uint8_t, uint8_t>(input0_def, input1_def, out_qparams_u8)});

  if (!HasQnnJsonGraph(json_dir)) return;

  // The fused DQ->Q must appear as exactly one Convert.
  AssertOpInQnnGraph(json_dir, "Convert", /*count=*/1);
}

// uint16 -> uint16 requantize at the graph output.
TEST_F(QnnHTPBackendTests, DQQFusion_U16_Convert) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  std::vector<float> input0_data = GetFloatDataInRange(-8.0f, 8.0f, 8);
  std::vector<float> input1_data = GetFloatDataInRange(-8.0f, 8.0f, 8);
  TestInputDef<float> input0_def({1, 2, 2, 2}, false, input0_data);
  TestInputDef<float> input1_def({1, 2, 2, 2}, false, input1_data);
  QuantParams<uint16_t> out_qparams_u16 = {1.0f, 32768};

  const std::filesystem::path json_dir = "DQQFusion_U16_Convert";
  RunFusionTest({json_dir,
                 BuildDQQConvertAtOutputTestCase<uint16_t, uint16_t>(input0_def, input1_def, out_qparams_u16)});

  if (!HasQnnJsonGraph(json_dir)) return;

  AssertOpInQnnGraph(json_dir, "Convert", /*count=*/1);
}

// ==========================================================================
// No-fusion test — guard condition not met; no Convert produced
// ==========================================================================

// The DQ feeding the final Q has a second consumer (the Add branch), so the
// DQ->Q pair is not a lone sequence and DQQFusion must be rejected. No Convert
// must appear in the compiled graph.
TEST_F(QnnHTPBackendTests, DQQFusion_ExtraConsumer_NoFusion) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  std::vector<float> input0_data = GetFloatDataInRange(-8.0f, 8.0f, 8);
  std::vector<float> input1_data = GetFloatDataInRange(-8.0f, 8.0f, 8);
  TestInputDef<float> input0_def({1, 2, 2, 2}, false, input0_data);
  TestInputDef<float> input1_def({1, 2, 2, 2}, false, input1_data);

  const std::filesystem::path json_dir = "DQQFusion_ExtraConsumer_NoFusion";
  RunFusionTest({json_dir,
                 BuildDQQNoFusionExtraConsumerTestCase<uint8_t>(input0_def, input1_def)});

  if (!HasQnnJsonGraph(json_dir)) return;

  // The shared DQ disqualifies the lone-DQ->Q pattern; no Convert is produced.
  AssertOpInQnnGraph(json_dir, "Convert", /*count=*/0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
