// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <gsl/gsl>
#include <optional>
#include <string>
#include <filesystem>
#include <variant>

#include "onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a non-QDQ model on the QNN CPU backend and compares output to CPU EP.
template <typename InputType = float>
static void RunOpTestOnCPU(const std::string& op_type,
                           const std::vector<TestInputDef<InputType>>& input_defs,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           const std::string& op_domain = kOnnxDomain) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<InputType>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain),
                  provider_options,
                  opset_version,
                  EPVerificationParams{expected_ep_assignment});
}

// Test f32 Relu on the CPU backend.
// TODO: When this is fixed, enable ActivationOpTest.Relu test in cpu/activation/activation_op_test tests.
// Disabled because QNN SDK 2.17 Relu treats inf as FLT_MAX.
// Log: the value pair (inf, 3.40282347e+38) at index #12 don't match
TEST_F(QnnCPUBackendTests, DISABLED_UnaryOp_Relu) {
  std::vector<float> input_data{-1.0f, 0, 1.0f,
                                100.0f, -100.0f, 1000.0f, -1000.0f,
                                FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,
                                FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity()};
  RunOpTestOnCPU("Relu",
                 {TestInputDef<float>({13}, false, input_data)},
                 {},
                 14,
                 ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, UnaryOp_Softplus) {
  RunOpTestOnCPU("Softplus",
                 {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                 {},
                 14,
                 ExpectedEPNodeAssignment::All);
}

// Rank > 4D is supported on CPU (no HTP rank constraint).
TEST_F(QnnCPUBackendTests, UnaryOp_Softplus_Rank5) {
  RunOpTestOnCPU("Softplus",
                 {TestInputDef<float>({1, 2, 3, 4, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 120))},
                 {},
                 14,
                 ExpectedEPNodeAssignment::All);
}

// Verifies QNN_OP_HARD_SWISH computes x * clip((x+3)/6, 0, 1), not HardSigmoid.
TEST_F(QnnCPUBackendTests, UnaryOp_HardSwish) {
  RunOpTestOnCPU("HardSwish",
                 {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                 {},
                 14,
                 ExpectedEPNodeAssignment::All);
}

// Test float HardSwish on the QNN HTP backend.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSwish_FP32) {
  QNN_SKIP_TEST_ON_ARM64("Fails on ARM64/AARCH64");
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  RunQnnModelTest(BuildOpTestCase<float>("HardSwish_node", "HardSwish",
                                         {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                                         {}, {}),
                  provider_options,
                  14,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.004f)});
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Tests the accuracy of a QDQ model on QNN EP by comparing to CPU EP, which runs both the fp32 model
// and the QDQ model.
template <typename InputQType = uint8_t>
static void RunQDQOpTest(const std::string& op_type,
                         const std::vector<TestInputDef<float>>& input_defs,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                         int opset_version,
                         ExpectedEPNodeAssignment expected_ep_assignment,
                         const std::string& op_domain = kOnnxDomain,
                         bool use_contrib_qdq = false,
                         QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain),
                       BuildQDQOpTestCase<InputQType>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain, use_contrib_qdq),
                       provider_options,
                       opset_version,
                       expected_ep_assignment,
                       tolerance);
}
// Tests the accuracy of a QDQ model with indices inputs on QNN EP by comparing to CPU EP, which runs
// both the fp32 model and the QDQ model.
template <typename InputQType = uint8_t, typename InputType2 = int64_t>
static void RunQDQOpTest(const std::string& op_type,
                         const std::vector<TestInputDef<float>>& input_defs_1,
                         const std::vector<TestInputDef<InputType2>>& input_defs_2,
                         const std::vector<TestInputDef<float>>& input_defs_3,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                         int opset_version,
                         ExpectedEPNodeAssignment expected_ep_assignment,
                         const std::string& op_domain = kOnnxDomain,
                         bool use_contrib_qdq = false,
                         QDQTolerance tolerance = QDQTolerance(),
                         bool combine_quant_inputs_qparams = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float, InputType2>(op_type + "_node", op_type, input_defs_1, input_defs_2, input_defs_3, attrs, op_domain),
                       BuildQDQOpTestCase<InputQType, InputType2>(op_type + "_node", op_type, input_defs_1, input_defs_2, input_defs_3, attrs,
                                                                  op_domain, use_contrib_qdq, combine_quant_inputs_qparams),
                       provider_options,
                       opset_version,
                       expected_ep_assignment,
                       tolerance);
}

// Runs a non-QDQ model on HTP and compares output to CPU EP.
template <typename InputType = float>
static void RunOpTest(const std::string& op_type,
                      const std::vector<TestInputDef<InputType>>& input_defs,
                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                      int opset_version,
                      ExpectedEPNodeAssignment expected_ep_assignment,
                      const std::string& op_domain = kOnnxDomain,
                      float fp32_abs_err = 1e-5f,
                      bool enable_htp_fp16_precision = false,
                      [[maybe_unused]] std::optional<std::string> soc_model = std::nullopt) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  if (enable_htp_fp16_precision) {
#if defined(_WIN32)
    SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
#if defined(__linux__) && !defined(__aarch64__)
    provider_options["soc_model"] = soc_model.has_value() ? *soc_model : std::to_string(QNN_SOC_MODEL_SM8850);
#endif
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildOpTestCase<InputType>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain),
                  provider_options,
                  opset_version,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(fp32_abs_err)});
}

// Runs an FP16 model on the QNN HTP backend and compares QNN EP's accuracy to CPU EP.
static void RunFP16OpTest(const std::string& op_type,
                          const std::vector<TestInputDef<float>>& input_defs,
                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                          int opset_version,
                          ExpectedEPNodeAssignment expected_ep_assignment,
                          const std::string& op_domain = kOnnxDomain,
                          float tolerance = 0.004f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  std::vector<TestInputDef<Ort::Float16_t>> input_fp16_defs;
  input_fp16_defs.reserve(input_defs.size());

  for (size_t i = 0; i < input_defs.size(); i++) {
    input_fp16_defs.push_back(ConvertToFP16InputDef(input_defs[i]));
  }

  auto model_fp32_fn = BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain);
  auto model_fp16_fn = BuildOpTestCase<Ort::Float16_t>(op_type + "_node", op_type, input_fp16_defs, {}, attrs, op_domain);

  TestFp16ModelAccuracy(model_fp32_fn,
                        model_fp16_fn,
                        provider_options,
                        opset_version,
                        expected_ep_assignment,
                        tolerance);
}

// Test the accuracy of QDQ Sigmoid.
TEST_F(QnnHTPBackendTests, UnaryOp_Sigmoid) {
  RunQDQOpTest<uint8_t>("Sigmoid",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ Sigmoid.
TEST_F(QnnHTPBackendTests, UnaryOp_Sigmoid_U16) {
  RunQDQOpTest<uint16_t>("Sigmoid",
                         {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                         {},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use MS domain Q/DQ ops
}

// Test the accuracy of QDQ Tanh.
TEST_F(QnnHTPBackendTests, UnaryOp_Tanh) {
  RunQDQOpTest<uint8_t>("Tanh",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// disabled for QNN 2.28.0.241029 backendValidateOpConfig failed
// still fails on QNN 2.28.2 and QNN 2.30.0
// QnnDsp <E> [4294967295] has incorrect Value -32768, expected equal to 0.
// QnnDsp <V> validateNativeOps node_token_6:qti.aisw:Tanh htp op validator failed 3110
// QnnDsp <V> registered validator failed => 3110
// QnnDsp <E> QnnBackend_validateOpConfig failed 3110
// QnnDsp <V> Wake up free backend (id: 1)'s thread(s)
// QnnDsp <E> Failed to validate op node_token_6 with error 0xc26
// Tests accuracy of 16-bit QDQ Tanh.
//
// We now skip QNN validation as a workaround for QNN SDK 2.28.0 to 2.30.0
TEST_F(QnnHTPBackendTests, UnaryOp_Tanh_U16) {
  RunQDQOpTest<uint16_t>("Tanh",
                         {TestInputDef<float>({1, 2, 64}, false, GetFloatDataInRange(-10.0f, 10.0f, 128))},
                         {},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use MS domain Q/DQ ops
}

// Check that QNN compiles DQ -> Gelu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Gelu) {
  RunQDQOpTest<uint8_t>("Gelu",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All,
                        kMSDomain);  // GeLu is a contrib op.
}

// Tests accuracy of 16-bit QDQ GeLu.
// TODO(adrianlizarraga): Inaccuracy detected for output 'output', element 5.
// Output quant params: scale=0.00015259021893143654, zero_point=0.
// Expected val: 10
// QNN QDQ val: 9.997406005859375 (err 0.002593994140625)
// CPU QDQ val: 9.999847412109375 (err 0.000152587890625)
TEST_F(QnnHTPBackendTests, UnaryOp_Gelu_U16) {
  const std::vector<float> input_data = {-10.0f, -8.4f, 0.0f, 4.3f, 7.1f, 10.0f};
  RunQDQOpTest<uint16_t>("Gelu",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         11,
                         ExpectedEPNodeAssignment::All,
                         kMSDomain,  // GeLu is a contrib op.
                         true);      // Use MS domain Q/DQ ops.
}

// Check that QNN compiles DQ -> Elu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Elu) {
  RunQDQOpTest<uint8_t>("Elu",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ Elu.
// TODO(adrianlizarraga): Re-enable. This works on QNN SDK 2.14.1!
// Inaccuracy detected for output 'output', element 1.
// Output quant params: scale=0.00011093531065853313, zero_point=8992.
// Expected val: -0.99751651287078857
// QNN QDQ val: 6.2726154327392578 (err 7.2701320648193359)
// CPU QDQ val: -0.99753034114837646 (err 1.3828277587890625e-05)
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, UnaryOp_Elu_U16) {
  RunQDQOpTest<uint16_t>("Elu",
                         {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                         {},
                         11,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

// Tests accuracy of QDQ Relu
// TODO: Relu does not set negative values to zero!
// Could be due to ORT's ReluQuantFusion!
//
// Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.039215687662363052, zero_point=0.
// Expected val: 0
// QNN QDQ val: -10 (err 10)
// CPU QDQ val: 0 (err 0)
TEST_F(QnnHTPBackendTests, UnaryOp_Relu) {
  RunQDQOpTest<uint8_t>("Relu",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        14,
                        ExpectedEPNodeAssignment::All);
}

// Returns true if at least one QNN JSON graph file exists in `dump_dir`. Used to skip graph
// assertions when the test was not executed (e.g., FP32 HTP unavailable on this architecture and
// no JSON dump is produced).
static bool HasQnnJsonGraph(const std::filesystem::path& dump_dir) {
  if (!std::filesystem::exists(dump_dir)) return false;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      return true;
    }
  }
  return false;
}

// Builds a uint8 QDQ model (Q -> <op_type> -> DQ), dumps the composed QNN graph, and asserts the
// op is emitted as the unified "ElementWiseNeuron" op (and that the op's old dedicated QNN op name
// is absent). Uses QDQ rather than a float model so the test runs on the x86 HTP emulator, where
// float ElementWiseNeuron ops are unsupported as standalone graph outputs.
static void RunNeuronOpTypeTest(const std::filesystem::path& json_qnn_graph_dir,
                                const std::string& op_type,
                                const std::string& legacy_qnn_op_name) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup =
      gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  std::vector<TestInputDef<float>> input_defs = {
      TestInputDef<float>({1, 2, 2, 2}, /*is_initializer=*/false, -1.0f, 1.0f)};
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, input_defs, {}, {}),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  if (!HasQnnJsonGraph(json_qnn_graph_dir)) {
    return;
  }

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron", /*count=*/1);
  AssertOpInQnnGraph(json_qnn_graph_dir, legacy_qnn_op_name, /*count=*/0);
}

// Standalone Relu/Sigmoid/Tanh/Elu now map to QNN_OP_ELEMENT_WISE_NEURON. Assert the emitted op
// type rather than just accuracy.
TEST_F(QnnHTPBackendTests, NeuronOpType_Relu) {
  RunNeuronOpTypeTest("NeuronOpType_Relu", "Relu", "Relu");
}

TEST_F(QnnHTPBackendTests, NeuronOpType_Sigmoid) {
  RunNeuronOpTypeTest("NeuronOpType_Sigmoid", "Sigmoid", "Sigmoid");
}

TEST_F(QnnHTPBackendTests, NeuronOpType_Tanh) {
  RunNeuronOpTypeTest("NeuronOpType_Tanh", "Tanh", "Tanh");
}

TEST_F(QnnHTPBackendTests, NeuronOpType_Elu) {
  RunNeuronOpTypeTest("NeuronOpType_Elu", "Elu", "Elu");
}

TEST_F(QnnHTPBackendTests, UnaryOp_Softplus_U8) {
  RunQDQOpTest<uint8_t>("Softplus",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        14,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_Softplus_U16) {
  RunQDQOpTest<uint16_t>("Softplus",
                         {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                         {},
                         14,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

// Check that QNN compiles DQ -> HardSwish -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSwish) {
  RunQDQOpTest<uint8_t>("HardSwish",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        14,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ HardSwish
TEST_F(QnnHTPBackendTests, UnaryOp_HardSwish_U16) {
  const std::vector<float> input_data = {-10.0f, -8.4f, 0.0f, 4.3f, 7.1f, 10.0f};
  RunQDQOpTest<uint16_t>("HardSwish",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         14,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_QU8) {
  RunQDQOpTest<uint8_t>("HardSigmoid",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {test::MakeAttribute("alpha", 0.1f),
                         test::MakeAttribute("beta", 0.4f)},
                        21,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_QU16) {
  RunQDQOpTest<uint16_t>("HardSigmoid",
                         {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                         {},
                         21,
                         ExpectedEPNodeAssignment::All);
}

// Test that QDQ uint16 HardSigmoid -> Mul produces correct output
// Reproduces MobileNetV3 SE block accuracy bug where HardSigmoid output scale must be
// overridden to 1/65536 for HTP to compute Mul correctly
TEST_F(QnnHTPBackendTests, HardSigmoidMul_QU16_ScaleOverride) {
  // Build: input -> HardSigmoid -> Mul(input, hardsigmoid_output) -> output
  // This mimics SE attention: x * sigmoid(fc(x))
  auto input_def = TestInputDef<float>({1, 16, 1, 1}, false, GetFloatDataInRange(-3.0f, 3.0f, 16));

  auto build_f32_model = [input_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    builder.AddNode("HardSigmoid", "HardSigmoid", {"input"}, {"hsig_out"});
    builder.AddNode("Mul", "Mul", {"input", "hsig_out"}, {"output"});
    builder.MakeOutput("output");
  };

  auto build_qdq_model = [input_def](ModelTestBuilder& builder,
                                     std::vector<QuantParams<uint16_t>>& output_qparams) {
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<uint16_t> input_qparams = GetTestInputQuantParams<uint16_t>(input_def);

    std::string input_dq = AddQDQNodePair<uint16_t>(builder, "input_qdq", "input",
                                                    input_qparams.scale, input_qparams.zero_point, true);
    builder.AddNode("HardSigmoid", "HardSigmoid", {input_dq}, {"hsig_out"});
    std::string hsig_dq = AddQDQNodePair<uint16_t>(builder, "hsig_qdq", "hsig_out",
                                                   1.0f / 65536.0f, static_cast<uint16_t>(0), true);
    builder.AddNode("Mul", "Mul", {input_dq, hsig_dq}, {"mul_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "output_qdq", "mul_out",
                                                    output_qparams[0].scale, output_qparams[0].zero_point, true);
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  GetTestQDQModelFn<uint16_t> qdq_fn = build_qdq_model;
  TestQDQModelAccuracy(build_f32_model, qdq_fn, provider_options, 21,
                       ExpectedEPNodeAssignment::All, QDQTolerance());
}

// Test that QDQ HardSigmoid is supported by QNN EP.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_QDQ_Supported) {
  RunQDQOpTest<uint8_t>("HardSigmoid",
                        {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f)},
                        {},
                        19,
                        ExpectedEPNodeAssignment::All);
}

// Check that QNN EP can support float32 HardSigmoid on HTP.
// Enables running f32 ops using fp16 precision.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_FP32_as_FP16) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 16);

  RunOpTest<float>("HardSigmoid",
                   {TestInputDef<float>({1, 2, 8}, false, input_data)},
                   {},
                   21,
                   ExpectedEPNodeAssignment::All,
                   kOnnxDomain,
                   0.004f,  // Tolerance. Comparing fp16 (QNN) with fp32 (CPU EP), so expect to need larger tolerance.
                   true);   // enable_htp_fp16_precision

  // Rank 4, non-default alpha and beta
  RunOpTest<float>("HardSigmoid",
                   {TestInputDef<float>({1, 2, 2, 4}, false, input_data)},
                   {test::MakeAttribute("alpha", 0.1f),
                    test::MakeAttribute("beta", 0.4f)},
                   21,
                   ExpectedEPNodeAssignment::All,
                   kOnnxDomain,
                   0.004f,  // Tolerance. Comparing fp16 (QNN) with fp32 (CPU EP), so expect to need larger tolerance.
                   true);   // enable_htp_fp16_precision
}

// Check that QNN EP can support float16 HardSigmoid on HTP
TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_FP16) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 16);

  RunFP16OpTest("HardSigmoid",
                {TestInputDef<float>({1, 2, 8}, false, input_data)},
                {},
                21,
                ExpectedEPNodeAssignment::All,
                kOnnxDomain);

  // Rank 4, non-default alpha and beta
  RunFP16OpTest("HardSigmoid",
                {TestInputDef<float>({1, 2, 2, 4}, false, input_data)},
                {test::MakeAttribute("alpha", 0.1f),
                 test::MakeAttribute("beta", 0.4f)},
                21,
                ExpectedEPNodeAssignment::All,
                kOnnxDomain);
}

// Returns a function that creates the model `X * HardSigmoid(X)`, which can be potentially fused
// into a single HardSwish(X) operator.
template <typename FloatType>
static GetTestModelFn BuildHardSigmoidFusionTestCase(TestInputDef<FloatType>& input_def,
                                                     std::optional<float> alpha,
                                                     std::optional<float> beta) {
  return [input_def, alpha, beta](ModelTestBuilder& builder) {
    MakeTestInput<FloatType>(builder, "input", input_def);

    // input -> HardSigmoid<alpha, beta> -> hs_output
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.reserve((alpha.has_value() ? 1u : 0u) + (beta.has_value() ? 1u : 0u));

    if (alpha.has_value()) {
      attrs.push_back(MakeAttribute("alpha", alpha.value()));
    }

    if (beta.has_value()) {
      attrs.push_back(MakeAttribute("beta", beta.value()));
    }

    builder.AddNode("HardSigmoid",
                    "HardSigmoid",
                    {"input"},
                    {"hs_out"},
                    kOnnxDomain,
                    attrs);

    // hs_out -> Mul -> output
    //             ^
    //             |
    // input ------+
    builder.MakeOutput("Y");
    builder.AddNode("Mul",
                    "Mul",
                    {"hs_out", "input"},
                    {"Y"});
  };
}

// Test FP32 fusion of HardSigmoid into HardSwish on the HTP backend with the enable_htp_fp16_precision option enabled
// to run it with fp16 precision.
TEST_F(QnnHTPBackendTests, HardSigmoidFusedIntoHardSwish_FP32_as_FP16) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  provider_options["enable_htp_fp16_precision"] = "1";

  std::vector<float> input_data = {-8.0f, -2.0f, 0.0f, 0.5f, 0.9f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, 0.0f, 0.2f, 0.4f, 0.8f, 2.1f, 4.3f, 7.0f};

  auto input_def = TestInputDef<float>({2, 2, 2, 2}, false, input_data);
  constexpr float alpha = 1.0f / 6.0f;
  constexpr float beta = 0.5f;
  auto model_fn = BuildHardSigmoidFusionTestCase<float>(input_def, alpha, beta);

  RunQnnModelTest(model_fn,
                  provider_options,
                  18,  // opset
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       // abs err. Comparing fp16 (QNN) vs fp32 (CPU EP) so can't expect too much.
                                       ElementwiseAbsoluteVerifier(0.01f)});
}

// Test FP16 fusion of HardSigmoid into HardSwish on the HTP backend.
TEST_F(QnnHTPBackendTests, HardSigmoidFusedIntoHardSwish_FP16) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  std::vector<float> input_data = {-8.0f, -2.0f, 0.0f, 0.5f, 0.9f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, 0.0f, 0.2f, 0.4f, 0.8f, 2.1f, 4.3f, 7.0f};

  auto input_def = TestInputDef<float>({2, 2, 2, 2}, false, input_data);
  auto input_fp16_def = ConvertToFP16InputDef(input_def);

  constexpr float alpha = 1.0f / 6.0f;
  constexpr float beta = 0.5f;
  auto model_fp32_fn = BuildHardSigmoidFusionTestCase<float>(input_def, alpha, beta);
  auto model_fp16_fn = BuildHardSigmoidFusionTestCase<Ort::Float16_t>(input_fp16_def, alpha, beta);

  TestFp16ModelAccuracy(model_fp32_fn,
                        model_fp16_fn,
                        provider_options,
                        18,  // opset
                        ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
