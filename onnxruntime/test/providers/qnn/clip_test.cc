// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Clip operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunClipTest(const TestInputDef<DataType>& input_def,
                        const std::vector<TestInputDef<DataType>>& min_max_defs,
                        ExpectedEPNodeAssignment expected_ep_assignment,
                        const std::string& backend_name = "cpu",
                        int opset = 13,
                        float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;

  RunQnnModelTest(BuildOpTestCase<DataType, DataType>("Clip_node", "Clip", {input_def}, min_max_defs, {}),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests:
//

// Test that Clip with a dynamic min or max input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Clip_Dynamic_MinMax_Unsupported) {
  // Dynamic min input is not supported.
  RunClipTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                     {TestInputDef<float>({}, false /* is_initializer */, {-5.0f})},
                     ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
  // Dynamic max input is not supported.
  RunClipTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                     {TestInputDef<float>({}, true, {-5.0f}),
                      TestInputDef<float>({}, false, {5.0f})},
                     ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test Clip with default min/max.
TEST_F(QnnCPUBackendTests, Clip_4D_f32_DefaultMinMax) {
  RunClipTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                     {},  // Don't specify min/max inputs.
                     ExpectedEPNodeAssignment::All);
}

// Test Clip with 5D input.
TEST_F(QnnCPUBackendTests, Clip_5D_f32) {
  RunClipTest<float>(TestInputDef<float>({1, 1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                     {TestInputDef<float>({}, true, {-5.0f}),
                      TestInputDef<float>({}, true, {5.0f})},
                     ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test Clip with float32 on HTP.
// Since QAIRT 2.35, default float precision on QNN HTP became FP16.
// Converting FP32 -> FP16 -> FP32 may introduce minor accuracy loss.
// For example, a value of -4.54545403 could become -4.54687548 after the conversion.
// The expected difference is approximately 0.00142145, so the tolerance is adjusted to 5e-3f.
TEST_F(QnnHTPBackendTests, Clip_f32) {
  RunClipTest<float>(TestInputDef<float>({1, 1, 3, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 12)),
                     {TestInputDef<float>({}, true, {-5.0f}),
                      TestInputDef<float>({}, true, {5.0f})},
                     ExpectedEPNodeAssignment::All,
                     "htp",
                     13,
                     5e-3f);
}

// Test Clip with int32 on HTP
TEST_F(QnnHTPBackendTests, Clip_int32) {
  RunClipTest<int32_t>(TestInputDef<int32_t>({1, 1, 3, 2}, false, {1, 2, -5, 3, -10, 25}),
                       {TestInputDef<int32_t>({}, true, {-5}),
                        TestInputDef<int32_t>({}, true, {5})},
                       ExpectedEPNodeAssignment::All,
                       "htp");
}

// Runs a QDQ Clip model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename QType>
static void RunQDQClipTestOnHTP(const TestInputDef<float>& input_def,
                                const std::vector<TestInputDef<float>>& min_max_defs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13,
                                bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildOpTestCase<float, float>("Clip_node", "Clip", {input_def}, {min_max_defs}, {});
  auto qdq_model_builder = BuildQDQOpTestCase<QType, float>("Clip_node", "Clip", {input_def}, {min_max_defs}, {},
                                                            kOnnxDomain, use_contrib_qdq);

  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test 8-bit QDQ Clip with default min/max.
// NOTE: The Clip operator is *optimized* away during L1 optimizations, so QNN EP does not get a graph with a Clip op.
// Instead, QNN EP will get a graph with a Q -> DQ.
// - Original sequence: Q1 -> DQ1 -> Clip -> Q2 -> DQ2
// - ClipQuantFusion: Fuses Clip -> QuantizeLinear resulting in Q1 -> DQ1 -> Q2' -> DQ2
// - DoubleQDQPairsRemover: Simplifies remaining Q1 -> DQ1 -> Q2' -> DQ2 sequence to Q1 -> DQ2.
TEST_F(QnnHTPBackendTests, Clip_U8_DefaultMinMax_Rank4) {
  RunQDQClipTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               {},  // Don't specify min/max inputs.
                               ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Clip with default min/max.
// NOTE: The Clip operator is *opt with QDQ Mninimized* away during L1 optimizations, so QNN EP does not get a graph with a Clip op.
// Instead, QNN EP will get a graph with a Q -> DQ.
// - Original sequence: Q1 -> DQ1 -> Clip -> Q2 -> DQ2
// - ClipQuantFusion: Fuses Clip -> QuantizeLinear resulting in Q1 -> DQ1 -> Q2' -> DQ2
// - DoubleQDQPairsRemover: Simplifies remaining Q1 -> DQ1 -> Q2' -> DQ2 sequence to Q1 -> DQ2.
TEST_F(QnnHTPBackendTests, Clip_U16_DefaultMinMax_Rank4) {
  RunQDQClipTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                {},  // Don't specify min/max inputs.
                                ExpectedEPNodeAssignment::All,
                                13,     // opset
                                true);  // Use com.microsoft Q/DQ ops
}

// Test 8-bit QDQ Clip with non-default min and max inputs. QNN EP will get a graph with a Clip operator.
TEST_F(QnnHTPBackendTests, Clip_U8_Rank4) {
  RunQDQClipTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               {TestInputDef<float>({}, true, {-5.0f}),
                                TestInputDef<float>({}, true, {5.0f})},
                               ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Clip with non-default min and max inputs. QNN EP will get a graph with a Clip operator.
TEST_F(QnnHTPBackendTests, Clip_U16_Rank4) {
  RunQDQClipTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                {TestInputDef<float>({}, true, {-5.0f}),
                                 TestInputDef<float>({}, true, {5.0f})},
                                ExpectedEPNodeAssignment::All,
                                13,     // opset
                                true);  // Use com.microsoft Q/DQ ops
}

// Clip with QDQ Min/Max: DQ(data) + DQ(min) + DQ(max) -> Clip -> Q(output).
TEST_F(QnnHTPBackendTests, Clip_U8_IndependentQDQ_MinMaxQDQ) {
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    std::vector<uint8_t> input_data(48);
    for (size_t i = 0; i < input_data.size(); ++i) {
      input_data[i] = static_cast<uint8_t>(i);
    }

    // input (u8) -> DQ ->
    NodeArg* quant_input = builder.MakeInput<uint8_t>({1, 3, 4, 4}, input_data);
    NodeArg* input_dq = builder.MakeIntermediate();
    const float scale = 0.1f;
    const uint8_t zp = 128;
    builder.AddDequantizeLinearNode<uint8_t>(quant_input, scale, zp, input_dq);

    // Quantized min/max -> DQ ->
    const float min_value = -1.0f;
    const float max_value = 1.0f;
    const uint8_t min_q = static_cast<uint8_t>(std::round(min_value / scale) + zp);
    const uint8_t max_q = static_cast<uint8_t>(std::round(max_value / scale) + zp);

    NodeArg* min_quant = builder.MakeInitializer<uint8_t>({}, {min_q});
    NodeArg* max_quant = builder.MakeInitializer<uint8_t>({}, {max_q});
    NodeArg* min_dq = builder.MakeIntermediate();
    NodeArg* max_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(min_quant, scale, zp, min_dq);
    builder.AddDequantizeLinearNode<uint8_t>(max_quant, scale, zp, max_dq);

    // Clip ->
    NodeArg* clip_output = builder.MakeIntermediate();
    builder.AddNode("Clip", {input_dq, min_dq, max_dq}, {clip_output});

    // Q -> output (u8)
    NodeArg* output = builder.MakeOutput();
    builder.AddQuantizeLinearNode<uint8_t>(clip_output, scale, zp, output);
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test QDQ Clip of rank 5.
TEST_F(QnnHTPBackendTests, Clip_U8_Rank5) {
  // We can't use the usual model-building functions because they add standalone Quantize and Dequantize nodes
  // at the input and output. These Q/DQ ops get lowered to QNN's Quantize and Dequantize operators, which DO NOT
  // support rank 5 tensors. Therefore, we have to create a test model that only instantiates the DQ -> Clip -> Q
  // QDQ node group, which gets lowered to a single QNN Clip node.
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    // input (u8) -> DQ ->
    builder.MakeInput<uint8_t>("X", {1, 1, 2, 2, 2}, {0, 1, 6, 10, 20, 100, 128, 255});
    builder.AddDequantizeLinearNode<uint8_t>("dq", "X", 1.0f, 0, "dq_out");  // scale = 1.0, zp = 0

    // Min/Max initializers
    builder.MakeScalarInitializer("min", 5.0f);
    builder.MakeScalarInitializer("max", 100.0f);

    // Clip ->
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode(
        "clip",
        "Clip",
        {"dq_out", "min", "max"},
        {"clip_out"},
        "",
        attributes);

    // Q -> output (u8)
    builder.AddQuantizeLinearNode<uint8_t>("q", "clip_out", 1.0f, 0, "Y");  // scale = 1.0, zp = 0
    builder.MakeOutput("Y");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test QDQ Clip with quantized min input only (and missing max input)
// This validates the pattern: DQ(data) + DQ(min) -> Clip -> Q(output)
TEST_F(QnnHTPBackendTests, Clip_U8_QuantizedMin) {
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    const float min_value = 0.0f;
    const float min_scale = 0.001f;
    const uint8_t min_zp = 128;

    uint8_t quantized_min = static_cast<uint8_t>(std::round(min_value / min_scale) + min_zp);
    builder.MakeInitializer<uint8_t>("min_quantized", {}, {quantized_min});
    builder.AddDequantizeLinearNode<uint8_t>("min_dq", "min_quantized", min_scale, min_zp, "min_dq_out");

    const float data_scale = 0.001f;
    const uint8_t data_zp = 128;
    std::vector<uint8_t> input_data(200);
    // Use values 28-227 so ~half are below and half above zero
    for (size_t i = 0; i < 200; i++) {
      input_data[i] = static_cast<uint8_t>(28 + i);
    }

    builder.MakeInput<uint8_t>("data_quantized", {200}, input_data);
    builder.AddDequantizeLinearNode<uint8_t>("data_dq", "data_quantized", data_scale, data_zp, "data_dq_out");

    // Clip with only min input (no max)
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("clip", "Clip", {"data_dq_out", "min_dq_out"}, {"clip_output"}, "", attributes);

    builder.AddQuantizeLinearNode<uint8_t>("q", "clip_output", data_scale, data_zp, "Y");
    builder.MakeOutput("Y");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(model_fn,
                  provider_options,
                  11,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test QDQ Clip with quantized max input only (and missing min input)
// This validates the pattern: DQ(data) + DQ(max) -> Clip -> Q(output)
TEST_F(QnnHTPBackendTests, Clip_U16_QuantizedMax) {
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    const float max_value = 0.0f;
    const float max_scale = 0.001f;
    const uint16_t max_zp = 32768;

    uint16_t quantized_max = static_cast<uint16_t>(std::round(max_value / max_scale) + max_zp);
    builder.MakeInitializer<uint16_t>("max_quantized", {}, {quantized_max});
    builder.AddDequantizeLinearNode<uint16_t>("max_dq", "max_quantized", max_scale, max_zp, "max_dq_out");

    const float data_scale = 0.001f;
    const uint16_t data_zp = 32768;
    std::vector<uint16_t> input_data(200);
    for (size_t i = 0; i < 200; i++) {
      input_data[i] = static_cast<uint16_t>(32768 - 100 + i);
    }

    builder.MakeInput<uint16_t>("data_quantized", {200}, input_data);
    builder.AddDequantizeLinearNode<uint16_t>("data_dq", "data_quantized", data_scale, data_zp, "data_dq_out");

    // Clip with only max input (no min) - provide empty input for min
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("clip", "Clip", {"data_dq_out", "", "max_dq_out"}, {"clip_output"}, "", attributes);

    builder.AddQuantizeLinearNode<uint16_t>("q", "clip_output", data_scale, data_zp, "Y");
    builder.MakeOutput("Y");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(model_fn,
                  provider_options,
                  21,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test QDQ Clip with both quantized min and max inputs
// This validates the pattern: DQ(data) + DQ(min) + DQ(max) -> Clip -> Q(output)
TEST_F(QnnHTPBackendTests, Clip_U8_QuantizedMinMax) {
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    const float min_value = -0.05f;
    const float min_scale = 0.001f;
    const uint8_t min_zp = 128;

    uint8_t quantized_min = static_cast<uint8_t>(std::round(min_value / min_scale) + min_zp);
    builder.MakeInitializer<uint8_t>("min_quantized", {}, {quantized_min});
    builder.AddDequantizeLinearNode<uint8_t>("min_dq", "min_quantized", min_scale, min_zp, "min_dq_out");

    const float max_value = 0.05f;
    const float max_scale = 0.001f;
    const uint8_t max_zp = 128;

    uint8_t quantized_max = static_cast<uint8_t>(std::round(max_value / max_scale) + max_zp);
    builder.MakeInitializer<uint8_t>("max_quantized", {}, {quantized_max});
    builder.AddDequantizeLinearNode<uint8_t>("max_dq", "max_quantized", max_scale, max_zp, "max_dq_out");

    const float data_scale = 0.001f;
    const uint8_t data_zp = 128;
    std::vector<uint8_t> input_data(200);
    for (size_t i = 0; i < 200; i++) {
      input_data[i] = static_cast<uint8_t>(28 + i);
    }

    builder.MakeInput<uint8_t>("data_quantized", {200}, input_data);
    builder.AddDequantizeLinearNode<uint8_t>("data_dq", "data_quantized", data_scale, data_zp, "data_dq_out");

    // Clip with both min and max inputs
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("clip", "Clip", {"data_dq_out", "min_dq_out", "max_dq_out"}, {"clip_output"}, "", attributes);

    builder.AddQuantizeLinearNode<uint8_t>("q", "clip_output", data_scale, data_zp, "Y");
    builder.MakeOutput("Y");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test FP16 Clip with min (FP16)
TEST_F(QnnHTPBackendTests, Clip_FP16) {
#if defined(_WIN32)
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "Test requires HTP FP16 support (arch > V68).";
  }
#endif
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  auto f32_input = TestInputDef<float>({1, 3, 2, 2}, false,
                                       {-10.0f, -8.0f, -3.5f, 2.2f,
                                        1.3f, 1.5f, 3.2f, 5.8f,
                                        5.8f, 9.7f, 8.5f, 8.9f});
  std::vector<Ort::Float16_t> f16_data;
  std::for_each(f32_input.GetRawData().begin(), f32_input.GetRawData().end(),
                [&f16_data](const float data) {
                  f16_data.push_back(static_cast<Ort::Float16_t>(data));
                });
  auto f16_input = TestInputDef<Ort::Float16_t>({1, 3, 2, 2}, false, f16_data);

  const float min_f32 = 1.2f;
  const Ort::Float16_t min_f16 = static_cast<Ort::Float16_t>(min_f32);
  auto f32_min_input = TestInputDef<float>({}, true, {min_f32});
  auto f16_min_input = TestInputDef<Ort::Float16_t>({}, true, {min_f16});

  auto f32_model_builder = BuildOpTestCase<float, float>("Clip_node", "Clip", {f32_input}, {f32_min_input}, {});
  auto f16_model_builder = BuildOpTestCase<Ort::Float16_t, Ort::Float16_t>("Clip_node", "Clip", {f16_input}, {f16_min_input}, {});
  int opset = 13;
  ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All;

  TestFp16ModelAccuracy(f32_model_builder,
                        f16_model_builder,
                        provider_options,
                        opset,
                        expected_ep_assignment);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)
//
// GPU tests:
//

// Test Clip with float32 on GPU
TEST_F(QnnGPUBackendTests, Clip_fp32) {
  RunClipTest<float>(TestInputDef<float>({1, 1, 3, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 12)),
                     {TestInputDef<float>({}, true, {-5.0f}),
                      TestInputDef<float>({}, true, {5.0f})},
                     ExpectedEPNodeAssignment::All,
                     "gpu",
                     13);
}

// Test Clip with int32 on GPU
// Disable Reason : Doesn't work.
TEST_F(QnnGPUBackendTests, DISABLED_Clip_int32) {
  RunClipTest<int32_t>(TestInputDef<int32_t>({1, 1, 3, 2}, false, {1, 2, -5, 3, -10, 25}),
                       {TestInputDef<int32_t>({}, true, {-5}),
                        TestInputDef<int32_t>({}, true, {5})},
                       ExpectedEPNodeAssignment::All,
                       "gpu");
}

#endif  // defined(_M_ARM64) GPU tests

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
