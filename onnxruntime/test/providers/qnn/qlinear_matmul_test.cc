// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "qnn_test_utils.h"
#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>
#include <unordered_map>

#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Graph builder helpers
// ---------------------------------------------------------------------------

// Quantize a float value to int8 using the given scale and zero_point.
template <typename QType>
static QType QuantizeVal(float val, float scale, int32_t zero_point) {
  constexpr float qmin = static_cast<float>(std::numeric_limits<QType>::min());
  constexpr float qmax = static_cast<float>(std::numeric_limits<QType>::max());
  float q = std::round(val / scale) + static_cast<float>(zero_point);
  return static_cast<QType>(std::max(qmin, std::min(qmax, q)));
}

// Quantize a float vector element-wise.
template <typename QType>
static std::vector<QType> QuantizeData(const std::vector<float>& data, float scale, int32_t zp) {
  std::vector<QType> out(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    out[i] = QuantizeVal<QType>(data[i], scale, zp);
  }
  return out;
}

// Compute per-tensor scale+zp from a float range.
template <typename QType>
static QuantParams<QType> ComputeQuantParams(float rmin, float rmax) {
  return QuantParams<QType>::Compute(rmin, rmax);
}

/**
 * Builds a QLinearMatMul graph and dequantizes its output to float.
 *
 * The graph is:  a, b (quantized) -> QLinearMatMul -> y_q (quantized) -> DequantizeLinear -> y (float)
 *
 * We dequantize the output so accuracy is compared in float space with a tolerance. QNN HTP and
 * ORT CPU use slightly different rounding/accumulation, so the raw quantized integer output can
 * differ by 1 LSB. Comparing the dequantized float output with an fp32 tolerance absorbs that
 * expected 1-unit quantization difference (same rationale as the QDQ MatMul tests).
 *
 * AType/BType/YType: int8 or uint8.
 * b_is_initializer: if true, B and its quant params are graph initializers.
 * dynamic_a_scale: if true, a_scale is a graph input (not initializer) — for rejection tests.
 */
template <typename AType, typename BType, typename YType>
static GetTestModelFn BuildQLinearMatMulTestCase(
    const std::vector<int64_t>& shape_a,
    const std::vector<int64_t>& shape_b,
    bool b_is_initializer = false,
    bool dynamic_a_scale = false) {
  return [shape_a, shape_b, b_is_initializer, dynamic_a_scale](ModelTestBuilder& builder) {
    const size_t num_a = static_cast<size_t>(
        std::accumulate(shape_a.begin(), shape_a.end(), int64_t{1}, std::multiplies<int64_t>()));
    const size_t num_b = static_cast<size_t>(
        std::accumulate(shape_b.begin(), shape_b.end(), int64_t{1}, std::multiplies<int64_t>()));

    // Float data in a modest range to avoid saturation.
    const auto float_a = GetFloatDataInRange(-1.0f, 1.0f, num_a);
    const auto float_b = GetFloatDataInRange(-0.5f, 0.5f, num_b);

    // Quant params — simple per-tensor.
    const auto qp_a = ComputeQuantParams<AType>(-1.0f, 1.0f);
    const auto qp_b = ComputeQuantParams<BType>(-0.5f, 0.5f);
    const auto qp_y = ComputeQuantParams<YType>(-2.0f, 2.0f);

    // Quantize A and B data.
    const auto q_a = QuantizeData<AType>(float_a, qp_a.scale, static_cast<int32_t>(qp_a.zero_point));
    const auto q_b = QuantizeData<BType>(float_b, qp_b.scale, static_cast<int32_t>(qp_b.zero_point));

    // Build input A — always a graph input (dynamic).
    builder.MakeInput<AType>("a", shape_a, q_a);

    // Build input B.
    if (b_is_initializer) {
      builder.MakeInitializer<BType>("b", shape_b, q_b);
    } else {
      builder.MakeInput<BType>("b", shape_b, q_b);
    }

    // a_scale
    std::string a_scale_name = "a_scale";
    if (dynamic_a_scale) {
      // Make a_scale a dynamic graph input to trigger rejection.
      builder.MakeInput<float>(a_scale_name, {}, std::vector<float>{qp_a.scale});
    } else {
      builder.MakeScalarInitializer<float>(a_scale_name, qp_a.scale);
    }

    // b_scale, y_scale — always scalar initializers.
    builder.MakeScalarInitializer<float>("b_scale", qp_b.scale);
    builder.MakeScalarInitializer<float>("y_scale", qp_y.scale);

    // QLinearMatMul (opset 10 and 21) requires all 8 inputs — scale and zero_point
    // for a, b, and y. Build the full input list.
    builder.MakeScalarInitializer<AType>("a_zp", qp_a.zero_point);
    builder.MakeScalarInitializer<BType>("b_zp", qp_b.zero_point);
    builder.MakeScalarInitializer<YType>("y_zp", qp_y.zero_point);

    std::vector<std::string> node_inputs = {
        "a", a_scale_name, "a_zp", "b", "b_scale", "b_zp", "y_scale", "y_zp"};

    if (dynamic_a_scale) {
      // Rejection test: emit the quantized output directly (no DequantizeLinear). The DQ op is
      // itself supported by QNN EP, so wrapping it would let QNN grab the lone DQ node and break
      // an ExpectedEPNodeAssignment::None check. Accuracy isn't verified for rejection tests.
      builder.MakeOutput("y_q");
      builder.AddNode("QLinearMatMul", "QLinearMatMul", node_inputs, {"y_q"}, kOnnxDomain);
    } else {
      builder.AddNode("QLinearMatMul", "QLinearMatMul", node_inputs, {"y_q"}, kOnnxDomain);

      // Dequantize the quantized output to float (reuse y_scale / y_zp), so accuracy is
      // compared in float space with tolerance rather than via exact integer equality.
      builder.MakeOutput("y");
      builder.AddNode("DequantizeLinear", "DequantizeLinear", {"y_q", "y_scale", "y_zp"}, {"y"}, kOnnxDomain);
    }
  };
}

// Helper: run QLinearMatMul on the given backend, checking IsOpSupported behavior.
template <typename AType = uint8_t, typename BType = uint8_t, typename YType = uint8_t>
static void RunQLinearMatMulTest(
    const std::vector<int64_t>& shape_a,
    const std::vector<int64_t>& shape_b,
    const std::string& backend_name,
    ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
    bool b_is_initializer = false,
    int opset = 10,
    bool dynamic_a_scale = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildQLinearMatMulTestCase<AType, BType, YType>(
          shape_a, shape_b, b_is_initializer, dynamic_a_scale),
      provider_options, opset, expected_ep_assignment,
      // Output is dequantized with y_scale = (2 - -2)/255 ≈ 0.0157 per LSB. HTP and CPU EP can
      // differ by a couple of quantization units on deeper reductions, so allow ~2.5 LSB.
      /*fp32_abs_err=*/0.04f);
}

// ---------------------------------------------------------------------------
// Negative / IsOpSupported rejection tests (CPU and HTP)
// ---------------------------------------------------------------------------

TEST_F(QnnCPUBackendTests, QLinearMatMulOp_DynamicScale_Unsupported) {
  // a_scale is a graph input (not initializer) — must not be assigned to QNN EP.
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>(
      {2, 3}, {3, 2}, "cpu", ExpectedEPNodeAssignment::None,
      /*b_is_initializer=*/false, /*opset=*/10,
      /*dynamic_a_scale=*/true);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_DynamicZeroPoint_Unsupported) {
  // a_zero_point is a dynamic graph input — must not be assigned to QNN EP.
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    builder.MakeInput<uint8_t>("a", {2, 3}, std::vector<uint8_t>(6, 128u));
    builder.MakeScalarInitializer<float>("a_scale", 0.02f);
    builder.MakeInput<uint8_t>("a_zp", {}, std::vector<uint8_t>{128u});
    builder.MakeInput<uint8_t>("b", {3, 2}, std::vector<uint8_t>(6, 128u));
    builder.MakeScalarInitializer<float>("b_scale", 0.02f);
    builder.MakeScalarInitializer<uint8_t>("b_zp", 128u);
    builder.MakeScalarInitializer<float>("y_scale", 0.04f);
    builder.MakeScalarInitializer<uint8_t>("y_zp", 128u);
    builder.MakeOutput("y");
    builder.AddNode("QLinearMatMul", "QLinearMatMul",
                    {"a", "a_scale", "a_zp", "b", "b_scale", "b_zp", "y_scale", "y_zp"},
                    {"y"}, kOnnxDomain);
  };

  RunQnnModelTest(model_fn, provider_options, 10, ExpectedEPNodeAssignment::None);
}

// ---------------------------------------------------------------------------
// CPU backend accuracy tests
// ---------------------------------------------------------------------------

TEST_F(QnnCPUBackendTests, QLinearMatMulOp_CPU_u8u8u8_2D) {
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>({2, 3}, {3, 2}, "cpu");
}

TEST_F(QnnCPUBackendTests, QLinearMatMulOp_CPU_u8u8u8_2D_InitB) {
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>(
      {2, 3}, {3, 2}, "cpu", ExpectedEPNodeAssignment::All, /*b_is_initializer=*/true);
}

// ---------------------------------------------------------------------------
// HTP backend accuracy tests
// ---------------------------------------------------------------------------

// --- uint8 shape coverage ---

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_2D) {
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>({2, 3}, {3, 2}, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_2D_InitB) {
  // B is a static initializer — exercises FullyConnected path.
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>(
      {2, 3}, {3, 2}, "htp", ExpectedEPNodeAssignment::All, /*b_is_initializer=*/true);
}

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_3D) {
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>({2, 3, 4}, {4, 2}, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_4D) {
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>(
      {2, 3, 3, 3}, {3, 2}, "htp", ExpectedEPNodeAssignment::All, /*b_is_initializer=*/true);
}

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_4D_Batched) {
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>({2, 3, 3, 4}, {2, 3, 4, 2}, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_Rank1A) {
  // A is rank-1 — triggers Reshape insertion before MatMul.
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>({4}, {4, 2}, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_Rank1B) {
  // B is rank-1 — triggers Reshape insertion after MatMul (dot product / mv case).
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>({2, 3}, {3}, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_Rank1Both) {
  // Both rank-1 — dot product, output is scalar.
  RunQLinearMatMulTest<uint8_t, uint8_t, uint8_t>({4}, {4}, "htp");
}

// --- int8 coverage ---

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_s8_2D) {
  RunQLinearMatMulTest<int8_t, int8_t, int8_t>({2, 3}, {3, 2}, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_s8_InitB) {
  RunQLinearMatMulTest<int8_t, int8_t, int8_t>(
      {2, 3}, {3, 2}, "htp", ExpectedEPNodeAssignment::All, /*b_is_initializer=*/true);
}

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_Mixed_u8s8u8) {
  RunQLinearMatMulTest<uint8_t, int8_t, uint8_t>({2, 3}, {3, 2}, "htp");
}

// --- Opset 21: float16 and bfloat16 scales ---

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_Float16Scale) {
  // Scales provided as float16 — tests the ReadScaleAsFloat32 upcast path.
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    const std::vector<float> q_a(6, 128.0f);
    const std::vector<float> q_b(6, 64.0f);
    builder.MakeInput<uint8_t>("a", {2, 3}, std::vector<uint8_t>(6, 128u));
    builder.MakeInput<uint8_t>("b", {3, 2}, std::vector<uint8_t>(6, 64u));

    // Encode 0.02f as float16 bits.
    Ort::Float16_t a_scale_fp16(0.02f);
    Ort::Float16_t b_scale_fp16(0.02f);
    Ort::Float16_t y_scale_fp16(0.04f);

    builder.MakeInitializer("a_scale", gsl::span<const int64_t>{},
                            ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                            AsByteSpan(&a_scale_fp16.val, sizeof(uint16_t)));
    builder.MakeInitializer("b_scale", gsl::span<const int64_t>{},
                            ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                            AsByteSpan(&b_scale_fp16.val, sizeof(uint16_t)));
    builder.MakeInitializer("y_scale", gsl::span<const int64_t>{},
                            ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                            AsByteSpan(&y_scale_fp16.val, sizeof(uint16_t)));

    builder.MakeScalarInitializer<uint8_t>("a_zp", 128u);
    builder.MakeScalarInitializer<uint8_t>("b_zp", 64u);
    builder.MakeScalarInitializer<uint8_t>("y_zp", 128u);

    // Dequantize the quantized output to float so accuracy is compared with tolerance (HTP and CPU
    // EP can differ by 1 LSB; VerifyOutput uses exact EXPECT_EQ for integer outputs). y_scale is
    // float16 here, which DequantizeLinear does not accept, so use a float32 scale for the DQ.
    builder.MakeScalarInitializer<float>("y_scale_f32", 0.04f);
    builder.AddNode("QLinearMatMul", "QLinearMatMul",
                    {"a", "a_scale", "a_zp", "b", "b_scale", "b_zp", "y_scale", "y_zp"},
                    {"y_q"}, kOnnxDomain);
    builder.MakeOutput("y");
    builder.AddNode("DequantizeLinear", "DequantizeLinear", {"y_q", "y_scale_f32", "y_zp"}, {"y"}, kOnnxDomain);
  };

  RunQnnModelTest(model_fn, provider_options, 21, ExpectedEPNodeAssignment::All, /*fp32_abs_err=*/0.02f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// BFloat16 scale inputs are only supported on HTP v81+ (ARM64 only).
#if defined(__aarch64__) || defined(_M_ARM64)

TEST_F(QnnHTPBackendTests, QLinearMatMulOp_HTP_u8_BFloat16Scale) {
  // Scales provided as bfloat16 (opset 21 TS type) — BF16 initializers require v81+.
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    builder.MakeInput<uint8_t>("a", {2, 3}, std::vector<uint8_t>(6, 128u));
    builder.MakeInput<uint8_t>("b", {3, 2}, std::vector<uint8_t>(6, 64u));

    Ort::BFloat16_t a_scale_bf16(0.02f);
    Ort::BFloat16_t b_scale_bf16(0.02f);
    Ort::BFloat16_t y_scale_bf16(0.04f);

    builder.MakeInitializer("a_scale", gsl::span<const int64_t>{},
                            ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
                            AsByteSpan(&a_scale_bf16.val, sizeof(uint16_t)));
    builder.MakeInitializer("b_scale", gsl::span<const int64_t>{},
                            ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
                            AsByteSpan(&b_scale_bf16.val, sizeof(uint16_t)));
    builder.MakeInitializer("y_scale", gsl::span<const int64_t>{},
                            ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
                            AsByteSpan(&y_scale_bf16.val, sizeof(uint16_t)));

    builder.MakeScalarInitializer<uint8_t>("a_zp", 128u);
    builder.MakeScalarInitializer<uint8_t>("b_zp", 64u);
    builder.MakeScalarInitializer<uint8_t>("y_zp", 128u);

    // Dequantize the quantized output to float so accuracy is compared with tolerance (HTP and CPU
    // EP can differ by 1 LSB; VerifyOutput uses exact EXPECT_EQ for integer outputs). y_scale is
    // bfloat16 here, which DequantizeLinear does not accept, so use a float32 scale for the DQ.
    builder.MakeScalarInitializer<float>("y_scale_f32", 0.04f);
    builder.AddNode("QLinearMatMul", "QLinearMatMul",
                    {"a", "a_scale", "a_zp", "b", "b_scale", "b_zp", "y_scale", "y_zp"},
                    {"y_q"}, kOnnxDomain);
    builder.MakeOutput("y");
    builder.AddNode("DequantizeLinear", "DequantizeLinear", {"y_q", "y_scale_f32", "y_zp"}, {"y"}, kOnnxDomain);
  };

  RunQnnModelTest(model_fn, provider_options, 21, ExpectedEPNodeAssignment::All, /*fp32_abs_err=*/0.02f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
