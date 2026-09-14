// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>

#include <gsl/gsl_util>
#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Re-implement testcases from test/contrib_ops/matmul_4bits_test.cc.

template <int bits>
void QuantizeDequantize(std::vector<float>& raw_vals,
                        std::vector<uint8_t>& quant_vals,
                        std::vector<float>& scales,
                        std::vector<uint8_t>* zp,
                        int32_t N,
                        int32_t K,
                        int32_t block_size) {
  QuantizeBlockwise<float, bits>(quant_vals.data(),
                                 scales.data(),
                                 zp != nullptr ? zp->data() : nullptr,
                                 raw_vals.data(),
                                 block_size,
                                 true,
                                 K,
                                 N,
                                 N);

  // Note that raw_vals is NxK after dequant
  DequantizeBlockwise<float, bits>(raw_vals.data(),                       // dequantized output
                                   quant_vals.data(),                     // quantized input
                                   scales.data(),                         // quantization scales
                                   zp != nullptr ? zp->data() : nullptr,  // quantization zero points
                                   block_size,                            // quantization block size
                                   true,                                  // columnwise quantization
                                   K,                                     // number of rows
                                   N);                                    // number of columns
}

struct TestParams {
  int64_t batch_count{1};
  int64_t M;
  int64_t N;
  int64_t K;
  int64_t block_size;
  // accuracy_level=1 selects fp32 compute in the CPU EP kernel, which keeps the CPU-side result an accurate
  // reference to compare QNN EP against. QNN EP ignores this attribute entirely.
  int64_t accuracy_level{1};

  bool has_zero_point{false};
  bool is_zp_symmetric{false};
  bool enable_lpbq{false};
};

// Adds the MatMulNBits weight B (packed uint8 BQ), float scales, and optional uint8 zero_point.
template <int bits>
static void AddMatMulNBitsWeightInputs(ModelTestBuilder& builder,
                                       const TestParams& params,
                                       std::vector<float>& input1_f_vals,
                                       bool is_symmetric_zp,
                                       std::vector<std::string>& input_names) {
  const int64_t k_blocks = (params.K + params.block_size - 1) / params.block_size;
  const int64_t blob_size = (params.block_size * bits + 7) / 8;
  const size_t q_scale_size = static_cast<size_t>(params.N * k_blocks);
  const size_t q_data_size_in_bytes = static_cast<size_t>(params.N * k_blocks * blob_size);  // packed as UInt4x2
  const int64_t zero_point_blob_size = (k_blocks * bits + 7) / 8;
  const size_t q_zp_size_in_bytes = static_cast<size_t>(params.N * zero_point_blob_size);  // packed as UInt4x2

  std::vector<uint8_t> input1_vals(q_data_size_in_bytes);
  std::vector<float> scales(q_scale_size);
  std::vector<uint8_t> zp(q_zp_size_in_bytes);

  // Hardcode zp instead of computing it when symmetric is required.
  if (is_symmetric_zp) {
    switch (bits) {
      case 2:
        std::fill(zp.begin(), zp.end(), 0b10101010);
        break;
      case 4:
        std::fill(zp.begin(), zp.end(), 0b10001000);
        break;
      case 8:
        std::fill(zp.begin(), zp.end(), 0b10000000);
        break;
    }
  }

  QuantizeDequantize<bits>(input1_f_vals,
                           input1_vals,
                           scales,
                           params.has_zero_point && !is_symmetric_zp ? &zp : nullptr,
                           static_cast<int32_t>(params.N),
                           static_cast<int32_t>(params.K),
                           static_cast<int32_t>(params.block_size));

  auto input1_def = TestInputDef<uint8_t>({params.N, k_blocks, blob_size}, true, input1_vals);
  MakeTestInput<uint8_t>(builder, "input1", input1_def);
  input_names.push_back("input1");

  auto scales_def = TestInputDef<float>({params.N, k_blocks}, true, scales);
  MakeTestInput<float>(builder, "scales", scales_def);
  input_names.push_back("scales");

  if (params.has_zero_point) {
    auto zp_def = TestInputDef<uint8_t>({params.N, zero_point_blob_size}, true, zp);
    MakeTestInput<uint8_t>(builder, "zero_point", zp_def);
    input_names.push_back("zero_point");
  }
}

// Adds the MatMulNBits contrib node (inputs `input_names` → "Y") with the K/N/block_size/bits/accuracy_level attrs.
static void AddMatMulNBitsNode(ModelTestBuilder& builder,
                               const TestParams& params,
                               int64_t bits,
                               const std::vector<std::string>& input_names) {
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
  attributes.push_back(builder.MakeScalarAttribute("K", static_cast<int64_t>(params.K)));
  attributes.push_back(builder.MakeScalarAttribute("N", static_cast<int64_t>(params.N)));
  attributes.push_back(builder.MakeScalarAttribute("block_size", static_cast<int64_t>(params.block_size)));
  attributes.push_back(builder.MakeScalarAttribute("bits", bits));
  attributes.push_back(builder.MakeScalarAttribute("accuracy_level", static_cast<int64_t>(params.accuracy_level)));

  builder.AddNode("matmul_nbits", "MatMulNBits", input_names, {"Y"}, kMSDomain, attributes);
}

template <int bits>
static void RunMatMulNBitsTest(const TestParams params,
                               const std::string& backend_name = "gpu",
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                               float fp32_abs_err = 0.05f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_block_quant_weight_optimization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  if (backend_name == "htp") {
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  }
#endif

  auto model_builder = [&params, &backend_name](ModelTestBuilder& builder) {
    std::vector<std::string> input_names;

    RandomValueGenerator random{1234};
    std::vector<float> input0_vals(random.Gaussian<float>(AsSpan({params.batch_count, params.M, params.K}),
                                                          0.0f,
                                                          0.25f));
    std::vector<float> input1_f_vals(random.Gaussian<float>(AsSpan({params.K, params.N}), 0.0f, 0.25f));

    auto input0_def = TestInputDef<float>({params.batch_count, params.M, params.K}, false, input0_vals);
    MakeTestInput<float>(builder, "input0", input0_def);
    input_names.push_back("input0");

    AddMatMulNBitsWeightInputs<bits>(builder,
                                     params,
                                     input1_f_vals,
                                     /*is_symmetric_zp=*/bits == 4 && backend_name == "gpu",
                                     input_names);

    builder.MakeOutput("Y");

    AddMatMulNBitsNode(builder, params, bits, input_names);
  };

  RunQnnModelTest(model_builder,
                  provider_options,
                  13,  // opset version for contrib ops
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(fp32_abs_err)});
}

// Tests the accuracy of a QDQ MatMulNBits graph that exercises the HTP block-quantized (BQ) *activation* paths:
//   - activation A: float → Q(ActQType) → DQ → MatMulNBits input A
//   - weight B    : packed uint8 BQ initializer (bits/block_size), symmetric unless has_zero_point
//   - output Y    : MatMulNBits → Q(ActQType) → DQ → graph output
template <int bits, typename ActQType>
static void RunHtpQDQMatMulNBitsTest(const TestParams params,
                                     std::optional<bool> expect_native_bq = std::nullopt,
                                     ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                                     QDQTolerance tolerance = QDQTolerance()) {
  static_assert(std::is_same_v<ActQType, uint16_t> || std::is_same_v<ActQType, int16_t>,
                "MatMulNBits QDQ activations on HTP must be uint16 or int16.");

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  if (params.enable_lpbq) {
    provider_options["enable_block_quant_weight_optimization"] = "1";
  } else {
    // Explicit: keep the BQ→LPBQ conversion off so the BQ (native BLOCK / BW_FLOAT_BLOCK) path is exercised.
    provider_options["enable_block_quant_weight_optimization"] = "0";
  }
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

#ifndef QNN_HTP_NATIVE_BQ_AVAILABLE
  // HTP native BQ support starts from QAIRT 2.51 (QNN API 2.40).
  // Disable here to avoid guarding everywhere.
  if (expect_native_bq.has_value()) {
    *expect_native_bq = false;
  }
#endif

  // When expect_native_bq is set, dump the QNN graph JSON so the selected BQ encoding can be verified:
  //   - Native BQ (QNN_QUANTIZATION_ENCODING_BLOCK): consumes/produces INT16 directly, so only the
  //     activation Quantize and output Dequantize of the QDQ model remain → Quantize=1, Dequantize=1.
  //   - Fallback (QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK): computes in FP16, so an extra INT16→FP16
  //     activation Dequantize and FP16→INT16 output Quantize are inserted → Quantize=2, Dequantize=2.
  const std::filesystem::path json_qnn_graph_dir = "MatMulNBitsNativeBQ";
  auto cleanup = gsl::finally([&expect_native_bq, &json_qnn_graph_dir]() {
    if (expect_native_bq.has_value()) {
      std::filesystem::remove_all(json_qnn_graph_dir);
    }
  });
  if (expect_native_bq.has_value()) {
    std::filesystem::remove_all(json_qnn_graph_dir);
    ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
    provider_options["dump_json_qnn_graph"] = "1";
    provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  }

  // Generate the raw float data once so that the float and QDQ models use identical inputs and weights.
  RandomValueGenerator random{1234};
  std::vector<float> input0_vals(random.Gaussian<float>(AsSpan({params.batch_count, params.M, params.K}),
                                                        0.0f,
                                                        0.25f));
  const std::vector<float> weight_f_vals(random.Gaussian<float>(AsSpan({params.K, params.N}), 0.0f, 0.25f));

  const TestInputDef<float> input0_def({params.batch_count, params.M, params.K}, false, input0_vals);

  // ── Float baseline: input0 (float) → MatMulNBits → Y ──
  auto f32_model_builder = [&params, &input0_def, &weight_f_vals](ModelTestBuilder& builder) {
    std::vector<std::string> input_names;

    MakeTestInput<float>(builder, "input0", input0_def);
    input_names.push_back("input0");

    // AddMatMulNBitsWeightInputs() overwrites its float weights with the dequantized values, so pass a copy.
    std::vector<float> weight_vals(weight_f_vals);
    AddMatMulNBitsWeightInputs<bits>(builder, params, weight_vals, params.is_zp_symmetric, input_names);

    AddMatMulNBitsNode(builder, params, bits, input_names);
    builder.MakeOutput("Y");
  };

  // ── QDQ model: input0 → Q(ActQType) → DQ → MatMulNBits → Q(ActQType) → DQ → output ──
  auto qdq_model_builder = [&params, &input0_def, &weight_f_vals](
                               ModelTestBuilder& builder,
                               std::vector<QuantParams<ActQType>>& output_qparams) {
    std::vector<std::string> input_names;

    // ── Activation A: float → Q(ActQType) → DQ ──
    MakeTestInput<float>(builder, "input0", input0_def);
    QuantParams<ActQType> act_qparams = GetTestInputQuantParams<ActQType>(input0_def);
    input_names.push_back(AddQDQNodePair<ActQType>(builder,
                                                   "act",
                                                   "input0",
                                                   act_qparams.scale,
                                                   act_qparams.zero_point));

    // ── Weight B + scales (+ optional zero_point) ──
    std::vector<float> weight_vals(weight_f_vals);
    AddMatMulNBitsWeightInputs<bits>(builder, params, weight_vals, params.is_zp_symmetric, input_names);

    // ── MatMulNBits → Y ──
    AddMatMulNBitsNode(builder, params, bits, input_names);

    // ── Output Y: → Q(ActQType) → DQ → graph output ──
    AddQDQNodePairWithOutputAsGraphOutput<ActQType>(builder,
                                                    "out",
                                                    "Y",
                                                    output_qparams[0].scale,
                                                    output_qparams[0].zero_point);
  };

  TestQDQModelAccuracy<ActQType>(f32_model_builder,
                                 qdq_model_builder,
                                 provider_options,
                                 21,  // opset 21 for 16-bit Q/DQ
                                 expected_ep_assignment,
                                 tolerance);

  if (expect_native_bq.has_value()) {
    const size_t expected_count = *expect_native_bq ? 1 : 2;
    AssertOpInQnnGraph(json_qnn_graph_dir, "Quantize", expected_count);
    AssertOpInQnnGraph(json_qnn_graph_dir, "Dequantize", expected_count);
  }
}

#if defined(_M_ARM64)
// QNN GPU only support FP16 activations and Q4_0 weights, with zero_points = 8
// Accumulation with larger channel accumulates more error. Set higher abs_error with respect to K.
TEST_F(QnnGPUBackendTests, MatMulNBits_Basic_M1_N128_K512_withZp) {
  TestParams params;
  params.M = 1;
  params.N = 128;
  params.K = 512;
  params.block_size = 32;
  params.has_zero_point = true;
  RunMatMulNBitsTest<4>(params);
}

TEST_F(QnnGPUBackendTests, MatMulNBits_Basic_M1_N128_K512) {
  TestParams params;
  params.M = 1;
  params.N = 128;
  params.K = 512;
  params.block_size = 32;
  params.has_zero_point = false;
  RunMatMulNBitsTest<4>(params);
}

TEST_F(QnnGPUBackendTests, MatMulNBits_Basic_M10_N128_K512_withZp) {
  TestParams params;
  params.M = 10;
  params.N = 128;
  params.K = 512;
  params.block_size = 32;
  params.has_zero_point = true;
  RunMatMulNBitsTest<4>(params);
}

TEST_F(QnnGPUBackendTests, MatMulNBits_Basic_M10_N128_K512) {
  TestParams params;
  params.M = 10;
  params.N = 128;
  params.K = 512;
  params.block_size = 32;
  params.has_zero_point = false;
  RunMatMulNBitsTest<4>(params);
}
#endif  // defined(_M_ARM64)

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B2_BS16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 16;
  params.has_zero_point = false;
  RunMatMulNBitsTest<2>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B2_BS16_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 16;
  params.has_zero_point = true;
  RunMatMulNBitsTest<2>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B2_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunMatMulNBitsTest<2>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B2_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  RunMatMulNBitsTest<2>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B2_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 64;
  params.has_zero_point = false;
  RunMatMulNBitsTest<2>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B2_BS64_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 64;
  params.has_zero_point = true;
  RunMatMulNBitsTest<2>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B4_BS16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 16;
  params.has_zero_point = false;
  RunMatMulNBitsTest<4>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B4_BS16_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 16;
  params.has_zero_point = true;
  RunMatMulNBitsTest<4>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B4_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunMatMulNBitsTest<4>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B4_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  RunMatMulNBitsTest<4>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B4_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 64;
  params.has_zero_point = false;
  RunMatMulNBitsTest<4>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B4_BS64_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 64;
  params.has_zero_point = true;
  RunMatMulNBitsTest<4>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B8_BS16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 16;
  params.has_zero_point = false;
  RunMatMulNBitsTest<8>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B8_BS16_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 16;
  params.has_zero_point = true;
  RunMatMulNBitsTest<8>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B8_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunMatMulNBitsTest<8>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B8_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  RunMatMulNBitsTest<8>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B8_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 64;
  params.has_zero_point = false;
  RunMatMulNBitsTest<8>(params, "htp");
}

TEST_F(QnnHTPBackendTests, MatMulNBits_M1_N2_K64_B8_BS64_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 64;
  params.block_size = 64;
  params.has_zero_point = true;
  RunMatMulNBitsTest<8>(params, "htp");
}

// QDQ MatMulNBits with UINT16 (UFIXED_POINT_16) activations/output.

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K64_B2_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<2, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K64_B2_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<2, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K128_B2_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<2, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K128_B2_BS64_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<2, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N64_K256_B2_BS128) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<2, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N64_K256_B2_BS128_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<2, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K64_B4_BS32) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto htp_arch = GetPlatformAttributes().htp_arch;

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, /*expect_native_bq=*/htp_arch >= QNN_HTP_DEVICE_ARCH_V81);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K64_B4_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K64_B4_BS16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 16;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N16_K64_B4_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 16;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K128_B4_BS64) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto htp_arch = GetPlatformAttributes().htp_arch;

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, /*expect_native_bq=*/htp_arch >= QNN_HTP_DEVICE_ARCH_V81);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K128_B4_BS64_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N64_K256_B4_BS128) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto htp_arch = GetPlatformAttributes().htp_arch;

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, /*expect_native_bq=*/htp_arch >= QNN_HTP_DEVICE_ARCH_V81);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N64_K256_B4_BS128_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K64_B8_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<8, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K64_B8_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<8, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K128_B8_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<8, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N32_K128_B8_BS64_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<8, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N64_K256_B8_BS128) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<8, uint16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_U16_M1_N64_K256_B8_BS128_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<8, uint16_t>(params, /*expect_native_bq=*/false);
}

// QDQ MatMulNBits with INT16 (SFIXED_POINT_16) activations/output.

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K64_B2_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<2, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K64_B2_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<2, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K128_B2_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<2, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K128_B2_BS64_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<2, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N64_K256_B2_BS128) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<2, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N64_K256_B2_BS128_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<2, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K64_B4_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<4, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K64_B4_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<4, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K128_B4_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<4, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K128_B4_BS64_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<4, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N64_K256_B4_BS128) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<4, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N64_K256_B4_BS128_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<4, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K64_B8_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<8, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K64_B8_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<8, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K128_B8_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<8, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N32_K128_B8_BS64_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<8, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N64_K256_B8_BS128) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = false;
  RunHtpQDQMatMulNBitsTest<8, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_QDQ_S16_M1_N64_K256_B8_BS128_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 64;
  params.K = 256;
  params.block_size = 128;
  params.has_zero_point = true;
  RunHtpQDQMatMulNBitsTest<8, int16_t>(params, /*expect_native_bq=*/false);
}

TEST_F(QnnHTPBackendTests, MatMulNBits_LPBQ_M1_N4_K64_B4_BS16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 4;
  params.K = 64;
  params.block_size = 16;
  params.has_zero_point = false;
  params.enable_lpbq = true;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, std::nullopt, ExpectedEPNodeAssignment::All, QDQTolerance(0.02f));
}

TEST_F(QnnHTPBackendTests, MatMulNBits_LPBQ_M1_N4_K128_B4_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 4;
  params.K = 128;
  params.block_size = 32;
  params.has_zero_point = false;
  params.enable_lpbq = true;
  RunHtpQDQMatMulNBitsTest<4, int16_t>(params, std::nullopt, ExpectedEPNodeAssignment::All, QDQTolerance(0.01f));
}

TEST_F(QnnHTPBackendTests, MatMulNBits_LPBQ_M1_N2_K128_B4_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 2;
  params.K = 128;
  params.block_size = 64;
  params.has_zero_point = false;
  params.enable_lpbq = true;
  RunHtpQDQMatMulNBitsTest<4, uint16_t>(params, std::nullopt, ExpectedEPNodeAssignment::All, QDQTolerance(0.01f));
}

TEST_F(QnnHTPBackendTests, MatMulNBits_LPBQ_M1_N8_K64_B4_BS32_ZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 8;
  params.K = 64;
  params.block_size = 32;
  params.has_zero_point = true;
  params.is_zp_symmetric = true;
  params.enable_lpbq = true;
  RunHtpQDQMatMulNBitsTest<4, int16_t>(params, std::nullopt, ExpectedEPNodeAssignment::All, QDQTolerance(0.02f));
}

// Should fallback to BW_FLOAT_BLOCK (bits=8)
TEST_F(QnnHTPBackendTests, MatMulNBits_LPBQ_M1_N32_K256_B8_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 32;
  params.K = 256;
  params.block_size = 32;
  params.has_zero_point = false;
  params.enable_lpbq = true;
  RunHtpQDQMatMulNBitsTest<8, int16_t>(params);
}

// Should fallback to BW_FLOAT_BLOCK (asymmetric zp)
TEST_F(QnnHTPBackendTests, MatMulNBits_LPBQ_M1_N4_K64_B4_BS16_AZP) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  TestParams params;
  params.M = 1;
  params.N = 4;
  params.K = 64;
  params.block_size = 16;
  params.has_zero_point = true;
  params.enable_lpbq = true;
  RunHtpQDQMatMulNBitsTest<4, int16_t>(params);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
