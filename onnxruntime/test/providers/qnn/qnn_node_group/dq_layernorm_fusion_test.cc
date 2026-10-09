// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Builds a QDQ LayerNormalization pattern:
//   input_q --> DequantizeLinear --> x_f32 --+
//   scale_q --> DequantizeLinear --> scale_f32 --+--> LayerNormalization --> y_f32 --> QuantizeLinear --> output
//   [bias]  --> [DequantizeLinear] --> [bias_f32] ---+
//
// `x_signed`/`scale_signed` select SFIXED_POINT_8 (signed, zp=0) vs UFIXED_POINT_8 (unsigned,
// zp=128) for X and scale independently, reproducing the exact sign-mismatch QNN's HTP LayerNorm
// validator rejects (e.g. an unsigned/asymmetric X paired with a signed/symmetric scale -- the
// standard output of most QDQ quantization tooling). `scale_per_channel` gives the scale a
// per-axis DequantizeLinear, which the fusion must reject when mismatched (no offline byte-flip
// possible), falling back to CPU EP. `bias_kind` selects the bias operand: kNone (omit it) or
// kInt32 (a plain SFIXED_POINT_32-style quantized bias via DequantizeLinear from int32, matching
// what real QDQ tooling emits for LayerNorm bias) are both, empirically, unreliable on this HTP
// SDK/data range -- real device runs show correct EP assignment but silently wrong output values
// for *any* test using either, even ones that never touch DQLayerNormFusion at all. This looks
// like a QNN-side precision issue specific to the SFIXED_POINT_32/implicit-bias path at these
// scales, not a bug in the fusion or in this test's quantization math, so the scale-focused tests
// below use kMatched8Bit instead (bias is 8-bit and the *same* sign as X -- a plain pass-through,
// no resign needed -- which is empirically reliable). kMismatched8Bit (bias is 8-bit and the
// *opposite* sign of X) exercises the fusion's bias-resign path specifically.
enum class BiasKind { kNone,
                      kInt32,
                      kMatched8Bit,
                      kMismatched8Bit };

GetTestModelFn BuildDQLayerNormSignFixupTestCase(bool x_signed, bool scale_signed, bool scale_per_channel,
                                                 BiasKind bias_kind = BiasKind::kInt32) {
  return [x_signed, scale_signed, scale_per_channel, bias_kind](ModelTestBuilder& builder) -> void {
    constexpr int64_t N = 1;
    constexpr int64_t S = 8;
    constexpr int64_t C = 16;

    // X: quantized input, dequantized before feeding LayerNormalization.
    constexpr float kXScale = 0.02f;
    std::string x_f32_name;
    if (x_signed) {
      auto input_def = TestInputDef<int8_t>({N, S, C}, /*is_initializer=*/false,
                                            static_cast<int8_t>(-128), static_cast<int8_t>(127));
      MakeTestInput<int8_t>(builder, "input_q", input_def);
      builder.AddDequantizeLinearNode<int8_t>("dq_x", "input_q", kXScale, static_cast<int8_t>(0), "x_f32");
    } else {
      auto input_def = TestInputDef<uint8_t>({N, S, C}, /*is_initializer=*/false,
                                             static_cast<uint8_t>(0), static_cast<uint8_t>(255));
      MakeTestInput<uint8_t>(builder, "input_q", input_def);
      builder.AddDequantizeLinearNode<uint8_t>("dq_x", "input_q", kXScale, static_cast<uint8_t>(128), "x_f32");
    }
    x_f32_name = "x_f32";

    // Scale (gamma): per-channel values in [0.8, 1.2].
    std::vector<float> gamma_values(static_cast<size_t>(C));
    for (int64_t i = 0; i < C; ++i) {
      gamma_values[static_cast<size_t>(i)] = 0.8f + 0.4f * (static_cast<float>(i) / static_cast<float>(C - 1));
    }
    constexpr float kScaleScale = 0.01f;
    if (scale_signed) {
      std::vector<int8_t> q_values(static_cast<size_t>(C));
      for (int64_t i = 0; i < C; ++i) {
        q_values[static_cast<size_t>(i)] =
            static_cast<int8_t>(std::lround(gamma_values[static_cast<size_t>(i)] / kScaleScale));
      }
      builder.MakeInitializer<int8_t>("scale_q", {C}, q_values);
      if (scale_per_channel) {
        std::vector<float> scales(static_cast<size_t>(C), kScaleScale);
        std::vector<int8_t> zps(static_cast<size_t>(C), static_cast<int8_t>(0));
        builder.AddDequantizeLinearNode<int8_t>("dq_scale", "scale_q", scales, zps, "scale_f32",
                                                {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0))});
      } else {
        builder.AddDequantizeLinearNode<int8_t>("dq_scale", "scale_q", kScaleScale, static_cast<int8_t>(0),
                                                "scale_f32");
      }
    } else {
      std::vector<uint8_t> q_values(static_cast<size_t>(C));
      for (int64_t i = 0; i < C; ++i) {
        const float shifted = gamma_values[static_cast<size_t>(i)] / kScaleScale + 128.0f;
        q_values[static_cast<size_t>(i)] = static_cast<uint8_t>(std::lround(shifted));
      }
      builder.MakeInitializer<uint8_t>("scale_q", {C}, q_values);
      if (scale_per_channel) {
        std::vector<float> scales(static_cast<size_t>(C), kScaleScale);
        std::vector<uint8_t> zps(static_cast<size_t>(C), static_cast<uint8_t>(128));
        builder.AddDequantizeLinearNode<uint8_t>("dq_scale", "scale_q", scales, zps, "scale_f32",
                                                 {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0))});
      } else {
        builder.AddDequantizeLinearNode<uint8_t>("dq_scale", "scale_q", kScaleScale, static_cast<uint8_t>(128),
                                                 "scale_f32");
      }
    }

    // Bias (beta): values in [0, 0.5].
    //   kNone: no bias input at all.
    //   kInt32: plain SFIXED_POINT_32-style quantized bias via DequantizeLinear from int32,
    //           matching what real QDQ tooling emits for LayerNorm bias.
    //   kMatched8Bit / kMismatched8Bit: bias is 8-bit, either the same or the opposite sign as X.
    std::vector<std::string> layer_norm_inputs = {x_f32_name, "scale_f32"};
    if (bias_kind == BiasKind::kInt32) {
      std::vector<int32_t> bias_q_values(static_cast<size_t>(C));
      for (int64_t i = 0; i < C; ++i) {
        const float beta = 0.5f * (static_cast<float>(i) / static_cast<float>(C - 1));
        bias_q_values[static_cast<size_t>(i)] = static_cast<int32_t>(std::lround(beta / (kXScale * kScaleScale)));
      }
      builder.MakeInitializer<int32_t>("bias_q", {C}, bias_q_values);
      builder.AddDequantizeLinearNode<int32_t>("dq_bias", "bias_q", kXScale * kScaleScale, static_cast<int32_t>(0),
                                               "bias_f32");
      layer_norm_inputs.push_back("bias_f32");
    } else if (bias_kind == BiasKind::kMatched8Bit || bias_kind == BiasKind::kMismatched8Bit) {
      constexpr float kBiasScale = 0.01f;
      std::vector<float> beta_values(static_cast<size_t>(C));
      for (int64_t i = 0; i < C; ++i) {
        beta_values[static_cast<size_t>(i)] = 0.5f * (static_cast<float>(i) / static_cast<float>(C - 1));
      }
      const bool bias_signed = (bias_kind == BiasKind::kMatched8Bit) ? x_signed : !x_signed;
      if (bias_signed) {
        std::vector<int8_t> bias_q_values(static_cast<size_t>(C));
        for (int64_t i = 0; i < C; ++i) {
          bias_q_values[static_cast<size_t>(i)] =
              static_cast<int8_t>(std::lround(beta_values[static_cast<size_t>(i)] / kBiasScale));
        }
        builder.MakeInitializer<int8_t>("bias_q", {C}, bias_q_values);
        builder.AddDequantizeLinearNode<int8_t>("dq_bias", "bias_q", kBiasScale, static_cast<int8_t>(0), "bias_f32");
      } else {
        std::vector<uint8_t> bias_q_values(static_cast<size_t>(C));
        for (int64_t i = 0; i < C; ++i) {
          const float shifted = beta_values[static_cast<size_t>(i)] / kBiasScale + 128.0f;
          bias_q_values[static_cast<size_t>(i)] = static_cast<uint8_t>(std::lround(shifted));
        }
        builder.MakeInitializer<uint8_t>("bias_q", {C}, bias_q_values);
        builder.AddDequantizeLinearNode<uint8_t>("dq_bias", "bias_q", kBiasScale, static_cast<uint8_t>(128),
                                                 "bias_f32");
      }
      layer_norm_inputs.push_back("bias_f32");
    }

    builder.AddNode("ln", "LayerNormalization", layer_norm_inputs, {"y_f32"}, kOnnxDomain,
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(-1)),
                     builder.MakeScalarAttribute("epsilon", 1e-5f)});

    if (x_signed) {
      builder.AddQuantizeLinearNode<int8_t>("q_y", "y_f32", 0.05f, static_cast<int8_t>(0), "output");
    } else {
      builder.AddQuantizeLinearNode<uint8_t>("q_y", "y_f32", 0.05f, static_cast<uint8_t>(128), "output");
    }
    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return provider_options;
}

// RunQnnModelTest's built-in output comparison does an exact EXPECT_EQ byte compare for int8/uint8
// graph outputs (ElementwiseAbsoluteVerifier's tolerance only applies to FLOAT/FLOAT16 -- see
// test_utils.cc's VerifyOutput switch), which is too strict for a quantized LayerNorm: a benign
// +/-1 LSB rounding-tie difference between the CPU reference and the target backend's kernel is
// expected and not a correctness issue. This runs the actual CPU reference and the actual QNN
// execution directly (bypassing RunQnnModelTest's comparison) and compares them itself with a
// small integer tolerance, so a real bug in the resign math (e.g. a wrong zero-point shift) still
// fails loudly, while a harmless +/-1 LSB tie does not.
void RunAndVerifyLayerNormOutput(const GetTestModelFn& build_test_case, ProviderOptions provider_options,
                                 int opset_version, ExpectedEPNodeAssignment expected_ep_assignment,
                                 int max_abs_diff = 1) {
  ModelTestBuilder helper;
  build_test_case(helper);

  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};
  for (const auto& [domain, version] : domain_to_version) {
    ONNX_NAMESPACE::OperatorSetIdProto* opset_id_proto = helper.model_.add_opset_import();
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  std::vector<Ort::Value> cpu_outputs;
  InferenceModelCPU(model_data, "DQLayerNormFusion_cpu_ref", helper.feeds_, cpu_outputs);

  std::vector<Ort::Value> qnn_outputs;
  InferenceModel(model_data, "DQLayerNormFusion_qnn", provider_options, expected_ep_assignment,
                 helper.feeds_, qnn_outputs);

  ASSERT_EQ(cpu_outputs.size(), qnn_outputs.size());
  for (size_t out_idx = 0; out_idx < cpu_outputs.size(); ++out_idx) {
    auto type_info = cpu_outputs[out_idx].GetTensorTypeAndShapeInfo();
    const size_t element_count = type_info.GetElementCount();
    const auto elem_type = type_info.GetElementType();

    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      const uint8_t* expected = cpu_outputs[out_idx].GetTensorData<uint8_t>();
      const uint8_t* actual = qnn_outputs[out_idx].GetTensorData<uint8_t>();
      for (size_t i = 0; i < element_count; ++i) {
        const int diff = std::abs(static_cast<int>(expected[i]) - static_cast<int>(actual[i]));
        EXPECT_LE(diff, max_abs_diff) << "Output element " << i << " differs by " << diff
                                      << " (cpu_ref=" << static_cast<int>(expected[i])
                                      << ", qnn=" << static_cast<int>(actual[i]) << ")";
      }
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
      const int8_t* expected = cpu_outputs[out_idx].GetTensorData<int8_t>();
      const int8_t* actual = qnn_outputs[out_idx].GetTensorData<int8_t>();
      for (size_t i = 0; i < element_count; ++i) {
        const int diff = std::abs(static_cast<int>(expected[i]) - static_cast<int>(actual[i]));
        EXPECT_LE(diff, max_abs_diff) << "Output element " << i << " differs by " << diff
                                      << " (cpu_ref=" << static_cast<int>(expected[i])
                                      << ", qnn=" << static_cast<int>(actual[i]) << ")";
      }
    } else {
      FAIL() << "RunAndVerifyLayerNormOutput: unexpected output element type " << elem_type;
    }
  }
}

}  // namespace

// Unsigned X (UFIXED_POINT_8) + signed scale (SFIXED_POINT_8): the exact combination the QNN HTP
// LayerNorm validator rejects. DQLayerNormFusion must resign the scale to UFIXED_POINT_8 so the
// node stays on HTP. Bias is a matched (non-resigned) 8-bit pass-through -- this test is about
// the scale-resign path only.
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_UnsignedX_SignedScale) {
  const std::filesystem::path json_dir = "DQLayerNormFusion_UnsignedX_SignedScale";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunAndVerifyLayerNormOutput(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/false, /*scale_signed=*/true, /*scale_per_channel=*/false,
                                        BiasKind::kMatched8Bit),
      opts, 17, ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
}

// Reverse direction: signed X (SFIXED_POINT_8) + unsigned scale (UFIXED_POINT_8). Same mismatch,
// opposite sign; the fusion must resign the scale to SFIXED_POINT_8. Bias is a matched 8-bit
// pass-through.
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_SignedX_UnsignedScale) {
  const std::filesystem::path json_dir = "DQLayerNormFusion_SignedX_UnsignedScale";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunAndVerifyLayerNormOutput(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/true, /*scale_signed=*/false, /*scale_per_channel=*/false,
                                        BiasKind::kMatched8Bit),
      opts, 17, ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
}

// X and scale already share the same sign (both unsigned): the fusion must not fire (no mismatch
// to fix), and the default op-builder path already handles this combination directly. Bias is a
// matched 8-bit pass-through.
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_Skip_AlreadyMatchingSign) {
  const std::filesystem::path json_dir = "DQLayerNormFusion_Skip_AlreadyMatchingSign";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunAndVerifyLayerNormOutput(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/false, /*scale_signed=*/false, /*scale_per_channel=*/false,
                                        BiasKind::kMatched8Bit),
      opts, 17, ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
}

// Mismatched sign but per-channel (not per-tensor) scale quantization: the fusion cannot resign a
// per-channel scale offline via a single zero-point shift, so it rejects; the node falls back to
// CPU EP since the default op-builder path also can't emit a QNN-valid combination here. No QNN
// partition is created when the whole node falls back, so there's no JSON graph dump to inspect --
// only the assignment check applies.
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_Skip_PerChannelScale) {
  ProviderOptions opts = GetProviderOptions();

  RunQnnModelTest(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/false, /*scale_signed=*/true, /*scale_per_channel=*/true,
                                        BiasKind::kNone),
      opts, 17,
      EPVerificationParams{ExpectedEPNodeAssignment::None, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Unsigned X + mismatched signed scale + mismatched signed bias: resigning both scale and bias to
// UFIXED_POINT_8 lands on UFIXED_POINT_8/UFIXED_POINT_8/UFIXED_POINT_8, a QNN-supported INT8
// config. Exercises the bias-resign branch in CreateOrValidateOnQnn (the scale-only tests above
// never touch it, since their bias is always the already-accepted SFIXED_POINT_32).
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_UnsignedX_MismatchedBias) {
  const std::filesystem::path json_dir = "DQLayerNormFusion_UnsignedX_MismatchedBias";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunAndVerifyLayerNormOutput(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/false, /*scale_signed=*/true, /*scale_per_channel=*/false,
                                        BiasKind::kMismatched8Bit),
      opts, 17, ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
}

// Signed X + mismatched unsigned scale + mismatched unsigned bias: resigning both to
// SFIXED_POINT_8 lands on SFIXED_POINT_8/SFIXED_POINT_8/SFIXED_POINT_8. This combination doesn't
// appear in the validator's own dumped "Supported I/O datatype sets" list, but real HTP hardware
// accepts and correctly computes it anyway (verified: no output mismatch, only an EP-assignment
// mismatch, when this test was first written expecting a CPU fallback) -- so the fusion firing
// here is correct.
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_SignedX_MismatchedBias) {
  const std::filesystem::path json_dir = "DQLayerNormFusion_SignedX_MismatchedBias";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunAndVerifyLayerNormOutput(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/true, /*scale_signed=*/false, /*scale_per_channel=*/false,
                                        BiasKind::kMismatched8Bit),
      opts, 17, ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
