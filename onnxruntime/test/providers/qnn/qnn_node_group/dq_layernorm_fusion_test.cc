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
// possible), falling back to CPU EP.
GetTestModelFn BuildDQLayerNormSignFixupTestCase(bool x_signed, bool scale_signed, bool scale_per_channel) {
  return [x_signed, scale_signed, scale_per_channel](ModelTestBuilder& builder) -> void {
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

    // Bias (beta): plain SFIXED_POINT_32-style quantized bias via DequantizeLinear from int32,
    // matching what real QDQ tooling emits for LayerNorm bias. Values in [0, 0.5].
    std::vector<int32_t> bias_q_values(static_cast<size_t>(C));
    for (int64_t i = 0; i < C; ++i) {
      const float beta = 0.5f * (static_cast<float>(i) / static_cast<float>(C - 1));
      bias_q_values[static_cast<size_t>(i)] = static_cast<int32_t>(std::lround(beta / (kXScale * kScaleScale)));
    }
    builder.MakeInitializer<int32_t>("bias_q", {C}, bias_q_values);
    builder.AddDequantizeLinearNode<int32_t>("dq_bias", "bias_q", kXScale * kScaleScale, static_cast<int32_t>(0),
                                             "bias_f32");

    builder.AddNode("ln", "LayerNormalization", {x_f32_name, "scale_f32", "bias_f32"}, {"y_f32"}, kOnnxDomain,
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

}  // namespace

// Unsigned X (UFIXED_POINT_8) + signed scale (SFIXED_POINT_8): the exact combination the QNN HTP
// LayerNorm validator rejects. DQLayerNormFusion must resign the scale to UFIXED_POINT_8 so the
// node stays on HTP.
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_UnsignedX_SignedScale) {
  const std::filesystem::path json_dir = "DQLayerNormFusion_UnsignedX_SignedScale";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/false, /*scale_signed=*/true, /*scale_per_channel=*/false),
      opts, 17,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
}

// Reverse direction: signed X (SFIXED_POINT_8) + unsigned scale (UFIXED_POINT_8). Same mismatch,
// opposite sign; the fusion must resign the scale to SFIXED_POINT_8.
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_SignedX_UnsignedScale) {
  const std::filesystem::path json_dir = "DQLayerNormFusion_SignedX_UnsignedScale";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/true, /*scale_signed=*/false, /*scale_per_channel=*/false),
      opts, 17,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
}

// X and scale already share the same sign (both unsigned): the fusion must not fire (no mismatch
// to fix), and the default op-builder path already handles this combination directly.
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_Skip_AlreadyMatchingSign) {
  const std::filesystem::path json_dir = "DQLayerNormFusion_Skip_AlreadyMatchingSign";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/false, /*scale_signed=*/false, /*scale_per_channel=*/false),
      opts, 17,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
}

// Mismatched sign but per-channel (not per-tensor) scale quantization: the fusion cannot resign a
// per-channel scale offline via a single zero-point shift, so it rejects; the node falls back to
// CPU EP since the default op-builder path also can't emit a QNN-valid combination here.
TEST_F(QnnHTPBackendTests, DQLayerNormFusion_Skip_PerChannelScale) {
  const std::filesystem::path json_dir = "DQLayerNormFusion_Skip_PerChannelScale";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildDQLayerNormSignFixupTestCase(/*x_signed=*/false, /*scale_signed=*/true, /*scale_per_channel=*/true),
      opts, 17,
      EPVerificationParams{ExpectedEPNodeAssignment::None, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
