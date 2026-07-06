// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

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

// Build the pattern:
//   input -> DynamicQuantizeLinear
//     (a_q,  a_scale, a_zp)
//   ConvInteger(a_q, W, [a_zp], [W_zp]) -> Cast(FLOAT) -> ci_out_f32
//   parallel_Mul(a_scale, b_scale_init) -> requant_Mul(ci_out_f32, scale_product)
//       -> (optional) Add(bias_init) -> output
//
// `include_bias` toggles the trailing Add. `per_channel_scale` uses a rank-4
// [1, C_out, 1, 1] B_scale; otherwise a scalar. `include_a_zp` / `include_b_zp` toggle
// whether ConvInteger consumes the corresponding zero-point input. `weight_signed` selects
// int8 weight (zp 0) or uint8 weight (symmetric zp 128); the underlying float weights are the
// same so output values match the int8 case within numerical tolerance.
// `depthwise` selects the QNN_OP_DEPTH_WISE_CONV_2D path: group = C_in = C_out so the fusion
// emits DepthWiseConv2d instead of Conv2d.
GetTestModelFn BuildDQConvIntegerFusionTestCase(bool include_bias,
                                                bool per_channel_scale,
                                                bool include_a_zp = true,
                                                bool include_b_zp = true,
                                                bool weight_signed = true,
                                                bool depthwise = false) {
  return [include_bias, per_channel_scale, include_a_zp, include_b_zp, weight_signed, depthwise](
             ModelTestBuilder& builder) -> void {
    constexpr int64_t N = 1;
    constexpr int64_t H = 8;
    constexpr int64_t W = 8;
    constexpr int64_t K = 3;
    constexpr int kUint8Zp = 128;  // symmetric zero-point used for uint8 weight tests

    // Depthwise: group = C_in = C_out so the fusion picks the DepthWiseConv2d path.
    const int64_t C_in = depthwise ? 4 : 3;
    const int64_t C_out = 4;
    const int64_t group = depthwise ? C_in : 1;
    // ConvInteger weight layout is [M, C/group, kH, kW]; for depthwise C/group == 1.
    const int64_t W_C = depthwise ? 1 : C_in;

    // Float input.
    auto input_def = TestInputDef<float>({N, C_in, H, W}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // DynamicQuantizeLinear: input -> (a_q, a_scale, a_zp)
    builder.AddNode("dql", "DynamicQuantizeLinear", {"input"},
                    {"a_q", "a_scale", "a_zp"});

    // Weight B (int8 or uint8 NCHW). For uint8 we shift by 128 around a symmetric zero-point so
    // the float-domain weight matches the int8 case for like-for-like accuracy comparisons.
    const size_t w_count = static_cast<size_t>(C_out * W_C * K * K);
    if (weight_signed) {
      std::vector<int8_t> b_values(w_count);
      for (size_t i = 0; i < w_count; ++i) {
        b_values[i] = static_cast<int8_t>((i % 7) - 3);  // small deterministic int8 values
      }
      builder.MakeInitializer<int8_t>("B", {C_out, W_C, K, K}, b_values);
      if (include_b_zp) {
        builder.MakeScalarInitializer<int8_t>("B_zp", static_cast<int8_t>(0));
      }
    } else {
      std::vector<uint8_t> b_values(w_count);
      for (size_t i = 0; i < w_count; ++i) {
        b_values[i] = static_cast<uint8_t>(static_cast<int>((i % 7) - 3) + kUint8Zp);
      }
      builder.MakeInitializer<uint8_t>("B", {C_out, W_C, K, K}, b_values);
      if (include_b_zp) {
        builder.MakeScalarInitializer<uint8_t>("B_zp", static_cast<uint8_t>(kUint8Zp));
      }
    }

    // Build the ConvInteger input list with empty strings for absent optional inputs.
    std::vector<std::string> conv_int_inputs;
    conv_int_inputs.push_back("a_q");
    conv_int_inputs.push_back("B");
    if (include_a_zp || include_b_zp) {
      conv_int_inputs.push_back(include_a_zp ? std::string("a_zp") : std::string());
    }
    if (include_b_zp) {
      conv_int_inputs.push_back("B_zp");
    }

    builder.AddNode("conv_int", "ConvInteger",
                    conv_int_inputs, {"ci_out"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("kernel_shape", std::vector<int64_t>{K, K}),
                     builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}),
                     builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                     builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}),
                     builder.MakeScalarAttribute("group", group)});

    // Cast int32 -> float.
    builder.AddNode("cast_int_to_float", "Cast", {"ci_out"}, {"ci_out_f32"}, kOnnxDomain,
                    {builder.MakeScalarAttribute("to",
                                                 static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});

    // B_scale: scalar (per-tensor) or [1, C_out, 1, 1] (per-channel, NCHW-broadcastable).
    if (per_channel_scale) {
      std::vector<float> bs_values(static_cast<size_t>(C_out));
      for (int64_t i = 0; i < C_out; ++i) {
        bs_values[static_cast<size_t>(i)] = 0.01f + 0.005f * static_cast<float>(i);
      }
      builder.MakeInitializer<float>("b_scale", {1, C_out, 1, 1}, bs_values);
    } else {
      builder.MakeScalarInitializer<float>("b_scale", 0.01f);
    }

    // parallel_Mul(a_scale, b_scale) -> scale_product.
    builder.AddNode("scale_mul", "Mul", {"a_scale", "b_scale"}, {"scale_product"});

    // requant_Mul(ci_out_f32, scale_product) -> requant_out.
    builder.AddNode("requant_mul", "Mul", {"ci_out_f32", "scale_product"}, {"requant_out"});

    const std::string final_name = include_bias ? std::string("output") : std::string("requant_out");
    if (include_bias) {
      // Bias shape [1, C_out, 1, 1] to broadcast correctly against NCHW conv output.
      std::vector<float> bias_values(static_cast<size_t>(C_out), 0.1f);
      builder.MakeInitializer<float>("bias", {1, C_out, 1, 1}, bias_values);
      builder.AddNode("bias_add", "Add", {"requant_out", "bias"}, {final_name});
    }

    builder.MakeOutput(final_name);
  };
}

// Build a graph with two parallel ConvIntegers fed by the SAME DynamicQuantizeLinear, each
// followed by its own Cast / parallel_Mul / requant_Mul (no bias). The two outputs are
// concatenated to keep both branches alive in the graph.
GetTestModelFn BuildSharedDqlTwoConvIntegersTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    constexpr int64_t N = 1;
    constexpr int64_t C_in = 3;
    constexpr int64_t H = 8;
    constexpr int64_t W = 8;
    constexpr int64_t C_out = 4;
    constexpr int64_t K = 3;

    auto input_def = TestInputDef<float>({N, C_in, H, W}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("dql", "DynamicQuantizeLinear", {"input"},
                    {"a_q", "a_scale", "a_zp"});

    auto add_branch = [&](const std::string& tag, float b_scale_value) {
      const std::string b_name = "B_" + tag;
      const std::string b_zp_name = "B_zp_" + tag;
      const std::string ci_out = "ci_out_" + tag;
      const std::string ci_out_f32 = "ci_out_f32_" + tag;
      const std::string b_scale = "b_scale_" + tag;
      const std::string scale_prod = "scale_product_" + tag;
      const std::string out = "out_" + tag;

      std::vector<int8_t> b_values(C_out * C_in * K * K);
      for (size_t i = 0; i < b_values.size(); ++i) {
        b_values[i] = static_cast<int8_t>((i % 5) - 2);
      }
      builder.MakeInitializer<int8_t>(b_name, {C_out, C_in, K, K}, b_values);
      builder.MakeScalarInitializer<int8_t>(b_zp_name, static_cast<int8_t>(0));

      builder.AddNode("conv_int_" + tag, "ConvInteger",
                      {"a_q", b_name, "a_zp", b_zp_name}, {ci_out}, kOnnxDomain,
                      {builder.MakeIntsAttribute("kernel_shape", std::vector<int64_t>{K, K}),
                       builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}),
                       builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                       builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}),
                       builder.MakeScalarAttribute("group", static_cast<int64_t>(1))});

      builder.AddNode("cast_" + tag, "Cast", {ci_out}, {ci_out_f32}, kOnnxDomain,
                      {builder.MakeScalarAttribute("to",
                                                   static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});

      // Distinct b_scale per branch so ORT's CSE pass cannot merge the two parallel_Muls.
      builder.MakeScalarInitializer<float>(b_scale, b_scale_value);
      builder.AddNode("scale_mul_" + tag, "Mul", {"a_scale", b_scale}, {scale_prod});
      builder.AddNode("requant_mul_" + tag, "Mul", {ci_out_f32, scale_prod}, {out});
      return out;
    };

    const std::string out_a = add_branch("a", 0.01f);
    const std::string out_b = add_branch("b", 0.013f);

    // Concatenate along channel axis to keep both fusions reachable from a graph output.
    builder.AddNode("concat", "Concat", {out_a, out_b}, {"output"}, kOnnxDomain,
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(1))});

    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

// Returns true if at least one QNN JSON graph file exists in `dump_dir`. Used to skip graph
// assertions when the test was not executed (e.g., skipped on architectures where FP16/FP32
// HTP is unavailable and no JSON dump is produced).
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

void RunFusionTestAndAssertFused(const std::filesystem::path& json_qnn_graph_dir,
                                 GetTestModelFn build_model,
                                 size_t expected_dequantize_count,
                                 size_t expected_add_count,
                                 float fp32_abs_err = 1e-3f,
                                 size_t expected_conv2d_count = 1,
                                 size_t expected_depthwise_conv2d_count = 0) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup =
      gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(build_model,
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(fp32_abs_err)});

  // RunQnnModelTest skips silently when the fixture's backend is unavailable (no ::testing::Test
  // GTEST_SKIP propagates back here). Detect that via the absence of any QNN JSON graph dump
  // and skip explicitly so CI sees a SKIPPED test instead of a green test that asserted nothing.
  if (!HasQnnJsonGraph(json_qnn_graph_dir)) {
    GTEST_SKIP() << "No QNN JSON graph dumped; HTP backend likely unavailable on this host.";
  }

  // The fused QNN graph should contain the rewritten ops.
  AssertOpInQnnGraph(json_qnn_graph_dir, "Conv2d", expected_conv2d_count);
  AssertOpInQnnGraph(json_qnn_graph_dir, "DepthWiseConv2d", expected_depthwise_conv2d_count);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Dequantize", expected_dequantize_count);
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseBinary", expected_add_count);
  AssertOpInQnnGraph(json_qnn_graph_dir, "ConvInteger", /*count=*/0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "DynamicQuantizeLinear", /*count=*/0);
}

}  // namespace

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_WithBias) {
  // Per-tensor B_scale (scalar) + Bias: expect Dequantize (weight) + ElementWiseAdd (bias).
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_WithBias",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/1);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_NoBias) {
  // Per-tensor B_scale (scalar), no Bias: expect Dequantize only.
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_NoBias",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/false, /*per_channel_scale=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/0);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_PerChannelBScale) {
  // Per-channel B_scale + Bias: weight is pre-dequantized offline, so no Dequantize op in the
  // QNN graph; bias still emits as ElementWiseAdd.
  // Tolerance is slightly looser than per-tensor: float activations vs. the reference's uint8
  // quantized activations introduces a small per-channel quantization discrepancy.
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_PerChannelBScale",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/true),
      /*expected_dequantize_count=*/0,
      /*expected_add_count=*/1,
      /*fp32_abs_err=*/2e-3f);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_PerChannelBScale_NoBias) {
  // Per-channel B_scale, no Bias: pre-dequantized weight + no ElementWiseAdd.
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_PerChannelBScale_NoBias",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/false, /*per_channel_scale=*/true),
      /*expected_dequantize_count=*/0,
      /*expected_add_count=*/0,
      /*fp32_abs_err=*/2e-3f);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_NoBZp) {
  // ConvInteger without B_zp input: defaults to 0, so the per-tensor weight-quant params have
  // a zero offset. The fusion still fires and emits the standard Dequantize + Conv2D path.
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_NoBZp",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/false,
                                       /*include_a_zp=*/true, /*include_b_zp=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/1);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_NoAZp_RejectsFusion) {
  // ConvInteger without A_zp: ConvInteger silently treats A_zp as 0, but the fused float Conv
  // takes the pre-DQL float input which still carries DQL's offset, so the rewrite would
  // change semantics. The fusion must decline. ConvInteger and DQL fall back to CPU EP, while
  // peripheral Cast/Mul nodes still land on QNN, so EP assignment is Some (not All). The
  // load-bearing check is that the QNN graph contains no Conv2d.
  std::filesystem::path json_dir{"DQConvIntegerFusion_NoAZp_RejectsFusion"};
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildDQConvIntegerFusionTestCase(/*include_bias=*/false,
                                                   /*per_channel_scale=*/false,
                                                   /*include_a_zp=*/false,
                                                   /*include_b_zp=*/true),
                  provider_options,
                  /*opset_version=*/13,
                  // Peripheral Cast/Mul on HTP fp16 introduces small rounding vs CPU EP reference.
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(1e-3f)});

  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "No QNN JSON graph dumped; HTP backend likely unavailable on this host.";
  }

  AssertOpInQnnGraph(json_dir, "Conv2d", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "ConvInteger", /*count=*/0);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_TwoConvIntegersShareDQL) {
  // Two ConvIntegers consume a_q (and a_scale via two parallel_Muls) from the same DQL. Both
  // fusions must fire and emit float QNN Conv nodes. The first fusion to be processed claims
  // DQL on behalf of its sibling; the second detects the existing claim and constructs its
  // Pattern with `dql=nullptr` so it does not double-claim. Keeping DQL in the QNN partition
  // is what allows the pattern walk-up to still succeed in the second GetCapability pass.
  std::filesystem::path json_dir{"DQConvIntegerFusion_TwoConvIntegersShareDQL"};
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildSharedDqlTwoConvIntegersTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});

  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "No QNN JSON graph dumped; HTP backend likely unavailable on this host.";
  }

  AssertOpInQnnGraph(json_dir, "Conv2d", /*count=*/2);
  AssertOpInQnnGraph(json_dir, "ConvInteger", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "DynamicQuantizeLinear", /*count=*/0);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_Uint8Weight_WithBias) {
  // uint8 weight with symmetric B_zp=128 + bias. Per-tensor B_scale: emits Dequantize(uint8) + Add.
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_Uint8Weight_WithBias",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/false,
                                       /*include_a_zp=*/true, /*include_b_zp=*/true,
                                       /*weight_signed=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/1);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_Uint8Weight_PerChannelBScale) {
  // uint8 weight with per-channel B_scale + bias: weight is pre-dequantized offline through the
  // unsigned dispatch path (no Dequantize op in the QNN graph).
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_Uint8Weight_PerChannelBScale",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/true,
                                       /*include_a_zp=*/true, /*include_b_zp=*/true,
                                       /*weight_signed=*/false),
      /*expected_dequantize_count=*/0,
      /*expected_add_count=*/1,
      /*fp32_abs_err=*/2e-3f);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_Uint8Weight_NoBZp) {
  // uint8 weight without B_zp input: ConvInteger defaults to zero. The dequantized float
  // weights are then `scale * [125..131]` (vs `scale * [-3..3]` when B_zp=128 is included),
  // so the conv output magnitude is ~40x larger and fp16 abs error scales with it.
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_Uint8Weight_NoBZp",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/false,
                                       /*include_a_zp=*/true, /*include_b_zp=*/false,
                                       /*weight_signed=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/1,
      /*fp32_abs_err=*/5e-2f);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_DepthWise) {
  // Depthwise conv (group == C_in == C_out): the fusion picks the QNN_OP_DEPTH_WISE_CONV_2D
  // path. Per-tensor B_scale + bias.
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_DepthWise",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/false,
                                       /*include_a_zp=*/true, /*include_b_zp=*/true,
                                       /*weight_signed=*/true, /*depthwise=*/true),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/1,
      /*fp32_abs_err=*/1e-3f,
      /*expected_conv2d_count=*/0,
      /*expected_depthwise_conv2d_count=*/1);
}

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_DepthWise_PerChannelBScale) {
  // Depthwise conv with per-channel B_scale: weight is pre-dequantized offline, so no
  // Dequantize op in the QNN graph; bias still emits as ElementWiseAdd.
  RunFusionTestAndAssertFused(
      "DQConvIntegerFusion_DepthWise_PerChannelBScale",
      BuildDQConvIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/true,
                                       /*include_a_zp=*/true, /*include_b_zp=*/true,
                                       /*weight_signed=*/true, /*depthwise=*/true),
      /*expected_dequantize_count=*/0,
      /*expected_add_count=*/1,
      /*fp32_abs_err=*/2e-3f,
      /*expected_conv2d_count=*/0,
      /*expected_depthwise_conv2d_count=*/1);
}

namespace {
// Build a graph with two ConvIntegers sharing one DQL where the first has B_zp as a constant
// initializer (fusible) but the second has B_zp as a runtime graph input. The runtime-input
// sibling fails IsConvIntegerStructurallyFusible's "B_zp must be a constant initializer"
// check while still passing ONNX schema validation. The fusion must reject the partition
// entirely so the well-formed sibling does not absorb DQL and orphan its sibling.
GetTestModelFn BuildSharedDqlOneSiblingRejectedTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    constexpr int64_t N = 1;
    constexpr int64_t C_in = 3;
    constexpr int64_t H = 8;
    constexpr int64_t W = 8;
    constexpr int64_t C_out = 4;
    constexpr int64_t K = 3;

    auto input_def = TestInputDef<float>({N, C_in, H, W}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("dql", "DynamicQuantizeLinear", {"input"},
                    {"a_q", "a_scale", "a_zp"});

    auto add_branch = [&](const std::string& tag, float b_scale_value,
                          bool b_zp_is_constant) {
      const std::string b_name = "B_" + tag;
      const std::string b_zp_name = "B_zp_" + tag;
      const std::string ci_out = "ci_out_" + tag;
      const std::string ci_out_f32 = "ci_out_f32_" + tag;
      const std::string b_scale = "b_scale_" + tag;
      const std::string scale_prod = "scale_product_" + tag;
      const std::string out = "out_" + tag;

      std::vector<int8_t> b_values(C_out * C_in * K * K);
      for (size_t i = 0; i < b_values.size(); ++i) {
        b_values[i] = static_cast<int8_t>((i % 5) - 2);
      }
      builder.MakeInitializer<int8_t>(b_name, {C_out, C_in, K, K}, b_values);
      if (b_zp_is_constant) {
        builder.MakeScalarInitializer<int8_t>(b_zp_name, static_cast<int8_t>(0));
      } else {
        builder.MakeInput<int8_t>(b_zp_name, /*shape=*/{}, std::vector<int8_t>{0});
      }

      builder.AddNode("conv_int_" + tag, "ConvInteger",
                      {"a_q", b_name, "a_zp", b_zp_name}, {ci_out}, kOnnxDomain,
                      {builder.MakeIntsAttribute("kernel_shape", std::vector<int64_t>{K, K}),
                       builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}),
                       builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                       builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}),
                       builder.MakeScalarAttribute("group", static_cast<int64_t>(1))});

      builder.AddNode("cast_" + tag, "Cast", {ci_out}, {ci_out_f32}, kOnnxDomain,
                      {builder.MakeScalarAttribute("to",
                                                   static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});

      builder.MakeScalarInitializer<float>(b_scale, b_scale_value);
      builder.AddNode("scale_mul_" + tag, "Mul", {"a_scale", b_scale}, {scale_prod});
      builder.AddNode("requant_mul_" + tag, "Mul", {ci_out_f32, scale_prod}, {out});
      return out;
    };

    // Branch a is well-formed (B_zp is a constant initializer). Branch b's B_zp is a runtime
    // graph input, which makes ConvInteger valid per ONNX schema but fails the fusion's
    // "B_zp must be a constant initializer" feasibility check. The fusion must refuse to
    // claim DQL for branch a so branch b is not stranded on CPU EP without a producing DQL.
    const std::string out_a = add_branch("a", 0.01f, /*b_zp_is_constant=*/true);
    const std::string out_b = add_branch("b", 0.013f, /*b_zp_is_constant=*/false);

    builder.AddNode("concat", "Concat", {out_a, out_b}, {"output"}, kOnnxDomain,
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(1))});

    builder.MakeOutput("output");
  };
}
}  // namespace

TEST_F(QnnHTPBackendTests, DQConvIntegerFusion_SiblingNotFusible_RejectsAll) {
  // When two ConvIntegers share one DQL but only one is structurally fusible, the fusion must
  // reject both: claiming DQL for the fusible sibling would strand the rejected sibling on CPU
  // EP without a producing DQL. Assertion: the QNN graph contains no Conv2d (i.e., neither
  // sibling was fused) and DynamicQuantizeLinear remains visible because both ConvIntegers
  // (and their DQL) fall back to CPU EP.
  std::filesystem::path json_dir{"DQConvIntegerFusion_SiblingNotFusible_RejectsAll"};
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  // Some peripheral nodes may still land on QNN; the load-bearing assertion is "no Conv2d".
  RunQnnModelTest(BuildSharedDqlOneSiblingRejectedTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(1e-3f)});

  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "No QNN JSON graph dumped; HTP backend likely unavailable on this host.";
  }

  AssertOpInQnnGraph(json_dir, "Conv2d", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "ConvInteger", /*count=*/0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
