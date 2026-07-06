// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cstdint>
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

// Builds the pattern:
//   input -> DynamicQuantizeLinear
//     (a_q, a_scale, a_zp)
//   MatMulInteger(a_q, B, [a_zp], [B_zp]) -> Cast(FLOAT) -> mm_out_f32
//   parallel_Mul(a_scale, b_scale_init) -> requant_Mul(mm_out_f32, scale_product)
//       -> (optional) Add(bias_init) -> output
//
// `include_bias` toggles the trailing Add. `per_channel_scale` uses [N] B_scale; otherwise a
// scalar. `include_a_zp` / `include_b_zp` toggle whether MatMulInteger consumes the
// corresponding zero-point input. `weight_signed` selects int8 weight (zp 0) or uint8 weight
// (symmetric zp 128); the underlying float weights are the same so output values match the
// int8 case within numerical tolerance.
GetTestModelFn BuildDQMatMulIntegerFusionTestCase(bool include_bias,
                                                  bool per_channel_scale,
                                                  bool include_a_zp = true,
                                                  bool include_b_zp = true,
                                                  bool weight_signed = true) {
  return [include_bias, per_channel_scale, include_a_zp, include_b_zp, weight_signed](
             ModelTestBuilder& builder) -> void {
    constexpr int64_t M = 4;
    constexpr int64_t K = 5;
    constexpr int64_t N = 3;
    constexpr int kUint8Zp = 128;  // symmetric zero-point used for uint8 weight tests

    auto input_def = TestInputDef<float>({M, K}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("dql", "DynamicQuantizeLinear", {"input"},
                    {"a_q", "a_scale", "a_zp"});

    // Weight B (int8 or uint8 [K, N]). For uint8 we shift by 128 around a symmetric zero-point
    // so the float-domain weight matches the int8 case for like-for-like accuracy comparisons.
    const size_t w_count = static_cast<size_t>(K * N);
    if (weight_signed) {
      std::vector<int8_t> b_values(w_count);
      for (size_t i = 0; i < w_count; ++i) {
        b_values[i] = static_cast<int8_t>((i % 7) - 3);
      }
      builder.MakeInitializer<int8_t>("B", {K, N}, b_values);
      if (include_b_zp) {
        builder.MakeScalarInitializer<int8_t>("B_zp", static_cast<int8_t>(0));
      }
    } else {
      std::vector<uint8_t> b_values(w_count);
      for (size_t i = 0; i < w_count; ++i) {
        b_values[i] = static_cast<uint8_t>(static_cast<int>((i % 7) - 3) + kUint8Zp);
      }
      builder.MakeInitializer<uint8_t>("B", {K, N}, b_values);
      if (include_b_zp) {
        builder.MakeScalarInitializer<uint8_t>("B_zp", static_cast<uint8_t>(kUint8Zp));
      }
    }

    std::vector<std::string> mmi_inputs;
    mmi_inputs.push_back("a_q");
    mmi_inputs.push_back("B");
    if (include_a_zp || include_b_zp) {
      mmi_inputs.push_back(include_a_zp ? std::string("a_zp") : std::string());
    }
    if (include_b_zp) {
      mmi_inputs.push_back("B_zp");
    }

    builder.AddNode("mm_int", "MatMulInteger", mmi_inputs, {"mm_out"});

    builder.AddNode("cast_int_to_float", "Cast", {"mm_out"}, {"mm_out_f32"}, kOnnxDomain,
                    {builder.MakeScalarAttribute("to",
                                                 static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});

    if (per_channel_scale) {
      std::vector<float> bs_values(static_cast<size_t>(N));
      for (int64_t i = 0; i < N; ++i) {
        bs_values[static_cast<size_t>(i)] = 0.01f + 0.005f * static_cast<float>(i);
      }
      builder.MakeInitializer<float>("b_scale", {N}, bs_values);
    } else {
      builder.MakeScalarInitializer<float>("b_scale", 0.01f);
    }

    builder.AddNode("scale_mul", "Mul", {"a_scale", "b_scale"}, {"scale_product"});
    builder.AddNode("requant_mul", "Mul", {"mm_out_f32", "scale_product"}, {"requant_out"});

    const std::string final_name = include_bias ? std::string("output") : std::string("requant_out");
    if (include_bias) {
      std::vector<float> bias_values(static_cast<size_t>(N), 0.1f);
      builder.MakeInitializer<float>("bias", {N}, bias_values);
      builder.AddNode("bias_add", "Add", {"requant_out", "bias"}, {final_name});
    }

    builder.MakeOutput(final_name);
  };
}

// Builds a graph with two parallel MatMulIntegers fed by the SAME DynamicQuantizeLinear, each
// followed by its own Cast / parallel_Mul / requant_Mul (no bias). The two outputs are
// concatenated to keep both branches alive in the graph. Distinct b_scale values prevent ORT's
// CSE pass from merging the two parallel_Muls.
GetTestModelFn BuildSharedDqlTwoMatMulIntegersTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    constexpr int64_t M = 4;
    constexpr int64_t K = 5;
    constexpr int64_t N = 3;

    auto input_def = TestInputDef<float>({M, K}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    builder.AddNode("dql", "DynamicQuantizeLinear", {"input"},
                    {"a_q", "a_scale", "a_zp"});

    auto add_branch = [&](const std::string& tag, float b_scale_value) {
      const std::string b_name = "B_" + tag;
      const std::string b_zp_name = "B_zp_" + tag;
      const std::string mm_out = "mm_out_" + tag;
      const std::string mm_out_f32 = "mm_out_f32_" + tag;
      const std::string b_scale = "b_scale_" + tag;
      const std::string scale_prod = "scale_product_" + tag;
      const std::string out = "out_" + tag;

      std::vector<int8_t> b_values(K * N);
      for (size_t i = 0; i < b_values.size(); ++i) {
        b_values[i] = static_cast<int8_t>((i % 5) - 2);
      }
      builder.MakeInitializer<int8_t>(b_name, {K, N}, b_values);
      builder.MakeScalarInitializer<int8_t>(b_zp_name, static_cast<int8_t>(0));

      builder.AddNode("mm_int_" + tag, "MatMulInteger",
                      {"a_q", b_name, "a_zp", b_zp_name}, {mm_out});

      builder.AddNode("cast_" + tag, "Cast", {mm_out}, {mm_out_f32}, kOnnxDomain,
                      {builder.MakeScalarAttribute("to",
                                                   static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});

      builder.MakeScalarInitializer<float>(b_scale, b_scale_value);
      builder.AddNode("scale_mul_" + tag, "Mul", {"a_scale", b_scale}, {scale_prod});
      builder.AddNode("requant_mul_" + tag, "Mul", {mm_out_f32, scale_prod}, {out});
      return out;
    };

    const std::string out_a = add_branch("a", 0.01f);
    const std::string out_b = add_branch("b", 0.013f);

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
                                 float fp32_abs_err = 1e-3f) {
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

  // If the test was skipped (e.g., FP16/FP32 HTP unavailable on this architecture) no JSON
  // graph dump is produced; skip the graph assertions in that case.
  if (!HasQnnJsonGraph(json_qnn_graph_dir)) {
    return;
  }

  AssertOpInQnnGraph(json_qnn_graph_dir, "MatMul", /*count=*/1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Dequantize", expected_dequantize_count);
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseBinary", expected_add_count);
  AssertOpInQnnGraph(json_qnn_graph_dir, "MatMulInteger", /*count=*/0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "DynamicQuantizeLinear", /*count=*/0);
}

}  // namespace

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_WithBias) {
  RunFusionTestAndAssertFused(
      "DQMatMulIntegerFusion_WithBias",
      BuildDQMatMulIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/1);
}

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_NoBias) {
  RunFusionTestAndAssertFused(
      "DQMatMulIntegerFusion_NoBias",
      BuildDQMatMulIntegerFusionTestCase(/*include_bias=*/false, /*per_channel_scale=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/0);
}

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_PerChannelBScale) {
  // Per-channel B_scale: weight is pre-dequantized offline, so no Dequantize op in the QNN
  // graph; bias still emits as ElementWiseAdd. Looser tolerance than per-tensor: float
  // activations vs the reference's uint8 quantized activations introduces a small per-channel
  // quantization discrepancy.
  RunFusionTestAndAssertFused(
      "DQMatMulIntegerFusion_PerChannelBScale",
      BuildDQMatMulIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/true),
      /*expected_dequantize_count=*/0,
      /*expected_add_count=*/1,
      /*fp32_abs_err=*/2e-3f);
}

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_PerChannelBScale_NoBias) {
  RunFusionTestAndAssertFused(
      "DQMatMulIntegerFusion_PerChannelBScale_NoBias",
      BuildDQMatMulIntegerFusionTestCase(/*include_bias=*/false, /*per_channel_scale=*/true),
      /*expected_dequantize_count=*/0,
      /*expected_add_count=*/0,
      /*fp32_abs_err=*/2e-3f);
}

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_NoBZp) {
  RunFusionTestAndAssertFused(
      "DQMatMulIntegerFusion_NoBZp",
      BuildDQMatMulIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/false,
                                         /*include_a_zp=*/true, /*include_b_zp=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/1);
}

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_NoAZp_RejectsFusion) {
  // Without A_zp, MatMulInteger silently uses 0; the fused float MatMul on the pre-DQL input
  // would diverge from the original semantics, so the fusion declines. MatMulInteger has no
  // standalone QNN op-builder, so it falls back to CPU EP; peripheral Cast/Mul nodes may still
  // land on QNN. Load-bearing assertion: no MatMul in the QNN graph.
  std::filesystem::path json_dir{"DQMatMulIntegerFusion_NoAZp_RejectsFusion"};
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildDQMatMulIntegerFusionTestCase(/*include_bias=*/false,
                                                     /*per_channel_scale=*/false,
                                                     /*include_a_zp=*/false,
                                                     /*include_b_zp=*/true),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(1e-3f)});

  if (!HasQnnJsonGraph(json_dir)) {
    return;
  }

  AssertOpInQnnGraph(json_dir, "MatMul", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "MatMulInteger", /*count=*/0);
}

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_TwoMatMulIntegersShareDQL) {
  // Two MatMulIntegers share one DQL. Both fusions must fire and emit float QNN MatMul nodes.
  // The first sibling claims DQL; the second detects the existing claim and constructs its
  // Pattern with `dql=nullptr` so it does not double-claim.
  std::filesystem::path json_dir{"DQMatMulIntegerFusion_TwoMatMulIntegersShareDQL"};
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(BuildSharedDqlTwoMatMulIntegersTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});

  if (!HasQnnJsonGraph(json_dir)) {
    return;
  }

  AssertOpInQnnGraph(json_dir, "MatMul", /*count=*/2);
  AssertOpInQnnGraph(json_dir, "MatMulInteger", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "DynamicQuantizeLinear", /*count=*/0);
}

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_Uint8Weight_WithBias) {
  // uint8 weight with symmetric B_zp=128 + bias. Per-tensor B_scale: emits Dequantize(uint8) + Add.
  RunFusionTestAndAssertFused(
      "DQMatMulIntegerFusion_Uint8Weight_WithBias",
      BuildDQMatMulIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/false,
                                         /*include_a_zp=*/true, /*include_b_zp=*/true,
                                         /*weight_signed=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/1);
}

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_Uint8Weight_PerChannelBScale) {
  // uint8 weight with per-channel B_scale + bias: weight is pre-dequantized offline through the
  // unsigned dispatch path (no Dequantize op in the QNN graph).
  RunFusionTestAndAssertFused(
      "DQMatMulIntegerFusion_Uint8Weight_PerChannelBScale",
      BuildDQMatMulIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/true,
                                         /*include_a_zp=*/true, /*include_b_zp=*/true,
                                         /*weight_signed=*/false),
      /*expected_dequantize_count=*/0,
      /*expected_add_count=*/1,
      /*fp32_abs_err=*/2e-3f);
}

TEST_F(QnnHTPBackendTests, DQMatMulIntegerFusion_Uint8Weight_NoBZp) {
  // uint8 weight without B_zp input: MatMulInteger defaults to zero. The dequantized float
  // weights are then `scale * [125..131]` (vs `scale * [-3..3]` when B_zp=128 is included),
  // so the matmul output magnitude is much larger and fp16 abs error scales with it.
  RunFusionTestAndAssertFused(
      "DQMatMulIntegerFusion_Uint8Weight_NoBZp",
      BuildDQMatMulIntegerFusionTestCase(/*include_bias=*/true, /*per_channel_scale=*/false,
                                         /*include_a_zp=*/true, /*include_b_zp=*/false,
                                         /*weight_signed=*/false),
      /*expected_dequantize_count=*/1,
      /*expected_add_count=*/1,
      /*fp32_abs_err=*/5e-2f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
