// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cmath>
#include <string>
#include <vector>

#include <gsl/gsl_util>
#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(_M_ARM64)

// QMoE (quantized Mixture-of-Experts, com.microsoft) is only supported on QNN GPU.
// Layout mirrors the gpt-oss style MoE block seen in real models: SwiGLU activation with
// activation_alpha=1.702 (sigmoid slope), activation_beta=1.0, a swiglu_limit clamp of 7.0,
// top-k routing (k=4) over normalize_routing_weights, and 4-bit block(32) quantized experts
// with no fc3 (fused gate_up SwiGLU) and no zero-points (symmetric quantization).

struct QMoETestParams {
  int64_t batch_size{1};  // batch dimension of the 3D input
  int64_t num_tokens;     // sequence length; routed rows are batch_size * num_tokens
  int64_t hidden_size;    // must be a multiple of block_size
  int64_t inter_size;     // per-expert intermediate size (pre-SwiGLU-fusion), multiple of block_size
  int64_t num_experts;
  int64_t k{4};
  int64_t block_size{32};
};

// Quantizes a single expert's [K, N] row-major weight matrix into the packed uint8 blob (N x
// K_blocks x blob_size, symmetric zp=8) + float32 scales (N x K_blocks) that
// QnnQuantParamsWrapper::BwBlockMapped()/AddQMoEWeight() expect, matching the same blockwise
// scheme used by MatMulNBits (see matmulnbits_test.cc's QuantizeDequantize<bits>).
template <int bits>
static void QuantizeExpertSlice(const float* raw_vals,  // [K, N] row-major
                                uint8_t* quant_vals,    // [N, K_blocks, blob_size] packed
                                float* scales,          // [N, K_blocks]
                                int32_t K,
                                int32_t N,
                                int32_t block_size) {
  QuantizeBlockwise<float, bits>(quant_vals,
                                 scales,
                                 nullptr,
                                 raw_vals,
                                 block_size,
                                 /*columnwise=*/true,
                                 K,
                                 N,
                                 N);
}

// Builds one QMoE expert-weight input (fc1 or fc2): quantizes `num_experts` independent [K, N]
// slices and concatenates them into the 3D {num_experts, N, K_blocks*blob_size} weight tensor
// and {num_experts, N, K_blocks} scales tensor that the QMoE op expects.
template <int bits>
static void AddQMoEExpertWeight(ModelTestBuilder& builder,
                                const QMoETestParams& params,
                                const std::string& weight_name,
                                const std::string& scale_name,
                                int64_t K,
                                int64_t N,
                                RandomValueGenerator& random,
                                std::vector<std::string>& input_names) {
  const int64_t k_blocks = K / params.block_size;
  const int64_t blob_size = (params.block_size * bits + 7) / 8;
  const int64_t num_experts = params.num_experts;

  std::vector<uint8_t> packed(static_cast<size_t>(num_experts * N * k_blocks * blob_size));
  std::vector<float> scales(static_cast<size_t>(num_experts * N * k_blocks));

  for (int64_t e = 0; e < num_experts; ++e) {
    std::vector<float> raw(random.Gaussian<float>(AsSpan({K, N}), 0.0f, 0.25f));
    QuantizeExpertSlice<bits>(raw.data(),
                              packed.data() + static_cast<size_t>(e * N * k_blocks * blob_size),
                              scales.data() + static_cast<size_t>(e * N * k_blocks),
                              static_cast<int32_t>(K),
                              static_cast<int32_t>(N),
                              static_cast<int32_t>(params.block_size));
  }

  auto weight_def = TestInputDef<uint8_t>({num_experts, N, k_blocks * blob_size}, true, packed);
  MakeTestInput<uint8_t>(builder, weight_name, weight_def);
  input_names.push_back(weight_name);

  auto scale_def = TestInputDef<float>({num_experts, N, k_blocks}, true, scales);
  MakeTestInput<float>(builder, scale_name, scale_def);
  input_names.push_back(scale_name);
}

// Builds the full 14-input QMoE (com.microsoft) node: input, router_probs, fc1/fc2 quantized
// weights + scales + bias, and empty fc3/zero_point slots -- matching a fused (swiglu_fusion=1)
// gate_up_proj + down_proj gpt-oss-style expert with no zero-points (symmetric 4-bit).
static void RunQMoETest(const QMoETestParams params,
                        ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                        TensorVerifier verifier = ElementwiseAbsoluteVerifier{0.1f}) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";
  provider_options["offload_graph_io_quantization"] = "0";

  auto model_builder = [&params](ModelTestBuilder& builder) {
    std::vector<std::string> input_names;
    RandomValueGenerator random{1234};

    // input: {1, num_tokens, hidden_size}
    std::vector<float> input_vals(
        random.Gaussian<float>(AsSpan({params.batch_size, params.num_tokens, params.hidden_size}), 0.0f, 0.25f));
    auto input_def = TestInputDef<float>({params.batch_size, params.num_tokens, params.hidden_size}, false, input_vals);
    MakeTestInput<float>(builder, "input", input_def);
    input_names.push_back("input");

    // router_probs: {num_tokens, num_experts} raw router logits.
    std::vector<float> router_vals(
        random.Gaussian<float>(AsSpan({params.batch_size * params.num_tokens, params.num_experts}), 0.0f, 1.0f));
    auto router_def = TestInputDef<float>({params.batch_size * params.num_tokens, params.num_experts}, false, router_vals);
    MakeTestInput<float>(builder, "router_probs", router_def);
    input_names.push_back("router_probs");

    // fc1 (gate_up_proj, SwiGLU-fused): N1 = 2 * inter_size, K1 = hidden_size.
    const int64_t fused_inter_size = 2 * params.inter_size;
    AddQMoEExpertWeight<4>(builder, params, "fc1_experts_weights", "fc1_scales",
                           /*K=*/params.hidden_size, /*N=*/fused_inter_size, random, input_names);
    std::vector<float> fc1_bias_vals(
        random.Gaussian<float>(AsSpan({params.num_experts, fused_inter_size}), 0.0f, 0.1f));
    auto fc1_bias_def = TestInputDef<float>({params.num_experts, fused_inter_size}, true, fc1_bias_vals);
    MakeTestInput<float>(builder, "fc1_experts_bias", fc1_bias_def);
    input_names.push_back("fc1_experts_bias");

    // fc2 (down_proj): N2 = hidden_size, K2 = inter_size.
    AddQMoEExpertWeight<4>(builder, params, "fc2_experts_weights", "fc2_scales",
                           /*K=*/params.inter_size, /*N=*/params.hidden_size, random, input_names);
    std::vector<float> fc2_bias_vals(
        random.Gaussian<float>(AsSpan({params.num_experts, params.hidden_size}), 0.0f, 0.1f));
    auto fc2_bias_def = TestInputDef<float>({params.num_experts, params.hidden_size}, true, fc2_bias_vals);
    MakeTestInput<float>(builder, "fc2_experts_bias", fc2_bias_def);
    input_names.push_back("fc2_experts_bias");

    // fc3 (unused, fused SwiGLU has no third projection) and zero-points (symmetric): all empty.
    for (const char* name : {"fc3_experts_weights", "fc3_scales", "fc3_experts_bias",
                             "fc1_zero_points", "fc2_zero_points", "fc3_zero_points"}) {
      input_names.push_back("");
      (void)name;
    }

    builder.MakeOutput("output");

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(builder.MakeScalarAttribute("k", params.k));
    attrs.push_back(builder.MakeStringAttribute("activation_type", std::string("swiglu")));
    attrs.push_back(builder.MakeScalarAttribute("block_size", params.block_size));
    attrs.push_back(builder.MakeScalarAttribute("expert_weight_bits", static_cast<int64_t>(4)));
    attrs.push_back(builder.MakeScalarAttribute("swiglu_fusion", static_cast<int64_t>(1)));
    attrs.push_back(builder.MakeScalarAttribute("swiglu_limit", 7.0f));
    attrs.push_back(builder.MakeScalarAttribute("normalize_routing_weights", static_cast<int64_t>(1)));
    attrs.push_back(builder.MakeScalarAttribute("use_sparse_mixer", static_cast<int64_t>(0)));
    attrs.push_back(builder.MakeScalarAttribute("activation_alpha", 1.7020000219345093f));
    attrs.push_back(builder.MakeScalarAttribute("activation_beta", 1.0f));

    builder.AddNode("QMoE", "QMoE", input_names, {"output"}, kMSDomain, attrs);
  };
  RunQnnModelTest(model_builder,
                  provider_options,
                  13,  // opset version for contrib ops
                  EPVerificationParams{expected_ep_assignment, verifier});
}

TEST_F(QnnGPUBackendTests, QMoE_Tokens1_H128_I128_E8_K4) {
  QMoETestParams params;
  params.num_tokens = 1;
  params.hidden_size = 128;
  params.inter_size = 128;
  params.num_experts = 8;
  params.k = 4;
  RunQMoETest(params);
}

TEST_F(QnnGPUBackendTests, QMoE_Tokens8_H128_I128_E8_K4) {
  QMoETestParams params;
  params.num_tokens = 8;
  params.hidden_size = 128;
  params.inter_size = 128;
  params.num_experts = 8;
  params.k = 4;
  RunQMoETest(params);
}

TEST_F(QnnGPUBackendTests, QMoE_Tokens128_H128_I128_E8_K4) {
  QMoETestParams params;
  params.num_tokens = 128;
  params.hidden_size = 128;
  params.inter_size = 128;
  params.num_experts = 8;
  params.k = 4;
  RunQMoETest(params);
}

TEST_F(QnnGPUBackendTests, QMoE_Batch128_Tokens1_H128_I128_E8_K4) {
  QMoETestParams params;
  params.batch_size = 128;
  params.num_tokens = 1;
  params.hidden_size = 128;
  params.inter_size = 128;
  params.num_experts = 8;
  params.k = 4;
  RunQMoETest(params);
}

// GPT-OSS production-shaped coverage, matching the GPU OpPackage MoE tests.
TEST_F(QnnGPUBackendTests, QMoE_GptOss_Tokens1_H2880_I2880_E32_K4) {
  QMoETestParams params;
  params.num_tokens = 1;
  params.hidden_size = 2880;
  params.inter_size = 2880;
  params.num_experts = 32;
  params.k = 4;
  RunQMoETest(params,
              ExpectedEPNodeAssignment::All,
              CosineSimilarityVerifier{0.999f});
}

TEST_F(QnnGPUBackendTests, QMoE_GptOss_Tokens128_H2880_I2880_E32_K4) {
  QMoETestParams params;
  params.num_tokens = 128;
  params.hidden_size = 2880;
  params.inter_size = 2880;
  params.num_experts = 32;
  params.k = 4;
  RunQMoETest(params,
              ExpectedEPNodeAssignment::All,
              CosineSimilarityVerifier{0.999f});
}

#endif  // defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif
