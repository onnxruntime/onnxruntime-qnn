// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

// QNN_QMOE_AVAILABLE is available from QNN API 2.42 (QAIRT 2.53).
// Defined here (duplicated from core/providers/qnn/builder/qnn_def.h) so this test file does not
// need to include that EP-private header.
#if QNN_API_VERSION_MAJOR > 2 || \
    (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 42)
#define QNN_QMOE_AVAILABLE
#endif

namespace onnxruntime {
namespace test {

#if defined(QNN_QMOE_AVAILABLE) && (defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC))

// QMoE (quantized Mixture-of-Experts, com.microsoft) is only supported on QNN GPU.
// The default configuration mirrors the large MoE block seen in real models: SwiGLU activation with
// activation_alpha=1.702 (sigmoid slope), activation_beta=1.0, a swiglu_limit clamp of 7.0,
// top-k routing (k=4) over normalize_routing_weights, and 4-bit block(32) quantized experts
// with no fc3 (fused gate_up SwiGLU) and no zero-points (symmetric quantization).

struct QMoETestParams {
  int64_t batch_size{1};  // batch dimension of the 3D input
  int64_t num_tokens{1};  // sequence length; routed rows are batch_size * num_tokens
  int64_t hidden_size{128};
  int64_t inter_size{128};  // per-expert intermediate size (pre-SwiGLU-fusion)
  int64_t num_experts{8};
  std::optional<int64_t> k{4};                   // nullopt: omit the attribute (schema default 1)
  std::optional<int64_t> expert_weight_bits{4};  // nullopt: omit the attribute (schema default 4)
  int64_t block_size{32};
  std::string activation_type{"swiglu"};
  int64_t swiglu_fusion{1};
  std::optional<float> swiglu_limit{7.0f};  // nullopt: omit the attribute (no clamp)
  bool fc1_bias{true};
  bool fc2_bias{true};
  bool input_2d{false};  // (num_tokens, hidden_size) instead of (batch, seq, hidden_size)
  // Optional input slots >= 14 to populate with a real tensor. Supported: 14 (router_weights) and
  // 15/16 (fc1/fc2_global_scale).
  std::vector<int> extra_optional_inputs{};
};

// Quantizes a single expert's [K, N] row-major weight matrix into the packed uint8 blob
// (N x K_blocks x blob_size, symmetric) and float32 scales (N x K_blocks) expected by QMoE.
// The packing matches MatMulNBits (see matmulnbits_test.cc's QuantizeDequantize<bits>).
template <int bits>
static void QuantizeExpertSlice(const float* raw_vals,  // [K, N] row-major
                                uint8_t* quant_vals,    // [N, K_blocks, blob_size] packed
                                float* scales,          // [N, K_blocks]
                                int32_t K,
                                int32_t N,
                                int32_t block_size) {
  QuantizeBlockwise<float, bits>(quant_vals, scales, /*zero_points=*/nullptr, raw_vals, block_size,
                                 /*columnwise=*/true, K, N, N);
}

// Builds one QMoE expert-weight input (fc1 or fc2): quantizes `num_experts` independent [K, N]
// slices into the 3D {num_experts, N, K_blocks*blob_size} weight tensor and the
// {num_experts, N, K_blocks} scales tensor.
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

  MakeTestInput<uint8_t>(builder, weight_name,
                         TestInputDef<uint8_t>({num_experts, N, k_blocks * blob_size}, true, packed));
  input_names.push_back(weight_name);

  MakeTestInput<float>(builder, scale_name, TestInputDef<float>({num_experts, N, k_blocks}, true, scales));
  input_names.push_back(scale_name);
}

static void AddQMoEExpertWeight(ModelTestBuilder& builder,
                                const QMoETestParams& params,
                                const std::string& weight_name,
                                const std::string& scale_name,
                                int64_t K,
                                int64_t N,
                                RandomValueGenerator& random,
                                std::vector<std::string>& input_names) {
  if (params.expert_weight_bits.value_or(4) == 8) {
    AddQMoEExpertWeight<8>(builder, params, weight_name, scale_name, K, N, random, input_names);
  } else {
    AddQMoEExpertWeight<4>(builder, params, weight_name, scale_name, K, N, random, input_names);
  }
}

static void AddQMoEBias(ModelTestBuilder& builder, const std::string& name, bool present, int64_t num_experts,
                        int64_t size, RandomValueGenerator& random, std::vector<std::string>& input_names) {
  if (!present) {
    input_names.emplace_back();
    return;
  }
  std::vector<float> vals(random.Gaussian<float>(AsSpan({num_experts, size}), 0.0f, 0.1f));
  MakeTestInput<float>(builder, name, TestInputDef<float>({num_experts, size}, true, vals));
  input_names.push_back(name);
}

// Builds a single QMoE (com.microsoft) node: input, router_probs, fc1/fc2 quantized weights, scales
// and biases, empty fc3/zero-point slots, and any requested optional inputs beyond slot 13.
static void RunQMoETest(const QMoETestParams& params,
                        ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                        TensorVerifier verifier = ElementwiseAbsoluteVerifier{0.1f}) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";
  provider_options["offload_graph_io_quantization"] = "0";

  auto model_builder = [&params](ModelTestBuilder& builder) {
    std::vector<std::string> input_names;
    RandomValueGenerator random{1234};
    const int64_t rows = params.batch_size * params.num_tokens;

    // input: (batch, seq, hidden) or (num_tokens, hidden)
    const std::vector<int64_t> input_shape = params.input_2d
                                                 ? std::vector<int64_t>{rows, params.hidden_size}
                                                 : std::vector<int64_t>{params.batch_size, params.num_tokens,
                                                                        params.hidden_size};
    std::vector<float> input_vals(random.Gaussian<float>(gsl::span<const int64_t>(input_shape), 0.0f, 0.25f));
    MakeTestInput<float>(builder, "input", TestInputDef<float>(input_shape, false, input_vals));
    input_names.push_back("input");

    // router_probs: (num_tokens, num_experts) raw router logits.
    std::vector<float> router_vals(random.Gaussian<float>(AsSpan({rows, params.num_experts}), 0.0f, 1.0f));
    MakeTestInput<float>(builder, "router_probs", TestInputDef<float>({rows, params.num_experts}, false, router_vals));
    input_names.push_back("router_probs");

    // fc1 (gate_up_proj): N1 = 2 * inter_size when SwiGLU is fused, K1 = hidden_size.
    const bool fused_swiglu = params.activation_type == "swiglu" && params.swiglu_fusion != 0;
    const int64_t fc1_out_size = fused_swiglu ? 2 * params.inter_size : params.inter_size;
    AddQMoEExpertWeight(builder, params, "fc1_experts_weights", "fc1_scales",
                        /*K=*/params.hidden_size, /*N=*/fc1_out_size, random, input_names);
    AddQMoEBias(builder, "fc1_experts_bias", params.fc1_bias, params.num_experts, fc1_out_size, random,
                input_names);

    // fc2 (down_proj): N2 = hidden_size, K2 = inter_size.
    AddQMoEExpertWeight(builder, params, "fc2_experts_weights", "fc2_scales",
                        /*K=*/params.inter_size, /*N=*/params.hidden_size, random, input_names);
    AddQMoEBias(builder, "fc2_experts_bias", params.fc2_bias, params.num_experts, params.hidden_size, random,
                input_names);

    // fc3_experts_weights, fc3_scales, fc3_experts_bias, fc1/fc2/fc3_zero_points: all empty.
    input_names.insert(input_names.end(), 6, std::string());

    // Optional inputs beyond slot 13 (router_weights, fc1/fc2_global_scale): empty unless requested.
    int last_extra = 13;
    for (int slot : params.extra_optional_inputs) {
      last_extra = std::max(last_extra, slot);
    }
    for (int slot = 14; slot <= last_extra; ++slot) {
      if (std::find(params.extra_optional_inputs.begin(), params.extra_optional_inputs.end(), slot) ==
          params.extra_optional_inputs.end()) {
        input_names.emplace_back();
        continue;
      }
      const std::string name = "optional_input_" + std::to_string(slot);
      const std::vector<int64_t> shape = slot == 14 ? std::vector<int64_t>{rows, params.num_experts}
                                                    : std::vector<int64_t>{params.num_experts};
      std::vector<float> vals(random.Uniform<float>(gsl::span<const int64_t>(shape), 0.5f, 1.0f));
      MakeTestInput<float>(builder, name, TestInputDef<float>(shape, slot != 14, vals));
      input_names.push_back(name);
    }

    builder.MakeOutput("output");

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    if (params.k.has_value()) {
      attrs.push_back(builder.MakeScalarAttribute("k", *params.k));
    }
    if (params.expert_weight_bits.has_value()) {
      attrs.push_back(builder.MakeScalarAttribute("expert_weight_bits", *params.expert_weight_bits));
    }
    if (params.swiglu_limit.has_value()) {
      attrs.push_back(builder.MakeScalarAttribute("swiglu_limit", *params.swiglu_limit));
    }
    attrs.push_back(builder.MakeStringAttribute("activation_type", params.activation_type));
    attrs.push_back(builder.MakeScalarAttribute("block_size", params.block_size));
    attrs.push_back(builder.MakeScalarAttribute("swiglu_fusion", params.swiglu_fusion));
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
  RunQMoETest(params);
}

// Batched (num_tokens > 1) QMoE requires an 8xx-tier GPU.
TEST_F(QnnGPUBackendTests, QMoE_Tokens8_H128_I128_E8_K4) {
  QMoETestParams params;
  params.num_tokens = 8;
  RunQMoETest(params);
}

// k and expert_weight_bits omitted: the schema defaults (k=1, expert_weight_bits=4) are supported.
TEST_F(QnnGPUBackendTests, QMoE_DefaultAttributes) {
  QMoETestParams params;
  params.k = std::nullopt;
  params.expert_weight_bits = std::nullopt;
  RunQMoETest(params);
}

// QNN GPU supports at most the first 14 QMoE inputs. A node that provides any input beyond them
// (router_weights, global scales, ...) must fall back instead of having that input silently dropped.
TEST_F(QnnGPUBackendTests, QMoE_InputsBeyond14_NotSupported) {
  for (int slot : {14, 15, 16}) {
    SCOPED_TRACE("optional input " + std::to_string(slot));
    QMoETestParams params;
    params.extra_optional_inputs = {slot};
    RunQMoETest(params, ExpectedEPNodeAssignment::None);
  }
}

// Configurations the QNN GPU QMoE validator rejects must fall back. Each case is a valid model for
// the CPU EP so the fallback result can still be verified.
TEST_F(QnnGPUBackendTests, QMoE_UnsupportedConfig_NotSupported) {
  struct UnsupportedCase {
    const char* name;
    std::function<void(QMoETestParams&)> apply;
  };
  const std::vector<UnsupportedCase> cases = {
      {"expert_weight_bits=8", [](QMoETestParams& p) { p.expert_weight_bits = 8; }},
      {"block_size=64", [](QMoETestParams& p) { p.block_size = 64; }},
      {"activation_type=gelu", [](QMoETestParams& p) {
         p.activation_type = "gelu";
         p.swiglu_fusion = 0;
       }},
      {"swiglu_limit absent", [](QMoETestParams& p) { p.swiglu_limit = std::nullopt; }},
      {"fc2_experts_bias absent", [](QMoETestParams& p) { p.fc2_bias = false; }},
      {"2D input", [](QMoETestParams& p) { p.input_2d = true; }},
      {"hidden_size=96", [](QMoETestParams& p) { p.hidden_size = 96; }},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    QMoETestParams params;
    test_case.apply(params);
    RunQMoETest(params, ExpectedEPNodeAssignment::None);
  }
}

// Larger multi-block coverage (hidden/inter span many 32-wide blocks, 16 experts).
TEST_F(QnnGPUBackendTests, QMoE_Large_Tokens1_H1024_I1024_E16_K4) {
  QMoETestParams params;
  params.num_tokens = 1;
  params.hidden_size = 1024;
  params.inter_size = 1024;
  params.num_experts = 16;
  RunQMoETest(params, ExpectedEPNodeAssignment::All, CosineSimilarityVerifier{0.999f});
}

// Batched (num_tokens > 1) QMoE requires an 8xx-tier GPU.
TEST_F(QnnGPUBackendTests, QMoE_Large_Tokens8_H1024_I1024_E16_K4) {
  QMoETestParams params;
  params.num_tokens = 8;
  params.hidden_size = 1024;
  params.inter_size = 1024;
  params.num_experts = 16;
  RunQMoETest(params, ExpectedEPNodeAssignment::All, CosineSimilarityVerifier{0.999f});
}

#endif  // defined(QNN_QMOE_AVAILABLE) && (defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC))

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
