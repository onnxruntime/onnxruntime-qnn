// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>
#include <cmath>
#include <optional>
#include <utility>
#include <array>
#include <memory>
#include <unordered_map>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "test/util/include/int4.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64)

namespace {

GetQDQTestCaseFn BuildLPBQGemmTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    // Define the test case for LPBQGemm fusion here
    constexpr int64_t input_channels = 16;
    constexpr int64_t output_channels = 16;
    constexpr int64_t blocks_per_axis = 4;
    const std::vector<int64_t> input_shape{1, input_channels};

    builder.graph_->set_name("lpbqgemm_fusion_graph");

    // input
    const auto input_def = TestInputDef<float>(input_shape, false, -0.5f, 0.5f);
    MakeTestInput<float>(builder, "input", input_def);

    // Activation QDQ: input -> Q -> DQ -> act_dq
    constexpr float act_scale = 0.00005509183756657876f;
    constexpr uint16_t act_zp = 23715;
    const std::string act_dq =
        AddQDQNodePair<uint16_t>(builder, "qdq_act", "input", act_scale, act_zp);

    // DequantizeLinear for Scale
    // scale_dq_in -> DQ (per-channel) -> scale_dq_out
    builder.MakeInitializer<uint8_t>("scale_dq_in", {blocks_per_axis, output_channels}, 1, 15);
    builder.MakeInitializer<float>("scale_dq_scale", {output_channels}, 0.01f, 0.02f);
    std::vector<uint8_t> dql_zero_points_data(output_channels, 0);
    builder.Make1DInitializer<uint8_t>("scale_dq_zp", dql_zero_points_data);

    builder.AddNode("DequantizeLinear",
                    "scale_dq",
                    {"scale_dq_in", "scale_dq_scale", "scale_dq_zp"},
                    {"scale_dq_out"},
                    kOnnxDomain,
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(1))});

    // Weight QuantizeLinear
    builder.MakeInitializer<float>("w_fp", {input_channels, output_channels}, -1.0f, 1.0f);

    std::vector<Int4x2> w_zp_data;
    const size_t num_storage_elems = static_cast<size_t>(blocks_per_axis * output_channels);
    w_zp_data.resize(Int4x2::CalcNumInt4Pairs(num_storage_elems));
    for (size_t i = 0; i < num_storage_elems; ++i) {
      const size_t r = i >> 1;
      const size_t c = i & 0x1;
      w_zp_data[r].SetElem(c, 0);
    }
    builder.MakeInitializer<Int4x2>("w_zp", {blocks_per_axis, output_channels}, w_zp_data);

    builder.AddNode("QuantizeLinear",
                    "w_q",
                    {"w_fp", "scale_dq_out", "w_zp"},
                    {"w_q_out"},
                    kOnnxDomain,
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)),
                     builder.MakeScalarAttribute("block_size", static_cast<int64_t>(4))});

    // Weight DequantizeLinear
    builder.AddNode("DequantizeLinear",
                    "w_dq",
                    {"w_q_out", "scale_dq_out", "w_zp"},
                    {"w_dq_out"},
                    kOnnxDomain,
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)),
                     builder.MakeScalarAttribute("block_size", static_cast<int64_t>(4))});

    // Gemm
    builder.MakeInitializer<float>("bias", {output_channels}, -1.0f, 1.0f);
    builder.AddNode("Gemm",
                    "gemm",
                    {act_dq, "w_dq_out", "bias"},
                    {"gemm_out"},
                    kOnnxDomain);

    // Output QDQ: gemm_out -> Q -> DQ -> output
    constexpr float out_scale = 0.00019595865160226822f;
    constexpr uint16_t out_zp = 31693;
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_out", "gemm_out", out_scale, out_zp);
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

}  // namespace

#if defined(_WIN32)
// Graph fails to compose on ARM64 Windows since QNN 2.37.0
TEST_F(QnnHTPBackendTests, DISABLED_LPBQGemmFusion) {
#else
TEST_F(QnnHTPBackendTests, LPBQGemmFusion) {
#endif
  ProviderOptions provider_options = GetProviderOptions();
  RunQnnModelTest(BuildLPBQGemmTestCase(),
                  provider_options,
                  /*opset_version=*/21,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::Some,
                  /*fp32_abs_err=*/1e-2f,
                  /*log_severity =*/OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
