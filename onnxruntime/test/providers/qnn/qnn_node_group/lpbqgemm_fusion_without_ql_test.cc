// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>
#include <cmath>
#include <optional>
#include <utility>
#include <array>
#include <memory>
#include <unordered_map>
#include <type_traits>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64)

namespace {

template <typename INT4_DTYPE = Int4x2>
GetQDQTestCaseFn BuildLPBQGemmWithoutQLTestCase(uint8_t scale_min = 1, uint8_t scale_max = 17) {
  static_assert(std::is_same<INT4_DTYPE, Int4x2>::value || std::is_same<INT4_DTYPE, UInt4x2>::value,
                "Only INT4x2 or UINT4x2 data type is supported");

  return [scale_min, scale_max](ModelTestBuilder& builder) -> void {
    // Define the test case for LPBQGemm fusion without QuantizeLinear node
    const int64_t input_channels = 16;
    const int64_t output_channels = 16;
    const int64_t blocks_per_axis = 4;
    const std::vector<int64_t> input_shape{1, input_channels};
    auto input_def = TestInputDef<float>(input_shape, false, -0.5f, 0.5f);
    MakeTestInput<float>(builder, "input", input_def);

    // QuantizeLinear for Activation
    builder.MakeScalarInitializer<float>("act_ql_scale", 0.00005509183756657876f);
    builder.MakeScalarInitializer<uint16_t>("act_ql_zero_point", static_cast<uint16_t>(23715));
    builder.AddNode("act_ql", "QuantizeLinear",
                    {"input", "act_ql_scale", "act_ql_zero_point"}, {"act_ql_output"});

    // DequantizeLinear for Activation
    builder.MakeScalarInitializer<float>("act_dql_scale", 0.00005509183756657876f);
    builder.MakeScalarInitializer<uint16_t>("act_dql_zero_point", static_cast<uint16_t>(23715));
    builder.AddNode("act_dql", "DequantizeLinear",
                    {"act_ql_output", "act_dql_scale", "act_dql_zero_point"}, {"act_dql_output"});

    // DequantizeLinear for Scale
    builder.MakeInitializer<uint8_t>("scale_dql_input", {blocks_per_axis, output_channels}, scale_min, scale_max);
    builder.MakeInitializer<float>("scale_dql_scale", {output_channels}, 0.01f, 0.02f);
    std::vector<uint8_t> dql_zero_points_data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    builder.Make1DInitializer<uint8_t>("scale_dql_zero_point", dql_zero_points_data);
    builder.AddNode("scale_dql", "DequantizeLinear",
                    {"scale_dql_input", "scale_dql_scale", "scale_dql_zero_point"},
                    {"scale_dql_output"}, "",
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(1))});

    // Create quantized weight directly (skip QuantizeLinear)
    // We're creating a pre-quantized weight tensor that would normally be the output of QuantizeLinear
    std::vector<INT4_DTYPE> quantized_weight_data;
    size_t num_storage_elems = input_channels * output_channels;
    if (std::is_same<INT4_DTYPE, Int4x2>::value) {
      quantized_weight_data.resize(Int4x2::CalcNumInt4Pairs(num_storage_elems));
    } else {
      quantized_weight_data.resize(UInt4x2::CalcNumInt4Pairs(num_storage_elems));
    }
    for (size_t i = 0; i < num_storage_elems / 2; ++i) {
      // Set some pattern in the quantized weights
      quantized_weight_data[i].SetElem(0, i % 16);
      quantized_weight_data[i].SetElem(1, (i + 8) % 16);
    }
    builder.MakeInitializer<INT4_DTYPE>("w_quantized", {input_channels, output_channels}, quantized_weight_data);

    // DequantizeLinear for Weight (directly using the quantized weight)
    std::vector<INT4_DTYPE> zero_points_data;
    size_t num_zp_elems = blocks_per_axis * output_channels;
    if (std::is_same<INT4_DTYPE, Int4x2>::value) {
      zero_points_data.resize(Int4x2::CalcNumInt4Pairs(num_zp_elems));
    } else {
      zero_points_data.resize(UInt4x2::CalcNumInt4Pairs(num_zp_elems));
    }
    for (size_t i = 0; i < num_zp_elems; ++i) {
      size_t r = i >> 1;
      size_t c = i & 0x1;
      zero_points_data[r].SetElem(c, 0);
    }
    builder.MakeInitializer<INT4_DTYPE>("w_dql_zero_point", {blocks_per_axis, output_channels}, zero_points_data);
    builder.AddNode("w_dql", "DequantizeLinear",
                    {"w_quantized", "scale_dql_output", "w_dql_zero_point"},
                    {"w_dql_output"}, "",
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)),
                     builder.MakeScalarAttribute("block_size", static_cast<int64_t>(4))});

    // Gemm
    builder.MakeInitializer<float>("gemm_bias", {output_channels}, -1.0f, 1.0f);
    builder.AddNode("gemm", "Gemm", {"act_dql_output", "w_dql_output", "gemm_bias"}, {"gemm_output"});

    // QuantizeLinear for Output
    builder.MakeScalarInitializer<float>("output_ql_scale", 0.00019595865160226822f);
    builder.MakeScalarInitializer<uint16_t>("output_ql_zero_point", static_cast<uint16_t>(31693));
    builder.AddNode("output_ql", "QuantizeLinear",
                    {"gemm_output", "output_ql_scale", "output_ql_zero_point"},
                    {"output_ql_output"});

    // DequantizeLinear for Output
    builder.MakeScalarInitializer<float>("output_dql_scale", 0.00019595865160226822f);
    builder.MakeScalarInitializer<uint16_t>("output_dql_zero_point", static_cast<uint16_t>(31693));
    builder.MakeOutput("output_dql_output");
    builder.AddNode("output_dql", "DequantizeLinear",
                    {"output_ql_output", "output_dql_scale", "output_dql_zero_point"},
                    {"output_dql_output"});
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

}  // namespace

TEST_F(QnnHTPBackendTests, LPBQGemmFusionWithoutQL) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "LPBQGemmFusionWithoutQL";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  RunQnnModelTest(BuildLPBQGemmWithoutQLTestCase(),
                  provider_options,
                  /*opset_version=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(1e-2f)},
                  /*log_severity =*/OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);

  AssertOpInQnnGraph(json_qnn_graph_dir, "FullyConnected");
}

TEST_F(QnnHTPBackendTests, LPBQGemmFusionWithoutQL_UnsupportUInt4) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  ProviderOptions provider_options = GetProviderOptions();
  const auto graph_verifier = CreateEPNodeAssignmentVerifier(kQnnExecutionProvider,
                                                             {"act_ql", "act_dql", "gemm", "output_ql", "output_dql"});
  RunQnnModelTest(BuildLPBQGemmWithoutQLTestCase<UInt4x2>(),
                  provider_options,
                  /*opset_version=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some,
                                       ElementwiseAbsoluteVerifier(1e-2f),
                                       &graph_verifier},
                  /*log_severity =*/OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

TEST_F(QnnHTPBackendTests, LPBQGemmFusionWithoutQL_OutOfRangeScaleMin) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  ProviderOptions provider_options = GetProviderOptions();
  const auto graph_verifier = CreateEPNodeAssignmentVerifier(kQnnExecutionProvider,
                                                             {"act_ql", "act_dql", "gemm", "output_ql", "output_dql"});
  RunQnnModelTest(BuildLPBQGemmWithoutQLTestCase(/*scale_min=*/0, /*scale_max=*/1),
                  provider_options,
                  /*opset_version=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some,
                                       ElementwiseAbsoluteVerifier(1e-2f),
                                       &graph_verifier},
                  /*log_severity =*/OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

TEST_F(QnnHTPBackendTests, LPBQGemmFusionWithoutQL_OutOfRangeScaleMax) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  ProviderOptions provider_options = GetProviderOptions();
  const auto graph_verifier = CreateEPNodeAssignmentVerifier(kQnnExecutionProvider,
                                                             {"act_ql", "act_dql", "gemm", "output_ql", "output_dql"});
  RunQnnModelTest(BuildLPBQGemmWithoutQLTestCase(/*scale_min=*/17, /*scale_max=*/18),
                  provider_options,
                  /*opset_version=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some,
                                       ElementwiseAbsoluteVerifier(1e-2f),
                                       &graph_verifier},
                  /*log_severity =*/OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
