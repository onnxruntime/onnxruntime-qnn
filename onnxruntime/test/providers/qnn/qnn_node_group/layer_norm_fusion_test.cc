// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

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

// Builds the decomposed LayerNorm pattern:
//
//                    +--------------------------------------------+
//                    |                                            |
//                    v                                            |
//   [x] --> ReduceMean --> Sub --> Pow(2) --> ReduceMean --> Add(eps) --> Sqrt --> Div --> Mul(gamma) --> Add(beta) ==>
//                                  |                                               ^
//                                  |                                               |
//                                  +-----------------------------------------------+
GetTestModelFn BuildLayerNormFusionTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& axes,
    float epsilon,
    const std::vector<int64_t>& gamma_shape,
    const std::vector<float>& gamma_values,
    const std::vector<int64_t>& beta_shape,
    const std::vector<float>& beta_values) {
  return [=](ModelTestBuilder& builder) -> void {
    MakeTestInput<float>(builder, "input", input_def);

    builder.MakeInitializer<float>("gamma", gamma_shape, gamma_values);
    builder.MakeInitializer<float>("beta", beta_shape, beta_values);
    builder.MakeScalarInitializer<float>("pow_exp", 2.0f);
    builder.MakeScalarInitializer<float>("eps", epsilon);

    std::vector<ONNX_NAMESPACE::AttributeProto> rm_attrs;
    rm_attrs.push_back(test::MakeAttribute("axes", axes));
    rm_attrs.push_back(test::MakeAttribute("keepdims", static_cast<int64_t>(1)));

    builder.AddNode("rm1", "ReduceMean", {"input"}, {"rm1_out"}, "", rm_attrs);
    builder.AddNode("sub", "Sub", {"input", "rm1_out"}, {"sub_out"});
    builder.AddNode("pow", "Pow", {"sub_out", "pow_exp"}, {"pow_out"});
    builder.AddNode("rm2", "ReduceMean", {"pow_out"}, {"rm2_out"}, "", rm_attrs);
    builder.AddNode("add_eps", "Add", {"rm2_out", "eps"}, {"add_eps_out"});
    builder.AddNode("sqrt", "Sqrt", {"add_eps_out"}, {"sqrt_out"});
    builder.AddNode("div", "Div", {"sub_out", "sqrt_out"}, {"div_out"});
    builder.AddNode("mul_gamma", "Mul", {"div_out", "gamma"}, {"mul_out"});

    builder.MakeOutput("output");
    builder.AddNode("add_beta", "Add", {"mul_out", "beta"}, {"output"});
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

// 3D input [1, 8, 16], axes=[-1], 1D gamma/beta {16}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_3D_1D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_3D_1D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  RunQnnModelTest(
      BuildLayerNormFusionTestCase(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f,
          {C}, std::vector<float>(static_cast<size_t>(C), 1.0f),
          {C}, std::vector<float>(static_cast<size_t>(C), 0.0f)),
      opts, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// 3D input, padded gamma/beta {1, 1, 16} — fusion must squeeze to {16}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_3D_PaddedGammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_3D_PaddedGammaBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  RunQnnModelTest(
      BuildLayerNormFusionTestCase(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f,
          {1, 1, C}, std::vector<float>(static_cast<size_t>(C), 1.0f),
          {1, 1, C}, std::vector<float>(static_cast<size_t>(C), 0.0f)),
      opts, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// 4D input [1, 4, 8, 16], axes=[-1], 1D gamma/beta {16}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_4D_1D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_4D_1D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  RunQnnModelTest(
      BuildLayerNormFusionTestCase(
          TestInputDef<float>({1, 4, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f,
          {C}, std::vector<float>(static_cast<size_t>(C), 1.0f),
          {C}, std::vector<float>(static_cast<size_t>(C), 0.0f)),
      opts, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// Transformer shape [1, 256, 64], axes=[-1], 1D gamma/beta {64}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_TransformerShape) {
  const std::filesystem::path json_dir = "LayerNormFusion_TransformerShape";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 64;
  RunQnnModelTest(
      BuildLayerNormFusionTestCase(
          TestInputDef<float>({1, 256, 64}, false, -1.5f, 1.5f),
          {-1}, 1e-5f,
          {C}, std::vector<float>(static_cast<size_t>(C), 1.0f),
          {C}, std::vector<float>(static_cast<size_t>(C), 0.0f)),
      opts, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// gamma {1, 64, 1} with axes=[-1] — non-normalized dim is non-unit, fusion must be skipped.
TEST_F(QnnHTPBackendTests, LayerNormFusion_Skip_InvalidGammaShape) {
  const std::filesystem::path json_dir = "LayerNormFusion_Skip_InvalidGammaShape";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  RunQnnModelTest(
      BuildLayerNormFusionTestCase(
          TestInputDef<float>({1, 64, 16}, false, -1.0f, 1.0f),
          {-1}, 1e-5f,
          {1, 64, 1}, std::vector<float>(64, 1.0f),
          {C}, std::vector<float>(static_cast<size_t>(C), 0.0f)),
      opts, 13, ExpectedEPNodeAssignment::All, 1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
