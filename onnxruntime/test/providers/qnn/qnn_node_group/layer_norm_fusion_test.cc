// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

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
template <typename T>
GetTestModelFn BuildLayerNormFusionTestCase(
    const TestInputDef<T>& input_def,
    const std::vector<int64_t>& axes,
    T epsilon,
    std::optional<std::reference_wrapper<const TestInputDef<T>>> gamma_def,
    std::optional<std::reference_wrapper<const TestInputDef<T>>> beta_def) {
  return [=](ModelTestBuilder& builder) -> void {
    MakeTestInput<T>(builder, "input", input_def);
    if (gamma_def.has_value()) {
      MakeTestInput<T>(builder, "gamma", gamma_def.value());
    }
    if (beta_def.has_value()) {
      MakeTestInput<T>(builder, "beta", beta_def.value());
    }
    builder.MakeOutput("output");
    builder.MakeScalarInitializer<T>("pow_exp", static_cast<T>(2.0f));
    builder.MakeScalarInitializer<T>("eps", epsilon);

    std::vector<ONNX_NAMESPACE::AttributeProto> rm_attrs;
    rm_attrs.push_back(test::MakeAttribute("axes", axes));
    rm_attrs.push_back(test::MakeAttribute("keepdims", static_cast<int64_t>(1)));

    const std::string div_out_name = (gamma_def.has_value() || beta_def.has_value()) ? "div_out" : "output";
    const std::string mul_out_name = beta_def.has_value() ? "mul_out" : "output";

    builder.AddNode("rm1", "ReduceMean", {"input"}, {"rm1_out"}, "", rm_attrs);
    builder.AddNode("sub", "Sub", {"input", "rm1_out"}, {"sub_out"});
    builder.AddNode("pow", "Pow", {"sub_out", "pow_exp"}, {"pow_out"});
    builder.AddNode("rm2", "ReduceMean", {"pow_out"}, {"rm2_out"}, "", rm_attrs);
    builder.AddNode("add_eps", "Add", {"rm2_out", "eps"}, {"add_eps_out"});
    builder.AddNode("sqrt", "Sqrt", {"add_eps_out"}, {"sqrt_out"});
    builder.AddNode("div", "Div", {"sub_out", "sqrt_out"}, {div_out_name});

    if (gamma_def.has_value()) {
      builder.AddNode("mul_gamma", "Mul", {div_out_name, "gamma"}, {mul_out_name});
      if (beta_def.has_value()) {
        builder.AddNode("add_beta", "Add", {mul_out_name, "beta"}, {"output"});
      }
    } else if (beta_def.has_value()) {
      builder.AddNode("add_beta", "Add", {div_out_name, "beta"}, {"output"});
    }
  };
}

[[maybe_unused]] ProviderOptions GetProviderOptions(const std::string& backend_type) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_type;
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// 3D input [1, 8, 16], axes=[-1], 1D gamma/beta {16}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_3D_1D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_3D_1D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

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

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({1, 1, C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({1, 1, C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

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

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 4, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

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

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 64;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 256, 64}, false, -1.5f, 1.5f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// gamma {1, 64, 1} with axes=[-1] — non-normalized dim is non-unit.
// Partial fusion not supported for HTP, so fusion must be skipped.
TEST_F(QnnHTPBackendTests, LayerNormFusion_InvalidGammaShape) {
  const std::filesystem::path json_dir = "LayerNormFusion_InvalidGammaShape";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({1, 64, 1}, true, std::vector<float>(64, 1.0f));
  const auto beta_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 64, 16}, false, -1.0f, 1.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 0);
}

// beta {1, 64, 1} with axes=[-1] — non-normalized dim is non-unit.
// Partial fusion not supported for HTP, so fusion must be skipped.
TEST_F(QnnHTPBackendTests, LayerNormFusion_InvalidBetaShape) {
  const std::filesystem::path json_dir = "LayerNormFusion_InvalidBetaShape";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({1, 64, 1}, true, std::vector<float>(64, 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 64, 16}, false, -1.0f, 1.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 0);
}

// FP16 activations: 3D input [1, 8, 16], axes=[-1], 1D gamma/beta {16}
TEST_F(QnnHTPBackendTests, LayerNormFusion_FP16_3D_1D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_FP16_3D_1D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<Ort::Float16_t>({C}, true, std::vector<Ort::Float16_t>(static_cast<size_t>(C), Ort::Float16_t(1.0f)));
  const auto beta_def = TestInputDef<Ort::Float16_t>({C}, true, std::vector<Ort::Float16_t>(static_cast<size_t>(C), Ort::Float16_t(0.0f)));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<Ort::Float16_t>(
          TestInputDef<Ort::Float16_t>({1, 8, 16}, false, Ort::Float16_t(-2.0f), Ort::Float16_t(2.0f)),
          {-1}, Ort::Float16_t(1e-3f), gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

// Dynamic beta with a constant gamma
TEST_F(QnnHTPBackendTests, LayerNormFusion_DynamicBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_DynamicBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({C}, false, -0.5f, 0.5f);
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

// Dynamic gamma and beta
TEST_F(QnnHTPBackendTests, LayerNormFusion_DynamicGammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_DynamicGammaBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, false, 0.5f, 1.5f);
  const auto beta_def = TestInputDef<float>({C}, false, -0.5f, 0.5f);
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

// 4D input [1, 4, 4, 64], axes=[-1], 3D gamma/beta {1, 1, 64}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_4D_3D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_4D_3D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("htp");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 64;
  const auto gamma_def = TestInputDef<float>({1, 1, C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({1, 1, C}, true, -0.5f, 0.5f);
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 4, 4, C}, false, -2.0f, 2.0f),
          {-1}, 1e-6f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)
//
// GPU tests:
//

// 3D input [1, 8, 16], axes=[-1], 1D gamma/beta {16}.
TEST_F(QnnGPUBackendTests, LayerNormFusion_3D_1D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_3D_1D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// 3D input, padded gamma/beta {1, 1, 16} — fusion must squeeze to {16}.
TEST_F(QnnGPUBackendTests, LayerNormFusion_3D_PaddedGammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_3D_PaddedGammaBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({1, 1, C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({1, 1, C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// 4D input [1, 4, 8, 16], axes=[-1], 1D gamma/beta {16}.
TEST_F(QnnGPUBackendTests, LayerNormFusion_4D_1D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_4D_1D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 4, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// Transformer shape [1, 256, 64], axes=[-1], 1D gamma/beta {64}.
TEST_F(QnnGPUBackendTests, LayerNormFusion_TransformerShape) {
  const std::filesystem::path json_dir = "LayerNormFusion_TransformerShape";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 64;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 256, 64}, false, -1.5f, 1.5f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// gamma {1, 64, 1} with axes=[-1] — non-normalized dim is non-unit, partial fusion with standalone trailing Mul+Add.
TEST_F(QnnGPUBackendTests, LayerNormFusion_InvalidGammaShape) {
  const std::filesystem::path json_dir = "LayerNormFusion_InvalidGammaShape";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({1, 64, 1}, true, std::vector<float>(static_cast<size_t>(64), 1.0f));
  const auto beta_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 64, 16}, false, -1.0f, 1.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);
}

// beta {1, 64, 1} with axes=[-1] — non-normalized dim is non-unit, partial fusion with standalone trailing Add.
TEST_F(QnnGPUBackendTests, LayerNormFusion_InvalidBetaShape) {
  const std::filesystem::path json_dir = "LayerNormFusion_InvalidBetaShape";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({1, 64, 1}, true, std::vector<float>(64, 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 64, 16}, false, -1.0f, 1.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);
}

// FP16 activations: 3D input [1, 8, 16], axes=[-1], 1D gamma/beta {16}
TEST_F(QnnGPUBackendTests, LayerNormFusion_FP16_3D_1D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_FP16_3D_1D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<Ort::Float16_t>({C}, true, std::vector<Ort::Float16_t>(static_cast<size_t>(C), Ort::Float16_t(1.0f)));
  const auto beta_def = TestInputDef<Ort::Float16_t>({C}, true, std::vector<Ort::Float16_t>(static_cast<size_t>(C), Ort::Float16_t(0.0f)));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<Ort::Float16_t>(
          TestInputDef<Ort::Float16_t>({1, 8, 16}, false, Ort::Float16_t(-2.0f), Ort::Float16_t(2.0f)),
          {-1}, Ort::Float16_t(1e-3f), gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

// Dynamic beta with a constant gamma
TEST_F(QnnGPUBackendTests, LayerNormFusion_DynamicBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_DynamicBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({C}, false, -0.5f, 0.5f);
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

// Dynamic gamma and beta
TEST_F(QnnGPUBackendTests, LayerNormFusion_DynamicGammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_DynamicGammaBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, false, 0.5f, 1.5f);
  const auto beta_def = TestInputDef<float>({C}, false, -0.5f, 0.5f);
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

// 3D input [1, 8, 16], axes=[-1], no gamma/beta.
TEST_F(QnnGPUBackendTests, LayerNormFusion_NoGamma_NoBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_NoGamma_NoBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, std::nullopt, std::nullopt),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

// 3D input [1, 8, 16], axes=[-1], 1D gamma {16}, no beta.
TEST_F(QnnGPUBackendTests, LayerNormFusion_Gamma_NoBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_Gamma_NoBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, gamma_def, std::nullopt),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

// 3D input [1, 8, 16], axes=[-1], no gamma, 1D beta {16}.
TEST_F(QnnGPUBackendTests, LayerNormFusion_NoGamma_Beta) {
  const std::filesystem::path json_dir = "LayerNormFusion_NoGamma_Beta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto beta_def = TestInputDef<float>({C}, true, std::vector<float>(static_cast<size_t>(C), 0.0f));
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
          {-1}, 1e-5f, std::nullopt, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);
}

// 4D input [1, 4, 4, 64], axes=[-1], 3D gamma/beta {1, 1, 64}.
TEST_F(QnnGPUBackendTests, LayerNormFusion_4D_3D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_4D_3D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 64;
  const auto gamma_def = TestInputDef<float>({1, 1, C}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({1, 1, C}, true, -0.5f, 0.5f);
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, 4, 4, C}, false, -2.0f, 2.0f),
          {-1}, 1e-6f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

// 4D input [1, 16, 64, 64], axes=[1], 3D gamma/beta {16, 1, 1}.
TEST_F(QnnGPUBackendTests, LayerNormFusion_4D_3D_Axis1_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_4D_3D_Axis1_GammaBeta";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetProviderOptions("gpu");
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  const int64_t C = 16;
  const auto gamma_def = TestInputDef<float>({C, 1, 1}, true, std::vector<float>(static_cast<size_t>(C), 1.0f));
  const auto beta_def = TestInputDef<float>({C, 1, 1}, true, -0.5f, 0.5f);
  RunQnnModelTest(
      BuildLayerNormFusionTestCase<float>(
          TestInputDef<float>({1, C, 64, 64}, false, -2.0f, 2.0f),
          {1}, 1e-6f, gamma_def, beta_def),
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
}

#endif  // defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
