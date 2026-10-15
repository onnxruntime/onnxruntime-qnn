// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <type_traits>
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

std::vector<Ort::Float16_t> ToFp16(const std::vector<float>& v) {
  std::vector<Ort::Float16_t> out;
  out.reserve(v.size());
  for (float f : v) out.emplace_back(f);
  return out;
}

// fp16 variant of BuildLayerNormFusionTestCase: input, gamma, beta, Pow exponent and epsilon are
// all FLOAT16 — this mirrors models converted with onnxconverter_common.float16 (keep_io_types).
GetTestModelFn BuildLayerNormFusionFp16TestCase(
    const TestInputDef<float>& float_input_def,
    const std::vector<int64_t>& axes,
    float epsilon,
    const std::vector<int64_t>& gamma_shape,
    const std::vector<float>& gamma_values,
    const std::vector<int64_t>& beta_shape,
    const std::vector<float>& beta_values) {
  return [=](ModelTestBuilder& builder) -> void {
    const TestInputDef<Ort::Float16_t> input_def = ConvertToFP16InputDef(float_input_def);
    MakeTestInput<Ort::Float16_t>(builder, "input", input_def);

    builder.MakeInitializer<Ort::Float16_t>("gamma", gamma_shape, ToFp16(gamma_values));
    builder.MakeInitializer<Ort::Float16_t>("beta", beta_shape, ToFp16(beta_values));
    builder.MakeScalarInitializer<Ort::Float16_t>("pow_exp", Ort::Float16_t(2.0f));
    builder.MakeScalarInitializer<Ort::Float16_t>("eps", Ort::Float16_t(epsilon));

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

// Normalization core whose Div output is shared by two heads, each applying its own gamma/beta:
//
//   [x] --> ReduceMean --> Sub --> ... --> Div --+--> Mul(gamma_a) --> Add(beta_a) ==> output_a
//                                                 +--> Mul(gamma_b) --> Add(beta_b) ==> output_b
//
// Mirrors a transformer where one normalized tensor feeds multiple heads. The full fusion cannot
// absorb either head, so the core must fuse into LayerNorm(x, ones, zeros) and both heads stay.
template <typename T>
GetTestModelFn BuildLayerNormSharedDivTestCase(const TestInputDef<float>& float_input_def, int64_t C) {
  return [=](ModelTestBuilder& builder) -> void {
    auto conv = [](const std::vector<float>& v) {
      if constexpr (std::is_same_v<T, float>) {
        return v;
      } else {
        return ToFp16(v);
      }
    };
    if constexpr (std::is_same_v<T, float>) {
      MakeTestInput<float>(builder, "input", float_input_def);
    } else {
      MakeTestInput<Ort::Float16_t>(builder, "input", ConvertToFP16InputDef(float_input_def));
    }

    const size_t n = static_cast<size_t>(C);
    std::vector<float> ga(n), gb(n), ba(n), bb(n);
    for (size_t i = 0; i < n; ++i) {
      ga[i] = 0.5f + 0.05f * static_cast<float>(i);
      gb[i] = 1.5f - 0.03f * static_cast<float>(i);
      ba[i] = 0.1f * static_cast<float>(i % 4);
      bb[i] = -0.2f + 0.02f * static_cast<float>(i);
    }
    builder.MakeInitializer<T>("gamma_a", {C}, conv(ga));
    builder.MakeInitializer<T>("beta_a", {C}, conv(ba));
    builder.MakeInitializer<T>("gamma_b", {C}, conv(gb));
    builder.MakeInitializer<T>("beta_b", {C}, conv(bb));
    builder.MakeScalarInitializer<T>("pow_exp", T(2.0f));
    builder.MakeScalarInitializer<T>("eps", T(1e-5f));

    std::vector<ONNX_NAMESPACE::AttributeProto> rm_attrs;
    rm_attrs.push_back(test::MakeAttribute("axes", std::vector<int64_t>{-1}));
    rm_attrs.push_back(test::MakeAttribute("keepdims", static_cast<int64_t>(1)));

    builder.AddNode("rm1", "ReduceMean", {"input"}, {"rm1_out"}, "", rm_attrs);
    builder.AddNode("sub", "Sub", {"input", "rm1_out"}, {"sub_out"});
    builder.AddNode("pow", "Pow", {"sub_out", "pow_exp"}, {"pow_out"});
    builder.AddNode("rm2", "ReduceMean", {"pow_out"}, {"rm2_out"}, "", rm_attrs);
    builder.AddNode("add_eps", "Add", {"rm2_out", "eps"}, {"add_eps_out"});
    builder.AddNode("sqrt", "Sqrt", {"add_eps_out"}, {"sqrt_out"});
    builder.AddNode("div", "Div", {"sub_out", "sqrt_out"}, {"div_out"});

    builder.AddNode("mul_a", "Mul", {"div_out", "gamma_a"}, {"mul_a_out"});
    builder.AddNode("add_a", "Add", {"mul_a_out", "beta_a"}, {"output_a"});
    builder.AddNode("mul_b", "Mul", {"div_out", "gamma_b"}, {"mul_b_out"});
    builder.AddNode("add_b", "Add", {"mul_b_out", "beta_b"}, {"output_b"});
    builder.MakeOutput("output_a");
    builder.MakeOutput("output_b");
  };
}

// Runs `build_fn` with QNN JSON graph dumping and returns the dump directory (cleaned up by caller).
void RunAndDump(const std::filesystem::path& json_dir, const GetTestModelFn& build_fn, float abs_err) {
  ProviderOptions opts = GetProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();
  RunQnnModelTest(build_fn, opts, 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(abs_err)});
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
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// gamma {1, 64, 1} with axes=[-1] — non-normalized dim is non-unit, so gamma cannot be folded
// into LayerNorm. Fusion falls back to core-only LayerNorm(x, ones, zeros); the Mul(gamma) and
// Add(beta) remain as standalone ops.
TEST_F(QnnHTPBackendTests, LayerNormFusion_InvalidGammaShape_FallsBackToCoreOnly) {
  const std::filesystem::path json_dir = "LayerNormFusion_InvalidGammaShape_FallsBackToCoreOnly";
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
      opts,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// ---- fp16 constants ---------------------------------------------------------
// fp16 models store the Pow exponent and epsilon as FLOAT16 initializers. Before fp16 support the
// fusion silently bailed out and the decomposed chain ran op-by-op in fp16.

// fp16 3D input [1, 8, 16], axes=[-1], 1D gamma/beta {16}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_Fp16_3D_1D_GammaBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_Fp16_3D_1D_GammaBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  const int64_t C = 16;
  std::vector<float> gamma(static_cast<size_t>(C)), beta(static_cast<size_t>(C));
  for (size_t i = 0; i < gamma.size(); ++i) {
    gamma[i] = 0.5f + 0.05f * static_cast<float>(i);
    beta[i] = -0.1f + 0.02f * static_cast<float>(i);
  }
  RunAndDump(json_dir,
             BuildLayerNormFusionFp16TestCase(TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
                                              {-1}, 1e-5f, {C}, gamma, {C}, beta),
             2e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWisePower", 0);
}

// fp16 transformer shape matching the style_encoder model: [4, 15, 256], axes=[-1].
TEST_F(QnnHTPBackendTests, LayerNormFusion_Fp16_TransformerShape) {
  const std::filesystem::path json_dir = "LayerNormFusion_Fp16_TransformerShape";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  const int64_t C = 256;
  RunAndDump(json_dir,
             BuildLayerNormFusionFp16TestCase(TestInputDef<float>({4, 15, 256}, false, -1.5f, 1.5f),
                                              {-1}, 1e-5f,
                                              {C}, std::vector<float>(static_cast<size_t>(C), 1.0f),
                                              {C}, std::vector<float>(static_cast<size_t>(C), 0.0f)),
             2e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
}

// ---- Shared Div (core-only fusion) ------------------------------------------

// fp32: Div output feeds two heads with distinct gamma/beta. Core fuses to one LayerNorm; the
// two Mul/Add heads remain (2 Mul, 2 Add in the QNN graph).
TEST_F(QnnHTPBackendTests, LayerNormFusion_SharedDiv_CoreOnly) {
  const std::filesystem::path json_dir = "LayerNormFusion_SharedDiv_CoreOnly";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunAndDump(json_dir,
             BuildLayerNormSharedDivTestCase<float>(TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f), 16),
             1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 2);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 2);
}

// fp16 version of the shared-Div case — matches the writer_head/glyph_head norm in style_encoder.
TEST_F(QnnHTPBackendTests, LayerNormFusion_Fp16_SharedDiv_CoreOnly) {
  const std::filesystem::path json_dir = "LayerNormFusion_Fp16_SharedDiv_CoreOnly";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunAndDump(json_dir,
             BuildLayerNormSharedDivTestCase<Ort::Float16_t>(
                 TestInputDef<float>({4, 15, 256}, false, -2.0f, 2.0f), 256),
             3e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 2);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 2);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
