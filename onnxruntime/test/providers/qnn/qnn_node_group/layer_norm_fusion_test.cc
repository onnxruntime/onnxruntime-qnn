// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "nlohmann/json.hpp"
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

// Deterministic non-trivial values (so gamma/beta actually affect the output): start + step * (i % 16).
std::vector<float> Ramp(size_t n, float start, float step) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = start + step * static_cast<float>(i % 16);
  }
  return v;
}

// Same chain as BuildLayerNormFusionTestCase but without the trailing Add(beta): the graph output is
// Mul(gamma). Exercises gamma-only fusion with a synthesized zeros beta.
GetTestModelFn BuildLayerNormFusionNoBetaTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& axes,
    float epsilon,
    const std::vector<int64_t>& gamma_shape,
    const std::vector<float>& gamma_values) {
  return [=](ModelTestBuilder& builder) -> void {
    MakeTestInput<float>(builder, "input", input_def);

    builder.MakeInitializer<float>("gamma", gamma_shape, gamma_values);
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

    builder.MakeOutput("output");
    builder.AddNode("mul_gamma", "Mul", {"div_out", "gamma"}, {"output"});
  };
}

// Same chain as BuildLayerNormFusionTestCase but without Mul(gamma): Div feeds Add(beta) directly.
// There is no beta-only fusion, so only the core is fused and Add(beta) stays a standalone op.
GetTestModelFn BuildLayerNormFusionNoGammaTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<int64_t>& axes,
    float epsilon,
    const std::vector<int64_t>& beta_shape,
    const std::vector<float>& beta_values) {
  return [=](ModelTestBuilder& builder) -> void {
    MakeTestInput<float>(builder, "input", input_def);

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

    builder.MakeOutput("output");
    builder.AddNode("add_beta", "Add", {"div_out", "beta"}, {"output"});
  };
}

// Asserts the shapes of the gamma (input 1) and beta (input 2) tensors read by the single fused QNN
// LayerNorm in the dumped graph. Tensors are found through the LayerNorm node's input list rather than
// by name: synthesized ones/zeros get generated names whose "_N" suffix depends on how many fusion
// candidates were validated first.
void ExpectLayerNormGammaBetaShapes(const std::filesystem::path& dump_dir,

                                    const std::vector<uint32_t>& expected_gamma_dims,
                                    const std::vector<uint32_t>& expected_beta_dims) {
  if (::testing::Test::IsSkipped()) {
    return;
  }

  std::filesystem::path json_path;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dump_dir, ec)) {
    const auto& p = entry.path();
    if (entry.is_regular_file() && p.extension() == ".json" &&
        p.filename().string().find("_tensor_log") == std::string::npos) {
      json_path = p;
      break;
    }
  }
  ASSERT_FALSE(json_path.empty()) << "No QNN JSON graph file found in " << dump_dir;

  std::ifstream json_file(json_path);
  ASSERT_TRUE(json_file.is_open()) << "Cannot open " << json_path;
  const nlohmann::json root = nlohmann::json::parse(json_file, nullptr, /*allow_exceptions=*/false);
  ASSERT_FALSE(root.is_discarded()) << "Failed to parse " << json_path;
  ASSERT_TRUE(root.is_object() && root.contains("graph") && root["graph"].is_object() &&
              root["graph"].contains("nodes") && root["graph"]["nodes"].is_object())
      << "JSON missing 'graph.nodes' in " << json_path;

  std::vector<std::string> ln_inputs;
  size_t ln_count = 0;
  try {
    for (const auto& [node_name, node] : root["graph"]["nodes"].items()) {
      if (node.is_object() && node.value("type", "") == "LayerNorm") {
        ++ln_count;
        ln_inputs = node.at("input_names").get<std::vector<std::string>>();
      }
    }
  } catch (const std::exception& ex) {
    FAIL() << "Failed to read LayerNorm inputs in " << json_path << ": " << ex.what();
  }
  ASSERT_EQ(ln_count, 1u) << "Expected exactly one LayerNorm in " << json_path;
  ASSERT_EQ(ln_inputs.size(), 3u) << "LayerNorm must have 3 inputs (x, gamma, beta) in " << json_path;

  AssertTensorShapeInQnnGraph(dump_dir, ln_inputs[1], expected_gamma_dims);
  AssertTensorShapeInQnnGraph(dump_dir, ln_inputs[2], expected_beta_dims);
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

// ---- Gamma-only fusion (synthesized zeros beta) -----------------------------

// Beta missing: the chain ends at Mul(gamma). Gamma is folded into LayerNorm, beta is a synthesized
// zeros tensor, and no element-wise op remains.
TEST_F(QnnHTPBackendTests, LayerNormFusion_GammaOnly_NoBeta) {
  const std::filesystem::path json_dir = "LayerNormFusion_GammaOnly_NoBeta";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  const int64_t C = 16;
  RunAndDump(json_dir,
             BuildLayerNormFusionNoBetaTestCase(TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
                                                {-1}, 1e-5f, {C}, Ramp(static_cast<size_t>(C), 0.5f, 0.05f)),
             1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
  ExpectLayerNormGammaBetaShapes(json_dir, {16}, {16});
}

// Beta invalid: gamma {16} is valid but beta {1, 8, 1} varies along a non-normalized axis, so it
// cannot be folded. The full candidate is rejected and gamma-only fusion is used; Add(beta) stays.
TEST_F(QnnHTPBackendTests, LayerNormFusion_GammaOnly_InvalidBetaShape) {
  const std::filesystem::path json_dir = "LayerNormFusion_GammaOnly_InvalidBetaShape";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  const int64_t C = 16;
  RunAndDump(json_dir,
             BuildLayerNormFusionTestCase(TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
                                          {-1}, 1e-5f,
                                          {C}, Ramp(static_cast<size_t>(C), 0.5f, 0.05f),
                                          {1, 8, 1}, Ramp(8, -0.2f, 0.05f)),
             1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);
  ExpectLayerNormGammaBetaShapes(json_dir, {16}, {16});
}

// ---- No beta-only fusion ----------------------------------------------------

// Div feeds Add(beta) directly, with no Mul(gamma). There is no beta-only candidate, so the fusion
// falls back to core-only LayerNorm(x, ones, zeros) and Add(beta) remains a standalone op.
TEST_F(QnnHTPBackendTests, LayerNormFusion_BetaOnly_NotFused_FallsBackToCoreOnly) {
  const std::filesystem::path json_dir = "LayerNormFusion_BetaOnly_NotFused_FallsBackToCoreOnly";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  const int64_t C = 16;
  RunAndDump(json_dir,
             BuildLayerNormFusionNoGammaTestCase(TestInputDef<float>({1, 8, 16}, false, -2.0f, 2.0f),
                                                 {-1}, 1e-5f, {C}, Ramp(static_cast<size_t>(C), -0.2f, 0.03f)),
             1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);
  // Both are synthesized (ones / zeros): beta was not folded.
  ExpectLayerNormGammaBetaShapes(json_dir, {16}, {16});
}

// ---- Unsupported axes (negative tests) --------------------------------------
// The fusion only handles axes that form a contiguous suffix ending at the last dim. Anything else
// must be left unfused: the decomposed chain runs as-is and stays numerically correct.

// axes=[0, 2] on [4, 8, 16]: ends at the last dim but skips axis 1 (not contiguous).
TEST_F(QnnHTPBackendTests, LayerNormFusion_Skip_NonContiguousAxes) {
  const std::filesystem::path json_dir = "LayerNormFusion_Skip_NonContiguousAxes";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  // gamma/beta {4, 1, 16} broadcast over the reduced dims, so the ONNX graph itself is valid.
  RunAndDump(json_dir,
             BuildLayerNormFusionTestCase(TestInputDef<float>({4, 8, 16}, false, -2.0f, 2.0f),
                                          {0, 2}, 1e-5f,
                                          {4, 1, 16}, Ramp(4 * 16, 0.5f, 0.05f),
                                          {4, 1, 16}, Ramp(4 * 16, -0.2f, 0.03f)),
             1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 0);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 2);
}

// axes=[0, 1] on [4, 8, 16]: contiguous but does not include the last dim (not trailing).
TEST_F(QnnHTPBackendTests, LayerNormFusion_Skip_NonTrailingAxes) {
  const std::filesystem::path json_dir = "LayerNormFusion_Skip_NonTrailingAxes";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunAndDump(json_dir,
             BuildLayerNormFusionTestCase(TestInputDef<float>({4, 8, 16}, false, -2.0f, 2.0f),
                                          {0, 1}, 1e-5f,
                                          {4, 8, 1}, Ramp(4 * 8, 0.5f, 0.05f),
                                          {4, 8, 1}, Ramp(4 * 8, -0.2f, 0.03f)),
             1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 0);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 2);
}

// ---- Multi-axis fusion ------------------------------------------------------
// Normalizing over the trailing axes [-2, -1] of [1, 4, 8, 16]. QNN HTP LayerNorm only normalizes the
// last axis, so the fusion flattens: Reshape [1,4,8,16] -> [4,128] -> LayerNorm(axis 1) -> Reshape
// back. Gamma/beta (real or synthesized) must therefore be {128}, and exactly two Reshapes appear.

// Full, multi-axis: gamma/beta {8, 16} fold into the flattened LayerNorm as {128}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_MultiAxis_Full) {
  const std::filesystem::path json_dir = "LayerNormFusion_MultiAxis_Full";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunAndDump(json_dir,
             BuildLayerNormFusionTestCase(TestInputDef<float>({1, 4, 8, 16}, false, -2.0f, 2.0f),
                                          {-2, -1}, 1e-5f,
                                          {8, 16}, Ramp(8 * 16, 0.5f, 0.05f),
                                          {8, 16}, Ramp(8 * 16, -0.2f, 0.03f)),
             1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "Reshape", 2);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
  ExpectLayerNormGammaBetaShapes(json_dir, {128}, {128});
}

// Core-only, multi-axis: gamma {1, 4, 1, 1} varies along a non-normalized axis, so neither Mul nor
// Add can be folded. LayerNorm gets synthesized ones/zeros of shape {128}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_MultiAxis_CoreOnly) {
  const std::filesystem::path json_dir = "LayerNormFusion_MultiAxis_CoreOnly";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunAndDump(json_dir,
             BuildLayerNormFusionTestCase(TestInputDef<float>({1, 4, 8, 16}, false, -2.0f, 2.0f),
                                          {-2, -1}, 1e-5f,
                                          {1, 4, 1, 1}, Ramp(4, 0.5f, 0.25f),
                                          {8, 16}, Ramp(8 * 16, -0.2f, 0.03f)),
             1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "Reshape", 2);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);
  ExpectLayerNormGammaBetaShapes(json_dir, {128}, {128});
}

// Gamma-only, multi-axis: padded gamma {1, 1, 8, 16} is collapsed to {128}; no beta, so a zeros
// beta of shape {128} is synthesized.
TEST_F(QnnHTPBackendTests, LayerNormFusion_MultiAxis_GammaOnly) {
  const std::filesystem::path json_dir = "LayerNormFusion_MultiAxis_GammaOnly";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunAndDump(json_dir,
             BuildLayerNormFusionNoBetaTestCase(TestInputDef<float>({1, 4, 8, 16}, false, -2.0f, 2.0f),
                                                {-2, -1}, 1e-5f,
                                                {1, 1, 8, 16}, Ramp(8 * 16, 0.5f, 0.05f)),
             1e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "Reshape", 2);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 0);
  ExpectLayerNormGammaBetaShapes(json_dir, {128}, {128});
}

// fp16 multi-axis core-only: synthesized ones/zeros must be FLOAT16 tensors of shape {128}.
TEST_F(QnnHTPBackendTests, LayerNormFusion_Fp16_MultiAxis_CoreOnly) {
  const std::filesystem::path json_dir = "LayerNormFusion_Fp16_MultiAxis_CoreOnly";
  std::filesystem::remove_all(json_dir);
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunAndDump(json_dir,
             BuildLayerNormFusionFp16TestCase(TestInputDef<float>({1, 4, 8, 16}, false, -2.0f, 2.0f),
                                              {-2, -1}, 1e-5f,
                                              {1, 4, 1, 1}, Ramp(4, 0.5f, 0.25f),
                                              {8, 16}, Ramp(8 * 16, -0.2f, 0.03f)),
             3e-2f);

  AssertOpInQnnGraph(json_dir, "LayerNorm", 1);
  AssertOpInQnnGraph(json_dir, "Reshape", 2);
  AssertOpInQnnGraph(json_dir, "ReduceMean", 0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);
  ExpectLayerNormGammaBetaShapes(json_dir, {128}, {128});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
