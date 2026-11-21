// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <fstream>
#include "nlohmann/json.hpp"
#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
namespace {

enum class KPrescaleTestCase { Fold,
                               SharedKey,
                               MismatchedScale,
                               NoDiv };

// Build the same float graph for the reference and QDQ graph for QNN.
void BuildKPrescaleTestModel(ModelTestBuilder& builder, KPrescaleTestCase test_case,
                             const std::vector<QuantParams<uint16_t>>* output_qparams = nullptr) {
  const bool qdq = output_qparams != nullptr;
  TestInputDef<float> x_def({1, 4, 8}, false, GetFloatDataInRange(0.0f, 4.0f, 32));
  TestInputDef<float> q_def({1, 2, 4, 4}, false, GetFloatDataInRange(0.0f, 1.0f, 32));
  TestInputDef<float> w_def({8, 8}, true, GetFloatDataInRange(0.0f, 4.0f, 64));
  MakeTestInput<float>(builder, "X", x_def);
  MakeTestInput<float>(builder, "Q", q_def);
  MakeTestInput<float>(builder, "W", w_def);
  builder.MakeInitializer<int64_t>("shape", {4}, {1, 4, 2, 4});

  auto quantize_input = [&](const std::string& name, const TestInputDef<float>& def) {
    if (!qdq) return name;
    const auto params = GetTestInputQuantParams<uint16_t>(def);
    return AddQDQNodePair<uint16_t>(builder, name + "_qdq", name, params.scale, params.zero_point);
  };
  const std::string x = quantize_input("X", x_def);
  const std::string query = quantize_input("Q", q_def);
  std::string weight = "W";
  if (qdq) {
    const auto params = GetTestInputQuantParams<uint8_t>(w_def);
    weight = AddQDQNodePair<uint8_t>(builder, "W_qdq", weight, params.scale, params.zero_point);
  }
  // K fits in [0,100]. Its U8 half-step exceeds the budget before /8, but not after.
  constexpr float kKeyScale = 100.0f / 65535.0f;
  auto quantize_key = [&](const std::string& name) {
    return qdq ? AddQDQNodePair<uint16_t>(builder, name + "_qdq", name, kKeyScale, 0) : name;
  };
  builder.AddNode("key_projection", "MatMul", {x, weight}, {"K"});
  const std::string key = quantize_key("K");
  builder.AddNode("key_reshape", "Reshape", {key, "shape"}, {"K_reshape"});
  builder.AddNode("key_transpose", "Transpose", {quantize_key("K_reshape")}, {"K_transpose"}, kOnnxDomain,
                  {builder.MakeIntsAttribute("perm", {0, 2, 3, 1})});
  builder.AddNode("qk", "MatMul", {query, quantize_key("K_transpose")}, {"score"});
  if (test_case == KPrescaleTestCase::NoDiv) {
    if (qdq) {
      AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "score_out", "score",
                                                      (*output_qparams)[0].scale, (*output_qparams)[0].zero_point);
    } else {
      builder.MakeOutput("Y");
      builder.AddNode("output", "Identity", {"score"}, {"Y"});
    }
    return;
  }

  std::string score = "score", divisor = "divisor";
  if (qdq) {
    const float scale = (*output_qparams)[0].scale * 8.0f;
    score = AddQDQNodePair<uint16_t>(builder, "score_qdq", score,
                                     test_case == KPrescaleTestCase::MismatchedScale ? scale * 1.05f : scale,
                                     (*output_qparams)[0].zero_point);
    // Cover the quantized scalar-DQ divisor used by exported QDQ graphs.
    builder.MakeScalarInitializer<uint16_t>("divisor_quant", 65535);
    builder.AddDequantizeLinearNode<uint16_t>("divisor_dq", "divisor_quant", 8.0f / 65535.0f,
                                              0, divisor);
  } else {
    builder.MakeScalarInitializer<float>(divisor, 8.0f);
  }
  builder.AddNode("attention_scale", "Div", {score, divisor}, {"scaled_score"});
  if (qdq) {
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "output_qdq", "scaled_score",
                                                    (*output_qparams)[0].scale, (*output_qparams)[0].zero_point);
  } else {
    builder.MakeOutput("Y");
    builder.AddNode("output", "Identity", {"scaled_score"}, {"Y"});
  }
  if (test_case == KPrescaleTestCase::SharedKey) {
    builder.AddNode("shared_key", "Identity", {key}, {"shared"});
    if (qdq) {
      AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "shared_out", "shared",
                                                      (*output_qparams)[1].scale, (*output_qparams)[1].zero_point);
    } else {
      builder.MakeOutput("shared");
    }
  }
}

}  // namespace

TEST_F(QnnHTPBackendTests, KPrescaleFusion_QDQ) {
  for (KPrescaleTestCase test_case : {KPrescaleTestCase::Fold, KPrescaleTestCase::SharedKey,
                                      KPrescaleTestCase::MismatchedScale, KPrescaleTestCase::NoDiv}) {
    SCOPED_TRACE(static_cast<int>(test_case));
    const auto graph_dir = std::filesystem::temp_directory_path() / "KPrescaleFusion_QDQ";
    std::filesystem::remove_all(graph_dir);
    ASSERT_TRUE(std::filesystem::create_directories(graph_dir));
    auto cleanup = gsl::finally([&]() { std::filesystem::remove_all(graph_dir); });
    ProviderOptions options;
    options["backend_type"] = "htp";
    options["offload_graph_io_quantization"] = "0";
    options["dump_json_qnn_graph"] = "1";
    options["json_qnn_graph_dir"] = graph_dir.string();
#if defined(__linux__) && !defined(__aarch64__)
    // Match the U16 MatMul tests' architecture for host HTP simulation.
    options["htp_arch"] = "73";
#endif
    QuantParams<uint16_t> final_params{};

    TestQDQModelAccuracy<uint16_t>(
        [test_case](ModelTestBuilder& builder) { BuildKPrescaleTestModel(builder, test_case); },
        [test_case, &final_params](ModelTestBuilder& builder, std::vector<QuantParams<uint16_t>>& params) {
          final_params = params[0];
          BuildKPrescaleTestModel(builder, test_case, &params);
        },
        options, 21, ExpectedEPNodeAssignment::All, QDQTolerance(0.01f));
    if (::testing::Test::IsSkipped()) return;
    const bool folded = test_case == KPrescaleTestCase::Fold;
    AssertOpInQnnGraph(graph_dir, "ElementWiseDivide", folded || test_case == KPrescaleTestCase::NoDiv ? 0 : 1);
    if (folded) {
      AssertOpWithNameInQnnGraph(graph_dir, "FullyConnected", "_k_prescale_fc");
      AssertOpWithNameInQnnGraph(graph_dir, "MatMul", "_k_prescale_matmul");
      AssertConvertOutputDataType(graph_dir, QNN_DATATYPE_UFIXED_POINT_8);
      for (const auto& entry : std::filesystem::directory_iterator(graph_dir)) {
        if (entry.path().extension() != ".json" ||
            entry.path().filename().string().find("_tensor_log") != std::string::npos) continue;
        std::ifstream stream(entry.path());
        const auto root = nlohmann::json::parse(stream, nullptr, false);
        ASSERT_FALSE(root.is_discarded());
        const auto& tensors = root.at("graph").at("tensors");
        size_t scaled_keys = 0, scaled_weights = 0;
        for (const auto& tensor : tensors.items()) {
          const auto& name = tensor.key();
          if (name.find("_k_prescale_projection") != std::string::npos ||
              name.find("_k_prescale_reshape") != std::string::npos ||
              name.find("_k_prescale_transpose") != std::string::npos) {
            const auto& encoding = tensor.value().at("quant_params").at("scale_offset");
            EXPECT_FLOAT_EQ(encoding.at("scale").get<float>(), 100.0f / 65535.0f / 8.0f);
            EXPECT_EQ(encoding.at("offset").get<int32_t>(), 0);
            ++scaled_keys;
          } else if (name.find("_k_prescale_weight") != std::string::npos) {
            const auto& encoding = tensor.value().at("quant_params").at("scale_offset");
            EXPECT_FLOAT_EQ(encoding.at("scale").get<float>(), 4.0f / 255.0f / 8.0f);
            EXPECT_EQ(encoding.at("offset").get<int32_t>(), 0);
            ++scaled_weights;
          }
        }
        EXPECT_EQ(scaled_keys, 3u);
        EXPECT_EQ(scaled_weights, 1u);
        for (const auto& node : root.at("graph").at("nodes").items()) {
          if (node.key().find("_k_prescale_matmul") == std::string::npos) continue;
          const auto& encoding = tensors.at(node.value().at("output_names").at(0).get<std::string>())
                                     .at("quant_params")
                                     .at("scale_offset");
          EXPECT_FLOAT_EQ(encoding.at("scale").get<float>(), final_params.scale);
          EXPECT_EQ(encoding.at("offset").get<int32_t>(), -static_cast<int32_t>(final_params.zero_point));
        }
      }
    }
  }
}

#endif
}  // namespace test
}  // namespace onnxruntime
#endif
