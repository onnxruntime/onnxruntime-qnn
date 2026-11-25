// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <array>
#include <memory>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

ProviderOptions MultinomialOptions(const char* backend) {
  ProviderOptions options{{"backend_type", backend}};
#if defined(__linux__) && !defined(__aarch64__)
  if (std::string(backend) == "htp") {
    // Match CumSum tests: host simulation needs a target supporting FP16 CumSum.
    options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  }
#endif
  return options;
}

GetTestModelFn MultinomialCase(int64_t dtype, int64_t samples, bool seed, bool fp16 = false,
                               bool downstream = false, float other_logit = -255.0f, float selected_logit = 0.0f) {
  return [=](ModelTestBuilder& builder) {
    const std::vector<int64_t> shape{3, 4};
    // One supported class per row after Exp underflow. Verify first, middle,
    // and last selection without requiring CPU and QNN RNG sequences to match.
    const std::vector<float> logits{selected_logit, other_logit, other_logit, other_logit, other_logit, other_logit, selected_logit, other_logit,
                                    other_logit, other_logit, other_logit, selected_logit};
    if (fp16) {
      std::vector<Ort::Float16_t> values;
      for (float value : logits) {
        values.emplace_back(value);
      }
      builder.MakeInput<Ort::Float16_t>("logits", shape, values);
    } else {
      builder.MakeInput<float>("logits", shape, logits);
    }
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs{MakeAttribute("sample_size", samples)};
    if (dtype != 0) {
      attrs.push_back(MakeAttribute("dtype", dtype));
    }
    if (seed) {
      attrs.push_back(MakeAttribute("seed", 42.0f));
    }
    builder.AddNode("sample", "Multinomial", {"logits"}, {"sampled"}, kOnnxDomain, attrs);
    if (downstream) {
      builder.AddNode("cast", "Cast", {"sampled"}, {"out"}, kOnnxDomain,
                      {MakeAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT})});
      builder.MakeOutput("out");
    } else {
      builder.MakeOutput("sampled");
    }
  };
}

void CheckOneClassSamples(const GetTestModelFn& build, const char* backend, int64_t dtype, bool downstream = false) {
  const auto provider_options = MultinomialOptions(backend);
  CONDITIONAL_SKIP_TEST_ON_LINUX_ARM64(provider_options, QNN_HTP_DEVICE_ARCH_V68, "FP16");
  std::unique_ptr<ModelAndBuilder> model;
  CreateModelInMemory(model, build, 13);
  std::vector<Ort::Value> outputs;
  InferenceModel(model->model_data, "Multinomial", provider_options, ExpectedEPNodeAssignment::All,
                 model->builder.feeds_, outputs);
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].GetTensorTypeAndShapeInfo().GetShape(), (std::vector<int64_t>{3, 1}));
  const std::vector<int64_t> expected{0, 2, 3};
  for (size_t i = 0; i < expected.size(); ++i) {
    if (downstream) {
      EXPECT_EQ(outputs[0].GetTensorData<float>()[i], static_cast<float>(expected[i]));
    } else if (dtype == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      EXPECT_EQ(outputs[0].GetTensorData<int64_t>()[i], expected[i]);
    } else {
      EXPECT_EQ(outputs[0].GetTensorData<int32_t>()[i], expected[i]);
    }
  }
}

template <typename QuantType>
GetTestModelFn DequantizedLogitsCase(int64_t batch, int64_t classes) {
  return [=](ModelTestBuilder& builder) {
    std::vector<QuantType> values(static_cast<size_t>(batch * classes), QuantType{0});
    for (int64_t row = 0; row < batch; ++row) {
      values[static_cast<size_t>(row * classes + row % classes)] = QuantType{255};
    }
    builder.MakeInput<QuantType>("quant_logits", {batch, classes}, values);
    builder.AddDequantizeLinearNode<QuantType>("dq", "quant_logits", 1.0f, QuantType{0}, "logits");
    builder.AddNode("sample", "Multinomial", {"logits"}, {"out"}, kOnnxDomain,
                    {MakeAttribute("sample_size", int64_t{1})});
    builder.MakeOutput("out");
  };
}

void CheckDequantizedSamples(const GetTestModelFn& build) {
  // Multinomial consumes floating-point logits after DQ and produces integer
  // indices. Quantizing its integer output would not be a QDQ Multinomial.
  RunQnnModelTest(build, MultinomialOptions("htp"), 21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.0f)});
}

// This is a stochastic property test, not a CPU/QNN sequence comparison.
// A fixed stream replaying each Run will put all 256 samples in one bucket.
void CheckRandomnessAcrossRuns(const char* backend) {
  const auto provider_options = MultinomialOptions(backend);
  CONDITIONAL_SKIP_TEST_ON_LINUX_ARM64(provider_options, QNN_HTP_DEVICE_ARCH_V68, "FP16");
  std::unique_ptr<ModelAndBuilder> model;
  CreateModelInMemory(model, [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("logits", {1, 4}, std::vector<float>(4, 0.0f));
    builder.AddNode("sample", "Multinomial", {"logits"}, {"out"}, kOnnxDomain,
                    {MakeAttribute("sample_size", int64_t{1}), MakeAttribute("seed", 42.0f)});
    builder.MakeOutput("out"); }, 13);
  Ort::SessionOptions options;
  options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  RegisteredEpDeviceUniquePtr ep;
  RegisterQnnEpLibrary(ep, options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(ep), Ort::Session(*ort_env, model->model_data.data(), model->model_data.size(), options));
  const char* input_names[]{"logits"};
  const char* output_names[]{"out"};
  std::array<int, 4> histogram{};
  for (int iteration = 0; iteration < 256; ++iteration) {
    auto results = scoped.session().Run(Ort::RunOptions{nullptr}, input_names,
                                        &model->builder.feeds_.at("logits"), 1, output_names, 1);
    const int32_t sample = results[0].GetTensorData<int32_t>()[0];
    ASSERT_GE(sample, 0);
    ASSERT_LT(sample, 4);
    ++histogram[static_cast<size_t>(sample)];
  }
  for (int count : histogram) {
    // Wide bounds avoid flaky tests while detecting repeated or degenerate draws.
    EXPECT_GT(count, 20);
    EXPECT_LT(count, 108);
  }
}

}  // namespace

// This SDK has no QNN CPU RandomUniformLike implementation. Verify clean
// fallback rather than claiming that the CPU reference backend can sample.
TEST_F(QnnCPUBackendTests, Multinomial_DefaultInt32_NoSeed_Fallback) {
  RunQnnModelTest(MultinomialCase(0, 1, false), MultinomialOptions("cpu"), 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::None, ElementwiseAbsoluteVerifier(0.0f)});
}

TEST_F(QnnCPUBackendTests, Multinomial_Int64_Seed_Fallback) {
  RunQnnModelTest(MultinomialCase(ONNX_NAMESPACE::TensorProto_DataType_INT64, 1, true), MultinomialOptions("cpu"), 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::None, ElementwiseAbsoluteVerifier(0.0f)});
}

TEST_F(QnnCPUBackendTests, Multinomial_Int64_DownstreamCast_Fallback) {
  RunQnnModelTest(MultinomialCase(ONNX_NAMESPACE::TensorProto_DataType_INT64, 1, true, false, true),
                  MultinomialOptions("cpu"), 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(0.0f)});
}

TEST_F(QnnCPUBackendTests, Multinomial_MultipleSamples_Fallback) {
  RunQnnModelTest(MultinomialCase(0, 2, false), MultinomialOptions("cpu"), 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::None, ElementwiseAbsoluteVerifier(0.0f)});
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__linux__)

TEST_F(QnnHTPBackendTests, Multinomial_Float32_Int32_NoSeed) {
  CheckOneClassSamples(MultinomialCase(ONNX_NAMESPACE::TensorProto_DataType_INT32, 1, false), "htp",
                       ONNX_NAMESPACE::TensorProto_DataType_INT32);
}

TEST_F(QnnHTPBackendTests, Multinomial_Float16_Int64_Seed) {
  CheckOneClassSamples(MultinomialCase(ONNX_NAMESPACE::TensorProto_DataType_INT64, 1, true, true), "htp",
                       ONNX_NAMESPACE::TensorProto_DataType_INT64);
}

TEST_F(QnnHTPBackendTests, Multinomial_Int64_DownstreamCast) {
  CheckOneClassSamples(MultinomialCase(ONNX_NAMESPACE::TensorProto_DataType_INT64, 1, true, false, true), "htp",
                       ONNX_NAMESPACE::TensorProto_DataType_INT64, true);
}

// Diagnostic for HTP RNG advancement between executions. Disabled because
// an advancing random stream is outside this sampler's supported contract.
TEST_F(QnnHTPBackendTests, DISABLED_Multinomial_RandomnessAcrossRuns) {
  CheckRandomnessAcrossRuns("htp");
}

// Exercise large negative centered logits across the HTP Exp underflow range.
// The first/middle/last supported classes must be selected regardless of RNG state.
TEST_F(QnnHTPBackendTests, Multinomial_ExtremeLogits) {
  for (float other_logit : {-104.0f, -128.0f, -512.0f, -1000.0f, -10000.0f, -65504.0f}) {
    SCOPED_TRACE(other_logit);
    CheckOneClassSamples(MultinomialCase(ONNX_NAMESPACE::TensorProto_DataType_INT32, 1, false, false, false, other_logit),
                         "htp", ONNX_NAMESPACE::TensorProto_DataType_INT32);
  }
  CheckOneClassSamples(MultinomialCase(ONNX_NAMESPACE::TensorProto_DataType_INT32, 1, false, false, false, -65504.0f, 65504.0f),
                       "htp", ONNX_NAMESPACE::TensorProto_DataType_INT32);
}

TEST_F(QnnHTPBackendTests, Multinomial_ExtremeLogits_Float16) {
  for (float other_logit : {-104.0f, -128.0f, -512.0f, -1000.0f, -10000.0f, -65504.0f}) {
    SCOPED_TRACE(other_logit);
    CheckOneClassSamples(MultinomialCase(ONNX_NAMESPACE::TensorProto_DataType_INT64, 1, true, true, false, other_logit),
                         "htp", ONNX_NAMESPACE::TensorProto_DataType_INT64);
  }
  CheckOneClassSamples(MultinomialCase(ONNX_NAMESPACE::TensorProto_DataType_INT64, 1, true, true, false, -65504.0f, 65504.0f),
                       "htp", ONNX_NAMESPACE::TensorProto_DataType_INT64);
}

TEST_F(QnnHTPBackendTests, Multinomial_MultipleSamples_Fallback) {
  RunQnnModelTest(MultinomialCase(0, 2, false), MultinomialOptions("htp"), 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::None, ElementwiseAbsoluteVerifier(0.0f)});
}

TEST_F(QnnHTPBackendTests, Multinomial_DQ_U8_Small) {
  CheckDequantizedSamples(DequantizedLogitsCase<uint8_t>(1, 2));
}

TEST_F(QnnHTPBackendTests, Multinomial_DQ_U8_Batched) {
  CheckDequantizedSamples(DequantizedLogitsCase<uint8_t>(3, 4));
}

TEST_F(QnnHTPBackendTests, Multinomial_DQ_U16_Small) {
  CheckDequantizedSamples(DequantizedLogitsCase<uint16_t>(1, 2));
}

TEST_F(QnnHTPBackendTests, Multinomial_DQ_U16_Batched) {
  CheckDequantizedSamples(DequantizedLogitsCase<uint16_t>(3, 4));
}

#endif

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
