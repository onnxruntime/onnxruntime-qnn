// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP backend tests
//

// Cover inferred and explicit FP16 output types. The source input only
// supplies shape/type information and may remain FP32 with dtype=FLOAT16.
template <typename InputType>
static void RunFP16RandomNormalLikeTest(bool has_seed, bool explicit_dtype, bool rnl_is_graph_output) {
  auto build_test_case = [=](ModelTestBuilder& builder) {
    const TestInputDef<float> input_def({1, 4, 3}, false,
                                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    if constexpr (std::is_same_v<InputType, Ort::Float16_t>) {
      MakeTestInput<Ort::Float16_t>(builder, "input", ConvertToFP16InputDef(input_def));
    } else {
      MakeTestInput<float>(builder, "input", input_def);
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("mean", 0.5f));
    attrs.push_back(MakeAttribute("scale", 1.5f));
    if (has_seed) {
      attrs.push_back(MakeAttribute("seed", 42.0f));
    }
    if (explicit_dtype) {
      attrs.push_back(MakeAttribute("dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)));
    }

    builder.AddNode("RandomNormalLike", "RandomNormalLike", {"input"}, {"random_out"}, kOnnxDomain, attrs);
    // HTP execution requires a runtime input consumed by the graph. RandomNormalLike
    // only consumes static shape/seed tensors, so keep an Add consumer in every case.
    std::string add_input = "input";
    if constexpr (!std::is_same_v<InputType, Ort::Float16_t>) {
      add_input = "add_input";
      MakeTestInput<Ort::Float16_t>(builder, add_input, ConvertToFP16InputDef(input_def));
    }
    builder.AddNode("Add", "Add", {add_input, "random_out"}, {"Y"});
    builder.MakeOutput("Y");
    if (rnl_is_graph_output) {
      builder.MakeOutput<Ort::Float16_t>("random_out", input_def.GetShape());
    }
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // The CPU EP lacks an FP16 RandomNormalLike kernel, and random number
  // algorithms differ across backends. Check assignment and inference instead.
  RunQnnModelTest(build_test_case,
                  provider_options,
                  14,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-5f)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

TEST_F(QnnHTPBackendTests, RandomNormalLike_FP16) {
  RunFP16RandomNormalLikeTest<Ort::Float16_t>(true, false, true);
}

TEST_F(QnnHTPBackendTests, RandomNormalLike_FP16_NoSeed) {
  RunFP16RandomNormalLikeTest<Ort::Float16_t>(false, false, true);
}

TEST_F(QnnHTPBackendTests, RandomNormalLike_FP16_ExplicitDtype) {
  RunFP16RandomNormalLikeTest<float>(true, true, true);
}

TEST_F(QnnHTPBackendTests, RandomNormalLike_FP16_AddDownstream) {
  RunFP16RandomNormalLikeTest<Ort::Float16_t>(true, false, false);
}

// Test RandomNormalLike + Add to verify the output can be consumed by a downstream op.
// This tests the full HTP path: RandomNormalLike (uint8) -> Dequantize -> Add -> output.
TEST_F(QnnHTPBackendTests, RandomNormalLike_AddDownstream) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    builder.MakeInput<float>("input", {1, 4, 3}, input_data);

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("mean", 0.0f));
    attrs.push_back(MakeAttribute("scale", 1.0f));
    attrs.push_back(MakeAttribute("seed", 42.0f));

    builder.AddNode("RandomNormalLike",
                    "RandomNormalLike",
                    {"input"},
                    {"random_out"},
                    kOnnxDomain,
                    attrs);

    // Add node: input + random_output
    builder.MakeOutput("Y");
    builder.AddNode("Add",
                    "Add",
                    {"input", "random_out"},
                    {"Y"});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // Do not verify outputs since RandomNormalLike randomness algo differs from ORT CPU EP.
  RunQnnModelTest(build_test_case,
                  provider_options,
                  14,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-5f)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

TEST_F(QnnHTPBackendTests, RandomNormalLike_GraphInputRegistered) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    builder.graph_->set_name("solo_random_normal_like_graph");

    MakeTestInput<float>(builder, "x", TestInputDef<float>({2, 2}, false, {0.0f, 1.0f, 2.0f, 3.0f}));

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("mean", 0.0f));
    attrs.push_back(MakeAttribute("scale", 1.0f));
    attrs.push_back(MakeAttribute("seed", 42.0f));

    builder.AddNode("RandomNormalLike", "RandomNormalLike", {"x"}, {"y"}, kOnnxDomain, attrs);
    builder.MakeOutput("y");
  };

  std::unique_ptr<ModelAndBuilder> model;
  CreateModelInMemory(model, build_test_case, 14);

  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

  onnxruntime::test::ProviderOptions options;
#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NO_THROW({
    Ort::Session session(*ort_env, model->model_data.data(), model->model_data.size(), so);
  });
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
