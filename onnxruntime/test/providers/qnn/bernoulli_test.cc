// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <memory>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Bernoulli with dtype unset (defaults to input dtype, float32).
// Output is non-deterministic: we only verify the session builds without error.
TEST_F(QnnHTPBackendTests, Bernoulli_Float32_HTP) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    builder.graph_->set_name("bernoulli_float32_htp");

    MakeTestInput<float>(builder, "prob", TestInputDef<float>({2, 4}, false, {0.1f, 0.5f, 0.9f, 0.3f, 0.2f, 0.6f, 0.8f, 0.4f}));

    builder.AddNode("Bernoulli", "Bernoulli", {"prob"}, {"out"}, kOnnxDomain, {});
    builder.MakeOutput("out");
  };

  std::unique_ptr<ModelAndBuilder> model;
  CreateModelInMemory(model, build_test_case, 15);

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

// Bernoulli with dtype=6 (TensorProto::INT32) explicitly set.
// Exercises the builder's dtype-conversion path (Cast(BOOL_8 → INT32) on HTP).
TEST_F(QnnHTPBackendTests, Bernoulli_Int32Output_HTP) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    builder.graph_->set_name("bernoulli_int32_output_htp");

    MakeTestInput<float>(builder, "prob", TestInputDef<float>({3, 4}, false, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.1f, 0.3f, 0.5f, 0.9f}));

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("dtype", static_cast<int64_t>(6)));  // TensorProto::INT32

    builder.AddNode("Bernoulli", "Bernoulli", {"prob"}, {"out"}, kOnnxDomain, attrs);
    builder.MakeOutput("out");
  };

  std::unique_ptr<ModelAndBuilder> model;
  CreateModelInMemory(model, build_test_case, 15);

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
