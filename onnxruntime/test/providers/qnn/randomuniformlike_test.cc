// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <memory>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "core/graph/node_attr_utils.h"
#include "gtest/gtest.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

struct ModelAndBuilder {
  std::string model_data;
  ModelTestBuilder builder;
};

void CreateModelInMemory(std::unique_ptr<ModelAndBuilder>& result,
                         const GetTestModelFn& model_build_fn,
                         int opset_version) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};
  result = std::make_unique<ModelAndBuilder>();
  model_build_fn(result->builder);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{result->builder.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  result->builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  result->builder.model_.SerializeToString(&result->model_data);
}

}  // namespace

TEST_F(QnnHTPBackendTests, RandomUniformLike_GraphInputRegistered) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    builder.graph_->set_name("solo_random_uniform_like_graph");

    MakeTestInput<float>(builder, "x", TestInputDef<float>({2, 2}, false, {0.0f, 1.0f, 2.0f, 3.0f}));

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("low", 0.0f));
    attrs.push_back(MakeAttribute("high", 1.0f));
    attrs.push_back(MakeAttribute("seed", 42.0f));

    builder.AddNode("RandomUniformLike", "RandomUniformLike", {"x"}, {"y"}, kOnnxDomain, attrs);
    builder.MakeOutput("y");
  };

  std::unique_ptr<ModelAndBuilder> model;
  CreateModelInMemory(model, build_test_case, 14);

  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

  onnxruntime::ProviderOptions options;
#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);

  ASSERT_NO_THROW({
    Ort::Session session(*ort_env, model->model_data.data(), model->model_data.size(), so);
  });
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
