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

// RandomUniform has zero ONNX inputs: shape/low/high/seed all come from attributes.
// Since there is no reference input tensor to chain a downstream op off of (unlike
// RandomNormalLike), a separate input tensor matching the `shape` attribute is created
// purely to be the other operand of a downstream Add -- this lets RunQnnModelTest
// verify EP assignment and that the whole graph actually executes, without needing to
// compare random output values.
TEST_F(QnnCPUBackendTests, RandomUniform_CPUBackend) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    builder.MakeInput<float>("input", {2, 2}, input_data);

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("shape", std::vector<int64_t>{2, 2}));
    attrs.push_back(MakeAttribute("low", 0.0f));
    attrs.push_back(MakeAttribute("high", 1.0f));
    attrs.push_back(MakeAttribute("seed", 42.0f));

    builder.AddNode("RandomUniform", "RandomUniform", {}, {"random_out"}, kOnnxDomain, attrs);
    builder.MakeOutput("Y");
    builder.AddNode("Add", "Add", {"input", "random_out"}, {"Y"});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  // Do not verify outputs since RandomUniform randomness algo differs from ORT CPU EP.
  RunQnnModelTest(build_test_case,
                  provider_options,
                  14,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-5f)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP backend tests
//

// Test RandomUniform + Add to verify the output can be consumed by a downstream op,
// and that the node is actually assigned to QNN EP.
TEST_F(QnnHTPBackendTests, RandomUniform_AddDownstream) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    builder.MakeInput<float>("input", {2, 2}, input_data);

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("shape", std::vector<int64_t>{2, 2}));
    attrs.push_back(MakeAttribute("low", 0.0f));
    attrs.push_back(MakeAttribute("high", 1.0f));
    attrs.push_back(MakeAttribute("seed", 42.0f));

    builder.AddNode("RandomUniform", "RandomUniform", {}, {"random_out"}, kOnnxDomain, attrs);
    builder.MakeOutput("Y");
    builder.AddNode("Add", "Add", {"input", "random_out"}, {"Y"});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // Do not verify outputs since RandomUniform randomness algo differs from ORT CPU EP.
  RunQnnModelTest(build_test_case,
                  provider_options,
                  14,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-5f)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

// Same as above but without the optional `seed` attribute, to exercise the
// single-input (shape tensor only, no seed tensor) path through ProcessInputs.
TEST_F(QnnHTPBackendTests, RandomUniform_NoSeedAttribute_AddDownstream) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    std::vector<float> input_data(12, 1.0f);
    builder.MakeInput<float>("input", {3, 4}, input_data);

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("shape", std::vector<int64_t>{3, 4}));
    attrs.push_back(MakeAttribute("low", -1.0f));
    attrs.push_back(MakeAttribute("high", 1.0f));

    builder.AddNode("RandomUniform", "RandomUniform", {}, {"random_out"}, kOnnxDomain, attrs);
    builder.MakeOutput("Y");
    builder.AddNode("Add", "Add", {"input", "random_out"}, {"Y"});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // Do not verify outputs since RandomUniform randomness algo differs from ORT CPU EP.
  RunQnnModelTest(build_test_case,
                  provider_options,
                  14,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-5f)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
