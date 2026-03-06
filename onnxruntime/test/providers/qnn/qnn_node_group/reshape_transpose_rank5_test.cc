// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Build float test: Add -> Reshape(rank-6) -> Transpose -> Reshape -> Add
// Uses smaller dimensions for testing
GetTestModelFn BuildRank6ToRank5FloatTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("rank6_to_rank5_fusion_float_graph");

    // input
    const auto input_def = TestInputDef<float>({256, 64}, false, -10.0f, 10.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // input + 1 -> add1_out
    builder.MakeScalarInitializer<float>("add_const1", 1.0f);
    builder.AddNode("Add1",
                    "Add",
                    {"input", "add_const1"},
                    {"add1_out"},
                    kOnnxDomain);

    // Reshape: (256, 64) -> (1, 4, 4, 4, 4, 64)
    builder.Make1DInitializer<int64_t>("reshape1_shape", {1, 4, 4, 4, 4, 64});
    builder.AddNode("Reshape1",
                    "Reshape",
                    {"add1_out", "reshape1_shape"},
                    {"reshape1_out"},
                    kOnnxDomain);

    // Transpose: perm [0, 2, 1, 3, 4, 5]
    builder.AddNode("Transpose",
                    "Transpose",
                    {"reshape1_out"},
                    {"transpose_out"},
                    kOnnxDomain,
                    {builder.MakeIntsAttribute("perm", {0, 2, 1, 3, 4, 5})});

    // Reshape: (1, 4, 4, 4, 4, 64) -> (1, 256, 64)
    builder.Make1DInitializer<int64_t>("reshape2_shape", {1, 256, 64});
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "reshape2_shape"},
                    {"reshape2_out"},
                    kOnnxDomain);

    // reshape2_out + 1 -> output
    builder.MakeScalarInitializer<float>("add_const2", 1.0f);
    builder.AddNode("Add2",
                    "Add",
                    {"reshape2_out", "add_const2"},
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

TEST_F(QnnHTPBackendTests, Rank6ToRank5Fusion_Float) {
  ProviderOptions provider_options = GetProviderOptions();
  RunQnnModelTest(BuildRank6ToRank5FloatTestCase(),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
