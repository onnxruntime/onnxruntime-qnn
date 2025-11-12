// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>
#include <cmath>
#include <optional>
#include <utility>
#include <array>
#include <memory>
#include <unordered_map>

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64)

namespace {

GetQDQTestCaseFn BuildReshapeGemmTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    // Define the test case for Reshape+Gemm fusion here
    const std::vector<int64_t> input_shape{4, 3, 2, 8};
    auto input_def = TestInputDef<float>(input_shape, false, -0.5f, 0.5f);
    NodeArg* input = MakeTestInput<float>(builder, input_def);

    NodeArg* reshape_shape_arg = builder.Make1DInitializer<int64_t>({24, 8});
    NodeArg* reshaped_output = builder.MakeIntermediate();
    builder.AddNode("Reshape", {input, reshape_shape_arg}, {reshaped_output});

    // NodeArg* gemm_bias = builder.MakeInitializer<float>({5}, -1.0f, 1.0f);
    NodeArg* gemm_weights = builder.MakeInitializer<float>({8, 5}, -1.0f, 1.0f);
    NodeArg* gemm_output = builder.MakeOutput();
    builder.AddNode("Gemm", {reshaped_output, gemm_weights}, {gemm_output});
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

}  // namespace

TEST_F(QnnHTPBackendTests, ReshapeGemmFusion) {
  ProviderOptions provider_options = GetProviderOptions();
  RunQnnModelTest(BuildReshapeGemmTestCase(),
                  provider_options,
                  /*opset_version=*/21,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f,
                  /*log_severity =*/logging::Severity::kVERBOSE,
                  /*verify_outputs=*/false);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
