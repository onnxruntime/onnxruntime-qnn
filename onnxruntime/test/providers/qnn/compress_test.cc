// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

static GetTestModelFn BuildFixedShapeCompressTestCase() {
  return [](ModelTestBuilder& builder) {
    constexpr int64_t kRows = 25;
    constexpr int64_t kColumns = 1536;
    MakeTestInput<float>(builder, "input", TestInputDef<float>({kRows, kColumns}, false, 0.0f, 1.0f));
    MakeTestInput<bool>(builder, "condition", TestInputDef<bool>({kRows}, true, std::vector<bool>(kRows, true)));
    builder.MakeOutput<float>("output", std::vector<int64_t>{kRows, kColumns});
    builder.AddNode("compress", "Compress", {"input", "condition"}, {"output"}, kOnnxDomain,
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0))});
  };
}

TEST_F(QnnGPUBackendTests, CompressFixedShapeAxis0) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";

  // This exercises only the fixed-shape audio recipe. Generic packed Compress
  // remains unsupported because its output length is dynamic.
  RunQnnModelTest(BuildFixedShapeCompressTestCase(),
                  provider_options,
                  13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-5f)});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
