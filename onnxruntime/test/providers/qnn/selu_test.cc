// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "core/graph/onnx_protobuf.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a Selu model on the given QNN backend and compares output to CPU EP.
// SeLU has no QDQ variant (float-only type constraints), so this helper covers
// all three backends (cpu / htp / gpu) via backend_name.
static void RunSeluTest(const std::vector<TestInputDef<float>>& input_defs,
                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                        ExpectedEPNodeAssignment expected_ep_assignment,
                        const std::string& backend_name = "cpu",
                        int opset = 6,
                        float fp32_abs_err = 1e-5f,
                        bool enable_htp_fp16_precision = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  if (enable_htp_fp16_precision) {
#if defined(_WIN32)
    SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
#if defined(__linux__) && !defined(__aarch64__)
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  RunQnnModelTest(BuildOpTestCase<float>("Selu_node", "Selu", input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Runs a native FP16 Selu model on the QNN HTP backend.
static void RunSeluFP16Test(const std::vector<TestInputDef<float>>& input_defs,
                            const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            int opset = 6,
                            float tolerance = 0.004f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  std::vector<TestInputDef<Ort::Float16_t>> input_fp16_defs;
  input_fp16_defs.reserve(input_defs.size());
  for (const auto& def : input_defs) {
    input_fp16_defs.push_back(ConvertToFP16InputDef(def));
  }

  RunQnnModelTest(BuildOpTestCase<Ort::Float16_t>("Selu_node", "Selu", input_fp16_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  tolerance);
}

//
// CPU tests
//

// Default alpha and gamma, opset 6.
TEST_F(QnnCPUBackendTests, Selu_DefaultAttrs) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {},
              ExpectedEPNodeAssignment::All);
}

// Custom alpha and gamma, opset 6.
TEST_F(QnnCPUBackendTests, Selu_CustomAlphaGamma) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {test::MakeAttribute("alpha", 1.0f),
               test::MakeAttribute("gamma", 2.0f)},
              ExpectedEPNodeAssignment::All);
}

// Opset 1 — non-function variant of SeLU.
TEST_F(QnnCPUBackendTests, Selu_Opset1) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {},
              ExpectedEPNodeAssignment::All,
              "cpu",
              1);
}

// Opset 22 — extends type constraints to include bfloat16 (float32 still valid).
TEST_F(QnnCPUBackendTests, Selu_Opset22) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {},
              ExpectedEPNodeAssignment::All,
              "cpu",
              22);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// HTP tests
//

// FP32 with default attrs, opset 6.
TEST_F(QnnHTPBackendTests, Selu_FP32_DefaultAttrs) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {},
              ExpectedEPNodeAssignment::All,
              "htp");
}

// FP32 with custom alpha and gamma, opset 6.
TEST_F(QnnHTPBackendTests, Selu_FP32_CustomAlphaGamma) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {test::MakeAttribute("alpha", 1.0f),
               test::MakeAttribute("gamma", 2.0f)},
              ExpectedEPNodeAssignment::All,
              "htp");
}

// FP32 executed at FP16 precision on HTP.
TEST_F(QnnHTPBackendTests, Selu_FP32_as_FP16) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {},
              ExpectedEPNodeAssignment::All,
              "htp", 6, 0.004f, true);
}

// FP32-as-FP16 with custom attributes.
TEST_F(QnnHTPBackendTests, Selu_FP32_as_FP16_CustomAlphaGamma) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {test::MakeAttribute("alpha", 1.0f),
               test::MakeAttribute("gamma", 2.0f)},
              ExpectedEPNodeAssignment::All,
              "htp", 6, 0.004f, true);
}

// Native FP16 model with default attrs.
TEST_F(QnnHTPBackendTests, Selu_FP16) {
  RunSeluFP16Test({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                  {},
                  ExpectedEPNodeAssignment::All);
}

// Native FP16 model with custom alpha and gamma.
TEST_F(QnnHTPBackendTests, Selu_FP16_CustomAlphaGamma) {
  RunSeluFP16Test({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                  {test::MakeAttribute("alpha", 1.0f),
                   test::MakeAttribute("gamma", 2.0f)},
                  ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)

//
// GPU tests
//

// FP32 with default attrs.
TEST_F(QnnGPUBackendTests, Selu_DefaultAttrs) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {},
              ExpectedEPNodeAssignment::All,
              "gpu");
}

// FP32 with custom alpha and gamma.
TEST_F(QnnGPUBackendTests, Selu_CustomAlphaGamma) {
  RunSeluTest({TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
              {test::MakeAttribute("alpha", 1.0f),
               test::MakeAttribute("gamma", 2.0f)},
              ExpectedEPNodeAssignment::All,
              "gpu");
}

#endif  // defined(_M_ARM64) — GPU tests

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
