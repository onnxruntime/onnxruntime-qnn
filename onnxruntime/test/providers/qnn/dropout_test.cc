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

// Dropout has no QDQ variant (float-only type constraints).
//
// In inference mode (training_mode absent or false) Dropout is a pure identity:
//   output = data,  mask = all-ones bool tensor.

// Builds a Dropout test model.
//   with_mask: if true, the optional mask output is included in the graph.
template <typename DataType>
inline GetTestModelFn BuildDropoutTestCase(const TestInputDef<DataType>& data_def,
                                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                           bool with_mask = false) {
  return [data_def, attrs, with_mask](ModelTestBuilder& builder) {
    MakeTestInput<DataType>(builder, "data", data_def);

    builder.MakeOutput("output");
    std::vector<std::string> output_names = {"output"};

    if (with_mask) {
      builder.MakeOutput("mask");
      output_names.push_back("mask");
    }

    builder.AddNode("Dropout_node", "Dropout", {"data"}, output_names, kOnnxDomain, attrs);
  };
}

// ---------------------------------------------------------------------------
// Test runner helpers
// ---------------------------------------------------------------------------

static void RunDropoutTest(const TestInputDef<float>& data_def,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           bool with_mask = false,
                           const std::string& backend_name = "cpu",
                           int opset = 13,
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

  RunQnnModelTest(BuildDropoutTestCase<float>(data_def, attrs, with_mask),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

static void RunDropoutFP16Test(const TestInputDef<float>& data_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                bool with_mask = false,
                                int opset = 13,
                                float tolerance = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  TestInputDef<Ort::Float16_t> data_fp16 = ConvertToFP16InputDef(data_def);

  RunQnnModelTest(BuildDropoutTestCase<Ort::Float16_t>(data_fp16, attrs, with_mask),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  tolerance);
}

// ---------------------------------------------------------------------------
// CPU tests
// ---------------------------------------------------------------------------

TEST_F(QnnCPUBackendTests, Dropout_Default) {
  RunDropoutTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                 {},
                 ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Dropout_WithMask) {
  RunDropoutTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                 {},
                 ExpectedEPNodeAssignment::All,
                 /*with_mask=*/true);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ---------------------------------------------------------------------------
// HTP FP32 tests
// x86_64 and ARM64 Windows, x86_64 and ARM64 Linux
// ---------------------------------------------------------------------------

TEST_F(QnnHTPBackendTests, Dropout_FP32_Default) {
  RunDropoutTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                 {},
                 ExpectedEPNodeAssignment::All,
                 /*with_mask=*/false,
                 "htp", 13, 1e-5f);
}

TEST_F(QnnHTPBackendTests, Dropout_FP32_WithMask) {
  RunDropoutTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                 {},
                 ExpectedEPNodeAssignment::All,
                 /*with_mask=*/true,
                 "htp", 13, 1e-5f);
}

// FP32 executed at FP16 precision on HTP.
TEST_F(QnnHTPBackendTests, Dropout_FP32_as_FP16) {
  RunDropoutTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                 {},
                 ExpectedEPNodeAssignment::All,
                 /*with_mask=*/false,
                 "htp", 13, 0.01f, /*enable_htp_fp16_precision=*/true);
}

// ---------------------------------------------------------------------------
// HTP native FP16 tests
// ---------------------------------------------------------------------------

TEST_F(QnnHTPBackendTests, Dropout_FP16) {
  RunDropoutFP16Test(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                     {},
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Dropout_FP16_WithMask) {
  RunDropoutFP16Test(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                     {},
                     ExpectedEPNodeAssignment::All,
                     /*with_mask=*/true);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(__aarch64__) || defined(_M_ARM64)

// ---------------------------------------------------------------------------
// HTP BF16 tests only on ARM64 Architecture (opset 22 adds bfloat16 to type constraints;)
// v81+ required
// ---------------------------------------------------------------------------

static void RunDropoutHTPBF16Test(const TestInputDef<float>& data_def,
                                   const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   bool with_mask = false,
                                   int opset = 13,
                                   float tolerance = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["htp_bf16_enable"] = "1";
  provider_options["soc_model"] = "88";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildDropoutTestCase<float>(data_def, attrs, with_mask),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  tolerance);
}

TEST_F(QnnHTPBackendTests, Dropout_HTP_BF16_Default) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunDropoutHTPBF16Test(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                        {},
                        ExpectedEPNodeAssignment::All,
                        /*with_mask=*/false,
                        /*opset=*/22);
}

TEST_F(QnnHTPBackendTests, Dropout_HTP_BF16_WithMask) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunDropoutHTPBF16Test(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                        {},
                        ExpectedEPNodeAssignment::All,
                        /*with_mask=*/true,
                        /*opset=*/22);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

#if defined(_M_ARM64)

// ---------------------------------------------------------------------------
// GPU tests
// ---------------------------------------------------------------------------

TEST_F(QnnGPUBackendTests, Dropout_Default) {
  RunDropoutTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                 {},
                 ExpectedEPNodeAssignment::All,
                 /*with_mask=*/false,
                 "gpu");
}

TEST_F(QnnGPUBackendTests, Dropout_WithMask) {
  RunDropoutTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                 {},
                 ExpectedEPNodeAssignment::All,
                 /*with_mask=*/true,
                 "gpu");
}

#endif  // defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
