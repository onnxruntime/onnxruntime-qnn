// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {

/**
 * Creates a graph with a single Cast operator.
 *
 * \param shape The shape of the input and output. Input data is randomly generated with this shape.
 * \param dst_type The destination type as an instance of the DataType enum in TensorProto.
 *
 * \return A function that builds the graph with the provided builder.
 */
template <typename InputType>
static GetTestModelFn BuildCastTestCase(const std::vector<int64_t>& shape,
                                        ONNX_NAMESPACE::TensorProto_DataType dst_type) {
  return [shape, dst_type](ModelTestBuilder& builder) {
    // Random input data
    builder.MakeInput<InputType>("X", shape, static_cast<InputType>(0), static_cast<InputType>(20));
    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("to", static_cast<int64_t>(dst_type)));
    builder.AddNode(
        "cast",
        "Cast",
        {"X"},
        {"Y"},
        "",
        attributes);
    builder.MakeOutput("Y");
  };
}

static GetTestModelFn BuildCastFP64TestCase(const std::vector<int64_t>& shape) {
  // Wrap FP64 in between since QNN cannot handle model IO in FP64 currently.
  return [shape](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", shape, static_cast<float>(0), static_cast<float>(20));

    builder.AddNode("cast1",
                    "Cast",
                    {"input"},
                    {"cast1_output"},
                    kOnnxDomain,
                    {builder.MakeScalarAttribute(
                        "to",
                        static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE))});

    builder.AddNode("cast2",
                    "Cast",
                    {"cast1_output"},
                    {"output"},
                    kOnnxDomain,
                    {builder.MakeScalarAttribute(
                        "to",
                        static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT))});
    builder.MakeOutput("output");
  };
}

ProviderOptions GetProviderOption(const std::string& backend_name, bool enable_fp16_precision) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  if (backend_name == "htp") {
    if (enable_fp16_precision) {
#if defined(__linux__) && !defined(__aarch64__)
      provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
      provider_options["enable_htp_fp16_precision"] = "1";
    } else {
      provider_options["enable_htp_fp16_precision"] = "0";
    }
  }

  return provider_options;
}

/**
 * Runs a Cast model on the QNN CPU or HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param shape The shape of the input and output. Input data is randomly generated with this shape.
 * \param dst_type The destination type as an instance of the DataType enum in TensorProto.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param use_htp True to run on HTP backend. Otherwise, runs on CPU.
 */
template <typename InputType>
static void RunCastOpTest(const std::vector<int64_t>& shape, ONNX_NAMESPACE::TensorProto_DataType dst_type,
                          ExpectedEPNodeAssignment expected_ep_assignment,
                          const std::string& backend_name = "cpu",
                          bool enable_fp16_precision = true) {
  if (backend_name == "htp" && enable_fp16_precision) {
#if defined(_WIN32)
    if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
      GTEST_SKIP() << "Test requires HTP FP16 support (arch > V68).";
    }
#endif
  }

  RunQnnModelTest(BuildCastTestCase<InputType>(shape, dst_type),
                  GetProviderOption(backend_name, enable_fp16_precision),
                  13,  // opset
                  expected_ep_assignment);
}

static void RunCastFP64OpTest(const std::vector<int64_t>& shape,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              const std::string& backend_name = "cpu") {
#if defined(_WIN32)
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "Test requires HTP FP16 support (arch > V68).";
  }
#endif

  RunQnnModelTest(BuildCastFP64TestCase(shape),
                  GetProviderOption(backend_name, true),
                  13,  // opset
                  expected_ep_assignment);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
static void RunCastFP16HTPTest(const std::vector<int64_t>& shape,
                               ONNX_NAMESPACE::TensorProto_DataType dst_type,
                               ExpectedEPNodeAssignment expected_ep_assignment) {
#if defined(_WIN32)
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "Test requires HTP FP16 support (arch > V68).";
  }
#endif
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto testcase = [shape, dst_type](ModelTestBuilder& builder) {
    auto input_def_fp = TestInputDef(shape, false, static_cast<float>(0), static_cast<float>(20));
    auto input_def = ConvertToFP16InputDef(input_def_fp);
    MakeTestInput<Ort::Float16_t>(builder, "X", input_def);

    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("to", static_cast<int64_t>(dst_type)));
    builder.AddNode(
        "cast",
        "Cast",
        {"X"},
        {"Y"},
        "",
        attributes);
    builder.MakeOutput("Y");
  };

  RunQnnModelTest(testcase, provider_options, /* opset */ 13, expected_ep_assignment);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// CPU tests:
//

// Cast int32_t to float on CPU
TEST_F(QnnCPUBackendTests, TestCastInt32ToFloat) {
  RunCastOpTest<int32_t>({2, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All);
}

// Cast uint8_t to float on CPU
TEST_F(QnnCPUBackendTests, TestCastUInt8ToFloat) {
  RunCastOpTest<uint8_t>({2, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All);
}

// Cast float to int32_t on CPU
TEST_F(QnnCPUBackendTests, TestCastFloatToInt32) {
  RunCastOpTest<float>({2, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32, ExpectedEPNodeAssignment::All);
}

// Cast float to double on CPU
TEST_F(QnnCPUBackendTests, TestCastFloatToDouble) {
  RunCastFP64OpTest({2, 3}, ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Cast int32_t to float on HTP
TEST_F(QnnHTPBackendTests, TestCastInt32ToFloatHTP) {
  RunCastOpTest<int32_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All,
                         "htp", false);
}

// Cast uint8_t to float on HTP
// Fails with QNN SDK 2.35.0:
// value pair (13, 1.00000012) at index #0 don't match, which is -12 from 13
TEST_F(QnnHTPBackendTests, DISABLED_TestCastUInt8ToFloatHTP) {
  RunCastOpTest<uint8_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All,
                         "htp", false);
}

// Cast float to int32_t on HTP
TEST_F(QnnHTPBackendTests, TestCastFloatToInt32HTP) {
  RunCastOpTest<float>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32, ExpectedEPNodeAssignment::All,
                       "htp", false);
}

// Cast int64_t to int32_t on HTP
// Supported in QNN SDK 2.23
TEST_F(QnnHTPBackendTests, TestCastInt64ToInt32HTP) {
  RunCastOpTest<int64_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                         ExpectedEPNodeAssignment::All, "htp");
}

// Cast int32_t to int64_t on HTP
// Supported in QNN SDK 2.23
TEST_F(QnnHTPBackendTests, TestCastInt32ToInt64HTP) {
  RunCastOpTest<int32_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                         ExpectedEPNodeAssignment::All, "htp");
}

// Cast float to bool on HTP.
TEST_F(QnnHTPBackendTests, TestCastFloatToBoolHTP) {
  RunCastOpTest<float>({3, 3},
                       ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL,
                       ExpectedEPNodeAssignment::All,
                       "htp");
}

// Cast float16 to bool on HTP.
TEST_F(QnnHTPBackendTests, TestCastFloat16ToBoolHTP) {
  RunCastFP16HTPTest({3, 3},
                     ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL,
                     ExpectedEPNodeAssignment::All);
}

// Cast float to double on HTP.
TEST_F(QnnHTPBackendTests, TestCastFloatToDoubleHTP) {
  RunCastFP64OpTest({3, 3}, ExpectedEPNodeAssignment::All, "htp");
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)
//
// GPU tests:
//

// Cast int32 to float on GPU
TEST_F(QnnGPUBackendTests, TestCastInt32ToFloat) {
  RunCastOpTest<int32_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All,
                         "gpu", false);
}

// Cast uint8 to float on GPU
TEST_F(QnnGPUBackendTests, TestCastUInt8ToFloat) {
  RunCastOpTest<uint8_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All,
                         "gpu", false);
}

// Cast float to int32 on GPU
TEST_F(QnnGPUBackendTests, TestCastFloatToInt32) {
  RunCastOpTest<float>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32, ExpectedEPNodeAssignment::All,
                       "gpu", false);
}

// Cast int64 to int32 on GPU
TEST_F(QnnGPUBackendTests, TestCastInt64ToInt32) {
  RunCastOpTest<int64_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                         ExpectedEPNodeAssignment::All, "gpu");
}

// Cast int32 to int64 on GPU
// Disable Reason : Currently not supported.
// Can enable after CastOp int32 to int64 is implemented in QnnGpu.
TEST_F(QnnGPUBackendTests, DISABLED_TestCastInt32ToInt64) {
  RunCastOpTest<int32_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                         ExpectedEPNodeAssignment::All, "gpu");
}

#endif  // defined(_M_ARM64) GPU tests

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
