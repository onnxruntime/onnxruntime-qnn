// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#if defined(_WIN32)
#include <windows.h>
#else
#include <limits.h>
#endif

#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <variant>

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_c_api.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"
static std::string onnx_domain = "udo_domain";
namespace onnxruntime {
namespace test {

// The CPU UDO op-package is supported on both Linux x86_64 and Windows x86 hosts.
// HTP UDO compilation requires the Hexagon SDK toolchain, which is wired up only on Linux x86_64
// (see docs/execution_providers/QNN_UDO.md for details).
/*
The following is a custom op that registered in udo_domain for demo purpose.
The logic of MyAdd op is (y = x + c) where x is input and c is attribute.
*/
struct MyAdd {
  MyAdd(const OrtApi* ort_api, const OrtKernelInfo* info) {
    // 'constant' is optional; keep the default value (1.0) when the attribute is absent.
    OrtStatus* status = ort_api->KernelInfoGetAttribute_float(info, "constant", &constant_);
    if (status != nullptr) {
      ort_api->ReleaseStatus(status);
    }
  }
  Ort::Status Compute(const Ort::Custom::Tensor<float>& X,
                      Ort::Custom::Tensor<float>* Y) {
    const std::vector<int64_t>& shape = X.Shape();
    const float* input_data = X.Data();
    float* output_data = Y->Allocate(shape);
    for (int i = 0; i < X.NumberOfElement(); i++) {
      output_data[i] = input_data[i] + constant_;
    }
    return Ort::Status(nullptr);
  }
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    Ort::ShapeInferContext::Shape shape = ctx.GetInputShape(0);
    ctx.SetOutputShape(0, shape);
    return Ort::Status(nullptr);
  }
  float constant_ = 1.0;
};

template <typename InputType>
static GetTestModelFn BuildUDOTestCase(const std::string& op_type,
                                       const TestInputDef<InputType>& input_def,
                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                       const std::string& op_domain) {
  return [op_type, input_def, attrs, op_domain](ModelTestBuilder& builder) {
    auto* opset = builder.model_.add_opset_import();
    opset->set_domain(op_domain);
    opset->set_version(1);
    MakeTestInput<float>(builder, "input", input_def);
    builder.AddNode(
        op_type,
        op_type,
        {"input"},
        {"output"},
        op_domain,
        attrs);
    builder.MakeOutput("output");
  };
}

// Builds a QDQ model. The quantization parameters are computed from the provided input definition.
template <typename InputQType>
static GetTestQDQModelFn<InputQType> BuildUDOQDQTestCase(const std::string& op_type,
                                                         const TestInputDef<float>& input_def,
                                                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                         const std::string& op_domain) {
  return [op_type, input_def, attrs, op_domain](ModelTestBuilder& builder,
                                                std::vector<QuantParams<InputQType>>& output_qparams) {
    auto* opset = builder.model_.add_opset_import();
    opset->set_domain(op_domain);
    opset->set_version(1);
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    std::string input_qdq = AddQDQNodePair<InputQType>(builder, "input_qdq", "input", input_qparams.scale, input_qparams.zero_point);
    builder.AddNode(
        op_type,
        op_type,
        {input_qdq},
        {"output"},
        op_domain,
        attrs);
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "output_qdq", "output", output_qparams[0].scale,
                                                      output_qparams[0].zero_point);
  };
}

// Runs a non-QDQ model on the QNN CPU backend and compares output to CPU EP.
static void RunOpTestOnCPU(const std::string& op_type,
                           const TestInputDef<float>& input_def,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           const std::string& op_packages,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["op_packages"] = op_packages;
  Ort::CustomOpDomain v2_domain{onnx_domain.c_str()};
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> MyAdd_op_ptr{Ort::Custom::CreateLiteCustomOp<MyAdd>("MyAdd", "CPUExecutionProvider")};
  v2_domain.Add(MyAdd_op_ptr.get());

  RunQnnModelTest(BuildUDOTestCase<float>(op_type, input_def, attrs, onnx_domain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  1e-5f,
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  true,
                  nullptr,
                  &v2_domain);
}

// Runs a QDQ model on the QNN HTP backend and compares output to CPU EP.
static void RunOpTestOnHTP(const std::string& op_type,
                           const TestInputDef<float>& input_def,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           const std::string& op_packages,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["op_packages"] = op_packages;
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  Ort::CustomOpDomain v2_domain{onnx_domain.c_str()};
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> MyAdd_op_ptr{Ort::Custom::CreateLiteCustomOp<MyAdd>("MyAdd", "CPUExecutionProvider")};
  v2_domain.Add(MyAdd_op_ptr.get());

  TestQDQModelAccuracy<uint8_t>(BuildUDOTestCase<float>(op_type, input_def, attrs, onnx_domain),       // baseline float32 model
                                BuildUDOQDQTestCase<uint8_t>(op_type, input_def, attrs, onnx_domain),  // QDQ model
                                provider_options,
                                opset_version,
                                expected_ep_assignment,
                                QDQTolerance(),
                                OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                "",
                                {},
                                std::nullopt,
                                nullptr,
                                &v2_domain);
}

std::string getLibPath(std::string backend) {
  /*
  Assume udo package lib is put with same directory with onnxruntime_provider_test.
  We set the path of udo package to absolute path so we can execute onnxruntime_provider_test from any path.
  */
#if defined(_WIN32)
  char path[MAX_PATH];
  DWORD result = GetModuleFileNameA(NULL, path, MAX_PATH);
  if (result == 0) {
    std::cerr << "Failed to get executable path." << std::endl;
    return "";
  }
  std::filesystem::path exePath(path);
  return (exePath.parent_path() / ("MyAddOpPackage_" + backend + ".dll")).string();
#else
  char path[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
  std::filesystem::path exePath(std::string(path, count));
  return exePath.parent_path() / ("libMyAddOpPackage_" + backend + ".so");
#endif
}

TEST_F(QnnCPUBackendTests, UDO_Op_MyAdd) {
  auto input_def = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
  std::filesystem::path path = getLibPath("cpu");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "UDO CPU op package not found: " << path;
  }
  RunOpTestOnCPU("MyAdd",
                 input_def,
                 {onnxruntime::test::MakeAttribute("constant", static_cast<float>(2.0))},
                 "MyAdd:" + path.string() + ":MyAddOpPackageInterfaceProvider",
                 11,
                 ExpectedEPNodeAssignment::All);
}

// HTP UDO compilation requires the Hexagon SDK toolchain (currently Linux x86_64 only).
#if defined(__linux__) && defined(__x86_64__)
TEST_F(QnnHTPBackendTests, UDO_Op_MyAdd) {
  auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
  std::filesystem::path path = getLibPath("htp");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "UDO HTP op package not found: " << path;
  }
  RunOpTestOnHTP("MyAdd",
                 {input},
                 {onnxruntime::test::MakeAttribute("constant", static_cast<float>(2.0))},
                 "MyAdd:" + path.string() + ":MyAddOpPackageInterfaceProvider:CPU",
                 11,
                 ExpectedEPNodeAssignment::All);
}
#endif  // defined(__linux__) && defined(__x86_64__)

}  // namespace test
}  // namespace onnxruntime

#endif
