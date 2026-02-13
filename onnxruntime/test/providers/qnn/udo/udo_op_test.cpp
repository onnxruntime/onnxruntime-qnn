// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#if defined(_WIN32)
#include <windows.h>
#else
#include <limits.h>
#endif

#include <filesystem>
#include <random>
#include <string>
#include <variant>

#include "core/framework/op_kernel.h"
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


/*
The following is a custom op that registered in udo_domain for demo purpose.
The logic of MyAdd op is (y = x + c) where x is input and c is attribute.
*/
struct MyAdd {
  MyAdd(const OrtApi* ort_api, const OrtKernelInfo* info) {
    ort_api->KernelInfoGetAttribute_float(info, "constant", &constant_);
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
                                       const std::vector<TestInputDef<InputType>>& input_defs,
                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                       const std::string& op_domain) {
  return [op_type, input_defs, attrs, op_domain](ModelTestBuilder& builder) {
    std::vector<NodeArg*> inputs;
    for (auto it = input_defs.begin(); it != input_defs.end(); it++) {
      NodeArg* input = MakeTestInput(builder, *it);
      inputs.push_back(input);
    }

    auto* Y = builder.MakeOutput();
    Node& node = builder.AddNode(op_type, inputs, {Y}, op_domain);
    for (const auto& attr : attrs) {
      node.AddAttributeProto(attr);
    }
  };
}

// Builds a QDQ model. The quantization parameters are computed from the provided input definition.
template <typename InputQType>
static GetTestQDQModelFn<InputQType> BuildUDOQDQTestCase(const std::string& op_type,
                                                         const std::vector<TestInputDef<float>>& input_defs,
                                                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                         const std::string& op_domain) {
  return [op_type, input_defs, attrs, op_domain](ModelTestBuilder& builder,
                                                 std::vector<QuantParams<InputQType>>& output_qparams) {
    std::vector<std::string> op_input_names;
    for (int i = 0; i < input_defs.size(); i++) {
      QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_defs[i]);
      // input -> Q -> DQ ->
      NodeArg* input = MakeTestInput(builder, *it);
      NodeArg* qdq_input = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale, input_qparams.zero_point);
      qdq_inputs.push_back(qdq_input);
    }

    auto* Y = builder.MakeIntermediate();
    Node& node = builder.AddNode(op_type, qdq_inputs, {Y}, op_domain);
    for (const auto& attr : attrs) {
      node.AddAttributeProto(attr);
    }
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, Y, output_qparams[0].scale,
                                                      output_qparams[0].zero_point);
  };
}

// Runs a non-QDQ model on the QNN CPU backend and compares output to CPU EP.
static void RunOpTestOnCPU(const std::string& op_type,
                           const std::vector<TestInputDef<float>>& input_defs,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           const std::string& op_packages,
                           std::vector<Ort::Value>& fetches,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["op_packages"] = op_packages;
  Ort::CustomOpDomain v2_domain{onnx_domain.c_str()};
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> MyAdd_op_ptr{Ort::Custom::CreateLiteCustomOp<MyAdd>("MyAdd", "CPUExecutionProvider")};
  v2_domain.Add(MyAdd_op_ptr.get());
  std::shared_ptr<Ort::SessionOptions> session_options = std::make_shared<Ort::SessionOptions>();
  session_options->Add(v2_domain);

  // RunQnnModelTest(BuildUDOTestCase<float>(op_type, input_defs, attrs, onnx_domain),
  //                 provider_options,
  //                 opset,
  //                 expected_ep_assignment,
  //                 tolerance);
  RunQnnModelTest(BuildUDOTestCase<float>(op_type, input_defs, attrs, onnx_domain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  1e-5f,
                  session_options);
}

// Runs a QDQ model on the QNN HTP backend and compares output to CPU EP.
static void RunOpTestOnHTP(const std::string& op_type,
                           const std::vector<TestInputDef<float>>& input_defs,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           const std::string& op_packages,
                           KernelCreateInfo (*funcPtr)(),
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["op_packages"] = op_packages;
  Ort::CustomOpDomain v2_domain{onnx_domain.c_str()};
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> MyAdd_op_ptr{Ort::Custom::CreateLiteCustomOp<MyAdd>("MyAdd", "CPUExecutionProvider")};
  v2_domain.Add(MyAdd_op_ptr.get());
  std::shared_ptr<Ort::SessionOptions> session_options = std::make_shared<Ort::SessionOptions>();
  session_options->Add(v2_domain);
  session_options->AddConfigEntry("session.disable_cpu_ep_fallback", "1");

  TestQDQModelAccuracy<uint8_t>(BuildUDOTestCase<float>(op_type, input_defs, attrs, onnx_domain),       // baseline float32 model
                                BuildQDQOpTestCase<uint8_t>(op_type, input_defs, {}, attrs, onnx_domain),  // QDQ model
                                provider_options,
                                opset_version,
                                expected_ep_assignment,
                                QDQTolerance(),
                                session_options);
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
  std::vector<float> input_data;
  std::random_device rd;
  std::mt19937 gen(rd());
  int element_count = 32;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < element_count; i++) {
    input_data.push_back(dist(gen));
  }
  auto input = TestInputDef<float>({1, element_count}, false, input_data);
  std::vector<Ort::Value> fetches;
  std::filesystem::path path = getLibPath("cpu");
  RunOpTestOnCPU("MyAdd",
                 {input},
                 {},
                 "MyAdd:" + path.string() + ":MyAddOpPackageInterfaceProvider",
                 fetches,
                 11,
                 ExpectedEPNodeAssignment::All);
  const float* output_data = fetches[0].GetTensorData<float>();
  for (int i = 0; i < element_count; ++i) {
    input_data[i] += 1.0;
    EXPECT_EQ(input_data[i], output_data[i]) << "Element " << i << " mismatch. Expected: " << input_data[i] << ", actual: " << output_data[i];
  }
}

// TEST_F(QnnCPUBackendTests, UDO_Op_MyAdd_with_constant) {
//   auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
//   std::filesystem::path path = getLibPath("cpu");
//   RunOpTestOnCPU("MyAdd",
//                  {input},
//                  {MakeAttribute("constant", static_cast<float>(2.0))},
//                  "MyAdd:" + path.string() + ":MyAddOpPackageInterfaceProvider",
//                  11,
//                  ExpectedEPNodeAssignment::All);
// }

// #if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// TEST_F(QnnHTPBackendTests, UDO_Op_MyAdd) {
//   auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
//   std::filesystem::path path = getLibPath("htp");
//   RunOpTestOnHTP("MyAdd",
//                  {input},
//                  {},
//                  "MyAdd:" + path.string() + ":MyAddOpPackageInterfaceProvider:CPU",
//                  11,
//                  ExpectedEPNodeAssignment::All);
// }

// TEST_F(QnnHTPBackendTests, UDO_Op_MyAdd_with_constant) {
//   auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
//   std::filesystem::path path = getLibPath("htp");
//   RunOpTestOnHTP("MyAdd",
//                  {input},
//                  {utils::MakeAttribute("constant", static_cast<float>(2.0))},
//                  "MyAdd:" + path.string() + ":MyAddOpPackageInterfaceProvider:CPU",
//                  11,
//                  ExpectedEPNodeAssignment::All);
// }

// #endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
