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
// #include "onnx/defs/schema.h"
static std::string onnx_domain = "my_domain";
// namespace ONNX_NAMESPACE {
//   ONNX_OPERATOR_SET_SCHEMA(
//       Increment,
//       1,
//       ONNX_NAMESPACE::OpSchema()
//           .SetDomain(onnx_domain)
//           .SinceVersion(11)
//           .SetDoc("My custom op that increment input by constant c")
//           .Input(0, "X", "Input", "T")
//           .Attr("constant", "constant value.", ONNX_NAMESPACE::AttributeProto::FLOAT, 1.0f)
//           .Output(0, "Y", "Output", "T")
//           .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output to float tensors")
//           .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));
// } // namespace ONNX_NAMESPACE
namespace onnxruntime {
namespace test {


/*
The following is a custom op that registered in onnx domain for demo purpose.
The logic of increment op is y = f(x) = x+c where x is input and c is attribute.
*/
struct Increment {
  Increment(const OrtApi* ort_api, const OrtKernelInfo* info) {
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
// class Increment : public OpKernel {
//  public:
//   explicit Increment(const OpKernelInfo& info) : OpKernel(info), constant_(info.GetAttrOrDefault<float>("constant", 1.0f)) {}
//   Status Compute(OpKernelContext* context) const override {
//     auto input_tensor = context->Input<Tensor>(0);
//     if (input_tensor == nullptr) {
//       return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
//     }
//     auto data_type = input_tensor->DataType();
//     if (data_type != DataTypeImpl::GetType<float>()) {
//       return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
//                              "Unsupportted tensor data type:",
//                              data_type);
//     }
//     const Tensor* input = context->Input<Tensor>(0);
//     const float* input_data = input->Data<float>();

//     Tensor* output = context->Output(0, input->Shape());
//     float* output_data = output->MutableData<float>();
//     int64_t elem_count = input->Shape().Size();
//     for (int64_t i = 0; i < elem_count; i++) {
//       output_data[i] = input_data[i] + constant_;
//     }
//     return Status::OK();
//   }

//  private:
//   float constant_;
// };

// // register custom schema
// ONNX_NAMESPACE::OpSchema& RegisterIncrementOpSchema(ONNX_NAMESPACE::OpSchema&& op_schema) {
//   printf("%s[%d] register op\n", __FILE__, __LINE__);
//   auto& obj = OpSchemaRegistry::DomainToVersionRange::Instance();
//   obj.AddDomainToVersion(onnx_domain, 1, 11);
//   ONNX_NAMESPACE::OpSchema& res = op_schema
//       .SetDomain(onnx_domain)
//       .SinceVersion(11)
//       .SetDoc("My custom op that increment input by constant c")
//       .Input(0, "X", "Input", "T")
//       .Attr("constant", "constant value.", ONNX_NAMESPACE::AttributeProto::FLOAT, 1.0f)
//       .Output(0, "Y", "Output", "T")
//       .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output to float tensors")
//       .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);
//   printf("%s[%d] register op end\n", __FILE__, __LINE__);
//   return res;
// }
// #define ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(name, schema_func) \
//   ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(__COUNTER__, name, schema_func)
// #define ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(Counter, name, schema_func) \
//   ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func)
// #define ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func) \
//   static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(                \
//       op_schema_register_once##name##Counter) ONNX_UNUSED =                     \
//       schema_func(ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__))
// ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(Increment, RegisterIncrementOpSchema);
// class Increment_Onnx_ver1;
// template <>
// OpSchema ONNX_NAMESPACE::GetOpSchema<Increment_Onnx_ver1>() { 
//   return ONNX_NAMESPACE::OpSchema().SetDomain(onnx_domain).SinceVersion(11).SetDoc("My custom op that increment input by constant c").Input(0, "X", "Input", "T").Attr("constant", "constant value.", ONNX_NAMESPACE::AttributeProto::FLOAT, 1.0f).Output(0, "Y", "Output", "T").TypeConstraint("T", {"tensor(float)"}, "Constrain input and output to float tensors").TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput).SetName("Increment").SetDomain(ONNX_DOMAIN).SinceVersion(1).SetLocation("C:\\Users\\CHENWENG\\onnxruntime-qnn\\onnxruntime\\test\\providers\\qnn\\udo\\udo_op_test.cpp", 123); 
// }
// static size_t dbg_count_check_Onnx_1_verIncrement [[maybe_unused]] = (true) ? DbgOperatorSetTracker::Instance().IncrementCount() : 0;
// static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( op_schema_register_onceIncrement3)  = RegisterIncrementOpSchema(ONNX_NAMESPACE::OpSchema("Increment", "C:\\Users\\CHENWENG\\onnxruntime-qnn\\onnxruntime\\test\\providers\\qnn\\udo\\udo_op_test.cpp", 108))
// void register_custom_op_to_cpu_provider(const std::string& op_type, KernelCreateInfo (*funcPtr)()) {
//   // kernel_registry is static variable which will not be release across difference gtest, use a static set to avoid
//   // duplicate registration.
//   static std::set<std::string> registered_udo_ops;
//   if (registered_udo_ops.find(op_type) == registered_udo_ops.end()) {
//     auto cpu_provider = CPUExecutionProvider(CPUExecutionProviderInfo());
//     std::shared_ptr<KernelRegistry> kernel_registry = cpu_provider.GetKernelRegistry();
//     ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(funcPtr())));
//     registered_udo_ops.insert(op_type);
//   }
// }

// UDO unit tests

// function to return KernelCreateInfo
// KernelCreateInfo BuildIncrementKernelCreateInfo() {
//   return KernelCreateInfo(KernelDefBuilder()
//                               .SetName("Increment")
//                               .SetDomain(onnx_domain)
//                               .SinceVersion(1)
//                               .Provider(kCpuExecutionProvider)
//                               .Build(),
//                           static_cast<KernelCreatePtrFn>([](
//                                                              FuncManager&,
//                                                              const OpKernelInfo& info,
//                                                              std::unique_ptr<OpKernel>& out) -> Status {
//                             out = std::make_unique<Increment>(info);
//                             return Status::OK();
//                           }));
// }

template <typename InputType>
static GetTestModelFn BuildUDOTestCase(const std::string& op_type,
                                       const std::vector<TestInputDef<InputType>>& input_defs,
                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                       const std::string& op_domain) {
  return [op_type, input_defs, attrs, op_domain](ModelTestBuilder& builder) {
    std::vector<std::string> op_input_names;
    for (int i = 0; i < input_defs.size(); i++) {
      std::string input_name = "input_" + std::to_string(i);
      MakeTestInput(builder, input_name, input_defs[i]);
      op_input_names.push_back(input_name);
    }

    builder.MakeOutput("Y");
    builder.AddNode(
        op_type,
        op_type,
        op_input_names,
        {"Y"},
        op_domain,
        attrs);
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
// struct ExampleKernel {
//   void Compute(OrtKernelContext* context) {
//     auto input_tensor = context->Input<Tensor>(0);
//     if (input_tensor == nullptr) {
//       return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
//     }
//     auto data_type = input_tensor->DataType();
//     if (data_type != DataTypeImpl::GetType<float>()) {
//       return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
//                             "Unsupportted tensor data type:",
//                             data_type);
//     }
//     const Tensor* input = context->Input<Tensor>(0);
//     const float* input_data = input->Data<float>();

//     Tensor* output = context->Output(0, input->Shape());
//     float* output_data = output->MutableData<float>();
//     int64_t elem_count = input->Shape().Size();
//     for (int64_t i = 0; i < elem_count; i++) {
//       output_data[i] = input_data[i] + constant_;
//     }
//   }
// };
// struct ExampleOp : Ort::CustomOpBase<ExampleOp, ExampleKernel> {
//   const char* GetName() const { return "Example"; }
//   const char* GetExecutionProviderType() const { return nullptr; } // CPU by default
//   size_t GetInputTypeCount() const { return 1; }
//   ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
//   size_t GetOutputTypeCount() const { return 1; }
//   ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
//   void* CreateKernel(const OrtApi&, const OrtKernelInfo*) const { return new ExampleKernel(); }
// } g_example_op;

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
  // auto lite =
  // std::unique_ptr<Ort::Custom::OrtLiteCustomOp>{Ort::Custom::CreateLiteCustomOp("Increment", "CPUExecutionProvider", Filter)};

  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> increment_op_ptr{Ort::Custom::CreateLiteCustomOp<Increment>("Increment", "CPUExecutionProvider")};
  v2_domain.Add(increment_op_ptr.get());
  Ort::SessionOptions session_options;
  session_options.Add(v2_domain);
  // session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");

  // register_custom_op_to_cpu_provider(op_type, funcPtr);
  RunQnnEpTest(BuildUDOTestCase<float>(op_type, input_defs, attrs, onnx_domain),
               provider_options, session_options,
               opset_version, fetches);
}

// Runs a QDQ model on the QNN HTP backend and compares output to CPU EP.
// static void RunOpTestOnHTP(const std::string& op_type,
//                            const std::vector<TestInputDef<float>>& input_defs,
//                            const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
//                            const std::string& op_packages,
//                            KernelCreateInfo (*funcPtr)(),
//                            int opset_version,
//                            ExpectedEPNodeAssignment expected_ep_assignment) {
//   ProviderOptions provider_options;
//   provider_options["backend_type"] = "htp";
//   provider_options["offload_graph_io_quantization"] = "0";
//   provider_options["op_packages"] = op_packages;

//   // register_custom_op_to_cpu_provider(op_type, funcPtr);
//   TestQDQModelAccuracy<uint8_t>(BuildUDOTestCase<float>(op_type, input_defs, attrs, onnx_domain),       // baseline float32 model
//                                 BuildQDQOpTestCase<uint8_t>(op_type, op_type, input_defs, {}, attrs, onnx_domain),  // QDQ model
//                                 provider_options,
//                                 opset_version,
//                                 expected_ep_assignment);
// }

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
  return (exePath.parent_path() / ("IncrementOpPackage_" + backend + ".dll")).string();
#else
  char path[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
  std::filesystem::path exePath(std::string(path, count));
  return exePath.parent_path() / ("libIncrementOpPackage_" + backend + ".so");
#endif
}

TEST_F(QnnCPUBackendTests, UDO_Op_Increment) {
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
  RunOpTestOnCPU("Increment",
                 {input},
                 {},
                 "Increment:" + path.string() + ":IncrementOpPackageInterfaceProvider",
                 fetches,
                 11,
                 ExpectedEPNodeAssignment::All);
  const float* output_data = fetches[0].GetTensorData<float>();
  for (int i = 0; i < element_count; ++i) {
    input_data[i] += 1.0;
    EXPECT_EQ(input_data[i], output_data[i]) << "Element " << i << " mismatch. Expected: " << input_data[i] << ", actual: " << output_data[i];
  }
}

// TEST_F(QnnCPUBackendTests, UDO_Op_Increment_with_constant) {
//   auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
//   std::filesystem::path path = getLibPath("cpu");
//   RunOpTestOnCPU("Increment",
//                  {input},
//                  {MakeAttribute("constant", static_cast<float>(2.0))},
//                  "Increment:" + path.string() + ":IncrementOpPackageInterfaceProvider",
//                  11,
//                  ExpectedEPNodeAssignment::All);
// }

// #if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// TEST_F(QnnHTPBackendTests, UDO_Op_Increment) {
//   auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
//   std::filesystem::path path = getLibPath("htp");
//   RunOpTestOnHTP("Increment",
//                  {input},
//                  {},
//                  "Increment:" + path.string() + ":IncrementOpPackageInterfaceProvider:CPU",
//                  11,
//                  ExpectedEPNodeAssignment::All);
// }

// TEST_F(QnnHTPBackendTests, UDO_Op_Increment_with_constant) {
//   auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
//   std::filesystem::path path = getLibPath("htp");
//   RunOpTestOnHTP("Increment",
//                  {input},
//                  {utils::MakeAttribute("constant", static_cast<float>(2.0))},
//                  "Increment:" + path.string() + ":IncrementOpPackageInterfaceProvider:CPU",
//                  11,
//                  ExpectedEPNodeAssignment::All);
// }

// #endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif