// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#if defined(_WIN32)
#include <windows.h>
#else
#include <limits.h>
#endif

#include <filesystem>
#include <string>
#include <variant>

#include "core/framework/op_kernel.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn-abi/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Leaving onnx_domain empty ("") registers the custom op in the default ONNX domain for UDO unit testing.
static std::string onnx_domain = "";

class Increment : public OpKernel {
 public:
  explicit Increment(const OpKernelInfo& info) : OpKernel(info), constant_(info.GetAttrOrDefault<float>("constant", 1.0f)) {}
  Status Compute(OpKernelContext* context) const override {
    auto input_tensor = context->Input<Tensor>(0);
    if (input_tensor == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    }
    auto data_type = input_tensor->DataType();
    if (data_type != DataTypeImpl::GetType<float>()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupportted tensor data type:",
                             data_type);
    }
    const Tensor* input = context->Input<Tensor>(0);
    const float* input_data = input->Data<float>();

    Tensor* output = context->Output(0, input->Shape());
    float* output_data = output->MutableData<float>();
    int64_t elem_count = input->Shape().Size();
    for (int64_t i = 0; i < elem_count; i++) {
      output_data[i] = input_data[i] + constant_;
    }
    return Status::OK();
  }

 private:
  float constant_;
};

// register custom schema
ONNX_NAMESPACE::OpSchema& RegisterIncrementOpSchema(ONNX_NAMESPACE::OpSchema&& op_schema) {
  return op_schema
      .SetDomain(onnx_domain)
      .SinceVersion(1)
      .SetDoc("My custom op that increment input by constant c")
      .Input(0, "X", "Input", "T")
      .Attr("constant", "constant value.", ONNX_NAMESPACE::AttributeProto::FLOAT, 1.0f)
      .Output(0, "Y", "Output", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output to float tensors")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);
}
ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(Increment, RegisterIncrementOpSchema);

void register_custom_op_to_cpu_provider(const std::string& op_type, KernelCreateInfo (*funcPtr)()) {
  // kernel_registry is static variable which will not be release across difference gtest, use a static set to avoid
  // duplicate registration.
  static std::set<std::string> registered_udo_ops;
  if (registered_udo_ops.find(op_type) == registered_udo_ops.end()) {
    auto cpu_provider = CPUExecutionProvider(CPUExecutionProviderInfo());
    std::shared_ptr<KernelRegistry> kernel_registry = cpu_provider.GetKernelRegistry();
    ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(funcPtr())));
    registered_udo_ops.insert(op_type);
  }
}

// UDO unit tests

// function to return KernelCreateInfo
KernelCreateInfo BuildIncrementKernelCreateInfo() {
  return KernelCreateInfo(KernelDefBuilder()
                              .SetName("Increment")
                              .SetDomain(onnx_domain)
                              .SinceVersion(1)
                              .Provider(kCpuExecutionProvider)
                              .Build(),
                          static_cast<KernelCreatePtrFn>([](
                                                             FuncManager&,
                                                             const OpKernelInfo& info,
                                                             std::unique_ptr<OpKernel>& out) -> Status {
                            out = std::make_unique<Increment>(info);
                            return Status::OK();
                          }));
}

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
    ORT_UNUSED_PARAMETER(output_qparams);
    std::vector<NodeArg*> qdq_inputs;
    for (auto it = input_defs.begin(); it != input_defs.end(); it++) {
      QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(*it);
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
                           KernelCreateInfo (*funcPtr)(),
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["op_packages"] = op_packages;

  register_custom_op_to_cpu_provider(op_type, funcPtr);
  RunQnnModelTestABI(BuildUDOTestCase<float>(op_type, input_defs, attrs, onnx_domain),
                     provider_options,
                     opset_version,
                     expected_ep_assignment);
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

  register_custom_op_to_cpu_provider(op_type, funcPtr);
  TestQDQModelAccuracy<uint8_t>(BuildUDOTestCase<float>(op_type, input_defs, attrs, onnx_domain),       // baseline float32 model
                                BuildUDOQDQTestCase<uint8_t>(op_type, input_defs, attrs, onnx_domain),  // QDQ model
                                provider_options,
                                opset_version,
                                expected_ep_assignment);
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
  return (exePath.parent_path() / ("IncrementOpPackage_" + backend + ".dll")).string();
#else
  char path[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
  std::filesystem::path exePath(std::string(path, count));
  return exePath.parent_path() / ("libIncrementOpPackage_" + backend + ".so");
#endif
}

TEST_F(QnnABICPUBackendTests, UDO_Op_Increment) {
  auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
  std::filesystem::path path = getLibPath("cpu");
  RunOpTestOnCPU("Increment",
                 {input},
                 {},
                 "Increment:" + path.string() + ":IncrementOpPackageInterfaceProvider",
                 BuildIncrementKernelCreateInfo,
                 11,
                 ExpectedEPNodeAssignment::All);
}

TEST_F(QnnABICPUBackendTests, UDO_Op_Increment_with_constant) {
  auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
  std::filesystem::path path = getLibPath("cpu");
  RunOpTestOnCPU("Increment",
                 {input},
                 {utils::MakeAttribute("constant", static_cast<float>(2.0))},
                 "Increment:" + path.string() + ":IncrementOpPackageInterfaceProvider",
                 BuildIncrementKernelCreateInfo,
                 11,
                 ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnABIHTPBackendTests, UDO_Op_Increment) {
  auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
  std::filesystem::path path = getLibPath("htp");
  RunOpTestOnHTP("Increment",
                 {input},
                 {},
                 "Increment:" + path.string() + ":IncrementOpPackageInterfaceProvider:CPU",
                 BuildIncrementKernelCreateInfo,
                 11,
                 ExpectedEPNodeAssignment::All);
}

TEST_F(QnnABIHTPBackendTests, UDO_Op_Increment_with_constant) {
  auto input = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
  std::filesystem::path path = getLibPath("htp");
  RunOpTestOnHTP("Increment",
                 {input},
                 {utils::MakeAttribute("constant", static_cast<float>(2.0))},
                 "Increment:" + path.string() + ":IncrementOpPackageInterfaceProvider:CPU",
                 BuildIncrementKernelCreateInfo,
                 11,
                 ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
