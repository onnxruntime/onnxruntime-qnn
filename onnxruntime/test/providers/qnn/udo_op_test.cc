// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <limits.h>

#include <filesystem>
#include <random>
#include <string>
#include <variant>

#include "onnxruntime_session_options_config_keys.h"
#include "onnxruntime_lite_custom_op.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"

#include "core/providers/qnn/qnn_custom_op.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"
namespace onnxruntime {
namespace test {
// qnn-op-package-generator supports Python 3.10 and 3.12 on Linux x86_64 (via version-tagged
// SDK extensions). Windows is excluded by the platform guard below because the Hexagon toolchain
// only targets HTP on Linux and the Windows CI environment uses Python 3.11+.
#if defined(__linux__) && defined(__x86_64__) && defined(BUILD_QNN_UDO_TEST)
constexpr std::string_view kUdoDomain = "udo_domain";
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
    return Ort::Status{nullptr};
  }
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    Ort::ShapeInferContext::Shape shape = ctx.GetInputShape(0);
    ctx.SetOutputShape(0, shape);
    return Ort::Status{nullptr};
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
// When `register_domain_manually` is true (default), a real MyAdd kernel is registered in the
// session so the CPU EP fallback path also works. When false, no manual Ort::CustomOpDomain is
// registered — the test relies solely on the QNN EP factory having registered the domain via
// ORT_QNN_CUSTOM_OP_DOMAINS. In that case the node must be fully assigned to QNN EP (no CPU
// fallback) for the session to run correctly.
static void RunOpTestOnCPU(const std::string& op_type,
                           const TestInputDef<float>& input_def,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           const std::string& op_packages,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           bool register_domain_manually = true) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["op_packages"] = op_packages;

  if (register_domain_manually) {
    Ort::CustomOpDomain v2_domain{kUdoDomain.data()};
    std::unique_ptr<Ort::Custom::OrtLiteCustomOp> my_add_op_ptr{Ort::Custom::CreateLiteCustomOp<MyAdd>("MyAdd", "CPUExecutionProvider")};
    v2_domain.Add(my_add_op_ptr.get());
    RunQnnModelTest(BuildUDOTestCase<float>(op_type, input_def, attrs, std::string(kUdoDomain)),
                    provider_options,
                    opset_version,
                    EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(1e-5f)},
                    OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                    true,
                    &v2_domain);
  } else {
    RunQnnModelTest(BuildUDOTestCase<float>(op_type, input_def, attrs, std::string(kUdoDomain)),
                    provider_options,
                    opset_version,
                    EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(1e-5f)},
                    OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                    /*verify_outputs=*/false,
                    /*custom_op_domain=*/nullptr);
  }
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
  Ort::CustomOpDomain v2_domain{kUdoDomain.data()};
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> my_add_op_ptr{Ort::Custom::CreateLiteCustomOp<MyAdd>("MyAdd", "CPUExecutionProvider")};
  v2_domain.Add(my_add_op_ptr.get());

  TestQDQModelAccuracy<uint8_t>(BuildUDOTestCase<float>(op_type, input_def, attrs, std::string(kUdoDomain)),       // f32_model_fn
                                BuildUDOQDQTestCase<uint8_t>(op_type, input_def, attrs, std::string(kUdoDomain)),  // qdq_model_fn
                                provider_options,
                                opset_version,
                                expected_ep_assignment,
                                QDQTolerance(),
                                OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                /*qnn_ctx_model_path=*/"",
                                /*session_option_pairs=*/{},
                                /*graph_optimization_level=*/std::nullopt,
                                /*qnn_ep_graph_checker=*/nullptr,
                                /*custom_op_domain=*/&v2_domain);
}

std::string getLibPath(std::string backend) {
  /*
  Assume udo package lib is put with same directory with onnxruntime_provider_test.
  We set the path of udo package to absolute path so we can execute onnxruntime_provider_test from any path.
  */
  char path[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
  std::filesystem::path exePath(std::string(path, count));
  return exePath.parent_path() / ("libMyAddOpPackage_" + backend + ".so");
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

// Verifies the full factory-level custom-op domain registration path:
//   ORT_QNN_CUSTOM_OP_DOMAINS env var → QnnEpFactory construction → GetNumCustomOpDomains /
//   GetCustomOpDomains callbacks → ORT registers domain → ONNX model containing a
//   udo_domain::MyAdd node loads successfully with no manual Ort::CustomOpDomain.
// We bypass the RunQnnModelTest helper (which also creates a CPU-EP reference session that
// needs the domain) and directly assert that Ort::Session construction succeeds — confirming
// the factory hook delivers the domain. An "Unknown domain" exception indicates the hook
// did NOT fire, which would be a regression.
TEST_F(QnnCPUBackendTests, UDO_Op_MyAdd_AutoDomainFromEnvVar) {
  std::filesystem::path path = getLibPath("cpu");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "UDO CPU op package not found: " << path;
  }

  // Set ORT_QNN_CUSTOM_OP_DOMAINS before RegisterExecutionProviderLibrary so the factory
  // reads it during CreateEpFactories() (called from RegisterExecutionProviderLibrary).
  // The previous test unregisters the QNN library on teardown, so a fresh factory is
  // constructed for this test. Restore the env var on exit to avoid cross-test leakage.
  const int set_result = setenv("ORT_QNN_CUSTOM_OP_DOMAINS", "udo_domain:MyAdd", /*overwrite=*/1);
  ASSERT_EQ(set_result, 0) << "setenv failed";
  auto env_guard = std::unique_ptr<void, std::function<void(void*)>>(
      reinterpret_cast<void*>(1),
      [](void*) { unsetenv("ORT_QNN_CUSTOM_OP_DOMAINS"); });

  // Build a minimal ONNX model with a udo_domain::MyAdd node.
  const auto input_def = TestInputDef<float>({1, 32}, false, -1.0f, 1.0f);
  const std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      onnxruntime::test::MakeAttribute("constant", static_cast<float>(2.0))};
  const std::unordered_map<std::string, int> domain_to_version = {{"", 11}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildUDOTestCase<float>("MyAdd", input_def, attrs, std::string(kUdoDomain))(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  // Register QNN EP — AppendExecutionProvider_V2 fires GetCustomOpDomains on the factory,
  // which reads ORT_QNN_CUSTOM_OP_DOMAINS and injects udo_domain::MyAdd into session_options.
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["op_packages"] = "MyAdd:" + path.string() + ":MyAddOpPackageInterfaceProvider";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;
  // Intentionally do NOT add a domain manually here.
  session_options.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR);
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  // Session construction must succeed — the factory hook delivers the domain.
  // Throws "not a registered function/op" if GetCustomOpDomains did NOT fire.
  ASSERT_NO_THROW(Ort::Session(*GetOrtEnv(),
                               model_data.data(),
                               static_cast<int>(model_data.size()),
                               session_options));
}

TEST_F(QnnHTPBackendTests, UDO_Op_MyAdd) {
  // Skip cleanly on hosts where the HTP backend / x86 simulator libs are not usable.
  // QnnHTPBackendTests::SetUp() already gates on cached_htp_support_; this macro adds the
  // arch-floor check that the rest of the HTP test suite uses (no-op on Linux x86_64 today,
  // but keeps the test consistent with TestAddEpUsingPublicApi et al.).
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
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

#endif  // defined(__linux__) && defined(__x86_64__) && defined(BUILD_QNN_UDO_TEST)

// ── QnnUdoPlaceholderOp tests ────────────────────────────────────────────────
// These tests do not require a QNN backend or UDO op package; they verify the
// placeholder op's metadata and that its kernel returns an explicit error (not a
// silent no-op) when Compute() is accidentally invoked.

TEST(QnnEP, UDO_PlaceholderOp_Metadata) {
  onnxruntime::qnn::QnnUdoPlaceholderOp op{"MyAdd", "QNNExecutionProvider"};
  EXPECT_STREQ(op.GetName(), "MyAdd");
  EXPECT_STREQ(op.GetExecutionProviderType(), "QNNExecutionProvider");
  EXPECT_EQ(op.GetInputTypeCount(), 1u);
  EXPECT_EQ(op.GetOutputTypeCount(), 1u);
  EXPECT_EQ(op.GetInputCharacteristic(0),
            OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC);
  EXPECT_EQ(op.GetOutputCharacteristic(0),
            OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC);
  EXPECT_FALSE(op.GetVariadicInputHomogeneity());
  EXPECT_FALSE(op.GetVariadicOutputHomogeneity());
}

TEST(QnnEP, UDO_PlaceholderKernel_ComputeReturnsError) {
  onnxruntime::qnn::QnnUdoPlaceholderKernel kernel;
  OrtStatusPtr status = kernel.ComputeV2(/*context=*/nullptr);
  ASSERT_NE(status, nullptr) << "Expected an error status but got nullptr (success)";
  // Confirm it's ORT_FAIL and the message mentions 'fused'.
  EXPECT_EQ(Ort::GetApi().GetErrorCode(status), ORT_FAIL);
  std::string msg = Ort::GetApi().GetErrorMessage(status);
  EXPECT_NE(msg.find("fused"), std::string::npos) << "Error message: " << msg;
  Ort::GetApi().ReleaseStatus(status);
}

}  // namespace test
}  // namespace onnxruntime

#endif
