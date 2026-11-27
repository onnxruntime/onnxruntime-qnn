// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <chrono>
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#endif

#include <filesystem>
#include <random>
#include <mutex>
#include <string>
#include <variant>

#include "onnxruntime_session_options_config_keys.h"
#include "onnxruntime_lite_custom_op.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"
namespace onnxruntime {
namespace test {
// Construct before registering the EP library: each registration creates a factory
// that reads the domain variable. Restore the caller's value when the session ends.
class ScopedUdoDomain {
 public:
  explicit ScopedUdoDomain(const char* value) {
    const char* old = std::getenv("ORT_QNN_CUSTOM_OP_DOMAINS");
    if (old != nullptr) original_ = old;
    Set(value);
  }
  ~ScopedUdoDomain() { Set(original_ ? original_->c_str() : nullptr); }
  ScopedUdoDomain(const ScopedUdoDomain&) = delete;
  ScopedUdoDomain& operator=(const ScopedUdoDomain&) = delete;

 private:
  static void Set(const char* value) {
#ifdef _WIN32
    const int result = _putenv_s("ORT_QNN_CUSTOM_OP_DOMAINS", value ? value : "");
#else
    const int result = value ? setenv("ORT_QNN_CUSTOM_OP_DOMAINS", value, 1)
                             : unsetenv("ORT_QNN_CUSTOM_OP_DOMAINS");
#endif
    EXPECT_EQ(result, 0);
  }
  std::optional<std::string> original_;
};

constexpr std::string_view kUdoDomain = "udo_domain";
constexpr float kMyAddConstant = 2.0f;
constexpr float kOutputQuantScale = 4.0f / 255.0f;
constexpr float kOutputTolerance = 2.0f * kOutputQuantScale;
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
    // Declare the output type and shape explicitly so ORT's type-inference pass can
    // resolve the output type without a custom shape-inference function on the placeholder op.
    // Real ONNX models exported by conversion tools carry this information in the graph's
    // output value_info; this mirrors that behavior.
    builder.MakeOutput<float>("output", input_def.GetShape());
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
    // The automatic domain uses an untyped variadic placeholder. Exporters provide
    // value_info for custom-op outputs; retain it for model type and shape inference.
    auto* value_info = builder.model_.mutable_graph()->add_value_info();
    value_info->set_name("output");
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (const auto dim : input_def.GetShape()) tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "output_qdq", "output", output_qparams[0].scale,
                                                      output_qparams[0].zero_point);
  };
}

// Runs a QDQ model on the QNN HTP backend and compares output to CPU EP.
static void RunOpTestOnHTP(const std::string& op_type,
                           const TestInputDef<float>& input_def,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           const ProviderOptions& provider_options,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment) {
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

// Lookup is independent of the process working directory and the build toolchain.
static std::filesystem::path ExecutableDirectory() {
#ifdef _WIN32
  std::wstring path(32768, L'\0');
  const auto count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (count == 0 || count == path.size()) throw std::runtime_error("Cannot locate test executable");
  path.resize(count);
  return std::filesystem::path(path).parent_path();
#else
  return std::filesystem::canonical("/proc/self/exe").parent_path();
#endif
}

struct UdoArtifact {
  std::filesystem::path path;
  std::filesystem::path dsp;
  std::string searched;
};

static UdoArtifact FindUdoArtifact(const std::filesystem::path& executable_dir,
                                   const std::string& override_dir) {
  std::vector<std::filesystem::path> dirs;
  if (!override_dir.empty()) dirs.emplace_back(override_dir);
  dirs.push_back(executable_dir / "custom_op/op_packages/htp");
  UdoArtifact result;
  for (const auto& dir : dirs) {
    const auto host = dir / "libQnnMyAddOpPackage.so";
    result.searched += host.string() + "\n";
    std::filesystem::path dsp;
#if defined(__ANDROID__) && defined(QNN_UDO_HTP_ARCH)
    dsp = dir / ("hexagon-v" + std::to_string(QNN_UDO_HTP_ARCH)) / "libQnnMyAddOpPackage.so";
    result.searched += dsp.string() + "\n";
#endif
    if (std::filesystem::is_regular_file(host) && (dsp.empty() || std::filesystem::is_regular_file(dsp))) {
      result.path = std::filesystem::absolute(host);
      if (!dsp.empty()) result.dsp = std::filesystem::absolute(dsp);
      return result;
    }
  }
  return result;
}

class QnnHTPUdoTests : public QnnHTPBackendTests {
 protected:
  void SetUp() override {
    QnnHTPBackendTests::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;
#ifdef BUILD_QNN_UDO_TEST
    SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#ifdef QNN_UDO_HTP_ARCH
    const auto device_arch = GetPlatformAttributes().htp_arch;
    if (device_arch != QNN_UDO_HTP_ARCH) {
      FAIL() << "Custom Op skel V" << QNN_UDO_HTP_ARCH << " does not match device V" << device_arch;
    }
#endif
    const auto artifact = FindUdoArtifact(ExecutableDirectory(),
                                          ParseEnvironmentVariableWithDefault<std::string>("ORT_QNN_UDO_OP_PACKAGE_DIR", ""));
    ASSERT_FALSE(artifact.path.empty()) << "Required HTP Custom Op package pair missing. Searched:\n"
                                        << artifact.searched;
    package_path_ = artifact.path;
    dsp_path_ = artifact.dsp;
#else
    // Follow the repository skip pattern for MSVC warning compatibility.
    if (::testing::internal::AlwaysTrue()) {
      GTEST_SKIP() << "No HTP Custom Op host package is built for this target";
    }
#endif
  }

  const std::filesystem::path& package_path() const { return package_path_; }
  const std::filesystem::path& dsp_path() const { return dsp_path_; }

 private:
  std::filesystem::path package_path_;
  std::filesystem::path dsp_path_;
};

class QnnHTPUdoConfigurationErrorTests : public QnnHTPBackendTests {
 protected:
  void SetUp() override {
    QnnHTPBackendTests::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;
#if !((defined(__linux__) && defined(__x86_64__)) || (defined(__ANDROID__) && defined(__aarch64__)))
    if (::testing::internal::AlwaysTrue()) GTEST_SKIP() << "Custom Op configuration error tests are not validated on this platform";
#endif
  }
};

static ProviderOptions UdoProviderOptions(const std::filesystem::path& host, const std::filesystem::path& dsp = {}) {
  std::string packages = "MyAdd:" + host.string() + ":MyAddOpPackageInterfaceProvider:CPU";
  if (!dsp.empty()) packages += ",MyAdd:" + dsp.string() + ":MyAddOpPackageInterfaceProvider:HTP";
  return {{"backend_type", "htp"}, {"offload_graph_io_quantization", "0"}, {"op_packages", packages}};
}

static void MakeUdoModel(ModelTestBuilder& builder, bool mixed = false) {
  const TestInputDef<float> input({1, 32}, false, -1.0f, 1.0f);
  std::vector<QuantParams<uint8_t>> output_qparams{{kOutputQuantScale, 0}};
  BuildUDOQDQTestCase<uint8_t>("MyAdd", input, {MakeAttribute("constant", kMyAddConstant)},
                               std::string(kUdoDomain))(builder, output_qparams);
  if (mixed) {
    // QNN does not support double tensors; this branch must stay on the CPU EP.
    builder.AddNode("double_cast", "Cast", {"output_qdq_dq_out"}, {"double_output"}, "",
                    {MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))});
    builder.MakeOutput<double>("double_output", std::vector<int64_t>{1, 32});
  }
  for (const auto& [domain, version] : std::unordered_map<std::string, int>{{"", 11}, {kMSDomain, 1}}) {
    auto* opset = builder.model_.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(version);
  }
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
}

static void RunUdoWithoutCpuReference(ModelTestBuilder& builder, const ProviderOptions& options,
                                      ExpectedEPNodeAssignment assignment = ExpectedEPNodeAssignment::All,
                                      const std::unordered_map<std::string, std::string>& configs = {},
                                      const std::filesystem::path& model_path = {}) {
  const auto& input = builder.feeds_.at("input");
  const auto count = input.GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<float> expected(count);
  for (size_t i = 0; i < count; ++i) expected[i] = input.GetTensorData<float>()[i] + kMyAddConstant;
  const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
  auto provider_options = options;
  TryEnableQNNSaver(provider_options);
  auto session_configs = configs;
  std::string model_data = builder.model_.SerializeAsString();
  if (!model_path.empty()) {
    ONNX_NAMESPACE::ModelProto context;
    std::ifstream stream(model_path, std::ios::binary);
    ASSERT_TRUE(context.ParseFromIstream(&stream));
    model_data = context.SerializeAsString();
    // InferenceModel loads from bytes; this supplies the base for external binaries.
    session_configs[kOrtSessionOptionEpContextFilePath] = model_path.string();
  }
  std::vector<Ort::Value> outputs;
  std::vector<std::string> output_names;
  std::function<void(const Ort::Session&)> capture_output_names = [&output_names](const Ort::Session& session) {
    output_names = session.GetOutputNames();
  };
  ASSERT_NO_FATAL_FAILURE(InferenceModel(model_data, "UDONoCpuReference", provider_options, assignment,
                                         builder.feeds_, outputs, ORT_LOGGING_LEVEL_ERROR, session_configs, std::nullopt, &capture_output_names));
  ASSERT_EQ(outputs.size(), builder.model_.graph().output_size());
  for (int output_index = 0; output_index < builder.model_.graph().output_size(); ++output_index) {
    const auto& output_name = output_names.at(output_index);
    const auto& output = outputs[output_index];
    ASSERT_EQ(output.GetTensorTypeAndShapeInfo().GetShape(), shape);
    ASSERT_EQ(output.GetTensorTypeAndShapeInfo().GetElementCount(), count);
    if ((output_name == "output_qdq_dq_out" || output_name == "output")) {
      ASSERT_EQ(output.GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      for (size_t i = 0; i < count; ++i) EXPECT_NEAR(output.GetTensorData<float>()[i], expected[i], kOutputTolerance);
    } else if (output_name == "double_output") {
      ASSERT_EQ(output.GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
      for (size_t i = 0; i < count; ++i) EXPECT_NEAR(output.GetTensorData<double>()[i], expected[i], kOutputTolerance);
    } else {
      FAIL() << "Unexpected Custom Op output: " << output_name;
    }
  }
}

TEST_F(QnnHTPUdoTests, UDO_Op_MyAdd) {
  RunOpTestOnHTP("MyAdd", TestInputDef<float>({1, 32}, false, -1.0f, 1.0f),
                 {MakeAttribute("constant", kMyAddConstant)},
                 UdoProviderOptions(package_path(), dsp_path()), 11, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPUdoTests, UDO_Op_MyAdd_AutoDomainOnly) {
  ScopedUdoDomain env("udo_domain:MyAdd");
  ModelTestBuilder builder;
  MakeUdoModel(builder);
  RunUdoWithoutCpuReference(builder, UdoProviderOptions(package_path(), dsp_path()));
}

TEST_F(QnnHTPUdoTests, UDO_Op_MyAdd_ManualPlusAutoDomain) {
  ScopedUdoDomain env("udo_domain:MyAdd");
  RunOpTestOnHTP("MyAdd", TestInputDef<float>({1, 32}, false, -1.0f, 1.0f),
                 {MakeAttribute("constant", kMyAddConstant)},
                 UdoProviderOptions(package_path(), dsp_path()), 11, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPUdoConfigurationErrorTests, UDO_Op_MyAdd_InvalidOpPackagePath) {
  ScopedUdoDomain env("udo_domain:MyAdd");
  ModelTestBuilder builder;
  MakeUdoModel(builder);
  const auto missing = ExecutableDirectory() / "nonexistent_custom_op.so";
  const ProviderOptions options{{"backend_type", "htp"},
                                {"op_packages", "MyAdd:" + missing.string() + ":MyAddOpPackageInterfaceProvider:CPU"}};
  QnnTestLogCapture diagnostics;
  Ort::SessionOptions session_options;
  AttachQnnTestLogCapture(session_options, diagnostics);
  session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
  RegisteredEpDeviceUniquePtr registered_ep;
  ASSERT_NO_FATAL_FAILURE(RegisterQnnEpLibrary(registered_ep, session_options, "UDOInvalidPath", options));
  const auto model_data = builder.model_.SerializeAsString();
  // ORT logs the EP registration error before reporting the unassigned MyAdd
  // node as the session exception. A session logger also works on Android,
  // where the default logger writes to logcat rather than stderr.
  EXPECT_THROW(Ort::Session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options), Ort::Exception);
  EXPECT_GT(diagnostics.CountContaining("Failed to register op package"), 0u);
  EXPECT_GT(diagnostics.CountContaining("Could not open op package path"), 0u);
  EXPECT_GT(diagnostics.CountContaining(missing.string()), 0u);
}

TEST_F(QnnHTPUdoConfigurationErrorTests, UDO_Op_MyAdd_CustomDomainEnvVarUnset) {
  ScopedUdoDomain env(nullptr);
  ModelTestBuilder builder;
  MakeUdoModel(builder);
  std::vector<Ort::Value> outputs;
  try {
    InferenceModel(builder.model_.SerializeAsString(), "UDONoDomain", ProviderOptions{{"backend_type", "htp"}},
                   ExpectedEPNodeAssignment::All, builder.feeds_, outputs);
    FAIL() << "Model without a custom domain must fail to load";
  } catch (const Ort::Exception& error) {
    EXPECT_NE(std::string(error.what()).find("MyAdd"), std::string::npos);
  }
}

static void TestUdoContextRoundTrip(const std::filesystem::path& package, const std::filesystem::path& dsp, bool mixed) {
  QnnScopedTempDir temporary_directory;
  const auto& dir = temporary_directory.path();
  const auto context_path = dir / "myadd_ctx.onnx";
  ModelTestBuilder builder;
  MakeUdoModel(builder, mixed);
  const auto assignment = mixed ? ExpectedEPNodeAssignment::Some : ExpectedEPNodeAssignment::All;
  {
    ScopedUdoDomain env("udo_domain:MyAdd");
    ASSERT_NO_FATAL_FAILURE(RunUdoWithoutCpuReference(builder, UdoProviderOptions(package, dsp), assignment,
                                                      {{kOrtSessionOptionEpContextEnable, "1"}, {kOrtSessionOptionEpContextEmbedMode, "0"}, {kOrtSessionOptionEpContextFilePath, context_path.string()}}));
  }
  ONNX_NAMESPACE::ModelProto context;
  std::ifstream stream(context_path, std::ios::binary);
  ASSERT_TRUE(context.ParseFromIstream(&stream));
  bool has_context = false;
  bool has_cpu_cast = false;
  bool has_binary = false;
  for (const auto& node : context.graph().node()) {
    EXPECT_NE(node.op_type(), "MyAdd");
    has_context |= node.op_type() == "EPContext";
    has_cpu_cast |= node.op_type() == "Cast";
    if (node.op_type() == "EPContext") {
      for (const auto& attr : node.attribute()) {
        if (attr.name() == "ep_cache_context") {
          has_binary |= std::filesystem::is_regular_file(dir / attr.s());
        }
      }
    }
  }
  ASSERT_TRUE(has_context);
  ASSERT_TRUE(has_binary);
  EXPECT_EQ(has_cpu_cast, mixed);
  ScopedUdoDomain env(nullptr);
  RunUdoWithoutCpuReference(builder, UdoProviderOptions(package, dsp), assignment, {}, context_path);
}

TEST_F(QnnHTPUdoTests, UDO_Op_MyAdd_EPContext_GenAndLoad) {
  TestUdoContextRoundTrip(package_path(), dsp_path(), false);
}

TEST_F(QnnHTPUdoTests, UDO_Op_MyAdd_MixedEPContext) {
  TestUdoContextRoundTrip(package_path(), dsp_path(), true);
}

TEST(QnnUdoArtifactLookup, UDO_Op_MyAdd_ArtifactLookup) {
  EXPECT_TRUE(std::filesystem::is_directory(ExecutableDirectory()));
  QnnScopedTempDir temporary_directory;
  const auto& missing = temporary_directory.path();
  const auto override_dir = missing / "override";
  const auto packaged_dir = missing / "custom_op/op_packages/htp";
  const auto artifact = FindUdoArtifact(missing, override_dir.string());
  EXPECT_TRUE(artifact.path.empty());
  EXPECT_NE(artifact.searched.find("override"), std::string::npos);
  EXPECT_NE(artifact.searched.find("custom_op/op_packages/htp"), std::string::npos);
  std::filesystem::create_directories(override_dir);
  std::filesystem::create_directories(packaged_dir);
  const auto name = "libQnnMyAddOpPackage.so";
#if defined(__ANDROID__) && defined(QNN_UDO_HTP_ARCH)
  for (const auto& root : {override_dir, packaged_dir}) {
    const auto dir = root / ("hexagon-v" + std::to_string(QNN_UDO_HTP_ARCH));
    std::filesystem::create_directories(dir);
    std::ofstream file(dir / name);
  }
#endif
  {
    std::ofstream file(missing / name);
  }
  EXPECT_TRUE(FindUdoArtifact(missing, override_dir.string()).path.empty());
  {
    std::ofstream file(packaged_dir / name);
  }
  EXPECT_EQ(FindUdoArtifact(missing, override_dir.string()).path, packaged_dir / name);
  {
    std::ofstream file(override_dir / name);
  }
  EXPECT_EQ(FindUdoArtifact(missing, override_dir.string()).path, override_dir / name);
#if defined(__ANDROID__) && defined(QNN_UDO_HTP_ARCH)
  const auto dsp_dir = "hexagon-v" + std::to_string(QNN_UDO_HTP_ARCH);
  EXPECT_EQ(FindUdoArtifact(missing, override_dir.string()).dsp, override_dir / dsp_dir / name);
  std::filesystem::remove(override_dir / dsp_dir / name);
  // An incomplete override must never be paired with another root's DSP half.
  const auto fallback = FindUdoArtifact(missing, override_dir.string());
  EXPECT_EQ(fallback.path, packaged_dir / name);
  EXPECT_EQ(fallback.dsp, packaged_dir / dsp_dir / name);
  std::filesystem::remove(packaged_dir / dsp_dir / name);
  EXPECT_TRUE(FindUdoArtifact(missing, override_dir.string()).path.empty());
#endif
}

}  // namespace test
}  // namespace onnxruntime

#endif
