// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#if !defined(ORT_MINIMAL_BUILD)

#include "test/providers/qnn/qnn_test_utils.h"
#include <cassert>
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test/test_environment.h"

#include "test/util/env_var_utils.h"
#include "core/common/span_utils.h"
#include "core/framework/compute_capability.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/ep_api_types.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_ep_types.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/optimizer/graph_optimizer_registry.h"

// Platform-specific includes for dynamic library loading
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace onnxruntime {
namespace test {

// Self-contained dynamic library loading functions to avoid using internal ORT APIs
namespace {

#if defined(_WIN32)
// Windows implementation using Win32 API
void* LoadDynamicLibraryImpl(const std::string& library_path) {
  return static_cast<void*>(LoadLibraryA(library_path.c_str()));
}

void* GetSymbolFromLibraryImpl(void* library_handle, const std::string& symbol_name) {
  if (!library_handle) return nullptr;
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library_handle), symbol_name.c_str()));
}

void UnloadDynamicLibraryImpl(void* library_handle) {
  if (library_handle) {
    FreeLibrary(static_cast<HMODULE>(library_handle));
  }
}

#else
// Linux implementation using dlopen/dlsym/dlclose
void* LoadDynamicLibraryImpl(const std::string& library_path) {
  return dlopen(library_path.c_str(), RTLD_LAZY);
}

void* GetSymbolFromLibraryImpl(void* library_handle, const std::string& symbol_name) {
  if (!library_handle) return nullptr;
  return dlsym(library_handle, symbol_name.c_str());
}

void UnloadDynamicLibraryImpl(void* library_handle) {
  if (library_handle) {
    dlclose(library_handle);
  }
}
#endif

}  // anonymous namespace

std::vector<float> GetFloatDataInRange(float min_val, float max_val, size_t num_elems) {
  if (num_elems == 0) {
    return {};
  }

  if (num_elems == 1) {
    return {min_val};
  }

  std::vector<float> data;
  data.reserve(num_elems);

  const float step_size = (max_val - min_val) / static_cast<float>(num_elems - 1);
  float val = min_val;
  for (size_t i = 0; i < num_elems; i++) {
    data.push_back(val);
    val += step_size;
  }

  // Ensure that max_val is included exactly (due to rounding from adding step sizes).
  data[num_elems - 1] = max_val;

  return data;
}

std::vector<float> GetSequentialFloatData(const std::vector<int64_t>& shape, float start, float step) {
  if (shape.empty()) {
    return {};
  }

  int64_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }

  std::vector<float> data;
  data.reserve(static_cast<size_t>(count));

  float val = start;
  for (int64_t i = 0; i < count; i++) {
    data.push_back(val);
    val += step;
  }

  return data;
}

TestInputDef<Ort::Float16_t> ConvertToFP16InputDef(const TestInputDef<float>& input_def) {
  if (input_def.IsRawData()) {
    std::vector<Ort::Float16_t> input_data_fp16;
    input_data_fp16.reserve(input_def.GetRawData().size());
    for (float f32_val : input_def.GetRawData()) {
      input_data_fp16.push_back(Ort::Float16_t(f32_val));
    }

    return TestInputDef<Ort::Float16_t>(input_def.GetShape(), input_def.IsInitializer(), input_data_fp16);
  } else {
    auto rand_data = input_def.GetRandomDataInfo();
    return TestInputDef<Ort::Float16_t>(input_def.GetShape(), input_def.IsInitializer(),
                                        Ort::Float16_t(rand_data.min), Ort::Float16_t(rand_data.max));
  }
}

class SafeIntExceptionHandler : public std::exception {
 public:
  [[noreturn]] static void SafeIntOnOverflow() {
    throw std::runtime_error("Integer overflow");
  }

  [[noreturn]] static void SafeIntOnDivZero() {
    throw std::runtime_error("Divide by zero");
  }
};

size_t SizeHelper(std::vector<int64_t> shape, size_t start, size_t end) {
  // Must return 1 for an empty sequence
  SafeInt<int64_t, SafeIntExceptionHandler> size = 1;  // this is used to calculate the size, which is used for memory allocations, so validate no overflow
  for (size_t i = start; i < end; i++) {
    if (shape[i] < 0) return -1;
    size *= shape[i];
  }
  return size;
}

size_t SizeToDimension(std::vector<int64_t> shape, size_t dimension) {
  QNN_ASSERT(dimension <= shape.size());

  int64_t size = SizeHelper(shape, 0, dimension);
  return size;
}

size_t SizeFromDimension(std::vector<int64_t> shape, size_t dimension) {
  const size_t num_dims = shape.size();
  QNN_ASSERT(dimension <= num_dims);

  int64_t size = SizeHelper(shape, dimension, num_dims);
  return size;
}

size_t SizeOfShape(std::vector<int64_t> shape) {
  return SizeHelper(shape, 0, shape.size());
}

void TryEnableQNNSaver(ProviderOptions& qnn_options) {
  // Allow dumping QNN API calls to file by setting an environment variable that enables the QNN Saver backend.
  constexpr auto kEnableQNNSaverEnvironmentVariableName = "ORT_UNIT_TEST_ENABLE_QNN_SAVER";
  static std::optional<int> enable_qnn_saver = ParseEnvironmentVariable<int>(kEnableQNNSaverEnvironmentVariableName);

  if (enable_qnn_saver.has_value() && *enable_qnn_saver != 0) {
#if defined(_WIN32)
    qnn_options["qnn_saver_path"] = "QnnSaver.dll";
#else
    qnn_options["qnn_saver_path"] = "libQnnSaver.so";
#endif  // defined(_WIN32)
  }
}

void RegisterQnnEpLibrary(RegisteredEpDeviceUniquePtr& registered_ep_device,
                          Ort::SessionOptions& session_options,
                          const std::string& registration_name,
                          const std::unordered_map<std::string, std::string>& ep_options,
                          bool simulated) {
  Ort::Env* ort_env = GetOrtEnv();
  const OrtApi& c_api = Ort::GetApi();

  std::filesystem::path library_path = "";
  if (simulated) {
    library_path =
#if _WIN32
        "onnxruntime_providers_qnn_simulation.dll";
#else
        "libonnxruntime_providers_qnn_simulation.so";
#endif
  } else {
    library_path =
#if _WIN32
        "onnxruntime_providers_qnn.dll";
#else
        "libonnxruntime_providers_qnn.so";
#endif
  }

  ASSERT_ORTSTATUS_OK(c_api.RegisterExecutionProviderLibrary(*ort_env,
                                                             registration_name.c_str(),
                                                             library_path.c_str()));

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices;
  ASSERT_ORTSTATUS_OK(c_api.GetEpDevices(*ort_env, &ep_devices, &num_devices));

  auto target_hw_device_type = OrtHardwareDeviceType_CPU;
  if ((ep_options.find("backend_type") != ep_options.end() && ep_options.at("backend_type") == "htp") ||
      (ep_options.find("backend_path") != ep_options.end() && ep_options.at("backend_path") ==
#if _WIN32
                                                                  "QnnHtp.dll"
#else
                                                                  "libQnnHtp.so"
#endif
       )) {
#if defined(__linux__)
    target_hw_device_type = OrtHardwareDeviceType_CPU;
#else
    target_hw_device_type = OrtHardwareDeviceType_NPU;
#endif
  } else if ((ep_options.find("backend_type") != ep_options.end() && ep_options.at("backend_type") == "gpu") ||
             (ep_options.find("backend_path") != ep_options.end() && ep_options.at("backend_path") ==
#if _WIN32
                                                                         "QnnGpu.dll"
#else
                                                                         "libQnnGpu.so"
#endif
              )) {
#if defined(__linux__)
    target_hw_device_type = OrtHardwareDeviceType_CPU;
#else
    target_hw_device_type = OrtHardwareDeviceType_GPU;
#endif
  }

  auto it = std::find_if(ep_devices, ep_devices + num_devices,
                         [&c_api, &registration_name, &target_hw_device_type](const OrtEpDevice* ep_device) {
                           return (c_api.EpDevice_EpName(ep_device) == registration_name &&
                                   c_api.HardwareDevice_Type(c_api.EpDevice_Device(ep_device)) == target_hw_device_type);
                         });

  ASSERT_NE(it, ep_devices + num_devices);

  registered_ep_device = RegisteredEpDeviceUniquePtr(*it, [registration_name](const OrtEpDevice* /*ep*/) {
    OrtStatus* status = Ort::GetApi().UnregisterExecutionProviderLibrary(*GetOrtEnv(), registration_name.c_str());
    if (status != nullptr) {
      Ort::GetApi().ReleaseStatus(status);
    }
  });

  session_options.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(registered_ep_device.get())}, ep_options);
}

void RunQnnModelTest(const GetTestModelFn& build_test_case, ProviderOptions provider_options,
                     int opset_version, ExpectedEPNodeAssignment expected_ep_assignment,
                     float fp32_abs_err, OrtLoggingLevel log_severity, bool verify_outputs,
                     std::function<void(const Graph&)>* ep_graph_checker) {
  std::filesystem::path output_dir;
  if (QNNTestEnvironment::GetInstance().dump_onnx() ||
      QNNTestEnvironment::GetInstance().dump_json() ||
      QNNTestEnvironment::GetInstance().dump_dlc()) {
    output_dir = QNNTestEnvironment::GetInstance().CreateTestcaseDirs();
  }

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = expected_ep_assignment;
  verification_params.fp32_abs_err = fp32_abs_err;
  verification_params.graph_verifier = ep_graph_checker;
  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  build_test_case(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Serialize the model to a string.
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  if (QNNTestEnvironment::GetInstance().dump_dlc()) {
    provider_options["dump_qnn_ir_dlc"] = "1";
    provider_options["dump_qnn_ir_dlc_dir"] = output_dir.string();
#if defined(_WIN32)
    provider_options["qnn_ir_backend_path"] = "QnnIr.dll";
#else
    provider_options["qnn_ir_backend_path"] = "libQnnIr.so";
#endif  // defined(_WIN32)
  }
  if (QNNTestEnvironment::GetInstance().dump_json()) {
    provider_options["dump_json_qnn_graph"] = "1";
    provider_options["json_qnn_graph_dir"] = output_dir.string();
  }

  // Run with QNN.
  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;

  session_options.SetLogSeverityLevel(log_severity);

  TryEnableQNNSaver(provider_options);
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);
  RunAndVerifyOutputsWithEP(AsByteSpan(model_data.data(), model_data.size()),
                            session_options,
                            registration_name,
                            "QNN_EP_TestLogID",
                            helper.feeds_,
                            verification_params,
                            verify_outputs);
}

void InferenceModelCPU(const std::string& model_data,
                       const char* log_id,
                       ExpectedEPNodeAssignment expected_ep_assignment [[maybe_unused]],
                       std::unordered_map<std::string, Ort::Value>& feeds,
                       std::vector<Ort::Value>& output_vals,
                       std::optional<GraphOptimizationLevel> graph_optimization_level) {
  Ort::SessionOptions session_options;
  session_options.SetLogId(log_id);

  if (graph_optimization_level.has_value()) {
    session_options.SetGraphOptimizationLevel(graph_optimization_level.value());
  }

  Ort::Session session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options);

  // Prepare inputs using public API
  std::vector<std::string> ort_input_names = session.GetInputNames();
  std::vector<std::string> ort_output_names = session.GetOutputNames();
  size_t input_count = ort_input_names.size();
  size_t output_count = ort_output_names.size();
  std::vector<const char*> ort_input_names_cstr(input_count);
  std::vector<const char*> ort_output_names_cstr(output_count);
  std::transform(ort_input_names.begin(), ort_input_names.end(), ort_input_names_cstr.begin(),
                 [](const std::string& s) { return s.c_str(); });
  std::transform(ort_output_names.begin(), ort_output_names.end(), ort_output_names_cstr.begin(),
                 [](const std::string& s) { return s.c_str(); });

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    ort_inputs.emplace_back(Ort::Value::CreateTensor(
        memory_info,
        (void*)feeds.at(ort_input_names[i]).GetTensorRawData(),
        feeds.at(ort_input_names[i]).GetTensorSizeInBytes(),
        feeds.at(ort_input_names[i]).GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().data(),
        feeds.at(ort_input_names[i]).GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().size(),
        feeds.at(ort_input_names[i]).GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType()));
  }

  // Run inference
  output_vals = session.Run(
      Ort::RunOptions{nullptr},
      ort_input_names_cstr.data(),
      ort_inputs.data(),
      ort_inputs.size(),
      ort_output_names_cstr.data(),
      ort_output_names_cstr.size());
}

void InferenceModel(const std::string& model_data,
                    const char* log_id,
                    const ProviderOptions& provider_options,
                    ExpectedEPNodeAssignment expected_ep_assignment,
                    std::unordered_map<std::string, Ort::Value>& feeds,
                    std::vector<Ort::Value>& output_vals,
                    const std::unordered_map<std::string, std::string>& session_option_pairs,
                    std::optional<GraphOptimizationLevel> graph_optimization_level,
                    std::function<void(const Graph&)>* graph_checker [[maybe_unused]]) {
  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;
  if (graph_optimization_level.has_value()) {
    session_options.SetGraphOptimizationLevel(*graph_optimization_level);
  }
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  session_options.SetLogId(log_id);

  // Uncomment to dump verbose output to stdout.
  // session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);

  for (auto key_value : session_option_pairs) {
    session_options.AddConfigEntry(key_value.first.c_str(), key_value.second.c_str());
  }

  Ort::RunOptions ort_run_options;
  ort_run_options.SetRunTag(log_id);

  std::string test_suite_name = ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
  // TODO: Implement EP assignment verification once public API for ep partition is ready
  // This disable_cpu_ep_fallback is an workaround for ExpectedEPNodeAssignment::All
  if (test_suite_name != "QnnCPUBackendTests" &&
      expected_ep_assignment == ExpectedEPNodeAssignment::All) {
    // ASSERT_EQ(ep_nodes, graph.NumberOfNodes()) << "Not all nodes were assigned to " << registration_name;
    session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  }
  Ort::Session session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options);

  // TODO: Implement graph_checker once public API for ep partition is ready
  // const auto& graph = ort_session.GetGraph();
  // if (graph_checker) {
  //   (*graph_checker)(graph);
  // }

  RunWithEP(session, ort_run_options, feeds, output_vals);
}

std::string MakeTestQDQBiasInput(ModelTestBuilder& builder,
                                 const std::string& name,
                                 const TestInputDef<float>& bias_def,
                                 float bias_scale,
                                 bool use_contrib_qdq) {
  // Bias must be int32 to be detected as a QDQ node unit.
  // We must quantize the data.
  if (bias_def.IsRandomData()) {
    // Create random initializer def that is quantized to int32
    const auto& rand_info = bias_def.GetRandomDataInfo();
    TestInputDef<int32_t> bias_int32_def(bias_def.GetShape(), bias_def.IsInitializer(),
                                         static_cast<int32_t>(rand_info.min / bias_scale),
                                         static_cast<int32_t>(rand_info.max / bias_scale));
    MakeTestInput(builder, name, bias_int32_def);
  } else {
    QNN_ASSERT(bias_def.IsRawData());
    // Create raw data initializer def that is quantized to int32
    const auto& bias_f32_raw = bias_def.GetRawData();
    const size_t num_elems = bias_f32_raw.size();

    std::vector<int32_t> bias_int32_raw(num_elems);
    for (size_t i = 0; i < num_elems; i++) {
      bias_int32_raw[i] = static_cast<int32_t>(bias_f32_raw[i] / bias_scale);
    }

    TestInputDef<int32_t> bias_int32_def(bias_def.GetShape(), bias_def.IsInitializer(), bias_int32_raw);
    MakeTestInput(builder, name, bias_int32_def);
  }

  builder.AddDequantizeLinearNode<int32_t>(
      name + "_dq",
      name.c_str(),
      bias_scale,
      0,
      (name + "_dq_out").c_str(),
      use_contrib_qdq);
  return name + "_dq_out";
}

// TODO: Consider using public DeviceCompatibility API for this function
static BackendSupport GetHTPSupport() {
  return BackendSupport::SUPPORTED;
}

void QnnHTPBackendTests::SetUp() {
  if (cached_htp_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  Ort::Logger logger = Ort::Logger();

  if (cached_htp_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_htp_support_ = GetHTPSupport();
  }
  // Determine if HTP backend is supported only if we done so haven't before.
  if (cached_htp_support_ == BackendSupport::UNSUPPORTED) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "QNN HTP backend is not available! Skipping test.");
    GTEST_SKIP();
  } else if (cached_htp_support_ == BackendSupport::SUPPORT_ERROR) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, "Failed to check if QNN HTP backend is available.");
    FAIL();
  }

  // query the platform attributes if not already cached.
  if (!cached_platform_attrs_.has_value()) {
    QnnPlatformAttributes attrs;

    Ort::Status query_status = QueryQnnPlatformAttributesDirectly(attrs, logger);
    if (!query_status.IsOK()) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, ("QueryQnnPlatformAttributesDirectly failed: " + query_status.GetErrorMessage()).c_str());
    } else {
      // Create a string stream to build the output message
      std::stringstream ss;
      ss << "QNN platform attributes: "
         << "HTP arch: " << attrs.htp_arch
         << ", DLBC supported: " << attrs.dlbc_supported
         << ", VTCM size MB: " << attrs.vtcm_size_mb
         << ", SoC model: " << attrs.soc_model
         << ", SDK version: " << attrs.sdk_version;

      std::string platform_info_str = ss.str();
      std::cout << platform_info_str;
      // TODO: Fix the crash here with ORT_CXX_LOG
      // ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO, platform_info_str.c_str());

      cached_platform_attrs_ = attrs;
    }
  }
}

// Checks if Qnn Gpu backend can run a graph on the system.
// Creates a one node graph with relu op,
// to check if the GPU backend is available.
static BackendSupport GetGPUSupport() {
  return BackendSupport::SUPPORTED;
}

void QnnGPUBackendTests::SetUp() {
  if (cached_gpu_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  Ort::Logger logger = Ort::Logger();

  // Determine if GPU backend is supported only if we haven't done so before.
  if (cached_gpu_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_gpu_support_ = GetGPUSupport();
  }

  if (cached_gpu_support_ == BackendSupport::UNSUPPORTED) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "QNN GPU backend is not available! Skipping test.");
  } else if (cached_gpu_support_ == BackendSupport::SUPPORT_ERROR) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, "Failed to check if QNN GPU backend is available.");
    FAIL();
  }
}

static BackendSupport GetIRSupport();

BackendSupport QnnHTPBackendTests::IsIRBackendSupported() const {
  if (cached_ir_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_ir_support_ = test::GetIRSupport();
  }

  return cached_ir_support_;
}

// TODO: Consider using public DeviceCompatibility API for this function
static BackendSupport GetCPUSupport() {
  return BackendSupport::SUPPORTED;
}

void QnnCPUBackendTests::SetUp() {
  if (cached_cpu_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  Ort::Logger logger = Ort::Logger();

  // Determine if CPU backend is supported only if we done so haven't before.
  if (cached_cpu_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_cpu_support_ = GetCPUSupport();
  }

  if (cached_cpu_support_ == BackendSupport::UNSUPPORTED) {
    GTEST_SKIP();
  } else if (cached_cpu_support_ == BackendSupport::SUPPORT_ERROR) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, "Failed to check if QNN CPU backend is available.");
    FAIL();
  }
}

static BackendSupport GetIRSupport() {
  // QnnIr should be able to serialize any model supported by the QNN reference spec.
  // Use a model that works on QnnCpu to verify QnnIr availability.
  return GetCPUSupport();
}

void QnnIRBackendTests::SetUp() {
  if (cached_ir_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  Ort::Logger logger = Ort::Logger();

  // Determine if IR backend is supported only if we done so haven't before.
  if (cached_ir_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_ir_support_ = GetIRSupport();
  }

  if (cached_ir_support_ == BackendSupport::UNSUPPORTED) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "QNN IR backend is not available! Skipping test.");
    GTEST_SKIP();
  } else if (cached_ir_support_ == BackendSupport::SUPPORT_ERROR) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, "Failed to check if QNN IR backend is available.");
    FAIL();
  }
}

#if defined(_WIN32)
// TODO: Remove or set to SUPPORTED once HTP emulation is supported on win arm64.
BackendSupport QnnHTPBackendTests::cached_htp_support_ = BackendSupport::SUPPORT_UNKNOWN;

// TODO: Remove or set to SUPPORTED once CPU backend works on win arm64 (pipeline VM).
BackendSupport QnnCPUBackendTests::cached_cpu_support_ = BackendSupport::SUPPORT_UNKNOWN;
#else
BackendSupport QnnHTPBackendTests::cached_htp_support_ = BackendSupport::SUPPORTED;
BackendSupport QnnCPUBackendTests::cached_cpu_support_ = BackendSupport::SUPPORTED;
#endif  // defined(_WIN32)

BackendSupport QnnHTPBackendTests::cached_ir_support_ = BackendSupport::SUPPORT_UNKNOWN;
BackendSupport QnnIRBackendTests::cached_ir_support_ = BackendSupport::SUPPORT_UNKNOWN;
BackendSupport QnnGPUBackendTests::cached_gpu_support_ = BackendSupport::SUPPORT_UNKNOWN;

std::optional<QnnHTPBackendTests::QnnPlatformAttributes> QnnHTPBackendTests::cached_platform_attrs_ = std::nullopt;

/**
 * @brief Queries QNN platform attributes by directly calling QNN APIs.
 *
 * This function loads the QNN HTP backend library, and retrieves platform attributes
 * such as version, platform ID, and platform name.
 *
 * @param[out] out
 *   Reference to a QnnPlatformAttributes struct that will be populated with the queried attributes
 *   if the function succeeds.
 * @param[in] logger
 *   Logger instance for logging warnings and errors.
 *
 * @return Status
 *   Returns Status::OK() on success. On failure, returns a Status object with an appropriate error code and message.
 *
 * @error
 *   - If the QNN backend library cannot be loaded, returns an error status.
 *   - If required QNN API symbols cannot be resolved, returns an error status.
 *   - If QNN context initialization or attribute querying fails, returns an error status.
 *   - In all error cases, the output parameter 'out' is not modified.
 */
Ort::Status QnnHTPBackendTests::QueryQnnPlatformAttributesDirectly(QnnHTPBackendTests::QnnPlatformAttributes& out, const Ort::Logger& logger) {
  void* qnn_lib_handle = nullptr;

#if defined(_WIN32)
  const std::string backend_path = "QnnHtp.dll";
#else
  const std::string backend_path = "libQnnHtp.so";
#endif

  // Load QNN HTP backend library using self-contained implementation
  qnn_lib_handle = LoadDynamicLibraryImpl(backend_path);
  if (!qnn_lib_handle) {
    return Ort::Status(("Failed to load QNN HTP backend library: " + backend_path).c_str(), ORT_FAIL);
  }

  // Get QNN interface providers function
  using QnnInterfaceGetProvidersFn_t = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  QnnInterfaceGetProvidersFn_t qnn_interface_get_providers = nullptr;

  qnn_interface_get_providers = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
      GetSymbolFromLibraryImpl(qnn_lib_handle, "QnnInterface_getProviders"));
  if (!qnn_interface_get_providers) {
    UnloadDynamicLibraryImpl(qnn_lib_handle);
    return Ort::Status("Failed to get QnnInterface_getProviders symbol", ORT_FAIL);
  }

  // Get QNN interface
  const QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;
  Qnn_ErrorHandle_t qnn_status = qnn_interface_get_providers(&interface_providers, &num_providers);

  if (qnn_status != QNN_SUCCESS || num_providers == 0 || !interface_providers) {
    UnloadDynamicLibraryImpl(qnn_lib_handle);
    return Ort::Status("QnnInterface_getProviders failed", ORT_FAIL);
  }

  // Use the first provider
  const QnnInterface_t* qnn_interface = interface_providers[0];
  if (!qnn_interface) {
    UnloadDynamicLibraryImpl(qnn_lib_handle);
    return Ort::Status("QnnInterface_getProviders failed", ORT_FAIL);
  }

  // Extract function pointers from the versioned interface
  auto logCreateFn = qnn_interface->QNN_INTERFACE_VER_NAME.logCreate;
  auto logFreeFn = qnn_interface->QNN_INTERFACE_VER_NAME.logFree;
  auto getPlatformInfoFn = qnn_interface->QNN_INTERFACE_VER_NAME.deviceGetPlatformInfo;
  auto freePlatformInfoFn = qnn_interface->QNN_INTERFACE_VER_NAME.deviceFreePlatformInfo;
  auto backendGetApiVersionFn = qnn_interface->QNN_INTERFACE_VER_NAME.backendGetApiVersion;

  // Create a log handle (optional, can pass nullptr)
  Qnn_LogHandle_t log_handle = nullptr;
  if (logCreateFn) {
    qnn_status = logCreateFn(nullptr, QNN_LOG_LEVEL_WARN, &log_handle);
    if (qnn_status != QNN_SUCCESS) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "Failed to create QNN log handle, continuing without logging");
    }
  }

  // Get platform info
  const QnnDevice_PlatformInfo_t* platform_info_ptr = nullptr;
  if (!getPlatformInfoFn) {
    if (log_handle && logFreeFn) {
      logFreeFn(log_handle);
    }
    UnloadDynamicLibraryImpl(qnn_lib_handle);
    return Ort::Status("deviceGetPlatformInfo function not available", ORT_FAIL);
  }

  qnn_status = getPlatformInfoFn(log_handle, &platform_info_ptr);

  if (qnn_status != QNN_SUCCESS || !platform_info_ptr) {
    if (log_handle && logFreeFn) {
      logFreeFn(log_handle);
    }
    UnloadDynamicLibraryImpl(qnn_lib_handle);
    return Ort::Status("deviceGetPlatformInfo failed", ORT_FAIL);
  }

  auto ret = Ort::Status();
  // Extract platform attributes
  if (platform_info_ptr->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
    const QnnDevice_PlatformInfoV1_t& p = platform_info_ptr->v1;

    // Get SDK version from backend API version
    if (backendGetApiVersionFn) {
      Qnn_ApiVersion_t api_version;
      qnn_status = backendGetApiVersionFn(&api_version);
      if (qnn_status == QNN_SUCCESS) {
        out.sdk_version = std::to_string(api_version.coreApiVersion.major) + "." +
                          std::to_string(api_version.coreApiVersion.minor) + "." +
                          std::to_string(api_version.coreApiVersion.patch);
      }
    }

    // Extract HTP-specific device info
    for (uint32_t i = 0; i < p.numHwDevices; ++i) {
      const QnnDevice_HardwareDeviceInfo_t& dev = p.hwDevices[i];
      if (dev.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) continue;

      const QnnDevice_HardwareDeviceInfoV1_t& devV1 = dev.v1;
      const auto* htp_ext = reinterpret_cast<const QnnHtpDevice_DeviceInfoExtension_t*>(devV1.deviceInfoExtension);

      if (htp_ext && htp_ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
        const QnnHtpDevice_OnChipDeviceInfoExtension_t& oc = htp_ext->onChipDevice;
        out.vtcm_size_mb = static_cast<uint32_t>(oc.vtcmSize);
        out.soc_model = oc.socModel;
        out.dlbc_supported = oc.dlbcSupport;
        out.htp_arch = oc.arch;
        break;
      }
    }
  } else {
    ret = Ort::Status("Unsupported QNN device platform info version", ORT_FAIL);
  }

  // Free platform info
  if (freePlatformInfoFn) {
    freePlatformInfoFn(log_handle, platform_info_ptr);
  }

  // Free log handle
  if (log_handle && logFreeFn) {
    logFreeFn(log_handle);
  }

  // Unload library
  UnloadDynamicLibraryImpl(qnn_lib_handle);

  return ret;
}

bool ReduceOpHasAxesInput(const std::string& op_type, int opset_version) {
  static const std::unordered_map<std::string, int> opset_with_axes_as_input = {
      {"ReduceMax", 18},
      {"ReduceMin", 18},
      {"ReduceMean", 18},
      {"ReduceProd", 18},
      {"ReduceSum", 13},
      {"ReduceL2", 18},
  };

  const auto it = opset_with_axes_as_input.find(op_type);

  return (it != opset_with_axes_as_input.cend()) && (it->second <= opset_version);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
