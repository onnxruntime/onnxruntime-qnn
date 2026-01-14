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

namespace onnxruntime {
namespace test {

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
    return TestInputDef<Ort::Float16_t>(input_def.GetShape(),
                               input_def.IsInitializer(),
                                     Ort::Float16_t(rand_data.min),
                                     Ort::Float16_t(rand_data.max));
  }
}

class SafeIntExceptionHandler : public std::exception {
 public:
  [[noreturn]] static void SafeIntOnOverflow() {
    throw std::string("Integer overflow");
  }

  [[noreturn]] static void SafeIntOnDivZero() {
    throw std::string("Divide by zero");
  }
};

int64_t SizeHelper(std::vector<int64_t> shape, size_t start, size_t end) {
  // Must return 1 for an empty sequence
  SafeInt<int64_t, SafeIntExceptionHandler> size = 1;  // this is used to calculate the size, which is used for memory allocations, so validate no overflow
  for (size_t i = start; i < end; i++) {
    if (shape[i] < 0) return -1;
    size *= shape[i];
  }
  return size;
}

size_t SizeToDimension(std::vector<int64_t> shape, size_t dimension) {
  assert(dimension <= shape_.size() && "Invalid dimension of for SizeToDimension");

  int64_t size = SizeHelper(shape, 0, dimension);
  return size;
}

size_t SizeFromDimension(std::vector<int64_t> shape, size_t dimension) {
  const size_t num_dims = shape.size();
  assert(dimension <= num_dims && "Invalid dimension of for SizeFromDimension.");

  int64_t size = SizeHelper(shape, dimension, num_dims);
  return size;
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

void RunQnnModelTest(const BuildTestModelFn& build_test_case, ProviderOptions provider_options,
                     int opset_version, ExpectedEPNodeAssignment expected_ep_assignment,
                     float fp32_abs_err, OrtLoggingLevel log_severity, bool verify_outputs,
                     std::function<void(const Graph&)>* ep_graph_checker) {
  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = expected_ep_assignment;
  verification_params.fp32_abs_err = fp32_abs_err;
  verification_params.graph_verifier = ep_graph_checker;
  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};

  // auto& logging_manager = DefaultLoggingManager();
  // logging_manager.SetDefaultLoggerSeverity(log_severity);

  // onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
  //                          IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
  //                          logging_manager.DefaultLogger());
  // Graph& graph = model.MainGraph();
  ModelPublicBuilder helper;
  build_test_case(helper);
  // helper.SetGraphOutputs();
  // ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  // std::string model_data;
  // model.ToProto().SerializeToString(&model_data);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  // TryEnableQNNSaver(provider_options);

  // Run with QNN.
  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  RunAndVerifyOutputsWithEPABI(AsByteSpan(model_data.data(), model_data.size()),
                               session_options,
                               registration_name,
                               "QNN_EP_TestLogID",
                               helper.feeds_,
                               verification_params,
                               verify_outputs);
}

void InferenceModelCPU(const std::string& model_data,
                       const char* log_id,
                       ExpectedEPNodeAssignment expected_ep_assignment,
                       std::unordered_map<std::string, Ort::Value>& feeds,
                       std::vector<Ort::Value>& output_vals) {
  Ort::SessionOptions session_options;
  session_options.SetLogId(log_id);

  Ort::Session session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options);

  // TODO: Evaluate whether we can acquire graph from the public Ort::Session
  // const auto& graph = inference_session->GetGraph();

  std::string provider_type = kCpuExecutionProvider;
  // auto ep_nodes = CountAssignedNodes(graph, provider_type);
  // if (expected_ep_assignment == ExpectedEPNodeAssignment::All) {
  //   // Verify the entire graph is assigned to the EP
  //   ASSERT_EQ(ep_nodes, graph.NumberOfNodes()) << "Not all nodes were assigned to " << provider_type;
  // } else if (expected_ep_assignment == ExpectedEPNodeAssignment::None) {
  //   ASSERT_EQ(ep_nodes, 0) << "No nodes are supposed to be assigned to " << provider_type;
  // } else {
  //   ASSERT_GT(ep_nodes, 0) << "No nodes were assigned to " << provider_type;
  // }

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
      feeds.at(ort_input_names[i]).GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType())
    );
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
                    std::function<void(const Graph&)>* graph_checker) {
  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  session_options.SetLogId(log_id);
  session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
  for (auto key_value : session_option_pairs) {
    session_options.AddConfigEntry(key_value.first.c_str(), key_value.second.c_str());
  }

  Ort::RunOptions ort_run_options;
  ort_run_options.SetRunTag(log_id);

  Ort::Session session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options);

  // Verify node assignment.
  // const auto& graph = ort_session.GetGraph();

  // auto ep_nodes = CountAssignedNodes(graph, registration_name);
  // if (expected_ep_assignment == ExpectedEPNodeAssignment::All) {
  //   ASSERT_EQ(ep_nodes, graph.NumberOfNodes()) << "Not all nodes were assigned to " << registration_name;
  // } else if (expected_ep_assignment == ExpectedEPNodeAssignment::None) {
  //   ASSERT_EQ(ep_nodes, 0) << "No nodes are supposed to be assigned to " << registration_name;
  // } else {
  //   ASSERT_GT(ep_nodes, 0) << "No nodes were assigned to " << registration_name;
  // }

  // if (graph_checker) {
  //   (*graph_checker)(graph);
  // }

  RunWithEPABI(session, ort_run_options, feeds, output_vals);
}

std::string MakeTestQDQBiasInput(ModelPublicBuilder& builder,
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
    assert(bias_def.IsRawData());
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

// Testing helper function that calls QNN EP's GetCapability() function with a mock graph to check
// if the HTP backend is available.
// TODO: Remove once HTP can be emulated on Windows ARM64.
static BackendSupport GetHTPSupport(const OrtLogger* logger) {
  std::cout << "Check if HTP is available" << std::endl;
  ModelPublicBuilder helper;

  // Build simple QDQ graph: DQ -> InstanceNormalization -> Q
  BuildTestModelFn build_test_case = [](ModelPublicBuilder& builder) {
    const uint8_t quant_zero_point = 0;
    const float quant_scale = 1.0f;

    auto* scale = builder.MakeInitializer<uint8_t>(
      "scale", {2}, std::vector<uint8_t>{1, 2});
    builder.AddDequantizeLinearNode<uint8_t>(
      "dq1", scale->name().c_str(), quant_scale, quant_zero_point, "dq1_out");

    // Add bias (initializer) -> DQ ->
    auto* bias = builder.MakeInitializer<int32_t>(
      "bias", {2}, std::vector<int32_t>{1, 1});
    builder.AddDequantizeLinearNode<int32_t>(
      "dq2", bias->name().c_str(), 1.0f, 0, "dq2_out");

    // Add input_u8 -> DQ ->
    auto* input_u8 = builder.MakeInput<uint8_t>(
      "X", {1, 2, 3}, std::vector<uint8_t>{1, 2, 3, 4, 5, 6});
    builder.AddDequantizeLinearNode<uint8_t>(
      "dq3", input_u8->name().c_str(), quant_scale, quant_zero_point, "dq3_out");

    // Add dq_input_output -> InstanceNormalization ->
    std::vector<ONNX_NAMESPACE::AttributeProto*> attributes;
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    builder.AddNode("in",
      "InstanceNormalization",
      {"dq1_out", "dq2_out", "dq3_out"},
      {"in_out"},
      "",
      attributes);

    // Add instance_norm_output -> Q -> output_u8
    builder.AddQuantizeLinearNode<uint8_t>(
      "q1", "in_out", quant_scale, quant_zero_point, "Y");
    builder.MakeOutput("Y");
  };

  build_test_case(helper);
  // helper.SetGraphOutputs();
  // auto status = model.MainGraph().Resolve();

  // if (!status.IsOK()) {
  //   return BackendSupport::SUPPORT_ERROR;
  // }

  // Create QNN EP and call GetCapability().
  // TODO: Use public API to acquire OrtEpGraphSupportInfo
  // onnxruntime::GraphViewer graph_viewer(graph);
  // std::unique_ptr<EpGraph> ep_graph = nullptr;
  // if (!EpGraph::Create(graph_viewer, ep_graph).IsOK()) {
  //   return BackendSupport::UNSUPPORTED;
  // }
  // OrtEpGraphSupportInfo graph_support_info(*ep_graph);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;
  ProviderOptions provider_options = {{"backend_type", "htp"}, {"offload_graph_io_quantization", "0"}};
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  OrtEpFactory* qnn_ep_factory = registered_ep_device->GetMutableFactory();
  OrtEp* qnn_ep = nullptr;
  std::cout << "qnn_ep_factory->CreateEp" << std::endl;
  if (qnn_ep_factory->CreateEp(qnn_ep_factory, nullptr, nullptr, 0, session_options, logger, &qnn_ep)) {
    std::cout << "Finish qnn_ep_factory->CreateEp" << std::endl;
    qnn_ep_factory->ReleaseEp(qnn_ep_factory, qnn_ep);
    return BackendSupport::UNSUPPORTED;
  }

  std::cout << "Check if HTP is available 4" << std::endl;

  // auto status = ToStatusAndRelease(qnn_ep->GetCapability(qnn_ep, ep_graph->ToExternal(), &graph_support_info));
  qnn_ep_factory->ReleaseEp(qnn_ep_factory, qnn_ep);

  // return status.IsOK() ? BackendSupport::SUPPORTED : BackendSupport::UNSUPPORTED;
  return BackendSupport::SUPPORTED;
}

void QnnHTPBackendTests::SetUp() {
  if (cached_htp_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  Ort::Logger logger = Ort::Logger();

  // // Determine if HTP backend is supported only if we done so haven't before.
  // if (cached_htp_support_ == BackendSupport::SUPPORT_UNKNOWN) {
  //   cached_htp_support_ = GetHTPSupport(reinterpret_cast<OrtLogger*>(&logger));
  // }

  // if (cached_htp_support_ == BackendSupport::UNSUPPORTED) {
  //   ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "QNN HTP backend is not available! Skipping test.");
  //   GTEST_SKIP();
  // } else if (cached_htp_support_ == BackendSupport::SUPPORT_ERROR) {
  //   ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, "Failed to check if QNN HTP backend is available.");
  //   FAIL();
  // }
}

// TODO
// There is an unknown behavior that "soc_model" config somehow remains in HTP backend throughout different testcases
// within the same process. Once the option "soc_model=60" is set, all following testcases would be implicitly applied
// (which can be checked in verbose logging). This problem causes QnnHTPBackendTests.Inverse_2d/3d/4d on Linux failed
// with accuracy issue 0.259040415 vs. 0.256103545. Although this precision mismatch may be expected as HTP fp16/fp32
// behaviors could differ on different platforms, here adopts workaround to avoid modifying non-ABI parts. Concretely,
// "soc_model=30", which is the default setting, is set to HTP backend again after QnnHTPBackendTests testsuite is
// completed. Note that this is not an ABI-specific issue and exists in non-ABI UT as well but it is not observed
// previously due to the execution order of testcases.
void QnnHTPBackendTests::TearDownTestSuite() {
#if !defined(__aarch64__) && !defined(_M_ARM64)
  if (cached_htp_support_ != BackendSupport::SUPPORTED) {
    return;
  }

  Ort::Logger logger = Ort::Logger();

  // Build simple Relu graph.
  ModelPublicBuilder helper;

  auto build_test_case = BuildOpTestCase<float, float>(
      "Relu",
      {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
      {},
      {});

  build_test_case(helper);
  // helper.SetGraphOutputs();
  // auto status = model.MainGraph().Resolve();
  // if (!status.IsOK()) {
  //   LOGS(logger, WARNING) << "Failed to tear down QnnHTPBackendTests.";
  //   return;
  // }

  // Create QNN EP and call GetCapability().
  // onnxruntime::GraphViewer graph_viewer(graph);
  // std::unique_ptr<EpGraph> ep_graph = nullptr;
  // if (!EpGraph::Create(graph_viewer, ep_graph).IsOK()) {
  //   LOGS(logger, WARNING) << "Failed to tear down QnnHTPBackendTests.";
  //   return;
  // }
  // OrtEpGraphSupportInfo graph_support_info(*ep_graph);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;
  ProviderOptions provider_options = {{"backend_type", "htp"}, {"soc_model", "30"}};
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  OrtEpFactory* qnn_ep_factory = registered_ep_device->GetMutableFactory();
  OrtEp* qnn_ep = nullptr;
  if (qnn_ep_factory->CreateEp(qnn_ep_factory, nullptr, nullptr, 0, session_options, reinterpret_cast<OrtLogger*>(&logger), &qnn_ep)) {
    qnn_ep_factory->ReleaseEp(qnn_ep_factory, qnn_ep);
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "Failed to tear down QnnHTPBackendTests.");
    return;
  }

  // status = ToStatusAndRelease(qnn_ep->GetCapability(qnn_ep, ep_graph->ToExternal(), &graph_support_info));
  qnn_ep_factory->ReleaseEp(qnn_ep_factory, qnn_ep);
#endif  // !defined(__aarch64__) && !defined(_M_ARM64)
}

// Checks if Qnn Gpu backend can run a graph on the system.
// Creates a one node graph with relu op,
// then calls QNN EP's GetCapability() function
// to check if the GPU backend is available.
static BackendSupport GetGPUSupport(const OrtLogger* logger) {
  ModelPublicBuilder helper;

  // Build simple QDQ graph: DQ -> InstanceNormalization -> Q
  auto build_test_case = BuildOpTestCase<float, float>(
      "Relu",
      {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
      {},
      {});

  build_test_case(helper);
  // helper.SetGraphOutputs();
  // auto status = model.MainGraph().Resolve();

  // if (!status.IsOK()) {
  //   return BackendSupport::SUPPORT_ERROR;
  // }

  // Create QNN EP and call GetCapability().
  // onnxruntime::GraphViewer graph_viewer(graph);
  // std::unique_ptr<EpGraph> ep_graph = nullptr;
  // if (!EpGraph::Create(graph_viewer, ep_graph).IsOK()) {
  //   return BackendSupport::UNSUPPORTED;
  // }
  // OrtEpGraphSupportInfo graph_support_info(*ep_graph);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;
  ProviderOptions provider_options = {{"backend_type", "gpu"}, {"offload_graph_io_quantization", "0"}};
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  OrtEpFactory* qnn_ep_factory = registered_ep_device->GetMutableFactory();
  OrtEp* qnn_ep = nullptr;
  if (qnn_ep_factory->CreateEp(qnn_ep_factory, nullptr, nullptr, 0, session_options, logger, &qnn_ep)) {
    qnn_ep_factory->ReleaseEp(qnn_ep_factory, qnn_ep);
    return BackendSupport::UNSUPPORTED;
  }

  // status = ToStatusAndRelease(qnn_ep->GetCapability(qnn_ep, ep_graph->ToExternal(), &graph_support_info));
  qnn_ep_factory->ReleaseEp(qnn_ep_factory, qnn_ep);

  return BackendSupport::SUPPORTED;
}

void QnnGPUBackendTests::SetUp() {
  if (cached_gpu_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  Ort::Logger logger = Ort::Logger();

  // Determine if GPU backend is supported only if we haven't done so before.
  if (cached_gpu_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_gpu_support_ = GetGPUSupport(reinterpret_cast<OrtLogger*>(&logger));  // BackendSupport::SUPPORTED;
  }

  if (cached_gpu_support_ == BackendSupport::UNSUPPORTED) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "QNN GPU backend is not available! Skipping test.");
    GTEST_SKIP();
  } else if (cached_gpu_support_ == BackendSupport::SUPPORT_ERROR) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, "Failed to check if QNN GPU backend is available.");
    FAIL();
  }
}

static BackendSupport GetIRSupport(const OrtLogger* logger);

BackendSupport QnnHTPBackendTests::IsIRBackendSupported() const {
  Ort::Logger logger = Ort::Logger();

  if (cached_ir_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_ir_support_ = test::GetIRSupport(reinterpret_cast<OrtLogger*>(&logger));
  }

  return cached_ir_support_;
}

// Testing helper function that calls QNN EP's GetCapability() function with a mock graph to check
// if the QNN CPU backend is available.
// TODO: Remove once the QNN CPU backend works on Windows ARM64 pipeline VM.
static BackendSupport GetCPUSupport(const OrtLogger* logger, const std::string& backend_type = "cpu") {
  ModelPublicBuilder helper;

  auto get_test_model_func = [](const std::vector<int64_t>& input_shape) -> BuildTestModelFn {
    return [input_shape](ModelPublicBuilder& builder) {
      const int64_t num_channels = input_shape[1];

      builder.MakeInitializer<float>("scale", {num_channels}, 0.0f, 1.0f);
      builder.MakeInitializer<float>("bias", {num_channels}, 0.0f, 4.0f);
      builder.MakeInput<float>("X", input_shape, 0.0f, 10.0f);
      builder.MakeOutput("Y");
      builder.AddNode(
        "in",
        "InstanceNormalization",
        {"X", "scale", "bias"},
        {"Y"});
    };
  };

  // Build simple graph with a InstanceNormalization op.
  BuildTestModelFn build_test_case = get_test_model_func({1, 2, 3, 3});
  build_test_case(helper);
  // helper.SetGraphOutputs();
  // auto status = model.MainGraph().Resolve();

  // if (!status.IsOK()) {
  //   return BackendSupport::SUPPORT_ERROR;
  // }

  // Create QNN EP and call GetCapability().
  // onnxruntime::GraphViewer graph_viewer(graph);
  // std::unique_ptr<EpGraph> ep_graph = nullptr;
  // if (!EpGraph::Create(graph_viewer, ep_graph).IsOK()) {
  //   return BackendSupport::UNSUPPORTED;
  // }
  // OrtEpGraphSupportInfo graph_support_info(*ep_graph);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;
  ProviderOptions provider_options = {{"backend_type", backend_type}, {"offload_graph_io_quantization", "0"}};
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  OrtEpFactory* qnn_ep_factory = registered_ep_device->GetMutableFactory();
  OrtEp* qnn_ep = nullptr;
  if (qnn_ep_factory->CreateEp(qnn_ep_factory, nullptr, nullptr, 0, session_options, logger, &qnn_ep)) {
    qnn_ep_factory->ReleaseEp(qnn_ep_factory, qnn_ep);
    return BackendSupport::UNSUPPORTED;
  }

  // status = ToStatusAndRelease(qnn_ep->GetCapability(qnn_ep, ep_graph->ToExternal(), &graph_support_info));
  qnn_ep_factory->ReleaseEp(qnn_ep_factory, qnn_ep);

  return BackendSupport::SUPPORTED;
}

void QnnCPUBackendTests::SetUp() {
  return;
  if (cached_cpu_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  Ort::Logger logger = Ort::Logger();

  // Determine if CPU backend is supported only if we done so haven't before.
  if (cached_cpu_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_cpu_support_ = GetCPUSupport(reinterpret_cast<OrtLogger*>(&logger));
  }

  if (cached_cpu_support_ == BackendSupport::UNSUPPORTED) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "QNN CPU backend is not available! Skipping test.");
    GTEST_SKIP();
  } else if (cached_cpu_support_ == BackendSupport::SUPPORT_ERROR) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, "Failed to check if QNN CPU backend is available.");
    FAIL();
  }
}

static BackendSupport GetIRSupport(const OrtLogger* logger) {
  // QnnIr should be able to serialize any model supported by the QNN reference spec.
  // Use a model that works on QnnCpu to verify QnnIr availability.
  return GetCPUSupport(logger, "ir");
}

void QnnIRBackendTests::SetUp() {
  if (cached_ir_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  Ort::Logger logger = Ort::Logger();

  // Determine if IR backend is supported only if we done so haven't before.
  if (cached_ir_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_ir_support_ = GetIRSupport(reinterpret_cast<OrtLogger*>(&logger));
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
