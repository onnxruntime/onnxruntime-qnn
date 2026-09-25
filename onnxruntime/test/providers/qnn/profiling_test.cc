// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <thread>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#include <evntcons.h>
#include <evntrace.h>
#endif

#include "gtest/gtest.h"
#include "core/providers/qnn/ort_api.h"
#include "test/providers/qnn/qnn_test_utils.h"

#if QNN_ORT_EP_PROFILING_API_ENABLED

namespace onnxruntime {
namespace test {
namespace {

std::unique_ptr<ModelAndBuilder> BuildReluModel(size_t node_count = 1) {
  std::unique_ptr<ModelAndBuilder> model;
  CreateModelInMemory(model, [node_count](ModelTestBuilder& builder) {
    const std::vector<int64_t> shape{1, 4};
    MakeTestInput<float>(builder, "X", TestInputDef<float>(shape, false, {-1.0f, 2.0f, -3.0f, 4.0f}));
    std::string input_name = "X";
    for (size_t i = 0; i < node_count; ++i) {
      const std::string output_name = (i + 1 == node_count) ? "Y" : "Y" + std::to_string(i);
      builder.AddNode("relu" + std::to_string(i), "Relu", {input_name}, {output_name});
      input_name = output_name;
    }
    builder.MakeOutput("Y");
  });
  return model;
}

std::unique_ptr<ModelAndBuilder> BuildMixedEpContextModel() {
  std::unique_ptr<ModelAndBuilder> model;
  CreateModelInMemory(model, [](ModelTestBuilder& builder) {
    const std::vector<int64_t> shape{1, 4};
    MakeTestInput<float>(builder, "X", TestInputDef<float>(shape, false, {-1.0f, 2.0f, -3.0f, 4.0f}));
    builder.AddNode("qnn_abs", "Abs", {"X"}, {"abs_out"});
    builder.AddNode("cpu_nonzero", "NonZero", {"abs_out"}, {"Y"});
    builder.MakeOutput("Y");
  });
  return model;
}

void SetBackendType(ProviderOptions& provider_options, const char* backend_type) {
  provider_options["backend_type"] = backend_type;
#if defined(__linux__) && !defined(__aarch64__)
  if (std::string_view{backend_type} == "htp") {
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  }
#endif
}

size_t CountQnnRootEvents(const std::string& profile) {
  size_t count = 0;
  size_t pos = 0;
  // ORT's profiler serializes event args as `"level" : "ROOT"` (spaces around the colon).
  while ((pos = profile.find("\"level\" : \"ROOT\"", pos)) != std::string::npos) {
    ++count;
    ++pos;
  }
  return count;
}

size_t CountQnnRootEventsForOperation(const std::string& profile, const char* operation) {
  const std::string operation_arg = "\"qnn_operation\" : \"" + std::string(operation) + "\"";
  size_t count = 0;
  size_t event_begin = 0;
  while ((event_begin = profile.find("{\"cat\"", event_begin)) != std::string::npos) {
    const size_t next_event = profile.find("{\"cat\"", event_begin + 1);
    const std::string_view event(profile.data() + event_begin,
                                 (next_event == std::string::npos ? profile.size() : next_event) - event_begin);
    if (event.find("\"level\" : \"ROOT\"") != std::string_view::npos &&
        event.find(operation_arg) != std::string_view::npos) {
      ++count;
    }
    event_begin = next_event;
  }
  return count;
}

void ExpectQnnOrtProfileMetadata(const std::string& profile) {
  EXPECT_NE(profile.find("\"level\" : \"ROOT\""), std::string::npos);
  EXPECT_NE(profile.find("\"parent_ort_node\""), std::string::npos);
  EXPECT_NE(profile.find("\"unit\""), std::string::npos);
  EXPECT_NE(profile.find("\"qnn_event_type\""), std::string::npos);
  EXPECT_NE(profile.find("\"qnn_event_identifier\""), std::string::npos);
  EXPECT_NE(profile.find("\"qnn_timing_source\" : \"BACKEND\""), std::string::npos);
  EXPECT_NE(profile.find("\"qnn_graph_name\""), std::string::npos);
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void RemoveProfilesWithPrefix(const std::filesystem::path& prefix) {
  const std::filesystem::path dir = prefix.parent_path().empty() ? "." : prefix.parent_path();
  const std::string base = prefix.filename().string();
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec || !entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (filename.rfind(base, 0) == 0 && entry.path().extension() == ".json") {
      std::filesystem::remove(entry.path(), ec);
    }
  }
}

std::filesystem::path FindProfileWithPrefix(const std::filesystem::path& prefix) {
  const std::filesystem::path dir = prefix.parent_path().empty() ? "." : prefix.parent_path();
  const std::string base = prefix.filename().string();
  std::filesystem::path found;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec || !entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (filename.rfind(base, 0) == 0 && entry.path().extension() == ".json") {
      found = entry.path();
      break;
    }
  }
  return found;
}

std::filesystem::path MakeProfilePrefix(const std::string& stem, const void* unique) {
  return std::filesystem::temp_directory_path() /
         (stem + "_" + std::to_string(reinterpret_cast<uintptr_t>(unique)));
}

std::basic_string<ORTCHAR_T> ToOrtPathString(const std::filesystem::path& path) {
#ifdef _WIN32
  return path.wstring();
#else
  return path.string();
#endif
}

bool HasQnnOperation(const std::string& profile, const char* operation,
                     const char* parent_ort_node = nullptr) {
  const std::string operation_arg = "\"qnn_operation\" : \"" + std::string(operation) + "\"";
  const std::string parent_arg = parent_ort_node == nullptr
                                     ? std::string{}
                                     : "\"parent_ort_node\" : \"" + std::string(parent_ort_node) + "\"";

  size_t event_begin = 0;
  while ((event_begin = profile.find("{\"cat\"", event_begin)) != std::string::npos) {
    const size_t next_event = profile.find("{\"cat\"", event_begin + 1);
    const std::string_view event(profile.data() + event_begin,
                                 (next_event == std::string::npos ? profile.size() : next_event) - event_begin);
    if (event.find(operation_arg) != std::string_view::npos &&
        (parent_ort_node == nullptr || event.find(parent_arg) != std::string_view::npos)) {
      return true;
    }
    event_begin = next_event;
  }
  return false;
}

void ExpectSessionInitializationSetupOperations(const std::string& profile) {
  EXPECT_TRUE(HasQnnOperation(profile, "compose", "session_initialization"));
  EXPECT_TRUE(HasQnnOperation(profile, "finalize", "session_initialization"));
}

bool HasQnnEvent(const std::string& profile, const char* event_identifier,
                 const char* operation = nullptr) {
  const std::string identifier_arg =
      "\"qnn_event_identifier\" : \"" + std::string(event_identifier) + "\"";
  const std::string operation_arg = operation == nullptr
                                        ? std::string{}
                                        : "\"qnn_operation\" : \"" + std::string(operation) + "\"";

  size_t event_begin = 0;
  while ((event_begin = profile.find("{\"cat\"", event_begin)) != std::string::npos) {
    const size_t next_event = profile.find("{\"cat\"", event_begin + 1);
    const std::string_view event(profile.data() + event_begin,
                                 (next_event == std::string::npos ? profile.size() : next_event) - event_begin);
    if (event.find(identifier_arg) != std::string_view::npos &&
        (operation == nullptr || event.find(operation_arg) != std::string_view::npos)) {
      return true;
    }
    event_begin = next_event;
  }
  return false;
}

void ExpectNoQnnSetupOperations(const std::string& profile) {
  EXPECT_FALSE(HasQnnOperation(profile, "compose"));
  EXPECT_FALSE(HasQnnOperation(profile, "finalize"));
  EXPECT_FALSE(HasQnnOperation(profile, "context_load"));
}

Ort::Value MakeReluInput(std::array<float, 4>& input_data) {
  const std::array<int64_t, 2> input_shape{1, 4};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  return Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                         input_shape.data(), input_shape.size());
}

size_t RunReluWithRunProfiling(Ort::Session& session, const std::filesystem::path& profile_prefix) {
  RemoveProfilesWithPrefix(profile_prefix);

  std::array<float, 4> input_data{-1.0f, 2.0f, -3.0f, 4.0f};
  Ort::Value input = MakeReluInput(input_data);
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};

  Ort::RunOptions run_options;
  const auto ort_profile_prefix = ToOrtPathString(profile_prefix);
  run_options.EnableProfiling(ort_profile_prefix.c_str());
  auto outputs = session.Run(run_options, input_names, &input, 1, output_names, 1);
  if (outputs.size() != 1u) {
    ADD_FAILURE() << "Relu run did not return one output";
    return 0;
  }

  const std::filesystem::path profile_path = FindProfileWithPrefix(profile_prefix);
  if (profile_path.empty()) {
    ADD_FAILURE() << "Run profiling did not produce an ORT profile for prefix " << profile_prefix.string();
    return 0;
  }

  const std::string profile = ReadFile(profile_path);
  ExpectQnnOrtProfileMetadata(profile);
  ExpectNoQnnSetupOperations(profile);
  const size_t root_events = CountQnnRootEvents(profile);
  std::error_code ec;
  std::filesystem::remove(profile_path, ec);
  return root_events;
}

bool RunReluNoProfiling(Ort::Session& session) {
  std::array<float, 4> input_data{-1.0f, 2.0f, -3.0f, 4.0f};
  Ort::Value input = MakeReluInput(input_data);
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};

  auto outputs = session.Run(Ort::RunOptions{}, input_names, &input, 1, output_names, 1);
  return outputs.size() == 1u;
}

bool RunUnprofiledRelu(const char* backend_type, const std::filesystem::path* provider_profile_path = nullptr) {
  auto model = BuildReluModel();
  ProviderOptions provider_options;
  SetBackendType(provider_options, backend_type);
  if (provider_profile_path != nullptr) {
    provider_options["profiling_file_path"] = provider_profile_path->string();
  }
  Ort::SessionOptions session_options;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));
  return RunReluNoProfiling(scoped.session()) &&
         (provider_profile_path == nullptr || !std::filesystem::exists(*provider_profile_path));
}

#ifdef _WIN32

// The QNN plugin defines this TraceLogging provider in qnn_telemetry.cc.
constexpr GUID kOrtTraceLoggingProviderGuid{
    0x3a26b1ff, 0x7484, 0x7484, {0x74, 0x84, 0x15, 0x26, 0x1f, 0x42, 0x61, 0x4d}};
constexpr ULONGLONG kOrtProfilingKeyword = 0x100;

class ScopedEtwSession {
 public:
  ScopedEtwSession()
      : name_(L"qnn_profiling_test_" + std::to_wstring(GetCurrentProcessId()) + L"_" +
              std::to_wstring(reinterpret_cast<uintptr_t>(this))),
        etl_path_(std::filesystem::temp_directory_path() /
                  ("qnn_profiling_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".etl")) {}

  ~ScopedEtwSession() {
    Stop();
    std::error_code ec;
    std::filesystem::remove(etl_path_, ec);
  }

  bool Start() {
    const std::wstring etl_path = etl_path_.wstring();
    const size_t name_bytes = (name_.size() + 1) * sizeof(wchar_t);
    const size_t path_bytes = (etl_path.size() + 1) * sizeof(wchar_t);
    properties_buffer_.resize(sizeof(EVENT_TRACE_PROPERTIES) + name_bytes + path_bytes);
    auto* properties = reinterpret_cast<EVENT_TRACE_PROPERTIES*>(properties_buffer_.data());
    *properties = {};
    properties->Wnode.BufferSize = static_cast<ULONG>(properties_buffer_.size());
    properties->Wnode.Flags = WNODE_FLAG_TRACED_GUID;
    properties->LogFileMode = EVENT_TRACE_FILE_MODE_SEQUENTIAL;
    properties->LoggerNameOffset = sizeof(EVENT_TRACE_PROPERTIES);
    properties->LogFileNameOffset = sizeof(EVENT_TRACE_PROPERTIES) + static_cast<ULONG>(name_bytes);
    std::memcpy(properties_buffer_.data() + properties->LoggerNameOffset, name_.c_str(), name_bytes);
    std::memcpy(properties_buffer_.data() + properties->LogFileNameOffset, etl_path.c_str(), path_bytes);

    const ULONG start_status = StartTraceW(&session_handle_, name_.c_str(), properties);
    if (start_status != ERROR_SUCCESS) {
      error_ = "StartTraceW failed with " + std::to_string(start_status);
      return false;
    }

    const ULONG enable_status = EnableTraceEx2(session_handle_, &kOrtTraceLoggingProviderGuid,
                                               EVENT_CONTROL_CODE_ENABLE_PROVIDER, TRACE_LEVEL_VERBOSE,
                                               kOrtProfilingKeyword, 0, 0, nullptr);
    if (enable_status != ERROR_SUCCESS) {
      error_ = "EnableTraceEx2 failed with " + std::to_string(enable_status);
      Stop();
      return false;
    }
    return true;
  }

  void Stop() {
    if (session_handle_ == 0) {
      return;
    }
    auto* properties = reinterpret_cast<EVENT_TRACE_PROPERTIES*>(properties_buffer_.data());
    (void)ControlTraceW(session_handle_, name_.c_str(), properties, EVENT_TRACE_CONTROL_STOP);
    session_handle_ = 0;
  }

  size_t CountProviderEvents() const {
    current_counter_ = 0;
    EVENT_TRACE_LOGFILEW trace{};
    const std::wstring etl_path = etl_path_.wstring();
    trace.LogFileName = const_cast<LPWSTR>(etl_path.c_str());
    trace.ProcessTraceMode = PROCESS_TRACE_MODE_EVENT_RECORD;
    trace.EventRecordCallback = [](PEVENT_RECORD event) {
      if (IsEqualGUID(event->EventHeader.ProviderId, kOrtTraceLoggingProviderGuid)) {
        ++current_counter_;
      }
    };
    TRACEHANDLE trace_handle = OpenTraceW(&trace);
    if (trace_handle == INVALID_PROCESSTRACE_HANDLE) {
      return 0;
    }
    (void)ProcessTrace(&trace_handle, 1, nullptr, nullptr);
    (void)CloseTrace(trace_handle);
    return current_counter_;
  }

  const std::string& error() const { return error_; }

 private:
  std::wstring name_;
  std::filesystem::path etl_path_;
  std::vector<unsigned char> properties_buffer_;
  TRACEHANDLE session_handle_ = 0;
  std::string error_;
  inline static size_t current_counter_ = 0;
};

bool RunEtwProfiledRelu() {
  auto model = BuildReluModel();
  ProviderOptions provider_options;
  SetBackendType(provider_options, "htp");
  provider_options["profiling_level"] = "basic";

  Ort::SessionOptions session_options;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));
  return RunReluNoProfiling(scoped.session());
}

#endif  // _WIN32

size_t RunProfiledRelu(size_t run_count, bool set_profiling_level, const char* backend_type = "htp") {
  auto model = BuildReluModel();
  const auto profile_prefix = MakeProfilePrefix("qnn_ort_profile", model.get());

  ProviderOptions provider_options;
  SetBackendType(provider_options, backend_type);
  if (set_profiling_level) {
    provider_options["profiling_level"] = "basic";
  }

  Ort::SessionOptions session_options;
  const auto ort_profile_prefix = ToOrtPathString(profile_prefix);
  session_options.EnableProfiling(ort_profile_prefix.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));

  std::array<float, 4> input_data{-1.0f, 2.0f, -3.0f, 4.0f};
  Ort::Value input = MakeReluInput(input_data);
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};

  for (size_t i = 0; i < run_count; ++i) {
    auto outputs = scoped.session().Run(Ort::RunOptions{}, input_names, &input, 1, output_names, 1);
    if (outputs.size() != 1u) {
      ADD_FAILURE() << "Relu run did not return one output";
      return 0;
    }
  }

  Ort::AllocatorWithDefaultOptions allocator;
  const auto profile_path = scoped.session().EndProfilingAllocated(allocator);
  const std::string profile = ReadFile(profile_path.get());
  ExpectQnnOrtProfileMetadata(profile);
  std::error_code ec;
  std::filesystem::remove(profile_path.get(), ec);
  return CountQnnRootEventsForOperation(profile, "execute");
}

bool RunSessionProfiledReluThenUnprofiledRelu() {
  auto model = BuildReluModel();

  ProviderOptions provider_options;
  SetBackendType(provider_options, "htp");

  Ort::SessionOptions session_options;
  const auto profile_prefix = MakeProfilePrefix("qnn_ort_profile_end_then_run", model.get());
  const auto ort_profile_prefix = ToOrtPathString(profile_prefix);
  session_options.EnableProfiling(ort_profile_prefix.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));

  if (!RunReluNoProfiling(scoped.session())) {
    ADD_FAILURE() << "Relu run during session profiling did not return one output";
    return false;
  }

  Ort::AllocatorWithDefaultOptions allocator;
  const auto profile_path = scoped.session().EndProfilingAllocated(allocator);
  const std::string profile = ReadFile(profile_path.get());
  ExpectQnnOrtProfileMetadata(profile);
  std::error_code ec;
  std::filesystem::remove(profile_path.get(), ec);

  if (!RunReluNoProfiling(scoped.session())) {
    ADD_FAILURE() << "Relu run after EndProfiling did not return one output";
    return false;
  }

  return true;
}

size_t RunProfiledReluWithCsv(const char* backend_type = "htp", bool enable_ort_profiling = true,
                              const char* profiling_level = "basic", bool enable_framework_op_trace = false,
                              size_t run_count = 1) {
  auto model = BuildReluModel();

  const std::filesystem::path csv_path = MakeProfilePrefix("qnn_ort_profile_csv", model.get()).replace_extension(".csv");
  const std::filesystem::path qnn_log_path = csv_path.parent_path() / (csv_path.stem().string() + "_qnn.log");
  const std::filesystem::path ort_prefix = MakeProfilePrefix("qnn_ort_profile_json", model.get());
  const std::filesystem::path trace_dir = MakeProfilePrefix("qnn_ort_profile_trace", model.get());
  std::error_code ec;
  std::filesystem::remove(csv_path, ec);
  std::filesystem::remove(qnn_log_path, ec);
  std::filesystem::remove_all(trace_dir, ec);
  if (enable_framework_op_trace) {
    std::filesystem::create_directories(trace_dir, ec);
    if (ec) {
      ADD_FAILURE() << "Failed to create framework trace directory " << trace_dir << ": " << ec.message();
      return 0;
    }
  }

  ProviderOptions provider_options;
  SetBackendType(provider_options, backend_type);
  provider_options["profiling_level"] = profiling_level;
  provider_options["profiling_file_path"] = csv_path.string();
  if (enable_framework_op_trace) {
    provider_options["enable_framework_op_trace"] = "1";
    provider_options["framework_op_trace_dir"] = trace_dir.string();
  }

  Ort::SessionOptions session_options;
  if (enable_ort_profiling) {
    const auto ort_profile_prefix = ToOrtPathString(ort_prefix);
    session_options.EnableProfiling(ort_profile_prefix.c_str());
  }

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));

  std::array<float, 4> input_data{-1.0f, 2.0f, -3.0f, 4.0f};
  Ort::Value input = MakeReluInput(input_data);
  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  for (size_t i = 0; i < run_count; ++i) {
    auto outputs = scoped.session().Run(Ort::RunOptions{}, input_names, &input, 1, output_names, 1);
    if (outputs.size() != 1u) {
      ADD_FAILURE() << "Relu run did not return one output";
      return 0;
    }
  }

  size_t root_events = 0;
  std::filesystem::path profile_path;
  if (enable_ort_profiling) {
    Ort::AllocatorWithDefaultOptions allocator;
    const auto allocated_profile_path = scoped.session().EndProfilingAllocated(allocator);
    profile_path = allocated_profile_path.get();
    const std::string profile = ReadFile(profile_path);
    ExpectQnnOrtProfileMetadata(profile);
    root_events = CountQnnRootEventsForOperation(profile, "execute");
#ifdef _WIN32
    if (std::string_view{backend_type} == "htp" && std::string_view{profiling_level} == "detailed") {
      EXPECT_TRUE(HasQnnEvent(profile, "RPC (execute) time", "execute"));
      EXPECT_TRUE(HasQnnEvent(profile, "QNN accelerator (execute) time", "execute"));
      EXPECT_TRUE(HasQnnEvent(profile, "Accelerator (execute) time", "execute"));
    }
#endif
  }

  EXPECT_TRUE(std::filesystem::exists(csv_path));
  EXPECT_GT(std::filesystem::file_size(csv_path), 0u);
  const std::string csv = ReadFile(csv_path);
  EXPECT_NE(csv.find("Event Level"), std::string::npos);
  EXPECT_NE(csv.find("Time"), std::string::npos);
  EXPECT_NE(csv.find("Unit of Measurement"), std::string::npos);
  EXPECT_NE(csv.find(",ROOT,"), std::string::npos);
  if (enable_framework_op_trace && std::string_view{profiling_level} != "basic") {
    EXPECT_NE(csv.find("ONNX Source Ops"), std::string::npos);
    EXPECT_NE(csv.find(",NODE,"), std::string::npos);
  }

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  if (enable_ort_profiling && std::string_view{backend_type} == "htp") {
    const bool qnn_log_exists = std::filesystem::exists(qnn_log_path);
    EXPECT_TRUE(qnn_log_exists);
    if (qnn_log_exists) {
      EXPECT_GT(std::filesystem::file_size(qnn_log_path), 0u);
    }
  }
#endif

  if (!profile_path.empty()) {
    std::filesystem::remove(profile_path, ec);
  }
  std::filesystem::remove(csv_path, ec);
  std::filesystem::remove(qnn_log_path, ec);
  std::filesystem::remove_all(trace_dir, ec);
  return root_events;
}

bool RunConcurrentRunProfilingIsolationTest(const char* backend_type) {
  auto model = BuildReluModel(128);

  ProviderOptions provider_options;
  SetBackendType(provider_options, backend_type);

  Ort::SessionOptions session_options;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));

  const auto baseline_prefix = MakeProfilePrefix("qnn_ort_run_profile_baseline", model.get());
  const size_t baseline_root_events = RunReluWithRunProfiling(scoped.session(), baseline_prefix);
  if (baseline_root_events == 0u) {
    return false;
  }

  const auto profiled_prefix = MakeProfilePrefix("qnn_ort_run_profile_isolated", &scoped.session());
  size_t profiled_root_events = 0;
  std::string profiled_error;
  std::string unprofiled_error;
  std::atomic<bool> start{false};

  std::thread profiled_run([&]() {
    try {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      profiled_root_events = RunReluWithRunProfiling(scoped.session(), profiled_prefix);
    } catch (const std::exception& e) {
      profiled_error = e.what();
    }
  });

  std::thread unprofiled_run([&]() {
    try {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (size_t i = 0; i < 8; ++i) {
        if (!RunReluNoProfiling(scoped.session())) {
          unprofiled_error = "Relu run did not return one output";
          break;
        }
      }
    } catch (const std::exception& e) {
      unprofiled_error = e.what();
    }
  });

  start.store(true, std::memory_order_release);
  profiled_run.join();
  unprofiled_run.join();

  EXPECT_TRUE(profiled_error.empty()) << profiled_error;
  EXPECT_TRUE(unprofiled_error.empty()) << unprofiled_error;
  return profiled_error.empty() && unprofiled_error.empty() &&
         profiled_root_events == baseline_root_events;
}

bool RunConcurrentRunProfilingTest(const char* backend_type) {
  auto model = BuildReluModel(128);

  ProviderOptions provider_options;
  SetBackendType(provider_options, backend_type);

  Ort::SessionOptions session_options;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));

  const auto baseline_prefix = MakeProfilePrefix("qnn_ort_run_profile_concurrent_baseline", model.get());
  const size_t baseline_root_events = RunReluWithRunProfiling(scoped.session(), baseline_prefix);
  if (baseline_root_events == 0u) {
    return false;
  }

  const auto prefix1 = MakeProfilePrefix("qnn_ort_run_profile_1", model.get());
  const auto prefix2 = MakeProfilePrefix("qnn_ort_run_profile_2", &scoped.session());
  size_t root_events1 = 0;
  size_t root_events2 = 0;
  std::string error1;
  std::string error2;

  std::thread run1([&]() {
    try {
      root_events1 = RunReluWithRunProfiling(scoped.session(), prefix1);
    } catch (const std::exception& e) {
      error1 = e.what();
    }
  });
  std::thread run2([&]() {
    try {
      root_events2 = RunReluWithRunProfiling(scoped.session(), prefix2);
    } catch (const std::exception& e) {
      error2 = e.what();
    }
  });
  run1.join();
  run2.join();

  EXPECT_TRUE(error1.empty()) << error1;
  EXPECT_TRUE(error2.empty()) << error2;
  return error1.empty() && error2.empty() &&
         root_events1 == baseline_root_events && root_events2 == baseline_root_events;
}

}  // namespace

TEST_F(QnnHTPBackendTests, OrtProfilingApiIntegration) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const size_t events_from_one_run = RunProfiledRelu(1, true);
  const size_t events_from_two_runs = RunProfiledRelu(2, true);

  ASSERT_GT(events_from_one_run, 0u);
  EXPECT_EQ(events_from_two_runs, 2 * events_from_one_run);
}

TEST_F(QnnHTPBackendTests, OrtProfilingApiLazyInitNoProfilingLevel) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_GT(RunProfiledRelu(1, false), 0u);
}

TEST_F(QnnHTPBackendTests, OrtProfilingApiEndProfilingReleasesLazyHandle) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_TRUE(RunSessionProfiledReluThenUnprofiledRelu());
}

TEST_F(QnnHTPBackendTests, OrtProfilingWithCsvDoesNotDuplicatePriorRuns) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const size_t events_from_one_run = RunProfiledReluWithCsv("htp", true, "basic", false, 1);
  const size_t events_from_two_runs = RunProfiledReluWithCsv("htp", true, "basic", false, 2);

  ASSERT_GT(events_from_one_run, 0u);
  EXPECT_EQ(events_from_two_runs, 2 * events_from_one_run);
}

TEST_F(QnnHTPBackendTests, QnnOnlyProfilingSupportsMultipleRuns) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_EQ(RunProfiledReluWithCsv("htp", false, "basic", false, 3), 0u);
}

TEST_F(QnnHTPBackendTests, ProfilingDefaultDisabledDoesNotCreateConfiguredCsv) {
  const auto csv_path = MakeProfilePrefix("qnn_disabled_profile", this).replace_extension(".csv");
  std::error_code ec;
  std::filesystem::remove(csv_path, ec);
  EXPECT_TRUE(RunUnprofiledRelu("htp", &csv_path));
  std::filesystem::remove(csv_path, ec);
}

#ifdef _WIN32
TEST_F(QnnHTPBackendTests, EtwProfilingEmitsProviderEvents) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ScopedEtwSession etw_session;
  if (!etw_session.Start()) {
    GTEST_SKIP() << etw_session.error();
  }

  EXPECT_TRUE(RunEtwProfiledRelu());
  etw_session.Stop();
  EXPECT_GT(etw_session.CountProviderEvents(), 0u);
}
#endif  // _WIN32

TEST_F(QnnHTPBackendTests, OrtProfilingApiSessionBasic) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_GT(RunProfiledRelu(1, false, "htp"), 0u);
}

TEST_F(QnnHTPBackendTests, OrtSessionProfilingLazyBasicCapturesSetupOperations) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto model = BuildReluModel();
  const auto profile_prefix = MakeProfilePrefix("qnn_ort_lazy_setup_profile", model.get());

  ProviderOptions provider_options;
  SetBackendType(provider_options, "htp");
  // Session setup is attributed on the initializing thread; use the supported serial-finalize path.
  provider_options["num_graph_prepare_threads"] = "1";

  Ort::SessionOptions session_options;
  const auto ort_profile_prefix = ToOrtPathString(profile_prefix);
  session_options.EnableProfiling(ort_profile_prefix.c_str());
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));

  Ort::AllocatorWithDefaultOptions allocator;
  const auto profile_path = scoped.session().EndProfilingAllocated(allocator);
  const std::string profile = ReadFile(profile_path.get());
  ExpectSessionInitializationSetupOperations(profile);

  std::error_code ec;
  std::filesystem::remove(profile_path.get(), ec);
}

TEST_F(QnnHTPBackendTests, OrtSessionProfilingRetainsExecuteEventsAfterSetupExtraction) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto model = BuildReluModel();
  const auto profile_prefix = MakeProfilePrefix("qnn_ort_handle_lifecycle_profile", model.get());

  ProviderOptions provider_options;
  SetBackendType(provider_options, "htp");
  provider_options["num_graph_prepare_threads"] = "1";

  Ort::SessionOptions session_options;
  const auto ort_profile_prefix = ToOrtPathString(profile_prefix);
  session_options.EnableProfiling(ort_profile_prefix.c_str());
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));

  EXPECT_TRUE(RunReluNoProfiling(scoped.session()));
  Ort::AllocatorWithDefaultOptions allocator;
  const auto profile_path = scoped.session().EndProfilingAllocated(allocator);
  const std::string profile = ReadFile(profile_path.get());
  ExpectSessionInitializationSetupOperations(profile);
  EXPECT_TRUE(HasQnnEvent(profile, "QNN (execute) time", "execute"));

  std::error_code ec;
  std::filesystem::remove(profile_path.get(), ec);
}

TEST_F(QnnHTPBackendTests, OrtSessionProfilingCapturesSerialSetupOperationsAndCsv) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto model = BuildReluModel();
  const auto profile_prefix = MakeProfilePrefix("qnn_ort_setup_profile", model.get());
  const auto csv_path = MakeProfilePrefix("qnn_ort_setup_provider", model.get()).replace_extension(".csv");
  const auto qnn_log_path = csv_path.parent_path() / (csv_path.stem().string() + "_qnn.log");
  std::error_code ec;
  std::filesystem::remove(csv_path, ec);
  std::filesystem::remove(qnn_log_path, ec);

  ProviderOptions provider_options;
  SetBackendType(provider_options, "htp");
  provider_options["profiling_level"] = "basic";
  provider_options["profiling_file_path"] = csv_path.string();
  // The ORT profiler's event scope is thread-local. Keep finalize on the initialization
  // thread so the test covers the supported serial-finalize path.
  provider_options["num_graph_prepare_threads"] = "1";

  Ort::SessionOptions session_options;
  const auto ort_profile_prefix = ToOrtPathString(profile_prefix);
  session_options.EnableProfiling(ort_profile_prefix.c_str());
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));

  EXPECT_TRUE(RunReluNoProfiling(scoped.session()));
  Ort::AllocatorWithDefaultOptions allocator;
  const auto profile_path = scoped.session().EndProfilingAllocated(allocator);
  const std::string profile = ReadFile(profile_path.get());
  ExpectQnnOrtProfileMetadata(profile);
  ExpectSessionInitializationSetupOperations(profile);
  ASSERT_TRUE(std::filesystem::exists(csv_path));
  const std::string csv = ReadFile(csv_path);
  EXPECT_NE(csv.find("Event Level"), std::string::npos);
  EXPECT_NE(csv.find("Time"), std::string::npos);

  std::filesystem::remove(profile_path.get(), ec);
  std::filesystem::remove(csv_path, ec);
  std::filesystem::remove(qnn_log_path, ec);
}

TEST_F(QnnHTPBackendTests, OrtProfilingApiAotPhase2EmitsQnnEvents) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto model = BuildReluModel();
  const std::filesystem::path context_path = MakeProfilePrefix("qnn_ort_aot_context", model.get()).replace_extension(".onnx");
  const std::filesystem::path profile_prefix = MakeProfilePrefix("qnn_ort_aot_profile", model.get());
  const std::filesystem::path csv_path = MakeProfilePrefix("qnn_ort_aot_provider", model.get()).replace_extension(".csv");
  std::error_code ec;
  std::filesystem::remove(context_path, ec);
  std::filesystem::remove(csv_path, ec);

  // Phase 1 compiles the EPContext model without provider or ORT profiling.
  {
    ProviderOptions provider_options;
    SetBackendType(provider_options, "htp");
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_path.string().c_str());
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
    ScopedOrtSession scoped(
        std::move(registered_ep_device),
        Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));
  }
  ASSERT_TRUE(std::filesystem::exists(context_path));

  // Phase 2 loads the compiled context with both ORT and provider CSV profiling enabled.
  {
    ProviderOptions provider_options;
    SetBackendType(provider_options, "htp");
    provider_options["disable_file_mapped_weights"] = "1";
    provider_options["profiling_level"] = "basic";
    provider_options["profiling_file_path"] = csv_path.string();
    Ort::SessionOptions session_options;
    const auto ort_profile_prefix = ToOrtPathString(profile_prefix);
    session_options.EnableProfiling(ort_profile_prefix.c_str());
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
    ScopedOrtSession scoped(std::move(registered_ep_device),
                            Ort::Session(*GetOrtEnv(), context_path.c_str(), session_options));

    EXPECT_TRUE(RunReluNoProfiling(scoped.session()));
    Ort::AllocatorWithDefaultOptions allocator;
    const auto profile_path = scoped.session().EndProfilingAllocated(allocator);
    const std::string profile = ReadFile(profile_path.get());
    ExpectQnnOrtProfileMetadata(profile);
    EXPECT_TRUE(HasQnnOperation(profile, "context_load", "session_initialization"));
    ASSERT_TRUE(std::filesystem::exists(csv_path));
    const std::string csv = ReadFile(csv_path);
    EXPECT_NE(csv.find("Event Level"), std::string::npos);
    EXPECT_NE(csv.find("Time"), std::string::npos);
    std::filesystem::remove(profile_path.get(), ec);
  }
  std::filesystem::remove(context_path, ec);
  std::filesystem::remove(csv_path, ec);
}

TEST_F(QnnHTPBackendTests, AotPhase1QnnOnlyProfilingPreservesCsvOutput) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto model = BuildReluModel();
  const std::filesystem::path context_path = MakeProfilePrefix("qnn_aot_provider_context", model.get()).replace_extension(".onnx");
  const std::filesystem::path csv_path = MakeProfilePrefix("qnn_aot_provider_profile", model.get()).replace_extension(".csv");
  std::error_code ec;
  std::filesystem::remove(context_path, ec);
  std::filesystem::remove(csv_path, ec);
  std::filesystem::remove(csv_path.parent_path() / (csv_path.stem().string() + "_qnn.log"), ec);

  ProviderOptions provider_options;
  SetBackendType(provider_options, "htp");
  provider_options["profiling_level"] = "basic";
  provider_options["profiling_file_path"] = csv_path.string();
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_path.string().c_str());
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped(
      std::move(registered_ep_device),
      Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));

  ASSERT_TRUE(std::filesystem::exists(context_path));
  ASSERT_TRUE(std::filesystem::exists(csv_path));
  EXPECT_GT(std::filesystem::file_size(csv_path), 0u);
  EXPECT_NE(ReadFile(csv_path).find("Event Level"), std::string::npos);

  std::filesystem::remove(context_path, ec);
  std::filesystem::remove(csv_path, ec);
  std::filesystem::remove(csv_path.parent_path() / (csv_path.stem().string() + "_qnn.log"), ec);
}

TEST_F(QnnHTPBackendTests, OrtProfilingApiAotMixedEpContextIncludesQnnAndCpuEvents) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto model = BuildMixedEpContextModel();
  const std::filesystem::path context_path = MakeProfilePrefix("qnn_ort_aot_mixed_context", model.get()).replace_extension(".onnx");
  const std::filesystem::path profile_prefix = MakeProfilePrefix("qnn_ort_aot_mixed_profile", model.get());
  std::error_code ec;
  std::filesystem::remove(context_path, ec);

  {
    ProviderOptions provider_options;
    SetBackendType(provider_options, "htp");
    provider_options["offload_graph_io_quantization"] = "0";
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_path.string().c_str());
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
    ScopedOrtSession scoped(
        std::move(registered_ep_device),
        Ort::Session(*GetOrtEnv(), model->model_data.data(), model->model_data.size(), session_options));
  }
  ASSERT_TRUE(std::filesystem::exists(context_path));

  {
    ProviderOptions provider_options;
    SetBackendType(provider_options, "htp");
    provider_options["offload_graph_io_quantization"] = "0";
    provider_options["disable_file_mapped_weights"] = "1";
    Ort::SessionOptions session_options;
    const auto ort_profile_prefix = ToOrtPathString(profile_prefix);
    session_options.EnableProfiling(ort_profile_prefix.c_str());
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);
    ScopedOrtSession scoped(std::move(registered_ep_device),
                            Ort::Session(*GetOrtEnv(), context_path.c_str(), session_options));

    EXPECT_TRUE(RunReluNoProfiling(scoped.session()));
    Ort::AllocatorWithDefaultOptions allocator;
    const auto profile_path = scoped.session().EndProfilingAllocated(allocator);
    const std::string profile = ReadFile(profile_path.get());
    ExpectQnnOrtProfileMetadata(profile);
    EXPECT_NE(profile.find("cpu_nonzero"), std::string::npos)
        << "Mixed EPContext profile is missing the CPU fallback node event";
    std::filesystem::remove(profile_path.get(), ec);
  }
  std::filesystem::remove(context_path, ec);
}

TEST_F(QnnHTPBackendTests, OrtRunProfilingApiConcurrentRunsUseSeparateProfilers) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_TRUE(RunConcurrentRunProfilingTest("htp"));
}

TEST_F(QnnHTPBackendTests, OrtRunProfilingDoesNotCaptureConcurrentUnprofiledRun) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_TRUE(RunConcurrentRunProfilingIsolationTest("htp"));
}

TEST_F(QnnHTPBackendTests, OrtProfilingApiCoexistsWithCsvProfiling) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_GT(RunProfiledReluWithCsv("htp"), 0u);
}

TEST_F(QnnHTPBackendTests, OrtProfilingApiCoexistsWithDetailedFrameworkOpTrace) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_GT(RunProfiledReluWithCsv("htp", true, "detailed", true), 0u);
}

TEST_F(QnnHTPBackendTests, OrtProfilingApiCoexistsWithOptraceFrameworkOpTrace) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_GT(RunProfiledReluWithCsv("htp", true, "optrace", true), 0u);
}

TEST_F(QnnHTPBackendTests, QnnOnlyProfilingPreservesCsvOutput) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  EXPECT_EQ(RunProfiledReluWithCsv("htp", false), 0u);
}

TEST_F(QnnHTPBackendTests, ProfilingDefaultDisabled) {
  EXPECT_TRUE(RunUnprofiledRelu("htp"));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // QNN_ORT_EP_PROFILING_API_ENABLED
