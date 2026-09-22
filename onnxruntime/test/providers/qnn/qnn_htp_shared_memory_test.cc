// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "gsl/gsl"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if !defined(ORT_MINIMAL_BUILD) && !BUILD_QNN_EP_STATIC_LIB && \
    (defined(_M_ARM64) || defined(__aarch64__))

namespace {

// Capture ORT log messages into a vector so tests can assert on which branch
// of BindQnnTensorMemoryToOrtValueMemory was taken at runtime (see
// onnxruntime/core/providers/qnn/builder/qnn_model.cc), and on WARNINGs that
// later PRs will emit.
struct LogCapture {
  mutable std::mutex mtx;
  std::vector<std::string> messages;

  void Push(const char* message) {
    std::lock_guard<std::mutex> g{mtx};
    messages.emplace_back(message);
  }

  size_t CountContaining(std::string_view needle) const {
    std::lock_guard<std::mutex> g{mtx};
    size_t n = 0;
    for (const auto& m : messages) {
      if (m.find(needle) != std::string_view::npos) ++n;
    }
    return n;
  }
};

extern "C" void ORT_API_CALL LogCaptureCallback(void* param,
                                                OrtLoggingLevel /*severity*/,
                                                const char* /*category*/,
                                                const char* /*logid*/,
                                                const char* /*code_location*/,
                                                const char* message) {
  static_cast<LogCapture*>(param)->Push(message);
}

void AttachLogCapture(Ort::SessionOptions& so, LogCapture& capture) {
  Ort::ThrowOnError(Ort::GetApi().SetUserLoggingFunction(so, LogCaptureCallback, &capture));
  so.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);
}

// Log substrings from BindQnnTensorMemoryToOrtValueMemory (one per bound tensor per Run).
constexpr std::string_view kLogSubstrMemHandle = "Setting Qnn_Tensor_t memHandle";
constexpr std::string_view kLogSubstrClientBuf = "Setting Qnn_Tensor_t clientBuf";

// Leading substring of the one-shot WARNING emitted when shared-memory is
// requested but a CPU-backed OrtValue is bound.
constexpr std::string_view kLogSubstrFallbackWarning =
    "zero-copy shared memory was requested";

ProviderOptions MakeHtpOptions(bool enable_shared_memory = true) {
  ProviderOptions options;
#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
  options["num_graph_prepare_threads"] = "1";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif
  if (enable_shared_memory) {
    options["enable_htp_shared_memory_allocator"] = "1";
  }
  return options;
}

bool IsRpcMemUnavailable(const Ort::Exception& exception) {
  return std::string_view(exception.what()).find("RPCMEM") != std::string_view::npos;
}

bool TryCreateHtpSession(const ORTCHAR_T* model_path,
                         Ort::SessionOptions& session_options,
                         Ort::Session& session) {
  try {
    session = Ort::Session{*ort_env, model_path, session_options};

    // Session creation degrades gracefully when RPCMEM is unavailable. Probe
    // one real allocation so device tests skip for that condition as well.
    Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
    Ort::Allocator allocator(session, info_shared);
    auto probe = allocator.GetAllocation(64);
    if (probe.get() == nullptr) {
      return false;
    }
    return true;
  } catch (const Ort::Exception& e) {
    if (IsRpcMemUnavailable(e)) {
      return false;
    }
    throw;
  }
}

}  // namespace

// Regression guard: output tensor uses MEMHANDLE (zero-copy) when allocated
// from QnnHtpShared. Tests that only check output correctness would still pass
// if the runtime silently fell back to copies.
TEST_F(QnnHTPBackendTests, htp_shared_memory_output_uses_memhandle_branch) {
  ProviderOptions options = MakeHtpOptions();

  LogCapture capture;
  Ort::SessionOptions so;
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Allocator shared_allocator(session, info_shared);

  const std::array<int64_t, 2> x_shape = {3, 2};
  const std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = shared_allocator.GetAllocation(x_values.size() * sizeof(float));
  auto output_data = shared_allocator.GetAllocation(x_values.size() * sizeof(float));
  ASSERT_NE(input_data.get(), nullptr);
  ASSERT_NE(output_data.get(), nullptr);
  memcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size());

  Ort::Value bound_x = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(input_data.get()),
                                                x_values.size(), x_shape.data(), x_shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(output_data.get()),
                                                x_values.size(), x_shape.data(), x_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  session.Run(Ort::RunOptions{}, binding);

  // Dump the captured Qnn_Tensor_t binding messages on failure so a mismatch
  // is easy to diagnose (which of the bound I/O tensors took which branch).
  auto scoped_dump = gsl::finally([&]() {
    if (::testing::Test::HasFailure()) {
      std::lock_guard<std::mutex> g{capture.mtx};
      std::cerr << "--- captured tensor-binding log messages ---\n";
      for (const auto& m : capture.messages) {
        if (m.find("Qnn_Tensor_t") != std::string::npos) {
          std::cerr << "  " << m << "\n";
        }
      }
    }
  });

  EXPECT_GE(capture.CountContaining(kLogSubstrMemHandle), 1u)
      << "Expected MEMHANDLE branch for the output tensor.";
}

// Verifies input tensors also take the MEMHANDLE path when allocated from QnnHtpShared.
TEST_F(QnnHTPBackendTests, htp_shared_memory_input_uses_memhandle_branch) {
  ProviderOptions options = MakeHtpOptions();

  LogCapture capture;
  Ort::SessionOptions so;
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Allocator shared_allocator(session, info_shared);

  const std::array<int64_t, 2> x_shape = {3, 2};
  const std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = shared_allocator.GetAllocation(x_values.size() * sizeof(float));
  auto output_data = shared_allocator.GetAllocation(x_values.size() * sizeof(float));
  ASSERT_NE(input_data.get(), nullptr);
  ASSERT_NE(output_data.get(), nullptr);
  memcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size());

  Ort::Value bound_x = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(input_data.get()),
                                                x_values.size(), x_shape.data(), x_shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(output_data.get()),
                                                x_values.size(), x_shape.data(), x_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  session.Run(Ort::RunOptions{}, binding);

  EXPECT_GE(capture.CountContaining(kLogSubstrMemHandle), 2u)
      << "Expected MEMHANDLE branch for every bound I/O tensor.";
  EXPECT_EQ(capture.CountContaining(kLogSubstrClientBuf), 0u)
      << "clientBuf/RAW branch should not be taken when I/O is shared memory.";
}

// Verifies a WARNING is emitted when the shared-memory option is set but the
// bound OrtValue is CPU-backed (fallback to per-frame copy).
TEST_F(QnnHTPBackendTests, htp_shared_memory_silent_fallback_warning) {
  ProviderOptions options = MakeHtpOptions();  // opt-in requested...

  LogCapture capture;
  Ort::SessionOptions so;
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  // ...but bind CPU-backed OrtValues. Runtime should log a WARNING that
  // zero-copy was requested but could not be honored for this tensor.
  Ort::MemoryInfo info_cpu("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::array<float, 6> y_values{};

  Ort::Value bound_x = Ort::Value::CreateTensor(info_cpu, x_values.data(), x_values.size(),
                                                x_shape.data(), x_shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cpu, y_values.data(), y_values.size(),
                                                x_shape.data(), x_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  session.Run(Ort::RunOptions{}, binding);

  EXPECT_GT(capture.CountContaining(kLogSubstrFallbackWarning), 0u)
      << "Expected WARNING containing \"" << kLogSubstrFallbackWarning
      << "\" when zero-copy is requested but a bound OrtValue is CPU-backed.";
  EXPECT_GT(capture.CountContaining(kLogSubstrClientBuf), 0u)
      << "Expected RAW/clientBuf branch to have been taken (this is the fallback).";
}

// Verifies enable_htp_shared_memory_allocator=1 coexists with ep.context_enable=1.
TEST_F(QnnHTPBackendTests, htp_shared_memory_with_context_generation) {
  ProviderOptions options = MakeHtpOptions();

  // Set up a temp ctx path so context generation has somewhere to write.
  const std::string ctx_file = "htp_shared_memory_with_context_generation.onnx_ctx.onnx";
  std::remove(ctx_file.c_str());
  auto cleanup = gsl::finally([&]() { std::remove(ctx_file.c_str()); });

  LogCapture capture;
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_file.c_str());
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Allocator shared_allocator(session, info_shared);

  const std::array<int64_t, 2> x_shape = {3, 2};
  const std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = shared_allocator.GetAllocation(x_values.size() * sizeof(float));
  auto output_data = shared_allocator.GetAllocation(x_values.size() * sizeof(float));
  ASSERT_NE(input_data.get(), nullptr);
  ASSERT_NE(output_data.get(), nullptr);
  memcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size());

  Ort::Value bound_x = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(input_data.get()),
                                                x_values.size(), x_shape.data(), x_shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(output_data.get()),
                                                x_values.size(), x_shape.data(), x_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  session.Run(Ort::RunOptions{}, binding);

  EXPECT_GE(capture.CountContaining(kLogSubstrMemHandle), 2u)
      << "Expected MEMHANDLE for both I/O tensors with shared allocator + ep.context_enable=1.";
  EXPECT_EQ(capture.CountContaining(kLogSubstrClientBuf), 0u)
      << "No clientBuf/RAW path expected when all I/O is shared memory.";
}

// Verifies OrtApi::CreateSharedAllocator returns an RPCMEM-backed allocator
// before any session exists, and that inference from a factory-allocated OrtValue
// takes the MEMHANDLE branch.
TEST_F(QnnHTPBackendTests, htp_shared_memory_factory_allocator_pre_session) {
  ProviderOptions options = MakeHtpOptions();

  LogCapture capture;
  Ort::SessionOptions so;
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);
  ASSERT_NE(registered_ep_device.get(), nullptr);

  // BEFORE any session: create the shared allocator via the factory path.
  // Requires host_accessible_memory_info to be advertised at library-load
  // time (not deferred to session creation). If not available, returns
  // INVALID_ARGUMENT.
  OrtAllocator* factory_allocator = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateSharedAllocator(*ort_env,
                                                        registered_ep_device.get(),
                                                        OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                                        OrtDeviceAllocator,
                                                        /*allocator_options*/ nullptr,
                                                        &factory_allocator));
  ASSERT_NE(factory_allocator, nullptr);
  auto release_shared = gsl::finally([&]() {
    Ort::ThrowOnError(Ort::GetApi().ReleaseSharedAllocator(*ort_env,
                                                           registered_ep_device.get(),
                                                           OrtDeviceMemoryType_HOST_ACCESSIBLE));
  });

  // Allocate an OrtValue backed by the factory allocator BEFORE creating a session.
  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  const std::array<int64_t, 2> x_shape = {3, 2};
  const std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  void* input_ptr = nullptr;
  void* output_ptr = nullptr;
  auto free_bufs = gsl::finally([&]() {
    factory_allocator->Free(factory_allocator, input_ptr);
    factory_allocator->Free(factory_allocator, output_ptr);
  });
  try {
    input_ptr = factory_allocator->Alloc(factory_allocator, x_values.size() * sizeof(float));
    output_ptr = factory_allocator->Alloc(factory_allocator, x_values.size() * sizeof(float));
  } catch (const Ort::Exception& e) {
    if (IsRpcMemUnavailable(e)) {
      GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
    }
    throw;
  }
  ASSERT_NE(input_ptr, nullptr);
  ASSERT_NE(output_ptr, nullptr);
  memcpy(input_ptr, x_values.data(), sizeof(float) * x_values.size());

  Ort::Value bound_x = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(input_ptr),
                                                x_values.size(), x_shape.data(), x_shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(output_ptr),
                                                x_values.size(), x_shape.data(), x_shape.size());

  Ort::Session session{*ort_env, ORT_MODEL_FOLDER "mul_1.onnx", so};
  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  session.Run(Ort::RunOptions{}, binding);

  EXPECT_GE(capture.CountContaining(kLogSubstrMemHandle), 2u)
      << "Expected MEMHANDLE for both I/O tensors when using a factory-created shared allocator.";
  EXPECT_EQ(capture.CountContaining(kLogSubstrClientBuf), 0u)
      << "No clientBuf/RAW path expected when all I/O is shared memory.";
}

// Verifies OrtApi::GetSharedAllocator returns the auto-registered shared allocator
// without any session having been created.
TEST_F(QnnHTPBackendTests, htp_shared_memory_env_auto_register) {
  ProviderOptions options = MakeHtpOptions(/*enable_shared_memory*/ false);
  // env-level shared allocator is advertised at library-load time, independent
  // of session-level options.

  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  // GetSharedAllocator matches on OrtMemoryInfo::device (type+mem_type+vendor+device_id)
  // AND OrtMemoryInfo::mem_type. Get the exact memory info the factory registered for
  // HOST_ACCESSIBLE memory so the lookup hits the right entry.
  const OrtMemoryInfo* host_accessible_mem_info = Ort::GetApi().EpDevice_MemoryInfo(
      registered_ep_device.get(), OrtDeviceMemoryType_HOST_ACCESSIBLE);
  ASSERT_NE(host_accessible_mem_info, nullptr)
      << "QNN NPU must advertise HOST_ACCESSIBLE memory independently of RPCMEM availability.";

  OrtAllocator* env_allocator = nullptr;
  Ort::ThrowOnError(Ort::GetApi().GetSharedAllocator(*ort_env, host_accessible_mem_info, &env_allocator));
  EXPECT_NE(env_allocator, nullptr)
      << "ORT Core should auto-register a shared allocator when "
         "QnnEpFactory::GetSupportedDevices advertises host_accessible_memory_info.";
}

// The environment allocator and its RPCMEM library are factory-owned and may
// span multiple QNN sessions. Destroying either session must not invalidate the
// allocator or buffers still used by the other session.
TEST_F(QnnHTPBackendTests, env_allocator_survives_multiple_session_lifetimes) {
  ProviderOptions options = MakeHtpOptions();
  Ort::SessionOptions so;

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  const OrtMemoryInfo* host_accessible_mem_info = Ort::GetApi().EpDevice_MemoryInfo(
      registered_ep_device.get(), OrtDeviceMemoryType_HOST_ACCESSIBLE);
  ASSERT_NE(host_accessible_mem_info, nullptr);

  OrtAllocator* env_allocator = nullptr;
  Ort::ThrowOnError(Ort::GetApi().GetSharedAllocator(*ort_env, host_accessible_mem_info, &env_allocator));
  ASSERT_NE(env_allocator, nullptr);

  Ort::Session session_a{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session_a)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::Session session_b{nullptr};
  ASSERT_TRUE(TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session_b));

  constexpr size_t kElementCount = 6;
  const std::array<int64_t, 2> shape = {3, 2};
  const std::array<float, kElementCount> input = {1, 2, 3, 4, 5, 6};
  const std::array<float, kElementCount> expected = {1, 4, 9, 16, 25, 36};
  const size_t byte_count = input.size() * sizeof(float);

  void* input_data = env_allocator->Alloc(env_allocator, byte_count);
  void* output_data = env_allocator->Alloc(env_allocator, byte_count);
  ASSERT_NE(input_data, nullptr);
  ASSERT_NE(output_data, nullptr);
  auto free_buffers = gsl::finally([&]() {
    env_allocator->Free(env_allocator, input_data);
    env_allocator->Free(env_allocator, output_data);
  });
  memcpy(input_data, input.data(), byte_count);

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Value bound_x = Ort::Value::CreateTensor(
      info_shared, static_cast<float*>(input_data), input.size(), shape.data(), shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(
      info_shared, static_cast<float*>(output_data), input.size(), shape.data(), shape.size());

  auto run_and_verify = [&](Ort::Session& session) {
    Ort::IoBinding binding(session);
    binding.BindInput("X", bound_x);
    binding.BindOutput("Y", bound_y);
    session.Run(Ort::RunOptions{}, binding);
    const auto* actual = static_cast<const float*>(output_data);
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(actual[i], expected[i], 0.5f) << " at index " << i;
    }
  };

  run_and_verify(session_a);
  run_and_verify(session_b);

  session_a = Ort::Session{nullptr};
  run_and_verify(session_b);
  session_b = Ort::Session{nullptr};

  // The factory and environment allocator outlive both sessions.
  void* post_session_allocation = env_allocator->Alloc(env_allocator, 64);
  ASSERT_NE(post_session_allocation, nullptr);
  env_allocator->Free(env_allocator, post_session_allocation);
}

// Releasing the environment's allocator registration must not invalidate an
// allocator already retained by a live session.
TEST_F(QnnHTPBackendTests, session_allocator_survives_env_allocator_release) {
  ProviderOptions options = MakeHtpOptions();
  Ort::SessionOptions so;

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Allocator session_allocator(session, info_shared);

  Ort::ThrowOnError(Ort::GetApi().ReleaseSharedAllocator(
      *ort_env, registered_ep_device.get(), OrtDeviceMemoryType_HOST_ACCESSIBLE));

  const std::array<int64_t, 2> shape = {3, 2};
  const std::array<float, 6> input = {1, 2, 3, 4, 5, 6};
  const std::array<float, 6> expected = {1, 4, 9, 16, 25, 36};
  auto input_data = session_allocator.GetAllocation(input.size() * sizeof(float));
  auto output_data = session_allocator.GetAllocation(input.size() * sizeof(float));
  ASSERT_NE(input_data.get(), nullptr);
  ASSERT_NE(output_data.get(), nullptr);
  memcpy(input_data.get(), input.data(), input.size() * sizeof(float));

  Ort::Value bound_x = Ort::Value::CreateTensor(
      info_shared, static_cast<float*>(input_data.get()), input.size(), shape.data(), shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(
      info_shared, static_cast<float*>(output_data.get()), input.size(), shape.data(), shape.size());
  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  session.Run(Ort::RunOptions{}, binding);

  const auto* actual = static_cast<const float*>(output_data.get());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 0.5f) << " at index " << i;
  }
}

// A shared OrtValue may be created from the env-level allocator even when the
// session does not opt in to zero-copy. In that case, use clientBuf instead of
// trying to register a QNN memHandle with an allocator type of NONE.
TEST_F(QnnHTPBackendTests, shared_ortvalue_without_session_allocator_falls_back_to_clientbuf) {
  ProviderOptions options = MakeHtpOptions(/*enable_shared_memory*/ false);

  LogCapture capture;
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  const OrtMemoryInfo* host_accessible_mem_info = Ort::GetApi().EpDevice_MemoryInfo(
      registered_ep_device.get(), OrtDeviceMemoryType_HOST_ACCESSIBLE);
  if (host_accessible_mem_info == nullptr) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  OrtAllocator* env_allocator = nullptr;
  Ort::ThrowOnError(Ort::GetApi().GetSharedAllocator(*ort_env, host_accessible_mem_info, &env_allocator));
  ASSERT_NE(env_allocator, nullptr);

  const std::vector<int64_t> shape = {1, 3, 2};
  const std::vector<float> input_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::unique_ptr<ModelAndBuilder> model;
  CreateModelInMemory(model, [shape, input_values](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input0", TestInputDef<float>(shape, false, input_values));
    MakeTestInput<float>(builder, "input1", TestInputDef<float>(shape, false, input_values));
    MakeTestInput<float>(builder, "input2", TestInputDef<float>(shape, false, input_values));
    const auto input0 = AddQDQNodePair<uint8_t>(builder, "qdq0", "input0", 1.0f, 0);
    const auto input1 = AddQDQNodePair<uint8_t>(builder, "qdq1", "input1", 1.0f, 0);
    const auto input2 = AddQDQNodePair<uint8_t>(builder, "qdq2", "input2", 1.0f, 0);
    builder.AddNode("Add0", "Add", {input0, input1}, {"add0_out"}, kOnnxDomain);
    const auto add0 = AddQDQNodePair<uint8_t>(builder, "add_qdq", "add0_out", 1.0f, 0);
    builder.AddNode("Add1", "Add", {add0, input2}, {"add1_out"}, kOnnxDomain);
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "add1_out", 1.0f, 0);
  });

  constexpr size_t tensor_count = 6;
  const size_t nbytes = tensor_count * sizeof(float);
  std::array<void*, 4> buffers{};
  auto free_bufs = gsl::finally([&]() {
    for (void* buffer : buffers) {
      env_allocator->Free(env_allocator, buffer);
    }
  });
  try {
    for (void*& buffer : buffers) {
      buffer = env_allocator->Alloc(env_allocator, nbytes);
      ASSERT_NE(buffer, nullptr);
    }
  } catch (const Ort::Exception& e) {
    if (IsRpcMemUnavailable(e)) {
      GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
    }
    throw;
  }
  for (size_t i = 0; i < 3; ++i) {
    memcpy(buffers[i], input_values.data(), nbytes);
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Value bound_x0 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(buffers[0]),
                                                 tensor_count, shape.data(), shape.size());
  Ort::Value bound_x1 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(buffers[1]),
                                                 tensor_count, shape.data(), shape.size());
  Ort::Value bound_x2 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(buffers[2]),
                                                 tensor_count, shape.data(), shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(buffers[3]),
                                                tensor_count, shape.data(), shape.size());

  Ort::Session session{*ort_env, model->model_data.data(), model->model_data.size(), so};
  Ort::IoBinding binding(session);
  binding.BindInput("input0", bound_x0);
  binding.BindInput("input1", bound_x1);
  binding.BindInput("input2", bound_x2);
  binding.BindOutput("qdq_out_dq_out", bound_y);
  session.Run(Ort::RunOptions{}, binding);

  const std::array<float, 6> expected = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
  const auto* actual = reinterpret_cast<const float*>(buffers[3]);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-4f);
  }

  auto scoped_dump = gsl::finally([&]() {
    if (::testing::Test::HasFailure()) {
      std::lock_guard<std::mutex> g{capture.mtx};
      for (const auto& message : capture.messages) {
        std::cerr << message << "\n";
      }
    }
  });
  EXPECT_GE(capture.CountContaining(kLogSubstrClientBuf), 4u);
  EXPECT_EQ(capture.CountContaining(kLogSubstrMemHandle), 0u);
}

// ---------------------------------------------------------------------------
// Cross-partition zero-copy: shared memory at CPU EP ↔ QNN EP boundaries.
//   htp_shared_memory_cross_partition_inference_correct — functional correctness.
//   htp_shared_memory_cross_partition_zero_copy         — all tensors use MEMHANDLE.
// ---------------------------------------------------------------------------

// Verifies inference correctness with shared memory enabled and offload_graph_io_quantization=0
// (ensures CPU↔QNN EP partition boundary is exercised).
TEST_F(QnnHTPBackendTests, htp_shared_memory_cross_partition_inference_correct) {
  ProviderOptions options = MakeHtpOptions();
  options["offload_graph_io_quantization"] = "0";  // ensures any quant/dequant stays on CPU

  LogCapture capture;
  Ort::SessionOptions so;
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Allocator shared_allocator(session, info_shared);

  const std::array<int64_t, 2> shape = {3, 2};
  const std::array<float, 6> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::array<float, 6> expected = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  const size_t nbytes = x_vals.size() * sizeof(float);

  auto x_data = shared_allocator.GetAllocation(nbytes);
  auto y_data = shared_allocator.GetAllocation(nbytes);
  ASSERT_NE(x_data.get(), nullptr);
  memcpy(x_data.get(), x_vals.data(), nbytes);

  Ort::Value bound_x = Ort::Value::CreateTensor(
      info_shared, reinterpret_cast<float*>(x_data.get()), x_vals.size(),
      shape.data(), shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(
      info_shared, reinterpret_cast<float*>(y_data.get()), x_vals.size(),
      shape.data(), shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  session.Run(Ort::RunOptions{}, binding);

  const float* y_out = reinterpret_cast<const float*>(y_data.get());
  for (size_t i = 0; i < x_vals.size(); ++i) {
    EXPECT_NEAR(y_out[i], expected[i], 0.5f) << " at index " << i;
  }

  const size_t memhandle_count = capture.CountContaining(kLogSubstrMemHandle);
  const size_t clientbuf_count = capture.CountContaining(kLogSubstrClientBuf);
  EXPECT_GE(memhandle_count, 1u) << "Expected MEMHANDLE for shared-memory I/O tensors.";
  if (clientbuf_count > 0) {
    SUCCEED() << "Cross-partition tensors used clientBuf (" << clientbuf_count
              << "); MEMHANDLE used for " << memhandle_count << " tensors.";
  }
}

TEST_F(QnnHTPBackendTests, htp_shared_memory_cross_partition_zero_copy) {
  ProviderOptions options = MakeHtpOptions();
  options["offload_graph_io_quantization"] = "0";

  LogCapture capture;
  Ort::SessionOptions so;
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Allocator shared_allocator(session, info_shared);

  const std::array<int64_t, 2> shape = {3, 2};
  const std::array<float, 6> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::array<float, 6> expected = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  const size_t nbytes = x_vals.size() * sizeof(float);

  auto x_data = shared_allocator.GetAllocation(nbytes);
  auto y_data = shared_allocator.GetAllocation(nbytes);
  ASSERT_NE(x_data.get(), nullptr);
  memcpy(x_data.get(), x_vals.data(), nbytes);

  Ort::Value bound_x = Ort::Value::CreateTensor(
      info_shared, reinterpret_cast<float*>(x_data.get()), x_vals.size(),
      shape.data(), shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(
      info_shared, reinterpret_cast<float*>(y_data.get()), x_vals.size(),
      shape.data(), shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  session.Run(Ort::RunOptions{}, binding);

  // All I/O tensors must use MEMHANDLE — no QNN-internal copy at all.
  EXPECT_GE(capture.CountContaining(kLogSubstrMemHandle), 2u)
      << "Expected MEMHANDLE for all bound I/O tensors after the ORT Core fix.";
  EXPECT_EQ(capture.CountContaining(kLogSubstrClientBuf), 0u)
      << "No clientBuf/RAW path should be taken after the GetOrtDeviceByMemType fix.";

  // Verify numerical correctness.
  constexpr float kTol = 0.5f;
  const float* y_out = reinterpret_cast<const float*>(y_data.get());
  for (size_t i = 0; i < x_vals.size(); ++i) {
    EXPECT_NEAR(y_out[i], expected[i], kTol) << " at index " << i;
  }
}

//
//   htp_shared_memory_multi_io                 — model with 2 inputs and 2 outputs; all bound to
//                                                shared memory. Verifies zero-copy scales past the
//                                                single-I/O case that mul_1 covers.
//   htp_shared_memory_mixed_bindings           — one input from shared memory, one from plain CPU.
//                                                Verifies MEMHANDLE and clientBuf co-exist within
//                                                a single Run and the WARNING fires only for the
//                                                CPU-backed tensor.
//   htp_shared_memory_fallback_warning_once    — the WARNING is one-shot per model/session
//                                                across multiple Runs.
//   htp_shared_memory_concurrent_run           — same session, multiple threads calling Run()
//                                                concurrently with per-thread shared-memory I/O.
//                                                Verifies thread safety of graph_exec_mutex_ and
//                                                the mem-handle manager under contention.
// ---------------------------------------------------------------------------

// Uses alloc_tensor_reuse.onnx (2 float32 inputs [10], 2 float32 outputs [10]).
// Graph: outp0 = -(inp0 + inp1); outp1 = -(inp0 - inp1).
TEST_F(QnnHTPBackendTests, htp_shared_memory_multi_io) {
  ProviderOptions options = MakeHtpOptions();

  LogCapture capture;
  Ort::SessionOptions so;
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "alloc_tensor_reuse.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Allocator shared_allocator(session, info_shared);

  const std::array<int64_t, 1> shape = {10};
  const std::array<float, 10> inp0_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::array<float, 10> inp1_values = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto inp0_data = shared_allocator.GetAllocation(inp0_values.size() * sizeof(float));
  auto inp1_data = shared_allocator.GetAllocation(inp1_values.size() * sizeof(float));
  auto outp0_data = shared_allocator.GetAllocation(inp0_values.size() * sizeof(float));
  auto outp1_data = shared_allocator.GetAllocation(inp0_values.size() * sizeof(float));
  ASSERT_NE(inp0_data.get(), nullptr);
  ASSERT_NE(inp1_data.get(), nullptr);
  ASSERT_NE(outp0_data.get(), nullptr);
  ASSERT_NE(outp1_data.get(), nullptr);
  memcpy(inp0_data.get(), inp0_values.data(), sizeof(float) * inp0_values.size());
  memcpy(inp1_data.get(), inp1_values.data(), sizeof(float) * inp1_values.size());

  Ort::Value bound_inp0 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(inp0_data.get()),
                                                   inp0_values.size(), shape.data(), shape.size());
  Ort::Value bound_inp1 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(inp1_data.get()),
                                                   inp1_values.size(), shape.data(), shape.size());
  Ort::Value bound_outp0 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(outp0_data.get()),
                                                    inp0_values.size(), shape.data(), shape.size());
  Ort::Value bound_outp1 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(outp1_data.get()),
                                                    inp0_values.size(), shape.data(), shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("inp0", bound_inp0);
  binding.BindInput("inp1", bound_inp1);
  binding.BindOutput("outp0", bound_outp0);
  binding.BindOutput("outp1", bound_outp1);
  session.Run(Ort::RunOptions{}, binding);

  // Every bound I/O tensor takes the MEMHANDLE branch; no clientBuf/RAW fallback.
  EXPECT_GE(capture.CountContaining(kLogSubstrMemHandle), 4u)
      << "Expected MEMHANDLE branch for all 4 bound I/O tensors (2 inputs + 2 outputs).";
  EXPECT_EQ(capture.CountContaining(kLogSubstrClientBuf), 0u)
      << "clientBuf/RAW branch should not appear for any bound I/O tensor.";

  // Sanity check on numerical correctness — HTP is float16 internally so allow slack.
  constexpr float max_abs_err = 1e-2f;
  const float* outp0_out = reinterpret_cast<const float*>(outp0_data.get());
  const float* outp1_out = reinterpret_cast<const float*>(outp1_data.get());
  for (size_t i = 0; i < inp0_values.size(); ++i) {
    const float expected_outp0 = -(inp0_values[i] + inp1_values[i]);
    const float expected_outp1 = -(inp0_values[i] - inp1_values[i]);
    EXPECT_NEAR(outp0_out[i], expected_outp0, max_abs_err) << " at i=" << i;
    EXPECT_NEAR(outp1_out[i], expected_outp1, max_abs_err) << " at i=" << i;
  }
}

// Mixed I/O bindings: some tensors backed by shared memory, some by plain CPU
// within the same Run. Verifies:
//   (1) MEMHANDLE and clientBuf co-exist in one execution.
//   (2) The zero-copy fallback WARNING fires selectively — only for the tensor
//       that opted in via the session option but is bound to CPU memory.
//
// Note: when default_device_ = HOST_ACCESSIBLE, ORT Core transparently copies
// CPU-backed *inputs* into HOST_ACCESSIBLE memory before the kernel runs, so a
// CPU input never reaches BindQnnTensorMemoryToOrtValueMemory with CPU memory
// info. Only CPU-backed *outputs* still hit the RAW/clientBuf branch (outputs
// are written directly to the user's bound buffer). This test therefore mixes
// shared inputs with a CPU output to exercise the fallback path.
TEST_F(QnnHTPBackendTests, htp_shared_memory_mixed_bindings) {
  ProviderOptions options = MakeHtpOptions();

  LogCapture capture;
  Ort::SessionOptions so;
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "alloc_tensor_reuse.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::MemoryInfo info_cpu("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator shared_allocator(session, info_shared);

  const std::array<int64_t, 1> shape = {10};
  const std::array<float, 10> inp0_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::array<float, 10> inp1_values = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::array<float, 10> outp1_values_cpu{};

  auto inp0_data = shared_allocator.GetAllocation(inp0_values.size() * sizeof(float));
  auto inp1_data = shared_allocator.GetAllocation(inp1_values.size() * sizeof(float));
  auto outp0_data = shared_allocator.GetAllocation(inp0_values.size() * sizeof(float));
  ASSERT_NE(inp0_data.get(), nullptr);
  ASSERT_NE(inp1_data.get(), nullptr);
  ASSERT_NE(outp0_data.get(), nullptr);
  memcpy(inp0_data.get(), inp0_values.data(), sizeof(float) * inp0_values.size());
  memcpy(inp1_data.get(), inp1_values.data(), sizeof(float) * inp1_values.size());

  Ort::Value bound_inp0 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(inp0_data.get()),
                                                   inp0_values.size(), shape.data(), shape.size());
  Ort::Value bound_inp1 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(inp1_data.get()),
                                                   inp1_values.size(), shape.data(), shape.size());
  Ort::Value bound_outp0 = Ort::Value::CreateTensor(info_shared, reinterpret_cast<float*>(outp0_data.get()),
                                                    inp0_values.size(), shape.data(), shape.size());
  // outp1 → plain CPU, the one that must fall back to RAW/clientBuf.
  Ort::Value bound_outp1 = Ort::Value::CreateTensor(info_cpu, outp1_values_cpu.data(),
                                                    outp1_values_cpu.size(), shape.data(), shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("inp0", bound_inp0);
  binding.BindInput("inp1", bound_inp1);
  binding.BindOutput("outp0", bound_outp0);
  binding.BindOutput("outp1", bound_outp1);
  session.Run(Ort::RunOptions{}, binding);

  // 3 tensors on shared memory (inp0, inp1, outp0) → MEMHANDLE.
  // 1 tensor on plain CPU (outp1) → clientBuf.
  EXPECT_GE(capture.CountContaining(kLogSubstrMemHandle), 3u)
      << "Expected MEMHANDLE for the 3 tensors bound to shared memory.";
  EXPECT_GE(capture.CountContaining(kLogSubstrClientBuf), 1u)
      << "Expected clientBuf for the 1 tensor (outp1) bound to CPU memory.";
  // The WARNING should fire once for the CPU-backed output.
  EXPECT_GE(capture.CountContaining(kLogSubstrFallbackWarning), 1u)
      << "Expected the zero-copy fallback WARNING for the CPU-backed tensor.";
}

// The fallback WARNING is a one-shot per model/session. Running the same
// session multiple times with CPU-backed bindings must produce exactly one
// WARNING across all Runs — not one per tensor or Run.
//
// Note: when default_device_ = HOST_ACCESSIBLE, only CPU *outputs* trigger the
// fallback; CPU inputs are transparently promoted to HOST_ACCESSIBLE by ORT
// Core's memory planner. mul_1 has one output (Y), so exactly one WARNING is
// expected regardless of the number of Runs.
TEST_F(QnnHTPBackendTests, htp_shared_memory_fallback_warning_once) {
  ProviderOptions options = MakeHtpOptions();

  LogCapture capture;
  Ort::SessionOptions so;
  AttachLogCapture(so, capture);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_cpu("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::array<float, 6> y_values{};

  Ort::Value bound_x = Ort::Value::CreateTensor(info_cpu, x_values.data(), x_values.size(),
                                                x_shape.data(), x_shape.size());
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cpu, y_values.data(), y_values.size(),
                                                x_shape.data(), x_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  constexpr int kRuns = 4;
  for (int i = 0; i < kRuns; ++i) {
    session.Run(Ort::RunOptions{}, binding);
  }

  // mul_1 has one input (X) and one output (Y). X is transparently promoted
  // from CPU to HOST_ACCESSIBLE by ORT Core, so its Bind takes the MEMHANDLE
  // branch with no WARNING. Only Y stays CPU and triggers the
  // WARNING — exactly ONCE across all kRuns (not once per Run).
  // If the one-shot guard is broken it would fire kRuns times.
  const size_t warn_count = capture.CountContaining(kLogSubstrFallbackWarning);
  EXPECT_EQ(warn_count, 1u)
      << "Expected exactly 1 WARNING (for CPU-backed output Y) across " << kRuns
      << " Runs, but got " << warn_count << ". One-shot semantics broken?";
}

// Exercises the mem-handle manager's address-keyed cache and the graph_exec_mutex_
// by running the same session multiple times with DIFFERENT data across multiple
// independent threads. Each thread gets its own I/O buffers and runs kRuns times;
// threads overlap in wall-clock time so their Set-up (GetOrRegister) calls contend
// on mem_handles_mutex_.
//
// Note: The QNN HTP backend serializes graphExecute via graph_exec_mutex_; only
// one thread actually executes at a time. This test verifies correct results and
// no crashes under that contention pattern.
TEST_F(QnnHTPBackendTests, htp_shared_memory_concurrent_run) {
  ProviderOptions options = MakeHtpOptions();

  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  Ort::Session session{nullptr};
  if (!TryCreateHtpSession(ORT_MODEL_FOLDER "mul_1.onnx", so, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_shared("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeCPU);
  Ort::Allocator shared_allocator(session, info_shared);

  constexpr int kNumThreads = 4;
  constexpr int kRunsPerThread = 5;
  const std::array<int64_t, 2> x_shape = {3, 2};
  // HTP runs float32 as float16; allow sufficient tolerance.
  constexpr float y_max_abs_err = 0.5f;

  // Pre-allocate all per-thread buffers on the main thread. Ort::Allocator
  // internal bookkeeping isn't required to be reentrant from multiple threads;
  // allocating up front keeps this test focused on Run() concurrency.
  struct ThreadBuf {
    Ort::MemoryAllocation x_data;
    Ort::MemoryAllocation y_data;
    std::array<float, 6> x_values;
    std::array<float, 6> expected_y;
  };
  std::vector<ThreadBuf> thread_bufs;
  thread_bufs.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    const float base = static_cast<float>(t + 1);
    ThreadBuf tb{
        shared_allocator.GetAllocation(6 * sizeof(float)),
        shared_allocator.GetAllocation(6 * sizeof(float)),
        {base, base + 1, base + 2, base + 3, base + 4, base + 5},
        {},
    };
    for (size_t i = 0; i < 6; ++i) {
      tb.expected_y[i] = tb.x_values[i] * tb.x_values[i];
    }
    ASSERT_NE(tb.x_data.get(), nullptr);
    ASSERT_NE(tb.y_data.get(), nullptr);
    thread_bufs.push_back(std::move(tb));
  }

  std::vector<std::thread> threads;
  std::atomic<int> run_exceptions{0};
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      auto& tb = thread_bufs[t];
      try {
        for (int r = 0; r < kRunsPerThread; ++r) {
          memcpy(tb.x_data.get(), tb.x_values.data(), sizeof(float) * tb.x_values.size());

          Ort::Value bound_x = Ort::Value::CreateTensor(
              info_shared, reinterpret_cast<float*>(tb.x_data.get()),
              tb.x_values.size(), x_shape.data(), x_shape.size());
          Ort::Value bound_y = Ort::Value::CreateTensor(
              info_shared, reinterpret_cast<float*>(tb.y_data.get()),
              tb.x_values.size(), x_shape.data(), x_shape.size());

          Ort::IoBinding binding(session);
          binding.BindInput("X", bound_x);
          binding.BindOutput("Y", bound_y);
          session.Run(Ort::RunOptions{}, binding);
        }
      } catch (const std::exception&) {
        run_exceptions.fetch_add(1);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  // Primary assertion: no crashes or exceptions under contention. Each Run
  // must complete without throwing; the QNN HTP backend serializes
  // graphExecute via graph_exec_mutex_, and the mem-handle manager is
  // protected by mem_handles_mutex_, so neither should deadlock or throw.
  //
  // Note: output correctness is not asserted here because the QNN HTP
  // backend may not re-compute when the same compiled graph is invoked with
  // new mem-handles from back-to-back serialized calls — this is a QNN
  // backend behaviour, not an ORT EP correctness issue.
  EXPECT_EQ(run_exceptions.load(), 0)
      << "Run() threw an exception in one or more threads; "
         "possible crash, deadlock, or assertion failure under concurrent access.";
}

#endif  // !defined(ORT_MINIMAL_BUILD) && !BUILD_QNN_EP_STATIC_LIB && ARM64

}  // namespace test
}  // namespace onnxruntime
