// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for EpContextIoDispatch (ep_context_io_dispatch.cc).
//
// Covers the runtime-capability guard added to fix an access violation that
// occurred when a QNN EP built against ORT 1.29+ headers was loaded by an older
// (<1.28) ORT runtime: the experimental EPContext encryption APIs don't exist in
// that runtime's OrtApi struct, so touching Ort::Experimental::EpContextConfig
// reads past the end of the struct instead of cleanly throwing.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "core/providers/qnn/builder/ep_context_io_dispatch.h"
#include "test/providers/qnn/infra/qnn_fake_ort_graph.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"

namespace onnxruntime {
namespace test {

namespace {
#if ORT_API_HAS_EPCONTEXT_ENCRYPTION
struct ReadBufferState {
  void* returned = nullptr;
  std::string name;
  bool fail_after_free = false;
  bool return_null = false;
  bool return_empty = false;

  static OrtStatus* ORT_API_CALL Read(void* opaque, const char* name, OrtAllocator* allocator,
                                      void** data, size_t* size) {
    auto& state = *static_cast<ReadBufferState*>(opaque);
    state.name = name;
    if (state.return_null) {
      *data = nullptr;
      *size = 32;
      return nullptr;
    }
    *data = allocator->Alloc(allocator, 32);
    *size = state.return_empty ? 0 : 32;
    state.returned = *data;
    if (!*data) return Ort::GetApi().CreateStatus(ORT_FAIL, "Allocation failed");
    if (state.fail_after_free) {
      allocator->Free(allocator, *data);
      return Ort::GetApi().CreateStatus(ORT_FAIL, "Callback freed its failed output");
    }
    static_cast<char*>(*data)[0] = 42;
    return nullptr;
  }
};
#endif

// Minimal OrtApi stub: GetExperimentalFunction always reports "unavailable",
// standing in for a runtime that doesn't have the v28 experimental vtable.
// If EpContextIoDispatch's effective_api_version guard is skipped and it
// calls into Ort::Experimental::EpContextConfig anyway, this stub throws
// ORT_NOT_IMPLEMENTED (via GetExperimentalFunction returning nullptr) rather
// than crashing, so the guard itself is exercised through the call counter
// below, not through a real ABI mismatch.
struct ExperimentalFnStubEnv {
  OrtApi stub_ort_api{};
  int get_experimental_function_calls = 0;

  ExperimentalFnStubEnv() {
    static thread_local ExperimentalFnStubEnv* current = nullptr;
    current = this;
    stub_ort_api.GetExperimentalFunction = [](const char*) noexcept -> OrtExperimentalFnPtr {
      if (current != nullptr) ++current->get_experimental_function_calls;
      return nullptr;
    };
  }
};
}  // namespace

// A runtime without the named EPContext-encryption capability must short-circuit
// before ever touching the experimental API, even with a non-null session_options.
TEST(QnnUnit_EpContextIoDispatchTest, EpContextEncryptionUnsupported_NeverTouchesExperimentalApi) {
  OrtSessionOptions* session_options = nullptr;
  ASSERT_EQ(Ort::GetApi().CreateSessionOptions(&session_options), nullptr);
  ASSERT_NE(session_options, nullptr);

  {
    ExperimentalFnStubEnv env;
    OrtGlobalApiOverride api_override(&env.stub_ort_api);
    const OrtApiRuntimeCapabilities runtime_capabilities{
        OrtApiRuntimeCapabilities::kEpContextEncryptionMinApiVersion - 1};

    qnn::EpContextIoDispatch dispatch(*session_options, runtime_capabilities);

    EXPECT_EQ(env.get_experimental_function_calls, 0);
    EXPECT_FALSE(dispatch.HasReadCallback());
    EXPECT_FALSE(dispatch.HasWriteCallback());
  }

  Ort::GetApi().ReleaseSessionOptions(session_options);
}

// The default constructor represents a disabled/no-session dispatch. It must not require a
// fabricated runtime version, or callers could accidentally depend on the session constructor's
// short-circuit order.
TEST(QnnUnit_EpContextIoDispatchTest, DefaultConstructor_IsDisabled) {
  qnn::EpContextIoDispatch dispatch;

  EXPECT_FALSE(dispatch.HasReadCallback());
  EXPECT_FALSE(dispatch.HasWriteCallback());
}

TEST(QnnUnit_EpContextIoDispatchTest, DisabledDispatch_ReadAndWrite_ReturnFailures) {
  qnn::EpContextIoDispatch dispatch;
  qnn::EpContextBuffer bytes;
  Ort::Logger logger;

  EXPECT_FALSE(dispatch.Read("context.bin", bytes).IsOK());
  EXPECT_FALSE(dispatch.Write("context.bin", nullptr, 0, logger).IsOK());
}

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION
TEST(QnnUnit_EpContextIoDispatchTest, InvalidSuccessfulCallbackPreservesPreviousOutput) {
  auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&Ort::GetApi());
  if (!setter) {
    GTEST_SKIP() << "Runtime does not support EPContext read callbacks";
  }
  Ort::SessionOptions options;
  ReadBufferState state;
  Ort::ThrowOnError(setter(options, ReadBufferState::Read, &state));
  qnn::EpContextIoDispatch dispatch(*options, OrtApiRuntimeCapabilities{ORT_API_VERSION});
  qnn::EpContextBuffer bytes;
  ASSERT_TRUE(dispatch.Read("valid.bin", bytes).IsOK());
  auto* previous = bytes.data();
  const auto previous_size = bytes.size();
  state.return_null = true;
  EXPECT_FALSE(dispatch.Read("null.bin", bytes).IsOK());
  EXPECT_EQ(bytes.data(), previous);
  EXPECT_EQ(bytes.size(), previous_size);
  state.return_null = false;
  state.return_empty = true;
  EXPECT_FALSE(dispatch.Read("empty.bin", bytes).IsOK());
  EXPECT_EQ(bytes.data(), previous);
  EXPECT_EQ(bytes.size(), previous_size);
  EXPECT_EQ(bytes.data()[0], 42);
}

TEST(QnnUnit_EpContextIoDispatchTest, AdoptsCallbackAllocationAndPreservesOutputOnError) {
  auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&Ort::GetApi());
  if (!setter) {
    GTEST_SKIP() << "Runtime does not support EPContext read callbacks";
  }
  Ort::SessionOptions options;
  ReadBufferState state;
  Ort::ThrowOnError(setter(options, ReadBufferState::Read, &state));
  qnn::EpContextIoDispatch dispatch(*options, OrtApiRuntimeCapabilities{ORT_API_VERSION});
  qnn::EpContextBuffer bytes;
  ASSERT_TRUE(dispatch.Read("requested.bin", bytes).IsOK());
  EXPECT_EQ(state.name, "requested.bin");
  EXPECT_EQ(bytes.data(), state.returned);
  EXPECT_EQ(bytes.size(), 32u);
  EXPECT_EQ(bytes.data()[0], 42);
  auto* original = bytes.data();
  state.fail_after_free = true;
  EXPECT_FALSE(dispatch.Read("failed.bin", bytes).IsOK());
  EXPECT_EQ(bytes.data(), original);
  EXPECT_EQ(bytes.size(), 32u);
  EXPECT_EQ(bytes.data()[0], 42);
  bytes = {};
}
#endif

TEST(QnnUnit_EpContextIoDispatchTest, MovedBufferRetainsDataAndEmptiesSource) {
  qnn::EpContextBuffer source;
  source.AllocateUninitialized(32);
  source.data()[0] = 42;
  auto* original = source.data();
  auto destination = std::move(source);
  EXPECT_EQ(source.data(), nullptr);
  EXPECT_EQ(source.size(), 0u);
  EXPECT_EQ(destination.data(), original);
  EXPECT_EQ(destination.size(), 32u);
  EXPECT_EQ(destination.data()[0], 42);
  source = std::move(destination);
  EXPECT_EQ(source.data(), original);
  EXPECT_EQ(source.size(), 32u);
  EXPECT_EQ(destination.data(), nullptr);
  EXPECT_EQ(destination.size(), 0u);
}

TEST(QnnUnit_EpContextIoDispatchTest, MovedBufferOutlivesDefaultAllocatorWrapper) {
  qnn::EpContextBuffer retained;
  char* original = nullptr;
  {
    Ort::AllocatorWithDefaultOptions allocator;
    auto* data = static_cast<char*>(allocator.Alloc(32));
    ASSERT_NE(data, nullptr);
    data[0] = 42;
    qnn::EpContextBuffer bytes;
    bytes.Adopt(data, 32);
    retained = std::move(bytes);
    original = data;
    EXPECT_EQ(retained.data(), data);
  }
  EXPECT_EQ(retained.data(), original);
  EXPECT_EQ(retained.size(), 32u);
  EXPECT_EQ(retained.data()[0], 42);
  retained = {};
  EXPECT_EQ(retained.data(), nullptr);
  EXPECT_EQ(retained.size(), 0u);
}

// The named EPContext-encryption capability proceeds to resolve callbacks
// (and, with no experimental API registered by this stub, fails via the
// existing try/catch rather than the version guard).
TEST(QnnUnit_EpContextIoDispatchTest, EpContextEncryptionSupported_AttemptsExperimentalApi) {
  OrtSessionOptions* session_options = nullptr;
  ASSERT_EQ(Ort::GetApi().CreateSessionOptions(&session_options), nullptr);
  ASSERT_NE(session_options, nullptr);

  {
    ExperimentalFnStubEnv env;
    OrtGlobalApiOverride api_override(&env.stub_ort_api);
    const OrtApiRuntimeCapabilities runtime_capabilities{
        OrtApiRuntimeCapabilities::kEpContextEncryptionMinApiVersion};

    qnn::EpContextIoDispatch dispatch(*session_options, runtime_capabilities);

    EXPECT_GT(env.get_experimental_function_calls, 0);
    EXPECT_FALSE(dispatch.HasReadCallback());
    EXPECT_FALSE(dispatch.HasWriteCallback());
  }

  Ort::GetApi().ReleaseSessionOptions(session_options);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
