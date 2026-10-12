// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for EpContextIoDispatch (ep_context_io_dispatch.cc).
//
// Covers the effective_api_version guard added to fix an access violation that
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
constexpr uintptr_t kFakeSessionOptionsToken = 0x1;

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

// effective_api_version below EpContextIoDispatch::kMinApiVersion must short-circuit
// before ever touching the experimental API, even with a non-null session_options.
TEST(EpContextIoDispatchTest, EffectiveApiVersionBelowMin_NeverTouchesExperimentalApi) {
  ExperimentalFnStubEnv env;
  OrtGlobalApiOverride api_override(&env.stub_ort_api);

  qnn::EpContextIoDispatch dispatch(
      reinterpret_cast<const OrtSessionOptions*>(kFakeSessionOptionsToken),
      qnn::EpContextIoDispatch::kMinApiVersion - 1);

  EXPECT_EQ(env.get_experimental_function_calls, 0);
  EXPECT_FALSE(dispatch.HasReadCallback());
  EXPECT_FALSE(dispatch.HasWriteCallback());
}

// A null session_options always short-circuits regardless of effective_api_version
// (no application session to resolve callbacks from).
TEST(EpContextIoDispatchTest, NullSessionOptions_NeverTouchesExperimentalApi) {
  ExperimentalFnStubEnv env;
  OrtGlobalApiOverride api_override(&env.stub_ort_api);

  qnn::EpContextIoDispatch dispatch(nullptr, qnn::EpContextIoDispatch::kMinApiVersion);

  EXPECT_EQ(env.get_experimental_function_calls, 0);
  EXPECT_FALSE(dispatch.HasReadCallback());
  EXPECT_FALSE(dispatch.HasWriteCallback());
}

// effective_api_version at/above kMinApiVersion proceeds to resolve callbacks
// (and, with no experimental API registered by this stub, fails via the
// existing try/catch rather than the version guard).
TEST(EpContextIoDispatchTest, EffectiveApiVersionAtMin_AttemptsExperimentalApi) {
  ExperimentalFnStubEnv env;
  OrtGlobalApiOverride api_override(&env.stub_ort_api);

  qnn::EpContextIoDispatch dispatch(
      reinterpret_cast<const OrtSessionOptions*>(kFakeSessionOptionsToken),
      qnn::EpContextIoDispatch::kMinApiVersion);

  EXPECT_GT(env.get_experimental_function_calls, 0);
  EXPECT_FALSE(dispatch.HasReadCallback());
  EXPECT_FALSE(dispatch.HasWriteCallback());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
