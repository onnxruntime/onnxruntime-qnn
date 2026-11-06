// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "core/providers/qnn/builder/ep_context_io_dispatch.h"
#include "core/providers/qnn/builder/qnn_def.h"

#if QNN_EP_INTERNAL_SYMBOL_ACCESS && defined(QNN_FILE_MAPPED_WEIGHTS_AVAILABLE) && ORT_API_HAS_EPCONTEXT_ENCRYPTION

#include <windows.h>
#include <filesystem>
#include <fstream>
#include <cstring>

#include "core/providers/qnn/builder/qnn_context_decrypt_source.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"

namespace onnxruntime::test {
namespace {
struct SourceState {
  size_t legacy_calls = 0;
  size_t source_calls = 0;
  const void* source = nullptr;
  size_t source_bytes = 0;
  void* output = nullptr;
  bool fail = false;
  std::string name;

  static OrtStatus* ORT_API_CALL Legacy(void* state, const char* name, OrtAllocator* allocator,
                                        void** data, size_t* bytes) {
    auto& self = *static_cast<SourceState*>(state);
    ++self.legacy_calls;
    self.name = name;
    *data = allocator->Alloc(allocator, 4);
    *bytes = 4;
    if (!*data) return Ort::GetApi().CreateStatus(ORT_FAIL, "Test allocation failed.");
    self.output = *data;
    std::memcpy(*data, "data", 4);
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL Decrypt(void* state, const char* name, const void* source,
                                         size_t source_bytes, OrtAllocator* allocator,
                                         void** data, size_t* bytes) {
    auto& self = *static_cast<SourceState*>(state);
    ++self.source_calls;
    self.source = source;
    self.source_bytes = source_bytes;
    self.name = name;
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(source, &info, sizeof(info)) || info.Type != MEM_MAPPED ||
        source_bytes != 6 || std::memcmp(source, "XXdata", 6) != 0) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, "Expected exact mapped source bytes.");
    }
    *bytes = source_bytes - 2;
    *data = allocator->Alloc(allocator, *bytes);
    if (!*data) return Ort::GetApi().CreateStatus(ORT_FAIL, "Test allocation failed.");
    self.output = *data;
    std::memcpy(*data, static_cast<const char*>(source) + 2, *bytes);
    if (self.fail) {
      allocator->Free(allocator, *data);
      return Ort::GetApi().CreateStatus(ORT_FAIL, "Test decrypt failure after output cleanup.");
    }
    return nullptr;
  }
};

class QnnDecryptSourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    wchar_t folder[MAX_PATH], file[MAX_PATH];
    ASSERT_NE(GetTempPathW(MAX_PATH, folder), 0u);
    ASSERT_NE(GetTempFileNameW(folder, L"qnn", 0, file), 0u);
    path_ = file;
    std::ofstream source(path_, std::ios::binary);
    source << "XXdata";
    ASSERT_TRUE(source.good());
    Ort::ThrowOnError(Ort::GetApi().CreateSessionOptions(&options_));
    Ort::ThrowOnError(QnnEpCreateDecryptSourceBinding(SourceState::Legacy, SourceState::Decrypt,
                                                      &state_, &binding_, &bridge_));
    auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&Ort::GetApi());
    ASSERT_NE(setter, nullptr);
    Ort::ThrowOnError(setter(options_, bridge_, binding_));
  }
  void TearDown() override {
    Ort::GetApi().ReleaseSessionOptions(options_);
    QnnEpReleaseDecryptSourceBinding(binding_);
    if (!path_.empty()) EXPECT_TRUE(DeleteFileW(path_.c_str()));
  }
  void ExpectSourceReleased() {
    MEMORY_BASIC_INFORMATION info{};
    ASSERT_NE(VirtualQuery(state_.source, &info, sizeof(info)), 0u);
    EXPECT_EQ(info.State, static_cast<DWORD>(MEM_FREE));
  }
  std::filesystem::path path_;
  SourceState state_;
  OrtSessionOptions* options_ = nullptr;
  QnnEpDecryptSourceBinding* binding_ = nullptr;
  OrtReadNamedBufferFunc bridge_ = nullptr;
  Ort::Logger logger_ = MakeNullLogger();
};

TEST_F(QnnDecryptSourceTest, OffDelegatesLegacyWithOriginalStateWithoutOpeningSource) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION});
  qnn::EpContextBuffer out;
  ASSERT_TRUE(dispatch.Read("logical-name", out, "nonexistent-source").IsOK());
  EXPECT_EQ(state_.legacy_calls, 1u);
  EXPECT_EQ(state_.source_calls, 0u);
  EXPECT_EQ(state_.name, "logical-name");
  EXPECT_EQ(out.data(), state_.output);
}

TEST_F(QnnDecryptSourceTest, OnPassesMappedSourceAndAdoptsIndependentDifferentSizedOutput) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  qnn::EpContextBuffer out;
  ASSERT_TRUE(dispatch.Read("logical-name", out, path_.string()).IsOK());
  EXPECT_EQ(state_.legacy_calls, 0u);
  EXPECT_EQ(state_.source_calls, 1u);
  EXPECT_EQ(state_.source_bytes, 6u);
  EXPECT_EQ(out.size(), 4u);
  EXPECT_EQ(out.data(), state_.output);
  EXPECT_NE(out.data(), state_.source);
  ExpectSourceReleased();
  EXPECT_EQ(std::memcmp(out.data(), "data", 4), 0);
}

TEST_F(QnnDecryptSourceTest, FailedDecryptReleasesViewWithoutRetryOrReplacingPreviousOutput) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  qnn::EpContextBuffer out;
  out.AllocateUninitialized(4);
  auto* previous = out.data();
  state_.fail = true;
  EXPECT_FALSE(dispatch.Read("logical-name", out, path_.string()).IsOK());
  EXPECT_EQ(state_.source_calls, 1u);
  EXPECT_EQ(state_.legacy_calls, 0u);
  EXPECT_EQ(out.data(), previous);
  ExpectSourceReleased();
}

TEST_F(QnnDecryptSourceTest, MappingFailureDoesNotInvokeEitherCallback) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  qnn::EpContextBuffer out;
  EXPECT_FALSE(dispatch.Read("logical-name", out, path_.string() + ".absent").IsOK());
  EXPECT_EQ(state_.source_calls, 0u);
  EXPECT_EQ(state_.legacy_calls, 0u);
  EXPECT_EQ(out.data(), nullptr);
}

TEST_F(QnnDecryptSourceTest, UnrelatedCallbackIsRejectedWithoutInterpretingItsApplicationState) {
  auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&Ort::GetApi());
  Ort::ThrowOnError(setter(options_, SourceState::Legacy, &state_));
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  qnn::EpContextBuffer out;
  EXPECT_FALSE(dispatch.HasSourceReadCallback());
  EXPECT_FALSE(dispatch.Read("logical-name", out, path_.string()).IsOK());
  EXPECT_EQ(state_.legacy_calls, 0u);
  EXPECT_EQ(state_.source_calls, 0u);
}
}  // namespace
}  // namespace onnxruntime::test
#endif
