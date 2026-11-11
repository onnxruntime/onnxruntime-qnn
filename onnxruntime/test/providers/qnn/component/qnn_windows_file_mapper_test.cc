// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_def.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && defined(QNN_FILE_MAPPED_WEIGHTS_AVAILABLE)

#include <windows.h>

#include <filesystem>
#include <fstream>
#include <memory>

#include "gtest/gtest.h"
#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"

namespace onnxruntime::test {
namespace {

class QnnWindowsFileMapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    wchar_t folder[MAX_PATH], filename[MAX_PATH];
    ASSERT_NE(GetTempPathW(MAX_PATH, folder), 0u);
    ASSERT_NE(GetTempFileNameW(folder, L"qnn", 0, filename), 0u);
    path_ = filename;
  }
  void TearDown() override {
    if (!path_.empty()) {
      EXPECT_TRUE(DeleteFileW(path_.c_str()));
    }
  }
  void Fill() {
    std::ofstream file(path_, std::ios::binary);
    file << "context binary";
    ASSERT_TRUE(file.good());
  }
  std::filesystem::path path_;
};

TEST_F(QnnWindowsFileMapperTest, EmptyFileFailsAtCreateFileMappingAndPreservesOutput) {
  auto logger = MakeNullLogger();
  qnn::WindowsFileMapper mapper(logger);
  qnn::FileMappingInterface::MappedMemoryPtr owner(new char[1], [](void* ptr) { delete[] static_cast<char*>(ptr); });
  auto* original = owner.get();
  auto status = mapper.MapContextBinary(path_.string(), owner);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find("Failed to create file mapping handle"), std::string::npos);
  EXPECT_EQ(owner.get(), original);
}

TEST_F(QnnWindowsFileMapperTest, ContextOwnedViewsSurviveMapperAndReleaseIndependently) {
  Fill();
  auto logger = MakeNullLogger();
  qnn::FileMappingInterface::MappedMemoryPtr first, second;
  {
    qnn::WindowsFileMapper mapper(logger);
    ASSERT_TRUE(mapper.MapContextBinary(path_.string(), first).IsOK());
    ASSERT_TRUE(mapper.MapContextBinary(path_.string(), second).IsOK());
    ASSERT_NE(first.get(), second.get());
  }
  EXPECT_EQ(first.get()[0], 'c');
  EXPECT_EQ(second.get()[0], 'c');
  auto* address = first.get();
  first.reset();
  MEMORY_BASIC_INFORMATION info{};
  ASSERT_NE(VirtualQuery(address, &info, sizeof(info)), 0u);
  EXPECT_EQ(info.State, static_cast<DWORD>(MEM_FREE));
  EXPECT_EQ(second.get()[0], 'c');
}

TEST_F(QnnWindowsFileMapperTest, LegacyCachedViewRemainsAvailableUntilMapperDestruction) {
  Fill();
  auto logger = MakeNullLogger();
  void* first = nullptr;
  {
    qnn::WindowsFileMapper mapper(logger);
    void* second = nullptr;
    ASSERT_TRUE(mapper.GetContextBinMappedMemoryPtr(path_.string(), &first).IsOK());
    ASSERT_TRUE(mapper.GetContextBinMappedMemoryPtr(path_.string(), &second).IsOK());
    EXPECT_EQ(first, second);
    EXPECT_EQ(static_cast<char*>(first)[0], 'c');
    EXPECT_FALSE(mapper.GetContextBinMappedMemoryPtr(path_.string(), nullptr).IsOK());
  }
  MEMORY_BASIC_INFORMATION info{};
  ASSERT_NE(VirtualQuery(first, &info, sizeof(info)), 0u);
  EXPECT_EQ(info.State, static_cast<DWORD>(MEM_FREE));
}

}  // namespace
}  // namespace onnxruntime::test

#endif
