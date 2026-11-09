// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"

namespace onnxruntime::test {
namespace {

class QnnContextBinaryLoadingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    qnn::QnnBackendManagerConfig config;
    manager_ = qnn::QnnBackendManager::Create(config, api_ptrs_, logger_);
    current_ = this;
    // This manager is mutable and has no SDK library loaded. Override only its local API table.
    auto& interface = const_cast<QNN_INTERFACE_VER_TYPE&>(manager_->GetQnnInterface());
    interface.errorGetMessage = [](Qnn_ErrorHandle_t error, const char** message) -> Qnn_ErrorHandle_t {
      EXPECT_EQ(error, current_->sdk_result_);
      *message = "stub context loading failure";
      return QNN_SUCCESS;
    };
    interface.contextCreateFromBinary = [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                           const QnnContext_Config_t** configs, const void* binary,
                                           Qnn_ContextBinarySize_t bytes, Qnn_ContextHandle_t* context,
                                           Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      ++current_->ordinary_calls_;
      EXPECT_EQ(binary, current_->binary_);
      EXPECT_EQ(bytes, sizeof(current_->binary_));
      EXPECT_EQ(configs, current_->configs_);
      if (current_->sdk_result_ == QNN_SUCCESS) {
        *context = current_->binary_;
      }
      return current_->sdk_result_;
    };
  }
  void TearDown() override {
    manager_.reset();
    current_ = nullptr;
  }
  Ort::Status Load(bool request_dma, Qnn_ContextHandle_t& context) {
    qnn::EpContextIoDispatch io_dispatch;
    // A nonexistent filename makes an accidental fallback reread fail the test.
    return manager_->CreateContextHandleFromBinary(binary_, sizeof(binary_), request_dma,
                                                   configs_, "not-a-context-file.bin", io_dispatch,
                                                   nullptr, context);
  }
  static thread_local QnnContextBinaryLoadingTest* current_;
  OrtApi ort_api_{};
  OrtEpApi ep_api_{};
  OrtModelEditorApi editor_api_{};
  ApiPtrs api_ptrs_{ort_api_, ep_api_, editor_api_};
  Ort::Logger logger_{MakeNullLogger()};
  std::shared_ptr<qnn::QnnBackendManager> manager_;
  char binary_[4] = {1, 2, 3, 4};
  const QnnContext_Config_t* configs_[1] = {nullptr};
  size_t ordinary_calls_ = 0;
  Qnn_ErrorHandle_t sdk_result_ = QNN_SUCCESS;
};

thread_local QnnContextBinaryLoadingTest* QnnContextBinaryLoadingTest::current_ = nullptr;

TEST_F(QnnContextBinaryLoadingTest, OrdinaryLoadingUsesSuppliedBufferWithoutReadingFile) {
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(false, context).IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, binary_);
}

TEST_F(QnnContextBinaryLoadingTest, MissingDmaResourcesFallBackToSameBuffer) {
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, binary_);
}

TEST_F(QnnContextBinaryLoadingTest, OrdinarySdkFailureAfterDmaFallbackIsPropagated) {
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  Qnn_ContextHandle_t context = nullptr;
  auto status = Load(true, context);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, nullptr);
  EXPECT_NE(std::string(status.GetErrorMessage()).find("contextCreateFromBinary failed"), std::string::npos);
  EXPECT_NE(std::string(status.GetErrorMessage()).find("stub context loading failure"), std::string::npos);
}

}  // namespace
}  // namespace onnxruntime::test

#endif
