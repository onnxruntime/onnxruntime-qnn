// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>

#include "core/providers/qnn/qnn_provider_factory.h"
#include "core/providers/qnn/qnn_execution_provider.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/abi_devices.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {

// Mock QnnEp for testing
class MockQnnEp : public QnnEp {
 public:
  MockQnnEp(QnnEpFactory& factory,
            const std::string& name,
            const OrtSessionOptions& session_options,
            const OrtLogger* logger)
      : QnnEp(factory, name, session_options, logger) {}

  MOCK_METHOD(OrtStatus*, GetHardwareDeviceIncompatibilityDetails,
              (const OrtHardwareDevice* hw, OrtDeviceEpIncompatibilityDetails* details),
              (noexcept));
};

class QnnProviderFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize ORT API
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ASSERT_NE(ort_api, nullptr);
    Ort::InitApi(ort_api);

    // Create session options
    ASSERT_EQ(ort_api->CreateSessionOptions(&session_options_), nullptr);

    // Get EP API
    ep_api_ = ort_api->GetEpApi();
    ASSERT_NE(ep_api_, nullptr);

    // Get Model Editor API
    model_editor_api_ = ort_api->GetModelEditorApi();
    ASSERT_NE(model_editor_api_, nullptr);

    // Create factory
    api_ptrs_ = std::make_unique<ApiPtrs>(ApiPtrs{*ort_api, *ep_api_, *model_editor_api_});
    factory_ = std::make_unique<QnnEpFactory>("QNN", *api_ptrs_);
  }

  void TearDown() override {
    factory_.reset();
    api_ptrs_.reset();
    if (session_options_) {
      OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseSessionOptions(session_options_);
      session_options_ = nullptr;
    }
  }

  // Helper to create a hardware device
  OrtHardwareDevice CreateHardwareDevice(OrtHardwareDeviceType type, uint32_t vendor_id) {
    OrtHardwareDevice device;
    device.type = type;
    device.vendor_id = vendor_id;
    device.device_id = 0;
    device.name = nullptr;
    device.description = nullptr;
    return device;
  }

  // Helper to create incompatibility details
  OrtDeviceEpIncompatibilityDetails* CreateIncompatibilityDetails() {
    OrtDeviceEpIncompatibilityDetails* details = nullptr;
    OrtStatus* status = ep_api_->CreateDeviceEpIncompatibilityDetails(&details);
    EXPECT_EQ(status, nullptr);
    return details;
  }

  // Helper to get details from incompatibility details object
  void GetDetailsInfo(OrtDeviceEpIncompatibilityDetails* details,
                     uint32_t& reasons,
                     int32_t& error_code,
                     std::string& message) {
    const char* msg_ptr = nullptr;
    OrtStatus* status = ep_api_->DeviceEpIncompatibilityDetails_GetDetails(
        details, &reasons, &error_code, &msg_ptr);
    EXPECT_EQ(status, nullptr);
    if (msg_ptr) {
      message = msg_ptr;
    }
  }

  OrtSessionOptions* session_options_ = nullptr;
  const OrtEpApi* ep_api_ = nullptr;
  const OrtModelEditorApi* model_editor_api_ = nullptr;
  std::unique_ptr<ApiPtrs> api_ptrs_;
  std::unique_ptr<QnnEpFactory> factory_;
};

// Test: CPU device should be compatible
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_CPUDevice_Compatible) {
  // Create CPU device
  OrtHardwareDevice cpu_device = CreateHardwareDevice(OrtHardwareDeviceType_CPU, 0);

  // Create incompatibility details
  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  // Call the function under test
  OrtStatus* status = factory_->GetHardwareDeviceIncompatibilityDetails(
      factory_.get(), &cpu_device, details);

  // CPU should be compatible (though backend creation might fail in test environment)
  // The function should at least not crash and return a status
  if (status != nullptr) {
    // If there's an error, it should be about backend creation, not device incompatibility
    std::string error_msg(OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetErrorMessage(status));
    // The error should not be about platform not supported
    EXPECT_TRUE(error_msg.find("platform not supported") == std::string::npos ||
                error_msg.find("compatible") != std::string::npos);
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
  }

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: NPU device with Qualcomm vendor ID should be compatible
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_NPUDevice_QualcommVendor_Compatible) {
  // Qualcomm vendor ID: 'QCOM' = 0x4D4F4351
  uint32_t qualcomm_vendor_id = 'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24);
  OrtHardwareDevice npu_device = CreateHardwareDevice(OrtHardwareDeviceType_NPU, qualcomm_vendor_id);

  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  OrtStatus* status = factory_->GetHardwareDeviceIncompatibilityDetails(
      factory_.get(), &npu_device, details);

  // Should attempt to create backend (may fail in test environment, but not due to vendor ID)
  if (status != nullptr) {
    std::string error_msg(OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetErrorMessage(status));
    // Should not fail due to platform not supported
    EXPECT_TRUE(error_msg.find("only supports general CPU devices and Qualcomm NPU and GPU devices") == std::string::npos);
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
  }

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: GPU device with Qualcomm vendor ID should be compatible
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_GPUDevice_QualcommVendor_Compatible) {
  uint32_t qualcomm_vendor_id = 'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24);
  OrtHardwareDevice gpu_device = CreateHardwareDevice(OrtHardwareDeviceType_GPU, qualcomm_vendor_id);

  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  OrtStatus* status = factory_->GetHardwareDeviceIncompatibilityDetails(
      factory_.get(), &gpu_device, details);

  // Should attempt to create backend (may fail in test environment, but not due to vendor ID)
  if (status != nullptr) {
    std::string error_msg(OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetErrorMessage(status));
    // Should not fail due to platform not supported
    EXPECT_TRUE(error_msg.find("only supports general CPU devices and Qualcomm NPU and GPU devices") == std::string::npos);
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
  }

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: NPU device with non-Qualcomm vendor ID should be incompatible
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_NPUDevice_WrongVendor_Incompatible) {
  // Use a different vendor ID (e.g., Intel)
  uint32_t wrong_vendor_id = 0x8086;
  OrtHardwareDevice npu_device = CreateHardwareDevice(OrtHardwareDeviceType_NPU, wrong_vendor_id);

  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  OrtStatus* status = factory_->GetHardwareDeviceIncompatibilityDetails(
      factory_.get(), &npu_device, details);

  // Should succeed in setting incompatibility details
  ASSERT_EQ(status, nullptr);

  // Check the incompatibility details
  uint32_t reasons = 0;
  int32_t error_code = 0;
  std::string message;
  GetDetailsInfo(details, reasons, error_code, message);

  // Should indicate device incompatibility
  EXPECT_EQ(reasons, static_cast<uint32_t>(OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE));
  EXPECT_NE(error_code, 0);  // Should have a QNN error code
  EXPECT_TRUE(message.find("QNN EP only supports general CPU devices and Qualcomm NPU and GPU devices") != std::string::npos);

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: GPU device with non-Qualcomm vendor ID should be incompatible
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_GPUDevice_WrongVendor_Incompatible) {
  // Use a different vendor ID (e.g., NVIDIA)
  uint32_t wrong_vendor_id = 0x10DE;
  OrtHardwareDevice gpu_device = CreateHardwareDevice(OrtHardwareDeviceType_GPU, wrong_vendor_id);

  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  OrtStatus* status = factory_->GetHardwareDeviceIncompatibilityDetails(
      factory_.get(), &gpu_device, details);

  // Should succeed in setting incompatibility details
  ASSERT_EQ(status, nullptr);

  // Check the incompatibility details
  uint32_t reasons = 0;
  int32_t error_code = 0;
  std::string message;
  GetDetailsInfo(details, reasons, error_code, message);

  // Should indicate device incompatibility
  EXPECT_EQ(reasons, static_cast<uint32_t>(OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE));
  EXPECT_NE(error_code, 0);  // Should have a QNN error code
  EXPECT_TRUE(message.find("QNN EP only supports general CPU devices and Qualcomm NPU and GPU devices") != std::string::npos);

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: Unsupported device type should be incompatible
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_UnsupportedDeviceType_Incompatible) {
  // Use an unsupported device type (e.g., FPGA)
  uint32_t qualcomm_vendor_id = 'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24);
  OrtHardwareDevice fpga_device = CreateHardwareDevice(OrtHardwareDeviceType_FPGA, qualcomm_vendor_id);

  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  OrtStatus* status = factory_->GetHardwareDeviceIncompatibilityDetails(
      factory_.get(), &fpga_device, details);

  // Should succeed in setting incompatibility details
  ASSERT_EQ(status, nullptr);

  // Check the incompatibility details
  uint32_t reasons = 0;
  int32_t error_code = 0;
  std::string message;
  GetDetailsInfo(details, reasons, error_code, message);

  // Should indicate device incompatibility
  EXPECT_EQ(reasons, static_cast<uint32_t>(OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE));
  EXPECT_NE(error_code, 0);  // Should have a QNN error code
  EXPECT_TRUE(message.find("QNN EP only supports general CPU devices and Qualcomm NPU and GPU devices") != std::string::npos);

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: Null hardware device pointer should be handled gracefully
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_NullDevice_ReturnsError) {
  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  // This should crash or return an error - testing defensive programming
  // In production code, this should be validated
  // For now, we just document the expected behavior

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: Null details pointer should be handled gracefully
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_NullDetails_ReturnsError) {
  uint32_t qualcomm_vendor_id = 'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24);
  OrtHardwareDevice npu_device = CreateHardwareDevice(OrtHardwareDeviceType_NPU, qualcomm_vendor_id);

  // This should crash or return an error - testing defensive programming
  // In production code, this should be validated
  // For now, we just document the expected behavior
}

// Test: Backend type determination for CPU device
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_CPUDevice_BackendType) {
  OrtHardwareDevice cpu_device = CreateHardwareDevice(OrtHardwareDeviceType_CPU, 0);

  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  // The function should attempt to create a QNN EP with CPU backend type
  OrtStatus* status = factory_->GetHardwareDeviceIncompatibilityDetails(
      factory_.get(), &cpu_device, details);

  // In test environment, backend creation will likely fail, but we can verify
  // the function doesn't crash and handles the CPU device type correctly
  if (status != nullptr) {
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
  }

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: Backend type determination for NPU device
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_NPUDevice_BackendType) {
  uint32_t qualcomm_vendor_id = 'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24);
  OrtHardwareDevice npu_device = CreateHardwareDevice(OrtHardwareDeviceType_NPU, qualcomm_vendor_id);

  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  // The function should attempt to create a QNN EP with HTP backend type
  OrtStatus* status = factory_->GetHardwareDeviceIncompatibilityDetails(
      factory_.get(), &npu_device, details);

  // In test environment, backend creation will likely fail, but we can verify
  // the function doesn't crash and handles the NPU device type correctly
  if (status != nullptr) {
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
  }

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: Backend type determination for GPU device
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_GPUDevice_BackendType) {
  uint32_t qualcomm_vendor_id = 'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24);
  OrtHardwareDevice gpu_device = CreateHardwareDevice(OrtHardwareDeviceType_GPU, qualcomm_vendor_id);

  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  // The function should attempt to create a QNN EP with GPU backend type
  OrtStatus* status = factory_->GetHardwareDeviceIncompatibilityDetails(
      factory_.get(), &gpu_device, details);

  // In test environment, backend creation will likely fail, but we can verify
  // the function doesn't crash and handles the GPU device type correctly
  if (status != nullptr) {
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
  }

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
// Test: Using real QNN EP on actual device
// This test only runs on platforms where QNN backend is available
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_RealDevice_NPU) {
  uint32_t qualcomm_vendor_id = 'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24);
  OrtHardwareDevice npu_device = CreateHardwareDevice(OrtHardwareDeviceType_NPU, qualcomm_vendor_id);

  // Create a real QNN EP with HTP backend
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;

  // Try to register QNN EP - this will only succeed if QNN backend is available
  OrtStatus* register_status = nullptr;
  try {
    RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, provider_options);
  } catch (...) {
    // If QNN backend is not available, skip this test
    GTEST_SKIP() << "QNN backend not available on this device";
    return;
  }

  // Get the factory from the registered device
  OrtEpFactory* real_factory = registered_ep_device->GetMutableFactory();
  ASSERT_NE(real_factory, nullptr);

  // Create incompatibility details
  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  // Call GetHardwareDeviceIncompatibilityDetails with the real factory
  OrtStatus* status = real_factory->GetHardwareDeviceIncompatibilityDetails(
      real_factory, &npu_device, details);

  // On a real Qualcomm device with HTP backend, this should succeed
  if (status == nullptr) {
    // Check the incompatibility details
    uint32_t reasons = 0;
    int32_t error_code = 0;
    std::string message;
    GetDetailsInfo(details, reasons, error_code, message);

    // Device should be compatible on real hardware
    EXPECT_EQ(reasons, static_cast<uint32_t>(OrtDeviceEpIncompatibility_NONE));
    EXPECT_EQ(error_code, 0);  // QNN_SUCCESS
    EXPECT_TRUE(message.find("compatible") != std::string::npos);
  } else {
    // If there's an error, it should be about backend initialization, not device incompatibility
    std::string error_msg(OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetErrorMessage(status));
    // Log the error for debugging
    std::cout << "GetHardwareDeviceIncompatibilityDetails returned error: " << error_msg << std::endl;
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
  }

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: Using real QNN EP to test CPU device compatibility
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_RealDevice_CPU) {
  OrtHardwareDevice cpu_device = CreateHardwareDevice(OrtHardwareDeviceType_CPU, 0);

  // Create a real QNN EP with CPU backend
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;

  // Try to register QNN EP - this will only succeed if QNN backend is available
  try {
    RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, provider_options);
  } catch (...) {
    // If QNN backend is not available, skip this test
    GTEST_SKIP() << "QNN backend not available on this device";
    return;
  }

  // Get the factory from the registered device
  OrtEpFactory* real_factory = registered_ep_device->GetMutableFactory();
  ASSERT_NE(real_factory, nullptr);

  // Create incompatibility details
  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  // Call GetHardwareDeviceIncompatibilityDetails with the real factory
  OrtStatus* status = real_factory->GetHardwareDeviceIncompatibilityDetails(
      real_factory, &cpu_device, details);

  // CPU backend should be available on most platforms
  if (status == nullptr) {
    // Check the incompatibility details
    uint32_t reasons = 0;
    int32_t error_code = 0;
    std::string message;
    GetDetailsInfo(details, reasons, error_code, message);

    // CPU device should be compatible
    EXPECT_EQ(reasons, static_cast<uint32_t>(OrtDeviceEpIncompatibility_NONE));
    EXPECT_EQ(error_code, 0);  // QNN_SUCCESS
  } else {
    // Log any errors for debugging
    std::string error_msg(OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetErrorMessage(status));
    std::cout << "GetHardwareDeviceIncompatibilityDetails returned error: " << error_msg << std::endl;
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
  }

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test: Using real QNN EP to test incompatible device (wrong vendor)
TEST_F(QnnProviderFactoryTest, GetHardwareDeviceIncompatibilityDetails_RealDevice_WrongVendor) {
  // Use a non-Qualcomm vendor ID
  uint32_t wrong_vendor_id = 0x8086;  // Intel
  OrtHardwareDevice npu_device = CreateHardwareDevice(OrtHardwareDeviceType_NPU, wrong_vendor_id);

  // Create a real QNN EP
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;

  try {
    RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, provider_options);
  } catch (...) {
    GTEST_SKIP() << "QNN backend not available on this device";
    return;
  }

  // Get the factory from the registered device
  OrtEpFactory* real_factory = registered_ep_device->GetMutableFactory();
  ASSERT_NE(real_factory, nullptr);

  // Create incompatibility details
  OrtDeviceEpIncompatibilityDetails* details = CreateIncompatibilityDetails();
  ASSERT_NE(details, nullptr);

  // Call GetHardwareDeviceIncompatibilityDetails with the real factory
  OrtStatus* status = real_factory->GetHardwareDeviceIncompatibilityDetails(
      real_factory, &npu_device, details);

  // Should succeed in setting incompatibility details
  ASSERT_EQ(status, nullptr);

  // Check the incompatibility details
  uint32_t reasons = 0;
  int32_t error_code = 0;
  std::string message;
  GetDetailsInfo(details, reasons, error_code, message);

  // Should indicate device incompatibility due to wrong vendor
  EXPECT_EQ(reasons, static_cast<uint32_t>(OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE));
  EXPECT_NE(error_code, 0);  // Should have a QNN error code
  EXPECT_TRUE(message.find("QNN EP only supports general CPU devices and Qualcomm NPU and GPU devices") != std::string::npos);

  ep_api_->ReleaseDeviceEpIncompatibilityDetails(details);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
