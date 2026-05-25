// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Test fixture for device compatibility tests
class QnnDeviceCompatibilityTests : public ::testing::Test {
 protected:
  void SetUp() override {
    // Get the ORT environment using the same pattern as RegisterQnnEpLibrary
    Ort::Env* ort_env = GetOrtEnv();
    ASSERT_NE(ort_env, nullptr);

    // Get the ORT API
    api_ = &Ort::GetApi();
    ASSERT_NE(api_, nullptr);

    // Get the ORT EP API
    ep_api_ = &Ort::GetEpApi();
    ASSERT_NE(ep_api_, nullptr);

    env_ = static_cast<OrtEnv*>(*ort_env);
    ASSERT_NE(env_, nullptr);
  }

  void TearDown() override {
    if (mock_hw_device != nullptr) {
      ep_api_->ReleaseHardwareDevice(mock_hw_device);
    }
  }

  // Helper function to create a mock hardware device
  OrtStatus* CreateMockHardwareDevice(OrtHardwareDeviceType type, uint32_t vendor_id) {
    return ep_api_->CreateHardwareDevice(
        type,
        vendor_id,
        0 /* device_id */,
        "QCOM" /* vendor_name */,
        nullptr /* metadata */,
        &mock_hw_device);
  }

  const uint32_t qualcomm_vendor_id{'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24)};
  OrtHardwareDevice* mock_hw_device{};

  const OrtApi* api_ = nullptr;
  const OrtEpApi* ep_api_ = nullptr;
  OrtEnv* env_ = nullptr;
};

// Test that CPU devices are compatible
TEST_F(QnnDeviceCompatibilityTests, CPUDeviceIsCompatible) {
  ProviderOptions options;
  options["backend_type"] = "cpu";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock CPU device
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_CPU, 0));

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify compatible (no incompatibility reasons)
  uint32_t reasons_bitmask = 0xFFFFFFFF;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_EQ(reasons_bitmask, 0u) << "CPU device should be compatible with QNN EP";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, 0);

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that NPU devices with Qualcomm vendor ID are compatible
TEST_F(QnnDeviceCompatibilityTests, NPUDeviceWithQualcommVendorIsCompatible) {
  ProviderOptions options;
  options["backend_type"] = "htp";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock NPU device with Qualcomm vendor ID
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_NPU, qualcomm_vendor_id));

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify compatible (no incompatibility reasons)
  uint32_t reasons_bitmask = 0xFFFFFFFF;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_EQ(reasons_bitmask, 0u) << "NPU device with Qualcomm vendor should be compatible with QNN EP";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, 0);

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that GPU devices with Qualcomm vendor ID are compatible
TEST_F(QnnDeviceCompatibilityTests, GPUDeviceWithQualcommVendorIsCompatible) {
  ProviderOptions options;
  options["backend_type"] = "gpu";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock GPU device with Qualcomm vendor ID
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_GPU, qualcomm_vendor_id));

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  // Check if GPU device is incompatible for any reason - if so, skip this test
  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));

  if (reasons_bitmask != 0u) {
    api_->ReleaseDeviceEpIncompatibilityDetails(details);
    GTEST_SKIP() << "GPU device is not compatible with QNN EP on this machine, skipping compatibility test";
  }

  // Verify compatible (no incompatibility reasons)
  EXPECT_EQ(reasons_bitmask, 0u) << "GPU device with Qualcomm vendor should be compatible with QNN EP";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, 0);

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that NPU devices with non-Qualcomm vendor ID are incompatible
TEST_F(QnnDeviceCompatibilityTests, NPUDeviceWithNonQualcommVendorIsIncompatible) {
  ProviderOptions options;
  options["backend_type"] = "htp";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock NPU device with a different vendor ID (not Qualcomm)
  uint32_t non_qualcomm_vendor_id = qualcomm_vendor_id + 1;
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_NPU, non_qualcomm_vendor_id));

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify incompatible (should have incompatibility reasons)
  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_NE(reasons_bitmask, 0u) << "NPU device with non-Qualcomm vendor should be incompatible with QNN EP";

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that GPU devices with non-Qualcomm vendor ID are incompatible
TEST_F(QnnDeviceCompatibilityTests, GPUDeviceWithNonQualcommVendorIsIncompatible) {
  ProviderOptions options;
  options["backend_type"] = "gpu";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock GPU device with a different vendor ID (not Qualcomm)
  uint32_t non_qualcomm_vendor_id = qualcomm_vendor_id + 1;
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_GPU, non_qualcomm_vendor_id));

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify incompatible (should have incompatibility reasons)
  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_NE(reasons_bitmask, 0u) << "GPU device with non-Qualcomm vendor should be incompatible with QNN EP";

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that CPU device incompatibility details include MISSING_DEPENDENCY and QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED
// Note: This should be tested by manually removing the CPU dependency
TEST_F(QnnDeviceCompatibilityTests, CPUDeviceIncompatibilityDetailsWithMissingDependency) {
  ProviderOptions options;
  options["backend_type"] = "cpu";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock CPU device
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_CPU, 0));

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  // Check if device is compatible - if so, skip this test
  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));

  if (reasons_bitmask == 0u) {
    // Device is compatible, skip this test
    api_->ReleaseDeviceEpIncompatibilityDetails(details);
    GTEST_SKIP() << "CPU device is compatible with QNN EP, skipping incompatibility test";
  }

  // Verify incompatibility reason includes MISSING_DEPENDENCY
  EXPECT_TRUE((reasons_bitmask & OrtDeviceEpIncompatibility_MISSING_DEPENDENCY) != 0)
      << "Expected MISSING_DEPENDENCY flag in incompatibility reasons";

  // Verify error code is QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED (2006)
  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED)
      << "Expected QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED error code";

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that NPU device incompatibility details include MISSING_DEPENDENCY and QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED
// Note: This should be tested by manually removing the NPU dependency
TEST_F(QnnDeviceCompatibilityTests, NPUDeviceIncompatibilityDetailsWithMissingDependency) {
  ProviderOptions options;
  options["backend_type"] = "htp";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock NPU device with Qualcomm vendor ID
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_NPU, qualcomm_vendor_id));

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  // Check if device is compatible - if so, skip this test
  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));

  if (reasons_bitmask == 0u) {
    // Device is compatible, skip this test
    api_->ReleaseDeviceEpIncompatibilityDetails(details);
    GTEST_SKIP() << "NPU device is compatible with QNN EP, skipping incompatibility test";
  }

  // Verify incompatibility reason includes MISSING_DEPENDENCY
  EXPECT_TRUE((reasons_bitmask & OrtDeviceEpIncompatibility_MISSING_DEPENDENCY) != 0)
      << "Expected MISSING_DEPENDENCY flag in incompatibility reasons";

  // Verify error code is QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED (2006)
  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED)
      << "Expected QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED error code";

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that GPU device incompatibility details include MISSING_DEPENDENCY and QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED
// Note: This should be tested by manually removing the GPU dependency
TEST_F(QnnDeviceCompatibilityTests, GPUDeviceIncompatibilityDetailsWithMissingDependency) {
  ProviderOptions options;
  options["backend_type"] = "gpu";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock GPU device with Qualcomm vendor ID
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_GPU, qualcomm_vendor_id));

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  // Check if device is compatible - if so, skip this test
  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));

  if (reasons_bitmask == 0u) {
    // Device is compatible, skip this test
    api_->ReleaseDeviceEpIncompatibilityDetails(details);
    GTEST_SKIP() << "GPU device is compatible with QNN EP, skipping incompatibility test";
  }

  if ((reasons_bitmask & OrtDeviceEpIncompatibility_MISSING_DEPENDENCY) == 0) {
    // The GPU library is present but incompatible for another reason (e.g., driver failure).
    // This test requires the GPU library itself to be absent; skip if that's not the case.
    api_->ReleaseDeviceEpIncompatibilityDetails(details);
    GTEST_SKIP() << "GPU library is present; MISSING_DEPENDENCY scenario cannot be reproduced";
  }

  // Verify incompatibility reason includes MISSING_DEPENDENCY
  EXPECT_TRUE((reasons_bitmask & OrtDeviceEpIncompatibility_MISSING_DEPENDENCY) != 0)
      << "Expected MISSING_DEPENDENCY flag in incompatibility reasons";

  // Verify error code is QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED (2006)
  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED)
      << "Expected QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED error code";

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that when the GPU backend library is present but the driver cannot initialize the
// backend (DRIVER_INCOMPATIBLE), the details are reported correctly.
// Skipped when the backend is fully compatible or when the library itself is missing.
// Note: This should be tested by manually removing the GPU driver library
TEST_F(QnnDeviceCompatibilityTests, GPUDeviceIncompatibilityDetailsWithDriverIncompatible) {
  ProviderOptions options;
  options["backend_type"] = "gpu";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock GPU device with Qualcomm vendor ID
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_GPU, qualcomm_vendor_id));

  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));

  if (reasons_bitmask == 0u) {
    api_->ReleaseDeviceEpIncompatibilityDetails(details);
    GTEST_SKIP() << "GPU backend is fully compatible; DRIVER_INCOMPATIBLE scenario cannot be reproduced";
  }

  if ((reasons_bitmask & OrtDeviceEpIncompatibility_MISSING_DEPENDENCY) != 0) {
    api_->ReleaseDeviceEpIncompatibilityDetails(details);
    GTEST_SKIP() << "GPU library is missing; DRIVER_INCOMPATIBLE scenario cannot be reproduced";
  }

  // On this system the backend loaded but the driver could not be initialized.
  EXPECT_TRUE((reasons_bitmask & OrtDeviceEpIncompatibility_DRIVER_INCOMPATIBLE) != 0)
      << "Expected DRIVER_INCOMPATIBLE when the backend library loads but initialisation fails";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED);

  const char* notes = nullptr;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetNotes(details, &notes));
  ASSERT_NE(notes, nullptr) << "Expected a non-null error message for DRIVER_INCOMPATIBLE";
  EXPECT_FALSE(std::string(notes).empty()) << "Expected a non-empty error message for DRIVER_INCOMPATIBLE";

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that when the HTP backend library is present but the hardware device cannot be
// created (DEVICE_INCOMPATIBLE), the details are reported correctly.
// Skipped when the backend is fully compatible or when the failure is not DEVICE_INCOMPATIBLE.
// Note: This should be tested by manually removing the NPU driver library
TEST_F(QnnDeviceCompatibilityTests, NPUDeviceIncompatibilityDetailsWithDeviceIncompatible) {
  ProviderOptions options;
  options["backend_type"] = "htp";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock NPU device with Qualcomm vendor ID
  ASSERT_ORTSTATUS_OK(CreateMockHardwareDevice(OrtHardwareDeviceType_NPU, qualcomm_vendor_id));

  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, kQnnExecutionProvider, mock_hw_device, &details));
  ASSERT_NE(details, nullptr);

  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));

  if (reasons_bitmask == 0u) {
    api_->ReleaseDeviceEpIncompatibilityDetails(details);
    GTEST_SKIP() << "HTP backend is fully compatible; DEVICE_INCOMPATIBLE scenario cannot be reproduced";
  }

  if ((reasons_bitmask & OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE) == 0) {
    api_->ReleaseDeviceEpIncompatibilityDetails(details);
    GTEST_SKIP() << "Failure is not DEVICE_INCOMPATIBLE on this system; skipping test";
  }

  // On this system the backend loaded but the hardware device could not be created.
  EXPECT_TRUE((reasons_bitmask & OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE) != 0)
      << "Expected DEVICE_INCOMPATIBLE when the hardware device is not present or accessible";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED);

  const char* notes = nullptr;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetNotes(details, &notes));
  ASSERT_NE(notes, nullptr) << "Expected a non-null error message for DEVICE_INCOMPATIBLE";
  EXPECT_FALSE(std::string(notes).empty()) << "Expected a non-empty error message for DEVICE_INCOMPATIBLE";

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
