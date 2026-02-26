// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if !defined(ORT_MINIMAL_BUILD)

// Test fixture for device compatibility tests
class QnnDeviceCompatibilityTests : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure environment is initialized
    ASSERT_NE(ort_env, nullptr);

    // Get the ORT API
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ASSERT_NE(api_, nullptr);

    env_ = static_cast<OrtEnv*>(*ort_env);
    ASSERT_NE(env_, nullptr);
  }

  // Helper function to create a mock hardware device
  OrtHardwareDevice CreateMockDevice(OrtHardwareDeviceType type, uint32_t vendor_id) {
    OrtHardwareDevice device{};
    device.type = type;
    device.vendor_id = vendor_id;
    device.device_id = 0;
    return device;
  }

  const OrtApi* api_ = nullptr;
  OrtEnv* env_ = nullptr;
};

// Test that CPU devices are compatible
TEST_F(QnnDeviceCompatibilityTests, CPUDeviceIsCompatible) {
  onnxruntime::ProviderOptions options;
#if defined(_WIN32)
  options["backend_path"] = "QnnCpu.dll";
#else
  options["backend_path"] = "libQnnCpu.so";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock CPU device
  OrtHardwareDevice cpu_device = CreateMockDevice(OrtHardwareDeviceType_CPU, 0);

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, onnxruntime::kQnnExecutionProvider, &cpu_device, &details));
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
  onnxruntime::ProviderOptions options;
#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);
  ASSERT_NE(registered_ep_device->ep_factory, nullptr);

  // Get the Qualcomm vendor ID from the factory
  uint32_t qualcomm_vendor_id = registered_ep_device->ep_factory->GetVendorId(registered_ep_device->ep_factory);

  // Create a mock NPU device with Qualcomm vendor ID
  OrtHardwareDevice npu_device = CreateMockDevice(OrtHardwareDeviceType_NPU, qualcomm_vendor_id);

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, onnxruntime::kQnnExecutionProvider, &npu_device, &details));
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
  onnxruntime::ProviderOptions options;
#if defined(_WIN32)
  options["backend_path"] = "QnnGpu.dll";
#else
  options["backend_path"] = "libQnnGpu.so";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);
  ASSERT_NE(registered_ep_device->ep_factory, nullptr);

  // Get the Qualcomm vendor ID from the factory
  uint32_t qualcomm_vendor_id = registered_ep_device->ep_factory->GetVendorId(registered_ep_device->ep_factory);

  // Create a mock GPU device with Qualcomm vendor ID
  OrtHardwareDevice gpu_device = CreateMockDevice(OrtHardwareDeviceType_GPU, qualcomm_vendor_id);

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, onnxruntime::kQnnExecutionProvider, &gpu_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify compatible (no incompatibility reasons)
  uint32_t reasons_bitmask = 0xFFFFFFFF;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_EQ(reasons_bitmask, 0u) << "GPU device with Qualcomm vendor should be compatible with QNN EP";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, 0);

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that NPU devices with non-Qualcomm vendor ID are incompatible
TEST_F(QnnDeviceCompatibilityTests, NPUDeviceWithNonQualcommVendorIsIncompatible) {
  onnxruntime::ProviderOptions options;
#if defined(_WIN32)
  options["backend_path"] = "QnnHtp.dll";
#else
  options["backend_path"] = "libQnnHtp.so";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);
  ASSERT_NE(registered_ep_device->ep_factory, nullptr);

  // Get the Qualcomm vendor ID from the factory
  uint32_t qualcomm_vendor_id = registered_ep_device->ep_factory->GetVendorId(registered_ep_device->ep_factory);

  // Create a mock NPU device with a different vendor ID (not Qualcomm)
  uint32_t non_qualcomm_vendor_id = qualcomm_vendor_id + 1;
  OrtHardwareDevice npu_device = CreateMockDevice(OrtHardwareDeviceType_NPU, non_qualcomm_vendor_id);

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, onnxruntime::kQnnExecutionProvider, &npu_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify incompatible (should have incompatibility reasons)
  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_NE(reasons_bitmask, 0u) << "NPU device with non-Qualcomm vendor should be incompatible with QNN EP";

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test that GPU devices with non-Qualcomm vendor ID are incompatible
TEST_F(QnnDeviceCompatibilityTests, GPUDeviceWithNonQualcommVendorIsIncompatible) {
  onnxruntime::ProviderOptions options;
#if defined(_WIN32)
  options["backend_path"] = "QnnGpu.dll";
#else
  options["backend_path"] = "libQnnGpu.so";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);
  ASSERT_NE(registered_ep_device->ep_factory, nullptr);

  // Get the Qualcomm vendor ID from the factory
  uint32_t qualcomm_vendor_id = registered_ep_device->ep_factory->GetVendorId(registered_ep_device->ep_factory);

  // Create a mock GPU device with a different vendor ID (not Qualcomm)
  uint32_t non_qualcomm_vendor_id = qualcomm_vendor_id + 1;
  OrtHardwareDevice gpu_device = CreateMockDevice(OrtHardwareDeviceType_GPU, non_qualcomm_vendor_id);

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, onnxruntime::kQnnExecutionProvider, &gpu_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify incompatible (should have incompatibility reasons)
  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_NE(reasons_bitmask, 0u) << "GPU device with non-Qualcomm vendor should be incompatible with QNN EP";

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test with CPU backend type option
TEST_F(QnnDeviceCompatibilityTests, CPUBackendTypeOption) {
  onnxruntime::ProviderOptions options;
  options["backend_type"] = "cpu";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);

  // Create a mock CPU device
  OrtHardwareDevice cpu_device = CreateMockDevice(OrtHardwareDeviceType_CPU, 0);

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, onnxruntime::kQnnExecutionProvider, &cpu_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify compatible (no incompatibility reasons)
  uint32_t reasons_bitmask = 0xFFFFFFFF;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_EQ(reasons_bitmask, 0u) << "CPU device should be compatible with QNN EP (CPU backend)";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, 0);

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test with HTP backend type option
TEST_F(QnnDeviceCompatibilityTests, HTPBackendTypeOption) {
  onnxruntime::ProviderOptions options;
  options["backend_type"] = "htp";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);
  ASSERT_NE(registered_ep_device->ep_factory, nullptr);

  // Get the Qualcomm vendor ID from the factory
  uint32_t qualcomm_vendor_id = registered_ep_device->ep_factory->GetVendorId(registered_ep_device->ep_factory);

  // Create a mock NPU device with Qualcomm vendor ID
  OrtHardwareDevice npu_device = CreateMockDevice(OrtHardwareDeviceType_NPU, qualcomm_vendor_id);

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, onnxruntime::kQnnExecutionProvider, &npu_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify compatible (no incompatibility reasons)
  uint32_t reasons_bitmask = 0xFFFFFFFF;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_EQ(reasons_bitmask, 0u) << "NPU device should be compatible with QNN EP (HTP backend)";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, 0);

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Test with GPU backend type option
TEST_F(QnnDeviceCompatibilityTests, GPUBackendTypeOption) {
  onnxruntime::ProviderOptions options;
  options["backend_type"] = "gpu";

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, onnxruntime::kQnnExecutionProvider, options);

  ASSERT_NE(registered_ep_device, nullptr);
  ASSERT_NE(registered_ep_device->ep_factory, nullptr);

  // Get the Qualcomm vendor ID from the factory
  uint32_t qualcomm_vendor_id = registered_ep_device->ep_factory->GetVendorId(registered_ep_device->ep_factory);

  // Create a mock GPU device with Qualcomm vendor ID
  OrtHardwareDevice gpu_device = CreateMockDevice(OrtHardwareDeviceType_GPU, qualcomm_vendor_id);

  // Check compatibility using the ORT C API
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api_->GetHardwareDeviceEpIncompatibilityDetails(
      env_, onnxruntime::kQnnExecutionProvider, &gpu_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify compatible (no incompatibility reasons)
  uint32_t reasons_bitmask = 0xFFFFFFFF;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_EQ(reasons_bitmask, 0u) << "GPU device should be compatible with QNN EP (GPU backend)";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api_->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, 0);

  api_->ReleaseDeviceEpIncompatibilityDetails(details);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace test
}  // namespace onnxruntime
