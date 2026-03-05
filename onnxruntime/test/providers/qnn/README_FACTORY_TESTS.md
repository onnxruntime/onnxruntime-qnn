# QNN Provider Factory Unit Tests

## Overview
This document describes the unit tests for `GetHardwareDeviceIncompatibilityDetails` in `qnn_provider_factory.cc`.

## Test File
- **Location**: `onnxruntime/test/providers/qnn/qnn_provider_factory_test.cc`
- **Test Framework**: Google Test (gtest) and Google Mock (gmock)

## Test Coverage

The test suite covers the following scenarios:

### 1. Compatible Device Tests
- **CPU Device**: Tests that CPU devices are recognized as compatible
- **NPU with Qualcomm Vendor**: Tests NPU devices with correct Qualcomm vendor ID
- **GPU with Qualcomm Vendor**: Tests GPU devices with correct Qualcomm vendor ID

### 2. Incompatible Device Tests
- **NPU with Wrong Vendor**: Tests that NPU devices with non-Qualcomm vendor IDs are rejected
- **GPU with Wrong Vendor**: Tests that GPU devices with non-Qualcomm vendor IDs are rejected
- **Unsupported Device Type**: Tests that unsupported device types (e.g., FPGA) are rejected

### 3. Backend Type Determination Tests
- **CPU Backend**: Verifies correct backend type selection for CPU devices
- **HTP Backend**: Verifies correct backend type selection for NPU devices
- **GPU Backend**: Verifies correct backend type selection for GPU devices

### 4. Edge Case Tests
- **Null Device Pointer**: Documents expected behavior for null hardware device
- **Null Details Pointer**: Documents expected behavior for null incompatibility details

### 5. Real Device Tests (Platform-Specific)
These tests only run on platforms where QNN backend is available (ARM64/Linux):
- **Real NPU Device**: Tests with actual QNN HTP backend on Qualcomm NPU
- **Real CPU Device**: Tests with actual QNN CPU backend
- **Real Device Wrong Vendor**: Tests incompatibility detection with real QNN EP

## Test Structure

### Test Fixture: `QnnProviderFactoryTest`
The test fixture provides:
- Proper initialization of ORT API, EP API, and Model Editor API
- Factory creation and cleanup
- Helper methods for creating hardware devices and incompatibility details
- Helper methods for extracting incompatibility information

### Key Helper Methods
```cpp
OrtHardwareDevice CreateHardwareDevice(OrtHardwareDeviceType type, uint32_t vendor_id)
OrtDeviceEpIncompatibilityDetails* CreateIncompatibilityDetails()
void GetDetailsInfo(OrtDeviceEpIncompatibilityDetails* details, ...)
```

## Running the Tests

The tests are automatically included in the main test suite. They will be compiled and run as part of:
- `onnxruntime_test_all` target (if QNN EP is enabled)
- `onnxruntime_provider_test` target (for provider-specific tests)

### Build Requirements
- QNN EP must be enabled: `-Donnxruntime_USE_QNN=ON`
- Not a minimal build: `-Donnxruntime_MINIMAL_BUILD=OFF`
- Not a reduced ops build: `-Donnxruntime_REDUCED_OPS_BUILD=OFF`

### Running Specific Tests
```bash
# Run all QNN provider factory tests
./onnxruntime_test_all --gtest_filter="QnnProviderFactoryTest.*"

# Run a specific test
./onnxruntime_test_all --gtest_filter="QnnProviderFactoryTest.GetHardwareDeviceIncompatibilityDetails_NPUDevice_WrongVendor_Incompatible"
```

## Implementation Notes

### Vendor ID
The Qualcomm vendor ID is calculated as: `'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24)`
This matches the vendor ID used in the QNN EP factory.

### Test Environment Limitations
- Tests run in a test environment where actual QNN backend libraries may not be available
- Backend creation failures are expected and handled gracefully
- Tests focus on the device compatibility checking logic rather than full backend initialization

### Mock Considerations
While the test file includes a `MockQnnEp` class definition, the current tests primarily test the factory's device compatibility checking logic without requiring full mocking of the QNN EP. The mock class is available for future test expansion if needed.

## Future Enhancements

Potential areas for test expansion:
1. More comprehensive backend initialization testing with proper mocks
2. Testing of the temporary QNN EP creation and cleanup
3. Testing of error propagation from QNN EP to factory
4. Testing with actual QNN backend libraries (integration tests)
5. Performance testing for device compatibility checks

## Related Files
- **Implementation**: `onnxruntime/core/providers/qnn/qnn_provider_factory.cc`
- **Header**: `onnxruntime/core/providers/qnn/qnn_provider_factory.h`
- **QNN EP**: `onnxruntime/core/providers/qnn/qnn_execution_provider.cc`
- **Other QNN Tests**: `onnxruntime/test/providers/qnn/qnn_ep_context_test.cc`
