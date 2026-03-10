# QNN Execution Provider Development Guide

## Overview

This guide provides instructions for developing features for the ONNX Runtime (ORT) QNN Execution Provider using the ABI-compliant EP library. The ABI compatibility ensures stable interfaces and streamlined session creation.

- **Note**: QNN EP version < 2.0 is **NOT** included in this page.

## Contents

- [Objective](#objective)
- [Key Components](#key-components)
- [ABI API References](#abi-api-references)
- [Application Workflow](#application-workflow)
- [Development Process](#development-process)
- [Testing](#testing)

## Objective

Enable ONNX Runtime developers to integrate and maintain features using the ABI-compliant EP library, ensuring ABI compatibility and streamlined session creation.

## Key Components

### Execution Provider (EP) Library
A plugin library (`.dll` or `.so`) registered by ORT to provide hardware-specific execution capabilities.

### ABI Interfaces
- **Public C API**: `onnxruntime_c_api.h`
- **EP-specific structs**: `OrtEpApi`, `OrtEpFactory`, `OrtEpDevice`

## ABI API References

### Documentation
- [Plugin Execution Provider Libraries](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/)
- [Usage Guide](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html)
- [API Header](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_c_api.h)

### Sample Code
- **C++**: [ORT auto EP selection test](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/autoep/test_selection.cc)
- **C#**: [test_qnnep.cs](../../qcom/samples/test_qnnep.cs)
- **Python**: [test_wheel.py](../../qcom/samples/test_wheel.py)

## Application Workflow

### Step 1: Register EP Library

Use `RegisterExecutionProviderLibrary(env, "ep_lib_name", "ep_path.dll")` to load the EP dynamically.

**Functions involved:**
- `LoadLibrary()` - Loads the EP library
- `GetSymbols()` - Retrieves function pointers
- `CreateFactories()` - Returns `OrtEpFactory` objects

**Cleanup:**
Use `UnregisterExecutionProviderLibrary()` when cleaning up resources.

### Step 2: Create Session

#### Option A: Explicit EP Device

1. Query supported hardware via `GetEpDevices(env)`
2. Append EP device using `SessionOptions.AppendExecutionProvider_V2(OrtEpDevice, ProviderOptionsOverride)`
3. Create session: `CreateSession(SessionOptions)`

#### Option B: Automatic EP Selection

1. Set policy: `SessionOptions.SetEpSelectionPolicy(PREFER_NPU)`
2. Create session: `CreateSession(SessionOptions)`

**Reference:** [Usage Documentation](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html)

## Development Process

### 1. Repository

The ORT QNN EP code is maintained in the onnxruntime/onnxruntime-qnn GitHub repository:
- **Repository**: [onnxruntime/onnxruntime-qnn](https://github.com/onnxruntime/onnxruntime-qnn)

### 2. Build Process

For detailed build instructions, including prerequisites, build options, and platform-specific commands, please refer to the [Build Instructions](build.md).

The build process involves:
1. Downloading the pre-built ONNX Runtime SDK
2. Downloading the Qualcomm AI Engine Direct SDK (QAIRT)
3. Building the QNN EP as an ABI-compatible plugin library

After building, you'll have:
- **ABI EP Plugin Library**: `onnxruntime_providers_qnn.dll`
- **Python Wheel** (optional): `onnxruntime_qnn-[version]-py3-none-[platform].whl`

### 3. Development Workflow

#### Workflow Steps

1. Pull latest commits from the [onnxruntime-qnn](https://github.com/onnxruntime/onnxruntime-qnn) repository
2. Implement QNN EP-specific changes in the repository
3. Test your changes locally
4. Submit a pull request

#### Coding Guidelines

- **Use ABI C API**: Work with `onnxruntime_c_api.h` and EP structs (`OrtEpApi`, `OrtEpFactory`, `OrtEpDevice`)
- **Avoid internal dependencies**: Do not directly depend on ORT internal headers
- **Follow existing patterns**: Review existing code in the repository for consistency

#### Key Features Supported

- `GetCapability` - Query operator support
- `Compile` - Model compilation
- `EPContext` ONNX models
- LPBQ (Low Precision Binary Quantization)
- Data layout customization via `ShouldConvertDataLayoutForOp()`

#### Debugging

IR writer integration and debug tools are in development for ABI workflows. For now, use standard debugging techniques:
- Enable verbose logging
- Use debugger breakpoints
- Review QNN SDK logs

## Testing

### Testing Applications

#### Pre-built Testing Applications

1. **onnxruntime_perf_test** - Performance testing
   ```bash
   onnxruntime_perf_test.exe `
      --plugin_ep_libs "QNNExecutionProvider|onnxruntime_providers_qnn.dll" `
      --plugin_eps "QNNExecutionProvider" `
      -i "backend_path|QnnHtp.dll" -C "session.disable_cpu_ep_fallback|1" `
      -m times -r 100 model.onnx
   ```

2. **onnx_test_runner** - Model validation
   - **Note:** `onnx_test_runner.exe` doesn't work with ABI library on ORT 1.24
   - **Note:** `onnx_test_runner.exe` in this repo is compatible with ABI library

#### Unit Testing Application

**onnxruntime_provider_test** - Unit tests for QNN EP
```bash
onnxruntime_provider_test.exe --gtest_filter=Qnn*
```

### CI Tests

Run the following tests to ensure the PR can pass CI testing:

```bash
# Apply lintrunner
python qcom/build_and_test.py lint_and_fix

# Run all QNN tests
./onnxruntime_provider_test --gtest_filter=Qnn*
```

## Contributing

When contributing to the QNN Execution Provider:

1. **Fork the repository** on GitHub
2. **Create a feature branch** for your changes
3. **Follow coding guidelines** as outlined above
4. **Add tests** for new functionality
5. **Submit a pull request** with a clear description of changes
6. **Respond to review feedback** promptly

## Additional Resources

- [QNN Execution Provider Documentation](QNN-ExecutionProvider.md)
- [Build Instructions](build.md)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Qualcomm AI Engine Direct SDK Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/introduction.html)

## Support

For issues and questions:
- **GitHub Issues**: [onnxruntime-qnn/issues](https://github.com/onnxruntime/onnxruntime-qnn/issues)
- **ONNX Runtime Discussions**: [onnxruntime-qnn/discussions](https://github.com/onnxruntime/onnxruntime-qnn/discussions)
