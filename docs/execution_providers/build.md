# Building QNN EP as ABI-Compatible Plugin Library

## Overview

The QNN Execution Provider can be built as an ABI-compatible plugin library that works with pre-built ONNX Runtime releases. This approach provides better compatibility and allows you to use the QNN EP with official ONNX Runtime distributions.

See more information on the QNN execution provider [here](./QNN-ExecutionProvider.md).
- **Note**: QNN EP version < 2.0 is **NOT** included in this page, please refer to [build page](https://onnxruntime.ai/docs/build/eps.html#qnn) for QNN EP version < 2.0.

## Prerequisites

### Required Software

* **Python 3.10 or higher** (Required):
  * [Python 3.12 for Windows Arm64](https://www.python.org/ftp/python/3.12.9/python-3.12.9-arm64.exe)
  * [Python 3.12 for Windows x86-64](https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe)
  * **Note**: Windows on Arm supports x86-64 Python via emulation. Ensure the Arm64 Python environment is activated for native Arm64 builds.

* **Visual Studio 2022 Professional** (Required for Windows)
  * [Visual Studio & VS Code Downloads for Windows, Mac, Linux](https://visualstudio.microsoft.com/downloads/)

* **Qualcomm AI Engine Direct SDK (QAIRT)** (optional): Download from [Qualcomm Package Manager](https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct)
  * **Note**: The build will fetch the appropriate QAIRT version automatically. If you have a local copy, specify its location using `--qairt-sdk` argument or `QAIRT_SDK_PATH` environment variable. The SDK must be version 2.17.0 or higher. For older versions, see [Build Instructions for QNN EP v1.x](https://onnxruntime.ai/docs/build/eps.html#qnn) for details.

* **Pre-built ONNX Runtime Release** (Optional): Download from [Microsoft ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
  * **Note**: The build will fetch the appropriate ORT version automaticall. If you have a local copy, extract the release package and set the path via `--ort-prebuilt` argument or `ORT_PREBUILT_ROOT` environment variable to specify the root directory of the prebuilt ORT package.

### Get the Source Code

Clone the onnxruntime-qnn repository:

```bash
git clone --recursive https://github.com/onnxruntime/onnxruntime-qnn.git
cd onnxruntime-qnn
```

## Build System Overview

The build system uses a task-based approach via `python qcom/build_and_test.py`. Each task represents a specific build or test operation, and tasks can have dependencies that are automatically resolved.

### Common Arguments

* `--config [Release/RelWithDebInfo/Debug]`: Build configuration (default: Release)
* `--qairt-sdk PATH`: Path to QAIRT SDK (overrides environment variables)
* `--ort-prebuilt PATH`: Path to pre-built ONNX Runtime SDK (optional)
* `--target-py-version [3.10/3.11/3.12/3.13/3.14]`: Python version for wheel building (default: 3.12 on Windows, 3.10 on Linux)
* `--build-nuget`: Enable building NuGet packages for .NET bindings
* `--build-zip`: Enable building Zip archives
* `--venv-path PATH`: Virtual environment path (default: ./venv)
* `--dry-run`: Print the build plan without executing
* `--only`: Run only specified tasks, skipping dependencies

### Environment Variables

The build system supports the following environment variables:

* `QAIRT_SDK_ROOT` / `QNN_SDK_ROOT` / `SNPE_ROOT`: Path to QAIRT SDK (searched in this order)
* `ORT_PREBUILT_ROOT`: Path to pre-built ONNX Runtime
* `ORT_BUILD_DOCKER_CCACHE_ROOT`: Docker ccache root directory
* `ORT_NIGHTLY_BUILD`: Set to 1 for nightly builds
* `ANDROID_HOME` / `ANDROID_NDK_HOME`: Android SDK/NDK paths
* `JAVA_HOME`: Java installation path

### Listing Available Tasks

To see all available tasks:

```bash
python qcom/build_and_test.py list
```

To see all tasks including internal ones:

```bash
python qcom/build_and_test.py list_all
```

## Build Instructions

### Windows Builds

#### Build for Host Architecture (Auto-detect)

Automatically detects whether you're on x86-64 or ARM64 and builds accordingly:

```batch
python qcom/build_and_test.py [--config <BUILD_CONFIG>] [--qairt-sdk <QNN_SDK_PATH>] [--ort-prebuilt <ORT_SDK_PATH>] [--target-py-version <PY_VERSION>] build
```

#### Build for ARM64 Windows

```batch
python qcom/build_and_test.py build_ort_windows_arm64
```

#### Build for ARM64EC Windows

```batch
python qcom/build_and_test.py build_ort_windows_arm64ec
```

#### Build for ARM64X Windows (Hybrid ARM64/ARM64EC)

```batch
python qcom/build_and_test.py build_ort_windows_arm64x
```

**Note**: ARM64X builds create both ARM64 and ARM64EC slices in a single binary.

#### Build for x86-64 Windows

```batch
python qcom/build_and_test.py build_ort_windows_x86_64
```

**Note**: Not all Qualcomm backends (e.g., HTP) support model execution on x86-64. Refer to the [Qualcomm SDK backend documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/backend.html) for details. However, the QNN EP can still [generate compiled models](./QNN-ExecutionProvider.md#qnn-context-binary-cache-feature) for backends that don't support x86-64 execution.

### Linux Builds

#### Build for Host Architecture (Auto-detect)

```bash
python qcom/build_and_test.py build
```

#### Build for x86-64 Linux

```bash
python qcom/build_and_test.py build_ort_linux_x86_64
```

#### Build for AArch64 Linux (OpenEmbedded GCC 11.2)

For Qualcomm Linux devices:

```bash
python qcom/build_and_test.py build_ort_linux_aarch64_oe_gcc11_2
```

### Android Builds

#### Build for Android AArch64

```bash
python qcom/build_and_test.py build_ort_android_aarch64
```

**Requirements**:
* Set `ANDROID_HOME` and `ANDROID_NDK_HOME` environment variables, or
* The build system will install known good versions automatically

## Testing

### Windows Testing

#### Test on Host Architecture

```batch
python qcom/build_and_test.py test
```

This automatically runs tests for your host architecture (ARM64 or x86-64).

#### Test Specific Architectures

```batch
# Test ARM64 build
python qcom/build_and_test.py test_ort_windows_arm64

# Test x86-64 build
python qcom/build_and_test.py test_ort_windows_x86_64

# Test ARM64EC build
python qcom/build_and_test.py test_ort_windows_arm64ec
```

### Linux Testing

```bash
# Test on host
python qcom/build_and_test.py test

# Test x86-64 build
python qcom/build_and_test.py test_ort_linux_x86_64
```

### Device Testing

#### Test on Locally Connected Devices

For Android devices connected via ADB:

```bash
python qcom/build_and_test.py test_ort_local_android_aarch64
```

For Qualcomm Linux devices connected via ADB:

```bash
python qcom/build_and_test.py test_ort_local_linux_aarch64_oe_gcc11_2
```

#### Test in Qualcomm Device Cloud (QDC)

**Requirements**: Set `QDC_API_TOKEN` environment variable with your QDC API token.

```bash
# Test Android in QDC
python qcom/build_and_test.py test_ort_qdc_android_aarch64

# Test Qualcomm Linux in QDC
python qcom/build_and_test.py test_ort_qdc_linux_aarch64_oe_gcc11_2

# Test Windows ARM64 in QDC
python qcom/build_and_test.py test_ort_qdc_windows_arm64
```

## Build Artifacts

### Directory Structure

Build artifacts are organized by configuration in the `build` directory:

```
build/
├── Release/
│   ├── Release/                          # Windows artifacts
│   │   ├── onnxruntime_providers_qnn.dll # QNN EP plugin
│   │   ├── onnxruntime.dll               # Main runtime (if built)
│   │   └── dist/                         # Python wheels
│   │       └── onnxruntime_qnn-*.whl
│   └── libonnxruntime_providers_qnn.so   # Linux artifacts
└── Debug/
    └── ...
```

### Artifact Types

#### ABI EP Plugin Library

* **Windows**: `build/Release/Release/onnxruntime_provider_qnn.dll`
* **Linux**: `build/Release/libonnxruntime_providers_qnn.so`

#### Python Wheels

Located in `build/Release/Release/dist/` (Windows) or `build/Release/dist/` (Linux):

* Windows ARM64: `onnxruntime_qnn-[version]-cp[py_version]-cp[py_version]-win_arm64.whl`
* Windows x86-64: `onnxruntime_qnn-[version]-cp[py_version]-cp[py_version]-win_amd64.whl`
* Linux x86-64: `onnxruntime_qnn-[version]-cp[py_version]-cp[py_version]-linux_x86_64.whl`
* Linux AArch64: `onnxruntime_qnn-[version]-cp[py_version]-cp[py_version]-manylinux_2_34_aarch64.whl`

#### NuGet Packages

Located in `build/Release/Release/nuget-local-artifacts/` (Windows):

* `Qualcomm.ML.OnnxRuntime.QNN.[version].nupkg`

#### Zip Archives

Located in `build/`:

* `onnxruntime-[platform]-[arch]-[version].zip`

## Advanced Usage

### Dry Run Mode

To see what would be built without actually building:

```batch
python qcom/build_and_test.py --dry-run build_ort_windows_arm64
```

### Skipping Dependencies

To build only a specific task without its dependencies:

```batch
python qcom/build_and_test.py --only build_ort_windows_arm64
```

**Warning**: This may fail if dependencies haven't been built previously.

### Task Dependency Graph

To visualize the task dependency graph in DOT format:

```batch
python qcom/build_and_test.py --print-task-graph build_ort_windows_arm64 > graph.dot
```

You can then render this with Graphviz:

```batch
dot -Tpng graph.dot -o graph.png
```

## Code Quality

### Running the Linter

```bash
python qcom/build_and_test.py lint
```

### Auto-fixing Linter Issues

```bash
python qcom/build_and_test.py lint_and_fix
```

## Troubleshooting

### Common Build Issues

**Issue**: Task not found
* **Solution**: Run `python qcom/build_and_test.py list` to see available tasks for your platform

**Issue**: QAIRT SDK not found
* **Solution**: Verify `--qairt-sdk` path or set `QAIRT_SDK_ROOT` / `QNN_SDK_ROOT` environment variable

**Issue**: ORT prebuilt not found
* **Solution**: Either provide `--ort-prebuilt` path or let the build fetch it automatically (omit the argument)

**Issue**: Python version mismatch
* **Solution**: Ensure you're using Python 3.10 or higher. Use `--target-py-version` to specify the target Python version for wheel builds

**Issue**: Build fails with dependency errors
* **Solution**: Don't use `--only` flag unless you're sure dependencies are already built

**Issue**: Docker build fails on Linux
* **Solution**: Ensure Docker is installed and your user has permission to run Docker commands

### Platform-Specific Notes

**Windows on Arm**:
* Use native Arm64 Python for best performance
* x86-64 Python works via emulation but may be slower
* Visual Studio 2022 with C++ development tools is required

**Linux**:
* Docker is required for manylinux builds
* For Android builds, set `ANDROID_HOME` and `ANDROID_NDK_HOME`

### Getting Help

For additional help:

1. Check the [QNN Execution Provider documentation](./QNN-ExecutionProvider.md)
2. Run `python qcom/build_and_test.py --help` for detailed argument information
3. Use `--dry-run` to preview what will be built
4. Check the build logs in the `build` directory
