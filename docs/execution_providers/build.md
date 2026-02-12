# Building QNN EP as ABI-Compatible Plugin Library

## Overview

The QNN Execution Provider can be built as an ABI-compatible plugin library that works with pre-built ONNX Runtime releases. This approach provides better compatibility and allows you to use the QNN EP with official ONNX Runtime distributions.

See more information on the QNN execution provider [here](./QNN-ExecutionProvider.md).

## Prerequisites

### Required Software

* **Qualcomm AI Engine Direct SDK (QAIRT)**: Download from [Qualcomm Package Manager](https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct)

* **CMake 3.28 or higher**: Download from [cmake.org](https://cmake.org/download/)

* **Python 3.10 or higher**:
  * [Python 3.12 for Windows Arm64](https://www.python.org/ftp/python/3.12.9/python-3.12.9-arm64.exe)
  * [Python 3.12 for Windows x86-64](https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe)
  * **Note**: Windows on Arm supports x86-64 Python via emulation. Ensure the Arm64 Python environment is activated for native Arm64 builds.

* **Pre-built ONNX Runtime Release**: Download from [Microsoft ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
  * Extract the release package and set `ORT_HOME` to the root directory
  * Example: `onnxruntime-win-arm64-1.24.1`

### Get the Source Code

Clone the onnxruntime-qnn repository:

```bash
git clone --recursive https://github.com/onnxruntime/onnxruntime-qnn.git
cd onnxruntime-qnn
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Build Options

### Required Options

* `--build_shared_lib`: Build as a shared library (required for ABI plugin)
* `--use_qnn`: Enable QNN Execution Provider
* `--qnn_home QNN_SDK_PATH`: Path to the Qualcomm AI Engine Direct SDK
  * Example: `--qnn_home "C:\Qualcomm\AIStack\QAIRT\2.42.0.251225"`
* `--ort_home ORT_SDK_PATH`: Path to the pre-built ONNX Runtime SDK
  * Example: `--ort_home "C:\onnxruntime-win-arm64-1.24.1"`
  * **Note**: `--ort_home` is not supported on `rel-2.0.0` branch

### Optional Options

* `--parallel`: Enable parallel build (recommended)
* `--build_wheel`: Build Python wheel package
* `--skip_tests`: Skip building tests
* `--config [Release/RelWithDebInfo/Debug]`: Build configuration (default: Release)
* `--build_dir BUILD_PATH`: Specify build output directory
* `--arm64`: Cross-compile for Arm64 (Windows only)
* `--cmake_generator GENERATOR`: Specify CMake generator (e.g., "Visual Studio 17 2022", "Ninja")

Run `python tools/ci_build/build.py --help` for a complete list of build options.

## Build Instructions

### Windows (native x86-64 or native Arm64)

```batch
.\build.bat --build_shared_lib --parallel --cmake_generator "Visual Studio 17 2022" --use_qnn --config Release --qnn_home [QNN_SDK_PATH] --ort_home [ORT_SDK_PATH] --build_dir .\build
```

**Notes:**
* Not all Qualcomm backends (e.g., HTP) support model execution on x86-64. Refer to the [Qualcomm SDK backend documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/backend.html) for details.
* Even if a backend doesn't support x86-64 execution, the QNN EP can still [generate compiled models](../execution-providers/QNN-ExecutionProvider.md#qnn-context-binary-cache-feature) for that backend.

### Windows (Arm64 cross-compile target)

```batch
.\build.bat --arm64 --build_shared_lib --parallel --cmake_generator "Visual Studio 17 2022" --use_qnn --build_wheel --skip_tests --config Release --qnn_home [QNN_SDK_PATH] --ort_home [ORT_SDK_PATH] --build_dir .\build
```

## Build Artifacts

After a successful build, the artifacts will be located in the build directory:

### ABI EP Plugin Library

* **Release**: `$BUILD_DIR\Release\Release\onnxruntime_provider_qnn.dll`

### Python Wheel (if `--build_wheel` was used)

* `$BUILD_DIR/Release/Release/dist/onnxruntime_qnn-[version]-py3-none-[platform].whl`

Example filenames:
* Windows Arm64: `onnxruntime_qnn-2.0.0-py3-none-win_arm64.whl`
* Windows x86-64: `onnxruntime_qnn-2.0.0-py3-none-win_amd64.whl`

## Troubleshooting

### Common Build Issues

**Issue**: CMake cannot find ONNX Runtime headers
* **Solution**: Verify that `ORT_HOME` points to the correct directory containing the `include` folder

**Issue**: CMake cannot find QNN SDK
* **Solution**: Verify that `QNN_HOME` points to the correct QAIRT SDK directory

**Issue**: Python version mismatch
* **Solution**: Ensure you're using Python 3.10 or higher. On Windows Arm64, use the native Arm64 Python installer.

**Issue**: Build fails with "Visual Studio not found"
* **Solution**: Install Visual Studio 2022 with C++ development tools, or specify a different generator with `--cmake_generator`

### Platform-Specific Notes

**Windows on Arm**:
* Use the native Arm64 Python environment for best performance
* x86-64 Python works via emulation but may have performance implications
