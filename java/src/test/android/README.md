# Android Test Application for ONNX Runtime QNN

This directory contains a simple Android application for testing the [ONNX Runtime QNN Execution Provider](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md).

## Background

For general usage and build instructions for ONNX Runtime Android, please see the [documentation](https://onnxruntime.ai/docs/tutorials/mobile/) here.

### Test Android Application Overview

This Android application is mainly aimed for testing:

- Model used: A simple [sigmoid ONNX model](https://github.com/onnx/onnx/blob/f9b0cc99344869c246b8f4011b8586a39841284c/onnx/backend/test/data/node/test_sigmoid/model.onnx) (converted to ORT format under `app/src/androidTest/assets` folder).
    - Here's [documentation](https://onnxruntime.ai/docs/reference/ort-format-models.html#convert-onnx-models-to-ort-format) about how you can convert an ONNX model into ORT format.
    - Run `python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style=Fixed /path/to/model.onnx` and rename the resulting .ort file accordingly.
- Main test file: An Android instrumentation test under `app/src/androidTest/java/ai/onnxruntime/example/javavalidator/SimpleTest.kt`
- The main dependencies of this application are the `onnxruntime-android` AAR from Maven Central, the `qnn-runtime` AAR from Maven Central, and the locally-built `onnxruntime-android-qnn` AAR. The `onnxruntime-android-qnn` AAR contains both `libonnxruntime_providers_qnn.so` and the `qnnpluginep` Kotlin helpers for registering the QNN EP at runtime.
- The MainActivity of this application is set to be empty.

### onnxruntime-android-qnn AAR

The `onnxruntime-android-qnn` AAR is built from the `qnnpluginep` module (`java/src/main/qnnpluginep/`) during the CMake build.
It bundles both `libonnxruntime_providers_qnn.so` and the Kotlin helper functions used by `SimpleTest.kt` to register the QNN EP at runtime:
`getLibraryPath()` returns the shared library filename passed to `OrtEnvironment.registerExecutionProviderLibrary()`,
and `getEpName()` returns `"QNNExecutionProvider"` used to select the QNN EP device.
CMake copies the built AAR to `app/libs/onnxruntime-android-qnn.aar` before the test APK build.

### Requirements

- JDK version 11 or later is required.
- The [Gradle](https://gradle.org/) build system is required for building the APKs used to run [Android instrumentation tests](https://source.android.com/compatibility/tests/development/instrumentation). Version 7.5 or newer is required.
  The Gradle wrapper at `java/gradlew[.bat]` may be used.

### Building

Use the main project's build script with `--build_java`.

Please note that you may need to set the `--android_abi=x86_64` (the default option is `arm64-v8a`). This is because Android instrumentation tests are run on an Android emulator which requires an ABI of `x86_64`.

#### QNN Builds
We use two AndroidManifest.xml files to manage different runtime requirements for QNN support. In the [build configuration](app/build.gradle), we specify which manifest file to use based on the build type.
In the [QNN manifest](app/src/main/AndroidManifestQnn.xml), we include the `<uses-native-library>` declaration for `libcdsprpc.so`, which is required for devices using QNN and Qualcomm DSP capabilities.
For QNN builds, it is also necessary to set the `ADSP_LIBRARY_PATH` environment variable to the [native library directory](https://developer.android.com/reference/android/content/pm/ApplicationInfo#nativeLibraryDir) depending on the device. This ensures that any native libraries downloaded as dependencies such as QNN libraries are found by the application. This is conditionally added by using the BuildConfig field `IS_QNN_BUILD` set in the build.gradle file.

#### Build Output

The build will generate two APKs required to run the test application in `$YOUR_BUILD_DIR/java/androidtest/android/app/build/outputs/apk`:

* `androidtest/debug/app-debug-androidtest.apk`
* `debug/app-debug.apk`

After running the build script, the two APKs will be installed on the `ort_android` emulator and it will automatically run the test application in an adb shell.
