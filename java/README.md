# ONNX Runtime QNN Android

This directory contains the Android build infrastructure for the ONNX Runtime QNN Execution Provider.

## Usage

This document pertains to developing, building, and testing the Android AAR and test application in your local environment.
For general purpose usage of the QNN Execution Provider, please see the [QNN EP documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md).

### Building

Use the main project's build script with the `--build_java` option.

#### Requirements

The [Gradle](https://gradle.org/) build system is used here to manage the Android project's dependency management, compilation, and assembly.
In particular, the Gradle [wrapper](https://docs.gradle.org/current/userguide/gradle_wrapper.html) at `java/gradlew[.bat]` is used, locking the Gradle version to the one specified in the `java/gradle/wrapper/gradle-wrapper.properties` configuration.
Using the Gradle wrapper removes the need to have the right version of Gradle installed on the system.

#### Build Output

The build will generate output in `$BUILD_DIR/java/`:

* `build/android-qnn-ep/outputs/aar/onnxruntime-android-qnn.aar` — the `onnxruntime-android-qnn` AAR containing both `libonnxruntime_providers_qnn.so` and the `qnnpluginep` Kotlin helpers
* `androidtest/android/app/build/outputs/apk/` - Test APKs

#### Build System Overview

The main CMake build system delegates building to Gradle.
This allows the CMake system to ensure all of the C/C++ compilation is achieved prior to the Android build.

When running the build script with `--build_java`, CMake will compile the `onnxruntime_providers_qnn` target and then invoke two Gradle builds:
1. `:qnnpluginep:bundleReleaseAar` (`build-android.gradle`) — packages `libonnxruntime_providers_qnn.so` and the `qnnpluginep` Kotlin helpers into the `onnxruntime-android-qnn` AAR.
2. `assembleDebug assembleDebugAndroidTest` (`src/test/android/`) — builds the Android instrumentation test APKs.

### qnnpluginep

`src/main/qnnpluginep/` is an Android library module (`ai.onnxruntime.qnnpluginep`) that is the source for the `onnxruntime-android-qnn` AAR.
It bundles `libonnxruntime_providers_qnn.so` (via `jniLibs.srcDirs`) together with Kotlin helper functions for registering the QNN Execution Provider at runtime.
The resulting AAR is the single deliverable that consumers depend on — it contains both the native library and the Kotlin helpers.
CMake passes `-DqnnpluginepBuildDir` to redirect the Gradle build output to `$BUILD_DIR/java/build/android-qnn-ep/`, keeping the source tree clean.

The module exposes the following top-level functions in `QnnPluginEpLibrary.kt`:

* `getLibraryPath()` - Returns the filename of the QNN EP shared library. This can be passed to `OrtEnvironment.registerExecutionProviderLibrary()`.
* `getEpName()` - Returns the EP name `"QNNExecutionProvider"`. Use this to filter `OrtEnvironment.epDevices` when selecting the QNN EP.
* `getEpNames()` - Returns an array of EP names exposed by the QNN EP library.
