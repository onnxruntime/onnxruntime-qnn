#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Builds the test APKs against a provided onnxruntime-android-qnn AAR file
# without polluting the source tree.  Run with --help for usage.

set -euo pipefail

AAR_PATH=""
QNN_VERSION="2.45.0"
ORT_VERSION="1.24.3"
OUTPUT_DIR="$PWD"
TEST_ASSETS_DIR=""

usage() {
    local stream="${1:-2}"   # 1 = stdout (user asked for help), 2 = stderr (wrong usage)
    cat >&"$stream" <<'EOF'
Builds the Android test APKs against a provided onnxruntime-android-qnn AAR
file.  Does NOT build the AAR.

Produces both APKs required to run instrumentation tests:
  - app-debug.apk              (the application under test)
  - app-debug-androidTest.apk  (the instrumentation test harness)

The source tree stays clean: the AAR is staged in a temp flatDir (via
--init-script) rather than app/libs/, Gradle's buildDir and project cache
are redirected to the temp dir, and the temp dir is removed on exit.

Usage:
  build_apk_from_aar.sh --aar PATH [--test-assets-dir DIR]
                        [--qnn-version VER] [--ort-version VER]
                        [--output-dir DIR]

Required:
  --aar PATH              Path to the onnxruntime-android-qnn AAR.

Optional:
  --test-assets-dir DIR   Directory containing test assets (e.g. sigmoid.ort).
                          Injected into the androidTest APK via Gradle sourceSets
                          without modifying the source tree.  If omitted, the
                          existing app/src/androidTest/assets/ is used (must
                          already contain sigmoid.ort).
  --android-sdk-root DIR  Android SDK root (default: $ANDROID_SDK_ROOT or $ANDROID_HOME).
  --qnn-version VER       QNN runtime version (default: 2.45.0).
  --ort-version VER       ORT Android version (default: 1.24.3).
  --output-dir DIR        Where to copy the built APKs (default: $PWD).
  --help                  Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --aar)              AAR_PATH="$2"; shift 2 ;;
        --test-assets-dir)  TEST_ASSETS_DIR="$2"; shift 2 ;;
        --android-sdk-root) ANDROID_SDK_ROOT="$2"; shift 2 ;;
        --qnn-version)      QNN_VERSION="$2"; shift 2 ;;
        --ort-version)      ORT_VERSION="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)          usage 1; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$AAR_PATH" ]]; then
    echo "ERROR: --aar is required" >&2
    usage
    exit 1
fi
if [[ ! -f "$AAR_PATH" ]]; then
    echo "ERROR: AAR file not found: $AAR_PATH" >&2
    exit 1
fi

# Resolve Android SDK root: --android-sdk-root > $ANDROID_SDK_ROOT > $ANDROID_HOME
ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
if [[ -z "$ANDROID_SDK_ROOT" ]]; then
    echo "ERROR: Android SDK root is not set." >&2
    echo "       Set ANDROID_SDK_ROOT / ANDROID_HOME, or pass --android-sdk-root." >&2
    exit 1
fi
if [[ ! -d "$ANDROID_SDK_ROOT" ]]; then
    echo "ERROR: Android SDK root does not exist: $ANDROID_SDK_ROOT" >&2
    exit 1
fi
export ANDROID_SDK_ROOT

AAR_ABS=$(readlink -f "$AAR_PATH")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(readlink -f "$OUTPUT_DIR")

REPO_ROOT=$(git rev-parse --show-toplevel)
ANDROID_DIR="${REPO_ROOT}/java/src/test/android"
GRADLEW="${REPO_ROOT}/java/gradlew"

if [[ ! -x "$GRADLEW" ]]; then
    echo "ERROR: Gradle wrapper not found or not executable: $GRADLEW" >&2
    exit 1
fi

TEMP_DIR=$(mktemp -d -t build-apk-from-aar.XXXXXX)
LOCAL_PROPS="${ANDROID_DIR}/local.properties"
LOCAL_PROPS_CREATED=false

cleanup() {
    rm -rf "$TEMP_DIR"
    if [[ "$LOCAL_PROPS_CREATED" == "true" ]]; then
        rm -f "$LOCAL_PROPS"
    fi
}
trap cleanup EXIT

# Write local.properties so AGP can locate the Android SDK.
# AGP reads this from the project root; there is no other way to inject sdk.dir
# without polluting the source tree.  We create the file only if it does not
# already exist, and delete it on exit.
if [[ ! -f "$LOCAL_PROPS" ]]; then
    echo "sdk.dir=${ANDROID_SDK_ROOT}" > "$LOCAL_PROPS"
    LOCAL_PROPS_CREATED=true
fi

# Validate and resolve test assets dir
if [[ -n "$TEST_ASSETS_DIR" ]]; then
    if [[ ! -d "$TEST_ASSETS_DIR" ]]; then
        echo "ERROR: --test-assets-dir does not exist: $TEST_ASSETS_DIR" >&2
        exit 1
    fi
    if [[ ! -f "${TEST_ASSETS_DIR}/sigmoid.ort" ]]; then
        echo "ERROR: sigmoid.ort not found in --test-assets-dir: $TEST_ASSETS_DIR" >&2
        exit 1
    fi
    TEST_ASSETS_DIR=$(readlink -f "$TEST_ASSETS_DIR")
fi

# Stage the AAR in a temp flatDir so Gradle's flatDir resolver can find it
# under the artifact name 'onnxruntime-android-qnn'.
LIBS_DIR="${TEMP_DIR}/libs"
mkdir -p "$LIBS_DIR"
cp "$AAR_ABS" "${LIBS_DIR}/onnxruntime-android-qnn.aar"

# Init script: merge our temp AAR dir into the existing flatDir (combining with
# the original 'libs' dir avoids a second flatDir registration and the resulting
# "flatDir2" warning), redirect every project's buildDir out of the source
# tree, and optionally inject test assets.  Unquoted heredoc is intentional:
# shell variables are expanded into the Groovy source
# (mktemp output is always /tmp/…/<suffix> — no special chars).
INIT_SCRIPT="${TEMP_DIR}/init.gradle"
BUILD_ROOT="${TEMP_DIR}/gradle-build"

# Build the optional assets sourceSets block — only emitted when --test-assets-dir is set.
if [[ -n "$TEST_ASSETS_DIR" ]]; then
    ASSETS_BLOCK="
    afterEvaluate { proj ->
        if (proj.plugins.hasPlugin('com.android.application')) {
            proj.android.sourceSets.androidTest.assets.srcDirs += ['${TEST_ASSETS_DIR}']
        }
    }"
else
    ASSETS_BLOCK=""
fi

cat > "$INIT_SCRIPT" <<EOF
allprojects {
    repositories {
        // Merge the temp AAR dir with the original 'libs' dir into one flatDir
        // entry to avoid Gradle registering a duplicate (flatDir2) repository.
        flatDir { dirs '${LIBS_DIR}', 'libs' }
    }
    buildDir = new File('${BUILD_ROOT}', project.name)${ASSETS_BLOCK}
}
EOF

PROJECT_CACHE_DIR="${TEMP_DIR}/gradle-cache"

# Set GRADLE_USER_HOME to avoid using ~/.gradle which may be prohibited
GRADLE_USER_HOME="${TEMP_DIR}/gradle-home"
export GRADLE_USER_HOME

# Ensure we build against the local flatDir AAR, not a Maven coord.
unset AAR_MAVEN_COORD

"$GRADLEW" -p "$ANDROID_DIR" \
    --init-script "$INIT_SCRIPT" \
    --project-cache-dir "$PROJECT_CACHE_DIR" \
    -PskipLibsFlatDir=true \
    -DqnnVersion="$QNN_VERSION" \
    -DortVersion="$ORT_VERSION" \
    assembleDebug assembleDebugAndroidTest \
    --no-daemon --console=plain

APK_DEBUG="${BUILD_ROOT}/app/outputs/apk/debug/app-debug.apk"
APK_TEST="${BUILD_ROOT}/app/outputs/apk/androidTest/debug/app-debug-androidTest.apk"

for apk in "$APK_DEBUG" "$APK_TEST"; do
    if [[ ! -f "$apk" ]]; then
        echo "ERROR: Expected APK not found: $apk" >&2
        exit 1
    fi
done

cp -f "$APK_DEBUG" "${OUTPUT_DIR}/app-debug.apk"
cp -f "$APK_TEST"  "${OUTPUT_DIR}/app-debug-androidTest.apk"

echo "APK build succeeded:"
echo "  ${OUTPUT_DIR}/app-debug.apk"
echo "  ${OUTPUT_DIR}/app-debug-androidTest.apk"
