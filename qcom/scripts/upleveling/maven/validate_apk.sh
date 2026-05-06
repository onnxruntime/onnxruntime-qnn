#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Validates a published AAR artifact by building the existing Android test APKs
# against it.  The APK project is at java/src/test/android/app/.
#
# Builds both APKs required for instrumentation tests:
#   - app-debug.apk              (the application under test)
#   - app-debug-androidTest.apk  (the instrumentation test harness)
#
# The source directory stays clean:
#   - Gradle's buildDir and project cache are redirected to a temp dir
#   - local.properties is written only if it does not already exist, and
#     deleted on exit
#   - the temp dir is removed on exit (the APKs are not retained; this is a
#     pass/fail gate, not a build artifact producer)
#
# Required environment variables:
#   AAR_MAVEN_COORD                 e.g. com.qualcomm.qti:onnxruntime-android-qnn:2.3.0-SNAPSHOT
#   AISW_MAVEN_ARTIFACTORY_USERNAME
#   AISW_MAVEN_ARTIFACTORY_PASSWORD
#   ANDROID_SDK_ROOT or ANDROID_HOME
#
# Optional environment variables:
#   QNN_VERSION       QNN runtime version (default: 2.45.0).
#   ORT_VERSION       ORT Android version (default: 1.24.3).
#
# The script is intentionally kept simple — it fails fast if the AAR is not
# consumable as a dependency, which proves the published artifact is well-formed.

set -euo pipefail

: "${AAR_MAVEN_COORD:?AAR_MAVEN_COORD is required}"
: "${AISW_MAVEN_ARTIFACTORY_USERNAME:?AISW_MAVEN_ARTIFACTORY_USERNAME is required}"
: "${AISW_MAVEN_ARTIFACTORY_PASSWORD:?AISW_MAVEN_ARTIFACTORY_PASSWORD is required}"

QNN_VERSION="${QNN_VERSION:-2.45.0}"
ORT_VERSION="${ORT_VERSION:-1.24.3}"

# Resolve Android SDK root: $ANDROID_SDK_ROOT > $ANDROID_HOME
ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
if [[ -z "${ANDROID_SDK_ROOT}" ]]; then
    echo "ERROR: Android SDK root is not set." >&2
    echo "       Set ANDROID_SDK_ROOT or ANDROID_HOME." >&2
    exit 1
fi
if [[ ! -d "${ANDROID_SDK_ROOT}" ]]; then
    echo "ERROR: Android SDK root does not exist: ${ANDROID_SDK_ROOT}" >&2
    exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
ANDROID_DIR="${REPO_ROOT}/java/src/test/android"
GRADLEW="${REPO_ROOT}/java/gradlew"
LOCAL_PROPS="${ANDROID_DIR}/local.properties"
LOCAL_PROPS_CREATED=false

if [[ ! -d "${ANDROID_DIR}" ]]; then
    echo "ERROR: Android test project directory not found: ${ANDROID_DIR}" >&2
    exit 1
fi

if [[ ! -x "${GRADLEW}" ]]; then
    echo "ERROR: Gradle wrapper not found or not executable: ${GRADLEW}" >&2
    exit 1
fi

TEMP_DIR=$(mktemp -d -t validate-apk.XXXXXX)
TRUSTSTORE_PATH=""
TRUSTSTORE_PASSWORD=""

cleanup() {
    rm -rf "${TEMP_DIR}"
    if [[ "${LOCAL_PROPS_CREATED}" == "true" ]]; then
        rm -f "${LOCAL_PROPS}"
    fi
}
trap cleanup EXIT

# Write local.properties so AGP can locate the Android SDK.
# Created only if it does not already exist; deleted on exit.
if [[ ! -f "${LOCAL_PROPS}" ]]; then
    echo "sdk.dir=${ANDROID_SDK_ROOT}" > "${LOCAL_PROPS}"
    LOCAL_PROPS_CREATED=true
fi

echo "Validating AAR: ${AAR_MAVEN_COORD}"
echo "Building APKs in: ${ANDROID_DIR}"

# Pass the Maven coord and Artifactory credentials to Gradle via env vars.
# build.gradle reads AAR_MAVEN_COORD to decide whether to pull from Artifactory.
export AAR_MAVEN_COORD
export AISW_MAVEN_ARTIFACTORY_USERNAME
export AISW_MAVEN_ARTIFACTORY_PASSWORD

# Set GRADLE_USER_HOME to avoid using ~/.gradle which may be prohibited
GRADLE_USER_HOME="${TEMP_DIR}/gradle-home"
export GRADLE_USER_HOME

# Init script: redirect every project's buildDir out of the source tree.
# Unquoted heredoc is intentional: ${BUILD_ROOT} is expanded by the shell into
# the Groovy source (mktemp output is always /tmp/…/<suffix> — no special chars).
INIT_SCRIPT="${TEMP_DIR}/init.gradle"
BUILD_ROOT="${TEMP_DIR}/gradle-build"
cat > "${INIT_SCRIPT}" <<EOF
allprojects {
    buildDir = new File('${BUILD_ROOT}', project.name)
}
EOF

PROJECT_CACHE_DIR="${TEMP_DIR}/gradle-cache"

# Set up SSL truststore with Qualcomm Artifactory CA certificate
ARTIFACTORY_CA_PATH="${REPO_ROOT}/qcom/scripts/upleveling/certs/artifactory-ca.pem"
# Probe known CA bundle locations in order; Debian/Ubuntu first, then RHEL/Fedora variants.
_SYSTEM_CA_CANDIDATES=(
    "/etc/ssl/certs/ca-certificates.crt"
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"
    "/etc/pki/tls/certs/ca-bundle.crt"
    "/etc/ssl/cert.pem"
    "/var/lib/ca-certificates/ca-bundle.pem"
)
SYSTEM_CA_BUNDLE=""
for _ca_candidate in "${_SYSTEM_CA_CANDIDATES[@]}"; do
    if [[ -f "${_ca_candidate}" ]]; then
        SYSTEM_CA_BUNDLE="${_ca_candidate}"
        break
    fi
done
if [[ -z "${SYSTEM_CA_BUNDLE}" ]]; then
    echo "WARNING: system CA bundle not found at any known path; Maven Central SSL may fail" >&2
fi

# Write JVM SSL flags to a Gradle JVM config file instead of GRADLE_OPTS to
# avoid exposing the truststore password in the process environment.
GRADLE_JVM_CONFIG=""
if [[ -f "${ARTIFACTORY_CA_PATH}" ]]; then
    TRUSTSTORE_PATH="${TEMP_DIR}/truststore.p12"
    TRUSTSTORE_PASSWORD=$(openssl rand -hex 16)
    CERTS_DIR="${TEMP_DIR}/certs"
    mkdir -p "${CERTS_DIR}"

    # Split system CA bundle into individual certificates
    if [[ -n "${SYSTEM_CA_BUNDLE}" ]]; then
        awk -v certs_dir="${CERTS_DIR}" '
            /BEGIN CERTIFICATE/ { cert++; in_cert=1 }
            in_cert             { print > certs_dir"/cert-"cert".pem" }
            /END CERTIFICATE/   { in_cert=0 }
        ' "${SYSTEM_CA_BUNDLE}"

        # Import each certificate; keep || true so one bad cert doesn't abort the
        # loop, but redirect stderr to a log file so failures are diagnosable.
        KEYTOOL_LOG="${TEMP_DIR}/keytool-errors.log"
        for cert_file in "${CERTS_DIR}"/cert-*.pem; do
            if [[ -f "${cert_file}" ]]; then
                cert_num=$(basename "${cert_file}" .pem)
                keytool -import -noprompt -trustcacerts -alias "system-${cert_num}" -file "${cert_file}" \
                    -storetype PKCS12 -keystore "${TRUSTSTORE_PATH}" -storepass "${TRUSTSTORE_PASSWORD}" \
                    2>>"${KEYTOOL_LOG}" || true
            fi
        done
    fi

    # Import Qualcomm CA
    keytool -import -noprompt -trustcacerts -alias qualcomm-ca -file "${ARTIFACTORY_CA_PATH}" \
        -storetype PKCS12 -keystore "${TRUSTSTORE_PATH}" -storepass "${TRUSTSTORE_PASSWORD}" 2>/dev/null

    if [[ -f "${TRUSTSTORE_PATH}" ]]; then
        CERT_COUNT=$(keytool -list -keystore "${TRUSTSTORE_PATH}" -storepass "${TRUSTSTORE_PASSWORD}" 2>/dev/null | grep -c "trustedCertEntry" || echo 0)
        echo "Built PKCS12 truststore with ${CERT_COUNT} certificates"
        # A healthy system CA bundle contains well over 100 certs.  Fewer than 50
        # suggests wholesale import failure (corrupt truststore, JDK incompatibility,
        # or a completely empty system bundle).  Fail early rather than letting Maven
        # discover the problem as an SSL handshake error later.
        MIN_EXPECTED_CERTS=50
        if (( CERT_COUNT < MIN_EXPECTED_CERTS )); then
            echo "ERROR: only ${CERT_COUNT} certs imported (expected >= ${MIN_EXPECTED_CERTS}); see ${KEYTOOL_LOG:-${TEMP_DIR}/keytool-errors.log} for details" >&2
            exit 1
        fi
        # Write SSL flags to gradle.properties under GRADLE_USER_HOME so the
        # password is not visible in `ps` output or the process environment.
        # Pre-create the file as 600 before writing so the password is never
        # readable by other users even transiently (matches _secure_tempfile()
        # in maven_publish_utils.py).  The file lives under TEMP_DIR which is
        # removed on EXIT trap; SIGKILL survivors are bounded to TEMP_DIR.
        mkdir -p "${GRADLE_USER_HOME}"
        install -m 600 /dev/null "${GRADLE_USER_HOME}/gradle.properties"
        cat >> "${GRADLE_USER_HOME}/gradle.properties" <<EOF
org.gradle.jvmargs=-Djavax.net.ssl.trustStore=${TRUSTSTORE_PATH} -Djavax.net.ssl.trustStoreType=PKCS12 -Djavax.net.ssl.trustStorePassword=${TRUSTSTORE_PASSWORD}
EOF
    else
        echo "WARNING: Failed to build PKCS12 truststore"
    fi
else
    echo "WARNING: Qualcomm CA cert not found at ${ARTIFACTORY_CA_PATH}"
fi
"${GRADLEW}" -p "${ANDROID_DIR}" \
    --init-script "${INIT_SCRIPT}" \
    --project-cache-dir "${PROJECT_CACHE_DIR}" \
    -PskipLibsFlatDir=true \
    -DqnnVersion="${QNN_VERSION}" \
    -DortVersion="${ORT_VERSION}" \
    assembleDebug assembleDebugAndroidTest \
    --no-daemon --console=plain

APK_DEBUG="${BUILD_ROOT}/app/outputs/apk/debug/app-debug.apk"
APK_TEST="${BUILD_ROOT}/app/outputs/apk/androidTest/debug/app-debug-androidTest.apk"

for apk in "${APK_DEBUG}" "${APK_TEST}"; do
    if [[ ! -f "${apk}" ]]; then
        echo "ERROR: Expected APK not found: ${apk}" >&2
        exit 1
    fi
done

echo "APK build succeeded — published AAR is consumable."
