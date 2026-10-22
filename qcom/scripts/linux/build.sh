#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"
source "${REPO_ROOT}/qcom/scripts/linux/tools.sh"

set_strict_mode

function update_needed() {
  if [ ! -f "${1}/build.ninja" ]; then
    log_debug "${1}/build.ninja does not exist"
    echo "1"
  elif [ ! -f "${qairt_sdk_file_path}" ]; then
    log_debug "No record of previous QAIRT SDK"
    echo "1"
  elif [ "$(cat "${qairt_sdk_file_path}")" != "${2}" ]; then
    log_debug "New QAIRT SDK detected"
    echo "1"
  elif [ ! -f "${qnn_build_mode_file_path}" ]; then
    log_debug "No record of previous QNN build mode"
    echo "1"
  elif [ "$(cat "${qnn_build_mode_file_path}")" != "${3}" ]; then
    log_debug "QNN coverage/internal-symbol build mode changed"
    echo "1"
  else
    log_debug "No update needed"
  fi
}

function save_qairt_sdk_path() {
  echo "${1}" > "${qairt_sdk_file_path}"
}

function save_qnn_build_mode() {
  echo "${1}" > "${qnn_build_mode_file_path}"
}

config="Release"
qairt_sdk_root=
ort_prebuilt_root=
qnn_arch_abi=
target_py_version=
use_cache=1
warnings_as_errors=1
build_java=
build_archive=
enable_coverage=
enable_qnn_internal_ut_symbols=
enable_asan=
build_dir_suffix=
for i in "$@"; do
  case $i in
    --build-archive)
      build_archive=1
      shift
      ;;
    --enable-asan)
      enable_asan=1
      shift
      ;;
    --config=*)
      config="${i#*=}"
      shift
      ;;
    --mode=*)
      mode="${i#*=}"
      shift
      ;;
    --no-use-cache)
      use_cache=
      shift
      ;;
    --no-warnings-as-errors)
      warnings_as_errors=
      shift
      ;;
    --build-java)
      build_java=1
      shift
      ;;
    --enable-coverage)
      enable_coverage=1
      shift
      ;;
    --enable-qnn-internal-ut-symbols)
      enable_qnn_internal_ut_symbols=1
      shift
      ;;
    --build-dir-suffix=*)
      build_dir_suffix="${i#*=}"
      shift
      ;;
    --ort-home=*)
      ort_prebuilt_root="${i#*=}"
      shift
      ;;
    --qairt-sdk-root=*)
      qairt_sdk_root="${i#*=}"
      shift
      ;;
    --qnn-arch-abi=*)
      qnn_arch_abi="${i#*=}"
      shift
      ;;
    --target-arch=*)
      target_arch="${i#*=}"
      shift
      ;;
    --target-platform=*)
      target_platform="${i#*=}"
      shift
      ;;
    --target-py-version=*)
      target_py_version="${i#*=}"
      shift
      ;;
    *)
      die "Unknown option: $i"
      ;;
  esac
done

cmake_generator="Ninja"

if [ -n "${build_dir_suffix}" ] && [[ ! "${build_dir_suffix}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  die "Invalid --build-dir-suffix: ${build_dir_suffix}"
fi

coverage_cmake_value=OFF
if [ -n "${enable_coverage}" ]; then
  coverage_cmake_value=ON
fi
internal_symbols_cmake_value=OFF
# Preserve the existing coverage-build contract: coverage includes the QNN EP
# component, snapshot, and accuracy tiers that require internal-symbol access.
if [ -n "${enable_coverage}" ] || [ -n "${enable_qnn_internal_ut_symbols}" ]; then
  internal_symbols_cmake_value=ON
fi
qnn_build_mode="coverage=${coverage_cmake_value};internal_symbols=${internal_symbols_cmake_value}"

build_root="${REPO_ROOT}/build"
build_dir_name="${target_platform}-${target_arch}"
if [ -n "${build_dir_suffix}" ]; then
  build_dir_name+="-${build_dir_suffix}"
fi
build_dir="${build_root}/${build_dir_name}"

qairt_sdk_file_path="${build_dir}/qairt-sdk-path-${config}.txt"
# build.py reruns CMake only when --update is provided. Track the resolved mode
# so switching coverage/internal-symbol settings cannot reuse a stale CMake cache.
qnn_build_mode_file_path="${build_dir}/qnn-build-mode-${config}.txt"

if [ -z "${qairt_sdk_root}" ]; then
    qairt_sdk_root="$(get_qairt_contentdir)"
fi

cmake_bindir="$(get_cmake_bindir)"
llvm_contentdir="$(get_llvm_contentdir)"
# Trigger Hexagon SDK download/extract; cmake discovers the path via ORT_BUILD_TOOLS_PATH.
get_hexagon_sdk_contentdir > /dev/null
# Surface the canonical tools dir to cmake so onnxruntime_unittests_udo.cmake can locate
# LLVM / Hexagon SDK without relying on a CMAKE_*_BINARY_DIR-relative fallback (which can
# resolve outside ${REPO_ROOT}/build/tools depending on where the cmake file is included).
export ORT_BUILD_TOOLS_PATH="$(get_tools_dir)"
PATH="${cmake_bindir}:$(get_ninja_bindir):${llvm_contentdir}/bin:${PATH}"

mkdir -p "${build_dir}/${config}"

build_is_dirty=
if [ "$(update_needed "${build_dir}/${config}" "${qairt_sdk_root}" "${qnn_build_mode}")" ]; then
  build_is_dirty=1
fi

common_args=(--cmake_generator "${cmake_generator}" \
             --config "${config}" \
             --parallel \
             --build_dir "${build_dir}" \
)

if [ -n "${qnn_arch_abi}" ]; then
  common_args+=(--cmake_extra_defines "QNN_ARCH_ABI=${qnn_arch_abi}")
fi

if [ -n "${target_py_version}" ]; then
  common_args+=(--build_wheel)

  build_venv="${build_dir}/venv-${target_py_version}"
  if [ ! -d "${build_venv}" ]; then
    log_debug "Creating venv for build in ${build_venv}"
    if command -v uv >/dev/null 2>&1; then
      uv venv --seed --python "${target_py_version}" "${build_venv}"
    elif command -v "python${target_py_version}" >/dev/null 2>&1; then
      "python${target_py_version}" -m venv "${build_venv}"
    else
      log_info "Neither uv nor python${target_py_version} found; bootstrapping uv into ${build_dir}/_uv_bootstrap."
      uv_bootstrap_dir="${build_dir}/_uv_bootstrap"
      python3 -m pip install --quiet --target="${uv_bootstrap_dir}" uv
      # Prefer the console-script binary; fall back to module form if the wheel didn't ship one.
      if [ -x "${uv_bootstrap_dir}/bin/uv" ]; then
        "${uv_bootstrap_dir}/bin/uv" venv --seed --python "${target_py_version}" "${build_venv}"
      else
        PYTHONPATH="${uv_bootstrap_dir}" python3 -m uv venv --seed --python "${target_py_version}" "${build_venv}"
      fi
    fi
  fi

  bash -c ". ${build_venv}/bin/activate && pip install uv"
  bash -c ". ${build_venv}/bin/activate && uv pip install -r ${REPO_ROOT}/tools/ci_build/github/linux/python/requirements.txt"

  python_for_build="${build_venv}/bin/python"

  log_info "Building wheel using ${python_for_build}."
else
  python_for_build=python
fi

if [ -n "${use_cache}" ]; then
  common_args+=("--use_cache")
  PATH="$(get_ccache_bindir):${PATH}"
fi

if [ -z "${warnings_as_errors}" ]; then
  common_args+=("--compile_no_warning_as_error")
fi

action_args=()
make_test_archive=
run_tests=
test_runner=

case "${target_platform}" in
  linux)
    qnn_args=(--use_qnn --qnn_home "${qairt_sdk_root}")
    if [ -n "${ort_prebuilt_root}" ]; then
      qnn_args+=("--ort_home")
      qnn_args+=("${ort_prebuilt_root}")
    fi
    platform_args=(--build_shared_lib --cmake_extra_defines CMAKE_BUILD_RPATH_USE_ORIGIN:BOOL=TRUE)

    test_runner="${REPO_ROOT}/qcom/scripts/linux/run_tests.sh"

    case "${mode}" in
      build)
        action_args+=("--build")
        if [ -n "${build_is_dirty}" ]; then
          action_args+=("--update")
        fi
        if [ "${target_arch}" == "aarch64_oe_gcc11_2" ]; then
          toolchain_root="$(get_linux_oe_gcc112_toolchain_root)"
          toolchain_cmake="${REPO_ROOT}/qcom/scripts/linux/linux-aarch64-gcc11.toolchain.cmake"

          # We need $toolchain_root from the toolchain.cmake, but the toolchain.cmake is sometimes
          # evaluated without the project's CMakeCache.txt entries. Pass it through the environment :-/
          export ORT_BUILD_LINUX_TOOLCHAIN_ROOT="${toolchain_root}"

          platform_args+=(--cmake_extra_defines
                          CMAKE_TOOLCHAIN_FILE:FILEPATH="${toolchain_cmake}"
                          ARM64:BOOL=TRUE)
        fi
        ;;
      test)
        run_tests=1
        ;;
      archive)
        make_test_archive=1
        ;;
      *)
        die "Invalid mode '${mode}'."
    esac
    ;;

  android)
    if [ -n "${build_is_dirty}" ]; then
      # The ORT Android build doesn't seem to support --update, but our QNN root has changed
      # so we really want to re-run cmake. Blow away the build.
      log_debug "Build is dirty: blowing away ${build_dir}/${config}"
      rm -fr "${build_dir}/${config}"
    fi

    # JDK 21's jlink is not compatible with Android 34's core modules,
    # so Java 17 is used instead.
    # JAVA_HOME must be exported unconditionally (not just for --build-java) because
    # sdkmanager checks JAVA_HOME before PATH; a pre-existing JAVA_HOME pointing to
    # JDK < 17 causes the NDK install step to fail even when java17 is on PATH.
    PATH="$(get_java17_bindir):${PATH}"
    export JAVA_HOME="$(get_java17_contentdir)"
    if [ -n "${build_java}" ]; then
      export GRADLE_USER_HOME="${build_root}/gradle-home"
    fi

    if [ -n "${ANDROID_HOME:-}" -a -n "${ANDROID_NDK_HOME:-}" ]; then
      android_sdk_path="${ANDROID_HOME}"
      android_ndk_path="${ANDROID_NDK_HOME}"
    else
      android_sdk_path="$(get_android_sdk_root)"
      android_ndk_path="$(get_android_ndk_root)"
    fi

    qnn_args=(--use_qnn static_lib --qnn_home "${qairt_sdk_root}")
    if [ -n "${ort_prebuilt_root}" ]; then
      qnn_args+=("--ort_home")
      qnn_args+=("${ort_prebuilt_root}")
    fi
    platform_args=(--build_shared_lib \
                   --android_sdk_path "${android_sdk_path}" \
                   --android_ndk_path "${android_ndk_path}" \
                   --android_abi "arm64-v8a" \
                   --android_api "27")
    if [ -n "${build_java}" ]; then
      platform_args+=(--build_java)
    fi
    case "${mode}" in
      build)
        action_args+=("--android")
        ;;
      test)
        die "--mode=test not supported with --target_platform=${target_platform}."
        ;;
      archive)
        make_test_archive=1
        ;;
      *)
        die "Invalid mode '${mode}'."
    esac
    ;;
  *)
    die "Unknown target platform ${target_platform}."
esac

if [ "${ORT_BUILD_PRUNE_PACKAGES:-1}" == "1" ]; then
  clean_tools_dir
fi

# Whatever happens, blow away mirror to avoid it showing up in git; it's okay, it's
# very cheap to regenerate.
function scrub_mirror() {
  rm -fr "${REPO_ROOT}/mirror"
}
trap scrub_mirror EXIT

if [ -n "${make_test_archive}" ]; then
  python "${REPO_ROOT}/qcom/scripts/all/archive_tests.py" \
    "--config=${config}" \
    "--target-platform=${target_platform}-${target_arch}" \
    "--qairt-sdk-root=${qairt_sdk_root}"
else
  cd "${REPO_ROOT}"

  # This platform supports running tests on the host. Prep the build directory
  # to run with our ctest wrapper.
  if [ -n "${test_runner}" ]; then
    cp "${test_runner}" "${build_dir}/${config}/"
    cp "${cmake_bindir}/ctest" "${build_dir}/${config}/"
    cp "${REPO_ROOT}/qcom/scripts/all/python_test_files.txt" "${build_dir}/${config}/"
  fi

  if [ "${#action_args[@]}" -gt 0 ]; then

    python "${REPO_ROOT}/qcom/scripts/all/fetch_cmake_deps.py"

    package_args=()
    if [ -n "${build_archive}" ]; then
      log_info "Building archive asset."
      package_args+=(--build_archive_asset)
    fi
    if [ -n "${ORT_VERSION_SUFFIX:-}" ]; then
      package_args+=(--version_suffix "${ORT_VERSION_SUFFIX}")
    fi
    if [[ "${ORT_NIGHTLY_BUILD:-}" == "1" ]]; then
      package_args+=(--wheel_name_suffix "qcom_internal" --nightly_build)
    fi

    # Always set both values, including OFF, so a reused build directory cannot
    # retain a previous coverage or internal-symbol configuration from CMakeCache.
    common_args+=(--cmake_extra_defines
                  "ENABLE_COVERAGE:BOOL=${coverage_cmake_value}"
                  "onnxruntime_QNN_ENABLE_INTERNAL_UT_SYMBOLS:BOOL=${internal_symbols_cmake_value}")

    if [ -n "${enable_asan}" ]; then
      common_args+=(--enable_address_sanitizer)
    fi

    # Reconfiguration may change the cache even if the subsequent build fails.
    # Keep the old mode invalid until the entire build succeeds.
    if [ -n "${build_is_dirty}" ]; then
      rm -f "${qnn_build_mode_file_path}"
    fi

    "${python_for_build}" ${REPO_ROOT}/tools/ci_build/build.py \
      "${action_args[@]}" \
      "${common_args[@]}" \
      "${qnn_args[@]}" \
      "${platform_args[@]}" \
      "${package_args[@]}"

    # Record the resolved configuration only after CMake/build succeeds. A
    # failed reconfigure must be retried rather than treated as cache-valid.
    if [ -n "${build_is_dirty}" ]; then
      save_qairt_sdk_path "${qairt_sdk_root}"
      save_qnn_build_mode "${qnn_build_mode}"
    fi
  fi

  if [ -n "${run_tests}" ]; then
    cd "${build_dir}/${config}/"

    # Run tests using our ctest wrapper.
    log_info "-=-=-=- Running unit tests -=-=-=-=-"
    "./$(basename ${test_runner})" --python="${python_for_build}"
  fi
fi
