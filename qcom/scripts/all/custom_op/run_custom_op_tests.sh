#!/bin/sh
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# POSIX shell: runs on Linux and Android without Python or a build toolchain.
set -eu
package=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build=$(dirname -- "$package")
export LD_LIBRARY_PATH="$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ORT_QNN_UDO_OP_PACKAGE_DIR="$package/op_packages/htp"
unset ORT_QNN_CUSTOM_OP_DOMAINS
cd "$build"

# Require actual executed tests, not successful processes containing only skips.
run_test() {
  name=$1
  report=$2
  "$build/onnxruntime_provider_test" "--gtest_filter=$name" "--gtest_output=xml:$report"
  [ -s "$report" ] || { echo "Missing test report: $report" >&2; exit 1; }
  if grep -q '<skipped\|<failure\|status="notrun"' "$report"; then
    echo "Custom Op tests failed or skipped: $report" >&2
    exit 1
  fi
  grep -q '<testcase ' "$report" || { echo "No Custom Op tests ran" >&2; exit 1; }
}

case "$(uname -s):$(uname -m)" in
  Linux:x86_64)
    arches=""
    for context in "$package"/contexts/*; do
      [ -d "$context" ] || continue
      arches="$arches $(basename -- "$context")"
    done
    [ -n "$arches" ] || { echo "No packaged Custom Op contexts" >&2; exit 1; }
    ;;
  Linux:aarch64)
    sdk_lib="$build/../../../lib"
    export LD_LIBRARY_PATH="$sdk_lib/aarch64-android:$LD_LIBRARY_PATH"
    dsp_dirs=""
    for directory in "$sdk_lib"/hexagon-v*/unsigned; do
      [ -d "$directory" ] || continue
      dsp_dirs="$dsp_dirs$directory;"
    done
    export ADSP_LIBRARY_PATH="$dsp_dirs$build;${ADSP_LIBRARY_PATH:-/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp}"
    export ORT_QNN_UDO_CONTEXT_DIR="$build"
    run_test 'QnnHTPBackendTests.UDO_AOT_QueryTarget' "$build/custom_op_target.results.xml"
    device_target=$(cat "$build/custom_op_device_target.txt")
    arches=${device_target%%:*}
    [ -f "$package/contexts/$arches/target.txt" ] || { echo "No packaged context for device V$arches" >&2; exit 1; }
    export ADSP_LIBRARY_PATH="$ORT_QNN_UDO_OP_PACKAGE_DIR/hexagon-v$arches;$ADSP_LIBRARY_PATH"
    ;;
  *) echo "Custom Op tests support Linux x86_64 and Android aarch64" >&2; exit 1 ;;
esac


run_test '*UDO_Op_MyAdd*' "$build/custom_op_jit.results.xml"
for arch in $arches; do
  ORT_QNN_UDO_CONTEXT_TARGET=$(cat "$package/contexts/$arch/target.txt")
  export ORT_QNN_UDO_CONTEXT_TARGET
  export ORT_QNN_UDO_CONTEXT_DIR="$package/contexts/$arch"
  # Load the archive's Linux-generated context in a fresh process without a domain.
  run_test 'QnnHTPUdoTests.UDO_AOT_LoadContext' "$build/custom_op_aot_load_v$arch.results.xml"
  # Independently cover native generation and consumption, without altering fixtures.
  native=$(mktemp -d "${TMPDIR:-$build}/custom-op-native.XXXXXX")
  trap 'rm -rf -- "$native"' EXIT HUP INT TERM
  export ORT_QNN_UDO_CONTEXT_DIR="$native"
  export ORT_QNN_UDO_CONTEXT_TARGET="${device_target:-$ORT_QNN_UDO_CONTEXT_TARGET}"
  run_test 'QnnHTPUdoTests.UDO_AOT_GenerateContext' "$build/custom_op_aot_generate_v$arch.results.xml"
  run_test 'QnnHTPUdoTests.UDO_AOT_LoadContext' "$build/custom_op_aot_reload_v$arch.results.xml"
  rm -rf -- "$native"
  trap - EXIT HUP INT TERM
done
