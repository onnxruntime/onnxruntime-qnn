#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Single source of truth for the QAIRT + ORT version strings that stamp the QNN
# EP unit-test golden store. Shared by the manifest WRITER (publish_goldens.sh)
# and — once it rebases on top of this script — the gate READER
# (accuracy_gate.py). Keeping one resolver on both sides guarantees the manifest
# is stamped with the same version string the gate later compares against.
#
# This file is dual-mode:
#   * Executable  -> prints versions to stdout (CLI below).
#   * Sourceable  -> exposes resolver functions resolve_qairt_version /
#                    resolve_ort_version for other scripts to call directly.
#
# CLI:
#   resolve_tool_versions.sh --bin-dir=<path> [qairt|ort|both]   (default: both)
#     qairt  -> print QAIRT version, or exit 3 if undeterminable
#     ort    -> print ORT version,   or exit 3 if undeterminable
#     both   -> print "qairt=<v>\nort=<v>"; exit 3 if EITHER is undeterminable
#
# Exit codes:
#   0  success
#   2  usage error (unknown argument / missing --bin-dir)
#   3  version undeterminable (graceful — callers treat this as a safe signal:
#      no version -> refuse to stamp a manifest / fall back to a full accuracy run)
# (1 and 99 are deliberately avoided so callers can distinguish "undeterminable"
#  from a generic die/setup failure.)
#
# Both resolvers key off <bin_dir>, the build output directory containing
# CMakeCache.txt (the same directory callers already locate to find
# onnxruntime_provider_test / accuracy_results.json). This -- not an env var --
# is the source of truth: no current CI workflow ever exports
# QAIRT_SDK_ROOT/QNN_SDK_ROOT/SNPE_ROOT/ORT_PREBUILT_ROOT, so build_and_test.py
# always lets the underlying build auto-fetch QAIRT + ORT. CMake records
# whatever path it actually resolved into CMakeCache.txt regardless of whether
# it was auto-fetched or explicitly given, so reading the build's own record is
# a strict superset of reading an env var that's never set in practice.
#
# QAIRT version precedence:
#   1. <bin_dir>/CMakeCache.txt's `onnxruntime_QNN_HOME` entry -> that root's
#      sdk.yaml `version:` key (see cmake/onnxruntime_providers_qnn.cmake,
#      which extracts the same value the same way at configure time).
#   2. otherwise undeterminable.
#
# ORT version precedence:
#   1. <bin_dir>/CMakeCache.txt's `onnxruntime_ORT_HOME` entry (set only when
#      --ort-prebuilt was passed; no current CI does) -> VERSION_NUMBER then
#      VERSION under that root.
#   2. otherwise, <bin_dir>/_deps/ort_core-src/VERSION_NUMBER -- the FetchContent
#      source tree CMake pulls ORT into when no prebuilt is given (see
#      cmake/external/onnxruntime_external_deps.cmake); this is the path every
#      current CI build takes.
#   3. otherwise undeterminable.
#
# This deliberately does NOT read repo-root VERSION_NUMBER: that file is the
# onnxruntime-qnn PLUGIN/wheel version (get_ort_version(), build.py:27-28, used
# only to name the wheel), NOT the ORT RUNTIME version.

_RTV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${_RTV_SCRIPT_DIR}/common.sh"

# ---------------------------------------------------------------------------
# _rtv_trim <string>
#   Echo the argument with leading/trailing whitespace removed. Always succeeds.
# ---------------------------------------------------------------------------
_rtv_trim() {
    printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# ---------------------------------------------------------------------------
# _rtv_parse_sdk_yaml_version <sdk.yaml path>
#   Parse a flat-YAML `version:` value. Echoes the trimmed, unquoted value and
#   returns 0 on success; returns 1 with no output otherwise. The first matching
#   line wins, mirroring the gate's re.match behavior. The regex is intentionally
#   loose (case-insensitive key); both sides pin it together when a real SDK
#   layout lands.
# ---------------------------------------------------------------------------
_rtv_parse_sdk_yaml_version() {
    local sdk_yaml="$1"
    [ -f "${sdk_yaml}" ] || return 1
    local line
    line="$(grep -m1 -E '^[[:space:]]*[Vv]ersion[[:space:]]*:' "${sdk_yaml}" 2>/dev/null || true)"
    [ -n "${line}" ] || return 1
    local value
    value="$(_rtv_trim "${line#*:}")"
    # Strip a single pair of surrounding single or double quotes.
    value="${value#[\"\']}"
    value="${value%[\"\']}"
    [ -n "${value}" ] || return 1
    printf '%s' "${value}"
    return 0
}

# ---------------------------------------------------------------------------
# _rtv_read_cmake_cache_var <CMakeCache.txt path> <var name>
#   Echo the value of a `<var>:<type>=<value>` cache entry, or return 1 with no
#   output if the file or entry is missing.
# ---------------------------------------------------------------------------
_rtv_read_cmake_cache_var() {
    local cache="$1" var="$2"
    [ -f "${cache}" ] || return 1
    local line
    line="$(grep -m1 -E "^${var}:" "${cache}" 2>/dev/null || true)"
    [ -n "${line}" ] || return 1
    local value="${line#*=}"
    [ -n "${value}" ] || return 1
    printf '%s' "${value}"
    return 0
}

# ---------------------------------------------------------------------------
# resolve_qairt_version <bin_dir>
#   Echo the resolved QAIRT version + return 0, or return 1 with no output.
# ---------------------------------------------------------------------------
resolve_qairt_version() {
    local bin_dir="$1"
    local qnn_home
    qnn_home="$(_rtv_read_cmake_cache_var "${bin_dir}/CMakeCache.txt" "onnxruntime_QNN_HOME")" || return 1
    _rtv_parse_sdk_yaml_version "${qnn_home}/sdk.yaml"
}

# ---------------------------------------------------------------------------
# resolve_ort_version <bin_dir>
#   Echo the resolved ORT version + return 0, or return 1 with no output.
# ---------------------------------------------------------------------------
resolve_ort_version() {
    local bin_dir="$1"
    local ort_home v f
    if ort_home="$(_rtv_read_cmake_cache_var "${bin_dir}/CMakeCache.txt" "onnxruntime_ORT_HOME")"; then
        for f in "${ort_home}/VERSION_NUMBER" "${ort_home}/VERSION"; do
            [ -f "${f}" ] || continue
            v="$(_rtv_trim "$(head -n1 "${f}" 2>/dev/null || true)")"
            if [ -n "${v}" ]; then
                printf '%s' "${v}"
                return 0
            fi
        done
    fi
    # No prebuilt given -> CMake FetchContent-ed ORT's own source tree.
    local fc="${bin_dir}/_deps/ort_core-src/VERSION_NUMBER"
    if [ -f "${fc}" ]; then
        v="$(_rtv_trim "$(head -n1 "${fc}" 2>/dev/null || true)")"
        if [ -n "${v}" ]; then
            printf '%s' "${v}"
            return 0
        fi
    fi
    return 1
}

_rtv_usage() {
    cat >&2 <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --bin-dir=<path> [qairt|ort|both]

  --bin-dir=<path>  Required. Build output dir containing CMakeCache.txt
                     (e.g. build/linux-x86_64/RelWithDebInfo).
  qairt   Print the resolved QAIRT version (exit 3 if undeterminable).
  ort     Print the resolved ORT version   (exit 3 if undeterminable).
  both    Print "qairt=<v>" and "ort=<v>" (default; exit 3 if either is
          undeterminable).

Exit codes: 0 success / 2 usage error / 3 version undeterminable.
EOF
}

main() {
    set_strict_mode
    local bin_dir="" what="both"
    for arg in "$@"; do
        case "${arg}" in
            --bin-dir=*)
                bin_dir="${arg#--bin-dir=}"
                ;;
            qairt|ort|both)
                what="${arg}"
                ;;
            -h|--help)
                _rtv_usage
                return 0
                ;;
            *)
                log_err "Unknown argument: ${arg}"
                _rtv_usage
                return 2
                ;;
        esac
    done
    if [ -z "${bin_dir}" ]; then
        log_err "--bin-dir=<path> is required."
        _rtv_usage
        return 2
    fi

    case "${what}" in
        qairt)
            local v
            if v="$(resolve_qairt_version "${bin_dir}")"; then
                printf '%s\n' "${v}"
                return 0
            fi
            log_err "QAIRT version undeterminable. Checked ${bin_dir}/CMakeCache.txt (onnxruntime_QNN_HOME) and its sdk.yaml."
            return 3
            ;;
        ort)
            local v
            if v="$(resolve_ort_version "${bin_dir}")"; then
                printf '%s\n' "${v}"
                return 0
            fi
            log_err "ORT version undeterminable. Checked ${bin_dir}/CMakeCache.txt (onnxruntime_ORT_HOME) and ${bin_dir}/_deps/ort_core-src/VERSION_NUMBER."
            return 3
            ;;
        both)
            local qv ov rc=0
            if ! qv="$(resolve_qairt_version "${bin_dir}")"; then
                rc=3
                log_err "QAIRT version undeterminable. Checked ${bin_dir}/CMakeCache.txt (onnxruntime_QNN_HOME) and its sdk.yaml."
            fi
            if ! ov="$(resolve_ort_version "${bin_dir}")"; then
                rc=3
                log_err "ORT version undeterminable. Checked ${bin_dir}/CMakeCache.txt (onnxruntime_ORT_HOME) and ${bin_dir}/_deps/ort_core-src/VERSION_NUMBER."
            fi
            if [ "${rc}" -ne 0 ]; then
                return 3
            fi
            printf 'qairt=%s\n' "${qv}"
            printf 'ort=%s\n' "${ov}"
            return 0
            ;;
    esac
}

# Dual-mode: run main only when executed directly, not when sourced.
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
    exit $?
fi
