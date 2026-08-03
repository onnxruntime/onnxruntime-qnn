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
#   resolve_tool_versions.sh [qairt|ort|both]        (default: both)
#     qairt  -> print QAIRT version, or exit 3 if undeterminable
#     ort    -> print ORT version,   or exit 3 if undeterminable
#     both   -> print "qairt=<v>\nort=<v>"; exit 3 if EITHER is undeterminable
#
# Exit codes:
#   0  success
#   2  usage error (unknown argument)
#   3  version undeterminable (graceful — callers treat this as a safe signal:
#      no version -> refuse to stamp a manifest / fall back to a full accuracy run)
# (1 and 99 are deliberately avoided so callers can distinguish "undeterminable"
#  from a generic die/setup failure.)
#
# QAIRT version precedence:
#   1. <root>/sdk.yaml `version:` key, trying $QAIRT_SDK_ROOT, $QNN_SDK_ROOT,
#      $SNPE_ROOT in that order (first hit wins).
#   2. otherwise undeterminable.
#
# ORT version precedence (mirrors accuracy_gate.detect_ort_version()):
#   1. $ORT_PREBUILT_ROOT/VERSION_NUMBER then $ORT_PREBUILT_ROOT/VERSION.
#   2. otherwise undeterminable.
#
# This deliberately does NOT read repo-root VERSION_NUMBER: that file is the
# onnxruntime-qnn PLUGIN/wheel version (get_ort_version(), build.py:27-28, used
# only to name the wheel), NOT the ORT RUNTIME version. Post-ABI-split the plugin
# is built against a prebuilt ORT, so the runtime version the gate's "runtime
# unchanged" invariant depends on is the prebuilt's ($ORT_PREBUILT_ROOT).

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
# resolve_qairt_version
#   Echo the resolved QAIRT version + return 0, or return 1 with no output.
# ---------------------------------------------------------------------------
resolve_qairt_version() {
    local root v
    for root in "${QAIRT_SDK_ROOT:-}" "${QNN_SDK_ROOT:-}" "${SNPE_ROOT:-}"; do
        [ -n "${root}" ] || continue
        if v="$(_rtv_parse_sdk_yaml_version "${root}/sdk.yaml")"; then
            printf '%s' "${v}"
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# resolve_ort_version
#   Echo the resolved ORT version + return 0, or return 1 with no output.
# ---------------------------------------------------------------------------
resolve_ort_version() {
    # The ORT RUNTIME version lives in the prebuilt ORT tree, not the repo (see
    # header). Try VERSION_NUMBER then VERSION under $ORT_PREBUILT_ROOT.
    local prebuilt="${ORT_PREBUILT_ROOT:-}"
    local v f
    if [ -n "${prebuilt}" ]; then
        for f in "${prebuilt}/VERSION_NUMBER" "${prebuilt}/VERSION"; do
            [ -f "${f}" ] || continue
            v="$(_rtv_trim "$(head -n1 "${f}" 2>/dev/null || true)")"
            if [ -n "${v}" ]; then
                printf '%s' "${v}"
                return 0
            fi
        done
    fi
    return 1
}

_rtv_usage() {
    cat >&2 <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [qairt|ort|both]

  qairt   Print the resolved QAIRT version (exit 3 if undeterminable).
  ort     Print the resolved ORT version   (exit 3 if undeterminable).
  both    Print "qairt=<v>" and "ort=<v>" (default; exit 3 if either is
          undeterminable).

Exit codes: 0 success / 2 usage error / 3 version undeterminable.
EOF
}

main() {
    set_strict_mode
    local what="${1:-both}"
    case "${what}" in
        qairt)
            local v
            if v="$(resolve_qairt_version)"; then
                printf '%s\n' "${v}"
                return 0
            fi
            log_err "QAIRT version undeterminable. Point QAIRT_SDK_ROOT/QNN_SDK_ROOT/SNPE_ROOT at an SDK containing sdk.yaml."
            return 3
            ;;
        ort)
            local v
            if v="$(resolve_ort_version)"; then
                printf '%s\n' "${v}"
                return 0
            fi
            log_err "ORT version undeterminable. Ensure \$ORT_PREBUILT_ROOT/VERSION_NUMBER exists."
            return 3
            ;;
        both)
            local qv ov rc=0
            if ! qv="$(resolve_qairt_version)"; then
                rc=3
                log_err "QAIRT version undeterminable. Point QAIRT_SDK_ROOT/QNN_SDK_ROOT/SNPE_ROOT at an SDK containing sdk.yaml."
            fi
            if ! ov="$(resolve_ort_version)"; then
                rc=3
                log_err "ORT version undeterminable. Ensure \$ORT_PREBUILT_ROOT/VERSION_NUMBER exists."
            fi
            if [ "${rc}" -ne 0 ]; then
                return 3
            fi
            printf 'qairt=%s\n' "${qv}"
            printf 'ort=%s\n' "${ov}"
            return 0
            ;;
        -h|--help)
            _rtv_usage
            return 0
            ;;
        *)
            log_err "Unknown argument: ${what}"
            _rtv_usage
            return 2
            ;;
    esac
}

# Dual-mode: run main only when executed directly, not when sourced.
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
    exit $?
fi
