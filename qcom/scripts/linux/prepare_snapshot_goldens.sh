#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Download and enable the latest QNN snapshot-golden store for a coverage run.
# A missing, corrupt, or version-mismatched store deliberately leaves
# QNN_UT_SNAPSHOT_GOLDEN_DIR unset, so accuracy_gate.py runs full accuracy.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
# shellcheck source=resolve_tool_versions.sh
source "${script_dir}/resolve_tool_versions.sh"

usage() {
    cat <<USAGE
Usage: $(basename "$0") --build-dir=<path> --golden-dir=<empty-path> [--repo-subpath=<path>]
USAGE
}

build_dir=""
golden_dir=""
repo_subpath="ci/qnn-ep-test-store/snapshot-goldens"
for arg in "$@"; do
    case "${arg}" in
        --build-dir=*) build_dir="${arg#--build-dir=}" ;;
        --golden-dir=*) golden_dir="${arg#--golden-dir=}" ;;
        --repo-subpath=*) repo_subpath="${arg#--repo-subpath=}" ;;
        -h|--help) usage; exit 0 ;;
        *) log_err "Unknown argument: ${arg}"; usage >&2; exit 2 ;;
    esac
done

clear_store_env() {
    if [ -n "${GITHUB_ENV:-}" ]; then
        printf '%s\n' "QNN_UT_SNAPSHOT_GOLDEN_DIR=" >> "${GITHUB_ENV}"
    fi
}

# Preflight starts disabled; only a fully verified archive enables the store.
clear_store_env

disable_store() {
    log_warn "Snapshot golden store disabled: $*"
    exit 0
}

[ -n "${build_dir}" ] || { usage >&2; exit 2; }
[ -n "${golden_dir}" ] || { usage >&2; exit 2; }
[ -n "${BUILD_ARTIFACTORY_REPO:-}" ] || disable_store "BUILD_ARTIFACTORY_REPO is unset."
command -v jf >/dev/null 2>&1 || disable_store "JFrog CLI is unavailable."
command -v unzip >/dev/null 2>&1 || disable_store "unzip is unavailable."

build_dir="$(realpath "${build_dir}")"
qairt_version="$(resolve_qairt_version "${build_dir}")" || disable_store "QAIRT version cannot be resolved from ${build_dir}."
ort_version="$(resolve_ort_version "${build_dir}")" || disable_store "ORT version cannot be resolved from ${build_dir}."

golden_parent="$(dirname "${golden_dir}")"
mkdir -p "${golden_parent}"
if ! golden_dir="$(realpath -m "${golden_dir}")"; then
    disable_store "failed to canonicalize destination: ${golden_dir}."
fi
golden_parent="$(dirname "${golden_dir}")"
if [ -e "${golden_dir}" ]; then
    if [ ! -d "${golden_dir}" ] || [ -n "$(find "${golden_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]; then
        disable_store "destination is not an empty directory: ${golden_dir}"
    fi
    rmdir "${golden_dir}" || disable_store "failed to prepare empty destination: ${golden_dir}"
fi

staging="$(mktemp -d "${golden_parent}/.snapshot-goldens.XXXXXX")"
trap 'rm -rf "${staging}"' EXIT

remote="${BUILD_ARTIFACTORY_REPO}/${repo_subpath}/latest/goldens.zip"
zip_path="${staging}/goldens.zip"
log_info "Downloading snapshot golden manifest candidate: ${remote}"
if ! jf rt download --flat "${remote}" "${staging}/"; then
    disable_store "download failed."
fi
[ -f "${zip_path}" ] || disable_store "download did not produce goldens.zip."

if ! unzip -tqq "${zip_path}"; then
    disable_store "downloaded goldens.zip is corrupt."
fi

if ! python3 - "${zip_path}" "${qairt_version}" "${ort_version}" <<'PY'
import json
import sys
import zipfile

zip_path, expected_qairt, expected_ort = sys.argv[1:]
try:
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
    print(f"Cannot read golden manifest: {exc}", file=sys.stderr)
    raise SystemExit(1)

actual_qairt = manifest.get("qairt_version")
actual_ort = manifest.get("ort_version")
if actual_qairt != expected_qairt or actual_ort != expected_ort:
    print(
        "Golden manifest version mismatch: "
        f"qairt={actual_qairt!r} (expected {expected_qairt!r}), "
        f"ort={actual_ort!r} (expected {expected_ort!r})",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
then
    disable_store "manifest is absent or versions do not match the coverage build."
fi

extract_dir="${staging}/extracted"
mkdir "${extract_dir}"
if ! unzip -q "${zip_path}" -d "${extract_dir}"; then
    disable_store "failed to extract goldens.zip."
fi
[ -f "${extract_dir}/manifest.json" ] || disable_store "extracted golden manifest is missing."

# Publish only a complete store. staging and golden_dir share golden_parent, so
# this rename is atomic and a failed extraction cannot leave partial goldens.
if ! mv -T "${extract_dir}" "${golden_dir}"; then
    disable_store "failed to publish extracted goldens."
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    echo "QNN_UT_SNAPSHOT_GOLDEN_DIR=${golden_dir}" >> "${GITHUB_ENV}"
fi
log_info "Enabled aligned snapshot golden store: ${golden_dir} (qairt=${qairt_version}, ort=${ort_version})"
