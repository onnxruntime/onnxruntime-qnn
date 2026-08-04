#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Publish QNN EP unit-test golden files to Artifactory.
#
# Thin orchestrator over run_snapshot_accuracy.sh:
#   1. Regenerate goldens into the golden store (--generate-goldens writes them,
#      then runs the accuracy tier to verify numerical correctness).
#   2. Read the accuracy JSON report to find which op GROUPS actually PASSED.
#   3. Package ONLY the passing groups' goldens + a version-stamped manifest.json.
#   4. Upload to Artifactory: a write-once dated/sha archive, then the mutable
#      latest/ pointer the gate reads.
#
# A group is published iff every one of its QnnUnit_Accuracy_<Group>Test cases
# PASSED (result COMPLETED, no failures). A group with any DRIFT/SKIPPED case is
# excluded and logged — we never stamp a golden as "known good at version X"
# unless the numerical gate agreed at that version.
#
# The manifest's qairt/ort versions come from resolve_tool_versions.sh, the same
# resolver the gate uses, so the version the store is stamped with is exactly
# the version the gate later compares against.
#
# SAFETY: default is dry-run. --publish is required to actually upload (it
# overwrites the shared latest/ pointer — a destructive, never-implicit action).
#
# Usage:
#   bash publish_goldens.sh --build-dir=<path> [options]
#
#   --build-dir=<path>    Required. Build root (passed through to
#                         run_snapshot_accuracy.sh).
#   --golden-dir=<path>   Golden store root. Default: $QNN_UT_SNAPSHOT_GOLDEN_DIR.
#                         Die if neither is set.
#   --filter=<g1,g2,...>  Scope regen to these op groups (passed through).
#   --repo-subpath=<p>    Artifactory path prefix. Default: "qnn-ut-goldens".
#   --dry-run             Do everything except the jf upload; print the exact
#                         jf commands. (Default when neither flag is given.)
#   --publish             Actually upload. Requires JF_URL / JF_ACCESS_TOKEN /
#                         BUILD_ARTIFACTORY_REPO.
#   --skip-regen          Reuse existing goldens + accuracy results (re-publish).
#   -h|--help

REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"
# shellcheck source=resolve_tool_versions.sh
source "${REPO_ROOT}/qcom/scripts/linux/resolve_tool_versions.sh"

set_strict_mode

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
build_dir=""
golden_dir=""
filter_groups=""
repo_subpath="qnn-ut-goldens"
dry_run=false
publish=false
skip_regen=false
mode_flag_given=false

for arg in "$@"; do
    case "${arg}" in
        --build-dir=*)    build_dir="${arg#--build-dir=}" ;;
        --golden-dir=*)   golden_dir="${arg#--golden-dir=}" ;;
        --filter=*)       filter_groups="${arg#--filter=}" ;;
        --repo-subpath=*) repo_subpath="${arg#--repo-subpath=}" ;;
        --dry-run)        dry_run=true;  mode_flag_given=true ;;
        --publish)        publish=true;  mode_flag_given=true ;;
        --skip-regen)     skip_regen=true ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --build-dir=<path> [options]

Regenerate QNN EP unit-test goldens, keep only accuracy-PASSING groups, and
publish them (+ a version-stamped manifest.json) to Artifactory.

  --build-dir=<path>    Required. Build root (passthrough to run_snapshot_accuracy.sh).
  --golden-dir=<path>   Golden store root. Default: \$QNN_UT_SNAPSHOT_GOLDEN_DIR.
  --filter=<g1,g2,...>  Scope regen to these op groups (passthrough).
  --repo-subpath=<p>    Artifactory path prefix. Default: qnn-ut-goldens.
  --dry-run             Do everything except upload; print the jf commands.
  --publish             Actually upload. Requires JF_URL / JF_ACCESS_TOKEN /
                        BUILD_ARTIFACTORY_REPO.
  --skip-regen          Reuse existing goldens + accuracy results.

Default when neither --dry-run nor --publish is given: dry-run (safe).
EOF
            exit 0
            ;;
        *) die "Unknown argument: ${arg}" ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate arguments & resolve defaults
# ---------------------------------------------------------------------------
if [ -z "${build_dir}" ]; then
    die "--build-dir is required. Run with --help for usage."
fi

if [ "${publish}" = true ] && [ "${dry_run}" = true ]; then
    die "--dry-run and --publish are mutually exclusive."
fi

# Default-safe: neither flag -> dry-run.
if [ "${mode_flag_given}" = false ]; then
    dry_run=true
    log_warn "Neither --dry-run nor --publish given; defaulting to dry-run (no upload)."
fi

# Golden store root: explicit flag, else env. No implicit creation.
if [ -z "${golden_dir}" ]; then
    golden_dir="${QNN_UT_SNAPSHOT_GOLDEN_DIR:-}"
fi
if [ -z "${golden_dir}" ]; then
    die "Golden store root unknown. Pass --golden-dir=<path> or set QNN_UT_SNAPSHOT_GOLDEN_DIR."
fi

# ---------------------------------------------------------------------------
# Early tool probes — fail before the expensive regen, not after.
# ---------------------------------------------------------------------------
for tool in python3 git zip; do
    command -v "${tool}" &>/dev/null || die "Required tool not found in PATH: ${tool}"
done

if [ "${publish}" = true ]; then
    command -v jf &>/dev/null || die "--publish requires the JFrog CLI (jf) in PATH."
    : "${JF_URL:?--publish requires JF_URL}"
    : "${JF_ACCESS_TOKEN:?--publish requires JF_ACCESS_TOKEN}"
    : "${BUILD_ARTIFACTORY_REPO:?--publish requires BUILD_ARTIFACTORY_REPO}"
fi

log_info "=== QNN EP Golden Publisher ==="
log_info "build_dir    : ${build_dir}"
log_info "golden_dir   : ${golden_dir}"
log_info "repo_subpath : ${repo_subpath}"
if [ -n "${filter_groups}" ]; then
    log_info "filter       : ${filter_groups}"
fi
if [ "${publish}" = true ]; then
    log_info "mode         : PUBLISH (will upload)"
else
    log_info "mode         : dry-run (no upload)"
fi

# ---------------------------------------------------------------------------
# Regenerate goldens + verify accuracy
# ---------------------------------------------------------------------------
runner="${REPO_ROOT}/qcom/scripts/linux/run_snapshot_accuracy.sh"

if [ "${skip_regen}" = true ]; then
    log_info "--- Skipping regen (--skip-regen); reusing existing goldens + results ---"
    [ -d "${golden_dir}" ] || die "--skip-regen but golden dir does not exist: ${golden_dir}"
else
    log_info "--- Regenerating goldens + verifying accuracy ---"
    [ -x "${runner}" ] || die "run_snapshot_accuracy.sh not found or not executable: ${runner}"

    regen_exit=0
    regen_args=(--build-dir="${build_dir}" --generate-goldens)
    if [ -n "${filter_groups}" ]; then
        regen_args+=(--filter="${filter_groups}")
    fi
    # The snapshot tests read the golden store root from the environment.
    QNN_UT_SNAPSHOT_GOLDEN_DIR="${golden_dir}" "${runner}" "${regen_args[@]}" || regen_exit=$?

    case "${regen_exit}" in
        0)
            log_info "Regen OK: goldens written and accuracy verified."
            ;;
        1)
            log_warn "Regen reported an accuracy regression (exit 1)."
            log_warn "Some groups FAILED accuracy. Continuing to publish ONLY the passing subset."
            ;;
        *)
            die "run_snapshot_accuracy.sh setup error (exit ${regen_exit}). Aborting; nothing published."
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# Locate the accuracy JSON report (replicate run_snapshot_accuracy.sh bin_dir search).
# ---------------------------------------------------------------------------
build_dir_abs="$(realpath "${build_dir}")"
bin_dir=""
for cfg in RelWithDebInfo Release Debug; do
    if [ -x "${build_dir_abs}/${cfg}/onnxruntime_provider_test" ]; then
        bin_dir="${build_dir_abs}/${cfg}"
        break
    fi
done
if [ -z "${bin_dir}" ] && [ -x "${build_dir_abs}/onnxruntime_provider_test" ]; then
    bin_dir="${build_dir_abs}"
fi
[ -n "${bin_dir}" ] || die "onnxruntime_provider_test not found under ${build_dir_abs}."

accuracy_json="${bin_dir}/snapshot_accuracy_results/accuracy_results.json"
if [ ! -f "${accuracy_json}" ]; then
    die "Accuracy report not found: ${accuracy_json}. The accuracy tier must run to \
prove which groups pass (is QNN_EP_ACCURACY_UT enabled?). Refusing to publish \
goldens with no pass evidence."
fi
log_info "Accuracy report: ${accuracy_json}"

# ---------------------------------------------------------------------------
# Select PASSING groups from the accuracy report.
#
# A group passes iff every QnnUnit_Accuracy_<Group>Test case is PASSED
# (result COMPLETED, no failures). Mirrors accuracy_gate.py's classify_testcase;
# implemented inline here so this PR does not import the not-yet-merged gate.
# ---------------------------------------------------------------------------
mapfile -t pass_groups < <(python3 - "${accuracy_json}" <<'PYEOF'
import json, re, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

pattern = re.compile(r'^QnnUnit_Accuracy_(\w+)Test')
# group -> all cases PASSED so far (True until a non-PASSED case flips it False)
group_all_pass = {}

for suite in data.get('testsuites', []):
    m = pattern.match(suite.get('name', ''))
    if not m:
        continue
    group = m.group(1)
    for case in suite.get('testsuite', []):
        result = case.get('result')      # COMPLETED / SKIPPED
        status = case.get('status')      # RUN / NOTRUN
        has_failures = bool(case.get('failures'))
        passed = (result == 'COMPLETED') and (not has_failures)
        # SKIPPED / NOTRUN / DRIFT all count as "not passed".
        if result == 'SKIPPED' or status == 'NOTRUN':
            passed = False
        group_all_pass[group] = group_all_pass.get(group, True) and passed
    # A suite with zero cases contributes no evidence of passing.
    group_all_pass.setdefault(group, False)

for group in sorted(g for g, ok in group_all_pass.items() if ok):
    print(group)
PYEOF
) || die "Failed to parse accuracy report ${accuracy_json}."

if [ "${#pass_groups[@]}" -eq 0 ]; then
    die "No group passed accuracy. Nothing to publish."
fi
log_info "Passing groups: ${pass_groups[*]}"

# ---------------------------------------------------------------------------
# Stage the passing groups' goldens into a temp tree (preserve relative paths).
#
# Golden layout (snapshot_golden_utils.h): $GOLDEN_DIR/<tier>/builder/opbuilder/
# <op_stem>/<Case>.json. The op_stem is the leaf dir basename; we match it
# against pass groups case-insensitively with underscores removed (Clip->clip,
# GeluFusion->gelu_fusion). Both the snapshot and session_snapshot tiers of a
# passing group are collected.
# ---------------------------------------------------------------------------
[ -d "${golden_dir}" ] || die "Golden dir does not exist: ${golden_dir}"
golden_dir_abs="$(realpath "${golden_dir}")"

staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT

_munge() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -d '_'; }

declare -A pass_munged=()
for g in "${pass_groups[@]}"; do
    pass_munged["$(_munge "${g}")"]=1
done

golden_count=0
declare -A excluded_leaf=()

while IFS= read -r -d '' jf_file; do
    rel="${jf_file#"${golden_dir_abs}"/}"
    leaf_dir="$(dirname "${rel}")"
    leaf_name="$(basename "${leaf_dir}")"
    if [ -n "${pass_munged[$(_munge "${leaf_name}")]:-}" ]; then
        mkdir -p "${staging}/${leaf_dir}"
        cp "${jf_file}" "${staging}/${rel}"
        golden_count=$((golden_count + 1))
    else
        excluded_leaf["${leaf_dir}"]=1
    fi
done < <(find "${golden_dir_abs}" -type f -name '*.json' -print0)

if [ "${#excluded_leaf[@]}" -gt 0 ]; then
    log_info "Excluded (not in a passing group):"
    for d in "${!excluded_leaf[@]}"; do
        log_info "  ${d}"
    done
fi

if [ "${golden_count}" -eq 0 ]; then
    die "Passing groups (${pass_groups[*]}) matched no golden files under ${golden_dir_abs}."
fi
log_info "Staged ${golden_count} golden file(s) from ${#pass_groups[@]} passing group(s)."

# ---------------------------------------------------------------------------
# Write manifest.json (version-stamped by resolve_tool_versions.sh).
# ---------------------------------------------------------------------------
qairt_version="$(resolve_qairt_version)" \
    || die "QAIRT version undeterminable — refusing to produce an unversioned manifest."
ort_version="$(resolve_ort_version)" \
    || die "ORT version undeterminable — refusing to produce an unversioned manifest."

git_sha="$(git -C "${REPO_ROOT}" rev-parse --short=10 HEAD)"
generated_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Build the passing-groups JSON array and manifest via python3 (safe quoting).
MANIFEST_QAIRT="${qairt_version}" \
MANIFEST_ORT="${ort_version}" \
MANIFEST_SHA="${git_sha}" \
MANIFEST_UTC="${generated_utc}" \
MANIFEST_COUNT="${golden_count}" \
MANIFEST_GROUPS="$(IFS=,; printf '%s' "${pass_groups[*]}")" \
python3 - "${staging}/manifest.json" <<'PYEOF'
import json, os, sys

groups = [g for g in os.environ["MANIFEST_GROUPS"].split(",") if g]
manifest = {
    "qairt_version": os.environ["MANIFEST_QAIRT"],
    "ort_version": os.environ["MANIFEST_ORT"],
    "git_sha": os.environ["MANIFEST_SHA"],
    "generated_utc": os.environ["MANIFEST_UTC"],
    "passing_groups": groups,
    "golden_count": int(os.environ["MANIFEST_COUNT"]),
    "generator": "publish_goldens.sh",
}
with open(sys.argv[1], "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
PYEOF

log_info "manifest.json: qairt=${qairt_version} ort=${ort_version} sha=${git_sha} groups=[${pass_groups[*]}] count=${golden_count}"

# ---------------------------------------------------------------------------
# Zip the staging tree. Archive root = manifest.json + tier subdirs.
# ---------------------------------------------------------------------------
zip_path="${staging}/goldens.zip"
(
    cd "${staging}"
    # Exclude the zip itself in case of re-runs; -x is a no-op on first pass.
    zip -r -q goldens.zip . -x goldens.zip
)
log_info "Packaged: goldens.zip ($(du -h "${zip_path}" | cut -f1))"

# ---------------------------------------------------------------------------
# Upload: write-once archive first, then the mutable latest/ pointer.
# ---------------------------------------------------------------------------
utc_date="$(date -u +%Y%m%d)"
if [ "${publish}" = true ]; then
    dest_base="${BUILD_ARTIFACTORY_REPO}/${repo_subpath}"
else
    dest_base="<BUILD_ARTIFACTORY_REPO>/${repo_subpath}"
fi
archive_dest="${dest_base}/archive/${utc_date}-${git_sha}/goldens.zip"
latest_dest="${dest_base}/latest/goldens.zip"

if [ "${publish}" = true ]; then
    log_info "--- Uploading (archive, then latest) ---"
    jf rt upload --flat "${zip_path}" "${archive_dest}"
    jf rt upload --flat "${zip_path}" "${latest_dest}"
    log_info "=== Published ==="
    log_info "archive: ${archive_dest}"
    log_info "latest : ${latest_dest}"
else
    log_info "--- DRY-RUN: would upload with these commands ---"
    log_info "jf rt upload --flat \"${zip_path}\" \"${archive_dest}\""
    log_info "jf rt upload --flat \"${zip_path}\" \"${latest_dest}\""
    log_warn "Dry-run: nothing uploaded. Re-run with --publish to upload."
fi
