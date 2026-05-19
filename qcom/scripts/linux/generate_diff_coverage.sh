#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Generate patch/diff coverage report using diff-cover.
#
# Prerequisites:
#   - Cobertura XML coverage report produced by generate_coverage.sh.
#   - A git diff file in unified diff format capturing the changes to analyse.
#     Typical usage: git diff origin/main...HEAD > patch.diff
#   - diff-cover installed in the active Python environment (see qcom/requirements.txt).
#     Activate the project venv before calling this script:
#       source venv/bin/activate
#
# Usage:
#   bash generate_diff_coverage.sh \
#       --coverage-xml=/path/to/coverage/coverage.xml \
#       --diff-file=/path/to/patch.diff \
#       [--output-dir=/path/to/report] \
#       [--fail-under=N]

REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"

set_strict_mode

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
coverage_xml=""
diff_file=""
output_dir=""
fail_under=""

for arg in "$@"; do
    case "${arg}" in
        --coverage-xml=*)
            coverage_xml="${arg#--coverage-xml=}"
            ;;
        --diff-file=*)
            diff_file="${arg#--diff-file=}"
            ;;
        --output-dir=*)
            output_dir="${arg#--output-dir=}"
            ;;
        --fail-under=*)
            fail_under="${arg#--fail-under=}"
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --coverage-xml=<path> --diff-file=<path> [--output-dir=<path>] [--fail-under=<N>]

  --coverage-xml=<path>   Required. Cobertura XML coverage report (output of generate_coverage.sh).
  --diff-file=<path>      Required. Unified diff file (e.g. git diff origin/main...HEAD > patch.diff).
  --output-dir=<path>     Optional. Output directory for the diff-coverage report.
                          Default: same directory as <coverage-xml>
  --fail-under=<N>        Optional. Exit non-zero if patch coverage is below N%. Default: disabled.
EOF
            exit 0
            ;;
        *)
            die "Unknown argument: ${arg}"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate arguments
# ---------------------------------------------------------------------------
if [ -z "${coverage_xml}" ]; then
    die "--coverage-xml is required. Run with --help for usage."
fi

if [ -z "${diff_file}" ]; then
    die "--diff-file is required. Run with --help for usage."
fi

coverage_xml="$(realpath "${coverage_xml}")"
diff_file="$(realpath "${diff_file}")"

if [ ! -f "${coverage_xml}" ]; then
    die "Coverage XML not found: ${coverage_xml}"
fi

if [ ! -f "${diff_file}" ]; then
    die "Diff file not found: ${diff_file}"
fi

if [ -z "${output_dir}" ]; then
    output_dir="$(dirname "${coverage_xml}")"
fi

log_info "=== Diff Coverage Report Generator ==="
log_info "coverage_xml : ${coverage_xml}"
log_info "diff_file    : ${diff_file}"
log_info "output_dir   : ${output_dir}"
if [ -n "${fail_under}" ]; then
    log_info "fail_under   : ${fail_under}%"
fi

# ---------------------------------------------------------------------------
# Check diff-cover is available
# ---------------------------------------------------------------------------
log_info "--- Checking diff-cover ---"
if ! command -v diff-cover &>/dev/null; then
    die "diff-cover not found in PATH. Activate the project venv or: pip install diff-cover"
fi
log_info "Using diff-cover: $(command -v diff-cover)  ($(diff-cover --version 2>&1 || true))"

# ---------------------------------------------------------------------------
# Generate diff-cover report
# ---------------------------------------------------------------------------
log_info "--- Generating diff coverage report ---"
mkdir -p "${output_dir}"

diff_cover_cmd=(
    diff-cover "${coverage_xml}"
    --diff-file "${diff_file}"
    --format "html:${output_dir}/diff-coverage.html"
)
if [ -n "${fail_under}" ]; then
    diff_cover_cmd+=(--fail-under "${fail_under}")
fi

# Emit text summary to CI log and save to file simultaneously.
"${diff_cover_cmd[@]}" | tee "${output_dir}/diff-coverage.txt"

log_info "=== Diff coverage report complete ==="
log_info "HTML report  : ${output_dir}/diff-coverage.html"
log_info "Text summary : ${output_dir}/diff-coverage.txt"
