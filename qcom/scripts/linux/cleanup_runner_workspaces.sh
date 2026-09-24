#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

runner_root="${ORT_RUNNER_ROOT:-/local/mnt/workspace/actions-runner}"
work_root="${runner_root}/_work"
runner_user="${ORT_RUNNER_USER:-ortqnnepci}"
retention_minutes="${ORT_WORKSPACE_RETENTION_MINUTES:-2880}"
warning_percent="${ORT_DISK_WARNING_PERCENT:-70}"
emergency_percent="${ORT_DISK_EMERGENCY_PERCENT:-85}"
lock_file="${ORT_WORKSPACE_CLEANUP_LOCK:-${runner_root}/.workspace-cleanup.lock}"
active_job_marker="${ORT_RUNNER_ACTIVE_JOB_MARKER:-${runner_root}/.job-active}"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

if [ ! -d "${runner_root}" ] || [ ! -d "${work_root}" ]; then
    log "Runner work directory does not exist: ${work_root}"
    exit 0
fi

resolved_runner_root=$(realpath "${runner_root}")
resolved_work_root=$(realpath "${work_root}")
if [ "${resolved_work_root}" != "${resolved_runner_root}/_work" ]; then
    log "Refusing unsafe work directory: ${resolved_work_root}"
    exit 1
fi

exec 9>"${lock_file}"
if ! flock --nonblock 9; then
    log "Another workspace cleanup is already running; skipping."
    exit 0
fi

if [ -e "${active_job_marker}" ]; then
    if pgrep -u "${runner_user}" -f '/bin/Runner.Worker' >/dev/null; then
        log "GitHub Actions job is active; skipping cleanup."
        exit 0
    fi

    log "Removing stale active-job marker: ${active_job_marker}"
    rm -f -- "${active_job_marker}"
fi

# This also covers jobs accepted before the job-started hook acquires the lock.
if pgrep -u "${runner_user}" -f '/bin/Runner.Worker' >/dev/null; then
    log "GitHub Actions worker is active; skipping cleanup."
    exit 0
fi

disk_percent=$(df --output=pcent "${runner_root}" | tail -n 1 | tr -cd '0-9')
log "Disk usage before cleanup: ${disk_percent}%"
du -sh "${work_root}" || true

delete_workspace() {
    local workspace="$1"
    local resolved_workspace

    resolved_workspace=$(realpath "${workspace}")
    if [ "$(dirname "${resolved_workspace}")" != "${resolved_work_root}" ]; then
        log "Refusing unsafe workspace path: ${resolved_workspace}"
        return 1
    fi

    log "Deleting runner workspace: ${resolved_workspace}"
    rm -rf -- "${resolved_workspace}"
}

while IFS= read -r -d '' workspace; do
    delete_workspace "${workspace}"
done < <(
    find "${resolved_work_root}" -mindepth 1 -maxdepth 1 -type d \
        ! -name '_*' -mmin "+${retention_minutes}" -print0
)

if [ "${disk_percent}" -ge "${emergency_percent}" ]; then
    log "Disk usage is at least ${emergency_percent}%; deleting all idle repository workspaces."
    while IFS= read -r -d '' workspace; do
        delete_workspace "${workspace}"
    done < <(
        find "${resolved_work_root}" -mindepth 1 -maxdepth 1 -type d ! -name '_*' -print0
    )
elif [ "${disk_percent}" -ge "${warning_percent}" ]; then
    log "Warning: disk usage is at least ${warning_percent}%."
fi

if [ -d "${resolved_work_root}/_temp" ]; then
    find "${resolved_work_root}/_temp" -mindepth 1 -mmin "+${retention_minutes}" -delete
fi

disk_percent=$(df --output=pcent "${runner_root}" | tail -n 1 | tr -cd '0-9')
log "Disk usage after cleanup: ${disk_percent}%"
du -sh "${work_root}" || true
