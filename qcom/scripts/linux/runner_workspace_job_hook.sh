#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

runner_root="${ORT_RUNNER_ROOT:-/local/mnt/workspace/actions-runner}"
lock_file="${ORT_WORKSPACE_CLEANUP_LOCK:-${runner_root}/.workspace-cleanup.lock}"
active_job_marker="${ORT_RUNNER_ACTIVE_JOB_MARKER:-${runner_root}/.job-active}"

exec 9>"${lock_file}"
flock 9

case "$(basename "$0")" in
    *job-started)
        touch "${active_job_marker}"
        ;;
    *job-completed)
        rm -f -- "${active_job_marker}"
        ;;
    *)
        echo "Unknown runner hook name: $0" >&2
        exit 1
        ;;
esac
