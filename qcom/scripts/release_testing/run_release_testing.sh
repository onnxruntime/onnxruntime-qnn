#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Bootstrap script for release testing on Linux.
# Creates a venv, installs dependencies, and runs release_testing.py with all
# forwarded arguments. Designed for both CI and local use.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
VENV_NAME="release_testing_venv"

# Clean stale venv
rm -rf "${VENV_NAME}"

# Create and activate venv
python3 -m venv "${VENV_NAME}"
source "${VENV_NAME}/bin/activate"

# Install dependencies
pip install uv
uv pip install --system-certs -r "${REPO_ROOT}/qcom/requirements.txt"

# Run release testing, forwarding all arguments
python "${REPO_ROOT}/qcom/scripts/release_testing/release_testing.py" "$@"
EXIT_CODE=$?

# Cleanup
deactivate
rm -rf "${VENV_NAME}"

exit ${EXIT_CODE}
