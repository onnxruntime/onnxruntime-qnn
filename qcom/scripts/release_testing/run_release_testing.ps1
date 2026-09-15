# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Bootstrap script for release testing on Windows.
# Creates a venv, installs dependencies, and runs release_testing.py with all
# forwarded arguments. Designed for both CI and local use.

$ErrorActionPreference = 'Stop'

$RepoRoot = git rev-parse --show-toplevel
$VenvName = "release_testing_venv"

# Clean stale venv
if (Test-Path $VenvName) { Remove-Item -Path $VenvName -Recurse -Force }

# Create and activate venv.
# This venv only runs the Python orchestrator (release_testing.py); its
# architecture does not affect which wheels are tested.  Wheel testing
# creates its own arch-specific venv via install_and_test_wheel.ps1 using
# the py launcher (e.g. py -3.12-arm64).
python -m venv $VenvName
. "$VenvName/Scripts/Activate.ps1"

# Install dependencies (--system-certs for corporate proxy environments)
pip install uv
uv pip install --system-certs -r "$RepoRoot/qcom/requirements.txt"

# Run release testing, forwarding all arguments
python "$RepoRoot/qcom/scripts/release_testing/release_testing.py" @args
$exitCode = $LASTEXITCODE

# Cleanup
deactivate
Remove-Item -Path $VenvName -Recurse -Force -ErrorAction SilentlyContinue

exit $exitCode
