# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Regression checks for the Linux internal-symbol CI workflow configuration."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/qualcomm-internal-build-and-test.yml"


def test_linux_x86_internal_symbol_validation_runs_on_the_build_runner() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    workflow_inputs = workflow["jobs"]["build-and-test-linux-x86_64-ubuntu-22-04"]["with"]

    assert workflow_inputs["qnn_internal_test_symbols"] is True
    assert workflow_inputs["build_test_same_runner"] is True
    assert workflow_inputs["upload_test_archive"] is False
    assert workflow_inputs["upload_wheel"] is True
