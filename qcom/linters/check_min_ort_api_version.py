#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Lintrunner adapter: verify that the QNN EP minimum ORT API version baseline
(``qcom/MIN_ORT_API_VERSION.txt``) matches the value computed from the ORT
headers and the EP source. Mismatch means a code change either added a call
into a newer ORT API (raising the floor) or removed one (lowering it); either
way the baseline must be refreshed in the same PR so the bump is visible in
the diff.

Delegates to ``qcom/scripts/all/compute_min_ort_api_version.py``. That script's
exit-code contract is what we route on:

  0  baseline matches  -> emit nothing
  1  baseline drift    -> emit one ``ort-api-floor-drift`` error
  2  cannot compute    -> emit one ``ort-api-floor-check-error`` error (distinct
                          finding name so reviewers can tell an environment
                          problem from a real baseline-refresh request)

Per-file paths are accepted on stdin (lintrunner convention) but ignored: the
check is project-wide.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

LINTER_CODE = "MIN-ORT-API-VERSION"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "qcom" / "scripts" / "all" / "compute_min_ort_api_version.py"
BASELINE = REPO_ROOT / "qcom" / "MIN_ORT_API_VERSION.txt"


def _msg(description: str, name: str, severity: str = "error", path: str | None = None) -> dict:
    return {
        "path": path or os.path.relpath(BASELINE, REPO_ROOT),
        "line": None,
        "char": None,
        "code": LINTER_CODE,
        "severity": severity,
        "name": name,
        "original": None,
        "replacement": None,
        "description": description,
    }


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("filenames", nargs="*")
    parser.parse_args()

    if not SCRIPT.is_file():
        print(
            json.dumps(_msg(f"compute script missing: {SCRIPT}", name="ort-api-floor-check-error")),
            flush=True,
        )
        return

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--baseline",
            str(BASELINE),
            "--fetch-from-deps-txt",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )

    if proc.returncode == 0:
        return

    description = (proc.stderr or proc.stdout or f"compute script exited {proc.returncode}").strip()
    if proc.returncode == 1:
        finding = "ort-api-floor-drift"
    elif proc.returncode == 2:
        finding = "ort-api-floor-check-error"
    else:
        finding = "ort-api-floor-check-error"
        description = f"unexpected compute-script exit {proc.returncode}: {description}"
    print(json.dumps(_msg(description, name=finding)), flush=True)


if __name__ == "__main__":
    main()
