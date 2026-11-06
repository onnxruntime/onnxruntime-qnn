# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Tests for snapshot/accuracy routing verification."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "verify_accuracy_routing.py"


def run_verifier(
    tmp_path: Path,
    summary: str,
    *,
    actual_count: int | None,
    crlf: bool = False,
) -> subprocess.CompletedProcess[str]:
    summary_path = tmp_path / "accuracy_gate_summary.txt"
    results_path = tmp_path / "accuracy_results.json"
    newline = "\r\n" if crlf else "\n"
    summary_path.write_bytes((newline.join(summary.splitlines()) + newline).encode())
    if actual_count is not None:
        test_cases = [{"name": f"Case/{index}"} for index in range(actual_count)]
        testsuites = [{"name": "QnnAcc_Clip_AccuracyTest", "testsuite": test_cases}] if test_cases else []
        results_path.write_text(json.dumps({"testsuites": testsuites}))
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"--summary={summary_path}",
            f"--accuracy-results={results_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_accepts_matched_versions_with_windows_crlf(tmp_path: Path) -> None:
    result = run_verifier(
        tmp_path,
        """[VERSION] qairt: current=2.51.0 manifest=2.51.0 | ort: current=1.30.0 manifest=1.30.0 -> MATCH
[ROUTING] 16 accuracy case(s): run 0, skip 16.""",
        actual_count=0,
        crlf=True,
    )

    assert result.returncode == 0, result.stderr
    assert "16 skipped accuracy case(s)" in result.stdout


def test_accepts_matched_versions_when_all_accuracy_cases_run(tmp_path: Path) -> None:
    result = run_verifier(
        tmp_path,
        """[VERSION] qairt: current=2.51.0 manifest=2.51.0 | ort: current=1.30.0 manifest=1.30.0 -> MATCH
[ROUTING] 16 accuracy case(s): run 16, skip 0.""",
        actual_count=16,
    )

    assert result.returncode == 0, result.stderr
    assert "16 selected and 0 skipped accuracy case(s)" in result.stdout


def test_accepts_version_mismatch_when_all_accuracy_cases_run(tmp_path: Path) -> None:
    result = run_verifier(
        tmp_path,
        """[VERSION] qairt: current=2.52.0 manifest=2.51.0 | ort: current=1.30.0 manifest=1.30.0 -> MISMATCH/absent => full accuracy run
[ROUTING] 16 accuracy case(s): run 16, skip 0.""",
        actual_count=16,
    )

    assert result.returncode == 0, result.stderr
    assert "all 16 accuracy case(s) executed" in result.stdout


def test_rejects_version_mismatch_when_accuracy_cases_are_skipped(tmp_path: Path) -> None:
    result = run_verifier(
        tmp_path,
        """[VERSION] qairt: current=2.52.0 manifest=2.51.0 | ort: current=1.30.0 manifest=1.30.0 -> MISMATCH/absent => full accuracy run
[ROUTING] 16 accuracy case(s): run 15, skip 1.""",
        actual_count=15,
    )

    assert result.returncode != 0
    assert "did not fall back to the full accuracy suite" in result.stderr


def test_rejects_accuracy_result_count_mismatch(tmp_path: Path) -> None:
    result = run_verifier(
        tmp_path,
        """[VERSION] qairt: current=2.51.0 manifest=2.51.0 | ort: current=1.30.0 manifest=1.30.0 -> MATCH
[ROUTING] 16 accuracy case(s): run 16, skip 0.""",
        actual_count=0,
    )

    assert result.returncode != 0
    assert "expected 16, executed 0" in result.stderr


def test_rejects_missing_accuracy_results(tmp_path: Path) -> None:
    result = run_verifier(
        tmp_path,
        """[VERSION] qairt: current=2.51.0 manifest=2.51.0 | ort: current=1.30.0 manifest=1.30.0 -> MATCH
[ROUTING] 16 accuracy case(s): run 0, skip 16.""",
        actual_count=None,
    )

    assert result.returncode != 0
    assert "Accuracy test results were not produced" in result.stderr
