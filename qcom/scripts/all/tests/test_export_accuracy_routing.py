# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Regression tests for exporting observed selection without changing routing."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "export_accuracy_routing.py"
RUNNER = SCRIPT.with_name("run_snapshot_accuracy.sh")
SNAPSHOT = "QnnSnapshot_Clip_OpBuilderTest"
ACCURACY = "QnnAcc_Clip_AccuracyTest"


def export(tmp_path, snapshot, manifest, extra=()):
    (tmp_path / "snapshots.txt").write_text(f"{SNAPSHOT}.\n  Case\n")
    (tmp_path / "accuracy.txt").write_text(f"{ACCURACY}.\n  Case\nQnnAcc_Other_AccuracyTest.\n  Other\n")
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"--snapshot-json={tmp_path / 'snapshot.json'}",
            f"--snapshot-list={tmp_path / 'snapshots.txt'}",
            f"--accuracy-list={tmp_path / 'accuracy.txt'}",
            f"--manifest={tmp_path / 'manifest.json'}",
            f"--output={tmp_path / 'result.json'}",
            "--source-commit=" + "a" * 40,
            "--qairt-version=2.52",
            "--ort-version=1.30",
            "--mode=compare",
            "--runner-exit=0",
            "--snapshot-exit=0",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((tmp_path / "result.json").read_text())


def snapshot(case):
    return {"testsuites": [{"name": SNAPSHOT, "testsuite": [{"name": "Case", **case}]}]}


MANIFEST = {"qairt_version": "2.52", "ort_version": "1.30", "golden_profile": "linux-x86_64-htp-simulator"}
PASS = {"status": "RUN", "result": "COMPLETED"}


@pytest.mark.parametrize(
    "case, manifest, targets, expected",
    [
        (PASS, MANIFEST, "", "match"),
        ({**PASS, "failures": [{"failure": "[QNN_SNAPSHOT_DRIFT]"}]}, MANIFEST, "Clip", "drift"),
        ({**PASS, "failures": [{"failure": "[QNN_GOLDEN_ABSENT]"}]}, MANIFEST, "Clip", "untrusted"),
        ({"status": "RUN", "result": "SKIPPED"}, MANIFEST, "Clip", "untrusted"),
        (PASS, {}, "", "untrusted"),
        (PASS, {**MANIFEST, "qairt_version": "different"}, "", "untrusted"),
    ],
)
def test_evidence_status_is_separate_from_observed_selection(tmp_path, case, manifest, targets, expected):
    result = export(tmp_path, snapshot(case), manifest, ("--selection-complete", f"--target-groups={targets}"))
    decision, unmapped = result["accuracy_decisions"]
    assert decision["status"] == expected
    assert decision["runner_action"] == ("run" if targets else "skip")
    assert decision["snapshot_test_ids"] == [SNAPSHOT + ".Case"]
    assert unmapped["runner_action"] == "out_of_scope"
    assert unmapped["status"] == "unmapped"
    assert result["producer"]["target_htp_arch"] is None
    digest = result.pop("payload_sha256")
    assert digest == hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_missing_case_and_incomplete_selection_cannot_claim_match(tmp_path):
    result = export(tmp_path, {"testsuites": []}, MANIFEST)
    assert result["accuracy_decisions"][0]["status"] == "untrusted"
    assert result["accuracy_decisions"][0]["runner_action"] == "unknown"


def test_duplicate_snapshot_identity_fails_closed(tmp_path):
    data = snapshot(PASS)
    data["testsuites"].append(data["testsuites"][0])
    result = export(tmp_path, data, MANIFEST, ("--selection-complete",))
    assert result["accuracy_decisions"][0]["status"] == "untrusted"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake executable fixture")
@pytest.mark.parametrize("drift, accuracy_exit", [(False, 0), (True, 0), (True, 1)])
def test_runner_exports_observed_selection_and_preserves_exit(tmp_path, drift, accuracy_exit):
    binary_dir = tmp_path / "build" / "Release"
    binary_dir.mkdir(parents=True)
    provider = binary_dir / "onnxruntime_provider_test"
    provider.write_text(
        f"""#!{sys.executable}
import json
import sys
from pathlib import Path
args = sys.argv[1:]
is_snapshot = any('QnnSnapshot_' in arg for arg in args)
if '--gtest_list_tests' in args:
    print('{SNAPSHOT}.' if is_snapshot else '{ACCURACY}.')
    print('  Case')
    raise SystemExit(0)
case = {{'name': 'Case', 'status': 'RUN', 'result': 'COMPLETED'}}
if is_snapshot and {drift!r}:
    case['failures'] = [{{'failure': '[QNN_SNAPSHOT_DRIFT]'}}]
suite = {{'name': '{SNAPSHOT}' if is_snapshot else '{ACCURACY}',
          'failures': int(is_snapshot and {drift!r}), 'errors': 0, 'testsuite': [case]}}
output = next(arg.removeprefix('--gtest_output=json:') for arg in args if arg.startswith('--gtest_output=json:'))
Path(output).write_text(json.dumps({{'testsuites': [suite]}}))
raise SystemExit(int({drift!r}) if is_snapshot else {accuracy_exit})
"""
    )
    provider.chmod(0o755)
    output = tmp_path / "routing.json"
    output.write_text("stale result")
    command = ["bash", str(RUNNER), f"--build-dir={binary_dir}"]
    env = os.environ | {"PYTHON_BIN": sys.executable}
    baseline = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    actual = subprocess.run(
        [*command, f"--emit-routing-result={output}"], env=env, capture_output=True, text=True, check=False
    )
    assert baseline.returncode == actual.returncode == accuracy_exit
    result = json.loads(output.read_text())
    assert result["selection_complete"]
    assert result["accuracy_decisions"][0]["runner_action"] == ("run" if drift else "skip")
    assert result["runner_exit_code"] == accuracy_exit

    # A failed write must preserve the same test outcome and leave no usable file.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    failed_export = subprocess.run(
        [*command, f"--emit-routing-result={blocker / 'routing.json'}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed_export.returncode == accuracy_exit
    assert "Routing artifact export failed" in failed_export.stderr
