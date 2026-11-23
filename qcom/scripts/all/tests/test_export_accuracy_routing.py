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
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "export_accuracy_routing.py"
RUNNER = SCRIPT.with_name("run_snapshot_accuracy.sh")
SNAPSHOT = "QnnSnapshot_Clip_OpBuilderTest"
ACCURACY = "QnnAcc_Clip_AccuracyTest"


def export(tmp_path, snapshot, manifest, extra=(), snapshot_list=None):
    (tmp_path / "snapshots.txt").write_text(snapshot_list or f"{SNAPSHOT}.\n  Case\n")
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


def test_pairing_excludes_other_cases_and_variants(tmp_path):
    other_variant = "QnnSnapshot_Clip_OpBuilder_QDQFloatTest"
    drift = {**PASS, "failures": [{"failure": "[QNN_SNAPSHOT_DRIFT]"}]}
    data = {
        "testsuites": [
            {"name": SNAPSHOT, "testsuite": [{"name": "Case", **PASS}, {"name": "Other", **drift}]},
            {"name": other_variant, "testsuite": [{"name": "Case", **drift}]},
        ]
    }
    result = export(
        tmp_path,
        data,
        MANIFEST,
        ("--selection-complete", "--target-groups=Clip"),
        snapshot_list=f"{SNAPSHOT}.\n  Case\n  Other\n{other_variant}.\n  Case\n",
    )
    decision = result["accuracy_decisions"][0]
    assert decision["snapshot_test_ids"] == [SNAPSHOT + ".Case"]
    assert decision["status"] == "match"
    assert decision["runner_action"] == "run"


def test_pairing_includes_both_snapshot_tiers_for_the_same_case(tmp_path):
    session = "QnnSnapshot_Clip_SessionTest"
    data = {"testsuites": [{"name": suite, "testsuite": [{"name": "Case", **PASS}]} for suite in (SNAPSHOT, session)]}
    result = export(
        tmp_path,
        data,
        MANIFEST,
        ("--selection-complete",),
        snapshot_list=f"{SNAPSHOT}.\n  Case\n{session}.\n  Case\n",
    )
    assert result["accuracy_decisions"][0]["snapshot_test_ids"] == [SNAPSHOT + ".Case", session + ".Case"]


def test_same_group_without_exact_pair_is_unmapped(tmp_path):
    data = {"testsuites": [{"name": SNAPSHOT, "testsuite": [{"name": "Other", **PASS}]}]}
    result = export(
        tmp_path,
        data,
        MANIFEST,
        ("--selection-complete", "--target-groups=Clip"),
        snapshot_list=f"{SNAPSHOT}.\n  Other\n",
    )
    decision = result["accuracy_decisions"][0]
    assert decision["snapshot_test_ids"] == []
    assert decision["status"] == "unmapped"
    assert decision["runner_action"] == "run"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake executable fixture")
@pytest.mark.parametrize("drift, accuracy_exit", [(False, 0), (True, 0), (True, 1)])
def test_runner_exports_observed_selection_and_preserves_exit(tmp_path, drift, accuracy_exit):
    binary_dir = tmp_path / "build" / "Release"
    binary_dir.mkdir(parents=True)
    provider = binary_dir / "onnxruntime_provider_test"
    provider.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path
args = sys.argv[1:]
is_snapshot = any('QnnSnapshot_' in arg for arg in args)
if '--gtest_list_tests' in args:
    print('{SNAPSHOT}.' if is_snapshot else '{ACCURACY}.')
    print('  Case')
    raise SystemExit(0)
with Path(os.environ['FAKE_EXECUTION_LOG']).open('a') as log:
    log.write(json.dumps(args) + '\\n')
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
    execution_log = tmp_path / "executions.log"
    github_output = tmp_path / "github_output"
    env = os.environ | {
        "PYTHON_BIN": sys.executable,
        "FAKE_EXECUTION_LOG": str(execution_log),
        "GITHUB_OUTPUT": str(github_output),
    }
    baseline = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    baseline_execution = execution_log.read_text()
    execution_log.unlink()
    actual = subprocess.run(
        [*command, f"--emit-routing-result={output}"], env=env, capture_output=True, text=True, check=False
    )
    assert baseline.returncode == actual.returncode == accuracy_exit
    result = json.loads(output.read_text())
    assert result["selection_complete"]
    assert result["accuracy_decisions"][0]["runner_action"] == ("run" if drift else "skip")
    assert result["runner_exit_code"] == accuracy_exit
    assert execution_log.read_text() == baseline_execution
    assert github_output.read_text().strip() == f"routing_result_path={output}"

    # A failed write must preserve the same test outcome and leave no usable file.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    github_output.unlink()
    execution_log.unlink()
    failed_export = subprocess.run(
        [*command, f"--emit-routing-result={blocker / 'routing.json'}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed_export.returncode == accuracy_exit
    assert "Routing artifact export failed" in failed_export.stderr
    assert execution_log.read_text() == baseline_execution
    assert not github_output.exists()

    # Preparation errors at the output or temporary path must not stop tests.
    output_directory = tmp_path / "output-directory"
    output_directory.mkdir()
    temporary_blocked = tmp_path / "temporary-blocked.json"
    temporary_blocked.with_suffix(".json.tmp").mkdir()
    for destination in (output_directory, temporary_blocked):
        execution_log.unlink()
        preparation_failure = subprocess.run(
            [*command, f"--emit-routing-result={destination}"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert preparation_failure.returncode == accuracy_exit
        assert execution_log.read_text() == baseline_execution
        assert "Routing artifact preparation failed" in preparation_failure.stderr
        assert not github_output.exists()

    # Inject canonicalization/cleanup failure for a regular stale destination.
    for tool, content in (
        ("realpath", '#!/bin/sh\nif [ "$1" = "-m" ]; then exit 73; fi\nexec /usr/bin/realpath "$@"\n'),
        ("rm", '#!/bin/sh\ncase "$*" in *routing.json*) exit 73 ;; esac\nexec /bin/rm "$@"\n'),
    ):
        fault_bin = tmp_path / tool
        fault_bin.mkdir()
        executable = fault_bin / tool
        executable.write_text(content)
        executable.chmod(0o755)
        output.write_text("stale result")
        execution_log.unlink()
        preparation_failure = subprocess.run(
            [*command, f"--emit-routing-result={output}"],
            env=env | {"PATH": str(fault_bin) + os.pathsep + env["PATH"]},
            capture_output=True,
            text=True,
            check=False,
        )
        assert preparation_failure.returncode == accuracy_exit
        assert execution_log.read_text() == baseline_execution
        assert output.read_text() == "stale result"
        assert not github_output.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX workflow shell fixture")
@pytest.mark.parametrize("preparation_fails", [False, True])
@pytest.mark.parametrize("test_exit", [0, 1])
def test_ci_directory_preparation_preserves_test_execution(tmp_path, preparation_fails, test_exit):
    workflow_path = RUNNER.parents[3] / ".github/workflows/qualcomm-internal-build-and-test-single-os.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    steps = workflow["jobs"]["build"]["steps"]
    step = next(item for item in steps if item.get("id") == "linux_snapshot_routing")
    upload = next(item for item in steps if item.get("name") == "Upload Linux snapshot routing result")
    cleanup = next(item for item in steps if item.get("name") == "Clean up routing output directory")
    assert upload["with"]["path"] == "${{ steps.linux_snapshot_routing.outputs.routing_result_path }}"
    assert "outputs.routing_result_path != ''" in upload["if"]
    assert "always()" in cleanup["if"]

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bash = fake_bin / "bash"
    execution_log = tmp_path / "command.log"
    fake_bash.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_EXECUTION_LOG"\nexit "$FAKE_TEST_EXIT"\n')
    fake_bash.chmod(0o755)
    if preparation_fails:
        fake_mktemp = fake_bin / "mktemp"
        fake_mktemp.write_text("#!/bin/sh\nexit 73\n")
        fake_mktemp.chmod(0o755)
    github_output = tmp_path / "github_output"
    env = os.environ | {
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(github_output),
        "GITHUB_WORKSPACE": str(tmp_path),
        "BUILD_DIR_ARCH": "x86_64-internal-symbols",
        "BUILD_CONFIG_DIR": "Release",
        "FAKE_EXECUTION_LOG": str(execution_log),
        "FAKE_TEST_EXIT": str(test_exit),
    }
    result = subprocess.run(
        ["/bin/bash", "-e", "-o", "pipefail", "-c", step["run"]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == test_exit
    arguments = execution_log.read_text().splitlines()
    assert arguments[0] == "qcom/scripts/all/run_snapshot_accuracy.sh"
    assert arguments[1] == f"--build-dir={tmp_path}/build/linux-x86_64-internal-symbols/Release"
    if preparation_fails:
        assert len(arguments) == 2
        assert not github_output.exists()
        assert "Routing directory preparation failed" in result.stdout
    else:
        directory = Path(github_output.read_text().strip().split("=", 1)[1])
        assert directory.parent == tmp_path
        assert directory.name.startswith("qnn-routing-123-2-")
        assert arguments[2] == f"--emit-routing-result={directory}/accuracy_routing.json"
        cleanup_env = env | {"QNN_ROUTING_OUTPUT_DIR": str(directory)}
        unrelated = tmp_path / "keep.txt"
        unrelated.write_text("unrelated")
        fake_rm = fake_bin / "rm"
        fake_rm.write_text("#!/bin/sh\nexit 73\n")
        fake_rm.chmod(0o755)
        cleanup_failure = subprocess.run(
            ["/bin/bash", "-e", "-o", "pipefail", "-c", cleanup["run"]],
            env=cleanup_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert cleanup_failure.returncode == 0
        assert "Routing directory cleanup failed" in cleanup_failure.stdout
        assert directory.exists()
        fake_rm.unlink()
        cleaned = subprocess.run(
            ["/bin/bash", "-e", "-o", "pipefail", "-c", cleanup["run"]],
            env=cleanup_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert cleaned.returncode == 0
        assert not directory.exists()
        assert unrelated.read_text() == "unrelated"
