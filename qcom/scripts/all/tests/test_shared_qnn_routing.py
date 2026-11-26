# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Run both Linux entry points against the same hermetic provider inventory."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[4]
RUNNER = REPO / "qcom/scripts/all/run_snapshot_accuracy.sh"
COVERAGE = REPO / "qcom/scripts/linux/generate_coverage.sh"
ACCURACY = "QnnAcc_Clip_AccuracyTest"


@pytest.fixture
def provider(tmp_path):
    build = tmp_path / "build/linux-x86_64_ubuntu_22_04-internal-symbols/Release"
    build.mkdir(parents=True)
    (build / "coverage.gcno").touch()
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "sdk.yaml").write_text("version: 2.52\n")
    (build / "CMakeCache.txt").write_text(f"onnxruntime_QNN_HOME:PATH={sdk}\n")
    ort = build / "_deps/ort_core-src"
    ort.mkdir(parents=True)
    (ort / "VERSION_NUMBER").write_text("1.30\n")
    golden = tmp_path / "goldens"
    golden.mkdir()
    (golden / "manifest.json").write_text(
        json.dumps(
            {
                "qairt_version": "2.52",
                "ort_version": "1.30",
                "golden_profile": "linux-x86_64-htp-simulator",
            }
        )
    )
    log = tmp_path / "executions.jsonl"
    binary = build / "onnxruntime_provider_test"
    binary.write_text(f"""#!{sys.executable}
import fnmatch, json, os, sys
from pathlib import Path
inventory = {{
 'QnnHTPBackendTests': ['Legacy'], 'QnnUnit_CoreTest': ['Core'], 'QnnInteg_CoreTest': ['Public'],
 'QnnSnapshot_Clip_OpBuilderTest': ['Match', 'Drift', 'Both'],
 'QnnSnapshot_Clip_SessionTest': ['Both'],
 '{ACCURACY}': ['Match', 'Drift', 'Both', 'Unpaired'],
}}
args = sys.argv[1:]
flt = next((a.split('=',1)[1] for a in args if a.startswith('--gtest_filter=')), '*')
positive, _, negative = flt.partition('-')
def selected(name):
 return any(fnmatch.fnmatchcase(name,p) for p in (positive or '*').split(':')) and not any(fnmatch.fnmatchcase(name,p) for p in negative.split(':') if p)
if '--gtest_list_tests' in args:
 for suite,names in inventory.items():
  chosen = [n for n in names if selected(suite+'.'+n)]
  if chosen:
   print(suite+'.')
   for n in chosen: print('  '+n)
 sys.exit(0)
mode = os.environ.get('FAKE_MODE', 'match')
report = {{'testsuites': []}}
rc = 0
for suite,names in inventory.items():
 cases = []
 for name in names:
  identity = suite+'.'+name
  if not selected(identity): continue
  with Path(os.environ['FAKE_LOG']).open('a') as f: f.write(json.dumps(identity)+'\\n')
  case = {{'name':name,'status':'RUN','result':'COMPLETED'}}
  if suite.startswith('QnnSnapshot_'):
   if mode == 'absent' or (mode == 'session_skip' and '_Session' in suite): case['result']='SKIPPED'
   elif name == 'Drift' and mode in ('mixed','error'):
    case['failures']=[{{'failure':'QNN_SNAPSHOT_DRIFT' if mode == 'mixed' else 'Backend initialization failed'}}]
    rc=1
  if name == 'Unpaired' and os.environ.get('ACCURACY_FAIL') == '1': rc=1
  cases.append(case)
 if cases: report['testsuites'].append({{'name':suite,'testsuite':cases,'errors':0}})
has_snapshot = any(s['name'].startswith('QnnSnapshot_') for s in report['testsuites'])
out = next((a.split('json:',1)[1] for a in args if a.startswith('--gtest_output=json:')), None)
if out:
 Path(out).write_text('{{' if mode == 'malformed' and has_snapshot else json.dumps(report))
if mode == 'malformed' and has_snapshot: rc=1
sys.exit(rc)
""")
    binary.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    (fake_bin / "python3").write_text(f'''#!/bin/bash
if [ "${{1:-}}" = "{REPO}/qcom/scripts/all/package_manager.py" ]; then
 case " $* " in *" --print-bin-dir "*) echo "{tools}" ;; esac
 exit 0
fi
exec "{sys.executable}" "$@"
''')
    (fake_bin / "python3").chmod(0o755)
    (tools / "lcov").write_text(f"""#!{sys.executable}
import json, sys
from pathlib import Path
a=sys.argv[1:]
if '--version' in a: print('lcov fake'); sys.exit(0)
if '--capture' in a:
 with Path({str(log)!r}).open('a') as f: f.write(json.dumps('coverage-capture')+'\\n')
if '--output-file' in a: Path(a[a.index('--output-file')+1]).touch()
""")
    (tools / "genhtml").write_text(f"""#!{sys.executable}
import sys
from pathlib import Path
a=sys.argv[1:]
Path(a[a.index('--output-directory')+1]).mkdir(parents=True,exist_ok=True)
""")
    (tools / "lcov_cobertura").write_text(f"""#!{sys.executable}
import sys
from pathlib import Path
a=sys.argv[1:]
Path(a[a.index('--output')+1]).write_text('<coverage/>')
""")
    for tool in tools.iterdir():
        tool.chmod(0o755)
    env = os.environ | {
        "PYTHON_BIN": sys.executable,
        "FAKE_LOG": str(log),
        "QNN_UT_SNAPSHOT_GOLDEN_DIR": str(golden),
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "ORT_BUILD_TOOLS_PATH": str(tmp_path / "packages"),
    }
    for key in ("GITHUB_OUTPUT", "GITHUB_ENV", "GITHUB_STEP_SUMMARY"):
        env.pop(key, None)
    return build, env, log


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX Linux provider fixture")
@pytest.mark.parametrize(
    "mode,force,expected,exit_code",
    [
        ("match", False, {"Unpaired"}, 0),
        ("mixed", False, {"Drift", "Unpaired"}, 0),
        ("session_skip", False, {"Both", "Unpaired"}, 0),
        ("absent", False, {"Match", "Drift", "Both", "Unpaired"}, 0),
        ("match", True, {"Match", "Drift", "Both", "Unpaired"}, 0),
        ("malformed", False, {"Match", "Drift", "Both", "Unpaired"}, 1),
        ("error", False, {"Match", "Drift", "Both", "Unpaired"}, 1),
    ],
)
def test_linux_and_coverage_share_inventory_selection_and_failure_paths(provider, mode, force, expected, exit_code):
    build, env, log = provider
    env = env | {"FAKE_MODE": mode}
    if mode == "absent":
        env = env | {"QNN_UT_SNAPSHOT_GOLDEN_DIR": str(build / "absent-goldens")}
    baseline = subprocess.run(
        [
            "bash",
            "-c",
            'source qcom/scripts/all/qnn_test_stages.sh; qnn_run_provider "$1" "$2" --gtest_filter="*:-${QNN_NON_ROUTED_EXCLUSIONS}"',
            "fixture",
            str(build / "onnxruntime_provider_test"),
            str(build),
        ],
        cwd=REPO,
        env=env,
        check=False,
    )
    assert baseline.returncode == 0
    options = ["--force-accuracy"] if force else []
    regular = subprocess.run(
        ["bash", str(RUNNER), f"--build-dir={build}", *options],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert regular.returncode == exit_code, regular.stderr
    regular_events = [json.loads(line) for line in log.read_text().splitlines()]
    assert {name.split(".")[-1] for name in regular_events if name.startswith(ACCURACY)} == expected
    log.unlink()
    coverage = subprocess.run(
        ["bash", str(COVERAGE), f"--build-dir={build.parent}", "--config=Release", *options],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert coverage.returncode == exit_code, coverage.stderr
    coverage_events = [json.loads(line) for line in log.read_text().splitlines()]
    assert coverage_events[-1] == "coverage-capture"
    assert coverage_events[:-1] == regular_events
    assert len(coverage_events[:-1]) == len(set(coverage_events[:-1]))
    assert (build / "coverage/coverage.xml").exists()


def test_active_ubuntu_caller_assigns_each_tier_to_one_owner():
    workflow = yaml.safe_load((REPO / ".github/workflows/qualcomm-internal-build-and-test-single-os.yml").read_text())
    caller = yaml.safe_load((REPO / ".github/workflows/qualcomm-internal-build-and-test.yml").read_text())
    active = next(
        job["with"]
        for job in caller["jobs"].values()
        if job.get("with", {}).get("target_arch") == "x86_64_ubuntu_22_04"
    )
    assert active["qnn_internal_test_symbols"] and active["run_tests"] and active["build_test_same_runner"]
    steps = workflow["jobs"]["build"]["steps"]
    baseline = next(step for step in steps if step.get("id") == "platform_validation")
    routed = next(step for step in steps if step.get("id") == "linux_snapshot_routing")
    assert "x86_64_ubuntu_22_04" in baseline["env"]["QNN_UT_TWO_PASS_SNAPSHOT_ACCURACY"]
    assert "x86_64_ubuntu_22_04" in routed["if"]
    assert "outputs.ready" not in routed["if"]  # artifact preparation cannot suppress owned tests
    assert "run_snapshot_accuracy.sh" in routed["run"]
    assert "observe_provider_routing.py" not in str(steps)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX Linux provider fixture")
@pytest.mark.parametrize(
    "mutation,expected_exit",
    [
        ("source", 0),
        ("versions", 0),
        ("observation", 0),
        ("incomplete", 0),
        ("digest", 1),
        ("inventory", 1),
        ("pair_type", 1),
    ],
)
def test_selector_rejects_untrusted_or_incomplete_verdicts(provider, mutation, expected_exit):
    build, env, _ = provider
    subprocess.run(
        ["bash", str(RUNNER), f"--build-dir={build}"], cwd=REPO, env=env, capture_output=True, text=True, check=True
    )
    results = build / "snapshot_accuracy_results"
    verdict = results / "selection.json"
    data = json.loads(verdict.read_text())
    source = data["producer"]["source_commit_sha"]
    if mutation == "source":
        data["producer"]["source_commit_sha"] = "0" * 40
    elif mutation == "versions":
        data["producer"]["qairt_version"] = "different"
    elif mutation == "observation":
        data["mode"] = "provider_observation"
    elif mutation == "incomplete":
        data["selection_complete"] = False
    elif mutation == "inventory":
        data["accuracy_decisions"].pop()
    elif mutation == "pair_type":
        data["accuracy_decisions"][0]["snapshot_test_ids"] = {}
    if mutation != "digest":
        data.pop("payload_sha256")
        data["payload_sha256"] = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    else:
        data["mode"] = "tampered"
    verdict.write_text(json.dumps(data))
    output = results / "rejected-filter.txt"
    actual = subprocess.run(
        [
            sys.executable,
            str(REPO / "qcom/scripts/all/select_accuracy_from_routing.py"),
            f"--verdict={verdict}",
            f"--accuracy-list={results / 'routing_accuracy_tests.txt'}",
            f"--output={output}",
            f"--source-commit={source}",
            "--qairt-version=2.52",
            "--ort-version=1.30",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert actual.returncode == expected_exit, actual.stderr
    if expected_exit:
        assert not output.exists()
    else:
        assert set(output.read_text().split(":")) == {
            f"{ACCURACY}.{name}" for name in ("Match", "Drift", "Both", "Unpaired")
        }


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX Linux provider fixture")
@pytest.mark.parametrize("artifact_ready,upstream_exit", [(False, "0"), (True, "0"), (True, "1")])
def test_actual_routing_step_runs_without_artifact_and_records_baseline_failure(
    provider, artifact_ready, upstream_exit
):
    build, env, log = provider
    workflow = yaml.safe_load((REPO / ".github/workflows/qualcomm-internal-build-and-test-single-os.yml").read_text())
    step = next(step for step in workflow["jobs"]["build"]["steps"] if step.get("id") == "linux_snapshot_routing")
    output = build.parents[2] / "github-output"
    artifact_dir = build.parents[2] / "artifact"
    artifact_dir.mkdir()
    actual = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=REPO,
        env=env
        | {
            "GITHUB_WORKSPACE": str(build.parents[2]),
            "BUILD_DIR_ARCH": "x86_64_ubuntu_22_04-internal-symbols",
            "BUILD_CONFIG_DIR": "Release",
            "GITHUB_OUTPUT": str(output),
            "QNN_ROUTING_OUTPUT_DIR": str(artifact_dir),
            "QNN_ROUTING_OUTPUT_READY": str(artifact_ready).lower(),
            "QNN_ROUTING_UPSTREAM_EXIT": upstream_exit,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert actual.returncode == 0, actual.stderr
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(name.startswith("QnnSnapshot_") for name in events) == 4
    assert [name for name in events if name.startswith(ACCURACY)] == [ACCURACY + ".Unpaired"]
    if artifact_ready:
        result_path = Path(output.read_text().strip().split("=", 1)[1])
        result = json.loads(result_path.read_text())
        assert result["runner_exit_code"] == int(upstream_exit)
    else:
        assert not output.exists()
        assert not (artifact_dir / "accuracy_routing.json").exists()


@pytest.mark.parametrize("exit_code,expected", [(1, 0), (2, 1), (124, 1), (139, 1)])
def test_crash_or_timeout_cannot_be_reclassified_as_golden_drift(tmp_path, exit_code, expected):
    report = tmp_path / "snapshot.json"
    report.write_text(json.dumps({"testsuites": [{"testsuite": [{"failures": [{"failure": "QNN_SNAPSHOT_DRIFT"}]}]}]}))
    actual = subprocess.run(
        [
            sys.executable,
            str(REPO / "qcom/scripts/all/select_accuracy_from_routing.py"),
            f"--snapshot-json={report}",
            f"--snapshot-exit={exit_code}",
            "--check-snapshot-exit",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert actual.returncode == expected, actual.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX Linux provider fixture")
def test_failed_selection_does_not_reuse_a_previous_verdict(provider):
    build, env, log = provider
    command = ["bash", str(RUNNER), f"--build-dir={build}"]
    subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=True)
    results = build / "snapshot_accuracy_results"
    assert (results / "selection.json").exists()
    log.unlink()
    failed = subprocess.run(
        command, cwd=REPO, env=env | {"FAKE_MODE": "error"}, capture_output=True, text=True, check=False
    )
    assert failed.returncode == 1
    assert not (results / "selection.json").exists()
    assert not (results / "accuracy_filter.txt").exists()
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(name.startswith(ACCURACY) for name in events) == 4
