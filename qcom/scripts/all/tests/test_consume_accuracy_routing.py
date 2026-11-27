# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Target routing stays observe-only and fails safe on incompatible evidence."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "consume_accuracy_routing.py"
SPEC = importlib.util.spec_from_file_location("routing_consumer", SCRIPT)
CONSUMER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONSUMER)
CASE = "QnnAcc_Clip_AccuracyTest.Case"
SNAPSHOT = "QnnSnapshot_Clip_OpBuilderTest.Case"
IDENTITY = {
    "source_commit_sha": "a" * 40,
    "workflow_sha": "a" * 40,
    "repository": "owner/repo",
    "workflow_ref": "owner/repo/workflow@refs/pull/1/merge",
    "run_id": "10",
    "run_attempt": "1",
}


def sign(artifact):
    payload = {key: value for key, value in artifact.items() if key != "payload_sha256"}
    artifact["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture
def inputs():
    artifact = {
        "evidence_granularity": "testcase",
        "decision_granularity": "operator_group",
        "producer": {
            **IDENTITY,
            "job": "build",
            "qairt_version": "2.52",
            "ort_version": "1.30",
            "golden_profile": "linux-simulator",
            "target_htp_arch": 73,
        },
        "golden_manifest": {"qairt_version": "2.52", "ort_version": "1.30", "golden_profile": "linux-simulator"},
        "versions_match": True,
        "mode": "compare",
        "selection_complete": True,
        "runner_exit_code": 0,
        "snapshots": [{"test_id": SNAPSHOT, "status": "match"}],
        "accuracy_decisions": [
            {"accuracy_test_id": CASE, "snapshot_test_ids": [SNAPSHOT], "status": "match", "runner_action": "skip"}
        ],
    }
    target = {
        "schema_version": 1,
        "identity": IDENTITY,
        "build": {"schema_version": 1, "identity": IDENTITY, "qairt_version": "2.52", "ort_version": "1.30"},
        "target_platform": "windows-arm64",
        "target_htp_arch": 73,
        "accuracy_outcomes": {CASE: "passed"},
        "errors": [],
    }
    policy = {
        "schema_version": 1,
        "default": "unresolved",
        "target_platform": "windows-arm64",
        "target_htp_arch": 73,
        "producer_golden_profile": "linux-simulator",
        "cases": {
            CASE: {
                "responsibility": "simulator_gated_common",
                "reviewed": True,
                "equivalence_evidence": "reviewed-evidence",
                "snapshot_test_ids": [SNAPSHOT],
            }
        },
    }
    sign(artifact)
    return artifact, target, policy


def test_match_proposes_skip_without_changing_execution(inputs):
    assert "schema_version" not in inputs[0]
    result = CONSUMER.select_shadow(*inputs, IDENTITY)
    assert result["would_skip"] == 1
    assert not result["target_execution_changed"]
    assert result["false_skip_candidates"] == 0


def test_provider_observation_never_authorizes_skip(inputs):
    artifact, target, policy = inputs
    artifact.update(mode="provider_observation", decision_granularity="none", selection_complete=False)
    sign(artifact)

    result = CONSUMER.select_shadow(artifact, target, policy, IDENTITY)

    assert result["would_skip"] == 0
    assert result["would_run"] == 1
    assert "provider_observation_does_not_authorize_skips" in result["fallback_reasons"]
    assert "invalid_artifact_contract" not in result["fallback_reasons"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX Linux producer fixture")
def test_actual_provider_observation_contract_is_consumed(inputs, tmp_path):
    repo = SCRIPT.parents[3]
    build = tmp_path / "build"
    build.mkdir()
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "sdk.yaml").write_text("version: 2.52.0\n")
    ort = tmp_path / "ort"
    ort.mkdir()
    (ort / "VERSION_NUMBER").write_text("1.30.0\n")
    (build / "CMakeCache.txt").write_text(f"onnxruntime_QNN_HOME:PATH={sdk}\nonnxruntime_ORT_HOME:PATH={ort}\n")
    (build / "onnxruntime_provider_test.Release.results.xml").write_text(
        '<testsuites><testsuite name="QnnSnapshot_Clip_OpBuilderTest">'
        '<testcase name="Case" status="run" result="completed" /></testsuite></testsuites>'
    )
    binary = build / "onnxruntime_provider_test"
    binary.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "assert '--gtest_list_tests' in sys.argv\n"
        "if any('QnnSnapshot_' in arg for arg in sys.argv):\n"
        " print('QnnSnapshot_Clip_OpBuilderTest.\\n  Case')\n"
        "else:\n print('QnnAcc_Clip_AccuracyTest.\\n  Case')\n"
    )
    binary.chmod(0o755)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    expected = IDENTITY | {"source_commit_sha": source, "workflow_sha": source}
    env = os.environ | {
        "GITHUB_SHA": source,
        "GITHUB_REPOSITORY": expected["repository"],
        "GITHUB_WORKFLOW_REF": expected["workflow_ref"],
        "GITHUB_RUN_ID": expected["run_id"],
        "GITHUB_RUN_ATTEMPT": expected["run_attempt"],
        "GITHUB_JOB": "build",
    }
    env.pop("QNN_UT_SNAPSHOT_GOLDEN_DIR", None)
    output = tmp_path / "observation.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT.with_name("observe_provider_routing.py")),
            f"--build-dir={build}",
            "--config=Release",
            f"--output={output}",
            "--validation-exit=0",
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(output.read_text())
    _, target, policy = inputs
    target["identity"] = expected
    target["build"].update(identity=expected, qairt_version="2.52.0", ort_version="1.30.0")

    result = CONSUMER.select_shadow(artifact, target, policy, expected)

    assert artifact["mode"] == "provider_observation"
    assert "schema_version" not in artifact
    assert result["would_skip"] == 0
    assert result["would_run"] == 1
    assert "provider_observation_does_not_authorize_skips" in result["fallback_reasons"]
    assert "artifact_integrity_failure" not in result["fallback_reasons"]
    assert "invalid_artifact_contract" not in result["fallback_reasons"]
    assert not any("identity_mismatch" in reason for reason in result["fallback_reasons"])


def test_shadow_workflow_uses_existing_jobs_and_active_producer_artifact():
    repo = SCRIPT.parents[3]
    workflow = yaml.safe_load((repo / ".github/workflows/qualcomm-internal-build-and-test.yml").read_text())
    jobs = workflow["jobs"]
    shadow = jobs["windows-arm64-v73-routing-shadow"]
    assert all(name in jobs for name in shadow["needs"])
    assert "build-and-test-linux-x86_64-ubuntu-22-04" in shadow["needs"]
    linux = next(step for step in shadow["steps"] if step.get("name") == "Download Linux routing result")
    single_os = yaml.safe_load((repo / ".github/workflows/qualcomm-internal-build-and-test-single-os.yml").read_text())
    producer = next(
        step
        for step in single_os["jobs"]["build"]["steps"]
        if step.get("name") == "Upload Linux snapshot routing result"
    )
    name = producer["with"]["name"].replace("${{ inputs.target_os }}", "linux")
    name = name.replace("${{ inputs.target_arch }}", "x86_64_ubuntu_22_04")
    name = name.replace("${{ inputs.build_config }}", "Release")
    assert linux["with"]["name"] == name
    windows = jobs["build-and-test-windows-arm64-internal-symbols"]["with"]
    assert windows["qnn_internal_test_symbols"] is True
    assert windows["run_tests"] is True
    assert windows["build_test_same_runner"] is False


@pytest.mark.parametrize(
    "change",
    [
        "commit",
        "attempt",
        "workflow",
        "integrity",
        "contract",
        "workflow_sha",
        "profile",
        "architecture",
        "qairt",
        "ort",
        "force",
        "incomplete",
        "producer_failure",
        "target_identity",
        "build_identity",
        "target_arch",
        "target_error",
        "drift",
        "missing_snapshot",
        "missing_mapping",
        "unresolved",
        "always_run",
        "unreviewed",
    ],
)
def test_uncertainty_proposes_run(inputs, change):
    artifact, target, policy = copy.deepcopy(inputs)
    if change in ("commit", "attempt", "workflow"):
        key = {"commit": "source_commit_sha", "attempt": "run_attempt", "workflow": "workflow_ref"}[change]
        artifact["producer"][key] = "other"
    elif change == "contract":
        artifact["evidence_granularity"] = "unknown"
    elif change == "workflow_sha":
        artifact["producer"]["workflow_sha"] = "other"
    elif change == "profile":
        artifact["producer"]["golden_profile"] = "other"
    elif change == "architecture":
        artifact["producer"]["target_htp_arch"] = None
    elif change in ("qairt", "ort"):
        target["build"][change + "_version"] = "other"
    elif change == "force":
        artifact["mode"] = "force_accuracy"
    elif change == "incomplete":
        artifact["selection_complete"] = False
    elif change == "producer_failure":
        artifact["runner_exit_code"] = 1
    elif change == "target_identity":
        target["identity"] = {**IDENTITY, "run_id": "other"}
    elif change == "build_identity":
        target["build"]["identity"] = {**IDENTITY, "source_commit_sha": "other"}
    elif change == "target_arch":
        target["target_htp_arch"] = 81
    elif change == "target_error":
        target["errors"] = ["missing report"]
    elif change == "drift":
        artifact["accuracy_decisions"][0]["status"] = "drift"
    elif change == "missing_snapshot":
        artifact["snapshots"] = []
    elif change == "missing_mapping":
        policy["cases"] = {}
    elif change == "unresolved":
        policy["cases"][CASE]["responsibility"] = "unresolved"
    elif change == "always_run":
        policy["cases"][CASE]["responsibility"] = "always_run_platform_specific"
    elif change == "unreviewed":
        policy["cases"][CASE]["reviewed"] = False
    if change != "integrity":
        sign(artifact)
    else:
        artifact["producer"]["qairt_version"] = "altered"
    result = CONSUMER.select_shadow(artifact, target, policy, IDENTITY)
    assert result["would_skip"] == 0
    assert result["would_run"] == 1
    assert not result["target_execution_changed"]


def test_failed_target_outcome_is_a_false_skip_candidate(inputs):
    artifact, target, policy = inputs
    target["accuracy_outcomes"][CASE] = "not_passed"
    result = CONSUMER.select_shadow(artifact, target, policy, IDENTITY)
    assert result["false_skip_candidates"] == 1


@pytest.mark.parametrize("data", [None, "not-json", "[]", '{"snapshots":null}'])
def test_cli_missing_or_malformed_input_emits_fallback_report(tmp_path, data):
    artifact = tmp_path / "artifact.json"
    if data is not None:
        artifact.write_text(data)
    target = tmp_path / "target.json"
    policy = tmp_path / "policy.json"
    target.write_text("{}")
    policy.write_text("{}")
    output = tmp_path / "observation.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"--artifact={artifact}",
            f"--target-evidence={target}",
            f"--policy={policy}",
            f"--output={output}",
            "--source-commit=" + "a" * 40,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["would_skip"] == 0


def test_cli_valid_artifact_and_unresolved_policy(inputs, tmp_path):
    artifact, target, policy = inputs
    policy["cases"] = {}
    for name, value in (("artifact", artifact), ("target", target), ("policy", policy)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"--artifact={tmp_path / 'artifact.json'}",
            f"--target-evidence={tmp_path / 'target.json'}",
            f"--policy={tmp_path / 'policy.json'}",
            f"--output={tmp_path / 'observation.json'}",
            "--source-commit=" + IDENTITY["source_commit_sha"],
        ],
        env=os.environ
        | {
            "GITHUB_REPOSITORY": IDENTITY["repository"],
            "GITHUB_WORKFLOW_REF": IDENTITY["workflow_ref"],
            "GITHUB_SHA": IDENTITY["workflow_sha"],
            "GITHUB_RUN_ID": "10",
            "GITHUB_RUN_ATTEMPT": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "observation.json").read_text())
    assert report["would_run"] == 1
    assert "unresolved_responsibility" in report["cases"][0]["reasons"]
