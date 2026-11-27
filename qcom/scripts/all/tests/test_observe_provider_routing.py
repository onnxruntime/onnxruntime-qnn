# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Regression coverage for observation on the active Ubuntu internal-symbol caller."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[4]
WORKFLOW = REPO / ".github/workflows/qualcomm-internal-build-and-test-single-os.yml"
CALLER = REPO / ".github/workflows/qualcomm-internal-build-and-test.yml"
REPORT = """<testsuites><testsuite name="QnnSnapshot_Clip_OpBuilderTest">
<testcase name="Match" status="run" result="completed" />
<testcase name="Drift" status="run" result="completed"><failure message="QNN_SNAPSHOT_DRIFT" /></testcase>
<testcase name="Skipped" status="run" result="skipped"><skipped message="QNN_GOLDEN_ABSENT" /></testcase>
</testsuite></testsuites>"""


def evaluate(expression, inputs, env=None):
    expression = expression.strip().removeprefix("${{").removesuffix("}}")
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"\btrue\b", "True", expression)
    expression = re.sub(r"\bfalse\b", "False", expression)
    return eval(
        expression,
        {"__builtins__": {}},
        {
            "inputs": SimpleNamespace(**inputs),
            "env": SimpleNamespace(**(env or {})),
            "format": lambda template, *args: template.format(*args),
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX active Linux caller fixture")
@pytest.mark.parametrize("build_config", ["Release", "RelWithDebInfo"])
@pytest.mark.parametrize(
    "validation_exit,write_report", [(0, True), (1, True), (1, False), (1, "malformed"), (1, "errors")]
)
def test_active_caller_exports_fresh_report_without_rerunning_tests(
    tmp_path, validation_exit, write_report, build_config
):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    caller = yaml.safe_load(CALLER.read_text())
    active = next(
        job for job in caller["jobs"].values() if job.get("with", {}).get("target_arch") == "x86_64_ubuntu_22_04"
    )
    inputs = {
        name: value.get("default")
        for name, value in workflow["on" if "on" in workflow else True]["workflow_call"]["inputs"].items()
    }
    inputs.update({k: v for k, v in active["with"].items() if not (isinstance(v, str) and "${{" in v)})
    inputs.update(has_code_changes=True, build_config=build_config, target_py_vsn="3.12")
    steps = workflow["jobs"]["build"]["steps"]
    prepare = next(step for step in steps if step.get("id") == "linux_observation_directory")
    baseline = next(step for step in steps if step.get("id") == "platform_validation")
    export = next(step for step in steps if step.get("id") == "linux_provider_observation")
    upload = next(step for step in steps if step.get("name") == "Upload Linux snapshot routing result")
    assert evaluate(prepare["if"], inputs)
    assert evaluate(baseline["if"], inputs)
    assert "QNN_UT_TWO_PASS_SNAPSHOT_ACCURACY" not in baseline.get("env", {})
    assert not any(step.get("id") == "linux_snapshot_routing" for step in steps)
    assert "prepare_snapshot_goldens.sh" not in WORKFLOW.read_text()
    for override in (
        {"qnn_internal_test_symbols": False},
        {"target_arch": "aarch64_manylinux_2_34"},
        {"has_code_changes": False},
        {"build_test_same_runner": False},
    ):
        assert not evaluate(prepare["if"], inputs | override)
    build_arch = evaluate(workflow["env"]["BUILD_DIR_ARCH"], inputs)
    build = tmp_path / "build" / f"linux-{build_arch}" / build_config
    build.mkdir(parents=True)
    # Derive the report suffix from CMake's GTest output contract, rather than
    # copying the exporter's expected filename into a synthetic fixture.
    cmake = (REPO / "cmake/onnxruntime_unittests.cmake").read_text()
    suffix = re.search(r'--gtest_output=xml:\$<TARGET_FILE_NAME:\$\{_UT_TARGET\}>([^"\s]+)', cmake)[1]
    report = build / ("onnxruntime_provider_test" + suffix.replace("$<CONFIG>", build_config))
    (build / "onnxruntime_provider_test.results.xml").write_text(REPORT)  # wrong-name stale report must never be used
    report.write_text(REPORT)  # prior-run evidence must be removed before validation
    binary = build / "onnxruntime_provider_test"
    binary.write_text(f"""#!{sys.executable}
import sys
assert '--gtest_list_tests' in sys.argv, 'Observation must not execute any tests'
if any('QnnSnapshot_' in arg for arg in sys.argv):
 print('QnnSnapshot_Clip_OpBuilderTest.\\n  Match\\n  Drift\\n  Skipped\\n  Missing')
else:
 print('QnnAcc_Clip_AccuracyTest.\\n  Match\\n  Drift\\n  Skipped\\n  Missing')
""")
    binary.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    log = tmp_path / "test-command.json"
    report_xml = REPORT
    if write_report == "malformed":
        report_xml = "<invalid"
    elif write_report == "errors":
        report_xml = REPORT.replace("<testsuites>", '<testsuites errors="1">')
    fake_python.write_text(f"""#!{sys.executable}
import json, sys
from pathlib import Path
Path({str(log)!r}).write_text(json.dumps(sys.argv[1:]))
if {write_report!r}:
 Path({str(report)!r}).write_text({report_xml!r})
sys.exit({validation_exit})
""")
    fake_python.chmod(0o755)
    output = tmp_path / "output"
    env = os.environ | {
        "GITHUB_WORKSPACE": str(tmp_path),
        "RUNNER_TEMP": str(tmp_path),
        "BUILD_DIR_ARCH": build_arch,
        "BUILD_CONFIG_DIR": build_config,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(output),
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
    }
    env.pop("QNN_UT_SNAPSHOT_GOLDEN_DIR", None)
    prepared = subprocess.run(
        ["bash", "-e", "-c", prepare["run"]], check=False, env=env, cwd=REPO, capture_output=True, text=True
    )
    assert prepared.returncode == 0, prepared.stderr
    assert not report.exists()
    directory = dict(line.split("=", 1) for line in output.read_text().splitlines())["routing_dir"]
    assert Path(directory).name == "qnn-routing-123-2"
    sentinel = Path(directory) / "keep.txt"
    sentinel.write_text("existing directory contents")
    collision_output = tmp_path / "collision-output"
    collision = subprocess.run(
        ["bash", "-e", "-c", prepare["run"]],
        check=False,
        env=env | {"GITHUB_OUTPUT": str(collision_output)},
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert collision.returncode == 0
    assert not collision_output.exists()  # no ready/path output means no export or cleanup ownership
    assert sentinel.read_text() == "existing directory contents"
    command = re.sub(r"\$\{\{(.*?)\}\}", lambda match: str(evaluate("${{" + match[1] + "}}", inputs)), baseline["run"])
    tested = subprocess.run(
        ["bash", "-e", "-c", command], check=False, env=env, cwd=REPO, capture_output=True, text=True
    )
    assert tested.returncode == validation_exit
    assert json.loads(log.read_text())[-2:] == ["test_ort_linux_x86_64_ubuntu_22_04_internal_symbols", "--only"]
    output.unlink()
    observed = subprocess.run(
        ["bash", "-e", "-c", export["run"]],
        check=False,
        env=env
        | {
            "QNN_ROUTING_OUTPUT_DIR": directory,
            "QNN_VALIDATION_OUTCOME": "success" if validation_exit == 0 else "failure",
        },
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert observed.returncode == 0, observed.stderr
    if not write_report or write_report in ("malformed", "errors"):
        assert not output.exists()
        assert not (Path(directory) / "accuracy_routing.json").exists()
        return
    result_path = Path(output.read_text().strip().split("=", 1)[1])
    assert "steps.linux_provider_observation.outputs.routing_result_path" in upload["with"]["path"]
    result = json.loads(result_path.read_text())
    assert "schema_version" not in result
    assert result["mode"] == "provider_observation"
    assert result["decision_granularity"] == "none"
    assert result["provider_validation_exit_code"] == validation_exit
    assert result["snapshot_exit_code"] is None
    assert not result["selection_complete"]
    assert all(case["runner_action"] == "unknown" for case in result["accuracy_decisions"])
    assert {case["test_id"].split(".")[-1]: case["status"] for case in result["snapshots"]} == {
        "Match": "match",
        "Drift": "drift",
        "Skipped": "untrusted",
        "Missing": "untrusted",
    }
    digest = result.pop("payload_sha256")
    assert digest == hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
