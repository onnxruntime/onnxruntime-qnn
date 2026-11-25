# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from prepare_custom_op import prepare

TARGETS = Path(__file__).parents[1] / "custom_op_targets.json"


@pytest.fixture
def producer(tmp_path: Path) -> tuple[Path, Path]:
    build = tmp_path / "build/linux-x86_64/Release"
    build.mkdir(parents=True)
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (build.parent / "qairt-sdk-path-Release.txt").write_text(str(sdk))
    runner = build / "onnxruntime_provider_test"
    runner.write_text(
        f"#!{sys.executable}\n"
        """import json, os, sys
from pathlib import Path
directory = Path(os.environ['ORT_QNN_UDO_CONTEXT_DIR'])
name = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--gtest_filter='))
report = Path(next(arg.split('xml:', 1)[1] for arg in sys.argv if arg.startswith('--gtest_output=')))
with Path('calls.jsonl').open('a') as stream:
    stream.write(json.dumps({'name': name, 'target': os.environ['ORT_QNN_UDO_CONTEXT_TARGET'],
                            'domain': os.environ.get('ORT_QNN_CUSTOM_OP_DOMAINS')}) + '\\n')
if 'Generate' in name:
    for file in ('myadd_ctx.onnx', 'myadd_qdq.onnx', 'myadd.bin'):
        (directory / file).write_text('fixture')
skipped = '<skipped message="unavailable"/>' if os.environ.get('FAKE_SKIP') else ''
report.write_text('<testsuites tests="1" failures="0" disabled="0"><testsuite><testcase>'
                  + skipped + '</testcase></testsuite></testsuites>')
"""
    )
    runner.chmod(0o755)
    return build, sdk


def test_prepare_validates_both_targets_in_fresh_processes(producer, tmp_path, monkeypatch):
    build, sdk = producer
    monkeypatch.setenv("ORT_QNN_CUSTOM_OP_DOMAINS", "caller:OtherOp")
    destination = tmp_path / "package"
    prepare(build, destination, TARGETS, sdk)
    calls = [json.loads(line) for line in (build / "calls.jsonl").read_text().splitlines()]
    assert [call["target"] for call in calls] == [
        "73:60:8",
        "73:60:8",
        "75:57:8",
        "75:57:8",
        "79:69:8",
        "79:69:8",
        "81:87:8",
        "81:87:8",
    ]
    assert all(call["domain"] is None for call in calls)
    assert all("Generate" in calls[i]["name"] and "Load" in calls[i + 1]["name"] for i in range(0, len(calls), 2))
    assert (destination / "contexts/81/myadd.bin").is_file()
    assert (destination / "run_custom_op_tests.sh").is_file()
    assert os.environ["ORT_QNN_CUSTOM_OP_DOMAINS"] == "caller:OtherOp"


def test_skipped_producer_fails_and_preserves_previous_contexts(producer, tmp_path, monkeypatch):
    build, sdk = producer
    destination = tmp_path / "package"
    old = destination / "contexts/73/previous.bin"
    old.parent.mkdir(parents=True)
    old.write_text("previous")
    monkeypatch.setenv("FAKE_SKIP", "1")
    with pytest.raises(RuntimeError, match="was skipped"):
        prepare(build, destination, TARGETS, sdk)
    assert old.read_text() == "previous"


def test_sdk_mismatch_rejected_before_generation(producer, tmp_path):
    build, _ = producer
    with pytest.raises(RuntimeError, match="same QAIRT SDK"):
        prepare(build, tmp_path / "package", TARGETS, tmp_path / "different-sdk")
    assert not (build / "calls.jsonl").exists()


@pytest.mark.parametrize("report", ["<testcase><skipped/></testcase>", '<testsuites tests="0"/>'])
def test_packaged_runner_rejects_skips_and_empty_suites(tmp_path, report):
    package = tmp_path / "custom_op"
    package.mkdir()
    (package / "contexts/73").mkdir(parents=True)
    script = package / "run_custom_op_tests.sh"
    shutil.copyfile(TARGETS.parent / "custom_op/run_custom_op_tests.sh", script)
    runner = tmp_path / "onnxruntime_provider_test"
    runner.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "report = next(arg.split('xml:', 1)[1] for arg in sys.argv if arg.startswith('--gtest_output='))\n"
        f"Path(report).write_text({report!r})\n"
    )
    runner.chmod(0o755)
    result = subprocess.run(["sh", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode != 0


@pytest.mark.parametrize("arch,soc", [(75, 57), (79, 69), (81, 87)])
def test_packaged_android_runner_selects_queried_device_target(tmp_path, monkeypatch, arch, soc):
    package = tmp_path / "custom_op"
    context = package / f"contexts/{arch}"
    context.mkdir(parents=True)
    (context / "target.txt").write_text(f"{arch}:{soc}:8\n")
    script = package / "run_custom_op_tests.sh"
    shutil.copyfile(TARGETS.parent / "custom_op/run_custom_op_tests.sh", script)
    tools = tmp_path / "tools"
    tools.mkdir()
    uname = tools / "uname"
    uname.write_text('#!/bin/sh\nif [ "$1" = "-s" ]; then echo Linux; else echo aarch64; fi\n')
    uname.chmod(0o755)
    runner = tmp_path / "onnxruntime_provider_test"
    runner.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "name = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--gtest_filter='))\n"
        "report = next(arg.split('xml:', 1)[1] for arg in sys.argv if arg.startswith('--gtest_output='))\n"
        "if 'QueryTarget' in name:\n"
        f"    (Path(os.environ['ORT_QNN_UDO_CONTEXT_DIR']) / 'custom_op_device_target.txt').write_text('{arch}:{soc}:8')\n"
        "with Path('calls.jsonl').open('a') as stream:\n"
        "    stream.write(json.dumps({'name': name, 'target': os.environ.get('ORT_QNN_UDO_CONTEXT_TARGET')}) + '\\n')\n"
        'Path(report).write_text(\'<testsuites tests="1"><testcase name="pass"/></testsuites>\')\n'
    )
    runner.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["sh", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert "QueryTarget" in calls[0]["name"]
    assert len(calls) == 5
    assert all(call["target"] == f"{arch}:{soc}:8" for call in calls[2:])
