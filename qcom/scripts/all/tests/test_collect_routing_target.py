# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "collect_routing_target.py"
SPEC = importlib.util.spec_from_file_location("routing_target", SCRIPT)
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def test_fresh_shard_reports_and_architecture_are_collected(tmp_path):
    context = tmp_path / "build.json"
    context.write_text(json.dumps({"schema_version": 1}))
    started = tmp_path / "started.txt"
    started.write_text(str(time.time() - 5))
    for index in range(2):
        (tmp_path / f"provider.shard{index}.results.xml").write_text(
            f'<testsuites><testsuite name="QnnAcc_Clip_AccuracyTest"><testcase name="Case{index}" status="run" result="completed" /></testsuite></testsuites>'
        )
    (tmp_path / "LastTest.log").write_text("QNN platform attributes: HTP arch: 73, SDK version: 2.52")
    result = COLLECTOR.collect_evidence(context, tmp_path, started)
    assert result["target_htp_arch"] == 73
    assert result["errors"] == []
    assert len(result["accuracy_outcomes"]) == 2


def test_stale_reports_and_logs_cannot_supply_target_evidence(tmp_path):
    context = tmp_path / "build.json"
    context.write_text("{}")
    started = tmp_path / "started.txt"
    started.write_text(str(time.time()))
    for path, content in (
        (
            tmp_path / "stale.results.xml",
            '<testsuites><testsuite name="QnnAcc_Clip_AccuracyTest"><testcase name="Case" status="run" result="completed" /></testsuite></testsuites>',
        ),
        (tmp_path / "LastTest.log", "HTP arch: 73"),
    ):
        path.write_text(content)
        os.utime(path, (1, 1))
    result = COLLECTOR.collect_evidence(context, tmp_path, started)
    assert not result["accuracy_outcomes"]
    assert result["target_htp_arch"] is None
    assert result["errors"]


@pytest.mark.parametrize("report", ["malformed", "duplicate", "skipped", "failed"])
def test_bad_or_nonpassing_reports_do_not_supply_pass_evidence(tmp_path, report):
    context = tmp_path / "build.json"
    context.write_text("{}")
    started = tmp_path / "started.txt"
    started.write_text(str(time.time() - 5))
    case = '<testcase name="Case" status="run" result="completed" />'
    if report == "skipped":
        case = '<testcase name="Case" status="run" result="skipped"><skipped /></testcase>'
    elif report == "failed":
        case = '<testcase name="Case" status="run" result="completed"><failure /></testcase>'
    elif report == "duplicate":
        case += case
    xml = f'<testsuites><testsuite name="QnnAcc_Clip_AccuracyTest">{case}</testsuite></testsuites>'
    (tmp_path / "provider.results.xml").write_text("broken" if report == "malformed" else xml)
    (tmp_path / "LastTest.log").write_text("HTP arch: 73")
    result = COLLECTOR.collect_evidence(context, tmp_path, started)
    if report in ("malformed", "duplicate"):
        assert result["errors"]
    else:
        assert result["accuracy_outcomes"]["QnnAcc_Clip_AccuracyTest.Case"] == "not_passed"


def test_collect_build_uses_actual_shared_resolver(tmp_path):
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "sdk.yaml").write_text("version: 2.52.0\n")
    ort = tmp_path / "ort"
    ort.mkdir()
    (ort / "VERSION_NUMBER").write_text("1.30.0\n")
    (tmp_path / "CMakeCache.txt").write_text(
        f"onnxruntime_QNN_HOME:PATH={sdk.as_posix()}\nonnxruntime_ORT_HOME:PATH={ort.as_posix()}\n"
    )
    result = COLLECTOR.collect_build(tmp_path)
    assert result["qairt_version"] == "2.52.0"
    assert result["ort_version"] == "1.30.0"
