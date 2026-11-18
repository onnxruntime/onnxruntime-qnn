# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Check production lifecycle logic without installing or invoking the QDC SDK."""

import ast
import logging
import shutil
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest
from qdc_windows_payload import validate_windows_results


@pytest.fixture
def lifecycle():
    # Only the lifecycle classes are loaded; SDK imports and upload decorators
    # are intentionally excluded. Network calls are mocked at their boundary.
    path = Path(__file__).parents[1] / "qdc_runner.py"
    source = ast.parse(path.read_text())
    names = {"Platform", "RunStatus", "Run", "TestRuns", "QdcRunner", "QdcRun"}
    body = ast.parse("from __future__ import annotations").body
    body.extend(node for node in source.body if isinstance(node, ast.ClassDef) and node.name in names)
    clock = [0.0]
    namespace = {
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "Enum": Enum,
        "NamedTuple": NamedTuple,
        "Path": Path,
        "logging": logging,
        "shutil": shutil,
        "time": SimpleNamespace(
            monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        ),
        "JobState": SimpleNamespace(CANCELED="Canceled", COMPLETED="Completed", RUNNING="Running"),
        "ErrorMessageType0": type("ErrorMessageType0", (), {}),
        "validate_windows_results": validate_windows_results,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def runner_with_client(lifecycle, client):
    runner = object.__new__(lifecycle.QdcRunner)
    runner._client = client
    return runner


def test_failed_cancellation_remains_eligible_for_cleanup(lifecycle):
    runner = SimpleNamespace(
        queue_timeout_seconds=-1,
        get_run_status=Mock(return_value=lifecycle.RunStatus.PENDING),
        abort_run=Mock(side_effect=[RuntimeError("Cancellation rejected"), None]),
    )
    run = lifecycle.QdcRun(runner, lifecycle.Platform.WINDOWS, "test", 42, "url")
    runs = lifecycle.TestRuns()
    runs.add_run(run)
    run.poll()
    assert run.status == lifecycle.RunStatus.FAILED
    assert run.needs_cleanup()
    runs.abort()
    assert runner.abort_run.call_count == 2
    assert not run.needs_cleanup()
    assert run.status == lifecycle.RunStatus.FAILED
    assert not runs.success()


@pytest.mark.parametrize("status", ("SUCCESS", "FAILED", "ABORTED"))
def test_normal_terminal_runs_are_not_canceled(lifecycle, status):
    runner = SimpleNamespace(get_run_status=Mock(return_value=lifecycle.RunStatus[status]), abort_run=Mock())
    run = lifecycle.QdcRun(runner, lifecycle.Platform.WINDOWS, "test", 42, "url")
    runs = lifecycle.TestRuns()
    runs.add_run(run)
    run.poll()
    runs.abort()
    runner.abort_run.assert_not_called()


def test_cancellation_requires_remote_confirmation(lifecycle):
    client = SimpleNamespace(
        get_job_status=Mock(side_effect=[SimpleNamespace(state="Running"), SimpleNamespace(state="Canceled")]),
        abort_job=Mock(return_value=SimpleNamespace(status_code=200, parsed=None)),
    )
    runner_with_client(lifecycle, client).abort_run(42)
    client.abort_job.assert_called_once_with(42)
    assert client.get_job_status.call_count == 2


def test_cancellation_rejects_http_failure(lifecycle):
    client = SimpleNamespace(
        get_job_status=Mock(return_value=SimpleNamespace(state="Running")),
        abort_job=Mock(return_value=SimpleNamespace(status_code=500, parsed=None)),
    )
    with pytest.raises(RuntimeError, match="HTTP=500"):
        runner_with_client(lifecycle, client).abort_run(42)


def test_cancellation_rejects_parsed_error(lifecycle):
    error = lifecycle.ErrorMessageType0()
    error.message = "Cancellation rejected"
    client = SimpleNamespace(
        get_job_status=Mock(return_value=SimpleNamespace(state="Running")),
        abort_job=Mock(return_value=SimpleNamespace(status_code=200, parsed=error)),
    )
    with pytest.raises(RuntimeError, match="Cancellation rejected"):
        runner_with_client(lifecycle, client).abort_run(42)


def test_cancellation_confirmation_timeout(lifecycle):
    client = SimpleNamespace(
        get_job_status=Mock(return_value=SimpleNamespace(state="Running")),
        abort_job=Mock(return_value=SimpleNamespace(status_code=200, parsed=None)),
    )
    with pytest.raises(TimeoutError, match="Termination not confirmed"):
        runner_with_client(lifecycle, client).abort_run(42)


def test_completed_remote_job_does_not_need_abort(lifecycle):
    client = SimpleNamespace(get_job_status=Mock(return_value=SimpleNamespace(state="Completed")), abort_job=Mock())
    runner_with_client(lifecycle, client).abort_run(42)
    client.abort_job.assert_not_called()


def test_old_windows_results_cannot_mask_empty_download(lifecycle, tmp_path):
    (tmp_path / "bootstrap.results.xml").write_text("<testsuite />")
    job_logs = tmp_path / "42"
    job_logs.mkdir()
    (job_logs / "bootstrap.results.xml").write_text("<testsuite />")
    client = SimpleNamespace(download_logs=Mock())
    runner = runner_with_client(lifecycle, client)
    runner._log_dir = tmp_path
    assert runner._finish_run(42, lifecycle.Platform.WINDOWS, lifecycle.RunStatus.SUCCESS) == lifecycle.RunStatus.FAILED
    client.download_logs.assert_called_once_with(42, job_logs)
    assert not (job_logs / "bootstrap.results.xml").exists()
    assert (tmp_path / "bootstrap.results.xml").exists()


def test_windows_download_failure_is_a_failed_run(lifecycle, tmp_path):
    client = SimpleNamespace(download_logs=Mock(side_effect=RuntimeError("Download failed")))
    runner = runner_with_client(lifecycle, client)
    runner._log_dir = tmp_path
    assert runner._finish_run(42, lifecycle.Platform.WINDOWS, lifecycle.RunStatus.SUCCESS) == lifecycle.RunStatus.FAILED


@pytest.mark.parametrize("platform", ("ANDROID", "QUALCOMM_LINUX"))
def test_other_platforms_keep_existing_result_handling(lifecycle, tmp_path, platform):
    client = SimpleNamespace(download_logs=Mock())
    runner = runner_with_client(lifecycle, client)
    runner._log_dir = tmp_path
    assert (
        runner._finish_run(42, lifecycle.Platform[platform], lifecycle.RunStatus.SUCCESS) == lifecycle.RunStatus.SUCCESS
    )
    client.download_logs.assert_called_once_with(42, tmp_path)
