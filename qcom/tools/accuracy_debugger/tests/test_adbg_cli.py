# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Tests for the snooper CLI: ``execute_ort_qnn_snooper``, ``main`` and ``python accuracy_debugger.py``."""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")

import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from common import Algorithm
from conftest import DEBUGGER_DIR
from ort_qnn_snooper import (
    LOGGER_NAME,
    OrtQnnSnooperOutputConfig,
    execute_ort_qnn_snooper,
    main,
)

MODULE = "ort_qnn_snooper"
LOG_FILE = "accuracy_debugger_user_info.log"


@pytest.fixture
def cli_args(qdq_chain_model: Path, qdq_input_sample, write_input_list, tmp_path: Path) -> list[str]:
    input_list = write_input_list(qdq_input_sample, tmp_path / "inputs")
    wd = tmp_path / "wd"
    wd.mkdir()
    return [
        "--qdq_model",
        str(qdq_chain_model),
        "--input_list",
        str(input_list),
        "--working_directory",
        str(wd),
        "--output_directory",
        "out",
    ]


@pytest.fixture(autouse=True)
def _close_snooper_logger():
    yield
    snooper_logger = logging.getLogger(LOGGER_NAME)
    for handler in list(snooper_logger.handlers):
        snooper_logger.removeHandler(handler)
        handler.close()


def _fake_output(tmp_path: Path) -> OrtQnnSnooperOutputConfig:
    csv_path, json_path = tmp_path / "r.csv", tmp_path / "r.json"
    csv_path.touch()
    json_path.touch()
    return OrtQnnSnooperOutputConfig(csv_snooping_report=csv_path, json_snooping_report=json_path)


def test_execute_ort_qnn_snooper(cli_args: list[str], tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}.OrtQnnSnooper") as mock_snooper,
        patch(f"{MODULE}.OrtQnnSnooperInputConfig") as mock_config,
    ):
        mock_snooper.return_value.run.return_value = _fake_output(tmp_path)
        execute_ort_qnn_snooper(cli_args)

    logger = logging.getLogger(LOGGER_NAME)
    mock_snooper.assert_called_once_with(logger, name=Algorithm.ONESHOT)
    mock_snooper.return_value.run.assert_called_once_with(mock_config.return_value)
    kwargs = mock_config.call_args.kwargs
    assert kwargs["dump_output_tensors"] is True
    assert kwargs["enable_framework_op_trace"] is False
    assert kwargs["algorithm"] == Algorithm.ONESHOT
    assert [c.name for c in kwargs["comparators"]] == ["mse"]
    run_dir = tmp_path / "wd" / "oneshot_snooping" / "out"
    assert kwargs["working_directory"] == run_dir
    assert run_dir.is_dir()
    assert (run_dir / LOG_FILE).exists()
    inputs = kwargs["input_processor"].get_framework_inputs()
    assert list(inputs[0]) == ["input"]


def test_execute_layerwise_algorithm(cli_args: list[str], tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}.OrtQnnSnooper") as mock_snooper,
        patch(f"{MODULE}.OrtQnnSnooperInputConfig") as mock_config,
    ):
        mock_snooper.return_value.run.return_value = _fake_output(tmp_path)
        execute_ort_qnn_snooper([*cli_args, "--algorithm", "layerwise", "--dump_elementwise_stats", "conv1=2;Conv=0"])
    mock_snooper.assert_called_once_with(logging.getLogger(LOGGER_NAME), name=Algorithm.LAYERWISE)
    assert mock_config.call_args.kwargs["dump_elementwise_stats"] == {"conv1": 2, "Conv": 0}
    assert (tmp_path / "wd" / "layerwise_snooping" / "out" / LOG_FILE).exists()


def test_execute_passes_trace_flags(cli_args: list[str], tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}.OrtQnnSnooper") as mock_snooper,
        patch(f"{MODULE}.OrtQnnSnooperInputConfig") as mock_config,
    ):
        mock_snooper.return_value.run.return_value = _fake_output(tmp_path)
        execute_ort_qnn_snooper([*cli_args, "--enable_framework_op_trace", "--dump_qnn_ep_input_graph"])
    kwargs = mock_config.call_args.kwargs
    assert kwargs["enable_framework_op_trace"] is True
    assert kwargs["dump_qnn_ep_input_graph"] is True


def test_main_success(cli_args: list[str], tmp_path: Path) -> None:
    with patch(f"{MODULE}.OrtQnnSnooper") as mock_snooper, patch(f"{MODULE}.OrtQnnSnooperInputConfig"):
        mock_snooper.return_value.run.return_value = _fake_output(tmp_path)
        assert main(cli_args) == 0
    log_text = (tmp_path / "wd" / "oneshot_snooping" / "out" / LOG_FILE).read_text()
    assert "Running ORT/QNN-EP snooping..." in log_text
    assert "Snooping Completed. CSV Report generated at:" in log_text
    assert "Json Report generated at:" in log_text


def test_main_argument_error_returns_2(cli_args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    args = list(cli_args)
    args[1] = "/does/not/exist.onnx"
    assert main(args) == 2
    assert "error: QDQ model file '/does/not/exist.onnx' does not exist." in capsys.readouterr().err


def test_main_runtime_error_returns_1_with_single_traceback(cli_args: list[str], tmp_path: Path) -> None:
    with patch(f"{MODULE}.OrtQnnSnooper") as mock_snooper, patch(f"{MODULE}.OrtQnnSnooperInputConfig"):
        mock_snooper.return_value.run.side_effect = RuntimeError("kaboom")
        assert main(cli_args) == 1
    log_text = (tmp_path / "wd" / "oneshot_snooping" / "out" / LOG_FILE).read_text()
    assert log_text.count("Traceback (most recent call last)") == 1
    assert "Error occurred while executing ORT/QNN-EP Snooping." in log_text
    assert "RuntimeError: kaboom" in log_text


def test_execute_reraises_after_logging(cli_args: list[str]) -> None:
    with patch(f"{MODULE}.OrtQnnSnooper") as mock_snooper, patch(f"{MODULE}.OrtQnnSnooperInputConfig"):
        mock_snooper.return_value.run.side_effect = ValueError("bad config")
        with pytest.raises(ValueError, match="bad config"):
            execute_ort_qnn_snooper(cli_args)


def test_main_unknown_flag_raises_system_exit(cli_args: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([*cli_args, "--bogus"])
    assert excinfo.value.code == 2


def test_main_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "python accuracy_debugger.py" in capsys.readouterr().out


def test_main_does_not_catch_keyboard_interrupt(cli_args: list[str]) -> None:
    with (
        patch(f"{MODULE}.OrtQnnSnooper") as mock_snooper,
        patch(f"{MODULE}.OrtQnnSnooperInputConfig", MagicMock()),
        pytest.raises(KeyboardInterrupt),
    ):
        mock_snooper.return_value.run.side_effect = KeyboardInterrupt
        main(cli_args)


@pytest.mark.parametrize("from_other_directory", [False, True], ids=["local-script", "absolute-script"])
def test_python_script_help(tmp_path: Path, from_other_directory: bool) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    script = str(DEBUGGER_DIR / "accuracy_debugger.py") if from_other_directory else "accuracy_debugger.py"
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=tmp_path if from_other_directory else DEBUGGER_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "--qdq_model" in result.stdout
    assert "usage: python accuracy_debugger.py" in result.stdout
