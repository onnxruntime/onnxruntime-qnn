# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for reporting."""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from common import Algorithm
from reporting import (
    ActivationInfo,
    append_csv_row,
    dump_csv,
    dump_json_report,
    get_activations_info,
    line_plot,
    plot_graphs,
)

ONESHOT_COLUMNS = [
    "Layer Name",
    "Layer Output",
    "Layer Type",
    "Layer Shape",
    "Source(Min, Max, Median)",
    "Target(Min, Max, Median)",
    "mse",
    "snr",
]
LAYERWISE_COLUMNS = [
    "Layer Name",
    "Layer Output",
    "Layer Status",
    "Layer Type",
    "Layer Shape",
    "Source(Min, Max, Median)",
    "Target(Min, Max, Median)",
    "mse",
    "snr",
    "Exception",
]


def _oneshot_rows() -> list[dict]:
    info = get_activations_info(
        {
            "conv": np.array([0.1, 0.9, 0.5], dtype=np.float32),
            "relu": np.array([0, 255, 12, 1], dtype=np.uint8),
        }
    )
    return [
        {
            "Layer Name": "conv1",
            "Layer Output": "conv",
            "Layer Type": "Conv",
            "Layer Shape": info["conv"].shape,
            "Source(Min, Max, Median)": info["conv"].distribution,
            "Target(Min, Max, Median)": info["conv"].distribution,
            "mse": 0.00474,
            "snr": math.inf,
        },
        {
            "Layer Name": "relu1",
            "Layer Output": "relu",
            "Layer Type": "Relu",
            "Layer Shape": info["relu"].shape,
            "Source(Min, Max, Median)": info["relu"].distribution,
            "Target(Min, Max, Median)": info["relu"].distribution,
            "mse": float("nan"),
            "snr": -math.inf,
        },
    ]


# --------------------------------------------------------------------------------------------- activation info
def test_activation_info_python_scalars() -> None:
    info = get_activations_info({"t": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})["t"]
    assert isinstance(info, ActivationInfo)
    assert info.dtype == "float32"
    assert info.shape == (2, 2)
    assert all(type(d) is int for d in info.shape)
    assert info.distribution == (1.0, 4.0, 2.5)
    assert all(type(v) is float for v in info.distribution)


def test_activation_info_is_frozen() -> None:
    info = get_activations_info({"t": np.ones(2)})["t"]
    with pytest.raises(AttributeError):
        info.dtype = "x"  # type: ignore[misc]


def test_activation_info_float32_cell_text() -> None:
    info = get_activations_info({"t": np.array([0.1, 0.9, 0.5], dtype=np.float32)})["t"]
    assert str(info.distribution) == "(0.1, 0.9, 0.5)"


def test_activation_info_uint8_cell_text() -> None:
    info = get_activations_info({"t": np.array([0, 255, 12, 1, 20], dtype=np.uint8)})["t"]
    assert info.dtype == "uint8"
    assert str(info.distribution) == "(0, 255, 12.0)"
    assert type(info.distribution[0]) is int
    assert type(info.distribution[2]) is float


def test_activation_info_bool() -> None:
    info = get_activations_info({"t": np.array([True, False, True])})["t"]
    assert info.distribution[0] is False
    assert info.distribution[1] is True


def test_activation_info_empty_tensor_gives_nan() -> None:
    info = get_activations_info({"t": np.zeros((0, 3), dtype=np.float32)})["t"]
    assert info.shape == (0, 3)
    assert all(math.isnan(v) for v in info.distribution)


def test_activation_info_empty_dict_raises() -> None:
    with pytest.raises(ValueError, match="outputs dictionary cannot be empty"):
        get_activations_info({})


def test_activation_info_non_array_raises() -> None:
    with pytest.raises(TypeError, match="Expected np.ndarray for tensor 't', got list"):
        get_activations_info({"t": [1, 2, 3]})  # type: ignore[dict-item]


# --------------------------------------------------------------------------------------------- CSV
def test_dump_csv(tmp_path: Path) -> None:
    df = pd.DataFrame(_oneshot_rows(), columns=ONESHOT_COLUMNS)
    csv_path = tmp_path / "oneshot.csv"
    dump_csv(df, csv_path)
    text = csv_path.read_text()
    assert text.splitlines()[0] == ",".join(f'"{c}"' if "," in c else c for c in ONESHOT_COLUMNS)
    assert '"(0.1, 0.9, 0.5)"' in text
    assert '"(0, 255, 6.5)"' in text
    assert "0.00474" in text
    assert text.splitlines() == df.to_csv(sep=",", index=False, header=True).splitlines()


def test_append_csv_row_matches_full_dump(tmp_path: Path) -> None:
    rows = _oneshot_rows()
    for row in rows:
        row["Layer Status"] = "SUCCESS"
        row["Exception"] = ""
    rows.append({"Layer Name": "skip", "Layer Output": "out", "Layer Status": "SKIP", "Layer Type": "Add"})
    rows[-1].update(
        {"Layer Shape": (), "Source(Min, Max, Median)": (), "Target(Min, Max, Median)": ()},
        mse="NaN",
        snr="NaN",
        Exception="Extraction failed with error: boom",
    )
    incremental = tmp_path / "incremental.csv"
    for row in rows:
        append_csv_row(row, LAYERWISE_COLUMNS, incremental)
    full = tmp_path / "full.csv"
    dump_csv(pd.DataFrame(rows, columns=LAYERWISE_COLUMNS), full)
    assert incremental.read_bytes() == full.read_bytes()
    assert pd.read_csv(incremental).shape == (3, len(LAYERWISE_COLUMNS))


def test_append_csv_row_missing_columns_left_empty(tmp_path: Path) -> None:
    csv_path = tmp_path / "r.csv"
    append_csv_row({"a": 1}, ["a", "b"], csv_path)
    assert csv_path.read_text().splitlines() == ["a,b", "1,"]


# --------------------------------------------------------------------------------------------- JSON
def test_dump_json_report_oneshot(tmp_path: Path) -> None:
    df = pd.DataFrame(_oneshot_rows(), columns=ONESHOT_COLUMNS)
    path = dump_json_report(df, tmp_path / "oneshot.json")
    assert path == tmp_path / "oneshot.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["header"] == {
        "header_version": {"major": 0, "minor": 1, "patch": 0},
        "artifact_type": "ONESHOT_SNOOPING_REPORT",
        "algorithm": "oneshot",
        "version": {"major": 0, "minor": 1, "patch": 0},
    }
    first, second = data["layers"]
    assert first == {
        "layer_name": "conv1",
        "layer_output": "conv",
        "layer_type": "Conv",
        "layer_shape": [3],
        "source_min_max_median": [0.1, 0.9, 0.5],
        "target_min_max_median": [0.1, 0.9, 0.5],
        "comparators": {"mse": 0.00474, "snr": "NaN"},
    }
    assert second["comparators"] == {"mse": "NaN", "snr": "NaN"}
    assert second["source_min_max_median"] == [0, 255, 6.5]


def test_dump_json_report_default_algorithm_is_oneshot(tmp_path: Path) -> None:
    data = json.loads(dump_json_report(pd.DataFrame([{"mse": 1.0}]), tmp_path / "r.json").read_text())
    assert data["header"]["algorithm"] == "oneshot"


def test_dump_json_report_layerwise(tmp_path: Path) -> None:
    rows = [
        {
            "Layer Name": "conv1",
            "Layer Output": "conv",
            "Layer Status": "SUCCESS",
            "Layer Type": "Conv",
            "Layer Shape": (1, 4),
            "Source(Min, Max, Median)": (0.0, 1.5, 0.25),
            "Target(Min, Max, Median)": (0.0, 1.5, 0.25),
            "mse": 0.5,
            "snr": None,
            "Exception": "",
        },
        {
            "Layer Name": "add",
            "Layer Output": "add_out",
            "Layer Status": "VERIFICATION_FAILURE",
            "Layer Type": "Add",
            "Layer Shape": (),
            "Source(Min, Max, Median)": (),
            "Target(Min, Max, Median)": (),
            "mse": "NaN",
            "snr": "NaN",
            "Exception": "Couldn't fetch golden reference for: add_out",
        },
    ]
    path = dump_json_report(
        pd.DataFrame(rows, columns=LAYERWISE_COLUMNS), tmp_path / "layerwise.json", Algorithm.LAYERWISE
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["header"]["artifact_type"] == "LAYERWISE_SNOOPING_REPORT"
    assert data["header"]["algorithm"] == "layerwise"
    first, second = data["layers"]
    assert first["layer_status"] == "SUCCESS"
    assert first["exception"] == ""
    assert first["layer_shape"] == [1, 4]
    assert first["comparators"] == {"mse": 0.5, "snr": "NaN"}
    assert second["layer_status"] == "VERIFICATION_FAILURE"
    assert second["exception"] == "Couldn't fetch golden reference for: add_out"
    assert second["comparators"] == {"mse": "NaN", "snr": "NaN"}
    assert "Layer Status" not in first["comparators"]
    assert "Exception" not in first["comparators"]


def test_dump_json_report_accepts_algorithm_string(tmp_path: Path) -> None:
    path = dump_json_report(pd.DataFrame([{"mse": 1.0}]), tmp_path / "r.json", "layerwise")  # type: ignore[arg-type]
    assert json.loads(path.read_text())["header"]["artifact_type"] == "LAYERWISE_SNOOPING_REPORT"


def test_dump_json_report_utf8_and_indent(tmp_path: Path) -> None:
    df = pd.DataFrame([{"Layer Name": "conv_é", "mse": 1.0}])
    text = dump_json_report(df, tmp_path / "r.json").read_text(encoding="utf-8")
    assert "conv_é" in text
    assert '\n    "header": {' in text


# --------------------------------------------------------------------------------------------- plots
def _write_report(tmp_path: Path) -> Path:
    csv_path = tmp_path / "oneshot.csv"
    pd.DataFrame({"Layer Name": ["a", "b", "c"], "mse": [0.1, "NaN", 0.3], "snr": [1.0, 2.0, 3.0]}).to_csv(
        csv_path, index=False
    )
    return csv_path


def test_plot_graphs_writes_pngs(tmp_path: Path, mock_logger: MagicMock) -> None:
    pytest.importorskip("matplotlib")
    csv_path = _write_report(tmp_path)
    plot_graphs(csv_path, "Layer Name", ["mse", "snr"], tmp_path, Algorithm.ONESHOT, mock_logger)
    assert (tmp_path / "plots" / "mse.png").is_file()
    assert (tmp_path / "plots" / "snr.png").is_file()
    mock_logger.warning.assert_not_called()
    mock_logger.info.assert_called_once_with(f"oneshot report plots saved at {tmp_path / 'plots'}")


def test_plot_graphs_nested_subdir(tmp_path: Path, mock_logger: MagicMock) -> None:
    pytest.importorskip("matplotlib")
    csv_path = _write_report(tmp_path)
    plot_graphs(csv_path, "Layer Name", ["snr"], tmp_path, "layerwise", mock_logger, plots_subdir="plots/base")
    assert (tmp_path / "plots" / "base" / "snr.png").is_file()
    mock_logger.info.assert_called_once_with(f"layerwise report plots saved at {tmp_path / 'plots' / 'base'}")


def test_plot_graphs_bad_column_never_raises(tmp_path: Path, mock_logger: MagicMock) -> None:
    pytest.importorskip("matplotlib")
    csv_path = _write_report(tmp_path)
    plot_graphs(csv_path, "Layer Name", ["missing", "snr"], tmp_path, Algorithm.ONESHOT, mock_logger)
    assert mock_logger.warning.call_count == 1
    assert mock_logger.warning.call_args[0][0].startswith("Plotting graph failed with error:")
    assert (tmp_path / "plots" / "snr.png").is_file()
    mock_logger.info.assert_called_once()


def test_plot_graphs_line_plot_failure_never_raises(tmp_path: Path, mock_logger: MagicMock) -> None:
    pytest.importorskip("matplotlib")
    csv_path = _write_report(tmp_path)
    with patch("reporting.line_plot", side_effect=RuntimeError("boom")):
        plot_graphs(csv_path, "Layer Name", ["mse", "snr"], tmp_path, Algorithm.ONESHOT, mock_logger)
    assert [c[0][0] for c in mock_logger.warning.call_args_list] == ["Plotting graph failed with error: boom"] * 2
    mock_logger.info.assert_called_once()


def test_plot_graphs_missing_csv_never_raises(tmp_path: Path, mock_logger: MagicMock) -> None:
    pytest.importorskip("matplotlib")
    plot_graphs(tmp_path / "nope.csv", "Layer Name", ["mse"], tmp_path, Algorithm.ONESHOT, mock_logger)
    mock_logger.warning.assert_called_once()


def test_plot_graphs_without_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, logger: logging.Logger
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    csv_path = _write_report(tmp_path)
    caplog.set_level(logging.DEBUG, logger=logger.name)
    plot_graphs(csv_path, "Layer Name", ["mse", "snr"], tmp_path, Algorithm.ONESHOT, logger)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == ["matplotlib is not installed; skipping plots."]
    assert "report plots saved at" not in caplog.text
    assert not (tmp_path / "plots").exists()


def test_line_plot_many_layers(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    labels = [f"layer_{i}" for i in range(60)]
    path = line_plot(labels, [float(i) for i in range(60)], "mse", tmp_path)
    assert path == tmp_path / "mse.png"
    assert path.is_file()
