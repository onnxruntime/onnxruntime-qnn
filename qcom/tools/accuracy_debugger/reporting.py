# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Snooping report helpers: activation statistics, CSV/JSON reports and comparator score plots."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from common import CSV_TO_JSON_FIELDS_MAP, Algorithm, to_py_scalar

__all__ = [
    "ActivationInfo",
    "append_csv_row",
    "dump_csv",
    "dump_json_report",
    "get_activations_info",
    "line_plot",
    "plot_graphs",
]

_MAX_LABELLED_TICKS = 50
_REPORT_VERSION: dict[str, int] = {"major": 0, "minor": 1, "patch": 0}


@dataclass(frozen=True)
class ActivationInfo:
    """Activation information of a tensor.

    Attributes:
        dtype: Data type of the tensor, e.g. ``"float32"``.
        shape: Shape of the tensor as Python ints.
        distribution: ``(min, max, median)`` as Python scalars; ``(nan, nan, nan)`` for empty tensors.
    """

    dtype: str
    shape: tuple[int, ...]
    distribution: tuple[float | int, float | int, float | int]


def get_activations_info(outputs: Mapping[str, np.ndarray]) -> dict[str, ActivationInfo]:
    """Computes the dtype, shape and (min, max, median) of each tensor.

    This method processes a dictionary of activation tensors and generates comprehensive
    metadata for each tensor, including data type, shape, and statistical distribution
    metrics (min, max, median).

    Args:
        outputs: Tensors by name.

    Returns:
        The :class:`ActivationInfo` of each tensor, in the input order.

    Raises:
        ValueError: If ``outputs`` is empty.
        TypeError: If a value is not a numpy array.
    """
    if not outputs:
        raise ValueError("outputs dictionary cannot be empty")

    activation_info: dict[str, ActivationInfo] = {}
    for name, tensor in outputs.items():
        # Validate that the tensor data is a NumPy array
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"Expected np.ndarray for tensor '{name}', got {type(tensor).__name__}")

        if tensor.size == 0:
            # Handle empty arrays gracefully to avoid computation errors
            distribution: tuple[Any, Any, Any] = (np.nan, np.nan, np.nan)
        else:
            # Compute statistical distribution metrics for the tensor
            distribution = (
                to_py_scalar(np.min(tensor)),
                to_py_scalar(np.max(tensor)),
                to_py_scalar(np.median(tensor)),
            )
        activation_info[name] = ActivationInfo(
            dtype=str(tensor.dtype),
            shape=tuple(int(d) for d in tensor.shape),
            distribution=distribution,
        )
    return activation_info


def dump_csv(data_frame: pd.DataFrame, csv_path: Path) -> None:
    """Writes a data frame to a CSV file with a header and without the index.

    Args:
        data_frame: Data to write.
        csv_path: Destination file (overwritten).
    """
    data_frame.to_csv(csv_path, sep=",", index=False, header=True)


def append_csv_row(row: Mapping[str, Any], columns: Sequence[str], csv_path: Path) -> None:
    """Appends one row to a CSV file, writing the header first if the file does not exist yet.

    The cell formatting is identical to writing the full data frame at once with :func:`dump_csv`.

    Args:
        row: Cell values by column name; missing columns are left empty.
        columns: Column order.
        csv_path: Destination file.
    """
    csv_path = Path(csv_path)
    pd.DataFrame([row], columns=list(columns)).to_csv(csv_path, mode="a", index=False, header=not csv_path.exists())


def dump_json_report(
    data_frame: pd.DataFrame, json_report_path: Path, algorithm: Algorithm = Algorithm.ONESHOT
) -> Path:
    """Writes the JSON version of a snooping report.

    Columns listed in ``CSV_TO_JSON_FIELDS_MAP`` are renamed; every other column is a comparator score and goes
    under ``"comparators"``, where missing values (NaN, +inf and -inf, which ``to_json`` emits as null) become
    ``"NaN"``.

    Args:
        data_frame: The report data frame (one row per layer).
        json_report_path: Destination file (overwritten).
        algorithm: Snooping algorithm that produced the report; sets the header's ``artifact_type``/``algorithm``.

    Returns:
        ``json_report_path``.
    """
    layers: list[dict[str, Any]] = []
    for record in json.loads(data_frame.to_json(orient="records")):
        layer: dict[str, Any] = {}
        comparators: dict[str, Any] = {}
        for key, value in record.items():
            if key in CSV_TO_JSON_FIELDS_MAP:
                layer[CSV_TO_JSON_FIELDS_MAP[key]] = value
            else:
                comparators[key] = "NaN" if value is None else value
        layer["comparators"] = comparators
        layers.append(layer)

    algorithm = Algorithm(algorithm)
    json_data = {
        "header": {
            "header_version": dict(_REPORT_VERSION),
            "artifact_type": f"{algorithm.name}_SNOOPING_REPORT",
            "algorithm": algorithm.value,
            "version": dict(_REPORT_VERSION),
        },
        "layers": layers,
    }
    json_report_path = Path(json_report_path)
    with json_report_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    return json_report_path


def line_plot(x: Sequence[Any], y: Sequence[float], plot_name: str, save_dir: Path) -> Path:
    """Saves a line plot of ``y`` over the layers ``x`` as ``<save_dir>/<plot_name>.png``.

    Args:
        x: Layer labels; used as tick labels when there are at most 50 of them.
        y: Scores, one per layer.
        plot_name: Plot name, used for the title, y label and file name.
        save_dir: Existing destination directory.

    Returns:
        The path of the saved PNG file.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    import matplotlib  # noqa: PLC0415  (optional dependency)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = list(x)
    fig, ax = plt.subplots()
    try:
        ax.plot(range(len(labels)), list(y), color="blue")
        if len(labels) <= _MAX_LABELLED_TICKS:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([str(label) for label in labels], rotation=90)
        ax.set_title(f"{plot_name} line plot")
        ax.set_xlabel("Layers")
        ax.set_ylabel(plot_name)
        png_path = Path(save_dir) / f"{plot_name}.png"
        fig.savefig(png_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return png_path


def plot_graphs(
    csv_path: Path,
    layer_names_column: str,
    comparator_columns: Sequence[str],
    output_dir: Path,
    algorithm: Algorithm | str,
    logger: logging.Logger,
    plots_subdir: str = "plots",
) -> None:
    """Plots the comparator scores of a CSV report, one PNG per comparator. Never raises.

    Args:
        csv_path: Path of the CSV report.
        layer_names_column: Column holding the layer names (x axis).
        comparator_columns: Comparator columns to plot.
        output_dir: Report directory; plots go to ``output_dir/plots_subdir``.
        algorithm: Algorithm name, used in the log message.
        logger: Logger.
        plots_subdir: Plot sub-directory, relative to ``output_dir`` (may be nested).
    """
    try:
        import matplotlib  # noqa: F401, PLC0415  (optional dependency probe)
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plots.")
        return

    try:
        plots_save_dir = Path(output_dir) / plots_subdir
        plots_save_dir.mkdir(parents=True, exist_ok=True)
        report_df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"Plotting graph failed with error: {e}")
        return

    for column in comparator_columns:
        try:
            logger.debug(f"Plotting graph for {column} scores...")
            line_plot(
                x=report_df[layer_names_column],
                y=[float(item) for item in report_df[column]],
                plot_name=column,
                save_dir=plots_save_dir,
            )
        except Exception as e:
            logger.warning(f"Plotting graph failed with error: {e}")

    logger.info(f"{algorithm} report plots saved at {plots_save_dir}")
