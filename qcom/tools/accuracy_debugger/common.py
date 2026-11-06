# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Shared constants, exceptions, pydantic bases and small helpers of the Accuracy Debugger."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict

__all__ = [
    "CSV_TO_JSON_FIELDS_MAP",
    "LAYERWISE_SKIP_OPS",
    "MAX_FILE_NAME_BYTES",
    "QDQ_OPS_INDEX_DEFAULT",
    "ActivationStatus",
    "Algorithm",
    "CustomException",
    "DebuggerConfig",
    "StrictModel",
    "SubgraphExtractionError",
    "VerificationError",
    "create_working_directory",
    "create_working_directory_with_timestamp",
    "sanitize_tensor_name",
    "to_py_scalar",
]


class Algorithm(str, Enum):
    """Snooping algorithms supported by the debugger."""

    ONESHOT = "oneshot"
    LAYERWISE = "layerwise"

    def __str__(self) -> str:
        """Returns the plain value so f-strings and argparse help print ``oneshot``/``layerwise``."""
        return self.value


class ActivationStatus:
    """Per-layer status values written to the CSV ``Layer Status`` column."""

    SKIP: Final = "SKIP"
    INFERENCE_FAILURE: Final = "INFERENCE_FAILURE"
    VERIFICATION_FAILURE: Final = "VERIFICATION_FAILURE"
    SUCCESS: Final = "SUCCESS"


CSV_TO_JSON_FIELDS_MAP: Final[dict[str, str]] = {
    "QNN_Op_Name": "qnn_op_name",
    "QNN_Op_Type": "qnn_op_type",
    "Fusion_Pattern": "fusion_pattern",
    "Original_ONNX_Nodes": "original_onnx_nodes",
    "CPU_Fallback_Reason": "cpu_fallback_reason",
    "Layer Name": "layer_name",
    "Layer Output": "layer_output",
    "Layer Status": "layer_status",
    "Layer Type": "layer_type",
    "Layer Shape": "layer_shape",
    "Source(Min, Max, Median)": "source_min_max_median",
    "Target(Min, Max, Median)": "target_min_max_median",
    "Exception": "exception",
}

MAX_FILE_NAME_BYTES: Final = 255

QDQ_OPS_INDEX_DEFAULT: Final[Mapping[str, int]] = MappingProxyType({"QuantizeLinear": 1, "DequantizeLinear": 2})

FP32_REFERENCE_QDQ_OPS_INDEX: Final = {"DequantizeLinear": 2}

LAYERWISE_SKIP_OPS: Final = frozenset({"QuantizeLinear", "DequantizeLinear", "Transpose", "Identity", "Constant"})


class CustomException(Exception):  # noqa: N818  (legacy public name kept)
    """Base class for the debugger's custom exceptions.

    Args:
        msg: Human-readable reason.
        exception: Optional underlying exception.
    """

    def __init__(self, msg: str, exception: BaseException | None = None) -> None:
        self.exception = exception
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self) -> str:
        """Returns ``msg``, or ``"Exception: <exception> Reason: <msg>"`` when an exception was given."""
        if self.exception is None:
            return f"{self.msg}"
        return f"Exception: {self.exception} Reason: {self.msg}"


class VerificationError(CustomException):
    """Defines error encountered during verification."""


class SubgraphExtractionError(Exception):
    """Raised when a subgraph cannot be extracted from a model."""


class MessageKeyError(KeyError):
    """``KeyError`` whose ``str()`` is the bare message (no repr quotes), so report cells stay readable."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class StrictModel(BaseModel):
    """Pydantic base model that forbids extra fields and validates on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=(), populate_by_name=True)


class DebuggerConfig(StrictModel):
    """Base pydantic class for debugger configurations; also allows arbitrary field types."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


def to_py_scalar(x: object) -> object:
    """Converts a numpy scalar to the equivalent Python scalar for CSV/JSON cells.

    numpy floating scalars go through ``float(str(x))`` so they keep the shortest repr of their native precision
    (``np.float32(0.1)`` becomes ``0.1``, not ``0.10000000149011612``), identically on numpy 1.x and 2.x.

    Args:
        x: Any value.

    Returns:
        A Python ``float``/``int``/``bool`` for numpy scalars, otherwise ``x`` unchanged.
    """
    if isinstance(x, np.floating):
        return float(str(x))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


_INVALID_TENSOR_NAME_CHARS: Final = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_tensor_name(name: object) -> str:
    """Transforms a node/tensor name to follow QNN tensor naming conventions.

    Every character other than ``[a-zA-Z0-9_]`` is replaced by ``_``, and a ``_`` is prepended when the result does
    not start with a letter or ``_``.

    Args:
        name: Original name; non-strings are converted with ``str``.

    Returns:
        The sanitized name.

    Raises:
        ValueError: If the name is empty.
    """
    transformed = _INVALID_TENSOR_NAME_CHARS.sub("_", str(name))
    if not transformed:
        raise ValueError("Tensor name must not be empty.")
    if not (transformed[0].isalpha() or transformed[0] == "_"):
        transformed = "_" + transformed
    return transformed


def _resolve_base_directory(working_directory: str | Path | None) -> Path:
    """Returns the base working directory, creating ``<cwd>/working_directory`` when none is given."""
    if working_directory is None:
        base = Path.cwd() / "working_directory"
        base.mkdir(parents=True, exist_ok=True)
        return base
    base = Path(working_directory)
    if not base.exists():
        raise FileNotFoundError(f"Working directory {working_directory} does not exist.")
    return base


def create_working_directory(
    sub_directory: str,
    working_directory: str | Path | None = None,
    output_directory: str | None = None,
) -> Path:
    """Creates ``<base>/<sub_directory>/<output_directory>``.

    When ``output_directory`` is None this delegates to :func:`create_working_directory_with_timestamp`. An empty
    string keeps the legacy semantics: the result is ``<base>/<sub_directory>`` with no leaf.

    Args:
        sub_directory: Subdirectory under the working directory.
        working_directory: Base working directory. Defaults to ``<cwd>/working_directory`` (created if missing).
        output_directory: Leaf directory name. None uses a timestamp.

    Returns:
        Path to the created directory.

    Raises:
        FileNotFoundError: If ``working_directory`` is given but does not exist.
    """
    if output_directory is None:
        return create_working_directory_with_timestamp(sub_directory, working_directory)
    working_dir = _resolve_base_directory(working_directory) / sub_directory / output_directory
    working_dir.mkdir(parents=True, exist_ok=True)
    return working_dir


def create_working_directory_with_timestamp(sub_directory: str, working_directory: str | Path | None = None) -> Path:
    """Creates ``<base>/<sub_directory>/<YYYY-MM-DD_HH-MM-SS>``.

    Args:
        sub_directory: Subdirectory under the working directory.
        working_directory: Base working directory. Defaults to ``<cwd>/working_directory`` (created if missing).

    Returns:
        Path to the created directory.

    Raises:
        FileNotFoundError: If ``working_directory`` is given but does not exist.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    working_dir = _resolve_base_directory(working_directory) / sub_directory / timestamp
    working_dir.mkdir(parents=True, exist_ok=True)
    return working_dir
