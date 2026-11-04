# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Input-list parsing and loading of raw input tensors for ONNX models.

An input list is a text file with one inference sample per line. Each line holds whitespace separated tokens,
either ``name:=path`` (named) or a bare ``path`` (positional). Blank lines and lines starting with ``#`` are ignored.
"""

from __future__ import annotations

import logging
import math
import shlex
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from onnx_utils import get_input_specs

__all__ = ["InputProcessor", "parse_input_list"]


def parse_input_list(input_list: Path) -> list[dict[str, str]]:
    """Reads an input-list file and returns one entry per data line.

    Args:
        input_list: Path to the input-list file.

    Returns:
        One dict per non-blank, non-comment line. Keys are tensor names for ``name:=path`` tokens and the token
        position (as a string) for bare tokens; values are the raw-file path strings.
    """
    parsed: list[dict[str, str]] = []
    with Path(input_list).open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entry: dict[str, str] = {}
            lexer = shlex.shlex(line, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            lexer.escape = ""  # Windows paths use literal backslashes.
            for pos, token in enumerate(lexer):
                if ":=" in token:
                    name, path = token.split(":=", 1)
                    entry[name.strip()] = path.strip()
                else:
                    entry[str(pos)] = token
            parsed.append(entry)
    return parsed


class InputProcessor:
    """Loads the samples of an input list as numpy arrays matching the inputs of an ONNX model.

    Attributes:
        input_list: Path to the input-list file.
        model: Path to the ONNX model whose graph inputs define names, dtypes and shapes.
        working_dir: Working directory (accepted for API compatibility, unused).
    """

    def __init__(
        self,
        input_list: str | Path,
        model: str | Path,
        working_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
        onnx_symbols: Mapping[str, int] | None = None,
    ) -> None:
        """Initialises the processor.

        Args:
            input_list: Path to the input-list file.
            model: Path to the ONNX model.
            working_dir: Working directory; accepted but unused.
            logger: Logger to use; defaults to the module logger.
            onnx_symbols: Values for symbolic (``dim_param``) input dimensions.

        Raises:
            ValueError: If ``model`` does not exist or is not an ``.onnx`` file.
        """
        self.input_list: Path = Path(input_list)
        self.model: Path = Path(model)
        self.working_dir: Path | None = Path(working_dir) if working_dir is not None else None
        self._logger = logger or logging.getLogger(__name__)
        self._onnx_symbols = onnx_symbols
        self._inputs_cache: list[dict[str, np.ndarray]] | None = None

        if not self.model.exists() or self.model.suffix != ".onnx":
            raise ValueError("Invalid source model. Supported framework types are ['onnx']")

    def get_framework_inputs(self) -> list[dict[str, np.ndarray]]:
        """Loads every sample of the input list.

        The result is memoised: later calls return the same list object.

        Returns:
            One dict per input-list line, mapping the original ONNX input names to arrays of the model's input
            dtype, reshaped to the model's input shape when it is known.

        Raises:
            ValueError: On a token-count mismatch, a missing file for an input, or a file whose element count does
                not match the input shape.
        """
        if self._inputs_cache is not None:
            return self._inputs_cache

        specs = get_input_specs(self.model, self._onnx_symbols)
        samples: list[dict[str, np.ndarray]] = []
        for line_idx, entry in enumerate(parse_input_list(self.input_list)):
            if len(entry) != len(specs):
                raise ValueError(f"Input list line {line_idx + 1}: expected {len(specs)} input(s), got {len(entry)}.")
            sample: dict[str, np.ndarray] = {}
            for idx, spec in enumerate(specs):
                raw = entry.get(spec.name) or entry.get(str(idx))
                if not raw:
                    raise ValueError(f"Input list line {line_idx + 1}: no file for input '{spec.name}'.")
                path = Path(raw)
                if not path.is_absolute():
                    path = self.input_list.parent / path
                arr = np.fromfile(path, dtype=spec.dtype)
                if spec.shape is not None:
                    expected = math.prod(spec.shape)
                    if arr.size != expected:
                        raise ValueError(
                            f"Input file '{path}' has {arr.size} elements of {spec.dtype}, "
                            f"expected {expected} for input '{spec.name}' with shape {spec.shape}."
                        )
                    arr = arr.reshape(spec.shape)
                sample[spec.name] = arr
            samples.append(sample)

        self._inputs_cache = samples
        return samples
