# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""ONNX Runtime CPU reference runner, reference-output dumper and golden-reference loader.

The reference model is run once on ``CPUExecutionProvider`` with the requested intermediate tensors
exposed as graph outputs.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnx
from common import MAX_FILE_NAME_BYTES, sanitize_tensor_name
from onnx_utils import TensorSpec, add_graph_outputs, session_model_source

import onnxruntime as ort

__all__ = [
    "dump_reference_outputs",
    "generate_framework_outputs",
    "get_all_node_outputs",
    "load_data_from_directory",
    "run_onnx_reference",
]

_PROFILE_FILE_NAME = "profile_info.json"
_BATCH_DIR_RE = re.compile(r"batch_\d+")


def get_all_node_outputs(model: onnx.ModelProto) -> list[str]:
    """Returns every non-empty output tensor name of the top-level graph nodes, in node order.

    Args:
        model: The ONNX model.

    Returns:
        The node output names.
    """
    return [name for node in model.graph.node for name in node.output if name]


def run_onnx_reference(
    model_path: str | Path,
    inputs: Mapping[str, np.ndarray],
    intermediate_outputs: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, np.ndarray]:
    """Runs the reference model on ONNX Runtime's CPU execution provider.

    Args:
        model_path: Path to the ONNX model (external data is loaded).
        inputs: One sample, keyed by graph input name.
        intermediate_outputs: Tensors to expose in addition to the graph outputs. ``None`` exposes every node
            output; an empty sequence means graph outputs only.
        logger: Logger to use; defaults to the module logger.

    Returns:
        Mapping of tensor name to output array: graph outputs first, then the requested intermediates.

    Raises:
        ValueError: If an input required by the session is missing from ``inputs``.
        RuntimeError: If ONNX Runtime fails while running the model.
    """
    logger = logger or logging.getLogger(__name__)
    model = onnx.load(str(model_path))
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    original_outputs = [out.name for out in graph.output if out.name not in initializer_names]
    known = {name for node in graph.node for name in node.output if name}
    known.update(out.name for out in graph.output)
    known.update(inp.name for inp in graph.input)

    requested = list(intermediate_outputs) if intermediate_outputs is not None else get_all_node_outputs(model)
    valid: list[str] = []
    for name in requested:
        if not name:
            continue
        if name not in known:
            logger.warning(f"Tensor '{name}' is not produced by any node of the reference model; it is ignored.")
            continue
        valid.append(name)

    final_names = add_graph_outputs(model, valid)
    head = set(original_outputs)
    names = original_outputs + [n for n in final_names if n not in head and n not in initializer_names]

    # The session is used inside the context so a temporary >2 GB model copy outlives it.
    with session_model_source(model) as src:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(src, sess_options=so, providers=["CPUExecutionProvider"])

        feed: dict[str, np.ndarray] = {}
        for session_input in sess.get_inputs():
            if session_input.name not in inputs:
                raise ValueError(f"Missing input '{session_input.name}' for reference model.")
            feed[session_input.name] = inputs[session_input.name]

        try:
            outs = sess.run(names, feed)
        except Exception as e:
            raise RuntimeError(f"ONNX inference execution failed with error: {e}") from e
        del sess
    return dict(zip(names, outs, strict=True))


def _profile_entry(data: np.ndarray) -> list[object]:
    """Builds one ``profile_info.json`` entry in the legacy format."""
    if data.size == 0:
        return ["-"] * 5
    if data.dtype == bool:
        values = data.tolist()
        return [str(data.dtype), list(data.shape), values, values, values]
    return [
        f"type = {data.dtype}",
        f"shape = {data.shape}",
        # ``!s`` (str) keeps the native-precision text of numpy scalars: format() would widen float32 to float64.
        f"min = {round(np.min(data), 3)!s}",
        f"max = {round(np.max(data), 3)!s}",
        f"median = {round(np.median(data), 3)!s}",
    ]


def dump_reference_outputs(
    outputs: Mapping[str, np.ndarray], output_dir: Path, logger: logging.Logger | None = None
) -> None:
    """Writes reference outputs as float32 ``.raw`` files plus ``profile_info.json``.

    File names are the sanitised tensor names. Names whose file name would exceed 255 bytes are skipped with a
    warning.

    Args:
        outputs: Mapping of tensor name to array.
        output_dir: Destination directory (created if missing).
        logger: Logger to use; defaults to the module logger.
    """
    logger = logger or logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile: dict[str, list[object]] = {}
    origin: dict[str, str] = {}
    for name, value in outputs.items():
        fname = sanitize_tensor_name(name)
        if len(f"{fname}.raw".encode()) > MAX_FILE_NAME_BYTES:
            logger.warning(
                f"Skipping dump of tensor '{name}': file name exceeds {MAX_FILE_NAME_BYTES} bytes after sanitisation."
            )
            continue
        if fname in origin:
            logger.warning(f"Tensors '{origin[fname]}' and '{name}' both map to file '{fname}.raw'; '{name}' wins.")
        origin[fname] = str(name)
        data = np.asarray(value)
        data.astype(np.float32).tofile(output_dir / f"{fname}.raw")
        profile[fname] = _profile_entry(data)

    with (output_dir / _PROFILE_FILE_NAME).open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=4)


def generate_framework_outputs(
    logger: logging.Logger,
    model: str | Path,
    inputs: Mapping[str, np.ndarray],
    dump_output_tensors: bool = False,
    output_dir: Path | None = None,
    intermediate_output_tensors: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Runs the reference model on one sample and optionally dumps its outputs.

    Args:
        logger: Logger to use.
        model: Path to the ONNX reference model.
        inputs: The (first) input sample keyed by graph input name.
        dump_output_tensors: When True, write outputs and their profile to ``output_dir/reference_outputs``.
        output_dir: Working directory for the dump; required when ``dump_output_tensors`` is True.
        intermediate_output_tensors: See ``run_onnx_reference``.

    Returns:
        Mapping of tensor name to output array.

    Raises:
        ValueError: If ``dump_output_tensors`` is True and ``output_dir`` is None, or an input is missing.
        RuntimeError: If ONNX Runtime fails while running the model.
    """
    if dump_output_tensors and output_dir is None:
        raise ValueError("output_dir must be specified when dump_output_tensors is True.")
    outputs = run_onnx_reference(model, inputs, intermediate_output_tensors, logger=logger)
    if dump_output_tensors and output_dir is not None:
        dump_reference_outputs(outputs, Path(output_dir) / "reference_outputs", logger=logger)
    return outputs


def _golden_root(directory: Path, logger: logging.Logger) -> Path:
    """Returns the directory to load goldens from, handling multi-sample ``batch_<n>`` layouts."""
    has_batches = any(child.is_dir() and _BATCH_DIR_RE.fullmatch(child.name) for child in directory.iterdir())
    if not has_batches:
        return directory
    root = directory / "batch_1"
    if not root.is_dir():
        raise ValueError("Golden reference has batch_<n> dirs but no batch_1.")
    logger.warning("Golden reference contains batch_<n> directories; only batch_1 is used.")
    return root


def load_data_from_directory(
    directory: Path,
    intermediate_outputs_info: Mapping[str, TensorSpec] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, np.ndarray]:
    """Recursively loads ``.raw`` golden tensors from a directory.

    Files are float32 on disk; they are cast to the dtype and reshaped to the shape found in
    ``intermediate_outputs_info`` (looked up by sanitized file stem).
    For multi-sample layouts (``batch_<n>`` dirs) only ``batch_1`` is loaded.

    Args:
        directory: Golden-reference directory.
        intermediate_outputs_info: Tensor specs keyed by sanitised tensor name.
        logger: Logger to use; defaults to the module logger.

    Returns:
        Mapping of file stem (unsanitised) to array.

    Raises:
        ValueError: If the directory has ``batch_<n>`` dirs but no ``batch_1``.
    """
    logger = logger or logging.getLogger(__name__)
    info = intermediate_outputs_info or {}
    root = _golden_root(Path(directory), logger)

    tensor_data: dict[str, np.ndarray] = {}
    for file in sorted(root.rglob("*.raw")):
        if {"target_outputs", "subgraph_data"}.intersection(file.relative_to(root).parts[:-1]):
            continue
        stem = file.stem
        if stem in tensor_data:
            logger.warning(f"Ignoring duplicate golden tensor '{stem}' at {file}")
            continue
        spec = info.get(sanitize_tensor_name(stem))
        arr = np.fromfile(file, np.float32).astype(spec.dtype if spec else np.float32)
        if spec is not None and spec.shape is not None:
            expected = math.prod(spec.shape)
            if arr.size == expected:
                arr = arr.reshape(spec.shape)
            else:
                logger.warning(
                    f"Golden tensor '{stem}' has {arr.size} elements, expected {expected} for shape {spec.shape}; "
                    "keeping it flat."
                )
        tensor_data[stem] = arr
    return tensor_data
