# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""ONNX model helpers shared by the reference runner, the inference engine and the subgraph extractor.

Structure-only checks load models with ``load_external_data=False``; shape inference goes through
:func:`infer_model_with_data` because it needs the constant data (Reshape/Slice/Expand shapes).
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from common import sanitize_tensor_name
from onnx import external_data_helper

__all__ = [
    "TensorSpec",
    "add_graph_outputs",
    "constant_tensors",
    "get_input_specs",
    "get_intermediate_outputs_info",
    "infer_model_with_data",
    "is_qdq_model",
    "load_model_structure",
    "session_model_source",
    "validate_model",
    "validate_qdq_graph",
]

_LOGGER = logging.getLogger(__name__)
_FALLBACK_DTYPE = np.dtype(np.float32)
_SMALL_CONSTANT_BYTES = 64 * 1024
_INTEGER_TYPES = frozenset(
    {
        onnx.TensorProto.INT8,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT64,
    }
)
# Ops whose outputs are not constant even when their inputs are.
_RANDOM_OPS = frozenset(
    {"Bernoulli", "Multinomial", "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}
)
_GRAPH_ATTRIBUTES = (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)


@dataclass(frozen=True)
class TensorSpec:
    """Name, static shape and numpy dtype of a model tensor.

    Attributes:
        name: Tensor name (original ONNX name for inputs, sanitized name for intermediate outputs).
        shape: Static shape, or ``None`` when unknown/dynamic (the caller must not reshape).
        dtype: Numpy dtype of the tensor.
    """

    name: str
    shape: tuple[int, ...] | None
    dtype: np.dtype


def load_model_structure(model_path: str | Path) -> onnx.ModelProto:
    """Loads the graph structure of an ONNX model without its external tensor data.

    Args:
        model_path: Path to the ONNX model.

    Returns:
        The model proto; external initializers keep their references and are not read.
    """
    return onnx.load(str(model_path), load_external_data=False)


def _external_tensors(model: onnx.ModelProto) -> list[onnx.TensorProto]:
    """Returns every initializer and Constant-attribute tensor of the top-level graph stored as external data."""
    tensors = list(model.graph.initializer)
    tensors.extend(
        attr.t for node in model.graph.node for attr in node.attribute if attr.type == onnx.AttributeProto.TENSOR
    )
    return [t for t in tensors if t.data_location == onnx.TensorProto.EXTERNAL]


def _external_length(tensor: onnx.TensorProto, base_dir: Path) -> int:
    """Returns the byte length of an externally stored tensor (from its ``length`` key, else the file size)."""
    info = {entry.key: entry.value for entry in tensor.external_data}
    if "length" in info:
        return int(info["length"])
    location = base_dir / info.get("location", "")
    return location.stat().st_size if location.is_file() else 0


def _inline_small_integer_constants(model: onnx.ModelProto, tensors: list[onnx.TensorProto], base_dir: Path) -> None:
    """Loads small integer external tensors (shape-like constants) into the proto, in place.

    ``infer_shapes_path`` does not read external data, so Reshape/Slice/Expand shape constants stored externally
    would stay invisible to shape inference. Inlining just these keeps the proto far below 2 GB.
    """
    for tensor in tensors:
        if tensor.data_type not in _INTEGER_TYPES or _external_length(tensor, base_dir) > _SMALL_CONSTANT_BYTES:
            continue
        external_data_helper.load_external_data_for_tensor(tensor, str(base_dir))
        tensor.data_location = onnx.TensorProto.DEFAULT
        del tensor.external_data[:]


def infer_model_with_data(model_path: str | Path, logger: logging.Logger | None = None) -> onnx.ModelProto:
    """Returns a shape-inferred model whose inference saw the external constant data.

    Up to 2 GB the model is loaded with its data and inferred in memory.
    Above 2 GB only the small integer (shape-like) external tensors are inlined and the model is inferred in memory;
    every other tensor keeps its external reference relative to ``model_path``'s directory. If even that proto is too
    large, ``onnx.shape_inference.infer_shapes_path`` runs into a temporary directory and the result is loaded
    structure-only.

    Args:
        model_path: Path to the ONNX model.
        logger: Logger for the shape-inference warning; defaults to this module's logger.

    Returns:
        The shape-inferred model, or the un-inferred model if shape inference failed.
    """
    log = logger or _LOGGER
    path = Path(model_path)
    model = load_model_structure(path)
    external = _external_tensors(model)
    if external:
        total = sum(_external_length(t, path.parent) for t in external)
        if total + model.ByteSize() <= onnx.checker.MAXIMUM_PROTOBUF:
            model = onnx.load(str(path))
        else:
            _inline_small_integer_constants(model, external, path.parent)
            if model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
                with tempfile.TemporaryDirectory() as tmp:
                    inferred_path = Path(tmp) / "inferred.onnx"
                    try:
                        onnx.shape_inference.infer_shapes_path(str(path), str(inferred_path))
                        return onnx.load(str(inferred_path), load_external_data=False)
                    except Exception as exc:
                        log.warning(f"Shape inference failed: {exc}")
                        return model
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        log.warning(f"Shape inference failed: {exc}")
        return model


def is_qdq_model(model: str | Path | onnx.ModelProto) -> bool:
    """Checks whether a model contains QuantizeLinear/DequantizeLinear style nodes.

    Only the top-level graph is scanned; a single Q/DQ node is enough.

    Args:
        model: Path to an ONNX model or an already loaded model proto.

    Returns:
        True if any top-level node's op type starts with ``Quantize`` or ``Dequantize``.
    """
    proto = model if isinstance(model, onnx.ModelProto) else load_model_structure(model)
    return any(node.op_type.startswith(("Quantize", "Dequantize")) for node in proto.graph.node)


def constant_tensors(nodes: Sequence[onnx.NodeProto], initializers: Iterable[str]) -> set[str]:
    """Returns the tensors that the initializers alone determine.

    Those are the initializers and the outputs of the nodes, such as Constant, the Identity of a shared initializer
    and the DequantizeLinear of a weight, all of whose inputs are such tensors. Random ops are excluded, and so are
    nodes with subgraphs, which can read tensors that are not their inputs.

    Args:
        nodes: Nodes of the model, in topological order.
        initializers: Initializer names.

    Returns:
        The initializer names and the names of the constant node outputs.
    """
    constants = set(initializers)
    for node in nodes:
        inputs = [name for name in node.input if name]
        if (
            (inputs or node.op_type == "Constant")
            and node.op_type not in _RANDOM_OPS
            and not any(attr.type in _GRAPH_ATTRIBUTES for attr in node.attribute)
            and all(name in constants for name in inputs)
        ):
            constants.update(name for name in node.output if name)
    return constants


def validate_qdq_graph(model: onnx.ModelProto) -> None:
    """Requires an original QDQ graph, rejecting cached EPContext partitions."""
    if any(node.op_type == "EPContext" and node.domain == "com.microsoft" for node in model.graph.node):
        raise ValueError("EPContext models are not supported. Please provide the original QDQ ONNX model.")
    if not is_qdq_model(model):
        raise ValueError("Model doesn't contain QDQ nodes. Please provide a valid QDQ ONNX file.")


def validate_model(model_path: str | Path, validate_qdq: bool = False) -> None:
    """Validates the model path, the ONNX suffix and, optionally, an original QDQ graph.

    Args:
        model_path: Path to the model file.
        validate_qdq: Also require at least one Q/DQ node and reject cached EPContext partitions.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the file is not an ``.onnx`` file, or ``validate_qdq`` is set and the graph lacks QDQ
            nodes or contains cached EPContext partitions.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")
    if model_path.suffix != ".onnx":
        raise ValueError("Invalid model format specified. Please provide a valid ONNX file.")
    if validate_qdq:
        validate_qdq_graph(load_model_structure(model_path))


def _np_dtype(elem_type: int) -> np.dtype | None:
    """Maps an ONNX element type to a numpy dtype, or ``None`` if it is unknown/unsupported."""
    if elem_type == onnx.TensorProto.UNDEFINED:
        return None
    try:
        return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type))
    except Exception:
        return None


def get_input_specs(model_path: str | Path, onnx_symbols: Mapping[str, int] | None = None) -> list[TensorSpec]:
    """Returns the specs of the model's real graph inputs (initializers excluded), in graph order.

    Args:
        model_path: Path to the ONNX model.
        onnx_symbols: Values for symbolic dimensions (``dim_param``).

    Returns:
        One :class:`TensorSpec` per graph input that is not an initializer.

    Raises:
        ValueError: If a symbolic dimension has no value in ``onnx_symbols``, or the input dtype is unsupported.
    """
    symbols = {str(k): int(v) for k, v in (onnx_symbols or {}).items()}
    graph = load_model_structure(model_path).graph
    initializers = {init.name for init in graph.initializer}
    specs: list[TensorSpec] = []
    for value_info in graph.input:
        if value_info.name in initializers:
            continue
        tensor_type = value_info.type.tensor_type
        shape: list[int] = []
        for dim in tensor_type.shape.dim:
            if dim.dim_param:
                if dim.dim_param not in symbols:
                    raise ValueError(f"Please supply value for onnx symbol: {dim.dim_param}")
                shape.append(symbols[dim.dim_param])
            else:
                shape.append(dim.dim_value)
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))
        specs.append(TensorSpec(value_info.name, tuple(shape), dtype))
    return specs


def _static_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    """Returns the static shape of a value info, or ``None`` if the rank or any dimension is unknown."""
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims = tensor_type.shape.dim
    if any(dim.dim_param or not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in dims):
        return None
    return tuple(int(dim.dim_value) for dim in dims)


def get_intermediate_outputs_info(model_path: str | Path) -> dict[str, TensorSpec]:
    """Returns dtype and shape of every inferred intermediate tensor and graph output.

    Shape inference sees the external constants (:func:`infer_model_with_data`). Unknown or unsupported element types
    fall back to float32; dynamic or unknown dims give ``shape=None``.

    Args:
        model_path: Path to the ONNX model.

    Returns:
        Mapping of sanitized tensor name to :class:`TensorSpec` (whose ``name`` is the original tensor name).
    """
    graph = infer_model_with_data(model_path).graph
    info: dict[str, TensorSpec] = {}
    for value_info in [*graph.value_info, *graph.output]:
        if not value_info.name:
            continue
        dtype = _np_dtype(value_info.type.tensor_type.elem_type) or _FALLBACK_DTYPE
        info[sanitize_tensor_name(value_info.name)] = TensorSpec(value_info.name, _static_shape(value_info), dtype)
    return info


def add_graph_outputs(model: onnx.ModelProto, names: Sequence[str]) -> list[str]:
    """Exposes extra tensors as graph outputs, in place.

    The new order is the requested names (empty names dropped, duplicates removed) followed by the existing outputs
    in their original order. Existing ``ValueInfoProto`` entries are reused; new ones carry no type (ORT infers it).

    Args:
        model: Model to modify.
        names: Tensor names to expose.

    Returns:
        The final ordered list of graph output names.
    """
    existing: dict[str, onnx.ValueInfoProto] = {}
    for value_info in model.graph.output:
        kept = onnx.ValueInfoProto()
        kept.CopyFrom(value_info)
        existing.setdefault(value_info.name, kept)

    ordered = list(dict.fromkeys([n for n in names if n] + list(existing)))
    del model.graph.output[:]
    model.graph.output.extend(
        existing[name] if name in existing else onnx.ValueInfoProto(name=name) for name in ordered
    )
    return ordered


@contextmanager
def session_model_source(model: onnx.ModelProto) -> Iterator[bytes | str]:
    """Yields something ``onnxruntime.InferenceSession`` can load: serialized bytes, or a file path above 2 GB.

    Above ``onnx.checker.MAXIMUM_PROTOBUF`` a copy of the model is saved with external data into a temporary
    directory that lives until the context exits.

    Args:
        model: Model to hand to ORT. It is not modified.

    Yields:
        The serialized model bytes, or the path (as ``str``) of the saved model.
    """
    if model.ByteSize() <= onnx.checker.MAXIMUM_PROTOBUF:
        yield model.SerializeToString()
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        copy = onnx.ModelProto()
        copy.CopyFrom(model)
        onnx.save(
            copy,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
        )
        yield str(path)
