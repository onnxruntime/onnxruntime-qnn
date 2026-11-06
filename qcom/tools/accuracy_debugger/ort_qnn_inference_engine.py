# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""ORT/QNN-EP inference engine.

Runs a QDQ ONNX model through ONNX Runtime with the QNN or CPU Execution Provider,
optionally exposing intermediate tensors as graph outputs and dumping every output as a float32 ``.raw`` file.

QNN EP library registration follows the legacy ordering: register, select devices, create the session, run,
delete the session, unregister. Un-registration happens in a ``finally`` block so a failed session never leaves
the library registered. ``use_cpu`` creates a regular ``"CPUExecutionProvider"`` session without loading or
registering the QNN plugin.

Classes:
    OrtQnnInferenceEngineInputConfig: Configuration for inference engine inputs.
    OrtQnnInferenceEngineOutputConfig: Configuration for inference engine outputs.
    OrtQnnInferenceEngine: Main inference engine class for model execution.
"""

from __future__ import annotations

import functools
import logging
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import onnx
from common import MAX_FILE_NAME_BYTES, DebuggerConfig, StrictModel, sanitize_tensor_name
from log import LOG_FORMAT
from numpy.typing import NDArray
from onnx_utils import add_graph_outputs, constant_tensors, session_model_source, validate_model, validate_qdq_graph
from pydantic import ConfigDict, DirectoryPath, FilePath, model_validator
from typing_extensions import Self

import onnxruntime as ort

__all__ = [
    "QNN_EP_NAME",
    "OrtQnnInferenceEngine",
    "OrtQnnInferenceEngineInputConfig",
    "OrtQnnInferenceEngineOutputConfig",
    "create_session",
    "ensure_qnn_ep_registered",
    "get_default_engine_logger",
    "run_inference_job",
]

QNN_EP_NAME: Final = "QNNExecutionProvider"
"""Registration (and EP) name of the QNN EP plugin for HTP execution."""

DEFAULT_ENGINE_LOGGER_NAME: Final = "OrtQnnInference"

# Registration names this module currently holds registered with ORT. Mutated in place, never rebound, so no
# ``global`` statement is needed (PLW0603).
_REGISTERED: Final[set[str]] = set()

_LOGGER = logging.getLogger(__name__)


class OrtQnnInferenceEngineInputConfig(DebuggerConfig):
    """Input configuration for the ORT/QNN-EP inference engine.

    Attributes:
        qdq_model: Path to the QDQ ONNX model. Must exist, be a ``.onnx`` file and contain Q/DQ nodes.
        input_sample: Mapping of graph input names to numpy arrays.
        outputs_list: Intermediate tensor names to expose as graph outputs. ``None`` or empty keeps the model's
            own outputs only.
        use_cpu: Run on ORT's built-in ``"CPUExecutionProvider"`` instead of QNN HTP.
        dump_output_tensors: Dump every output to ``working_directory/target_outputs/<sanitized>.raw``.
        working_directory: Existing directory for the dumped outputs and framework traces. Required.
        disable_cpu_fallback: Set ``session.disable_cpu_ep_fallback=1`` so a model that cannot run entirely on
            the QNN backend raises error instead of silently falling back to ORT CPU. Ignored for CPU runs.
        enable_framework_op_trace: Generate a framework op trace in a fresh ``working_directory/op_trace/run_*``
            directory per QNN compilation. Ignored for CPU runs, which do not generate QNN traces.
        dump_qnn_ep_input_graph: Also dump the EP input graph next to the trace for offline name recovery;
            requires enable_framework_op_trace for QNN runs. Ignored for CPU runs.
        compile_only: Create and release the session without running inference, for example to trace how QNN EP
            compiles the unmodified model. ``input_sample`` is not used and ``output_data`` is ``None``.
    """

    qdq_model: FilePath
    input_sample: dict[str, NDArray]
    outputs_list: list[str] | None = None
    use_cpu: bool = False
    dump_output_tensors: bool = False
    working_directory: DirectoryPath
    disable_cpu_fallback: bool = False
    enable_framework_op_trace: bool = False
    dump_qnn_ep_input_graph: bool = False
    compile_only: bool = False

    @model_validator(mode="after")
    def validate_qdq_model(self) -> Self:
        """Validates that ``qdq_model`` is an existing ``.onnx`` file with an original QDQ graph.

        Returns:
            The validated instance.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the file is not a QDQ ONNX model.
        """
        validate_model(self.qdq_model, validate_qdq=True)
        return self


class OrtQnnInferenceEngineOutputConfig(StrictModel):
    """Output configuration of the ORT/QNN-EP inference engine.

    Attributes:
        output_data: Mapping of output names to float32 numpy arrays, in session output order.
        output_dir: Directory holding the dumped ``.raw`` files, when dumping was requested.
        op_trace_file: Trace generated by this compilation, or None when the EP emitted no trace.
        cpu_fallback_reason: Explanation when ORT retried the QNN execution on CPU.
        host_dequantized_outputs: DQ outputs computed on ORT CPU from their quantized QNN outputs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_data: dict[str, np.ndarray] | None = None
    output_dir: DirectoryPath | None = None
    op_trace_file: FilePath | None = None
    cpu_fallback_reason: str = ""
    host_dequantized_outputs: tuple[str, ...] = ()


def _import_qnn_ep() -> ModuleType:
    """Imports the ``onnxruntime_qnn`` plugin package lazily.

    Returns:
        The ``onnxruntime_qnn`` module.

    Raises:
        RuntimeError: If ``onnxruntime_qnn`` is not installed.
    """
    try:
        import onnxruntime_qnn as qnn_ep  # noqa: PLC0415  (optional dependency, imported lazily)
    except ImportError as e:
        raise RuntimeError("onnxruntime_qnn is not installed; it is required to run on QNN EP.") from e
    return qnn_ep


def _register_ep_library(registration_name: str, qnn_ep: ModuleType) -> None:
    """Registers the QNN EP plugin library under ``registration_name``.

    Raises:
        RuntimeError: If registration fails (e.g. ORT older than the plugin's minimum API version).
    """
    try:
        ort.register_execution_provider_library(registration_name, qnn_ep.get_library_path())
    except Exception as e:
        raise RuntimeError(f"Failed to register QNN EP library: {e}") from e


@functools.cache
def ensure_qnn_ep_registered() -> str:
    """Registers the QNN EP plugin under its own EP name for the rest of the process.

    Used to probe for QNN EP availability (``requires_qnn_ep`` in the tests). Later inferences that use the same
    registration name reuse this registration and do not unregister it (BUG-IE-3). ``functools.cache`` does not
    cache exceptions, so a failed call retries.

    Returns:
        The EP name (``"QNNExecutionProvider"``).

    Raises:
        RuntimeError: If ``onnxruntime_qnn`` is not installed or the library cannot be registered.
    """
    qnn_ep = _import_qnn_ep()
    name = str(qnn_ep.get_ep_name())
    if name not in _REGISTERED:
        _register_ep_library(name, qnn_ep)
    # Persistent from here on: OrtQnnInferenceEngine.run neither re-registers nor unregisters it.
    _REGISTERED.add(name)
    return name


@contextmanager
def _ep_library_registered(use_cpu: bool, logger: logging.Logger) -> Iterator[None]:
    """Keeps the QNN EP library registered for one inference (register ... unregister, legacy ``IE:262-301``).

    Args:
        use_cpu: Bypass QNN registration for ORT CPU execution.
        logger: Logger for un-registration failures.

    Raises:
        RuntimeError: If ``onnxruntime_qnn`` is missing or registration fails.
    """
    if use_cpu:
        yield
        return
    qnn_ep = _import_qnn_ep()
    registration_name = QNN_EP_NAME
    if registration_name in _REGISTERED:
        yield
        return
    _register_ep_library(registration_name, qnn_ep)
    _REGISTERED.add(registration_name)
    try:
        yield
    finally:
        _REGISTERED.discard(registration_name)
        try:
            ort.unregister_execution_provider_library(registration_name)
        except Exception as e:  # never mask the error of the inference itself
            logger.warning(f"Failed to unregister execution provider library '{registration_name}': {e}")


def create_session(
    model: onnx.ModelProto,
    use_cpu: bool = False,
    disable_cpu_fallback: bool = False,
    *,
    op_trace_dir: Path | None = None,
    dump_qnn_ep_input_graph: bool = False,
) -> ort.InferenceSession:
    """Creates an ORT inference session for an QDQ ``model``.

    Backends:
        * ``use_cpu``: ORT's built-in ``"CPUExecutionProvider"``, without the QNN plugin.
        * otherwise: QNN EP plugin registered as ``"QNNExecutionProvider"`` with the QNN HTP backend
          (``get_qnn_htp_path()``).

    For QNN, if the registration name is not registered yet, the library is registered here and stays registered
    (standalone use). :meth:`OrtQnnInferenceEngine.run` registers it beforehand and unregisters it after the
    session has been released.

    Args:
        model: Model to run. It is not modified.
        use_cpu: Select ORT CPU execution. QNN trace and graph dump options are ignored.
        disable_cpu_fallback: Add ``session.disable_cpu_ep_fallback=1`` to the session config (QNN paths only).

    Returns:
        The inference session.

    Raises:
        ValueError: If the graph is not an original QDQ model.
        RuntimeError: For QNN execution, if ``onnxruntime_qnn`` is missing, the library cannot be registered,
            or no device matches (``"QNN EP device not found."``).
    """
    validate_qdq_graph(model)
    options = ort.SessionOptions()
    if use_cpu:
        with session_model_source(model) as src:
            return ort.InferenceSession(src, sess_options=options, providers=["CPUExecutionProvider"])

    qnn_ep = _import_qnn_ep()
    registration_name = QNN_EP_NAME
    if registration_name not in _REGISTERED:
        _register_ep_library(registration_name, qnn_ep)
        _REGISTERED.add(registration_name)

    devices = [device for device in ort.get_ep_devices() if device.ep_name == registration_name]
    if not devices:
        raise RuntimeError("QNN EP device not found.")

    ep_options = {"backend_path": str(qnn_ep.get_qnn_htp_path())}
    if op_trace_dir is not None:
        op_trace_dir = Path(op_trace_dir).resolve()
        op_trace_dir.mkdir(parents=True, exist_ok=True)
        ep_options.update(enable_framework_op_trace="1", framework_op_trace_dir=str(op_trace_dir))
        if dump_qnn_ep_input_graph:
            ep_options.update(dump_qnn_ep_input_graph="1", dump_qnn_ep_input_graph_dir=str(op_trace_dir))
    elif dump_qnn_ep_input_graph:
        raise ValueError("dump_qnn_ep_input_graph requires op_trace_dir")
    options.add_provider_for_devices(devices, ep_options)

    if disable_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    _LOGGER.debug(f"Creating ORT session with following options: {options}")
    with session_model_source(model) as src:
        return ort.InferenceSession(src, sess_options=options)


def _detach_paired_dq_outputs(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    """Removes the graph outputs of DQ nodes whose quantized input is also a graph output, in place.

    With its default ``offload_graph_io_quantization``, QNN EP leaves a graph-output DQ to ORT CPU and exposes the
    DQ input under the DQ output name. When the DQ input is itself a graph output, QNN EP lists that tensor twice,
    the compiled graph loses outputs (``Inconsistent output sizes``) and ORT reruns the whole model on CPU. Only DQ
    nodes whose scale and zero point the initializers alone determine are removed, so that
    :func:`_dequantize_on_cpu` can evaluate them. They include parameters that PyTorch exports share through
    ``Identity`` nodes: ORT folds such nodes before QNN EP sees the graph, so their pairs fail as well.

    Args:
        model: Model whose graph outputs are already final. Modified in place.

    Returns:
        The DQ nodes whose outputs were removed, in graph order.
    """
    outputs = {value_info.name for value_info in model.graph.output}
    constants = constant_tensors(model.graph.node, (tensor.name for tensor in model.graph.initializer))
    detached = [
        node
        for node in model.graph.node
        if node.op_type == "DequantizeLinear"
        and node.output[0] in outputs
        and node.input[0] in outputs
        and all(name in constants for name in node.input[1:] if name)
    ]
    names = {node.output[0] for node in detached}
    kept = [value_info for value_info in model.graph.output if value_info.name not in names]
    del model.graph.output[:]
    model.graph.output.extend(kept)
    return detached


def _dequantize_on_cpu(
    model: onnx.ModelProto, nodes: list[onnx.NodeProto], values: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Evaluates DQ ``nodes`` on ORT CPU, which also runs graph-output DQ nodes under QNN EP's default options.

    The nodes that compute their scales and zero points from initializers, such as ``Constant`` and ``Identity``,
    run with them.

    Args:
        model: Model that holds the nodes and their scale and zero point tensors.
        nodes: DQ nodes returned by :func:`_detach_paired_dq_outputs`.
        values: Session outputs, including every quantized input of ``nodes``.

    Returns:
        The DQ outputs keyed by output name.
    """
    graph_nodes = list(model.graph.node)
    producers = {output: index for index, node in enumerate(graph_nodes) for output in node.output if output}
    needed: set[int] = set()
    pending = [name for node in nodes for name in node.input[1:] if name]
    while pending:
        index = producers.get(pending.pop())
        if index is not None and index not in needed:
            needed.add(index)
            pending.extend(name for name in graph_nodes[index].input if name)
    dq_nodes = [graph_nodes[index] for index in sorted(needed)] + nodes
    used = {name for node in dq_nodes for name in node.input if name}
    inputs = list(dict.fromkeys(node.input[0] for node in nodes))
    graph = onnx.helper.make_graph(
        dq_nodes,
        "host_dequantize",
        [
            onnx.helper.make_tensor_value_info(
                name, onnx.helper.np_dtype_to_tensor_dtype(values[name].dtype), values[name].shape
            )
            for name in inputs
        ],
        [onnx.ValueInfoProto(name=node.output[0]) for node in nodes],
        [tensor for tensor in model.graph.initializer if tensor.name in used],
    )
    dq_model = onnx.helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    session = ort.InferenceSession(dq_model.SerializeToString(), providers=["CPUExecutionProvider"])
    results = session.run(None, {name: values[name] for name in inputs})
    return dict(zip((node.output[0] for node in nodes), results, strict=True))


def get_default_engine_logger() -> logging.Logger:
    """Returns the default ``"OrtQnnInference"`` logger at INFO with one stderr handler.

    Idempotent: the handler is attached only if no ``_adbg_handler``-marked handler is present.

    Returns:
        The ``"OrtQnnInference"`` logger.
    """
    logger = logging.getLogger(DEFAULT_ENGINE_LOGGER_NAME)
    if not any(getattr(handler, "_adbg_handler", False) for handler in logger.handlers):
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._adbg_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


class OrtQnnInferenceEngine:
    """Runs QDQ ONNX models on ONNX Runtime's QNN or CPU Execution Provider.

    Attributes:
        logger: Logger used for progress messages.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initializes the engine.

        Args:
            logger: Logger to use. Defaults to :func:`get_default_engine_logger`.
        """
        self.logger: logging.Logger = logger if logger is not None else get_default_engine_logger()

    def run(self, config: OrtQnnInferenceEngineInputConfig) -> OrtQnnInferenceEngineOutputConfig:
        """Runs inference on ``config.qdq_model``.

        Requested intermediate outputs come first, followed by the model's original outputs in their original order
        (duplicates removed). Every output is converted to float32.

        On QNN HTP with CPU fallback allowed, a DQ output whose quantized input is also an output is not exposed by
        the QNN session. It is computed on ORT CPU from the QNN result instead, as QNN EP does for any DQ graph
        output, because QNN EP cannot return both (see :func:`_detach_paired_dq_outputs`).

        With ``compile_only``, the unmodified model's session is created, which compiles it and writes the
        requested trace, and is released without running inference.

        Args:
            config: Engine input configuration.

        Returns:
            The outputs keyed by session output name, and the dump directory when dumping was requested.

        Raises:
            RuntimeError: If the QNN EP cannot be registered, no QNN device is found, or ORT fails.
        """
        model = onnx.load(config.qdq_model)

        outputs_list = config.outputs_list or []
        if outputs_list:
            self.logger.info(f"Adding {len(outputs_list)} intermediate output nodes to the model")
            add_graph_outputs(model, outputs_list)
        elif not config.compile_only:
            self.logger.warning("No intermediate outputs specified. Only final model outputs will be captured.")

        graph_outputs = [value_info.name for value_info in model.graph.output]
        # QNN EP turns offload_graph_io_quantization off when CPU fallback is disabled; then the pairs compile.
        keep_outputs = config.use_cpu or config.disable_cpu_fallback or config.compile_only
        host_dq_nodes = [] if keep_outputs else _detach_paired_dq_outputs(model)
        if host_dq_nodes:
            self.logger.info(
                f"Computing {len(host_dq_nodes)} DequantizeLinear outputs on ORT CPU from their quantized QNN outputs: "
                f"{', '.join(node.output[0] for node in host_dq_nodes)}"
            )

        # Every compilation gets its own directory, including repeated and spawned
        # layerwise runs. A stale trace can never be mistaken for this session's trace.
        trace_directory = None
        session_kwargs = {}
        if config.enable_framework_op_trace and not config.use_cpu:
            trace_root = Path(config.working_directory) / "op_trace"
            trace_root.mkdir(parents=True, exist_ok=True)
            trace_directory = Path(tempfile.mkdtemp(prefix="run_", dir=trace_root)).resolve()
            session_kwargs.update(
                op_trace_dir=trace_directory,
                dump_qnn_ep_input_graph=config.dump_qnn_ep_input_graph,
            )
        elif config.dump_qnn_ep_input_graph and not config.use_cpu:
            raise ValueError("dump_qnn_ep_input_graph requires enable_framework_op_trace")

        # register -> select devices -> session -> run -> del session -> unregister (legacy ordering, IE:262-301)
        with _ep_library_registered(config.use_cpu, self.logger):
            session = create_session(model, config.use_cpu, config.disable_cpu_fallback, **session_kwargs)
            cpu_fallback_reason = ""
            try:
                if not config.compile_only:
                    self.logger.info("Executing inference with %s", "ORT/CPU-EP" if config.use_cpu else "ORT/QNN-EP")
                    initial_providers = session.get_providers()
                    results = session.run(None, config.input_sample)
                    if QNN_EP_NAME in initial_providers and QNN_EP_NAME not in session.get_providers():
                        cpu_fallback_reason = "ORT retried this model on CPU after a QNN HTP execution failure."
                        self.logger.warning(f"{cpu_fallback_reason} Target outputs come from CPU, not HTP.")
                    output_names = [output.name for output in session.get_outputs()]
            finally:
                # Release the session before the library is unregistered, also when run() raises.
                del session

        trace_file = trace_directory / "qnn_op_trace.json" if trace_directory is not None else None
        if trace_file is not None and not trace_file.is_file():
            # A compile_only caller needs the trace itself and reports its absence.
            if not config.compile_only:
                self.logger.warning("QNN EP did not produce a framework op trace; using available tensor mappings.")
            trace_file = None
        if config.compile_only:
            self.logger.info("ORT session created without inference (compile_only).")
            return OrtQnnInferenceEngineOutputConfig(op_trace_file=trace_file)

        self.logger.info("Organizing generated inference outputs")
        values = dict(zip(output_names, results, strict=False))
        if host_dq_nodes:
            values.update(_dequantize_on_cpu(model, host_dq_nodes, values))
            output_names = graph_outputs
        inference_outputs = {name: np.asarray(values[name]).astype(np.float32) for name in output_names}

        output_dir = None
        if config.dump_output_tensors:
            output_dir = self._dump_inference_outputs(inference_outputs, Path(config.working_directory))

        output_config = OrtQnnInferenceEngineOutputConfig(
            output_data=inference_outputs,
            output_dir=output_dir,
            op_trace_file=trace_file,
            cpu_fallback_reason=cpu_fallback_reason,
            host_dequantized_outputs=tuple(node.output[0] for node in host_dq_nodes),
        )
        self.logger.info("ORT inference completed successfully!")
        return output_config

    def _dump_inference_outputs(self, inference_outputs: Mapping[str, np.ndarray], output_path: Path) -> Path:
        """Dumps every output tensor to ``output_path/target_outputs/<sanitized name>.raw``.

        Tensors whose sanitized file name exceeds 255 bytes are skipped with a warning.

        Args:
            inference_outputs: Mapping of output names to arrays.
            output_path: Base directory.

        Returns:
            The ``target_outputs`` directory.
        """
        dump_dir = Path(output_path) / "target_outputs"
        dump_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor in inference_outputs.items():
            file_name = f"{sanitize_tensor_name(name)}.raw"
            if len(file_name.encode("utf-8")) > MAX_FILE_NAME_BYTES:
                self.logger.warning(f"Skipping output tensor '{name}' as filename exceeds {MAX_FILE_NAME_BYTES} bytes")
                continue
            tensor.tofile(dump_dir / file_name)
        return dump_dir


def run_inference_job(config: OrtQnnInferenceEngineInputConfig, logger_name: str) -> OrtQnnInferenceEngineOutputConfig:
    """Top-level, picklable worker entry point for spawn process pools.

    Args:
        config: Engine input configuration.
        logger_name: Name of the logger to use (configured in the child by the pool initializer).

    Returns:
        The engine output configuration.
    """
    return OrtQnnInferenceEngine(logging.getLogger(logger_name)).run(config)
