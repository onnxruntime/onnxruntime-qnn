# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Accuracy Debugger for ONNX Runtime with the QNN Execution Provider.

Compares the intermediate tensors of a QDQ ONNX model executed on the QNN EP against an ONNX Runtime CPU
reference, either for the whole graph at once (``oneshot``) or one layer at a time (``layerwise``).

Run it with ``python accuracy_debugger.py --help``.

The snooper compares a reference (ORT CPU run of an FP32 or QDQ model, or a golden dump) against the
target (the QDQ model on QNN EP) and reports per-tensor comparator scores.

Algorithms:
    * ``oneshot``: one target run of the whole model with every selected intermediate exposed.
    * ``layerwise``: every layer is extracted as a subgraph, fed with reference activations and run in
      isolation (a fresh spawn process per layer), so errors do not accumulate across layers.

Reports are written to the working directory as ``<algorithm>.csv`` / ``<algorithm>.json`` plus optional plots.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import multiprocessing as mp
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnx
import pandas as pd
from common import (
    FP32_REFERENCE_QDQ_OPS_INDEX,
    LAYERWISE_SKIP_OPS,
    QDQ_OPS_INDEX_DEFAULT,
    ActivationStatus,
    Algorithm,
    DebuggerConfig,
    MessageKeyError,
    StrictModel,
    VerificationError,
    create_working_directory,
    sanitize_tensor_name,
    to_py_scalar,
)
from comparators import Comparator, MSEComparator
from inputs import InputProcessor
from log import configure_worker_logging, get_logger
from onnx_utils import (
    get_intermediate_outputs_info,
    is_qdq_model,
    load_model_structure,
    validate_model,
)
from op_trace import TRACE_COLUMNS, FusionGroup, OpTraceIndex
from ort_qnn_inference_engine import (
    OrtQnnInferenceEngine,
    OrtQnnInferenceEngineInputConfig,
    OrtQnnInferenceEngineOutputConfig,
    run_inference_job,
)
from parser import OrtQnnSnooperParser
from pydantic import DirectoryPath, Field, FilePath, model_validator
from reference import generate_framework_outputs, load_data_from_directory
from reporting import ActivationInfo, append_csv_row, dump_json_report, get_activations_info, plot_graphs
from subgraph_extractor import OnnxSubgraphExtractor
from typing_extensions import Self
from verification import VerifierResult, verify

__all__ = [
    "LOGGER_NAME",
    "OrtQnnSnooper",
    "OrtQnnSnooperInputConfig",
    "OrtQnnSnooperOutputConfig",
    "execute_ort_qnn_snooper",
    "main",
]

LOGGER_NAME: Final = "Snooping_OrtQnn"
FORCED_CPU_REASON: Final = "Layer forced to CPU by set_cpu_layers."
HOST_DEQUANTIZE_REASON: Final = (
    "Dequantized on ORT CPU from its captured Q output; QNN EP cannot return a Q output and its DQ output together."
)

_ONESHOT_COLUMNS: Final = (
    "Layer Name",
    "Layer Output",
    "Layer Type",
    "Layer Shape",
    "Source(Min, Max, Median)",
    "Target(Min, Max, Median)",
)
_LAYERWISE_COLUMNS: Final = (
    "Layer Name",
    "Layer Output",
    "Layer Status",
    "Layer Type",
    "Layer Shape",
    "Source(Min, Max, Median)",
    "Target(Min, Max, Median)",
)


class OrtQnnSnooperInputConfig(DebuggerConfig):
    """Input configuration of :class:`OrtQnnSnooper`.

    Constraints (checked by the model validators, in this order):
        * ``reference_model`` is supported only with ``Algorithm.ONESHOT`` and must be a valid ONNX file.
        * ``qdq_model`` must be a valid ONNX file containing QuantizeLinear/DequantizeLinear nodes.
        * ``set_cpu_layers`` is supported only with ``Algorithm.LAYERWISE``.
        * ``dump_elementwise_stats`` requires a quantized (QDQ) ``reference_model`` when one is given.

    Attributes:
        qdq_model: QDQ ONNX model to debug (the target).
        reference_model: Optional reference model (FP32 or QDQ); ``qdq_model`` is used when omitted.
        input_processor: Provides the input samples; only the first sample is snooped.
        algorithm: Snooping algorithm.
        comparators: Comparators used to score target against reference tensors.
        working_directory: Existing directory receiving every artefact.
        golden_reference_path: Directory of pre-computed reference ``.raw`` tensors (skips the reference run).
        set_intermediate_layers: Layer names / op types to debug; ``None`` means all.
        set_cpu_layers: Layer names / op types executed on ORT's CPUExecutionProvider (layerwise only).
        dump_elementwise_stats: Layer name / op type -> allowed step difference for elementwise logs.
        dump_output_tensors: Dump reference and target tensors to ``.raw`` files.
        op_trace_file: Existing framework op trace; it takes precedence over generated traces.
        enable_framework_op_trace: Generate a framework op trace per target compilation, under ``working_directory``.
        dump_qnn_ep_input_graph: Also dump the EP input graph next to each generated trace.
    """

    qdq_model: FilePath
    reference_model: FilePath | None = None
    input_processor: InputProcessor
    algorithm: Algorithm = Algorithm.ONESHOT
    comparators: list[Comparator] = Field(default_factory=lambda: [MSEComparator()])
    working_directory: DirectoryPath
    golden_reference_path: DirectoryPath | None = None
    set_intermediate_layers: list[str] | None = None
    set_cpu_layers: list[str] | None = None
    dump_elementwise_stats: dict[str, int] | None = None
    dump_output_tensors: bool = False
    op_trace_file: FilePath | None = None
    enable_framework_op_trace: bool = False
    dump_qnn_ep_input_graph: bool = False

    @model_validator(mode="after")
    def validate_trace_options(self) -> Self:
        """Graph dumps are written next to generated traces, so they need enable_framework_op_trace."""
        if self.dump_qnn_ep_input_graph and not self.enable_framework_op_trace:
            raise ValueError("dump_qnn_ep_input_graph requires enable_framework_op_trace")
        return self

    @model_validator(mode="after")
    def validate_reference_model(self) -> Self:
        """Checks that a reference model is only used with oneshot and is a valid ONNX file.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``reference_model`` is given with a non-oneshot algorithm or is not a valid ONNX file.
            FileNotFoundError: Propagated from :func:`validate_model`.
        """
        if self.reference_model:
            if self.algorithm == Algorithm.ONESHOT:
                validate_model(self.reference_model)
            else:
                raise ValueError(f"reference_model is supported only with {Algorithm.ONESHOT.name.lower()} algorithm.")
        return self

    @model_validator(mode="after")
    def validate_qdq_model(self) -> Self:
        """Checks that ``qdq_model`` is an original QDQ ONNX model.

        Returns:
            The validated config.

        Raises:
            ValueError: If the model is not an original QDQ ONNX model.
        """
        validate_model(self.qdq_model, validate_qdq=True)
        return self

    @model_validator(mode="after")
    def validate_set_cpu_layers(self) -> Self:
        """Checks that ``set_cpu_layers`` is only used with the layerwise algorithm.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``set_cpu_layers`` is given with a non-layerwise algorithm.
        """
        if self.set_cpu_layers and self.algorithm != Algorithm.LAYERWISE:
            raise ValueError(
                f"set_cpu_layers option is supported only with {Algorithm.LAYERWISE.name.lower()} algorithm."
            )
        return self

    @model_validator(mode="after")
    def validate_dump_elementwise_stats(self) -> Self:
        """Checks that elementwise stats are only requested against a quantized reference model.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``dump_elementwise_stats`` is given with a non-QDQ ``reference_model``.
        """
        if self.reference_model:
            reference_model_is_qdq = is_qdq_model(self.reference_model)
            if self.dump_elementwise_stats and not reference_model_is_qdq:
                raise ValueError(
                    "dump_elementwise_stats option is supported only when reference_model is a quantized model."
                )
        return self


class OrtQnnSnooperOutputConfig(StrictModel):
    """Output of :class:`OrtQnnSnooper`.

    Attributes:
        csv_snooping_report: Path of the CSV report.
        json_snooping_report: Path of the JSON report.
    """

    csv_snooping_report: FilePath
    json_snooping_report: FilePath


def _get_constant_tensors(graph: onnx.GraphProto) -> set[str]:
    """Returns the weight and bias tensors: constant data that QNN EP receives but does not compute.

    Nodes are visited in graph (topological) order. Initializers and ``Constant`` outputs are constant. So are the
    outputs of a node without subgraphs whose inputs are all constant (at least one input), unless QNN EP computes
    the node from dequantized weights:

    * ``DequantizeLinear`` outputs, and layout ops (``Reshape``, ``Transpose``, ...) of them, are dequantized weights.
    * ``QuantizeLinear`` outputs are constant, as are the outputs of nodes without dequantized inputs, which ORT
      folds (for example a ``Mul`` of two initializers).
    * Any other node with a dequantized input is constant only as the single-input ``X`` of a ``DQ -> X -> Q``
      unit, which ORT also folds. Otherwise QNN EP computes it (for example a ``MatMul`` of two weights), so its
      outputs are activations.

    Args:
        graph: Graph whose top-level nodes are scanned.

    Returns:
        The constant tensor names, initializers included.
    """
    # Ops that only move, select or cast their data: applied to a dequantized weight, they still give a weight.
    weight_layout_ops: Final = frozenset(
        {
            "Cast",
            "Concat",
            "DepthToSpace",
            "Expand",
            "Flatten",
            "Gather",
            "GatherElements",
            "GatherND",
            "Identity",
            "Reshape",
            "Slice",
            "SpaceToDepth",
            "Split",
            "Squeeze",
            "Tile",
            "Transpose",
            "Unsqueeze",
        }
    )
    constants = {init.name for init in graph.initializer}
    constants.update(init.values.name for init in graph.sparse_initializer)
    producers = {name: node for node in graph.node for name in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    graph_outputs = {output.name for output in graph.output}
    dequantized: set[str] = set()
    for node in graph.node:
        if node.op_type == "Constant":
            constants.update(node.output)
            continue
        inputs = [name for name in node.input if name]
        has_subgraph = any(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for attr in node.attribute
        )
        if not inputs or has_subgraph or not constants.issuperset(inputs):
            continue
        if node.op_type == "DequantizeLinear" or (
            node.op_type in weight_layout_ops and dequantized.intersection(inputs)
        ):
            dequantized.update(node.output)
        elif node.op_type != "QuantizeLinear" and dequantized.intersection(inputs):
            producer = producers.get(node.input[0])
            folded = (
                len(node.input) == 1
                and len(node.output) == 1
                and producer is not None
                and producer.op_type == "DequantizeLinear"
                and node.output[0] not in graph_outputs
                and [consumer.op_type for consumer in consumers.get(node.output[0], [])] == ["QuantizeLinear"]
            )
            if not folded:
                continue
        constants.update(node.output)
    return constants


def _get_qdq_consumers(nodes: Iterable[onnx.NodeProto]) -> dict[str, list[onnx.NodeProto]]:
    """Maps each tensor to the Q/DQ nodes that take it as their data input."""
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        if node.op_type in QDQ_OPS_INDEX_DEFAULT and node.input:
            consumers.setdefault(node.input[0], []).append(node)
    return consumers


def _get_downstream_qdq_outputs(
    node: onnx.NodeProto,
    qdq_consumers: Mapping[str, Sequence[onnx.NodeProto]],
    qdq_ops_index: Mapping[str, int],
) -> list[str]:
    """Returns the outputs of the Q/DQ nodes that follow ``node`` through Q/DQ nodes only.

    Q/DQ nodes need not be adjacent in topological order, so data edges are followed, including every output
    branch of a multi-output operator.

    Args:
        node: Node whose outputs start the walk.
        qdq_consumers: Result of :func:`_get_qdq_consumers` for the node's graph.
        qdq_ops_index: Q/DQ op types whose outputs are returned; the walk passes through the other Q/DQ nodes.

    Returns:
        Output tensor names in visiting order.
    """
    outputs: list[str] = []
    pending = list(node.output)
    visited: set[str] = set()
    while pending:
        tensor = pending.pop(0)
        if tensor in visited:
            continue
        visited.add(tensor)
        for child in qdq_consumers.get(tensor, []):
            if child.op_type in qdq_ops_index:
                outputs.extend(child.output)
            pending.extend(child.output)
    return outputs


class OrtQnnSnooper:
    """Runs the oneshot or layerwise snooping algorithm for a QDQ model on ORT/QNN-EP.

    This class provides comprehensive debugging capabilities for ONNX models running
    on the ONNXRuntime with QNN Execution provider. It compares outputs between a reference
    model (FP32 or QDQ) and a target quantized model to identify accuracy degradation
    and analyze tensor distributions.

    The snooper supports multiple debugging workflows:
    - Full model analysis (ONESHOT algorithm)
    - Layer-by-layer analysis (LAYERWISE algorithm - future implementation)
    - Custom intermediate layer selection
    - CPU fallback layer configuration

    Example:
        >>> snooper = OrtQnnSnooper(logging.getLogger("snooper"), name=Algorithm.LAYERWISE)
        >>> output = snooper.run(config)
        >>> print(output.csv_snooping_report, output.json_snooping_report)
    """

    def __init__(
        self, logger: logging.Logger, name: Algorithm | str = Algorithm.ONESHOT, *, isolate_inference: bool = True
    ) -> None:
        """Initialises the snooper.

        Args:
            logger: Logger used for every message (also forwarded to spawned workers by name).
            name: Algorithm name, used in log and plot messages.
            isolate_inference: Run every layerwise subgraph in a fresh spawn process (legacy behaviour, keeps
                QNN/HTP state from leaking between layers). ``False`` runs in-process.
        """
        self._logger = logger
        self._name = name
        self._isolate_inference = isolate_inference
        self._op_trace_index: OpTraceIndex | None = None
        self._supplied_op_trace: OpTraceIndex | None = None
        self._enable_framework_op_trace = False
        self._dump_qnn_ep_input_graph = False
        self._trace_requested = False
        self._reference_proto: onnx.ModelProto | None = None
        self._runtime_fallback_reason = ""
        self._host_dequantized: frozenset[str] = frozenset()
        self._forced_cpu = False
        # QNN HTP fusions of the unmodified QDQ model, kept intact by the selected outputs and subgraphs.
        self._fusion_groups: list[FusionGroup] = []

    def _load_op_trace(self, path: Path, model: onnx.ModelProto) -> OpTraceIndex | None:
        try:
            return OpTraceIndex.load(path, model, self._reference_proto)
        except (OSError, ValueError, TypeError) as error:
            self._logger.warning(f"Cannot use framework op trace '{path}': {error}. Using fallback mapping.")
            return None

    def _use_generated_trace(self, result: OrtQnnInferenceEngineOutputConfig, model: onnx.ModelProto) -> None:
        # Reset on every compilation, so a prior layer's fusion cannot label this layer.
        self._op_trace_index = None
        self._runtime_fallback_reason = result.cpu_fallback_reason
        self._host_dequantized = frozenset(result.host_dequantized_outputs)
        if result.op_trace_file:
            self._op_trace_index = self._load_op_trace(result.op_trace_file, model)

    def _trace_metadata(self, tensor: str) -> dict[str, str]:
        if not self._trace_requested:
            return {}
        result = dict.fromkeys(TRACE_COLUMNS, "")
        if self._forced_cpu:
            # No QNN compilation ran, so a supplied trace must not label this layer.
            result["CPU_Fallback_Reason"] = FORCED_CPU_REASON
            return result
        for index in (self._op_trace_index, self._supplied_op_trace):
            if index is not None:
                metadata = index.metadata(tensor)
                for key, value in metadata.items():
                    if value:
                        result[key] = value
        host_reason = HOST_DEQUANTIZE_REASON if tensor in self._host_dequantized else ""
        if host_reason or self._runtime_fallback_reason:
            result["CPU_Fallback_Reason"] = ";".join(
                part for part in (result["CPU_Fallback_Reason"], host_reason, self._runtime_fallback_reason) if part
            )
        return result

    def _probe_fusion_groups(
        self, qdq_model: Path, qdq_proto: onnx.ModelProto, working_directory: Path
    ) -> list[FusionGroup]:
        """Finds the QNN HTP fusions of the unmodified QDQ model.

        Exposing a tensor inside a fusion stops QNN EP from fusing it, so a trace of the snooped model cannot show
        the fusions to keep. The unmodified model is compiled once more, with the framework op trace and without
        inference; the trace is written to ``working_directory/fusion_probe/op_trace/run_*``.

        Args:
            qdq_model: QDQ model path.
            qdq_proto: Structure of the QDQ model.
            working_directory: Output directory.

        Returns:
            The fusion groups in graph order, or an empty list if QNN EP did not report them.
        """
        probe_directory = working_directory / "fusion_probe"
        probe_directory.mkdir(parents=True, exist_ok=True)
        config = OrtQnnInferenceEngineInputConfig(
            qdq_model=qdq_model,
            input_sample={},
            working_directory=probe_directory,
            enable_framework_op_trace=True,
            compile_only=True,
        )
        unprotected = "Outputs and subgraphs are selected without them, which can stop QNN EP from fusing layers."
        self._logger.info("Compiling the QDQ model on QNN HTP, without inference, to find its fusions")
        try:
            trace_file = self._run_inference(config).op_trace_file
        except RuntimeError as error:
            self._logger.warning(f"Cannot find the QNN HTP fusions: {error}. {unprotected}")
            return []
        if trace_file is None:
            self._logger.warning(f"QNN EP did not produce a framework op trace of its fusions. {unprotected}")
            return []
        try:
            groups = OpTraceIndex.load(trace_file, qdq_proto).fusion_groups()
        except (OSError, ValueError, TypeError) as error:
            self._logger.warning(f"Cannot read the QNN HTP fusions from '{trace_file}': {error}. {unprotected}")
            return []

        nodes = {node.name: node for node in qdq_proto.graph.node if node.name}
        for group in groups:
            layers = [name for name in group.nodes if nodes[name].op_type not in QDQ_OPS_INDEX_DEFAULT]
            self._logger.info(f"QNN HTP fuses {'+'.join(group.patterns)}: {', '.join(layers or group.nodes)}")
        if not groups:
            self._logger.info("QNN HTP fuses no layers whose tensors could be exposed.")
        return groups

    def run(self, config: OrtQnnSnooperInputConfig) -> OrtQnnSnooperOutputConfig:
        """Executes the configured snooping algorithm.

        This is the main entry point for the snooping workflow. It orchestrates the
        entire debugging process including:
        1. Loading and validating input data
        2. Generating reference outputs from the reference model
        3. Generating target outputs from the QDQ model on ORT/QNN-EP
        4. Comparing outputs using specified comparators
        5. Generating comprehensive reports (CSV and JSON)
        6. Creating visualization plots

        The method supports both ONESHOT (full model) and LAYERWISE (incremental) debugging algorithms.

        Args:
            config: Validated snooper configuration.

        Returns:
            Paths of the generated CSV and JSON reports.

        Raises:
            ValueError: For an unsupported algorithm.
            VerificationError: If no tensor could be compared (oneshot).
            RuntimeError: If the reference or target inference fails (oneshot).
        """
        qdq_model = Path(config.qdq_model)
        algorithm = config.algorithm
        working_directory = Path(config.working_directory)
        dump_output_tensors = config.dump_output_tensors
        intermediates_layers_list = config.set_intermediate_layers or []
        cpu_layers_list = config.set_cpu_layers or []
        reference_model = Path(config.reference_model) if config.reference_model else qdq_model

        self._logger.info(f"Started {algorithm} snooping")

        qdq_proto = load_model_structure(qdq_model)

        self._op_trace_index = self._supplied_op_trace = None
        self._runtime_fallback_reason = ""
        self._host_dequantized = frozenset()
        self._forced_cpu = False
        self._enable_framework_op_trace = config.enable_framework_op_trace
        self._dump_qnn_ep_input_graph = config.dump_qnn_ep_input_graph
        self._trace_requested = config.op_trace_file is not None or config.enable_framework_op_trace
        self._reference_proto = qdq_proto if reference_model == qdq_model else load_model_structure(reference_model)
        if config.op_trace_file:
            self._supplied_op_trace = self._load_op_trace(config.op_trace_file, qdq_proto)
        self._fusion_groups = []
        if self._trace_requested:
            self._fusion_groups = self._probe_fusion_groups(qdq_model, qdq_proto, working_directory)

        samples = config.input_processor.get_framework_inputs()
        if len(samples) > 1:
            self._logger.warning(f"Input list contains {len(samples)} samples; only the first sample is used.")
        input_sample = samples[0]

        # Determine if reference model is QDQ format (affects tensor mapping strategy)
        reference_model_is_qdq = is_qdq_model(reference_model)

        # Get valid reference intermediate layer outputs based on user specification and model type
        # This is done only for oneshot because for layerwise we would need all reference
        # intermediate outputs to chain previous inputs for subgraphs
        reference_intermediates_list: list[str] | None = None
        if algorithm == Algorithm.ONESHOT:
            reference_proto = qdq_proto if reference_model == qdq_model else load_model_structure(reference_model)
            # The CPU reference keeps weight and bias Q/DQ outputs: a reference dump reused as a layerwise golden
            # reference feeds them to the extracted layers.
            reference_intermediates_list = self._get_valid_intermediates(
                intermediates_layers_list,
                reference_proto,
                is_qdq_model=reference_model_is_qdq,
                skip_constant_tensors=False,
            )
            if reference_model == qdq_model and self._fusion_groups:
                # The target compares the outputs of a fusion in place of the tensors inside it.
                fusion_outputs = self._get_valid_intermediates(
                    intermediates_layers_list,
                    reference_proto,
                    is_qdq_model=True,
                    skip_constant_tensors=False,
                    keep_fusions=True,
                )
                reference_intermediates_list = list(dict.fromkeys([*reference_intermediates_list, *fusion_outputs]))
            if intermediates_layers_list and not reference_intermediates_list:
                if reference_model != qdq_model:
                    # Names may differ across models. Capture all the intermediate outputs of the
                    # reference graph before the target compile provides trace-based correspondence.
                    reference_intermediates_list = None  # Setting None will dump all intermediates
                    self._logger.warning("Reference layer names differ; capturing all reference intermediates.")
                else:
                    self._logger.warning("No layers matched set_intermediate_layers; only graph outputs are compared.")
            elif self._trace_requested and reference_model != qdq_model:
                # Names may differ across models. Capture all the intermediate outputs of the
                # reference graph before the target compile provides trace-based correspondence.
                reference_intermediates_list = None  # Setting None will dump all intermediates

        # If golden reference path is provided, load data from there
        if config.golden_reference_path:
            self._logger.info(f"Loading reference data from given {config.golden_reference_path}")
            reference_outputs = load_data_from_directory(
                Path(config.golden_reference_path), get_intermediate_outputs_info(reference_model), logger=self._logger
            )
        else:
            self._logger.info(f"Generating reference data for {reference_model}")
            reference_outputs = generate_framework_outputs(
                self._logger,
                model=reference_model,
                inputs=input_sample,
                dump_output_tensors=dump_output_tensors,
                output_dir=working_directory,
                intermediate_output_tensors=reference_intermediates_list,
            )

        if algorithm == Algorithm.ONESHOT:
            # ONESHOT: Analyze entire model in one pass
            csv_path, json_report_path = self._execute_oneshot(
                qdq_model=qdq_model,
                input_sample=input_sample,
                reference_outputs=reference_outputs,
                dump_output_tensors=dump_output_tensors,
                comparators=config.comparators,
                working_directory=working_directory,
                intermediates_layers_list=intermediates_layers_list,
                reference_model_is_qdq=reference_model_is_qdq,
                dump_elementwise_stats=config.dump_elementwise_stats,
            )
        elif algorithm == Algorithm.LAYERWISE:
            # LAYERWISE: Analyze model layer by layer
            csv_path, json_report_path = self._execute_layerwise(
                reference_model=reference_model,
                qdq_model=qdq_model,
                input_sample=input_sample,
                reference_outputs=reference_outputs,
                dump_output_tensors=dump_output_tensors,
                comparators=config.comparators,
                working_directory=working_directory,
                intermediates_layers_list=intermediates_layers_list,
                cpu_layers_list=cpu_layers_list,
                reference_model_is_qdq=reference_model_is_qdq,
                dump_elementwise_stats=config.dump_elementwise_stats,
            )
        else:
            raise ValueError(
                f"Invalid algorithm '{algorithm}' supplied. "
                f"Supported options are {Algorithm.ONESHOT.name.lower()} and {Algorithm.LAYERWISE.name.lower()}"
            )

        return OrtQnnSnooperOutputConfig(csv_snooping_report=csv_path, json_snooping_report=json_report_path)

    def _execute_oneshot(
        self,
        qdq_model: Path,
        input_sample: dict[str, np.ndarray],
        reference_outputs: dict[str, np.ndarray],
        dump_output_tensors: bool,
        comparators: list[Comparator],
        working_directory: Path,
        intermediates_layers_list: list[str] | None,
        reference_model_is_qdq: bool,
        dump_elementwise_stats: dict[str, int] | None,
    ) -> tuple[Path, Path]:
        """Runs the whole QDQ model once on the target and compares every selected intermediate.

        This method implements the ONESHOT algorithm which analyzes the entire model in a single pass.
        It performs the following steps:
        1. Runs inference on the QDQ model using QNN backend
        2. Maps target tensors to reference tensors
        3. Compares outputs using specified comparators
        4. Generates activation distribution statistics
        5. Creates comprehensive CSV and JSON reports
        6. Generates visualization plots

        The ONESHOT approach is faster than LAYERWISE but provides less granular
        debugging information.

        Args:
            qdq_model: QDQ model path.
            input_sample: Model inputs keyed by ONNX input name.
            reference_outputs: Reference tensors keyed by tensor name.
            dump_output_tensors: Dump the target tensors to ``target_outputs/``.
            comparators: Comparators to apply.
            working_directory: Output directory.
            intermediates_layers_list: Layer names / op types to compare; empty or ``None`` means all.
            reference_model_is_qdq: Whether the reference model is QDQ (selects Q+DQ vs DQ-only outputs and the
                tensor-mapping strategy).
            dump_elementwise_stats: Layer name / op type -> allowed steps, or ``None``.

        Returns:
            The CSV and JSON report paths.

        Raises:
            VerificationError: If no tensor pair could be compared.
        """
        # Capture Q and DQ outputs for a QDQ reference, DQ outputs only for an FP32 reference
        qdq_ops_index = QDQ_OPS_INDEX_DEFAULT if reference_model_is_qdq else FP32_REFERENCE_QDQ_OPS_INDEX
        qdq_proto = load_model_structure(qdq_model)
        outputs_list = self._get_valid_intermediates(
            intermediates_layers_list,
            qdq_proto,
            is_qdq_model=True,
            qdq_ops_index=qdq_ops_index,
            keep_fusions=True,
        )
        if self._fusion_groups:
            requested = self._get_valid_intermediates(
                intermediates_layers_list, qdq_proto, is_qdq_model=True, qdq_ops_index=qdq_ops_index
            )
            hidden = set(requested).difference(outputs_list)
            if hidden:
                self._logger.info(
                    f"Not exposing {len(hidden)} tensors inside QNN HTP fusions, so that the fusions stay intact."
                )
            if intermediates_layers_list:
                self._warn_fused_layers(intermediates_layers_list, qdq_proto, qdq_ops_index)
        if not outputs_list and self._get_valid_intermediates(
            intermediates_layers_list,
            qdq_proto,
            is_qdq_model=True,
            qdq_ops_index=qdq_ops_index,
            skip_constant_tensors=False,
            keep_fusions=True,
        ):
            # Without a filter, this is a weight-only QDQ model: its activations are not quantized.
            selection = (
                "set_intermediate_layers matched only weight and bias Q/DQ outputs"
                if intermediates_layers_list
                else "All Q/DQ outputs of the QDQ model are weights and biases"
            )
            self._logger.warning(f"{selection}, which are not compared; only graph outputs are compared.")

        # Execute inference with appropriate configuration
        inference_config = OrtQnnInferenceEngineInputConfig(
            qdq_model=qdq_model,
            input_sample=input_sample,
            dump_output_tensors=dump_output_tensors,
            working_directory=working_directory,
            outputs_list=outputs_list,
            enable_framework_op_trace=self._enable_framework_op_trace,
            dump_qnn_ep_input_graph=self._dump_qnn_ep_input_graph,
        )
        inference_result = OrtQnnInferenceEngine(self._logger).run(inference_config)
        self._use_generated_trace(inference_result, qdq_proto)
        target_outputs = inference_result.output_data or {}

        # Create mapping between target and reference tensor names
        tensor_mapping = self._get_onnx_tensor_mapping(
            list(reference_outputs), list(target_outputs), reference_model_is_qdq
        )

        # Compare reference outputs with target outputs using specified comparators
        verifier_output = verify(
            reference_outputs, target_outputs, self._logger, comparators=comparators, tensor_mapping=tensor_mapping
        )
        if not verifier_output:
            raise VerificationError("Verification of tensors failed. Please check the logs for more details.")

        if dump_elementwise_stats:
            for layer, steps_allowed in dump_elementwise_stats.items():
                elementwise_outputs_list = self._get_valid_intermediates(
                    [layer], qdq_proto, is_qdq_model=True, keep_fusions=True
                )
                target_outputs_subset: dict[str, np.ndarray] = {}
                reference_outputs_subset: dict[str, np.ndarray] = {}
                for output_name in elementwise_outputs_list:
                    reference_name = tensor_mapping.get(output_name)
                    if output_name in target_outputs and reference_name in reference_outputs:
                        target_outputs_subset[output_name] = target_outputs[output_name]
                        reference_outputs_subset[output_name] = reference_outputs[reference_name]
                self._dump_elementwise_stats(
                    reference_outputs_subset, target_outputs_subset, steps_allowed, working_directory
                )

        reference_activation_info = get_activations_info(reference_outputs)
        target_activation_info = get_activations_info(target_outputs)

        csv_path, json_path = self._generate_snooping_report(
            verifier_output,
            comparators,
            working_directory,
            reference_activation_info,
            target_activation_info,
            qdq_proto,
        )

        plot_graphs(
            csv_path=csv_path,
            layer_names_column="Layer Output",
            comparator_columns=[comparator.name for comparator in comparators],
            output_dir=working_directory,
            algorithm=self._name,
            logger=self._logger,
        )
        return csv_path, json_path

    def _execute_layerwise(
        self,
        reference_model: Path,
        qdq_model: Path,
        input_sample: dict[str, np.ndarray],
        reference_outputs: dict[str, np.ndarray],
        dump_output_tensors: bool,
        comparators: list[Comparator],
        working_directory: Path,
        intermediates_layers_list: list[str] | None,
        cpu_layers_list: list[str] | None,
        reference_model_is_qdq: bool,
        dump_elementwise_stats: dict[str, int] | None,
    ) -> tuple[Path, Path]:
        """Extracts every layer as a subgraph, runs it on the target with reference inputs and compares it.

        This method analyzes the model layer-by-layer by extracting per-layer subgraphs
        and running them through the ORT/QNN execution backend. For each layer:
        1. A subgraph is extracted from the layer's input to the selected de-quantized output
        2. Subgraph inputs are resolved from original inputs, golden reference outputs,
           or previously computed target outputs (for mixed CPU/QNN execution)
        3. The subgraph is executed (optionally forcing CPU for a layer if user specifies)
        4. Outputs are compared against the reference using provided comparators
        5. A CSV row is appended with per-layer status, shapes, distributions, and metrics
        One CSV row is appended (and the CSV flushed) per compared tensor or per failed layer.

        Args:
            reference_model: Reference model path (the QDQ model itself for layerwise).
            qdq_model: QDQ model path.
            input_sample: Model inputs keyed by ONNX input name.
            reference_outputs: Reference tensors keyed by tensor name.
            dump_output_tensors: Dump every subgraph's target tensors.
            comparators: Comparators to apply.
            working_directory: Output directory.
            intermediates_layers_list: Layer names / op types to debug; empty or ``None`` means all.
            cpu_layers_list: Layer names / op types to run on ORT CPU; their target outputs then feed
                later layers (mixed backend).
            reference_model_is_qdq: Whether the reference model is QDQ (tensor-mapping strategy).
            dump_elementwise_stats: Layer name / op type -> allowed steps, or ``None``.

        Returns:
            The CSV and JSON report paths.
        """
        # the layerwise reference is the QDQ model run on ORT CPU (already in reference_outputs)
        del reference_model

        intermediates_layers_list = intermediates_layers_list or []
        cpu_layers_list = cpu_layers_list or []

        # Define CSV schema: identifiers, metadata, distributions, and comparator columns
        comparator_names = [comparator.name for comparator in comparators]
        trace_columns = TRACE_COLUMNS if self._trace_requested else ()
        columns = [*_LAYERWISE_COLUMNS, *trace_columns, *comparator_names, "Exception"]

        # Pre-compute distributions for reference activations (used per-layer)
        reference_activation_info = get_activations_info(reference_outputs)

        # Initialize empty results table and output locations
        csv_path = working_directory / f"{Algorithm.LAYERWISE.value}.csv"
        csv_path.unlink(missing_ok=True)
        subgraphs_dir = working_directory / "subgraph_data"
        subgraphs_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []

        def add_row(row: dict[str, Any]) -> None:
            rows.append(row)
            append_csv_row(row, columns, csv_path)

        def add_failure_row(status: str, node: onnx.NodeProto, output_name: str, exception: str) -> None:
            add_row(
                self._build_layerwise_row(
                    status=status,
                    layer_name=node.name,
                    output_name=output_name,
                    verifier_scores=None,
                    layer_type=node.op_type,
                    layer_shape=(),
                    source_distribution=(),
                    target_distribution=(),
                    comparator_names=comparator_names,
                    exception=exception,
                )
            )

        qdq_proto = load_model_structure(qdq_model)
        extractor = OnnxSubgraphExtractor(qdq_model, logger=self._logger)
        mixed_backend = bool(cpu_layers_list)
        # Accumulator for target activations to support mixed backend chaining
        target_outputs: dict[str, np.ndarray] = {}
        used_subgraph_dirs: set[str] = set()
        # A QNN HTP fusion is snooped as one subgraph, from its first node to its outputs, so that it stays fused.
        nodes_by_name = {node.name: node for node in qdq_proto.graph.node if node.name}
        fusion_of = {name: group for group in self._fusion_groups for name in group.nodes}

        # Iterate over the layers of model
        for idx, current_layer in enumerate(qdq_proto.graph.node):
            self._op_trace_index = None
            self._runtime_fallback_reason = ""
            self._host_dequantized = frozenset()
            self._forced_cpu = False
            if current_layer.op_type in LAYERWISE_SKIP_OPS:
                self._logger.debug(f"Skipping layer {current_layer.name} as it is MATH INVARIANT")
                continue

            group = fusion_of.get(current_layer.name)
            members = [nodes_by_name[name] for name in group.nodes] if group else [current_layer]
            layers = [node for node in members if node.op_type not in LAYERWISE_SKIP_OPS]
            if layers[0].name != current_layer.name:
                self._logger.debug(f"Skipping layer {current_layer.name} as it is fused with layer {layers[0].name}")
                continue
            # The fusion's last layer produces the compared outputs, so it names the rows and the subgraph directory.
            layer = layers[-1]

            # Optional filter: process only specified nodes/op types if provided; any node of a fusion selects it
            if (
                intermediates_layers_list
                and all(node.name not in intermediates_layers_list for node in members)
                and all(node.op_type not in intermediates_layers_list for node in layers)
            ):
                continue

            # For the current node, identify the corresponding QDQ output nodes
            # that represents post-quantization values for fair comparison with reference outputs
            subgraph_intermediates = self._get_valid_intermediates(
                [current_layer.name], qdq_proto, is_qdq_model=True, keep_fusions=True
            )
            # If there are no QDQ nodes associated, skip that layer
            if not subgraph_intermediates:
                self._logger.debug(f"Skipping layer {layer.name} as it doesn't have QDQ nodes")
                continue
            # Take farthest output in the graph topology
            output_order = {name: order for order, node in enumerate(qdq_proto.graph.node) for name in node.output}
            subgraph_end_layer = max(subgraph_intermediates, key=output_order.__getitem__)

            # Create a per-layer working directory to store extracted model and dumps
            dir_name = sanitize_tensor_name(layer.name) if layer.name else f"node_{idx}"
            if dir_name in used_subgraph_dirs:
                dir_name = f"{dir_name}_{idx}"
            used_subgraph_dirs.add(dir_name)
            current_subgraph_dir = subgraphs_dir / dir_name
            current_subgraph_dir.mkdir(parents=True, exist_ok=True)

            # Decide backend for this subgraph: force CPU if configured for this node/op type.
            # This allows mixing CPU/QNN execution to isolate problematic layers while preserving
            # target outputs for subsequent subgraphs.
            use_cpu = mixed_backend and (
                any(node.name in cpu_layers_list for node in members)
                or any(node.op_type in cpu_layers_list for node in layers)
            )

            # ---- Subgraph extraction ----
            err_msg = ""
            extraction = None
            if not members[0].input or not members[0].input[0]:
                err_msg = "Layer has no primary input."
                self._logger.error(err_msg)
                add_failure_row(ActivationStatus.SKIP, layer, subgraph_end_layer, err_msg)
                continue

            try:
                if group is not None:
                    self._logger.info(
                        f"Extracting QNN HTP fusion {'+'.join(group.patterns)} of layers "
                        f"{', '.join(node.name for node in layers)} as one subgraph"
                    )
                # Extract subgraph from the layer's input to its de-quantized output
                self._logger.info(f"Extracting subgraph from layer {members[0].name} to {subgraph_end_layer}")
                extraction = extractor.extract(
                    members[0].input[0],
                    subgraph_end_layer,
                    current_subgraph_dir,
                    # The Q/DQ outputs of every output branch, which the closure of the end tensor can miss.
                    additional_outputs=subgraph_intermediates,
                    # Quantized inputs and static weights let QNN HTP build the model's quantized ops and fusions.
                    # ORT CPU would fuse such Q/DQ nodes into integer kernels, so a CPU subgraph keeps float inputs.
                    requantize_inputs=not use_cpu,
                )
            except Exception as exception:
                err_msg = f"Extraction failed with error: {exception}"
                self._logger.error(err_msg)
            if extraction is None:
                self._logger.error("Couldn't extract current layer, proceeding to next layer.")
                add_failure_row(ActivationStatus.SKIP, layer, subgraph_end_layer, err_msg)
                continue

            # ---- Subgraph inference ----
            try:
                # Resolve inputs for the subgraph with precedence: original -> target(mixed_backend) -> golden
                self._logger.info(f"Fetching inputs for the subgraph with names: {extraction.input_names}")
                subgraph_inputs = self._get_subgraph_inputs(
                    extraction.input_names, input_sample, target_outputs, reference_outputs, mixed_backend=mixed_backend
                )
                self._forced_cpu = use_cpu

                # Configure and invoke the ORT/QNN engine on the extracted subgraph
                inference_config = OrtQnnInferenceEngineInputConfig(
                    qdq_model=extraction.model_path,
                    input_sample=subgraph_inputs,
                    use_cpu=use_cpu,
                    dump_output_tensors=dump_output_tensors,
                    working_directory=current_subgraph_dir,
                    outputs_list=subgraph_intermediates,
                    enable_framework_op_trace=self._enable_framework_op_trace,
                    dump_qnn_ep_input_graph=self._dump_qnn_ep_input_graph,
                )
                self._logger.info("Executing subgraph")
                inference_output_config = self._run_inference(inference_config)
                self._use_generated_trace(inference_output_config, load_model_structure(extraction.model_path))
                target_output = inference_output_config.output_data or {}

                # Support mixed backend by chaining outputs as inputs to subsequent layers
                if mixed_backend:
                    target_outputs.update(target_output)
            except Exception as exception:
                err_msg = f"Couldn't execute extracted model, proceeding to next layer. Reason: {exception}"
                self._logger.error(err_msg)
                add_failure_row(ActivationStatus.INFERENCE_FAILURE, layer, subgraph_end_layer, err_msg)
                continue

            # ---- verification ----
            try:
                tensor_mapping = self._get_onnx_tensor_mapping(
                    list(reference_outputs), list(target_output), reference_model_is_qdq
                )

                # Slice reference outputs to only those produced by this subgraph
                reference_output: dict[str, np.ndarray] = {}
                for output_name in target_output:
                    reference_name = tensor_mapping.get(output_name)
                    if reference_name is None:
                        raise MessageKeyError(f"Couldn't fetch golden reference for: {output_name}")
                    reference_output[reference_name] = reference_outputs[reference_name]

                # Compute comparator metrics for this subgraph
                verifier_output = verify(
                    reference_output,
                    target_output,
                    self._logger,
                    comparators=comparators,
                    tensor_mapping=tensor_mapping,
                )

                if dump_elementwise_stats:
                    # Node names take precedence over op types.
                    keys = [*(node.name for node in members), *(node.op_type for node in layers)]
                    steps_allowed = next(
                        (dump_elementwise_stats[key] for key in keys if key in dump_elementwise_stats), None
                    )
                    if steps_allowed is not None:
                        self._dump_elementwise_stats(
                            {name: reference_output[ref] for name, ref in tensor_mapping.items()},
                            target_output,
                            steps_allowed,
                            working_directory,
                        )
            except Exception as exception:
                err_msg = f"Couldn't compare outputs, proceeding to next layer. Reason: {exception}"
                self._logger.error(err_msg)
                add_failure_row(ActivationStatus.VERIFICATION_FAILURE, layer, subgraph_end_layer, err_msg)
                continue

            if not verifier_output:
                err_msg = "No comparable tensors between target and reference outputs."
                self._logger.error(err_msg)
                add_failure_row(ActivationStatus.VERIFICATION_FAILURE, layer, subgraph_end_layer, err_msg)
                continue

            # ---- Record success rows ----
            # Flag rows whose target silently ran on CPU (their scores compare CPU with CPU).
            fallback_reason = inference_output_config.cpu_fallback_reason
            target_activation_info = get_activations_info(target_output)
            for (target_tensor_name, reference_tensor_name), verifier_scores in verifier_output.items():
                target_info = target_activation_info[target_tensor_name]
                reference_distribution: tuple = ()
                if reference_tensor_name in reference_activation_info:
                    reference_distribution = reference_activation_info[reference_tensor_name].distribution
                sanitized_reference_name = sanitize_tensor_name(reference_tensor_name)
                if sanitized_reference_name in reference_activation_info:
                    reference_distribution = reference_activation_info[sanitized_reference_name].distribution
                add_row(
                    self._build_layerwise_row(
                        status=ActivationStatus.SUCCESS,
                        layer_name=layer.name,
                        output_name=target_tensor_name,
                        verifier_scores=verifier_scores,
                        layer_type=layer.op_type,
                        layer_shape=target_info.shape,
                        source_distribution=reference_distribution,
                        target_distribution=target_info.distribution,
                        comparator_names=comparator_names,
                        exception=fallback_reason,
                    )
                )

        if not rows:
            pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

        json_path = self._generate_json_report(
            pd.DataFrame(rows, columns=columns), working_directory, Algorithm.LAYERWISE
        )
        plot_graphs(
            csv_path=csv_path,
            layer_names_column="Layer Output",
            comparator_columns=comparator_names,
            output_dir=working_directory,
            algorithm=self._name,
            logger=self._logger,
        )
        return csv_path, json_path

    def _run_inference(self, config: OrtQnnInferenceEngineInputConfig) -> OrtQnnInferenceEngineOutputConfig:
        """Runs one subgraph on the target, in a fresh spawn process when isolation is enabled.

        A new single-worker pool is created per call (per layer) so QNN/HTP state cannot leak between layers.
        Worker logs go to the same logger name and to the parent's log file (if it has one).

        Args:
            config: Inference engine configuration (picklable).

        Returns:
            The inference engine output.

        Raises:
            RuntimeError: ``"ORT session error - <reason>"`` for any failure, including a crashed worker.
        """
        try:
            if not self._isolate_inference:
                return OrtQnnInferenceEngine(self._logger).run(config)

            log_file = next(
                (Path(h.baseFilename) for h in self._logger.handlers if isinstance(h, logging.FileHandler)), None
            )
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp.get_context("spawn"),
                initializer=configure_worker_logging,
                initargs=(self._logger.name, self._logger.getEffectiveLevel(), log_file),
            ) as pool:
                return pool.submit(run_inference_job, config, self._logger.name).result()
        except Exception as e:
            raise RuntimeError(f"ORT session error - {e}") from e

    def _get_subgraph_inputs(
        self,
        graph_inputs: Sequence[str],
        original_inputs: Mapping[str, np.ndarray],
        target_outputs: Mapping[str, np.ndarray],
        golden_outputs: Mapping[str, np.ndarray],
        mixed_backend: bool = False,
    ) -> dict[str, np.ndarray]:
        """Resolves the input tensors of an extracted subgraph.

        Precedence: original model inputs, then (only with ``mixed_backend``) previously computed target outputs,
        then reference outputs by raw name, then by sanitized name.

        Args:
            graph_inputs: Subgraph input names.
            original_inputs: Model inputs.
            target_outputs: Target activations accumulated from earlier subgraphs.
            golden_outputs: Reference activations.
            mixed_backend: Allow feeding target activations (set when CPU layers are configured).

        Returns:
            Input name -> tensor.

        Raises:
            KeyError: ``"Couldn't fetch input: <name>"`` when an input cannot be resolved.
        """
        subgraph_inputs: dict[str, np.ndarray] = {}
        for ip in graph_inputs:
            # Highest precedence: use original model inputs if the subgraph expects them
            if ip in original_inputs:
                subgraph_inputs[ip] = original_inputs[ip]
            # Next: if mixing backends, allow feeding outputs of previously executed subgraphs
            elif mixed_backend and ip in target_outputs:
                subgraph_inputs[ip] = target_outputs[ip]
            # Fallback: use golden/reference activations for remaining inputs
            elif ip in golden_outputs:
                subgraph_inputs[ip] = golden_outputs[ip]
            # In case user passed reference outputs with sanitized names
            elif (sanitized := sanitize_tensor_name(ip)) in golden_outputs:
                subgraph_inputs[ip] = golden_outputs[sanitized]
            else:
                raise MessageKeyError(f"Couldn't fetch input: {ip}")
        return subgraph_inputs

    def _build_layerwise_row(
        self,
        status: str,
        layer_name: str,
        output_name: str,
        verifier_scores: Mapping[str, Any] | None,
        layer_type: str,
        layer_shape: tuple,
        source_distribution: tuple,
        target_distribution: tuple,
        comparator_names: Sequence[str],
        exception: str = "",
    ) -> dict[str, Any]:
        """Builds one layerwise CSV row.

        Args:
            status: One of the :class:`ActivationStatus` values.
            layer_name: Node name.
            output_name: Compared tensor (the subgraph end tensor for failed layers).
            verifier_scores: Comparator scores keyed by comparator name (``None`` unless SUCCESS).
            layer_type: Node op type.
            layer_shape: Target tensor shape (``()`` unless SUCCESS).
            source_distribution: Reference (min, max, median).
            target_distribution: Target (min, max, median).
            comparator_names: Comparator columns, in order.
            exception: Error text for failed layers.

        Returns:
            Column name -> cell value; comparator cells are Python scalars on SUCCESS, else ``"NaN"``.
        """
        row: dict[str, Any] = {
            "Layer Name": layer_name,
            "Layer Output": output_name,
            "Layer Status": status,
            "Layer Type": layer_type,
            "Layer Shape": layer_shape,
            "Source(Min, Max, Median)": source_distribution,
            "Target(Min, Max, Median)": target_distribution,
        }

        # Populate comparator metric columns when verification succeeded, otherwise "NaN"
        success = status == ActivationStatus.SUCCESS and verifier_scores is not None
        for comparator_name in comparator_names:
            row[comparator_name] = to_py_scalar(verifier_scores[comparator_name]) if success else "NaN"

        # Attach any error/exception message captured during extraction/inference/verification
        row["Exception"] = exception
        row.update(self._trace_metadata(output_name))

        return row

    def _build_data_frame_for_subgraph(
        self,
        status: str,
        layer_name: str,
        output_name: str,
        verifier_scores: Mapping[str, Any] | None,
        layer_type: str,
        layer_shape: tuple,
        source_distribution: tuple,
        target_distribution: tuple,
        comparator_names: Sequence[str],
        data_frame: pd.DataFrame,
        exception: str = "",
    ) -> pd.DataFrame:
        """Appends a layerwise row to ``data_frame`` (legacy-compatible wrapper of :meth:`_build_layerwise_row`).

        Args:
            status: One of the :class:`ActivationStatus` values.
            layer_name: Node name.
            output_name: Compared tensor.
            verifier_scores: Comparator scores keyed by comparator name.
            layer_type: Node op type.
            layer_shape: Target tensor shape.
            source_distribution: Reference (min, max, median).
            target_distribution: Target (min, max, median).
            comparator_names: Comparator columns, in order.
            data_frame: Frame to append to (modified in place).
            exception: Error text for failed layers.

        Returns:
            ``data_frame`` with the new row.
        """
        row = self._build_layerwise_row(
            status=status,
            layer_name=layer_name,
            output_name=output_name,
            verifier_scores=verifier_scores,
            layer_type=layer_type,
            layer_shape=layer_shape,
            source_distribution=source_distribution,
            target_distribution=target_distribution,
            comparator_names=comparator_names,
            exception=exception,
        )
        data_frame.loc[len(data_frame)] = row
        return data_frame

    def _dump_elementwise_stats(
        self,
        reference_output: Mapping[str, np.ndarray],
        target_output: Mapping[str, np.ndarray],
        steps_allowed: int | float | str,
        working_directory: Path,
    ) -> None:
        """Appends the elements whose absolute difference exceeds ``steps_allowed`` to ``elementwise_stats.log``.

        Compares each corresponding element in target vs reference outputs and logs
        detailed differences that exceed an allowed step threshold. This is useful
        for pinpointing exact indices where quantization deviates beyond tolerance.
        Integer/integer pairs are compared in int64 so unsigned tensors do not wrap around.
        Nothing is written for a tensor without out-of-tolerance elements.
        Never raises; problems are logged as warnings.

        Args:
            reference_output: Reference tensors by name.
            target_output: Target tensors by name.
            steps_allowed: Allowed absolute difference (coerced to float; invalid values fall back to 0).
            working_directory: Directory of ``elementwise_stats.log``.
        """
        try:
            allowed = float(steps_allowed)
        except Exception:
            self._logger.warning(f"Invalid steps_allowed='{steps_allowed}'. Falling back to 0.")
            allowed = 0.0

        try:
            # Iterate over outputs present in target_output; we will only compare those
            # that are also present in reference_output to avoid KeyErrors.
            for output_name in target_output:
                # Retrieve the candidate tensors. If either is missing, skip gracefully.
                reference = reference_output.get(output_name)
                target = target_output.get(output_name)
                if reference is None or target is None:
                    self._logger.warning(
                        f"Skipping elementwise stats for '{output_name}' as it is missing in reference/target outputs."
                    )
                    continue

                # Elementwise comparison requires shape equality so that indexing aligns.
                if reference.shape != target.shape:
                    self._logger.warning(
                        f"Skipping elementwise stats for '{output_name}' as reference shape: "
                        f"{reference.shape} and target shape: {target.shape} are not matching."
                    )
                    continue

                log = f"Starting Elementwise comparison for {output_name}\n"
                self._logger.debug(log)

                t, r = target, reference
                if t.dtype.kind in "iub" and r.dtype.kind in "iub":
                    t, r = t.astype(np.int64), r.astype(np.int64)
                diff = np.abs(t - r)
                over = np.argwhere(diff > allowed)
                lines = []
                for position in over:
                    index = tuple(int(i) for i in position)
                    lines.append(
                        f"Difference={diff[index]} at index {index}, Steps allowed={allowed}, "
                        f"Target value: {target[index]}, Reference value: {reference[index]}\n"
                    )
                diff_counter = len(lines)

                if diff_counter:
                    max_diff = diff.max() if diff.size else -np.inf
                    log += "".join(lines)
                    log += f"Total number of elements that are further than {allowed} steps: {diff_counter}\n"
                    log += f"Maximum step difference: {max_diff}\n\n\n"
                    save_path = Path(working_directory) / "elementwise_stats.log"
                    try:
                        with open(save_path, "a") as file:
                            file.write(log)
                    except Exception as file_err:
                        self._logger.warning(
                            f"Couldn't write elementwise stats for '{output_name}' to '{save_path}'. Reason: {file_err}"
                        )
        except Exception as e:
            self._logger.warning(f"Dumping elementwise stats failed. Reason: {e}")

    def _warn_fused_layers(
        self,
        intermediates_layers_list: Sequence[str],
        proto: onnx.ModelProto,
        qdq_ops_index: Mapping[str, int],
    ) -> None:
        """Warns about requested layers inside a QNN HTP fusion, whose fusion outputs are compared in their place.

        A requested layer is replaced when its own Q/DQ outputs are inside the fusion, or when it has none. The last
        layer of a fusion produces the fusion outputs, so it is not replaced.

        Args:
            intermediates_layers_list: Node names / op types of ``set_intermediate_layers``.
            proto: The QDQ model.
            qdq_ops_index: Q/DQ op types whose outputs are compared.
        """
        nodes = {node.name: node for node in proto.graph.node if node.name}
        qdq_consumers = _get_qdq_consumers(proto.graph.node)
        constant_tensors = _get_constant_tensors(proto.graph)

        def selected(names: Iterable[str]) -> list[str]:
            outputs = [
                tensor
                for name in names
                for tensor in _get_downstream_qdq_outputs(nodes[name], qdq_consumers, qdq_ops_index)
            ]
            return [tensor for tensor in dict.fromkeys(outputs) if tensor not in constant_tensors]

        for group in self._fusion_groups:
            outputs = [tensor for tensor in selected(group.nodes) if tensor not in group.internal_tensors]
            inside = []
            for name in group.nodes:
                if name in intermediates_layers_list or nodes[name].op_type in intermediates_layers_list:
                    own = selected([name])
                    if not own or group.internal_tensors.intersection(own):
                        inside.append(name)
            if outputs and inside:
                layers = [name for name in group.nodes if nodes[name].op_type not in QDQ_OPS_INDEX_DEFAULT]
                self._logger.warning(
                    f"{', '.join(inside)} {'is' if len(inside) == 1 else 'are'} inside QNN HTP fusion "
                    f"{'+'.join(group.patterns)} ({', '.join(layers or group.nodes)}). Comparing the fusion outputs "
                    f"instead, so that QNN EP keeps the fusion: {', '.join(outputs)}"
                )

    def _get_valid_intermediates(
        self,
        intermediates_layers_list: Sequence[str] | None,
        model: Path | str | onnx.ModelProto,
        is_qdq_model: bool = False,
        qdq_ops_index: Mapping[str, int] | None = None,
        skip_constant_tensors: bool = True,
        keep_fusions: bool = False,
    ) -> list[str]:
        """Selects the output tensor names to capture for the given layer filter.

        Decision Logic:
        1. If intermediates_layers_list is provided and node name or op_type matches:
              - QDQ model → capture the outputs of the downstream Q/DQ nodes whose op types are in qdq_ops_index
                    - When reference is QDQ, qdq_ops_index: {"QuantizeLinear": 1, "DequantizeLinear": 2}
                    - When reference is FP32, qdq_ops_index: {"DequantizeLinear": 2}
              - FP32 model → capture node outputs directly
        2. If intermediates_layers_list is empty:
           - QDQ model → capture outputs from nodes whose op_type is present in qdq_ops_index
           - FP32 model → capture all node outputs

        Downstream Q/DQ nodes are found by data edges (see ``_get_downstream_qdq_outputs``), not by node position:
        where branches interleave, as at a residual downsample, a node's Q and DQ need not directly follow it.

        By default, QDQ models never capture weight and bias Q/DQ outputs (see ``_get_constant_tensors``). They are
        not activations, and QNN EP does not take DQ nodes that generate a graph output. Q/DQ outputs that QNN EP
        computes from weights are captured.

        With ``keep_fusions``, the QNN HTP fusions found by the fusion probe stay intact: the tensors inside a fusion
        are never captured, and a selected node inside a fusion selects every node of the fusion, which captures the
        fusion outputs instead.

        Args:
            intermediates_layers_list: Node names / op types; empty or ``None`` means all.
            model: Model path or proto (a path is loaded structure-only).
            is_qdq_model: Treat the model as QDQ.
            qdq_ops_index: Q/DQ op types to capture, as its keys; ``None`` means ``{"QuantizeLinear": 1,
                "DequantizeLinear": 2}``. The values are the legacy look-ahead offsets and are not used.
            skip_constant_tensors: Drop weight and bias Q/DQ outputs of a QDQ model. The CPU reference keeps them.
            keep_fusions: Keep the QNN HTP fusions of a QDQ model intact (see ``FusionGroup``).

        Returns:
            Output tensor names in node order. The Q/DQ outputs of a selected node are in the order
            ``_get_downstream_qdq_outputs`` visits them.
        """
        proto = model if isinstance(model, onnx.ModelProto) else load_model_structure(model)
        qdq_ops_index = QDQ_OPS_INDEX_DEFAULT if qdq_ops_index is None else qdq_ops_index
        nodes = proto.graph.node
        constant_tensors = _get_constant_tensors(proto.graph) if is_qdq_model and skip_constant_tensors else set()
        groups = self._fusion_groups if keep_fusions and is_qdq_model else []
        # Weight and bias Q/DQ outputs, and the tensors inside kept fusions, are dropped from the QDQ selection.
        skipped = constant_tensors.union(*(group.internal_tensors for group in groups))
        valid_intermediates: list[str] = []

        if is_qdq_model and intermediates_layers_list:

            def selects(node: onnx.NodeProto) -> bool:
                return node.name in intermediates_layers_list or node.op_type in intermediates_layers_list

            # A node inside a kept fusion selects every node of the fusion, so the selection holds the fusion outputs.
            matched = {node.name for node in nodes if selects(node)}
            fused = {name for group in groups if matched.intersection(group.nodes) for name in group.nodes}
            qdq_consumers = _get_qdq_consumers(nodes)
            for node in nodes:
                if selects(node) or node.name in fused:
                    valid_intermediates.extend(_get_downstream_qdq_outputs(node, qdq_consumers, qdq_ops_index))
            return [name for name in dict.fromkeys(valid_intermediates) if name not in skipped]

        for node in nodes:
            if intermediates_layers_list:
                # Condition #1 - User specified specific layers to analyze; QDQ models are handled above
                if node.name in intermediates_layers_list or node.op_type in intermediates_layers_list:
                    # For FP32 models, capture node outputs directly
                    valid_intermediates.extend(node.output)
            else:
                # Condition #2 - No specific layers specified, capture based on model type
                if is_qdq_model:
                    # For QDQ models, select downstream QDQ outputs based on qdq_ops_index
                    if node.op_type in qdq_ops_index:
                        valid_intermediates.extend(node.output)
                else:
                    # For FP32 models, capture node outputs directly
                    valid_intermediates.extend(node.output)

        return [name for name in valid_intermediates if name not in skipped]

    def _get_onnx_tensor_mapping(
        self,
        reference_outputs_names: Sequence[str],
        target_outputs_names: Sequence[str],
        reference_model_is_qdq: bool,
    ) -> dict[str, str]:
        """Maps target tensor names to reference tensor names.

        This function establishes a correspondence between tensor names in the target
        (quantized) model and the reference (floating-point or QDQ) model. The mapping
        strategy differs based on whether the reference model is in QDQ format.

        Mapping Strategies:
        -------------------
        1. **Framework op-tracing** (Always attempted first):
            - Uses generated or supplied framework op-tracing data
        2. **Exact Name Match**:
           - Direct match: "conv1_output" → "conv1_output"
           - Applied to both QDQ and FP32 reference models

        3. **Transformed Name Match**:
           - Handles external golden output files scenario
           - Uses Helper.transform_node_names() to normalize names
           - Example: "/conv1/output" → "conv1_output"

        4. **Prefix Matching** (Only when reference is FP32):
           - Handles quantization-added suffixes
           - Example: "conv1_output_quantized" → "conv1_output"
           - Example: "layer2_dequantized" → "layer2"
           - First match wins (no ambiguity resolution)

        Rationale:
        ----------
        Quantization often modifies tensor names by adding suffixes like:
        - "_quantized", "_dequantized", "_qdq"
        - "_scale", "_zero_point"

        Notes:
            - Unmapped target tensors are silently excluded from the result
            - The function assumes quantization adds suffixes, not prefixes
            - First match wins in prefix matching (no ambiguity resolution)

        Args:
            reference_outputs_names: Reference tensor names.
            target_outputs_names: Target tensor names.
            reference_model_is_qdq: Disables prefix matching when ``True``.

        Returns:
            Target name -> reference name for every mapped target tensor.
        """
        reference_names = set(reference_outputs_names)
        tensor_mapping: dict[str, str] = {}

        def is_boundary(name: str, prefix_length: int) -> bool:
            return prefix_length == len(name) or name[prefix_length] in ("_", "/", ".", ":")

        # Iterate through all target model outputs to find corresponding reference outputs
        # Each target tensor needs to be matched to exactly one reference tensor
        for target_output_name in target_outputs_names:
            # ============================================================
            # Strategy 1: Use generated or supplied framework op-tracing
            # ============================================================
            mapped = None
            ambiguous = False
            for index in (self._supplied_op_trace, self._op_trace_index):
                if index is not None:
                    mapped, ambiguous = index.resolve(target_output_name, reference_names, reference_model_is_qdq)
                    if mapped is not None or ambiguous:
                        break
            if ambiguous:
                self._logger.warning(
                    f"Ambiguous framework op trace mapping for '{target_output_name}'; skipping tensor."
                )
                continue
            if mapped is not None:
                tensor_mapping[target_output_name] = mapped
                continue

            # ============================================================
            # Strategy 2: Exact Name Match
            # ============================================================
            if target_output_name in reference_names:
                tensor_mapping[target_output_name] = target_output_name
                continue  # Move to next target output (match found)

            # ============================================================
            # Strategy 3: Sanitized Name Match
            # ============================================================
            transformed_target_name = sanitize_tensor_name(target_output_name)
            if transformed_target_name in reference_names:
                tensor_mapping[target_output_name] = transformed_target_name
                continue  # Move to next target output (match found)

            # ============================================================
            # Strategy 4: Prefix Matching (FP32 Reference Only)
            # ============================================================
            # If no exact match and reference is FP32, try prefix matching
            # This handles quantization-introduced naming changes such as:
            # - "layer_output" (FP32) → "layer_output_quantized" (Quantized)
            # - "conv1" (FP32) → "conv1_dequantized" (Quantized)
            # - "fc" (FP32) → "fc_qdq" (Quantized)
            # Additional validation: ensure it's a true suffix match
            # Prevent false matches like "conv1" matching "conv10"
            # by checking if the next character is a separator or end of string
            if not reference_model_is_qdq:
                # Search through all reference outputs for potential prefix matches
                for reference_output_name in reference_outputs_names:
                    prefix_length = len(reference_output_name)
                    if target_output_name.startswith(reference_output_name) and is_boundary(
                        target_output_name, prefix_length
                    ):
                        # Use first match (assumes no ambiguous prefixes)
                        tensor_mapping[target_output_name] = reference_output_name
                        break
                    elif transformed_target_name.startswith(reference_output_name) and is_boundary(
                        transformed_target_name, prefix_length
                    ):
                        tensor_mapping[target_output_name] = reference_output_name
                        break

        # Calculate mapping statistics for debugging and monitoring
        mapped_count = len(tensor_mapping)
        total_target_count = len(target_outputs_names)
        self._logger.info(
            f"Tensor mapping created: {mapped_count} out of {total_target_count} target tensors mapped to reference."
        )
        if mapped_count < total_target_count:
            unmapped_tensors = set(target_outputs_names) - set(tensor_mapping)
            self._logger.debug(f"Unmapped target tensors: {sorted(unmapped_tensors)}")

        return tensor_mapping

    def _generate_snooping_report(
        self,
        verifier_output: VerifierResult,
        comparators: Sequence[Comparator],
        working_directory: Path,
        reference_activation_info: Mapping[str, ActivationInfo],
        target_activation_info: Mapping[str, ActivationInfo],
        qdq_model: Path | onnx.ModelProto,
    ) -> tuple[Path, Path]:
        """Writes the oneshot CSV and JSON reports.

        This method creates detailed reports containing:
        - Layer-wise comparison metrics (MSE, SQNR, etc.)
        - Tensor distribution statistics (min, max, median)
        - Layer metadata (name, type, shape)
        - Parent layer information for DequantizeLinear nodes

        The reports are generated in two formats:
        1. CSV: Human-readable tabular format for analysis in spreadsheet tools
        2. JSON: Machine-readable format for automation and further processing

        Args:
            verifier_output: ``(target name, reference name)`` -> comparator scores.
            comparators: Comparators whose names become score columns.
            working_directory: Output directory.
            reference_activation_info: Reference tensor statistics.
            target_activation_info: Target tensor statistics.
            qdq_model: QDQ model path or proto, used to name each row after the op feeding the Q/DQ node.

        Returns:
            The CSV (``oneshot.csv``) and JSON (``oneshot.json``) paths.

        Example:
            CSV Report Structure:
            | Layer Name | Layer Output   | Layer Type | Layer Shape | Source(...) | Target(...) | MSE | SQNR |
            |------------|----------------|------------|-------------|-------------|-------------|-----|------|
            | conv1      | conv1_dequant  | Conv       | [1,64,56,56]| (0,1,0.5)   | (0,1,0.5)   | 0.01| 45.2 |
        """
        model = qdq_model if isinstance(qdq_model, onnx.ModelProto) else load_model_structure(qdq_model)
        model_nodes = model.graph.node
        output_index: dict[str, int] = {}
        for idx, node in enumerate(model_nodes):
            for output in node.output:
                output_index.setdefault(output, idx)

        snooping_report = []
        for (target_tensor_name, reference_tensor_name), scores in verifier_output.items():
            # Get corresponding parent Op details for QDQ output nodes
            parent_op = self._get_qdq_parent_map(target_tensor_name, model_nodes, output_index)
            parent_op_name = parent_op_type = ""
            if parent_op:
                parent_op_name, parent_op_type = parent_op.name, parent_op.op_type

            target_info = target_activation_info[target_tensor_name]
            reference_distribution: tuple = ()
            if reference_tensor_name in reference_activation_info:
                reference_distribution = reference_activation_info[reference_tensor_name].distribution

            # Create report entry with layer information and statistics
            report_entry: dict[str, Any] = {
                "Layer Name": parent_op_name,
                "Layer Output": target_tensor_name,
                "Layer Type": parent_op_type,
                "Layer Shape": target_info.shape,
                "Source(Min, Max, Median)": reference_distribution,
                "Target(Min, Max, Median)": target_info.distribution,
            }
            report_entry.update(self._trace_metadata(target_tensor_name))
            # Add comparator results to report entry
            for comparator in comparators:
                report_entry[comparator.name] = scores[comparator.name]
            snooping_report.append(report_entry)

        trace_columns = TRACE_COLUMNS if self._trace_requested else ()
        columns = [*_ONESHOT_COLUMNS, *trace_columns, *(comparator.name for comparator in comparators)]
        data_frame = pd.DataFrame(snooping_report, columns=columns)
        csv_path = Path(working_directory) / f"{Algorithm.ONESHOT.value}.csv"
        data_frame.to_csv(csv_path, index=False)
        json_path = self._generate_json_report(data_frame, Path(working_directory), Algorithm.ONESHOT)
        return csv_path, json_path

    def _get_qdq_parent_map(
        self,
        tensor_name: str,
        model_nodes: Sequence[onnx.NodeProto],
        output_index: Mapping[str, int] | None = None,
    ) -> onnx.NodeProto | None:
        """Returns the op a Q/DQ output belongs to: the first non-Q/DQ op up the chain of Q/DQ data inputs.

        Report rows are named after this op (e.g. "conv1" instead of "conv1_dequantize"). Data edges are followed,
        not node positions: where branches interleave, as at a residual downsample, the op need not directly precede
        its Q/DQ nodes.

        Args:
            tensor_name: Tensor to trace.
            model_nodes: Graph nodes in order.
            output_index: Optional tensor name -> producing node index (first producer); built when omitted.

        Returns:
            node or None: The parent operation node if found, otherwise None.
                - For a Q/DQ output: the op that feeds the chain of Q/DQ nodes ending at the tensor.
                - For a Q/DQ chain that starts at a graph input or an initializer: the node producing the tensor.
                - For other nodes: the node that produces the tensor directly.
                - If tensor not found in any node outputs: None.

        Example:
            >>> # Conv "conv1" → QuantizeLinear → DequantizeLinear → "conv1_dequant_output"
            >>> parent = self._get_qdq_parent_map("conv1_dequant_output", model_nodes)
            >>> print(parent.name)  # "conv1"
            >>> print(parent.op_type)  # "Conv"
        """
        if output_index is None:
            output_index = {}
            for idx, node in enumerate(model_nodes):
                for output in node.output:
                    output_index.setdefault(output, idx)

        idx = output_index.get(tensor_name)
        if idx is None:
            return None

        producer = parent = model_nodes[idx]
        visited = {idx}
        while parent.op_type in QDQ_OPS_INDEX_DEFAULT and parent.input:
            idx = output_index.get(parent.input[0])
            if idx is None or idx in visited:
                return producer
            visited.add(idx)
            parent = model_nodes[idx]
        return parent

    def _generate_json_report(
        self, data_frame: pd.DataFrame, output_dir: Path, algorithm: Algorithm = Algorithm.ONESHOT
    ) -> Path:
        """Writes ``<output_dir>/<algorithm>.json`` from the report frame.

        Args:
            data_frame: Report rows.
            output_dir: Output directory.
            algorithm: Algorithm the report belongs to (file name and JSON header).

        Returns:
            The JSON report path.
        """
        algorithm = Algorithm(algorithm)
        return dump_json_report(data_frame, Path(output_dir) / f"{algorithm.value}.json", algorithm)


def execute_ort_qnn_snooper(args: Sequence[str] | None = None) -> None:
    """Parses the CLI arguments, runs the snooper and logs where the reports were written.

    Args:
        args: Command-line arguments (``None`` reads ``sys.argv[1:]``).

    Raises:
        argparse.ArgumentTypeError: For invalid arguments (before any logger exists).
        Exception: Any snooping error, after it has been logged with its traceback.
    """
    parsed = OrtQnnSnooperParser().parse(args)
    algorithm = Algorithm(parsed.algorithm)
    working_directory = create_working_directory(
        sub_directory=f"{algorithm.value}_snooping",
        working_directory=parsed.working_directory,
        output_directory=parsed.output_directory,
    )
    logger = get_logger(
        LOGGER_NAME, log_file_path=working_directory, log_file_name="accuracy_debugger", level=parsed.log_level
    )

    logger.info("Running ORT/QNN-EP snooping...")
    try:
        input_processor = InputProcessor(
            input_list=Path(parsed.input_list),
            model=Path(parsed.reference_model or parsed.qdq_model),
            working_dir=working_directory,
            logger=logger,
        )
        snooper_config = OrtQnnSnooperInputConfig(
            qdq_model=parsed.qdq_model,
            reference_model=parsed.reference_model,
            input_processor=input_processor,
            algorithm=algorithm,
            comparators=parsed.comparator,
            working_directory=working_directory,
            golden_reference_path=parsed.golden_reference,
            set_intermediate_layers=parsed.set_intermediate_layers,
            set_cpu_layers=parsed.set_cpu_layers,
            dump_elementwise_stats=parsed.dump_elementwise_stats,
            dump_output_tensors=True,
            op_trace_file=parsed.op_trace_file,
            enable_framework_op_trace=parsed.enable_framework_op_trace,
            dump_qnn_ep_input_graph=parsed.dump_qnn_ep_input_graph,
        )
        output = OrtQnnSnooper(logger, name=algorithm).run(snooper_config)
    except Exception as e:
        logger.exception("Error occurred while executing ORT/QNN-EP Snooping.")
        with contextlib.suppress(AttributeError):  # lets main() skip a second report of an already-logged error
            e._adbg_logged = True  # type: ignore[attr-defined]
        raise
    logger.info(f"Snooping Completed. CSV Report generated at: {output.csv_snooping_report}")
    logger.info(f"Json Report generated at: {output.json_snooping_report}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (``None`` reads ``sys.argv[1:]``).

    Returns:
        0 on success, 2 for invalid arguments, 1 for any other error. ``SystemExit`` from argparse (``--help``,
        unknown flags) and ``KeyboardInterrupt`` propagate.
    """
    try:
        execute_ort_qnn_snooper(argv)
    except argparse.ArgumentTypeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        if not getattr(e, "_adbg_logged", False):  # failed before the logger existed
            print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
