# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Layer-window subgraph extraction for layerwise snooping.

The model is loaded, shape-inferred and indexed once; every :meth:`OnnxSubgraphExtractor.extract` call then cuts the
window ``(producer(start), producer(end)]`` out of the graph with the iterative ``onnx.utils.Extractor``.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import onnx
import onnx.utils
from common import SubgraphExtractionError
from onnx_utils import constant_tensors, infer_model_with_data

__all__ = ["OnnxSubgraphExtractor", "SubgraphExtractionResult"]

_SYMBOLIC_PARTIAL_FILE = "sym_shape_infer_temp.onnx"
# Domains of the DequantizeLinear nodes whose outputs a requantized subgraph input is quantized again with.
_QDQ_DOMAINS = frozenset({"", "ai.onnx", "com.microsoft"})


@dataclass(frozen=True)
class SubgraphExtractionResult:
    """Result of one subgraph extraction.

    Attributes:
        model_path: Path of the saved, checked subgraph model.
        input_names: Real feeds of the subgraph (``graph.input`` minus initializers), in graph order.
        output_names: Graph outputs of the subgraph, ``end`` first.
    """

    model_path: Path
    input_names: list[str]
    output_names: list[str]


@contextmanager
def _temporary_cwd() -> Iterator[Path]:
    """Runs the body with a fresh temporary directory as cwd, so nothing is written to the user's cwd."""
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(previous)


class OnnxSubgraphExtractor:
    """Extracts the subgraph between two tensors of an ONNX model.

    For ``p`` = index of the producer of ``start`` (``-1`` for a graph input) and ``e`` = index of the producer of
    ``end``, the window is ``nodes[p+1 .. e]``:

    * outputs: ``end``, the original graph outputs that the nodes produce, then ``additional_outputs``. Graph outputs
      of other branches in the window are left out, as they would add those branches' nodes;
    * nodes: backward closure from ``end`` and ``additional_outputs``, stopping at ``start`` and at tensors available
      before the window;
    * inputs: ``start`` plus every non-initializer tensor consumed by the closure that is produced at or before ``p``
      or is a graph input, in first-use order; inputs no node consumes are pruned.

    With ``requantize_inputs``, the subgraph's nodes consume quantized tensors as they do in the model:

    * a constant tensor produced at or before ``p``, such as the output of a weight's DequantizeLinear, is computed in
      the subgraph by the model's nodes and initializers instead of being an input (``start`` stays an input);
    * an input produced by a DequantizeLinear with constant parameters is quantized again by an inserted
      QuantizeLinear -> DequantizeLinear pair with those parameters. Inputs with int32 data stay float, as
      QuantizeLinear cannot produce int32.

    Attributes:
        FILE_NAME: File name of the saved subgraph inside ``out_dir``.
    """

    FILE_NAME: ClassVar[str] = "extracted_model.onnx"

    def __init__(
        self,
        model_path: str | Path,
        *,
        logger: logging.Logger | None = None,
        symbolic_shape_fallback: bool = True,
    ) -> None:
        """Loads, shape-infers and indexes the model.

        Args:
            model_path: Path to the ONNX model.
            logger: Logger for warnings/debug messages; defaults to this module's logger.
            symbolic_shape_fallback: Fill value_info that ONNX shape inference could not produce (e.g. for
                ``com.microsoft`` Q/DQ nodes) with onnxruntime's symbolic shape inference, when available.
        """
        self._path = Path(model_path)
        self._log = logger or logging.getLogger(__name__)
        model = infer_model_with_data(self._path, self._log)
        if symbolic_shape_fallback:
            self._fill_missing_value_info(model)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):  # only above 2 GB
            onnx.load_external_data_for_model(model, str(self._path.parent))
        self._ir_lt4 = model.ir_version < 4
        graph = model.graph
        self._nodes = list(graph.node)
        self._inits = {t.name: t for t in graph.initializer}
        self._feeds = {vi.name for vi in graph.input if vi.name not in self._inits}
        self._producer = {out: idx for idx, node in enumerate(self._nodes) for out in node.output if out}
        self._graph_outputs = [vi.name for vi in graph.output]
        self._constants = constant_tensors(self._nodes, self._inits)
        self._elem_types = {t.name: t.data_type for t in graph.initializer}
        for vi in [*graph.input, *graph.value_info, *graph.output]:
            if vi.type.HasField("tensor_type"):
                self._elem_types.setdefault(vi.name, vi.type.tensor_type.elem_type)
        # Names used in the model, which the nodes and tensors added to a subgraph must not reuse.
        self._names = {name for node in self._nodes for name in (node.name, *node.input, *node.output)}
        self._names.update(self._inits, self._feeds)
        # onnx 1.18 only resolves extracted inputs against graph.input + value_info; make graph outputs resolvable too.
        known_vi = {vi.name for vi in graph.value_info}
        graph.value_info.extend(vi for vi in graph.output if vi.name not in known_vi)
        self._extractor = onnx.utils.Extractor(model)

    def _fill_missing_value_info(self, model: onnx.ModelProto) -> None:
        """Adds value_info for node outputs ONNX shape inference left untyped, using symbolic shape inference.

        Only the public ``SymbolicShapeInference.infer_shapes`` is used, inside a temporary cwd so its failure dump
        (``sym_shape_infer_temp.onnx``) never lands in the user's directory; on incomplete inference the partial
        result is read back from that dump. Any failure is logged and ignored; the ONNX shape-inference results are
        kept.

        Args:
            model: Model to complete, in place.
        """
        graph = model.graph
        known = {vi.name for vi in [*graph.value_info, *graph.input, *graph.output]}
        missing = {out for node in graph.node for out in node.output if out and out not in known}
        if not missing:
            return
        try:
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference  # noqa: PLC0415
        except ImportError as exc:
            self._log.debug(f"Symbolic shape inference unavailable: {exc}")
            return
        inferred = self._run_symbolic_shape_inference(SymbolicShapeInference, model)
        if inferred is None:
            return
        graph.value_info.extend(vi for vi in inferred.graph.value_info if vi.name in missing)

    def _run_symbolic_shape_inference(self, ssi_cls: type, model: onnx.ModelProto) -> onnx.ModelProto | None:
        """Runs the public ``SymbolicShapeInference.infer_shapes`` and returns its (possibly partial) result.

        Rank guessing for unknown ops is tried first and dropped on a hard failure (it cannot handle unranked
        inputs). On incomplete inference the public API dumps its partial result into the cwd before raising; it runs
        in a temporary cwd, so the dump never reaches the user's directory and is read back from there.

        Args:
            ssi_cls: The ``SymbolicShapeInference`` class.
            model: Model to infer (not modified).

        Returns:
            The inferred model, or ``None`` if nothing could be inferred.
        """
        for guess_output_rank in (True, False):
            with _temporary_cwd() as tmp:
                try:
                    inferred = ssi_cls.infer_shapes(model, auto_merge=True, guess_output_rank=guess_output_rank)
                except Exception as exc:
                    partial = tmp / _SYMBOLIC_PARTIAL_FILE
                    if partial.is_file():
                        self._log.warning(f"Symbolic shape inference incomplete, using partial result: {exc}")
                        try:
                            return onnx.load(str(partial), load_external_data=False)
                        except Exception as load_exc:
                            self._log.debug(f"Could not read partial symbolic shape inference result: {load_exc}")
                            return None
                    self._log.debug(f"Symbolic shape inference failed (guess_output_rank={guess_output_rank}): {exc}")
                    continue
            if inferred is None:
                self._log.debug("Symbolic shape inference skipped (opset < 7).")
            return inferred
        self._log.warning("Symbolic shape inference failed; continuing with ONNX shape inference results.")
        return None

    def extract(
        self,
        start: str,
        end: str,
        out_dir: str | Path,
        *,
        additional_outputs: tuple[str, ...] | list[str] = (),
        requantize_inputs: bool = False,
    ) -> SubgraphExtractionResult:
        """Extracts the subgraph from ``start`` (exclusive producer) to ``end`` and saves it to ``out_dir``.

        Args:
            start: Tensor where the subgraph begins (it becomes the first input).
            end: Tensor where the subgraph ends (it becomes the first output).
            out_dir: Directory for ``extracted_model.onnx`` (created if needed).
            additional_outputs: Other branch outputs to retain within the same extraction window.
            requantize_inputs: Compute the constant inputs in the subgraph and quantize the dequantized inputs again
                (see the class docstring).

        Returns:
            The saved model path and its input and output names.

        Raises:
            SubgraphExtractionError: If the tensors are not a valid window, or extraction, saving or checking fails.
        """
        if start in self._producer:
            p = self._producer[start]
        elif start in self._feeds:
            p = -1
        else:
            raise SubgraphExtractionError(f"Start tensor '{start}' is not produced by any node or graph input.")

        e = self._producer.get(end)
        if e is None or e <= p:
            raise SubgraphExtractionError(f"End tensor '{end}' is not produced after start tensor '{start}'.")

        no_producer = len(self._nodes)

        def is_boundary(tensor: str) -> bool:
            return self._producer.get(tensor, no_producer) <= p or tensor in self._feeds

        for output in additional_outputs:
            if not p < self._producer.get(output, -1) <= e:
                raise SubgraphExtractionError(f"Additional output '{output}' is outside the extraction window.")

        closure: set[int] = set()
        stack = [end, *additional_outputs]
        while stack:
            tensor = stack.pop()
            idx = self._producer.get(tensor)
            if tensor == start or idx is None or idx <= p or idx in closure:
                continue
            closure.add(idx)
            stack.extend(name for name in self._nodes[idx].input if name)

        # A graph output of another branch in the window is left out, so that its branch is not extracted too.
        outputs = [end] + [o for o in self._graph_outputs if o != end and self._producer.get(o) in closure]
        for output in additional_outputs:
            if output not in outputs:
                outputs.append(output)

        # The extractor copies the nodes and initializers computing each tensor that is not an input.
        computed = self._constants if requantize_inputs else self._inits
        inputs = [start]
        for idx in sorted(closure):
            for tensor in self._nodes[idx].input:
                if tensor and tensor not in computed and is_boundary(tensor) and tensor not in inputs:
                    inputs.append(tensor)

        try:
            sub = self._extractor.extract_model(inputs, outputs)
        except Exception as exc:
            raise SubgraphExtractionError(str(exc)) from exc

        used = {tensor for node in sub.graph.node for tensor in node.input}
        for vi in [vi for vi in sub.graph.input if vi.name not in used]:
            sub.graph.input.remove(vi)
        if requantize_inputs:
            self._requantize(sub.graph)
        if self._ir_lt4:
            names = {vi.name for vi in sub.graph.input}
            sub.graph.input.extend(
                onnx.helper.make_tensor_value_info(t.name, t.data_type, list(t.dims))
                for t in sub.graph.initializer
                if t.name not in names
            )

        out_path = Path(out_dir) / self.FILE_NAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if sub.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
                onnx.save(
                    sub,
                    str(out_path),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"{self.FILE_NAME}.data",
                )
            else:
                onnx.save(sub, str(out_path))
            onnx.checker.check_model(str(out_path))
        except Exception as exc:
            raise SubgraphExtractionError(f"Saving extracted model failed: {exc}") from exc
        initializers = {t.name for t in sub.graph.initializer}
        return SubgraphExtractionResult(
            out_path,
            [vi.name for vi in sub.graph.input if vi.name not in initializers],
            [vi.name for vi in sub.graph.output],
        )

    def _requantize(self, graph: onnx.GraphProto) -> None:
        """Inserts a QuantizeLinear -> DequantizeLinear pair behind each input that a DequantizeLinear produces.

        The pair has the parameters and attributes of the input's DequantizeLinear, and the nodes and initializers
        computing the parameters are copied from the model, so the subgraph's nodes consume a dequantized tensor as in
        the model. An input stays float if its DequantizeLinear has runtime parameters, its quantized data type is
        int32 or unknown, or it has no zero point, data other than uint8 and a scale that is not an initializer.

        Args:
            graph: Extracted subgraph, changed in place.
        """
        added: set[str] = set()
        inserted: list[tuple[int, onnx.NodeProto, onnx.NodeProto]] = []
        params: list[str] = []
        for name in [vi.name for vi in graph.input]:
            idx = self._producer.get(name)
            dq = self._nodes[idx] if idx is not None else None
            if dq is None or dq.op_type != "DequantizeLinear" or dq.domain not in _QDQ_DOMAINS or len(dq.input) < 2:
                continue
            scale = dq.input[1]
            zero_point = dq.input[2] if len(dq.input) > 2 else ""
            elem_type = self._elem_types.get(dq.input[0]) or self._elem_types.get(zero_point, 0)
            reason = ""
            if not all(param in self._constants for param in (scale, zero_point) if param):
                reason = f"the parameters of '{dq.name}' are computed at runtime"
            elif elem_type == onnx.TensorProto.UNDEFINED:
                reason = f"the data type of '{dq.input[0]}' is unknown"
            elif elem_type == onnx.TensorProto.INT32:
                reason = "QuantizeLinear cannot produce int32"
            elif not zero_point and elem_type != onnx.TensorProto.UINT8 and scale not in self._inits:
                reason = f"'{dq.name}' has no zero point and its scale is not an initializer"
            if reason:
                self._log.debug(f"Subgraph input '{name}' stays float: {reason}.")
                continue
            if not zero_point and elem_type != onnx.TensorProto.UINT8:
                # Without a zero point QuantizeLinear produces uint8; a zero zero point keeps the data type.
                dims = list(self._inits[scale].dims)
                zero_point = self._unique_name(f"{name}_requant_zero_point", added)
                graph.initializer.append(onnx.helper.make_tensor(zero_point, elem_type, dims, [0] * math.prod(dims)))
            q_params = [scale, zero_point] if zero_point else [scale]
            q_out = self._unique_name(f"{name}_requant_q", added)
            dq_out = self._unique_name(f"{name}_requant_dq", added)
            quantize = onnx.helper.make_node(
                "QuantizeLinear",
                [name, *q_params],
                [q_out],
                name=self._unique_name(f"{name}_requant_QuantizeLinear", added),
                domain=dq.domain,
            )
            quantize.attribute.extend(attr for attr in dq.attribute if attr.name in ("axis", "block_size"))
            dequantize = onnx.helper.make_node(
                "DequantizeLinear",
                [q_out, *q_params],
                [dq_out],
                name=self._unique_name(f"{name}_requant_DequantizeLinear", added),
                domain=dq.domain,
            )
            dequantize.attribute.extend(dq.attribute)
            for node in graph.node:
                for i, tensor in enumerate(node.input):
                    if tensor == name:
                        node.input[i] = dq_out
            inserted.append((idx, quantize, dequantize))
            params += q_params
            self._log.debug(f"Subgraph input '{name}' is quantized again with the parameters of '{dq.name}'.")
        if not inserted:
            return

        produced = {out for node in graph.node for out in node.output}
        available = {t.name for t in graph.initializer} | {vi.name for vi in graph.input}
        cone: set[int] = set()
        while params:
            tensor = params.pop()
            if not tensor or tensor in produced or tensor in available:
                continue
            if tensor in self._inits:
                graph.initializer.append(self._inits[tensor])
                available.add(tensor)
                continue
            producer = self._producer[tensor]
            cone.add(producer)
            produced.update(self._nodes[producer].output)
            params.extend(self._nodes[producer].input)

        # Sort by model position; each pair takes the place of its DequantizeLinear, which precedes the window.
        keyed = [((self._producer[next(out for out in node.output if out)], 0), node) for node in graph.node]
        keyed += [((producer, 0), self._nodes[producer]) for producer in cone]
        for idx, quantize, dequantize in inserted:
            keyed += [((idx, 1), quantize), ((idx, 2), dequantize)]
        nodes = [node for _, node in sorted(keyed, key=lambda item: item[0])]
        del graph.node[:]
        graph.node.extend(nodes)

    def _unique_name(self, name: str, added: set[str]) -> str:
        """Returns ``name``, or ``name_<n>`` if the model or ``added`` has it, and adds the result to ``added``."""
        unique, suffix = name, 0
        while unique in self._names or unique in added:
            suffix += 1
            unique = f"{name}_{suffix}"
        added.add(unique)
        return unique
