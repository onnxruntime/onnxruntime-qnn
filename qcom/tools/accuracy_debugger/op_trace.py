# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Framework op trace lookup, preserving fusion groups and tensor identities.

Trace op names and tensor names occupy separate namespaces. Q/DQ edges are
followed in the ONNX graph; neither node order nor quantizer suffixes are used.
An ambiguous trace match must never select the first reference tensor.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from common import sanitize_tensor_name

TRACE_COLUMNS = (
    "QNN_Op_Name",
    "QNN_Op_Type",
    "Fusion_Pattern",
    "Original_ONNX_Nodes",
    "CPU_Fallback_Reason",
)

# node_group_type of a plain ORT node unit; any other value names a QNN EP fusion.
_NODE_UNIT = "OrtNodeUnit"
# Trace name suffix of a DQ that ORT duplicated so that each node unit has its own DQ.
_DUPLICATED = "/duplicated"


@dataclass(frozen=True)
class FusionGroup:
    """ONNX nodes that QNN EP compiled as one fused node group.

    Attributes:
        nodes: Member node names in graph order. A DQ fed from outside the group is not a member: ORT gives the
            group its own copy of a DQ whose output is exposed, so exposing it does not change the fusion.
        patterns: Fusion names from the trace's ``node_group_type``, e.g. ``Rank6ToRank5Fusion``.
        qnn_ops: QNN ops the group compiled to.
        internal_tensors: Tensors that only the group produces and consumes. Exposing one as a graph output stops
            QNN EP from fusing the group, and the whole group then runs on CPU.
    """

    nodes: tuple[str, ...]
    patterns: tuple[str, ...]
    qnn_ops: tuple[str, ...]
    internal_tensors: frozenset[str]


class _Graph:
    def __init__(self, model: onnx.ModelProto | None) -> None:
        nodes = list(model.graph.node) if model is not None else []
        self.order = {node.name: index for index, node in enumerate(nodes) if node.name}
        self.nodes = {node.name: node for node in nodes if node.name}
        self.producers = {output: node for node in nodes for output in node.output if output}
        self.consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
        for node in nodes:
            for name in dict.fromkeys(node.input):
                if name:
                    self.consumers[name].append(node)
        self.outputs = {output.name for output in model.graph.output} if model is not None else set()

    def lineage(self, tensor: str) -> list[str]:
        """Return the tensor and its upstream Q/DQ data tensors, nearest first."""
        result = []
        while tensor and tensor not in result:
            result.append(tensor)
            node = self.producers.get(tensor)
            if node is None or node.op_type not in {"QuantizeLinear", "DequantizeLinear"} or not node.input:
                break
            tensor = node.input[0]
        return result

    def boundary_outputs(self, names: Iterable[str]) -> set[str]:
        nodes = [self.nodes[name] for name in names if name in self.nodes]
        inputs = {name for node in nodes for name in node.input}
        return {name for node in nodes for name in node.output if name and name not in inputs}


@dataclass(frozen=True)
class _Mapping:
    graph: str
    destination: str
    sources: tuple[tuple[str, str], ...]
    originals: tuple[tuple[str, str], ...]
    op_type: str = ""
    fusion: str = ""

    @property
    def preferred(self) -> tuple[tuple[str, str], ...]:
        return self.originals or self.sources


def _records(value: Any, field: str) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be an array of objects")
    return value


def _sources(value: Any) -> tuple[tuple[str, str], ...]:
    result = []
    for source in _records(value, "sources"):
        name, kind = source.get("name"), source.get("type")
        if not isinstance(name, str) or not name or kind not in {"OP", "TENSOR"}:
            raise ValueError("Trace sources require a name and type OP or TENSOR")
        result.append((name, kind))
    return tuple(result)


class OpTraceIndex:
    """Index a schema v2 trace, including the matcher's original_sources extension."""

    def __init__(
        self,
        document: dict,
        target_model: onnx.ModelProto | None = None,
        reference_model: onnx.ModelProto | None = None,
    ) -> None:
        if not isinstance(document, dict) or type(document.get("schema_version")) is not int:
            raise ValueError("Framework op trace requires an integer schema_version")
        if document["schema_version"] != 2:
            raise ValueError(f"Unsupported framework op trace schema_version: {document['schema_version']}")
        self.target = _Graph(target_model)
        self.reference = _Graph(reference_model)
        self.ops: dict[str, list[_Mapping]] = defaultdict(list)
        self.tensors: dict[str, list[_Mapping]] = defaultdict(list)
        self._groups: dict[tuple[str, tuple], list[_Mapping]] = defaultdict(list)
        self.unsupported: dict[str, list[str]] = defaultdict(list)
        # SOCs share this flat topology. Keep all entries, including duplicate
        # destination names in different subgraphs, instead of overwriting them.
        for graph in _records(document.get("subgraph_traces"), "subgraph_traces"):
            for field, index, kind in (("op_mappings", self.ops, "OP"), ("tensor_mappings", self.tensors, "TENSOR")):
                for item in _records(graph.get(field, []), field):
                    destination = item.get("dst_name")
                    if not isinstance(destination, str) or not destination:
                        raise ValueError("Trace mappings require a non-empty dst_name")
                    mapping = _Mapping(
                        str(graph.get("graph_name", "")),
                        destination,
                        _sources(item.get("sources", [])),
                        _sources(item.get("original_sources", [])),
                        str(item.get("dst_qnn_op_type", "")),
                        str(item.get("node_group_type", "")),
                    )
                    names = {destination}
                    for sources in (mapping.sources, mapping.originals):
                        # Some EP builds put the entire op group's provenance in
                        # tensor_mappings, including inputs and internal tensors.
                        # Those names are not aliases of the destination tensor.
                        if kind == "OP" or all(typ == "TENSOR" for _, typ in sources):
                            names.update(name for name, typ in sources if typ == kind)
                    for name in names:
                        index[name].append(mapping)
                    if kind == "OP":
                        self._groups[(mapping.graph, mapping.sources)].append(mapping)

        for node in _records(document.get("unsupported_nodes", []), "unsupported_nodes"):
            name, reason = node.get("node_name"), node.get("reason")
            if not isinstance(name, str) or not isinstance(reason, str):
                raise ValueError("Unsupported nodes require node_name and reason strings")
            self.unsupported[name].append(reason)
        # Rejection reasons keyed by produced tensor; filled from EP input graph dumps.
        self.unsupported_tensors: dict[str, list[str]] = defaultdict(list)

    @classmethod
    def load(
        cls,
        path: Path,
        target_model: onnx.ModelProto | None = None,
        reference_model: onnx.ModelProto | None = None,
    ) -> OpTraceIndex:
        path = Path(path)
        index = cls(json.loads(path.read_text(encoding="utf-8-sig")), target_model, reference_model)
        for dump in sorted(path.parent.glob("*_qnn_ep_input_graph.json")):
            index.add_ep_input_graph(json.loads(dump.read_text(encoding="utf-8-sig")))
        return index

    def add_ep_input_graph(self, document: Any) -> None:
        """Attribute rejection reasons of EP-renamed nodes (e.g. ``DequantizeLinear_token_N``) to their outputs."""
        nodes = document.get("graph", {}).get("nodes", {}) if isinstance(document, dict) else {}
        if not isinstance(nodes, dict):
            return
        for name, node in nodes.items():
            reasons = self.unsupported.get(name)
            outputs = node.get("output_names", []) if isinstance(node, dict) else []
            if reasons and isinstance(outputs, list):
                for output in outputs:
                    if isinstance(output, str):
                        self.unsupported_tensors[output].extend(
                            r for r in reasons if r not in self.unsupported_tensors[output]
                        )

    def fusion_groups(self) -> list[FusionGroup]:
        """Return the target model's fused node groups in graph order; groups sharing a node are merged.

        Op names are matched to target nodes. A duplicated DQ (``<name>/duplicated``) counts as its ONNX node, and
        EP-inserted ops are ignored. Groups with fewer than two member nodes cannot hide tensors and are dropped.
        """
        graph = self.target
        merged: list[tuple[set[str], list[_Mapping]]] = []
        for mappings in self._groups.values():
            if all(mapping.fusion in ("", _NODE_UNIT) for mapping in mappings):
                continue
            names = set()
            for mapping in mappings:
                for source, kind in mapping.preferred:
                    name = source
                    while name.endswith(_DUPLICATED):
                        name = name[: -len(_DUPLICATED)]
                    if kind == "OP" and name in graph.nodes:
                        names.add(name)
            members = set()
            for name in names:
                node = graph.nodes[name]
                producer = graph.producers.get(node.input[0]) if node.input else None
                if node.op_type != "DequantizeLinear" or (producer is not None and producer.name in names):
                    members.add(name)
            combined = mappings
            for entry in [entry for entry in merged if entry[0] & members]:
                merged.remove(entry)
                members |= entry[0]
                combined = entry[1] + combined
            merged.append((members, combined))

        groups = []
        for members, mappings in merged:
            if len(members) < 2:
                continue
            internal = frozenset(
                tensor
                for name in members
                for tensor in graph.nodes[name].output
                if tensor
                and tensor not in graph.outputs
                and graph.consumers.get(tensor)
                and all(consumer.name in members for consumer in graph.consumers[tensor])
            )
            groups.append(
                FusionGroup(
                    tuple(sorted(members, key=graph.order.__getitem__)),
                    tuple(dict.fromkeys(m.fusion for m in mappings if m.fusion not in ("", _NODE_UNIT))),
                    tuple(dict.fromkeys(mapping.destination for mapping in mappings)),
                    internal,
                )
            )
        return sorted(groups, key=lambda group: graph.order[group.nodes[0]])

    def _known(self, mapping: _Mapping) -> bool:
        """Whether an OP source of this group exists in either ONNX graph (always True without graphs)."""
        graphs = [graph for graph in (self.target, self.reference) if graph.nodes]
        return not graphs or any(
            name in graph.nodes for name, kind in mapping.preferred if kind == "OP" for graph in graphs
        )

    def _op_mappings(self, tensor: str) -> list[_Mapping]:
        direct = []
        for mapping in self.tensors.get(tensor, []):
            direct.extend(self._groups.get((mapping.graph, mapping.sources), []))
        # Skip groups made only of EP-inserted ops (e.g. layout Transposes); use graph edges instead.
        direct = [mapping for mapping in direct if self._known(mapping)]
        if direct:
            return list(dict.fromkeys(direct))
        tensors = set(self.target.lineage(tensor))
        for mapping in self.tensors.get(tensor, []):
            for name, kind in mapping.sources + mapping.originals:
                if kind == "TENSOR":
                    tensors.update(self.target.lineage(name))
        mappings = []
        quantize_mappings = []
        dequantize_mappings = []
        for name in sorted(tensors):
            for graph in (self.target, self.reference):
                node = graph.producers.get(name)
                if node is not None:
                    destination = mappings
                    if node.op_type == "QuantizeLinear":
                        destination = quantize_mappings
                    elif node.op_type == "DequantizeLinear":
                        destination = dequantize_mappings
                    destination.extend(self.ops.get(node.name, []))
        # A post-op DQ may belong to the following QNN op as an input. Prefer
        # the producer's output Q when the optimizer renamed the compute op.
        return list(dict.fromkeys(mappings or quantize_mappings or dequantize_mappings))

    def resolve(self, tensor: str, reference_names: set[str], reference_is_qdq: bool) -> tuple[str | None, bool]:
        """Return (reference name, ambiguous); a miss may use the legacy heuristic.

        Only tensor mappings supply aliases. Op groups supply graph outputs, never
        op names as tensor names. Multiple boundary outputs are deliberately unresolved.
        """

        def available(names: Iterable[str]) -> set[str]:
            result = set()
            for name in names:
                if name in reference_names:
                    result.add(name)
                elif sanitize_tensor_name(name) in reference_names:
                    result.add(sanitize_tensor_name(name))
            return result

        def answer(names: set[str]) -> tuple[str | None, bool]:
            return (next(iter(names)), False) if len(names) == 1 else (None, len(names) > 1)

        # An explicit tensor mapping has the strongest evidence. Prefer merged
        # names, but allow raw sources if originals are unavailable in this reference.
        for source_attribute in ("preferred", "sources"):
            candidates = set()
            for mapping in self.tensors.get(tensor, []):
                sources = getattr(mapping, source_attribute)
                if any(kind == "OP" for _, kind in sources):
                    continue
                for name, kind in sources:
                    if kind == "TENSOR":
                        aliases = [name] if reference_is_qdq else self.target.lineage(name)
                        candidates.update(available(aliases))
            if candidates:
                return answer(candidates)

        # Q and DQ values are numerically different. Preserve their exact identity
        # when comparing two QDQ executions, even if they share a fused QNN op.
        if reference_is_qdq and tensor in reference_names:
            return tensor, False

        mappings = self._op_mappings(tensor)
        if not mappings:
            return None, False

        # For an unchanged producer, retain its output slot (e.g. Split output 1).
        origin = self.target.lineage(tensor)[-1]
        producer = self.target.producers.get(origin)
        if producer is not None and not reference_is_qdq:
            candidates = available([origin])
            if candidates:
                return answer(candidates)
            reference_node = self.reference.nodes.get(producer.name)
            if reference_node is not None and producer.op_type == reference_node.op_type:
                slot = list(producer.output).index(origin)
                if slot < len(reference_node.output):
                    candidates = available([reference_node.output[slot]])
                    if candidates:
                        return answer(candidates)

        candidates = set()
        for mapping in mappings:
            source_nodes = [name for name, kind in mapping.preferred if kind == "OP"]
            outputs = self.reference.boundary_outputs(source_nodes)
            # A partial original graph cannot establish the boundary of a fusion.
            if source_nodes and all(name in self.reference.nodes for name in source_nodes):
                if len(outputs) > 1:
                    return None, True
                candidates.update(available(outputs))
        return answer(candidates)

    def metadata(self, tensor: str) -> dict[str, str]:
        """Return the TRACE_COLUMNS values: contributing QNN ops/fusions and CPU rejection reasons."""
        mappings = self._op_mappings(tensor)

        def joined(values: Iterable[str]) -> str:
            return ";".join(dict.fromkeys(value for value in values if value))

        nodes = joined(name for mapping in mappings for name, kind in mapping.preferred if kind == "OP")
        producer = self.target.producers.get(tensor)
        reasons = self.unsupported.get(producer.name, []) if producer is not None else []
        if not reasons:
            reasons = self.unsupported_tensors.get(tensor, [])
        if not reasons:
            origin = self.target.lineage(tensor)[-1]
            producer = self.target.producers.get(origin)
            reasons = self.unsupported.get(producer.name, []) if producer is not None else []
        return dict(
            zip(
                TRACE_COLUMNS,
                (
                    joined(mapping.destination for mapping in mappings),
                    joined(mapping.op_type for mapping in mappings),
                    joined(mapping.fusion for mapping in mappings),
                    nodes,
                    joined(reasons),
                ),
                strict=True,
            )
        )
