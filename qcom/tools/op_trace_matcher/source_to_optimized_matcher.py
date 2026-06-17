# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Source ONNX -> Optimized ONNX node and tensor mapping tool.

Computes a structural correspondence between a user-supplied ONNX model
(`source.onnx`) and an ORT-optimized graph. Output JSON aligns with
the QNN EP `FrameworkOpTrace` schema's `original_sources` extension;
combined with the QNN EP's existing `qnn_op_trace.json`, this yields
end-to-end provenance: original ONNX node -> optimized ONNX node -> QNN op.

See ./README.md for the feasibility assessment, known limitations, and the
output schema.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# `onnx` is needed only to load the models at run time. Import lazily so that
# `--help` and import-as-a-library (Mode B of enrich_profiling_csv.py guards its
# own use) work without the package installed; the hard error is raised in
# _require_onnx() at the point of actual use. `from __future__ import annotations`
# keeps the ModelProto type hints as strings, so they need no import here.
try:
    import onnx
    from onnx import ModelProto  # used in string annotations (ModelProto type hints)
    from onnx import numpy_helper as _onnx_numpy_helper
except ImportError:
    onnx = None
    _onnx_numpy_helper = None


def _require_onnx() -> None:
    if onnx is None:
        sys.stderr.write("error: this tool requires the `onnx` package (`pip install onnx`)\n")
        sys.exit(2)


# Public API for import-as-a-library use (e.g. enrich_profiling_csv.py Mode B).
__all__ = [
    "Confidence",
    "Match",
    "MatchMethod",
    "Matcher",
    "MatcherStats",
    "TensorMatch",
    "build_output",
    "join_qnn_trace",
]


# QNN-Netron tensor-type integer for an initializer (weights/constants). Mirrors
# `Qnn_TensorType_t::QNN_TENSOR_TYPE_STATIC` used by the QNN EP's
# dump_qnn_ep_input_graph output (see qnn_ep_input_graph_dumper.cc).
_QNN_TENSOR_TYPE_STATIC = 4


# Maximum BFS hop count for the lineage-extension walks (producer-preferred
# backward walk and consumer fallback forward walk). For any DAG the per-walk
# `seen` set already bounds total work to O(|tensors|); this ceiling is a
# defensive safety net for pathological or cyclic inputs, deliberately set
# well above any practical model depth so it never trips on real graphs.
_LINEAGE_WALK_MAX_HOPS = 1024


# ---------------------------------------------------------------------------
# Match result types
# ---------------------------------------------------------------------------


class MatchMethod(str, Enum):
    EXACT_NAME = "exact_name"
    EXACT_IO = "exact_io"
    INITIALIZER_NAME = "initializer_name"
    INITIALIZER_HASH = "initializer_hash"
    FUSION_PATTERN = "fusion_pattern"
    NODE_OUTPUT_POSITIONAL = "node_output_positional"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"


@dataclass
class Match:
    optimized_node: str
    optimized_op_type: str
    source_nodes: list[str]
    source_op_types: list[str]
    method: MatchMethod
    confidence: Confidence
    pattern: str | None = None
    notes: str | None = None


@dataclass
class TensorMatch:
    optimized_tensor: str
    original_tensors: list[str]
    method: MatchMethod
    confidence: Confidence
    notes: str | None = None


# ---------------------------------------------------------------------------
# Graph index
# ---------------------------------------------------------------------------


@dataclass
class NodeRecord:
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    initializer_inputs: tuple[str, ...] = ()


@dataclass
class GraphIndex:
    nodes: dict[str, NodeRecord]
    by_output_tensor: dict[str, str]
    by_input_tensor: dict[str, list[str]]
    initializers: dict[str, bytes]
    by_initializer_consumer: dict[str, list[str]]

    @classmethod
    def build(cls, model: ModelProto) -> GraphIndex:
        graph = model.graph

        initializers: dict[str, bytes] = {}
        for init in graph.initializer:
            # raw_data is the common case for large weights: hash it directly,
            # decoupled from any protobuf encoding details.
            #
            # For non-raw_data initializers (small constants stored in
            # field-typed lists like int32_data/float_data), hash the canonical
            # numpy bytes via onnx.numpy_helper.to_array(...).tobytes() rather
            # than init.SerializeToString(). The proto wire format is not
            # guaranteed deterministic across re-serializations, but the
            # underlying numpy buffer is — this keeps the hash stable when the
            # source and optimized graphs were each round-tripped through
            # different ONNX load/save passes.
            if init.raw_data:
                payload = init.raw_data
            elif _onnx_numpy_helper is not None:
                try:
                    payload = _onnx_numpy_helper.to_array(init).tobytes()
                except (ValueError, TypeError, RuntimeError):
                    # numpy_helper raises ValueError for malformed tensor
                    # field counts, TypeError for unsupported dtypes (BF16
                    # variants on older onnx versions), and RuntimeError when
                    # external_data is referenced but not yet resolved. Fall
                    # back to proto serialization in those cases. The hash is
                    # still byte-stable within a single protobuf process;
                    # cross-process stability is the only property weakened.
                    payload = init.SerializeToString()
            else:
                payload = init.SerializeToString()
            initializers[init.name] = hashlib.sha256(payload).digest()

        nodes: dict[str, NodeRecord] = {}
        by_output_tensor: dict[str, str] = {}
        by_input_tensor: dict[str, list[str]] = defaultdict(list)
        by_initializer_consumer: dict[str, list[str]] = defaultdict(list)

        seen_names: set[str] = set()
        for idx, node in enumerate(graph.node):
            name = node.name or f"unnamed_{node.op_type}_{idx}"
            if name in seen_names:
                name = f"{name}__dup{idx}"
            seen_names.add(name)

            init_inputs = tuple(i for i in node.input if i in initializers)
            record = NodeRecord(
                name=name,
                op_type=node.op_type,
                inputs=tuple(node.input),
                outputs=tuple(node.output),
                initializer_inputs=init_inputs,
            )
            nodes[name] = record
            for out in node.output:
                if out:
                    by_output_tensor[out] = name
            for inp in node.input:
                if inp:
                    by_input_tensor[inp].append(name)
            for init_name in init_inputs:
                by_initializer_consumer[init_name].append(name)

        return cls(
            nodes=nodes,
            by_output_tensor=by_output_tensor,
            by_input_tensor=dict(by_input_tensor),
            initializers=dict(initializers),
            by_initializer_consumer=dict(by_initializer_consumer),
        )

    @classmethod
    def build_from_qnn_json(cls, doc: dict) -> GraphIndex:
        """Build a GraphIndex from the QNN-Netron-schema JSON emitted by the
        QNN EP's `dump_qnn_ep_input_graph` option (see qnn_ep_input_graph_dumper.cc).

        That JSON is the ONNX graph the EP received at compile time. It carries
        node name/op_type/inputs/outputs and a tensor table that flags each
        tensor's role via an integer `type` (see _QNN_TENSOR_TYPE_STATIC for the
        initializer value).
        It does NOT carry initializer data bytes, so each initializer is given a
        name-derived sentinel hash: name-based and topology-based matching work
        unchanged, while data-hash matching (for renamed weights) is inert for
        this input — acceptable because the QNN-EP-direct workflow keeps weight
        names stable.
        """
        graph = doc.get("graph", {})
        json_nodes = graph.get("nodes", {}) or {}
        json_tensors = graph.get("tensors", {}) or {}

        # Initializers: tensors flagged STATIC. Sentinel hash per name.
        initializers: dict[str, bytes] = {}
        for tname, tinfo in json_tensors.items():
            if isinstance(tinfo, dict) and tinfo.get("type") == _QNN_TENSOR_TYPE_STATIC:
                initializers[tname] = hashlib.sha256(tname.encode("utf-8")).digest()

        nodes: dict[str, NodeRecord] = {}
        by_output_tensor: dict[str, str] = {}
        by_input_tensor: dict[str, list[str]] = defaultdict(list)
        by_initializer_consumer: dict[str, list[str]] = defaultdict(list)

        seen_names: set[str] = set()
        for idx, (node_name, raw_ninfo) in enumerate(json_nodes.items()):
            ninfo = raw_ninfo or {}
            name = node_name or f"unnamed_{ninfo.get('type', 'op')}_{idx}"
            if name in seen_names:
                name = f"{name}__dup{idx}"
            seen_names.add(name)

            inputs = tuple(ninfo.get("input_names", []) or [])
            outputs = tuple(ninfo.get("output_names", []) or [])
            init_inputs = tuple(i for i in inputs if i in initializers)
            nodes[name] = NodeRecord(
                name=name,
                op_type=ninfo.get("type", ""),
                inputs=inputs,
                outputs=outputs,
                initializer_inputs=init_inputs,
            )
            for out in outputs:
                if out:
                    by_output_tensor[out] = name
            for inp in inputs:
                if inp:
                    by_input_tensor[inp].append(name)
            for init_name in init_inputs:
                by_initializer_consumer[init_name].append(name)

        return cls(
            nodes=nodes,
            by_output_tensor=by_output_tensor,
            by_input_tensor=dict(by_input_tensor),
            initializers=dict(initializers),
            by_initializer_consumer=dict(by_initializer_consumer),
        )


# ---------------------------------------------------------------------------
# Fusion patterns
# ---------------------------------------------------------------------------

# Each pattern: optimized op_type -> list of candidate source-op-type sequences.
# Sequences are listed in source-graph topological order (input -> output),
# which matches how the matcher walks back from the optimized node's input.
#
# Only LINEAR source chains can be expressed here — the walker follows one
# input edge per step. Fusions with branching source patterns (LayerNorm,
# EmbedLayerNorm, Attention, GroupQueryAttention, the Tanh-form FastGelu, …)
# typically rely on the matcher's initializer-anchoring or lineage-extension
# strategies instead. Patterns derived by reading the corresponding fusion
# files in onnxruntime/core/optimizer/.
FUSION_PATTERNS: dict[str, list[list[str]]] = {
    # MatMul + Add  ->  Gemm                 (matmul_add_fusion)
    "Gemm": [["MatMul", "Add"]],
    # Div + Erf + Add + Mul + Mul  ->  Gelu  (gelu_fusion, Erf form)
    "Gelu": [["Div", "Erf", "Add", "Mul", "Mul"]],
    # Tanh-form approximation  ->  FastGelu  (fast_gelu_fusion)
    # The full source pattern branches; this is the linear chain through
    # the inner Mul·Tanh·Add expression. Best-effort.
    "FastGelu": [["Mul", "Tanh", "Add", "Mul", "Mul", "Add", "Mul"]],
    # Sigmoid + Mul  ->  QuickGelu           (quick_gelu_fusion)
    "QuickGelu": [["Sigmoid", "Mul"]],
    # Add + (Gelu | FastGelu)  ->  BiasGelu  (bias_gelu_fusion)
    "BiasGelu": [
        ["Add", "Gelu"],
        ["Add", "FastGelu"],
    ],
    # Add + Softmax  ->  BiasSoftmax         (bias_softmax_fusion)
    "BiasSoftmax": [["Add", "Softmax"]],
    # Add + LayerNormalization  ->  SkipLayerNormalization
    #                                       (skip_layer_norm_fusion)
    "SkipLayerNormalization": [["Add", "LayerNormalization"]],
    # Transpose + MatMul  ->  FusedMatMul    (matmul_transpose_fusion)
    # Only one MatMul input is transposed; whether the linear walk finds
    # Transpose depends on input ordering. Fallback strategies usually
    # cover the case when this pattern misses.
    "FusedMatMul": [["Transpose", "MatMul"]],
}


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


@dataclass
class MatcherStats:
    total_source: int = 0
    total_optimized: int = 0
    matched: int = 0
    unmatched: int = 0
    by_method: dict[str, int] = field(default_factory=dict)
    # Tensor-level stats
    total_optimized_tensors: int = 0
    matched_tensors: int = 0
    by_tensor_method: dict[str, int] = field(default_factory=dict)


class Matcher:
    def __init__(self, source, optimized):
        """`source` and `optimized` may each be either an onnx ModelProto (built
        via GraphIndex.build) or an already-constructed GraphIndex (e.g. from
        GraphIndex.build_from_qnn_json). This lets the optimized side come from
        the QNN EP's dump_qnn_ep_input_graph JSON instead of a saved .onnx."""
        self.src = source if isinstance(source, GraphIndex) else GraphIndex.build(source)
        self.opt = optimized if isinstance(optimized, GraphIndex) else GraphIndex.build(optimized)
        self.matches: dict[str, Match] = {}

    def run(self) -> tuple[list[Match], list[TensorMatch], MatcherStats, list[str], list[str]]:
        # Strategies are applied in confidence order. Each strategy only
        # operates on optimized nodes that have not yet been matched.
        self._exact_name_match()
        self._exact_io_match()
        self._initializer_name_match()
        self._initializer_hash_match()
        self._fusion_pattern_match()
        self._lineage_extension()

        # Tensor-level matching runs after node-level so it can derive
        # output-position mappings from the node matches.
        tensor_matches = self._match_tensors()

        unmatched_opt = [n for n in self.opt.nodes if n not in self.matches]

        claimed_src = self._collect_claimed_src()
        removed_src = [n for n in self.src.nodes if n not in claimed_src]

        stats = MatcherStats(
            total_source=len(self.src.nodes),
            total_optimized=len(self.opt.nodes),
            matched=len(self.matches),
            unmatched=len(unmatched_opt),
            total_optimized_tensors=len(self._all_tensor_names(self.opt)),
            matched_tensors=len(tensor_matches),
        )
        for m in self.matches.values():
            stats.by_method[m.method.value] = stats.by_method.get(m.method.value, 0) + 1
        for tm in tensor_matches:
            stats.by_tensor_method[tm.method.value] = stats.by_tensor_method.get(tm.method.value, 0) + 1

        return list(self.matches.values()), tensor_matches, stats, unmatched_opt, removed_src

    def _collect_claimed_src(self) -> set[str]:
        """Set of source node names already attributed by an earlier strategy.
        Used to keep each source node attributed to at most one optimized node.
        Strategies that build a multi-source chain (initializer anchoring,
        fusion patterns) read this and either filter or skip accordingly."""
        claimed: set[str] = set()
        for m in self.matches.values():
            claimed.update(m.source_nodes)
        return claimed

    # -- Strategy 1: exact name + op_type --------------------------------

    def _exact_name_match(self) -> None:
        for name, opt in self.opt.nodes.items():
            if name in self.matches:
                continue
            src = self.src.nodes.get(name)
            if src is None or src.op_type != opt.op_type:
                continue
            self._record(
                opt,
                source_nodes=[name],
                method=MatchMethod.EXACT_NAME,
                confidence=Confidence.HIGH,
            )

    # -- Strategy 2: exact I/O signature ---------------------------------

    def _exact_io_match(self) -> None:
        # Index source by (op_type, inputs, outputs) signature.
        sig_to_src: dict[tuple, list[str]] = defaultdict(list)
        for name, n in self.src.nodes.items():
            sig_to_src[(n.op_type, n.inputs, n.outputs)].append(name)

        already_claimed_singletons: set[str] = {
            m.source_nodes[0] for m in self.matches.values() if len(m.source_nodes) == 1
        }

        for name, opt in self.opt.nodes.items():
            if name in self.matches:
                continue
            sig = (opt.op_type, opt.inputs, opt.outputs)
            candidates = [c for c in sig_to_src.get(sig, []) if c not in already_claimed_singletons]
            if len(candidates) == 1:
                src = candidates[0]
                already_claimed_singletons.add(src)
                self._record(
                    opt,
                    source_nodes=[src],
                    method=MatchMethod.EXACT_IO,
                    confidence=Confidence.HIGH,
                )

    # -- Strategy 3: initializer name anchoring --------------------------

    def _initializer_name_match(self) -> None:
        # Note: this strategy intentionally does NOT filter against
        # _collect_claimed_src(). In QDQ models, a single source Q/DQ pair on a
        # quantized boundary tensor legitimately participates in BOTH the
        # upstream and downstream fused QLinear* ops' source sets — the same
        # scale/zero_point initializer is shared across the boundary. Skipping
        # already-claimed source nodes here would strip exactly those legitimate
        # multi-attributions and produce incomplete provenance for QDQ patterns.
        for name, opt in self.opt.nodes.items():
            if name in self.matches:
                continue
            if not opt.initializer_inputs:
                continue
            candidates: set[str] = set()
            for init_name in opt.initializer_inputs:
                for src_consumer in self.src.by_initializer_consumer.get(init_name, []):
                    candidates.add(src_consumer)
            if not candidates:
                continue
            src_list = self._topo_sort_in_src(list(candidates))
            confidence = Confidence.HIGH if len(src_list) == 1 else Confidence.MEDIUM
            notes = None if len(src_list) == 1 else "Multiple source nodes share initializers — likely fusion"
            self._record(
                opt,
                source_nodes=src_list,
                method=MatchMethod.INITIALIZER_NAME,
                confidence=confidence,
                notes=notes,
            )

    # -- Strategy 4: initializer data hash -------------------------------

    def _initializer_hash_match(self) -> None:
        # hash -> set of source node names that consume an initializer with that hash
        src_hash_to_consumers: dict[bytes, set[str]] = defaultdict(set)
        for init_name, h in self.src.initializers.items():
            for consumer in self.src.by_initializer_consumer.get(init_name, []):
                src_hash_to_consumers[h].add(consumer)

        # See _initializer_name_match: deliberately no claimed-source filter,
        # because QDQ-boundary Q/DQ pairs are legitimately shared across
        # adjacent fused QLinear ops.
        for name, opt in self.opt.nodes.items():
            if name in self.matches:
                continue
            if not opt.initializer_inputs:
                continue
            candidates: set[str] = set()
            for init_name in opt.initializer_inputs:
                h = self.opt.initializers.get(init_name)
                if h is None:
                    continue
                candidates.update(src_hash_to_consumers.get(h, set()))
            if not candidates:
                continue
            src_list = self._topo_sort_in_src(list(candidates))
            confidence = Confidence.HIGH if len(src_list) == 1 else Confidence.MEDIUM
            notes = (
                None
                if len(src_list) == 1
                else "Renamed initializer matched by data hash; multiple consumers suggest fusion"
            )
            self._record(
                opt,
                source_nodes=src_list,
                method=MatchMethod.INITIALIZER_HASH,
                confidence=confidence,
                notes=notes,
            )

    # -- Strategy 5: fusion pattern walk ---------------------------------

    def _fusion_pattern_match(self) -> None:
        # A fusion chain is a structural pattern (e.g. MatMul + Add -> Gemm)
        # that requires every node in the chain. If any chain node is already
        # attributed to a different optimized node, the fusion is incomplete
        # for this candidate, so skip — unlike initializer anchoring, we
        # cannot meaningfully drop the claimed node and keep the rest.
        claimed_src = self._collect_claimed_src()

        for name, opt in self.opt.nodes.items():
            if name in self.matches:
                continue
            patterns = FUSION_PATTERNS.get(opt.op_type)
            if not patterns:
                continue
            for pattern in patterns:
                chain = self._walk_back_for_pattern(opt, pattern)
                if chain and not any(s in claimed_src for s in chain):
                    self._record(
                        opt,
                        source_nodes=chain,
                        method=MatchMethod.FUSION_PATTERN,
                        confidence=Confidence.MEDIUM,
                        pattern="+".join(pattern),
                    )
                    claimed_src.update(chain)
                    break

    def _walk_back_for_pattern(self, opt: NodeRecord, pattern: list[str]) -> list[str] | None:
        """Try to find a chain of source nodes matching `pattern` whose final
        output tensor is one of `opt`'s inputs. Pattern is in source topological
        order (input side first); we walk backward starting from the last op.

        Limitation for non-commutative fusions (MatMul+Add -> Gemm,
        Conv+Add -> Conv-with-bias, …): when a node has more than one
        op-output input, this walker takes the FIRST `inputs[]` entry that
        has a producer in the source graph and stops. This relies on the
        assumption that the chain entry sits at the lower input index — true
        for all currently-supported fusions because their non-chain operand
        is an initializer (bias / weight) that never appears in
        `by_output_tensor`. Fusions whose chain entry sits at a higher input
        index, or whose siblings are both op outputs, will require an
        explicit per-pattern annotation (e.g. an `input_index` marker on
        `FUSION_PATTERNS`).
        """
        for inp in opt.inputs:
            producer_name = self.src.by_output_tensor.get(inp)
            if producer_name is None:
                continue
            chain: list[str] = []
            current: str | None = producer_name
            for op_type in reversed(pattern):
                if current is None:
                    break
                src = self.src.nodes.get(current)
                if src is None or src.op_type != op_type:
                    break
                chain.append(current)
                # Move to the producer of this node's first non-initializer input.
                current = None
                for src_inp in src.inputs:
                    if src_inp in self.src.by_output_tensor:
                        current = self.src.by_output_tensor[src_inp]
                        break
            if len(chain) == len(pattern):
                return list(reversed(chain))  # source topological order
        return None

    # -- Strategy 6: lineage extension -----------------------------------

    def _lineage_extension(self) -> None:
        """For source nodes not yet claimed, attribute them to a matched
        optimized node — preferring the *producer* (upstream) over the
        *consumer* (downstream).

        The producer-preferred rule reflects how ORT fusions actually work:
        Conv+BN -> Conv folds BN into Conv, so unmatched `bn1` belongs with
        `conv1` (its producer), not `relu1` (its consumer). MatMul+Add -> Gemm
        is similar — `add1` belongs with the node that absorbed `matmul1`.

        Iterates: a chain of unmatched nodes (e.g. BN -> Activation both folded)
        is propagated outward until no more progress is possible.
        """
        claimed: set[str] = set()
        src_to_opt: dict[str, str] = {}
        for opt_name, m in self.matches.items():
            for s in m.source_nodes:
                claimed.add(s)
                src_to_opt[s] = opt_name

        progress = True
        while progress:
            progress = False
            for src_name, src in self.src.nodes.items():
                if src_name in claimed:
                    continue
                # Producer first (upstream).
                target_opt = self._find_producer_match(src, src_to_opt)
                # Fall back to consumer (downstream).
                if target_opt is None:
                    target_opt = self._find_lineage_target(src)
                if target_opt is None:
                    continue
                opt_match = self.matches.get(target_opt)
                if opt_match is None or src_name in opt_match.source_nodes:
                    continue
                # Insert and re-sort topologically by source-graph order.
                merged = self._topo_sort_in_src([*opt_match.source_nodes, src_name])
                opt_match.source_nodes = merged
                opt_match.source_op_types = [self.src.nodes[n].op_type for n in merged]
                opt_match.confidence = Confidence.MEDIUM
                tag = f"lineage-extended: absorbed source node `{src_name}` ({src.op_type})"
                opt_match.notes = f"{opt_match.notes}; {tag}" if opt_match.notes else tag
                claimed.add(src_name)
                src_to_opt[src_name] = target_opt
                progress = True

    def _find_producer_match(self, src: NodeRecord, src_to_opt: dict[str, str]) -> str | None:
        """Walk *backward* in the source graph from `src.inputs` until we
        reach a node already claimed by an optimized match. Returns the
        optimized node's name."""
        frontier: list[str] = [t for t in src.inputs if t]
        seen: set[str] = set(frontier)
        for _hop in range(_LINEAGE_WALK_MAX_HOPS):
            if not frontier:
                return None
            new_frontier: list[str] = []
            for tensor in frontier:
                producer_in_src = self.src.by_output_tensor.get(tensor)
                if producer_in_src is None:
                    continue
                if producer_in_src in src_to_opt:
                    return src_to_opt[producer_in_src]
                producer_node = self.src.nodes.get(producer_in_src)
                if producer_node is None:
                    continue
                for inp in producer_node.inputs:
                    if inp and inp not in seen:
                        seen.add(inp)
                        new_frontier.append(inp)
            frontier = new_frontier
        return None

    def _topo_sort_in_src(self, node_names: list[str]) -> list[str]:
        """Sort source-graph node names in topological order (a node A comes
        before B if B transitively consumes A's output). Returns alphabetical
        as a fallback if a cycle or disconnected case is detected."""
        if len(node_names) <= 1:
            return list(node_names)
        name_set = set(node_names)
        in_degree: dict[str, int] = dict.fromkeys(node_names, 0)
        edges: dict[str, list[str]] = defaultdict(list)
        for n in node_names:
            node = self.src.nodes.get(n)
            if node is None:
                continue
            for inp in node.inputs:
                producer = self.src.by_output_tensor.get(inp)
                if producer in name_set and producer != n:
                    edges[producer].append(n)
                    in_degree[n] += 1
        queue: list[str] = sorted(n for n in node_names if in_degree[n] == 0)
        result: list[str] = []
        while queue:
            cur = queue.pop(0)
            result.append(cur)
            promoted = []
            for nxt in edges[cur]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    promoted.append(nxt)
            queue.extend(sorted(promoted))
        if len(result) != len(node_names):
            return sorted(node_names)
        return result

    def _find_lineage_target(self, src: NodeRecord) -> str | None:
        """Walk forward in the source graph from `src.outputs` until we reach a
        tensor name that is an output of some optimized node we already
        matched. Return that optimized node's name."""
        frontier: list[str] = [o for o in src.outputs if o]
        seen: set[str] = set(frontier)
        # Bound the walk to avoid pathological runtime on deep graphs.
        for _hop in range(_LINEAGE_WALK_MAX_HOPS):
            if not frontier:
                return None
            new_frontier: list[str] = []
            for tensor in frontier:
                # Direct hit: tensor is an output of an optimized node we matched.
                producer_in_opt = self.opt.by_output_tensor.get(tensor)
                if producer_in_opt and producer_in_opt in self.matches:
                    return producer_in_opt
                # Otherwise expand: who consumes this tensor in the source graph?
                for downstream in self.src.by_input_tensor.get(tensor, []):
                    ds = self.src.nodes.get(downstream)
                    if ds is None:
                        continue
                    for out in ds.outputs:
                        if out and out not in seen:
                            seen.add(out)
                            new_frontier.append(out)
            frontier = new_frontier
        return None

    # -- Tensor-level matching -------------------------------------------

    @staticmethod
    def _all_tensor_names(g: GraphIndex) -> set[str]:
        """All non-empty tensor names that appear in a graph: node outputs,
        node inputs (incl. model inputs), and initializers."""
        names: set[str] = set()
        names.update(t for t in g.by_output_tensor if t)
        names.update(t for t in g.by_input_tensor if t)
        names.update(g.initializers.keys())
        return names

    def _match_tensors(self) -> list[TensorMatch]:
        """Build tensor-level mapping (optimized_tensor -> original_tensors).
        Layered strategies, applied in confidence order:
          1. Exact tensor name match (model I/O + surviving intermediate tensors)
          2. Initializer name match
          3. Initializer SHA-256 hash match (renamed initializers)
          4. Node-output positional inheritance from node matches
        """
        out: dict[str, TensorMatch] = {}

        def record(
            opt_t: str, src_ts: list[str], method: MatchMethod, conf: Confidence, notes: str | None = None
        ) -> None:
            if not opt_t or opt_t in out:
                return
            out[opt_t] = TensorMatch(opt_t, list(src_ts), method, conf, notes)

        src_tensors = self._all_tensor_names(self.src)
        opt_tensors = self._all_tensor_names(self.opt)

        # Phase 1: exact name match (covers model I/O + surviving intermediates).
        for t in opt_tensors:
            if t in src_tensors and t not in self.opt.initializers:
                record(t, [t], MatchMethod.EXACT_NAME, Confidence.HIGH)

        # Phase 2: initializer name match.
        for init_name in self.opt.initializers:
            if init_name in self.src.initializers:
                record(init_name, [init_name], MatchMethod.INITIALIZER_NAME, Confidence.HIGH)

        # Phase 3: initializer hash match (renamed but byte-identical).
        src_hash_to_init: dict[bytes, list[str]] = defaultdict(list)
        for src_name, h in self.src.initializers.items():
            src_hash_to_init[h].append(src_name)
        for init_name, h in self.opt.initializers.items():
            if init_name in out:
                continue
            candidates = src_hash_to_init.get(h, [])
            if not candidates:
                continue
            cands = sorted(candidates)
            confidence = Confidence.HIGH if len(cands) == 1 else Confidence.MEDIUM
            notes = None if len(cands) == 1 else "Multiple source initializers share data hash"
            record(init_name, cands, MatchMethod.INITIALIZER_HASH, confidence, notes)

        # Phase 4: node-output positional inheritance from node matches.
        # Heuristic: optimized node X matched to source nodes [s1, ..., sN] —
        # the LAST source node (sN) is the deepest in topological order, so its
        # outputs are the most likely semantic match for X's outputs.
        for opt_node_name, opt_node in self.opt.nodes.items():
            node_match = self.matches.get(opt_node_name)
            if node_match is None:
                continue
            target_src_name = node_match.source_nodes[-1]
            src_node = self.src.nodes.get(target_src_name)
            if src_node is None:
                continue
            for i, opt_out in enumerate(opt_node.outputs):
                if not opt_out or opt_out in out:
                    continue
                if i >= len(src_node.outputs):
                    continue
                src_out = src_node.outputs[i]
                if not src_out:
                    continue
                if len(node_match.source_nodes) == 1:
                    confidence = Confidence.HIGH
                    notes = None
                else:
                    confidence = Confidence.MEDIUM
                    notes = (
                        f"derived from fused node mapping "
                        f"({len(node_match.source_nodes)} source nodes; "
                        f"output attributed to last source `{target_src_name}`)"
                    )
                record(opt_out, [src_out], MatchMethod.NODE_OUTPUT_POSITIONAL, confidence, notes)

        return list(out.values())

    # -- helpers ---------------------------------------------------------

    def _record(
        self,
        opt: NodeRecord,
        *,
        source_nodes: list[str],
        method: MatchMethod,
        confidence: Confidence,
        pattern: str | None = None,
        notes: str | None = None,
    ) -> None:
        self.matches[opt.name] = Match(
            optimized_node=opt.name,
            optimized_op_type=opt.op_type,
            source_nodes=list(source_nodes),
            source_op_types=[self.src.nodes[s].op_type for s in source_nodes],
            method=method,
            confidence=confidence,
            pattern=pattern,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Output classification
# ---------------------------------------------------------------------------


def _classify_removal(src: NodeRecord, matcher: Matcher) -> str:
    # Constant folding turns the source node's output tensors into initializers
    # in the optimized graph. Require at least one real (non-empty) output name
    # and require all real outputs to appear as optimized initializers — a node
    # with only placeholder/empty output names cannot be classified as folded.
    real_outputs = [o for o in src.outputs if o]
    if real_outputs and all(o in matcher.opt.initializers for o in real_outputs):
        return "constant_folded"
    # If a downstream optimized match claims this src node via lineage extension,
    # we'd already have moved it into matches. Reaching here means lineage
    # extension didn't link it — most likely it was removed (e.g. NoOp dropout).
    return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_output(
    args: argparse.Namespace,
    matcher: Matcher,
    matches: list[Match],
    tensor_matches: list[TensorMatch],
    stats: MatcherStats,
    unmatched: list[str],
    removed: list[str],
) -> dict:
    return {
        "source_model": str(args.source_model),
        "optimized_model": str(args.optimized_model),
        "summary": {
            "total_source_nodes": stats.total_source,
            "total_optimized_nodes": stats.total_optimized,
            "matched_optimized_nodes": stats.matched,
            "unmatched_optimized_nodes": stats.unmatched,
            "by_method": dict(stats.by_method),
            "total_optimized_tensors": stats.total_optimized_tensors,
            "matched_tensors": stats.matched_tensors,
            "by_tensor_method": dict(stats.by_tensor_method),
        },
        "node_mappings": [
            {
                "optimized_node": m.optimized_node,
                "optimized_op_type": m.optimized_op_type,
                "original_sources": _build_original_chain(m.source_nodes, m.source_op_types, matcher.src),
                "match_method": m.method.value,
                "confidence": m.confidence.value,
                **({"fusion_pattern": m.pattern} if m.pattern else {}),
                **({"notes": m.notes} if m.notes else {}),
            }
            for m in matches
        ],
        "tensor_mappings": [
            {
                "optimized_tensor": tm.optimized_tensor,
                "original_tensors": [{"name": n} for n in tm.original_tensors],
                "match_method": tm.method.value,
                "confidence": tm.confidence.value,
                **({"notes": tm.notes} if tm.notes else {}),
            }
            for tm in tensor_matches
        ],
        "unmapped_optimized_nodes": [
            {
                "name": n,
                "op_type": matcher.opt.nodes[n].op_type,
                "reason": "no_strategy_matched",
            }
            for n in unmatched
        ],
        "removed_original_nodes": [
            {
                "name": n,
                "op_type": matcher.src.nodes[n].op_type,
                "likely_reason": _classify_removal(matcher.src.nodes[n], matcher),
            }
            for n in removed
        ],
    }


# ---------------------------------------------------------------------------
# Join with QNN EP qnn_op_trace.json
# ---------------------------------------------------------------------------

# TraceTargetType string values as serialized by the QNN EP's
# SerializeFrameworkOpTrace() in qnn_op_tracing_serialization.cc.
_TRACE_TYPE_TENSOR = "TENSOR"
_TRACE_TYPE_OP = "OP"


def _connecting_tensor(src_index: GraphIndex, a: str, b: str) -> str | None:
    """Return a tensor that's an output of source-graph node `a` and an input
    of node `b` (the one that flows from a to b). Used to weave intermediate
    tensors into a multi-op source chain."""
    na = src_index.nodes.get(a)
    nb = src_index.nodes.get(b)
    if na is None or nb is None:
        return None
    b_inputs = set(nb.inputs)
    for out in na.outputs:
        if out and out in b_inputs:
            return out
    return None


def _build_original_chain(
    source_nodes: list[str],
    source_op_types: list[str],
    src_index: GraphIndex,
) -> list[dict]:
    """Build a QNN-EP-style source chain (`op1, tensor_out1, op2, ..., opN`)
    from a list of source-graph node names. Intermediate tensors are looked
    up via `_connecting_tensor` against the source graph. Tensor entries use
    `{name, type="TENSOR"}`, op entries use `{name, type="OP", op_type}`."""
    chain: list[dict] = []
    for i, name in enumerate(source_nodes):
        chain.append(
            {
                "name": name,
                "type": _TRACE_TYPE_OP,
                "op_type": source_op_types[i],
            }
        )
        if i < len(source_nodes) - 1:
            connecting = _connecting_tensor(src_index, name, source_nodes[i + 1])
            if connecting:
                chain.append({"name": connecting, "type": _TRACE_TYPE_TENSOR})
    return chain


def join_qnn_trace(
    matches: list[Match],
    tensor_matches: list[TensorMatch],
    qnn_trace: dict,
    src_index: GraphIndex,
) -> tuple[dict, dict]:
    """Extend a QNN EP qnn_op_trace.json with `original_sources` on each
    TraceMapping. Resolves both op-typed sources (via node matches) and
    tensor-typed sources (via tensor matches). Returns the extended trace plus
    a stats dict.

    The returned dict is a deep copy — `qnn_trace` is not mutated, so callers
    can keep using their original input unchanged.

    For multi-source op expansions (fusions), the source nodes are emitted in
    source-graph topological order and intermediate tensors are woven between
    consecutive ops, mirroring the QNN EP `op1, tensor1, op2, ..., opN`
    convention.

    Output entries follow the QNN EP `TraceSourcePair` schema (`{name, type}`),
    with an additional `op_type` field on op entries for human readability.

    Returns:
        A tuple ``(extended_trace, stats)``.

        ``extended_trace`` is a deep copy of ``qnn_trace`` with an
        ``original_sources`` field added next to ``sources`` on every
        op_mappings/tensor_mappings entry.

        ``stats`` is a dict with the following integer keys:

          - ``op_mappings_total`` / ``op_mappings_extended`` — count of
            ``op_mappings[]`` entries traversed and how many had at least one
            ``original_sources`` entry written.
          - ``tensor_mappings_total`` / ``tensor_mappings_extended`` — same
            counts for ``tensor_mappings[]``.
          - ``op_sources_unresolved`` / ``tensor_sources_unresolved`` — count
            of ``sources[]`` references (op-typed and tensor-typed) that
            could not be resolved to an original-ONNX name.
    """
    qnn_trace = copy.deepcopy(qnn_trace)
    node_index: dict[str, list[dict]] = {
        m.optimized_node: _build_original_chain(m.source_nodes, m.source_op_types, src_index) for m in matches
    }

    tensor_index: dict[str, list[str]] = {tm.optimized_tensor: list(tm.original_tensors) for tm in tensor_matches}

    stats = {
        "op_mappings_total": 0,
        "op_mappings_extended": 0,
        "tensor_mappings_total": 0,
        "tensor_mappings_extended": 0,
        "op_sources_unresolved": 0,
        "tensor_sources_unresolved": 0,
    }

    def lookup_originals(sources: list) -> list[dict]:
        """Collect deduplicated original sources for entries in `sources`,
        preserving QNN EP's `{name, type}` schema. Mutates `stats` counters
        for unresolved references.

        When an OP-typed source name is not in `node_index` (i.e. the optimized
        graph has no match for it in the source->optimized mapping), but the name
        exists as a node in the source graph, it is assumed the EP consumed the
        source ONNX directly without an intervening ORT optimization that would
        have renamed or fused the node. In that case the source name is already
        an original-ONNX name and is passed through unchanged (identity match)."""
        seen: set[tuple[int, str]] = set()
        result: list[dict] = []
        for src in sources or []:
            src_type = src.get("type")
            src_name = src.get("name", "")
            if src_type == _TRACE_TYPE_OP:
                chain = node_index.get(src_name)
                if chain is None:
                    # Identity fallback: the EP processed this source node directly
                    # from the user's ONNX (the source -> optimized step was a no-op
                    # for this node, or the user passed the same file as both).
                    src_node = src_index.nodes.get(src_name)
                    if src_node is not None:
                        entry = {
                            "name": src_name,
                            "type": _TRACE_TYPE_OP,
                            "op_type": src_node.op_type,
                        }
                        key = (entry["type"], entry["name"])
                        if key not in seen:
                            seen.add(key)
                            result.append(entry)
                        continue
                    stats["op_sources_unresolved"] += 1
                    continue
                for entry in chain:
                    key = (entry["type"], entry["name"])
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(dict(entry))
            elif src_type == _TRACE_TYPE_TENSOR:
                originals = tensor_index.get(src_name)
                if originals is None:
                    stats["tensor_sources_unresolved"] += 1
                    continue
                for tname in originals:
                    key = (_TRACE_TYPE_TENSOR, tname)
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append({"name": tname, "type": _TRACE_TYPE_TENSOR})
        return result

    for sg in qnn_trace.get("subgraph_traces", []):
        for mapping in sg.get("op_mappings", []) or []:
            stats["op_mappings_total"] += 1
            originals = lookup_originals(mapping.get("sources", []))
            if originals:
                mapping["original_sources"] = originals
                stats["op_mappings_extended"] += 1
        for mapping in sg.get("tensor_mappings", []) or []:
            stats["tensor_mappings_total"] += 1
            originals = lookup_originals(mapping.get("sources", []))
            if originals:
                mapping["original_sources"] = originals
                stats["tensor_mappings_extended"] += 1

    return qnn_trace, stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Match source ONNX nodes to ORT-optimized ONNX nodes.",
        allow_abbrev=False,
    )
    ap.add_argument("--source-model", required=True, type=Path, help="Original user-provided ONNX model")
    ap.add_argument(
        "--optimized-model",
        required=True,
        type=Path,
        help="ORT-optimized graph the EP consumed. Either an ONNX model (saved via "
        "SessionOptions::optimized_model_filepath) or a QNN-Netron-schema JSON dumped "
        "by the QNN EP's `dump_qnn_ep_input_graph` option (detected by a .json extension).",
    )
    ap.add_argument("--output", required=True, type=Path, help="Output JSON path")
    ap.add_argument(
        "--qnn-trace",
        type=Path,
        default=None,
        help="(Optional) QNN EP qnn_op_trace.json. When given, the trace is "
        "extended with `original_sources` and written to --joined-output.",
    )
    ap.add_argument(
        "--joined-output",
        type=Path,
        default=None,
        help="(Optional) Output path for the extended QNN trace. Defaults to "
        "<qnn-trace-stem>.with_original_sources.json next to --qnn-trace. "
        "Only used when --qnn-trace is given.",
    )
    ap.add_argument("--verbose", action="store_true", help="Print per-strategy diagnostics")
    args = ap.parse_args()

    if not args.source_model.is_file():
        sys.stderr.write(f"error: source model not found: {args.source_model}\n")
        return 2
    if not args.optimized_model.is_file():
        sys.stderr.write(f"error: optimized model not found: {args.optimized_model}\n")
        return 2
    if args.qnn_trace is not None and not args.qnn_trace.is_file():
        sys.stderr.write(f"error: QNN trace not found: {args.qnn_trace}\n")
        return 2
    if args.joined_output is not None and args.qnn_trace is None:
        sys.stderr.write("error: --joined-output requires --qnn-trace\n")
        return 2
    if args.output.exists():
        sys.stderr.write(f"warning: overwriting existing output: {args.output}\n")

    _require_onnx()
    source = onnx.load(str(args.source_model))

    # The optimized side may be a QNN-Netron-schema JSON (from the QNN EP's
    # dump_qnn_ep_input_graph) instead of an .onnx; detect by extension and
    # build the GraphIndex directly so no onnx proto load is needed for it.
    if args.optimized_model.suffix.lower() == ".json":
        try:
            opt_doc = json.loads(args.optimized_model.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            sys.stderr.write(f"error: failed to parse {args.optimized_model}: {e}\n")
            return 2
        optimized = GraphIndex.build_from_qnn_json(opt_doc)
    else:
        optimized = onnx.load(str(args.optimized_model))

    matcher = Matcher(source, optimized)
    matches, tensor_matches, stats, unmatched, removed = matcher.run()

    output = build_output(args, matcher, matches, tensor_matches, stats, unmatched, removed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(
        f"Matched {stats.matched}/{stats.total_optimized} optimized nodes "
        f"(unmatched: {stats.unmatched}, source-removed: {len(removed)}); "
        f"{stats.matched_tensors}/{stats.total_optimized_tensors} optimized tensors"
    )
    if args.verbose:
        print(f"  by node method: {dict(stats.by_method)}")
        print(f"  by tensor method: {dict(stats.by_tensor_method)}")
        if unmatched:
            print(f"  unmapped optimized nodes: {len(unmatched)}")
            for n in unmatched[:10]:
                print(f"    - {n} ({matcher.opt.nodes[n].op_type})")
            if len(unmatched) > 10:
                print(f"    ... and {len(unmatched) - 10} more")
        if removed:
            classified: dict[str, int] = defaultdict(int)
            for n in removed:
                classified[_classify_removal(matcher.src.nodes[n], matcher)] += 1
            print(f"  source-removed by reason: {dict(classified)}")
    print(f"Output: {args.output}")

    if args.qnn_trace is not None:
        joined_path = args.joined_output or (
            args.qnn_trace.with_name(args.qnn_trace.stem + ".with_original_sources.json")
        )
        if joined_path.exists():
            sys.stderr.write(f"warning: overwriting existing joined output: {joined_path}\n")
        try:
            qnn_trace = json.loads(args.qnn_trace.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            sys.stderr.write(f"error: failed to parse {args.qnn_trace}: {e}\n")
            return 2
        extended, join_stats = join_qnn_trace(matches, tensor_matches, qnn_trace, matcher.src)
        joined_path.parent.mkdir(parents=True, exist_ok=True)
        joined_path.write_text(json.dumps(extended, indent=2), encoding="utf-8")
        print(
            f"Joined QNN trace: {join_stats['op_mappings_extended']}/"
            f"{join_stats['op_mappings_total']} op_mappings extended, "
            f"{join_stats['tensor_mappings_extended']}/"
            f"{join_stats['tensor_mappings_total']} tensor_mappings extended"
        )
        if join_stats["op_sources_unresolved"]:
            print(
                f"  warning: {join_stats['op_sources_unresolved']} op-typed source "
                f"reference(s) had no matcher entry (optimized node not in source->optimized mapping)"
            )
        if join_stats["tensor_sources_unresolved"]:
            print(
                f"  warning: {join_stats['tensor_sources_unresolved']} tensor-typed source "
                f"reference(s) had no matcher entry (optimized tensor not in source->optimized mapping)"
            )
        print(f"Joined output: {joined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
