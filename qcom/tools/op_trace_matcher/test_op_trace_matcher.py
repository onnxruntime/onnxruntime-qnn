# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for `source_to_optimized_matcher` and `enrich_profiling_csv`.

The tests build minimal `onnx.ModelProto` graphs in memory (no real model
needed) and exercise:

  * each layered matching strategy (`MatchMethod` enum) at least once
  * the linear-chain fusion walk (MatMul + Add -> Gemm)
  * `join_qnn_trace`'s identity fallback when no real source ONNX is paired
  * each of the four CSV adaptation modes in `enrich_profiling_csv.enrich`
  * `build_lookups` on a degenerate empty trace

Tests are stdlib-only except for `onnx`, which is also the matcher's only
runtime dependency. Pytest is the runner; tests are self-contained so they
can be invoked via `python -m pytest qcom/tools/op_trace_matcher/`.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

# Make the sibling module importable regardless of how pytest discovers the
# file (rootdir / cwd / -m). The package's __init__.py also enables the
# `qcom.tools.op_trace_matcher` import path when invoked via `python -m`.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import enrich_profiling_csv as enr  # noqa: E402
import source_to_optimized_matcher as m  # noqa: E402

onnx = pytest.importorskip("onnx")
import numpy as np  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

sys.path.remove(str(_THIS_DIR))


# ---------------------------------------------------------------------------
# Helpers for building tiny ONNX graphs
# ---------------------------------------------------------------------------


def _make_value_info(name: str, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _make_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def _model_from_nodes(
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
    name: str = "test_graph",
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=nodes,
        name=name,
        inputs=inputs,
        outputs=outputs,
        initializer=initializers or [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    return model


# ---------------------------------------------------------------------------
# GraphIndex
# ---------------------------------------------------------------------------


class TestGraphIndex:
    def test_build_records_initializers_and_consumers(self) -> None:
        weight = _make_initializer("W", np.ones((2, 2), dtype=np.float32))
        x = _make_value_info("X", [2, 2])
        y = _make_value_info("Y", [2, 2])
        node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="matmul1")
        model = _model_from_nodes([node], [x], [y], [weight])

        idx = m.GraphIndex.build(model)

        assert "matmul1" in idx.nodes
        rec = idx.nodes["matmul1"]
        assert rec.op_type == "MatMul"
        assert rec.inputs == ("X", "W")
        assert "W" in rec.initializer_inputs
        assert "W" in idx.initializers
        assert idx.by_output_tensor["Y"] == "matmul1"
        assert "matmul1" in idx.by_initializer_consumer["W"]

    def test_build_from_qnn_json_uses_static_type_for_initializers(self) -> None:
        # The QNN-Netron schema emitted by the EP marks initializers via
        # `type == _QNN_TENSOR_TYPE_STATIC`. Verify that GraphIndex picks
        # them up under that contract.
        doc = {
            "graph": {
                "nodes": {
                    "n1": {"type": "MatMul", "input_names": ["X", "W"], "output_names": ["Y"]},
                },
                "tensors": {
                    "X": {"type": 0},  # APP_WRITE
                    "W": {"type": m._QNN_TENSOR_TYPE_STATIC},
                    "Y": {"type": 1},  # APP_READ
                },
            },
        }
        idx = m.GraphIndex.build_from_qnn_json(doc)
        assert "W" in idx.initializers
        assert "X" not in idx.initializers
        assert idx.nodes["n1"].op_type == "MatMul"


# ---------------------------------------------------------------------------
# Matcher strategies
# ---------------------------------------------------------------------------


class TestMatcherStrategies:
    def _single_node_model(self, op_type: str, name: str, *, inputs=("X",), outputs=("Y",)) -> onnx.ModelProto:
        node = helper.make_node(op_type, list(inputs), list(outputs), name=name)
        return _model_from_nodes(
            [node],
            [_make_value_info(t, [2, 2]) for t in inputs],
            [_make_value_info(t, [2, 2]) for t in outputs],
        )

    def test_exact_name_match(self) -> None:
        # Same node name on both sides — should land EXACT_NAME with HIGH.
        src = self._single_node_model("Relu", "relu1")
        opt = self._single_node_model("Relu", "relu1")
        matches, *_ = m.Matcher(src, opt).run()
        assert len(matches) == 1
        assert matches[0].method == m.MatchMethod.EXACT_NAME
        assert matches[0].confidence == m.Confidence.HIGH
        assert matches[0].source_nodes == ["relu1"]

    def test_exact_io_match_on_renamed_node(self) -> None:
        # Same input + output names, different node name — EXACT_IO.
        src = self._single_node_model("Relu", "relu_orig")
        opt = self._single_node_model("Relu", "relu_renamed")
        matches, *_ = m.Matcher(src, opt).run()
        assert len(matches) == 1
        assert matches[0].method == m.MatchMethod.EXACT_IO
        assert matches[0].source_nodes == ["relu_orig"]

    def test_initializer_name_match(self) -> None:
        # Same initializer name consumed by a node whose other inputs differ.
        weight = _make_initializer("W", np.ones((2, 2), dtype=np.float32))
        src = _model_from_nodes(
            [helper.make_node("MatMul", ["X_src", "W"], ["Y_src"], name="m_src")],
            [_make_value_info("X_src", [2, 2])],
            [_make_value_info("Y_src", [2, 2])],
            [weight],
        )
        opt = _model_from_nodes(
            [helper.make_node("MatMul", ["X_opt", "W"], ["Y_opt"], name="m_opt")],
            [_make_value_info("X_opt", [2, 2])],
            [_make_value_info("Y_opt", [2, 2])],
            [weight],
        )
        matches, *_ = m.Matcher(src, opt).run()
        assert len(matches) == 1
        assert matches[0].method == m.MatchMethod.INITIALIZER_NAME

    def test_initializer_hash_match_on_renamed_initializer(self) -> None:
        # Same bytes, different name — only INITIALIZER_HASH can recover this
        # since names differ AND I/O tensor names differ.
        arr = np.arange(4, dtype=np.float32).reshape(2, 2)
        weight_src = _make_initializer("W_src", arr)
        weight_opt = _make_initializer("W_opt", arr)
        src = _model_from_nodes(
            [helper.make_node("MatMul", ["X_src", "W_src"], ["Y_src"], name="m_src")],
            [_make_value_info("X_src", [2, 2])],
            [_make_value_info("Y_src", [2, 2])],
            [weight_src],
        )
        opt = _model_from_nodes(
            [helper.make_node("MatMul", ["X_opt", "W_opt"], ["Y_opt"], name="m_opt")],
            [_make_value_info("X_opt", [2, 2])],
            [_make_value_info("Y_opt", [2, 2])],
            [weight_opt],
        )
        matches, *_ = m.Matcher(src, opt).run()
        assert len(matches) == 1
        assert matches[0].method == m.MatchMethod.INITIALIZER_HASH

    def test_fusion_pattern_matmul_add_to_gemm(self) -> None:
        # Source: MatMul -> Add(bias). Optimized: a single Gemm. Walker should
        # link the Gemm node back to the MatMul + Add chain.
        weight = _make_initializer("W", np.ones((2, 2), dtype=np.float32))
        bias = _make_initializer("B", np.ones((2,), dtype=np.float32))
        src = _model_from_nodes(
            [
                helper.make_node("MatMul", ["X", "W"], ["mm_out"], name="matmul1"),
                helper.make_node("Add", ["mm_out", "B"], ["Y"], name="add1"),
            ],
            [_make_value_info("X", [2, 2])],
            [_make_value_info("Y", [2, 2])],
            [weight, bias],
        )
        opt = _model_from_nodes(
            [helper.make_node("Gemm", ["X", "W", "B"], ["Y"], name="gemm1")],
            [_make_value_info("X", [2, 2])],
            [_make_value_info("Y", [2, 2])],
            [weight, bias],
        )
        matches, *_ = m.Matcher(src, opt).run()
        assert len(matches) == 1
        # MatMul + Add -> Gemm via fusion pattern (HIGH); other strategies
        # may also fire on shared initializers, so accept either path that
        # produces the chain.
        assert matches[0].source_op_types == ["MatMul", "Add"]


# ---------------------------------------------------------------------------
# join_qnn_trace identity fallback
# ---------------------------------------------------------------------------


class TestJoinQnnTraceIdentityFallback:
    def test_identity_fallback_emits_dst_as_original(self) -> None:
        # When the matcher produced no matches (e.g. EP-input == source ONNX
        # in the QNN-direct workflow), join_qnn_trace falls back to identity:
        # `dst_name` itself becomes the original-source entry.
        weight = _make_initializer("W", np.ones((2, 2), dtype=np.float32))
        src_model = _model_from_nodes(
            [helper.make_node("MatMul", ["X", "W"], ["Y"], name="matmul1")],
            [_make_value_info("X", [2, 2])],
            [_make_value_info("Y", [2, 2])],
            [weight],
        )
        src_index = m.GraphIndex.build(src_model)

        qnn_trace = {
            "subgraph_traces": [
                {
                    "op_mappings": [
                        {
                            "dst_name": "matmul1",
                            "sources": [{"name": "matmul1", "type": m._TRACE_TYPE_OP}],
                        }
                    ]
                }
            ]
        }
        # No matches -> identity fallback fires for the op-typed source.
        extended, _stats = m.join_qnn_trace([], [], qnn_trace, src_index)
        mapping = extended["subgraph_traces"][0]["op_mappings"][0]
        assert "original_sources" in mapping
        names = [s["name"] for s in mapping["original_sources"]]
        assert "matmul1" in names


# ---------------------------------------------------------------------------
# enrich_profiling_csv: four input modes
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        for row in rows:
            w.writerow(row)


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="") as fh:
        return list(csv.reader(fh))


def _merged_trace_one_mapping(qnn_op: str, original: str) -> dict:
    return {
        "subgraph_traces": [
            {
                "op_mappings": [
                    {
                        "dst_name": qnn_op,
                        "sources": [{"name": "opt_" + qnn_op, "type": m._TRACE_TYPE_OP}],
                        "original_sources": [{"name": original, "type": m._TRACE_TYPE_OP}],
                    }
                ]
            }
        ]
    }


class TestEnrichCsvModes:
    def test_basic_no_node_events_copies_verbatim(self, tmp_path: Path) -> None:
        # BASIC profiling has no NODE-level rows. Whatever the merged trace
        # says, the CSV is copied byte-for-byte.
        profiling_csv = tmp_path / "basic.csv"
        _write_csv(
            profiling_csv,
            [
                ["Event Identifier", "Time (us)"],  # header only — no NODE rows
            ],
        )
        merged = _merged_trace_one_mapping("qnn_op", "orig_op")
        out = tmp_path / "out.csv"
        stats = enr.enrich(profiling_csv, merged, out)
        assert stats["node_events_present"] is False
        assert profiling_csv.read_bytes() == out.read_bytes()

    def test_populated_existing_column_appends_only_originals(self, tmp_path: Path) -> None:
        profiling_csv = tmp_path / "populated.csv"
        _write_csv(
            profiling_csv,
            [
                ["Event Identifier", "ONNX Source Ops"],
                ["qnn_op:OpId_3 (cycles)", "runtime_written"],
            ],
        )
        merged = _merged_trace_one_mapping("qnn_op", "orig_op")
        out = tmp_path / "out.csv"
        stats = enr.enrich(profiling_csv, merged, out)
        # Already-populated `ONNX Source Ops` => do not synthesize, do not
        # fill in place — only append `Original ONNX Source Ops`.
        assert stats["added_optimized_column"] is False
        assert stats["filled_existing_onnx_column"] is False
        rows = _read_csv(out)
        assert "Original ONNX Source Ops" in rows[0]
        # Runtime-written value preserved.
        assert "runtime_written" in rows[1]
        # Original ONNX Source Ops column populated.
        orig_idx = rows[0].index("Original ONNX Source Ops")
        assert rows[1][orig_idx] == "orig_op"

    def test_all_empty_existing_column_filled_in_place(self, tmp_path: Path) -> None:
        # AOT-no-sidecar case: column is present but every NODE row's cell
        # is empty. Enricher should fill that column from the merged trace
        # AND append Original ONNX Source Ops.
        profiling_csv = tmp_path / "aot.csv"
        _write_csv(
            profiling_csv,
            [
                ["Event Identifier", "ONNX Source Ops"],
                ["qnn_op:OpId_5 (cycles)", ""],
            ],
        )
        merged = _merged_trace_one_mapping("qnn_op", "orig_op")
        out = tmp_path / "out.csv"
        stats = enr.enrich(profiling_csv, merged, out)
        assert stats["added_optimized_column"] is False
        assert stats["filled_existing_onnx_column"] is True
        rows = _read_csv(out)
        onnx_idx = rows[0].index("ONNX Source Ops")
        orig_idx = rows[0].index("Original ONNX Source Ops")
        assert rows[1][onnx_idx] == "opt_qnn_op"
        assert rows[1][orig_idx] == "orig_op"

    def test_missing_column_synthesizes_both(self, tmp_path: Path) -> None:
        profiling_csv = tmp_path / "synth.csv"
        _write_csv(
            profiling_csv,
            [
                ["Event Identifier", "Time (us)"],
                ["qnn_op:OpId_7 (cycles)", "100"],
            ],
        )
        merged = _merged_trace_one_mapping("qnn_op", "orig_op")
        out = tmp_path / "out.csv"
        stats = enr.enrich(profiling_csv, merged, out)
        assert stats["added_optimized_column"] is True
        rows = _read_csv(out)
        assert "ONNX Source Ops" in rows[0]
        assert "Original ONNX Source Ops" in rows[0]
        onnx_idx = rows[0].index("ONNX Source Ops")
        orig_idx = rows[0].index("Original ONNX Source Ops")
        assert rows[1][onnx_idx] == "opt_qnn_op"
        assert rows[1][orig_idx] == "orig_op"

    def test_refuses_already_enriched_csv(self, tmp_path: Path) -> None:
        profiling_csv = tmp_path / "already.csv"
        _write_csv(
            profiling_csv,
            [
                ["Event Identifier", "ONNX Source Ops", "Original ONNX Source Ops"],
                ["qnn_op:OpId_1 (cycles)", "x", "y"],
            ],
        )
        out = tmp_path / "out.csv"
        with pytest.raises(ValueError):
            enr.enrich(profiling_csv, {}, out)


# ---------------------------------------------------------------------------
# build_lookups
# ---------------------------------------------------------------------------


class TestBuildLookups:
    def test_empty_trace_returns_empty_lookups(self) -> None:
        originals, optimized = enr.build_lookups({})
        assert originals == {}
        assert optimized == {}

    def test_skips_mappings_without_dst_name(self) -> None:
        trace = {
            "subgraph_traces": [
                {
                    "op_mappings": [
                        {"sources": [{"name": "a", "type": m._TRACE_TYPE_OP}]},  # no dst_name
                        {
                            "dst_name": "qnn_op",
                            "sources": [{"name": "opt_a", "type": m._TRACE_TYPE_OP}],
                            "original_sources": [{"name": "orig_a", "type": m._TRACE_TYPE_OP}],
                        },
                    ]
                }
            ]
        }
        originals, optimized = enr.build_lookups(trace)
        assert list(optimized.keys()) == ["qnn_op"]
        assert list(originals.keys()) == ["qnn_op"]
